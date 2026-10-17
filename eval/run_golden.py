"""Golden set: compare rag, oag and hybrid on the same corpus and questions.

Metrics per question group (single_hop, multi_hop, free_text) and mode:
  context hit   the expected answer is present in the context handed to the LLM
                (facts and/or passages). Needs no answer model, so it is cheap.
  answer acc.   with --answers: the LLM's answer contains the expected keywords.

The M2 gate from the plan: on multi_hop questions hybrid must beat rag.

Models come from flags or the RAG_OAG_* environment variables, e.g. local Ollama:

    python eval/run_golden.py --base-url http://localhost:11434/v1 \\
        --llm gemma4:e4b --embed nomic-embed-text --answers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from rag_with_oag import AsyncKnowledgeBase, EmbedConfig, LLMConfig, Ontology
from rag_with_oag.types import MODES

HERE = Path(__file__).parent
GOLDEN = HERE / "golden"
GROUPS = ("single_hop", "multi_hop", "free_text")


def norm(text: str) -> str:
    """Case- and Turkish-insensitive, punctuation kept: '%2' must not match any '2'."""
    text = text.replace("ı", "i").replace("İ", "I")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.casefold())


def matches(text: str, groups: list[list[str]]) -> bool:
    haystack = norm(text)
    return all(any(norm(alt) in haystack for alt in group) for group in groups)


@dataclass
class Row:
    group: str
    mode: str
    question: str
    mode_used: str
    context_hit: bool
    context_tokens: int
    retrieve_ms: float
    answer: str | None = None
    answer_ok: bool | None = None


@dataclass
class Summary:
    model: dict[str, str]
    ingest: dict[str, Any]
    rows: list[Row] = field(default_factory=list)

    def cell(self, group: str | None, mode: str) -> dict[str, float | None]:
        rows = [r for r in self.rows if r.mode == mode and (group is None or r.group == group)]
        answered = [r for r in rows if r.answer_ok is not None]
        return {
            "context_hit": 100 * sum(r.context_hit for r in rows) / len(rows),
            "answer_acc": (100 * sum(bool(r.answer_ok) for r in answered) / len(answered))
            if answered
            else None,
            "tokens": statistics.fmean(r.context_tokens for r in rows),
            "ms": statistics.median(r.retrieve_ms for r in rows),
            "fallbacks": sum(r.mode_used != r.mode for r in rows),
        }


async def run(args: argparse.Namespace) -> Summary:
    llm = LLMConfig(model=args.llm, base_url=args.base_url, timeout=600, max_tokens=None)
    extractor = LLMConfig(model=args.extractor or args.llm, base_url=args.base_url, timeout=600)
    embedder = EmbedConfig(
        model=args.embed,
        base_url=args.embed_base_url or args.base_url,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
    )
    db = args.out / "golden.db"
    kb = AsyncKnowledgeBase(
        args.kb,
        path=db,
        llm=llm,
        extractor_llm=extractor,
        embedder=embedder,
        ontology=Ontology.from_yaml(GOLDEN / "ontology.yaml"),
    )
    summary = Summary(
        model={"llm": args.llm, "extractor": args.extractor or args.llm, "embed": args.embed},
        ingest={},
    )
    try:
        if args.fresh:
            await kb.open()
            await kb.store.drop()
            await kb.close()
            kb = AsyncKnowledgeBase(
                args.kb,
                path=db,
                llm=llm,
                extractor_llm=extractor,
                embedder=embedder,
                ontology=Ontology.from_yaml(GOLDEN / "ontology.yaml"),
            )
        t0 = time.perf_counter()
        report = await kb.ingest(GOLDEN / "docs", concurrency=args.concurrency)
        summary.ingest = {
            "seconds": round(time.perf_counter() - t0, 1),
            "processed": report.documents_processed,
            "skipped": report.documents_skipped,
            "chunks": report.chunks_written,
            "entities_written": report.entities_written,
            "relations_written": report.relations_written,
            "validation_errors": len(report.validation_errors),
            "failures": report.failures,
            "unmapped": report.unmapped_concepts[:15],
            "stats": await kb.stats(),
            "tokens": report.token_usage,
        }
        print(f"ingest: {json.dumps(summary.ingest, ensure_ascii=False)}", flush=True)

        questions = yaml.safe_load((GOLDEN / "questions.yaml").read_text(encoding="utf-8"))
        modes = args.modes.split(",")
        for group in GROUPS:
            for item in questions[group]:
                rows = {}
                for mode in modes:
                    rows[mode] = await evaluate(kb, args, group, item, mode, args.k)
                if args.equal_budget and "hybrid" in rows:
                    k = await equal_budget_k(kb, item["q"], args.k, rows["hybrid"].context_tokens)
                    rows[RAG_EQ] = await evaluate(kb, args, group, item, "rag", k, label=RAG_EQ)
                for row in rows.values():
                    summary.rows.append(row)
                    mark = "✓" if row.context_hit else "·"
                    print(f"{mark} {group:10} {row.mode:7} {item['q'][:70]}", flush=True)
    finally:
        await kb.close()
    return summary


RAG_EQ = "rag_eq"  # rag with k raised until its context is as large as hybrid's
MAX_EQUAL_K = 40


async def evaluate(
    kb: AsyncKnowledgeBase,
    args: argparse.Namespace,
    group: str,
    item: dict[str, Any],
    mode: str,
    k: int,
    *,
    label: str | None = None,
) -> Row:
    t = time.perf_counter()
    ctx = await kb.retrieve(item["q"], mode=mode, k=k, hops=args.hops)
    row = Row(
        group=group,
        mode=label or mode,
        question=item["q"],
        # A labelled run that used its mode is no fallback.
        mode_used=label if label and ctx.mode_used == mode else ctx.mode_used,
        context_hit=matches(ctx.prompt_context, item["answer"]),
        context_tokens=ctx.token_count,
        retrieve_ms=(time.perf_counter() - t) * 1000,
    )
    if args.answers:
        result = await kb.query(item["q"], mode=mode, k=k, hops=args.hops)
        row.answer = result.answer
        row.answer_ok = matches(result.answer, item["answer"])
    return row


async def equal_budget_k(kb: AsyncKnowledgeBase, question: str, k: int, tokens: int) -> int:
    """Smallest k (from ``k`` up) whose rag context reaches ``tokens``, or the corpus end."""
    previous = -1
    while k < MAX_EQUAL_K:
        ctx = await kb.retrieve(question, mode="rag", k=k)
        if ctx.token_count >= tokens or ctx.token_count == previous:  # big enough, or no more
            break
        previous = ctx.token_count
        k += 1
    return k


def render(s: Summary, modes: list[str]) -> str:
    def pct(v: float | None) -> str:
        return "–" if v is None else f"{v:.0f}%"

    lines = [
        f"# Golden set — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Models: answer `{s.model['llm']}`, extraction `{s.model['extractor']}`, "
        f"embedding `{s.model['embed']}`",
        "",
        f"Ingest: {s.ingest['processed']} documents in {s.ingest['seconds']} s; graph "
        f"{s.ingest['stats'].get('entities')} entities, {s.ingest['stats'].get('relations')} "
        f"relations; {s.ingest['validation_errors']} extraction items rejected; "
        f"{len(s.ingest['failures'])} failures.",
        "",
        "| Group | Mode | Context hit | Answer acc. | Context tokens | Retrieval p50 | Fallbacks |",
        "|---|---|---|---|---|---|---|",
    ]
    for group in [*GROUPS, None]:
        for mode in modes:
            c = s.cell(group, mode)
            name = group or "**all**"
            lines.append(
                f"| {name} | {mode} | {pct(c['context_hit'])} | {pct(c['answer_acc'])} "
                f"| {c['tokens']:.0f} | {c['ms']:.0f} ms | {c['fallbacks']} |"
            )
    if "hybrid" in modes:
        hyb = s.cell("multi_hop", "hybrid")
        metric = "answer_acc" if hyb["answer_acc"] is not None else "context_hit"
        for baseline in ("rag", RAG_EQ):
            if baseline not in modes:
                continue
            rag = s.cell("multi_hop", baseline)
            delta = (hyb[metric] or 0) - (rag[metric] or 0)
            verdict = "PASSED" if delta > 0 else "NOT PASSED"
            lines += [
                "",
                f"**M2 gate ({metric.replace('_', ' ')}, multi_hop): hybrid "
                f"{pct(hyb[metric])} vs {baseline} {pct(rag[metric])} → {delta:+.0f} points, "
                f"{verdict}.**",
            ]
    if RAG_EQ in modes:
        lines += [
            "",
            f"`{RAG_EQ}`: rag with k raised per question until its context is as "
            "large as hybrid's, so the comparison is at an equal token budget.",
        ]
    misses = [r for r in s.rows if r.mode == "hybrid" and not r.context_hit]
    if misses:
        lines += ["", "Hybrid context misses:"]
        lines += [f"- [{r.group}] {r.question}" for r in misses]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--embed-base-url")
    parser.add_argument("--llm", required=True, help="answer model")
    parser.add_argument("--extractor", help="extraction model (default: --llm)")
    parser.add_argument("--embed", required=True, help="embedding model")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("--hops", type=int)
    parser.add_argument("--answers", action="store_true", help="also generate and score answers")
    parser.add_argument(
        "--equal-budget",
        action="store_true",
        help=f"add '{RAG_EQ}': rag at the same context size as hybrid",
    )
    parser.add_argument("--query-prefix", default="", help="embedder query prefix")
    parser.add_argument("--document-prefix", default="", help="embedder document prefix")
    parser.add_argument("--fresh", action="store_true", help="drop the knowledge base first")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--kb", default="golden")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = asyncio.run(run(args))
    modes = args.modes.split(",")
    if args.equal_budget and "hybrid" in modes:
        modes.append(RAG_EQ)
    text = render(summary, modes)
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    (args.out / f"golden_{stamp}.md").write_text(text + "\n", encoding="utf-8")
    (args.out / f"golden_{stamp}.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("\n" + text)


if __name__ == "__main__":
    main()
