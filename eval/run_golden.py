"""Evaluation sets: compare rag, oag and hybrid on the same corpus and questions.

Metrics per question group and mode:
  context hit   the expected answer is present in the context handed to the LLM
                (facts and/or passages). Needs no answer model, so it is cheap.
  answer acc.   with --answers: the LLM's answer contains the expected keywords.

The M2 gate compares hybrid with rag on multi_hop and deep_hop questions.

Models come from flags or the RAG_OAG_* environment variables, e.g. local Ollama:

    python eval/run_golden.py --base-url http://localhost:11434/v1 \\
        --llm gemma4:e4b --embed nomic-embed-text --answers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml
from redis.exceptions import ResponseError

from rag_with_oag import AsyncKnowledgeBase, EmbedConfig, LLMConfig, Ontology, RagOagError
from rag_with_oag.store.connection import GraphConnection
from rag_with_oag.store.queries import safe_identifier
from rag_with_oag.store.repository import GraphRepository
from rag_with_oag.types import MODES, Mode

if TYPE_CHECKING or __package__:
    from .graph_quality import (
        ExpectedGraph,
        FoundEntity,
        FoundRelation,
        GraphQuality,
        score_graph_quality,
    )
    from .synthetic.text import matches as matches
    from .synthetic.text import norm as norm
else:
    from graph_quality import (
        ExpectedGraph,
        FoundEntity,
        FoundRelation,
        GraphQuality,
        score_graph_quality,
    )
    from synthetic.text import matches as matches
    from synthetic.text import norm as norm

HERE = Path(__file__).parent
GOLDEN = HERE / "golden"


async def drop_graph(db: Path, name: str) -> None:
    connection = GraphConnection(path=db)
    repository = GraphRepository(connection, name)
    try:
        await connection.open()
        await repository.drop()
    except ResponseError as exc:
        if "empty key" not in str(exc).lower():
            raise
    finally:
        await connection.close()


async def read_graph_quality(
    store: GraphRepository, expected: ExpectedGraph, ontology: Ontology
) -> GraphQuality:
    type_keys = {entity.name: entity.key for entity in ontology.entities}
    keys = list(dict.fromkeys(key for key in type_keys.values() if key is not None))
    projection = "".join(f", e.{safe_identifier(key)}" for key in keys)
    nodes = await store.query(f"MATCH (e:Entity) RETURN id(e), labels(e), e.name{projection}")
    entities = []
    for node_id, labels, name, *values in nodes:
        entity_type = next((label for label in labels if label in type_keys), None)
        if entity_type is None:
            entity_type = next((label for label in labels if label != "Entity"), "Entity")
        props = dict(zip(keys, values, strict=True))
        key_property = type_keys.get(entity_type)
        key = props.get(key_property) if key_property else None
        entities.append(FoundEntity(node_id, entity_type, key, name or ""))
    edges = await store.query(
        "MATCH (s:Entity)-[r]->(t:Entity) RETURN type(r), id(s), id(t), r.rol, r.oran"
    )
    relations = [
        FoundRelation(relation_type, source, target, {"rol": rol, "oran": oran})
        for relation_type, source, target, rol, oran in edges
    ]
    return score_graph_quality(expected, entities, relations, ontology)


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
    groups: list[str] = field(default_factory=list)
    graph_quality: GraphQuality | None = None

    def cell(self, group: str | None, mode: str) -> dict[str, float | None]:
        rows = [r for r in self.rows if r.mode == mode and (group is None or r.group == group)]
        answered = [r for r in rows if r.answer_ok is not None]
        return {
            "context_hit": 100 * sum(r.context_hit for r in rows) / len(rows) if rows else None,
            "answer_acc": (100 * sum(bool(r.answer_ok) for r in answered) / len(answered))
            if answered
            else None,
            "tokens": statistics.fmean(r.context_tokens for r in rows) if rows else None,
            "ms": statistics.median(r.retrieve_ms for r in rows) if rows else None,
            "fallbacks": sum(r.mode_used != r.mode for r in rows),
        }


async def run(args: argparse.Namespace) -> Summary:
    questions = yaml.safe_load((args.set / "questions.yaml").read_text(encoding="utf-8"))
    ontology = Ontology.from_yaml(args.set / "ontology.yaml")
    expected_path = args.set / "expected_graph.json"
    expected = (
        json.loads(expected_path.read_text(encoding="utf-8")) if expected_path.exists() else None
    )
    llm = LLMConfig(model=args.llm, base_url=args.base_url, timeout=600, max_tokens=None)
    extractor = LLMConfig(model=args.extractor or args.llm, base_url=args.base_url, timeout=600)
    embedder = EmbedConfig(
        model=args.embed,
        base_url=args.embed_base_url or args.base_url,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
    )
    db = args.out / "golden.db"
    if args.fresh:
        await drop_graph(db, args.kb)
    kb = AsyncKnowledgeBase(
        args.kb,
        path=db,
        llm=llm,
        extractor_llm=extractor,
        embedder=embedder,
        ontology=ontology,
    )
    summary = Summary(
        model={"llm": args.llm, "extractor": args.extractor or args.llm, "embed": args.embed},
        ingest={},
        groups=list(questions),
    )
    try:
        t0 = time.perf_counter()
        report = await kb.ingest(args.set / "docs", concurrency=args.concurrency)
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
        if expected is not None:
            summary.graph_quality = await read_graph_quality(kb.store, expected, ontology)

        modes = cast(list[Mode], args.modes.split(","))
        for group, items in questions.items():
            for item in items[: args.limit]:
                rows: dict[str, Row] = {}
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
    mode: Mode,
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
        try:
            result = await kb.query(item["q"], mode=mode, k=k, hops=args.hops)
        except RagOagError as exc:  # e.g. the answer model ran out of context
            row.answer, row.answer_ok = f"{ANSWER_ERROR}{type(exc).__name__}: {exc}", False
        else:
            row.answer = result.answer
            row.answer_ok = matches(result.answer, item["answer"])
    return row


ANSWER_ERROR = "<error> "


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
    for group in [*s.groups, None]:
        for mode in modes:
            c = s.cell(group, mode)
            name = group or "**all**"
            tokens = "–" if c["tokens"] is None else f"{c['tokens']:.0f}"
            latency = "–" if c["ms"] is None else f"{c['ms']:.0f} ms"
            lines.append(
                f"| {name} | {mode} | {pct(c['context_hit'])} | {pct(c['answer_acc'])} "
                f"| {tokens} | {latency} | {c['fallbacks']} |"
            )
    for group in ("multi_hop", "deep_hop"):
        if group not in s.groups or "hybrid" not in modes:
            continue
        hyb = s.cell(group, "hybrid")
        if hyb["context_hit"] is None:
            continue
        metric = "answer_acc" if hyb["answer_acc"] is not None else "context_hit"
        for baseline in ("rag", RAG_EQ):
            if baseline not in modes:
                continue
            rag = s.cell(group, baseline)
            delta = (hyb[metric] or 0) - (rag[metric] or 0)
            verdict = "PASSED" if delta > 0 else "NOT PASSED"
            lines += [
                "",
                f"**M2 gate ({metric.replace('_', ' ')}, {group}): hybrid "
                f"{pct(hyb[metric])} vs {baseline} {pct(rag[metric])} → {delta:+.0f} points, "
                f"{verdict}.**",
            ]
    if RAG_EQ in modes:
        lines += [
            "",
            f"`{RAG_EQ}`: rag with k raised per question until its context is as "
            "large as hybrid's, so the comparison is at an equal token budget.",
        ]
    if s.graph_quality is not None:
        lines.extend(render_graph_quality(s.graph_quality))
    errors = [r for r in s.rows if (r.answer or "").startswith(ANSWER_ERROR)]
    if errors:
        lines += ["", f"**{len(errors)} answers failed with an error** (counted as wrong):"]
        lines += [f"- [{r.mode}] {r.question}: {r.answer}" for r in errors[:10]]
    misses = [r for r in s.rows if r.mode == "hybrid" and not r.context_hit]
    if misses:
        lines += ["", "Hybrid context misses:"]
        lines += [f"- [{r.group}] {r.question}" for r in misses]
    return "\n".join(lines)


def render_graph_quality(quality: GraphQuality) -> list[str]:
    def pct(value: float | None) -> str:
        return "–" if value is None else f"{value:.1%}"

    lines = [
        "",
        "## Graph quality",
        "",
        "| Entity type | Expected | Found | Missing | Extra nodes | Duplicates |",
        "|---|---|---|---|---|---|",
    ]
    for name, score in quality["entities"].items():
        lines.append(
            f"| {name} | {score['expected']} | {score['found']} | {score['missing']} "
            f"| {score['extra']} | {score['duplicates']} |"
        )
    lines += [
        "",
        "| Relation type | Expected | Found | Matched | Precision | Recall | Reversed |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, relation in quality["relations"].items():
        lines.append(
            f"| {name} | {relation['expected']} | {relation['found']} | {relation['matched']} "
            f"| {pct(relation['precision'])} | {pct(relation['recall'])} | {relation['reversed']} |"
        )
    lines += [
        "",
        "| Relation type | Property | Correct / matched | Accuracy |",
        "|---|---|---|---|",
    ]
    for name, relation in quality["relations"].items():
        for prop, prop_score in relation["properties"].items():
            lines.append(
                f"| {name} | {prop} | {prop_score['correct']} / {prop_score['matched']} "
                f"| {pct(prop_score['accuracy'])} |"
            )
    return lines


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
    parser.add_argument("--set", type=Path, default=GOLDEN, help="evaluation set directory")
    parser.add_argument("--kb", help="knowledge base name (default: the set directory name)")
    parser.add_argument("--limit", type=int, help="maximum questions per group")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.kb is None:
        args.kb = args.set.name or args.set.resolve().name
    args.out.mkdir(parents=True, exist_ok=True)

    summary = asyncio.run(run(args))
    modes = args.modes.split(",")
    if args.equal_budget and "hybrid" in modes:
        modes.append(RAG_EQ)
    text = render(summary, modes)
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    (args.out / f"golden_{stamp}.md").write_text(text + "\n", encoding="utf-8")
    data = asdict(summary)
    data.pop("groups")
    if summary.graph_quality is None:
        data.pop("graph_quality")
    (args.out / f"golden_{stamp}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("\n" + text)


if __name__ == "__main__":
    main()
