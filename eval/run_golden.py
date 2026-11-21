"""Evaluation sets: compare rag, oag and hybrid on the same corpus and questions.

Metrics per question group and mode:
  context hit   the expected answer string is present in the packed context
  chain hit     every hop is supported by the context (for sets with chains)
  chain cov     mean fraction of supported hops per question
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
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, NotRequired, TypedDict, cast

import yaml
from redis.exceptions import ResponseError

from rag_with_oag import (
    AsyncKnowledgeBase,
    EmbedConfig,
    LLMConfig,
    Ontology,
    RagOagError,
    RetrievalConfig,
)
from rag_with_oag.store.connection import GraphConnection
from rag_with_oag.store.queries import safe_identifier
from rag_with_oag.store.repository import GraphRepository
from rag_with_oag.types import MODES, Mode

if TYPE_CHECKING or __package__:
    from .chain import Chain, chain_coverage
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
    from chain import Chain, chain_coverage
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
DEFAULT_BASE_URL = "http://localhost:11434/v1"
RAG_EQ: Final = "rag_eq"
MAX_EQUAL_K = 40

type EvalMode = Mode | Literal["rag_eq"]


class QuestionItem(TypedDict):
    q: str
    answer: list[list[str]]
    chain: NotRequired[Chain]


def load_questions(set_dir: Path) -> dict[str, list[QuestionItem]]:
    return cast(
        dict[str, list[QuestionItem]],
        yaml.safe_load((set_dir / "questions.yaml").read_text(encoding="utf-8")),
    )


def add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", help=f"model server fallback (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--embed-base-url", help="embedding server override")
    parser.add_argument("--embed", help="embedding model (default: nomic-embed-text)")
    parser.add_argument("--query-prefix", help="embedder query prefix")
    parser.add_argument("--document-prefix", help="embedder document prefix")


def make_embedder(args: argparse.Namespace) -> EmbedConfig:
    return EmbedConfig(
        model=args.embed or os.environ.get("RAG_OAG_EMBED_MODEL", "nomic-embed-text"),
        base_url=args.embed_base_url
        or args.base_url
        or os.environ.get("RAG_OAG_EMBED_BASE_URL", DEFAULT_BASE_URL),
        query_prefix=args.query_prefix
        if args.query_prefix is not None
        else os.environ.get("RAG_OAG_EMBED_QUERY_PREFIX", ""),
        document_prefix=args.document_prefix
        if args.document_prefix is not None
        else os.environ.get("RAG_OAG_EMBED_DOCUMENT_PREFIX", ""),
    )


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


async def read_node_keys(store: GraphRepository, ontology: Ontology) -> dict[str, str]:
    node_keys: dict[str, str] = {}
    for entity in ontology.entities:
        if entity.key is None:
            continue
        label, key = safe_identifier(entity.name), safe_identifier(entity.key)
        rows = await store.query(
            f"MATCH (e:Entity:{label}) WHERE e.{key} IS NOT NULL RETURN e.id, e.{key}"
        )
        node_keys.update((str(node_id), str(value)) for node_id, value in rows)
    return node_keys


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
    chain_hit: bool | None = None
    chain_cov: float | None = None


@dataclass
class Summary:
    model: dict[str, str]
    ingest: dict[str, Any]
    rows: list[Row] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    graph_quality: GraphQuality | None = None

    @property
    def has_chains(self) -> bool:
        return any(row.chain_hit is not None for row in self.rows)

    def cell(self, group: str | None, mode: str) -> dict[str, float | None]:
        rows = [r for r in self.rows if r.mode == mode and (group is None or r.group == group)]
        answered = [r for r in rows if r.answer_ok is not None]
        chain_hits = [r.chain_hit for r in rows if r.chain_hit is not None]
        chain_coverages = [r.chain_cov for r in rows if r.chain_cov is not None]
        return {
            "context_hit": 100 * sum(r.context_hit for r in rows) / len(rows) if rows else None,
            "chain_hit": 100 * statistics.fmean(chain_hits) if chain_hits else None,
            "chain_cov": 100 * statistics.fmean(chain_coverages) if chain_coverages else None,
            "answer_acc": (100 * sum(bool(r.answer_ok) for r in answered) / len(answered))
            if answered
            else None,
            "tokens": statistics.fmean(r.context_tokens for r in rows) if rows else None,
            "ms": statistics.median(r.retrieve_ms for r in rows) if rows else None,
            "fallbacks": sum(r.mode_used != r.mode for r in rows),
        }


async def run(args: argparse.Namespace) -> Summary:
    questions = load_questions(args.set)
    ontology = Ontology.from_yaml(args.set / "ontology.yaml")
    expected_path = args.set / "expected_graph.json"
    expected = (
        json.loads(expected_path.read_text(encoding="utf-8")) if expected_path.exists() else None
    )
    base_url = args.base_url or DEFAULT_BASE_URL
    llm = LLMConfig(model=args.llm, base_url=base_url, timeout=600, max_tokens=None)
    extractor = LLMConfig(model=args.extractor or args.llm, base_url=base_url, timeout=600)
    embedder = make_embedder(args)
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
        model={"llm": args.llm, "extractor": args.extractor or args.llm, "embed": embedder.model},
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

        node_keys = (
            await read_node_keys(kb.store, ontology)
            if any("chain" in item for items in questions.values() for item in items)
            else {}
        )
        modes = runner_modes(args)
        for group, items in questions.items():
            for item in items[: args.limit]:
                rows = await evaluate_modes(
                    kb, group, item, modes, args.k, args.hops, node_keys, answers=args.answers
                )
                for row in rows:
                    summary.rows.append(row)
                    hit = row.context_hit if row.chain_hit is None else row.chain_hit
                    mark = "✓" if hit else "·"
                    print(f"{mark} {group:10} {row.mode:7} {item['q'][:70]}", flush=True)
    finally:
        await kb.close()
    return summary


def runner_modes(args: argparse.Namespace) -> list[EvalMode]:
    modes = cast(list[EvalMode], args.modes.split(","))
    if args.equal_budget and "hybrid" in modes:
        modes.append(RAG_EQ)
    return modes


async def evaluate_modes(
    kb: AsyncKnowledgeBase,
    group: str,
    item: QuestionItem,
    modes: Sequence[EvalMode],
    k: int,
    hops: int | None,
    node_keys: Mapping[str, str],
    *,
    answers: bool = False,
) -> list[Row]:
    rows: dict[EvalMode, Row] = {}
    for mode in modes:
        if mode != RAG_EQ:
            rows[mode] = await evaluate(kb, group, item, mode, k, hops, node_keys, answers=answers)
    if RAG_EQ in modes:
        hybrid = rows.get("hybrid")
        tokens = (
            hybrid.context_tokens
            if hybrid is not None
            else (await kb.retrieve(item["q"], mode="hybrid", k=k, hops=hops)).token_count
        )
        equal_k = await equal_budget_k(kb, item["q"], k, tokens)
        rows[RAG_EQ] = await evaluate(
            kb, group, item, "rag", equal_k, hops, node_keys, answers=answers, label=RAG_EQ
        )
    return list(rows.values())


async def evaluate(
    kb: AsyncKnowledgeBase,
    group: str,
    item: QuestionItem,
    mode: Mode,
    k: int,
    hops: int | None,
    node_keys: Mapping[str, str],
    *,
    answers: bool = False,
    label: str | None = None,
) -> Row:
    t = time.perf_counter()
    ctx = await kb.retrieve(item["q"], mode=mode, k=k, hops=hops)
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
    if "chain" in item:
        covered, total = chain_coverage(ctx, item["chain"], node_keys)
        row.chain_hit = covered == total
        row.chain_cov = covered / total if total else 1.0
    if answers:
        try:
            result = await kb.query(item["q"], mode=mode, k=k, hops=hops)
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


def pct(value: float | None, digits: int = 0) -> str:
    return "–" if value is None else f"{value:.{digits}f}%"


def render(s: Summary, modes: Sequence[EvalMode]) -> str:
    has_chains = s.has_chains
    loose_label = "Answer-string hit (loose)" if has_chains else "Context hit"
    chain_headers = "Chain hit | Chain coverage | " if has_chains else ""
    chain_dividers = "---|---|" if has_chains else ""
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
        f"| Group | Mode | {loose_label} | {chain_headers}Answer acc. | "
        "Context tokens | Retrieval p50 | Fallbacks |",
        f"|---|---|---|{chain_dividers}---|---|---|---|",
    ]
    for group in [*s.groups, None]:
        for mode in modes:
            c = s.cell(group, mode)
            name = group or "**all**"
            tokens = "–" if c["tokens"] is None else f"{c['tokens']:.0f}"
            latency = "–" if c["ms"] is None else f"{c['ms']:.0f} ms"
            chain_cells = f"{pct(c['chain_hit'])} | {pct(c['chain_cov'])} | " if has_chains else ""
            lines.append(
                f"| {name} | {mode} | {pct(c['context_hit'])} | {chain_cells}"
                f"{pct(c['answer_acc'])} | {tokens} | {latency} | {c['fallbacks']} |"
            )
    for group in ("multi_hop", "deep_hop"):
        if group not in s.groups or "hybrid" not in modes:
            continue
        hyb = s.cell(group, "hybrid")
        if hyb["context_hit"] is None:
            continue
        metric = "answer_acc" if hyb["answer_acc"] is not None else "context_hit"
        if hyb["chain_hit"] is not None:
            metric = "chain_hit"
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
        heading = "Hybrid answer-string misses (loose):" if has_chains else "Hybrid context misses:"
        lines += ["", heading]
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


def summary_data(summary: Summary, modes: Sequence[EvalMode]) -> dict[str, Any]:
    data = asdict(summary)
    data.pop("groups")
    if summary.graph_quality is None:
        data.pop("graph_quality")
    for row in data["rows"]:
        if row["chain_hit"] is None:
            row.pop("chain_hit")
            row.pop("chain_cov")
    if summary.has_chains:
        chain_metrics = {}
        for group in [*summary.groups, None]:
            cells = {mode: summary.cell(group, mode) for mode in modes}
            chain_metrics[group or "all"] = {
                mode: {metric: cell[metric] for metric in ("chain_hit", "chain_cov")}
                for mode, cell in cells.items()
            }
        data["chain_metrics"] = chain_metrics
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_embedding_arguments(parser)
    parser.add_argument("--llm", required=True, help="answer model")
    parser.add_argument("--extractor", help="extraction model (default: --llm)")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("-k", type=int, default=RetrievalConfig().k)
    parser.add_argument("--hops", type=int)
    parser.add_argument("--answers", action="store_true", help="also generate and score answers")
    parser.add_argument(
        "--equal-budget",
        action="store_true",
        help=f"add '{RAG_EQ}': rag at the same context size as hybrid",
    )
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
    modes = runner_modes(args)
    text = render(summary, modes)
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    (args.out / f"golden_{stamp}.md").write_text(text + "\n", encoding="utf-8")
    (args.out / f"golden_{stamp}.json").write_text(
        json.dumps(summary_data(summary, modes), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("\n" + text)


if __name__ == "__main__":
    main()
