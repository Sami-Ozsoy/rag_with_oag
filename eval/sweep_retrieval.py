"""Compare retrieval settings on an already ingested evaluation set."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, cast

from rag_with_oag import AsyncKnowledgeBase, EmbedConfig, Ontology, RagOagError, RetrievalConfig
from rag_with_oag.rerank import Reranker
from rag_with_oag.rerank.local import FastEmbedReranker
from rag_with_oag.types import MODES

if TYPE_CHECKING or __package__:
    from .run_golden import (
        HERE,
        RAG_EQ,
        EvalMode,
        QuestionItem,
        Summary,
        add_embedding_arguments,
        evaluate_modes,
        load_questions,
        make_embedder,
        pct,
        read_node_keys,
    )
else:
    from run_golden import (
        HERE,
        RAG_EQ,
        EvalMode,
        QuestionItem,
        Summary,
        add_embedding_arguments,
        evaluate_modes,
        load_questions,
        make_embedder,
        pct,
        read_node_keys,
    )

METRICS = ("chain_hit", "chain_cov", "context_hit", "tokens")
type Cells = dict[str, dict[str, dict[str, float | None]]]


def grid_values(text: str, *, sequence: bool) -> list[object]:
    """Parse comma-separated scalars or, for tuple fields, JSON arrays."""
    values: list[object] = []
    while True:
        text = text.strip()
        if sequence and text.startswith("["):
            value, end = json.JSONDecoder().raw_decode(text)
            values.append(value)
            rest = text[end:].lstrip()
            separator, tail = rest[:1], rest[1:]
            if separator not in ("", ","):
                raise ValueError("expected a comma after the grid array")
        else:
            value, separator, tail = text.partition(",")
            value = value.strip()
            if not value:
                raise ValueError("grid values must be non-empty")
            values.append([value] if sequence else value)
        if not separator:
            return values
        text = tail


def configurations(grids: Sequence[str], hops: str | None) -> list[tuple[RetrievalConfig, int]]:
    axes: dict[str, list[object]] = {}
    for spec in grids:
        name, separator, values = spec.partition("=")
        if not separator or name not in RetrievalConfig.model_fields:
            raise ValueError(f"invalid --grid {spec!r}; expected FIELD=V1,V2 for RetrievalConfig")
        axes.setdefault(name, []).extend(
            grid_values(
                values, sequence=RetrievalConfig.model_fields[name].annotation == tuple[str, ...]
            )
        )
    hop_overrides = (
        [RetrievalConfig.model_validate({"hops": value}).hops for value in hops.split(",")]
        if hops is not None
        else []
    )
    configs = [
        RetrievalConfig.model_validate(dict(zip(axes, values, strict=True)))
        for values in product(*axes.values())
    ]
    return [(config, hop) for hop in hop_overrides for config in configs] or [
        (config, config.hops) for config in configs
    ]


async def run_config(
    args: argparse.Namespace,
    ontology: Ontology,
    embedder: EmbedConfig,
    questions: dict[str, list[QuestionItem]],
    modes: Sequence[EvalMode],
    config: RetrievalConfig,
    hops: int,
    *,
    reranker: Reranker | None = None,
) -> Cells:
    kb = AsyncKnowledgeBase(
        args.kb,
        path=args.db,
        ontology=ontology,
        embedder=embedder,
        retrieval=config,
        reranker=reranker,
    )
    summary = Summary(model={}, ingest={})
    try:
        if (await kb.stats())["chunks"] == 0:
            raise ValueError(f"KB {args.kb!r} in {args.db} is empty; use an already ingested KB")
        node_keys = await read_node_keys(kb.store, ontology)
        for group, items in questions.items():
            for item in items[: args.limit]:
                summary.rows.extend(
                    await evaluate_modes(kb, group, item, modes, config.k, hops, node_keys)
                )
    finally:
        await kb.close()
    cells: Cells = {}
    for group in questions:
        by_mode = {mode: summary.cell(group, mode) for mode in modes}
        cells[group] = {
            mode: {metric: cell[metric] for metric in METRICS} for mode, cell in by_mode.items()
        }
    return cells


def print_config(hops: int, settings: dict[str, Any], cells: Cells) -> None:
    print(f"\nHops {hops} | {json.dumps(settings, ensure_ascii=False)}")
    print("| Group | Mode | Chain hit | Chain coverage | Answer-string hit (loose) | Mean tokens |")
    print("|---|---|---|---|---|---|")
    for group, modes in cells.items():
        for mode, cell in modes.items():
            tokens = cell["tokens"]
            print(
                f"| {group} | {mode} | {pct(cell['chain_hit'], 1)} | {pct(cell['chain_cov'], 1)} "
                f"| {pct(cell['context_hit'], 1)} | {'–' if tokens is None else f'{tokens:.0f}'} |"
            )
    print(flush=True)


def write_report(out: Path, report: dict[str, object]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=out.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.close()
            temporary.replace(out)
        finally:
            temporary.unlink(missing_ok=True)


async def sweep(
    args: argparse.Namespace,
    configs: Sequence[tuple[RetrievalConfig, int]],
    modes: Sequence[EvalMode],
) -> None:
    questions = load_questions(args.set)
    ontology = Ontology.from_yaml(args.set / "ontology.yaml")
    embedder = make_embedder(args)
    rerank_model = getattr(args, "rerank", None)
    reranker = FastEmbedReranker(rerank_model) if rerank_model is not None else None
    results: list[dict[str, object]] = []
    report: dict[str, object] = {
        "set": str(args.set),
        "kb": args.kb,
        "db": str(args.db),
        "embed": embedder.model,
        "rerank": rerank_model,
        "rag_lexical": list(dict.fromkeys(config.rag_lexical for config, _ in configs)),
        "limit_per_group": args.limit,
        "results": results,
    }
    print(f"Embedding `{embedder.model}`, reranker `{rerank_model or 'none'}`", flush=True)
    for config, hops in configs:
        cells = await run_config(
            args, ontology, embedder, questions, modes, config, hops, reranker=reranker
        )
        results.append({"hops": hops, "retrieval": config.model_dump(mode="json"), "groups": cells})
        write_report(args.out, report)
        settings = config.model_dump(mode="json", exclude_unset=True)
        settings["rag_lexical"] = config.rag_lexical
        print_config(hops, settings, cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_embedding_arguments(parser)
    parser.add_argument("--rerank", metavar="MODEL", help="FastEmbed cross-encoder model")
    parser.add_argument("--set", type=Path, required=True, help="evaluation set directory")
    parser.add_argument("--kb", help="knowledge base name (default: the set directory name)")
    parser.add_argument("--db", type=Path, default=HERE / "out" / "golden.db")
    parser.add_argument("--hops", help="comma-separated traversal overrides (default: config hops)")
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="FIELD=V1,V2",
        help="retrieval setting values; repeat for each field; tuple values use JSON arrays",
    )
    parser.add_argument("--limit", type=int, help="maximum questions per group")
    parser.add_argument("--modes", default=",".join((*MODES, RAG_EQ)))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    modes = cast(list[EvalMode], [mode.strip() for mode in args.modes.split(",")])
    if any(mode not in (*MODES, RAG_EQ) for mode in modes):
        parser.error(f"--modes must contain only {', '.join((*MODES, RAG_EQ))}")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.kb is None:
        args.kb = args.set.name or args.set.resolve().name
    if not args.db.is_file():
        parser.error(f"database {args.db} does not exist; use an already ingested KB")
    if args.out.resolve() == args.db.resolve():
        parser.error("--out must differ from --db")
    try:
        configs = configurations(args.grid, args.hops)
        asyncio.run(sweep(args, configs, modes))
    except (OSError, ValueError, RagOagError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
