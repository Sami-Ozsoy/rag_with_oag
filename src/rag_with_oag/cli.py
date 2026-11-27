"""Command line for managing documents, querying, and designing ontologies.

Models come from the environment, e.g.::

    export RAG_OAG_EMBED_MODEL=text-embedding-3-small
    export RAG_OAG_LLM_MODEL=gpt-4o-mini        # plus *_BASE_URL / *_API_KEY as needed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from rag_with_oag import __version__
from rag_with_oag.config import EmbedConfig, LLMConfig
from rag_with_oag.errors import RagOagError
from rag_with_oag.kb import KnowledgeBase
from rag_with_oag.llm.client import LLMClient
from rag_with_oag.ontology.model import Ontology
from rag_with_oag.ontology.questions import load_questions
from rag_with_oag.ontology.suggest import load_documents, suggest_with_coverage
from rag_with_oag.retrieval.context import fact_ref
from rag_with_oag.types import MODES

if TYPE_CHECKING:
    from rag_with_oag.ontology.preview import PreviewReport
    from rag_with_oag.ontology.questions import QuestionCoverageReport


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-with-oag", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--kb", default="default", help="knowledge base name (default: default)")
    where = common.add_mutually_exclusive_group()
    where.add_argument("--path", help="embedded database file (default: ./rag_with_oag.db)")
    where.add_argument("--url", help="FalkorDB server, e.g. redis://localhost:6379")
    common.add_argument("--ontology", help="ontology YAML; enables extraction, oag and hybrid")
    common.add_argument("--json", action="store_true", help="machine-readable output")
    common.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", parents=[common], help="ingest files or directories")
    ingest.add_argument("sources", nargs="+")
    ingest.add_argument("--concurrency", type=int, default=4)
    ingest.add_argument("--no-replace", action="store_true", help="keep older document versions")

    delete = sub.add_parser("delete", parents=[common], help="delete documents by ID or source")
    delete.add_argument("targets", nargs="+")

    query = sub.add_parser("query", parents=[common], help="ask a question")
    query.add_argument("question")
    query.add_argument("--mode", choices=MODES, help="default: hybrid with --ontology, else rag")
    query.add_argument("-k", type=int, default=None, help="chunks to retrieve")
    query.add_argument("--hops", type=int, default=None, help="graph expansion depth")
    query.add_argument("--retrieve-only", action="store_true", help="print context, skip the LLM")

    suggest = sub.add_parser("suggest", help="suggest an ontology from sample documents")
    suggest.add_argument("sources", nargs="+")
    suggest.add_argument("-o", metavar="FILE", dest="output", type=Path)
    suggest.add_argument("--force", action="store_true", help="overwrite the output file")
    suggest.add_argument("--base", metavar="ONTOLOGY_YAML", type=Path)
    suggest.add_argument("--name")
    suggest.add_argument("--instructions")
    suggest.add_argument("--sample-tokens", type=int, default=12_000)
    suggest.add_argument("--questions", metavar="FILE", type=Path, help="text or YAML questions")
    suggest.add_argument("--no-repair", action="store_true", help="skip the ontology repair round")
    suggest.add_argument(
        "--hops", type=_positive_int, default=3, help="question coverage hop limit"
    )
    suggest.add_argument("-v", "--verbose", action="store_true")

    preview = sub.add_parser(
        "preview", help="preview what an ontology extracts from sample documents"
    )
    preview.add_argument("sources", nargs="+")
    preview.add_argument("--ontology", required=True, type=Path, help="ontology YAML file")
    preview.add_argument("--docs", type=_positive_int, default=5)
    preview.add_argument("--chunks-per-doc", type=_positive_int, default=6)
    preview.add_argument("--concurrency", type=_positive_int, default=4)
    preview.add_argument("--json", action="store_true", help="machine-readable output")
    preview.add_argument("-v", "--verbose", action="store_true")

    check = sub.add_parser("check-questions", help="check whether an ontology covers questions")
    check.add_argument("--ontology", required=True, type=Path, help="ontology YAML file")
    check.add_argument("--questions", required=True, type=Path, help="text or YAML questions file")
    check.add_argument("--hops", type=_positive_int, default=3)
    check.add_argument("--concurrency", type=_positive_int, default=4)
    check.add_argument("--json", action="store_true", help="machine-readable output")
    check.add_argument("-v", "--verbose", action="store_true")
    return parser


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    if args.command == "suggest":
        return _suggest(args)
    if args.command == "preview":
        return _preview(args)
    if args.command == "check-questions":
        return _check_questions(args)
    try:
        kb = _open(args)
    except ValidationError as exc:
        return _model_settings_error(exc)
    except (RagOagError, ValueError, OSError) as exc:  # bad ontology file, kb name, ...
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        with kb:
            if args.command == "ingest":
                return _ingest(kb, args)
            if args.command == "delete":
                return _delete(kb, args)
            return _query(kb, args)
    except (RagOagError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _model_settings_error(exc: ValidationError) -> int:
    prefix = "EMBED" if "Embed" in exc.title else "LLM"
    missing = ", ".join(f"RAG_OAG_{prefix}_{str(e['loc'][0]).upper()}" for e in exc.errors())
    print(f"error: missing or invalid model settings: {missing}", file=sys.stderr)
    return 2


def _make_llm(config: LLMConfig) -> LLMClient:
    return LLMClient(config)


def _suggest(args: argparse.Namespace) -> int:
    output: Path | None = args.output
    try:
        if output is not None and not args.force and (output.exists() or output.is_symlink()):
            print(f"error: output file already exists: {output} (use --force)", file=sys.stderr)
            return 2
        try:
            questions = load_questions(args.questions) if args.questions is not None else []
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        try:
            config = LLMConfig()  # type: ignore[call-arg]  # filled from env
        except ValidationError as exc:
            return _model_settings_error(exc)
        base = Ontology.from_yaml(args.base) if args.base else None
        documents = load_documents(args.sources)

        async def generate() -> tuple[Ontology, QuestionCoverageReport | None]:
            llm = _make_llm(config)
            try:
                return await suggest_with_coverage(
                    documents,
                    llm,
                    base=base,
                    name=args.name,
                    instructions=args.instructions,
                    sample_tokens=args.sample_tokens,
                    questions=questions,
                    repair=not args.no_repair,
                    hops=args.hops,
                )
            finally:
                close = getattr(llm, "aclose", None)
                if close is not None:
                    await close()

        ontology, report = asyncio.run(generate())
        text = (
            f"# Suggested by rag_with_oag from {len(documents)} documents. Review before use.\n"
            + ontology.to_yaml()
        )
        if output is None:
            print(text, end="")
        else:
            try:
                with output.open("w" if args.force else "x", encoding="utf-8") as file:
                    file.write(text)
            except FileExistsError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2

            entities, relations = len(ontology.entities), len(ontology.relations)
            summary = f"Wrote {output} (entity types={entities}, relation types={relations})"
            if base is not None:
                summary += f" (added entity types={entities - len(base.entities)}, "
                summary += f"relation types={relations - len(base.relations)})"
            print(summary, file=sys.stderr)
        if report is not None:
            _print_question_summary(report)
        return 0
    except (RagOagError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _print_question_summary(report: QuestionCoverageReport) -> None:
    counts = report.counts
    gaps = report.uncovered
    print(
        f"Questions: {counts['covered']} covered, {counts['text']} answered from text, "
        f"{len(gaps)} gaps "
        f"({counts['not_covered']} not covered, {counts['invalid_path']} invalid path, "
        f"{counts['too_many_hops']} too many hops)",
        file=sys.stderr,
    )
    for item in gaps:
        label = item.status.replace("_", " ")
        detail = item.reason
        if item.status == "too_many_hops":
            label += f" ({item.hops} > {report.hops_limit})"
            detail = item.path or item.reason
        question = " ".join(item.question.splitlines())
        detail = " ".join(detail.splitlines())
        print(f"  {label}: {question} — {detail}", file=sys.stderr)


def _check_questions(args: argparse.Namespace) -> int:
    try:
        ontology = Ontology.from_yaml(args.ontology)
        questions = load_questions(args.questions)
    except (RagOagError, ValueError, OSError, AttributeError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        config = LLMConfig()  # type: ignore[call-arg]  # filled from env
    except ValidationError as exc:
        return _model_settings_error(exc)

    async def generate() -> QuestionCoverageReport:
        llm = _make_llm(config)
        try:
            return await ontology.acheck_questions(
                questions, llm, hops=args.hops, concurrency=args.concurrency
            )
        finally:
            close = getattr(llm, "aclose", None)
            if close is not None:
                await close()

    try:
        report = asyncio.run(generate())
    except (RagOagError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _print_json(asdict(report))
    else:
        print(
            f"Checked {len(report.questions)} questions against ontology {report.ontology!r} "
            f"(hops limit {report.hops_limit}): {report.covered} covered, "
            f"{len(report.text_only)} answered from text, {len(report.uncovered)} gaps."
        )
        for item in report.questions:
            label = item.status.replace("_", " ")
            detail = item.path if item.status in {"covered", "too_many_hops"} else item.reason
            print(f"  {label:<13}  {' '.join(item.question.splitlines())}")
            print(f"    {' '.join((detail or '').splitlines())}")
        usage = report.token_usage
        print(
            f"Tokens: prompt {usage.get('prompt_tokens', 0)}, "
            f"completion {usage.get('completion_tokens', 0)}"
        )
    return 0


def _preview(args: argparse.Namespace) -> int:
    try:
        ontology = Ontology.from_yaml(args.ontology)
    except (RagOagError, ValueError, OSError, AttributeError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        config = LLMConfig()  # type: ignore[call-arg]  # filled from env
    except ValidationError as exc:
        return _model_settings_error(exc)

    async def generate() -> PreviewReport:
        llm = _make_llm(config)
        try:
            return await ontology.apreview(
                args.sources,
                llm,
                documents=args.docs,
                chunks_per_doc=args.chunks_per_doc,
                concurrency=args.concurrency,
            )
        finally:
            close = getattr(llm, "aclose", None)
            if close is not None:
                await close()

    try:
        report = asyncio.run(generate())
    except (RagOagError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _print_json(asdict(report))
    else:
        _print_preview(report)
    return 1 if len(report.failed_batches) == report.batches else 0


def _print_preview(report: PreviewReport) -> None:
    print(
        f"Previewed ontology {report.ontology!r} on {report.documents_sampled} of "
        f"{report.documents_available} documents ({report.chunks_sampled} chunks, "
        f"{report.batches} batches)."
    )
    print(
        f"Coverage: {report.chunks_with_entities}/{report.chunks_sampled} chunks with entities, "
        f"{report.chunks_with_relations}/{report.chunks_sampled} with relations."
    )
    print("\nEntity types:")
    entity_width = max((len(entity.name) for entity in report.entity_types), default=0)
    for entity in report.entity_types:
        summary = f"{entity.mentions} mentions"
        if entity.mentions:
            summary += f", {entity.distinct} distinct"
            if entity.key_fill_rate is not None:
                summary += f", key {entity.key} {entity.key_fill_rate:.0%}"
            summary += f"   e.g. {', '.join(entity.examples)}"
        print(f"  {entity.name:<{entity_width}}  {summary}")
    if report.relation_types:
        print("\nRelation types:")
        relation_width = max(len(relation.name) for relation in report.relation_types)
        endpoints = [
            f"({relation.source} -> {relation.target})" for relation in report.relation_types
        ]
        endpoint_width = max(map(len, endpoints))
        for relation, pair in zip(report.relation_types, endpoints, strict=True):
            examples = f"   e.g. {', '.join(relation.examples)}" if relation.examples else ""
            print(
                f"  {relation.name:<{relation_width}}  {pair:<{endpoint_width}}  "
                f"{relation.count}{examples}"
            )
    if report.dropped:
        print("\nDropped:")
        for group in report.dropped:
            label = f"{group.kind} {group.type_name}" if group.type_name is not None else group.kind
            pairs = ", ".join(f"{pair} {count}" for pair, count in group.observed.items())
            details = f"  ({pairs})" if pairs else ""
            print(f"  {label}  {group.count}{details}")
    if report.hints:
        print("\nHints:")
        for hint in report.hints:
            print(f"  - {hint}")
    usage = report.token_usage
    print(
        f"\nTokens: prompt {usage.get('prompt_tokens', 0)}, "
        f"completion {usage.get('completion_tokens', 0)}"
    )
    for failure in report.failed_batches:
        print(f"  failed: {failure}", file=sys.stderr)


def _open(args: argparse.Namespace) -> KnowledgeBase:
    # ingest needs the model only to extract the graph, which needs an ontology
    query_needs = args.command == "query" and not args.retrieve_only
    needs_llm = query_needs or (args.command == "ingest" and args.ontology is not None)
    return KnowledgeBase(
        args.kb,
        path=args.path,
        url=args.url,
        # A Path, so a file name is never mistaken for inline YAML.
        ontology=Path(args.ontology) if args.ontology else None,
        embedder=EmbedConfig(),  # type: ignore[call-arg]  # filled from env
        llm=LLMConfig() if needs_llm else None,  # type: ignore[call-arg]
    )


def _ingest(kb: KnowledgeBase, args: argparse.Namespace) -> int:
    report = kb.ingest(args.sources, concurrency=args.concurrency, replace=not args.no_replace)
    if args.json:
        _print_json(asdict(report))
    else:
        print(
            f"processed {report.documents_processed}, skipped {report.documents_skipped} "
            f"(unchanged), replaced {report.documents_replaced}, chunks {report.chunks_written}, "
            f"entities {report.entities_written}, "
            f"relations {report.relations_written}"
        )
        if report.validation_errors:
            print(
                f"  {len(report.validation_errors)} extraction items failed validation (see --json)"
            )
        if report.unmapped_concepts:
            top = ", ".join(f"{t} ({c})" for t, c in report.unmapped_concepts[:10])
            print(f"  concepts the ontology cannot express: {top}")
        for source, error in report.failures:
            print(f"  failed: {source}: {error}", file=sys.stderr)
    return 0 if report.ok else 1


def _delete(kb: KnowledgeBase, args: argparse.Namespace) -> int:
    report = kb.delete(*args.targets)
    if args.json:
        _print_json(asdict(report))
    else:
        print(
            f"deleted documents {report.documents}, chunks {report.chunks}, "
            f"relations {report.relations_deleted}, entities {report.entities_deleted}, "
            f"trimmed relations {report.relations_trimmed}"
        )
        for target in report.not_found:
            print(f"  not found: {target}")
    return 0


def _query(kb: KnowledgeBase, args: argparse.Namespace) -> int:
    if args.retrieve_only:
        ctx = kb.retrieve(args.question, mode=args.mode, k=args.k, hops=args.hops)
        if args.json:
            _print_json(asdict(ctx))
        else:
            print(ctx.prompt_context or "(no results)")
        return 0
    result = kb.query(args.question, mode=args.mode, k=args.k, hops=args.hops)
    if args.json:
        _print_json(asdict(result))
        return 0
    print(result.answer)
    chunks = {hit.chunk_id: hit.location for hit in result.chunks}
    facts = {fact_ref(f): f.render() for f in result.facts}
    if result.citations:
        print("\nSources:")
        for c in result.citations:
            print(f"  [{c.id}] {chunks.get(c.ref) or facts.get(c.ref) or c.ref}")
    if args.mode and result.mode_used != args.mode:
        print(f"\n(mode fell back to {result.mode_used})")
    return 0


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
