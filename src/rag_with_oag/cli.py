"""Command line: ``rag-with-oag ingest`` and ``rag-with-oag query``.

Models come from the environment, e.g.::

    export RAG_OAG_EMBED_MODEL=text-embedding-3-small
    export RAG_OAG_LLM_MODEL=gpt-4o-mini        # plus *_BASE_URL / *_API_KEY as needed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from typing import Any

from pydantic import ValidationError

from rag_with_oag import __version__
from rag_with_oag.config import EmbedConfig, LLMConfig
from rag_with_oag.errors import RagOagError
from rag_with_oag.kb import KnowledgeBase
from rag_with_oag.retrieval.context import fact_ref
from rag_with_oag.types import MODES


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

    query = sub.add_parser("query", parents=[common], help="ask a question")
    query.add_argument("question")
    query.add_argument("--mode", choices=MODES, help="default: hybrid with --ontology, else rag")
    query.add_argument("-k", type=int, default=None, help="chunks to retrieve")
    query.add_argument("--hops", type=int, default=None, help="graph expansion depth")
    query.add_argument("--retrieve-only", action="store_true", help="print context, skip the LLM")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    try:
        kb = _open(args)
    except ValidationError as exc:
        missing = ", ".join(
            f"RAG_OAG_{'EMBED' if 'Embed' in exc.title else 'LLM'}_{str(e['loc'][0]).upper()}"
            for e in exc.errors()
        )
        print(f"error: missing or invalid model settings: {missing}", file=sys.stderr)
        return 2
    try:
        with kb:
            if args.command == "ingest":
                return _ingest(kb, args)
            return _query(kb, args)
    except (RagOagError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _open(args: argparse.Namespace) -> KnowledgeBase:
    # ingest needs the model only to extract the graph, which needs an ontology
    query_needs = args.command == "query" and not args.retrieve_only
    needs_llm = query_needs or (args.command == "ingest" and args.ontology is not None)
    return KnowledgeBase(
        args.kb,
        path=args.path,
        url=args.url,
        ontology=args.ontology,
        embedder=EmbedConfig(),  # type: ignore[call-arg]  # filled from env
        llm=LLMConfig() if needs_llm else None,  # type: ignore[call-arg]
    )


def _ingest(kb: KnowledgeBase, args: argparse.Namespace) -> int:
    report = kb.ingest(args.sources, concurrency=args.concurrency)
    if args.json:
        _print_json(asdict(report))
    else:
        print(
            f"processed {report.documents_processed}, skipped {report.documents_skipped} "
            f"(unchanged), chunks {report.chunks_written}, entities {report.entities_written}, "
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
