"""Ingest orchestration.

Steps 1–4 (load → chunk → embed → write) make a document queryable in rag
mode. With an ontology, steps 5–10 (extract → validate → resolve → embed
entities → write → report) run right after and open oag and hybrid mode; see
``graph_builder``. A document whose chunks are stored but whose graph is
missing (an ontology added later, or a failed extraction) only runs 5–10.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rag_with_oag._tokens import count_tokens
from rag_with_oag.embed.base import Embedder, embed_documents, embedder_identity
from rag_with_oag.errors import EmbeddingMismatchError
from rag_with_oag.ingest.chunker import Chunker, page_index
from rag_with_oag.ingest.graph_builder import BuildResult, GraphBuilder
from rag_with_oag.ingest.loaders import Source, iter_paths, load_path
from rag_with_oag.store.repository import GraphRepository
from rag_with_oag.types import Chunk, Document, IngestReport, chunk_id

logger = logging.getLogger(__name__)


MAX_REPORTED_ERRORS = 200


@dataclass(slots=True)
class _DocOutcome:
    chunks_written: int
    graph: BuildResult | None


class IngestPipeline:
    def __init__(
        self,
        repo: GraphRepository,
        embedder: Embedder,
        chunker: Chunker,
        graph_builder: GraphBuilder | None = None,
    ) -> None:
        self.repo = repo
        self.embedder = embedder
        self.chunker = chunker
        self.graph_builder = graph_builder

    async def run(self, sources: Iterable[Source], *, concurrency: int = 4) -> IngestReport:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        report = IngestReport()
        meters = [self.embedder.usage]
        if (
            self.graph_builder is not None
            and self.graph_builder.llm.usage is not self.embedder.usage
        ):
            meters.append(self.graph_builder.llm.usage)
        usage_before = [Counter(m) for m in meters]
        unmapped: Counter[str] = Counter()
        semaphore = asyncio.Semaphore(concurrency)

        async def handle(source: Source) -> None:
            async with semaphore:
                label = source.source if isinstance(source, Document) else str(source)
                try:
                    doc = (
                        source
                        if isinstance(source, Document)
                        else await asyncio.to_thread(
                            load_path,
                            source,  # type: ignore[arg-type]
                        )
                    )
                    outcome = await self._ingest_document(doc)
                except EmbeddingMismatchError:
                    raise  # every later document would fail the same way
                except Exception as exc:
                    logger.warning("ingest failed for %s: %s", label, exc, exc_info=True)
                    report.failures.append((label, f"{type(exc).__name__}: {exc}"))
                    return
                if outcome is None:
                    report.documents_skipped += 1
                    return
                report.documents_processed += 1
                report.chunks_written += outcome.chunks_written
                if outcome.graph is not None:
                    _merge_graph(report, unmapped, doc.source, outcome.graph)

        try:
            async with asyncio.TaskGroup() as tg:
                for source in _expand(sources, report):
                    tg.create_task(handle(source))
        except ExceptionGroup as group:
            # Surface the first real error, not a group the caller has to unpack.
            mismatch = group.subgroup(EmbeddingMismatchError)
            first = mismatch.exceptions[0] if mismatch else group.exceptions[0]
            raise first from None

        usage: Counter[str] = Counter()
        for meter, before in zip(meters, usage_before, strict=True):
            usage.update(Counter(meter) - before)
        report.token_usage = dict(usage)
        report.unmapped_concepts = unmapped.most_common()
        return report

    async def _ingest_document(self, doc: Document) -> _DocOutcome | None:
        """Bring one document up to date; None if there was nothing to do."""
        state = await self.repo.document_state(doc.id)
        builder = self.graph_builder
        needs_graph = builder is not None and state.extracted_version != builder.version
        if state.complete and not needs_graph:
            return None
        if state.complete:
            chunks = await self.repo.document_chunks(doc.id)
            written = 0
        else:
            chunks = await asyncio.to_thread(build_chunks, doc, self.chunker)
            if not chunks:
                raise ValueError("document has no text to index")
            vectors = await embed_documents(self.embedder, [embedding_text(doc, c) for c in chunks])
            await self.repo.ensure_embedding_dim(
                model=embedder_identity(self.embedder), dim=len(vectors[0])
            )
            await self.repo.write_document(doc, chunks, vectors)
            written = len(chunks)
        graph = await builder.build(doc, chunks) if builder is not None and needs_graph else None
        return _DocOutcome(chunks_written=written, graph=graph)


def build_chunks(doc: Document, chunker: Chunker) -> list[Chunk]:
    return [
        Chunk(
            id=chunk_id(doc.id, idx),
            doc_id=doc.id,
            idx=idx,
            text=span.text,
            token_count=count_tokens(span.text),
            start_char=span.start,
            end_char=span.end,
            section_path=span.section_path,
            page_index=page_index(doc.text, span.start),
            chunk_strategy=span.strategy,
        )
        for idx, span in enumerate(chunker.split(doc.text))
    ]


def embedding_text(doc: Document, chunk: Chunk) -> str:
    """Prefix the title and section path so short chunks keep their context in vector space."""
    header = " > ".join((doc.title, *chunk.section_path))
    return f"{header}\n\n{chunk.text}"


def _merge_graph(
    report: IngestReport, unmapped: Counter[str], source: str, graph: BuildResult
) -> None:
    report.entities_written += graph.entities
    report.relations_written += graph.relations
    unmapped.update(graph.unmapped)
    room = MAX_REPORTED_ERRORS - len(report.validation_errors)
    report.validation_errors.extend((source, e) for e in graph.validation_errors[: max(room, 0)])
    # The chunks are searchable; only the graph part is retried on the next ingest.
    report.failures.extend((source, f"extraction: {f}") for f in graph.failed_batches)


def _expand(sources: Iterable[Source], report: IngestReport) -> Sequence[Source]:
    expanded: list[Source] = []
    for source in sources:
        if isinstance(source, Document):
            expanded.append(source)
            continue
        try:
            expanded.extend(iter_paths([source]))
        except FileNotFoundError as exc:
            report.failures.append((str(source), str(exc)))
    return expanded
