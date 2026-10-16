"""Ingest steps 5–10: extract, validate, resolve, embed entities, write, report.

Runs per document after its chunks are stored, so a document is already
searchable in rag mode while its graph is being built. A document is marked
extracted only when every batch got an answer from the model; otherwise the
next ingest retries it, and the extraction cache makes the batches that did
succeed free.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from rag_with_oag.config import ExtractionConfig
from rag_with_oag.embed.base import Embedder
from rag_with_oag.ingest.extractor import ChunkGraph, Extractor
from rag_with_oag.ingest.resolver import EntityResolver
from rag_with_oag.llm.client import ChatModel
from rag_with_oag.ontology.model import Ontology
from rag_with_oag.store.repository import GraphRepository
from rag_with_oag.types import Chunk, Document, content_hash

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BuildResult:
    entities: int = 0
    relations: int = 0
    validation_errors: list[str] = field(default_factory=list)
    unmapped: Counter[str] = field(default_factory=Counter)
    failed_batches: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.failed_batches


class GraphBuilder:
    def __init__(
        self,
        repo: GraphRepository,
        llm: ChatModel,
        embedder: Embedder,
        ontology: Ontology,
        config: ExtractionConfig,
    ) -> None:
        self.repo = repo
        self.llm = llm
        self.ontology = ontology
        self.config = config
        self.version = ontology.fingerprint
        self.extractor = Extractor(
            llm,
            ontology,
            mode=config.structured_mode,
            validation_retries=config.validation_retries,
        )
        self.resolver = EntityResolver(
            repo, embedder, ontology, similarity=config.resolution_similarity
        )
        # Resolution reads the graph to find existing entities, so two documents
        # resolving at once could each create the same entity. Extraction (the slow
        # part) stays concurrent; resolve + write runs one document at a time.
        self._write_lock = asyncio.Lock()

    def cache_key(self, chunk: Chunk) -> str:
        raw = f"{content_hash(chunk.text)}|{self.version}|{self.llm.model}"
        return hashlib.sha256(raw.encode()).hexdigest()

    async def build(self, doc: Document, chunks: Sequence[Chunk]) -> BuildResult:
        result = BuildResult()
        keys = {c.id: self.cache_key(c) for c in chunks}
        cached = (
            await self.repo.cached_extractions(list(keys.values())) if self.config.cache else {}
        )
        graphs: dict[str, ChunkGraph] = {
            c.id: ChunkGraph.from_json(cached[keys[c.id]]) for c in chunks if keys[c.id] in cached
        }
        todo = [c for c in chunks if c.id not in graphs]
        fresh: dict[str, ChunkGraph] = {}
        size = self.config.batch_size
        for i in range(0, len(todo), size):
            batch = todo[i : i + size]
            try:
                outcome = await self.extractor.extract(batch)
            except Exception as exc:
                logger.warning(
                    "extraction failed for %s chunks %d-%d: %s",
                    doc.source,
                    batch[0].idx,
                    batch[-1].idx,
                    exc,
                )
                result.failed_batches.append(
                    f"chunks {batch[0].idx}-{batch[-1].idx}: {type(exc).__name__}: {exc}"
                )
                continue
            fresh.update(outcome.graphs)
            result.validation_errors.extend(outcome.errors)
        if fresh and self.config.cache:
            await self.repo.cache_extractions({keys[cid]: g.to_json() for cid, g in fresh.items()})
        graphs.update(fresh)

        for graph in graphs.values():
            for term, count in graph.unmapped:
                result.unmapped[term] += count
        async with self._write_lock:
            write = await self.resolver.resolve(graphs)
            await self.repo.write_graph(write, version=self.version)
        result.entities = len(write.entities)
        result.relations = len(write.relations)
        if result.complete:
            await self.repo.mark_extracted(doc.id, self.version)
        return result
