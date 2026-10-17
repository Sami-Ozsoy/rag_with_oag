"""The facade users work with: KnowledgeBase (sync) and AsyncKnowledgeBase.

The async class is the real implementation. The sync class runs it on a
private event loop in a background thread, which keeps it usable from
notebooks that already run their own loop.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine, Iterable
from concurrent.futures import Future
from pathlib import Path
from types import TracebackType
from typing import Any, Self, TypeVar

from rag_with_oag.config import EmbedConfig, ExtractionConfig, LLMConfig, RetrievalConfig
from rag_with_oag.embed.base import Embedder, embedder_identity
from rag_with_oag.embed.openai_compat import OpenAIEmbedder
from rag_with_oag.errors import ConfigError, KnowledgeBaseClosedError, ModeNotAvailableError
from rag_with_oag.generation.answer import answer
from rag_with_oag.ingest.chunker import AdaptiveChunker, Chunker
from rag_with_oag.ingest.graph_builder import GraphBuilder
from rag_with_oag.ingest.loaders import Source
from rag_with_oag.ingest.pipeline import IngestPipeline
from rag_with_oag.llm.client import ChatModel, LLMClient
from rag_with_oag.ontology.model import Ontology
from rag_with_oag.retrieval.hybrid import HybridRetriever
from rag_with_oag.retrieval.oag import OAGRetriever
from rag_with_oag.retrieval.rag import RAGRetriever
from rag_with_oag.store.connection import GraphConnection
from rag_with_oag.store.queries import safe_identifier
from rag_with_oag.store.repository import GraphRepository
from rag_with_oag.types import MODES, IngestReport, Mode, QueryResult, RetrievedContext

T = TypeVar("T")

DEFAULT_PATH = "./rag_with_oag.db"


class AsyncKnowledgeBase:
    """One knowledge base = one graph in FalkorDB.

    Pass ``path`` for the embedded database (default ``./rag_with_oag.db``) or
    ``url`` for a FalkorDB server; nothing else changes.

    Without an ontology only ``rag`` mode is available. With one, ingest also
    extracts entities and relations (using ``extractor_llm``, or ``llm``) and
    ``oag`` and ``hybrid`` open up; ``hybrid`` becomes the default mode.
    ``llm`` is needed for ``query`` and extraction; ``retrieve`` works without it.
    """

    def __init__(
        self,
        name: str,
        *,
        embedder: EmbedConfig | Embedder,
        llm: LLMConfig | ChatModel | None = None,
        ontology: Ontology | str | Path | None = None,
        extractor_llm: LLMConfig | ChatModel | None = None,
        path: str | Path | None = None,
        url: str | None = None,
        retrieval: RetrievalConfig | None = None,
        extraction: ExtractionConfig | None = None,
        chunker: Chunker | None = None,
        connection: GraphConnection | None = None,
    ) -> None:
        try:
            self.name = safe_identifier(name)
        except ValueError:
            raise ConfigError(
                f"knowledge base name {name!r} must match [A-Za-z_][A-Za-z0-9_]*"
            ) from None
        if connection is not None and (path is not None or url is not None):
            raise ConfigError("pass either connection or path/url, not both")
        if path is not None and url is not None:
            raise ConfigError("pass either path (embedded) or url (FalkorDB server), not both")
        self.retrieval_config = retrieval or RetrievalConfig()
        self.embedder: Embedder = (
            OpenAIEmbedder(embedder) if isinstance(embedder, EmbedConfig) else embedder
        )
        self.llm: ChatModel | None = LLMClient(llm) if isinstance(llm, LLMConfig) else llm
        self.extractor_llm: ChatModel | None = (
            LLMClient(extractor_llm) if isinstance(extractor_llm, LLMConfig) else extractor_llm
        ) or self.llm
        self.ontology: Ontology | None = (
            Ontology.from_yaml(ontology) if isinstance(ontology, str | Path) else ontology
        )
        self.extraction_config = extraction or ExtractionConfig()
        # Only clients built here are closed here; objects passed in belong to the caller.
        self._owned_clients: list[Any] = [
            client
            for client, cfg in (
                (self.embedder, embedder),
                (self.llm, llm),
                (self.extractor_llm, extractor_llm),
            )
            if isinstance(cfg, EmbedConfig | LLMConfig)
        ]
        self._owns_connection = connection is None
        self._connection = connection or GraphConnection(
            path=None if url else (path or DEFAULT_PATH), url=url
        )
        self.store = GraphRepository(self._connection, self.name)
        self._chunker = chunker or AdaptiveChunker(self.retrieval_config)
        self._rag = RAGRetriever(self.store, self.embedder, self.retrieval_config)
        self._oag: OAGRetriever | None = None
        self._hybrid: HybridRetriever | None = None
        if self.ontology is not None:
            args = (self.store, self.embedder, self.ontology, self.retrieval_config)
            self._oag = OAGRetriever(*args)
            self._hybrid = HybridRetriever(*args)
        self._open_lock = asyncio.Lock()
        self._opened = False
        self._closed = False

    # --- lifecycle --------------------------------------------------------------

    async def open(self) -> Self:
        async with self._open_lock:
            if self._closed:
                raise KnowledgeBaseClosedError(f"knowledge base '{self.name}' is closed")
            if not self._opened:
                await self._connection.open()
                await self.store.init(
                    embed_model=embedder_identity(self.embedder), ontology=self.ontology
                )
                self._opened = True
        return self

    async def close(self) -> None:
        """Release the model clients and connection built here. Closing twice is harmless."""
        async with self._open_lock:
            if self._closed:
                return
            self._closed = True
            self._opened = False
            try:
                for client in self._owned_clients:
                    await client.aclose()
            finally:
                if self._owns_connection:
                    await self._connection.close()

    async def __aenter__(self) -> Self:
        return await self.open()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # --- operations -----------------------------------------------------------

    async def ingest(
        self, sources: Iterable[Source] | Source, *, concurrency: int = 4
    ) -> IngestReport:
        """Load, chunk, embed and store documents, and extract their graph if there
        is an ontology. Content that is already up to date is skipped."""
        await self.open()
        if isinstance(sources, str | Path) or not isinstance(sources, Iterable):
            sources = [sources]
        builder = None
        if self.ontology is not None:
            if self.extractor_llm is None:
                raise ConfigError(
                    "extraction needs a model; pass llm=... or extractor_llm=LLMConfig(...)"
                )
            builder = GraphBuilder(
                self.store,
                self.extractor_llm,
                self.embedder,
                self.ontology,
                self.extraction_config,
            )
        pipeline = IngestPipeline(self.store, self.embedder, self._chunker, builder)
        return await pipeline.run(sources, concurrency=concurrency)

    @property
    def default_mode(self) -> Mode:
        return "hybrid" if self.ontology is not None else "rag"

    async def retrieve(
        self,
        question: str,
        *,
        mode: Mode | None = None,
        k: int | None = None,
        hops: int | None = None,
    ) -> RetrievedContext:
        """Retrieval only, without an LLM answer, for plugging into your own pipeline."""
        await self.open()
        mode = mode or self.default_mode
        self._check_mode(mode)
        if mode == "rag":
            return await self._rag.retrieve(question, k=k)
        if mode == "oag":
            assert self._oag is not None
            return await self._oag.retrieve(question, hops=hops)
        assert self._hybrid is not None
        return await self._hybrid.retrieve(question, k=k, hops=hops)

    async def query(
        self,
        question: str,
        *,
        mode: Mode | None = None,
        k: int | None = None,
        hops: int | None = None,
    ) -> QueryResult:
        if self.llm is None:
            raise ConfigError("query() needs an llm; pass llm=LLMConfig(...) or use retrieve()")
        ctx = await self.retrieve(question, mode=mode, k=k, hops=hops)
        return await answer(self.llm, ctx)

    async def stats(self) -> dict[str, int]:
        await self.open()
        return await self.store.stats()

    def _check_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode != "rag" and self.ontology is None:
            raise ModeNotAvailableError(
                f"mode '{mode}' needs an ontology; pass ontology=Ontology.from_yaml(...)"
            )


class KnowledgeBase:
    """Synchronous facade over AsyncKnowledgeBase; same arguments and methods."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name=f"rag_with_oag:{name}", daemon=True
        )
        self._thread.start()
        try:
            self._async = self._run(_construct(name, kwargs))
        except BaseException:
            self._stop_loop()
            raise

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        if self._loop.is_closed():
            coro.close()
            raise KnowledgeBaseClosedError("knowledge base is closed")
        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    @property
    def name(self) -> str:
        return self._async.name

    @property
    def store(self) -> _SyncStore:
        return _SyncStore(self)

    @property
    def llm(self) -> ChatModel | None:
        return self._async.llm

    def ingest(self, sources: Iterable[Source] | Source, *, concurrency: int = 4) -> IngestReport:
        return self._run(self._async.ingest(sources, concurrency=concurrency))

    def retrieve(
        self,
        question: str,
        *,
        mode: Mode | None = None,
        k: int | None = None,
        hops: int | None = None,
    ) -> RetrievedContext:
        return self._run(self._async.retrieve(question, mode=mode, k=k, hops=hops))

    def query(
        self,
        question: str,
        *,
        mode: Mode | None = None,
        k: int | None = None,
        hops: int | None = None,
    ) -> QueryResult:
        return self._run(self._async.query(question, mode=mode, k=k, hops=hops))

    def stats(self) -> dict[str, int]:
        return self._run(self._async.stats())

    def close(self) -> None:
        if self._loop.is_closed():
            return
        try:
            self._run(self._async.close())
        finally:
            self._stop_loop()

    def _stop_loop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        self._loop.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class _SyncStore:
    """``kb.store.query(cypher, params)`` for the sync facade."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self._kb = kb

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        async def run() -> list[list[Any]]:
            await self._kb._async.open()
            return await self._kb._async.store.query(cypher, params)

        return self._kb._run(run())


async def _construct(name: str, kwargs: dict[str, Any]) -> AsyncKnowledgeBase:
    # Built on the background loop so its asyncio primitives bind to that loop.
    return AsyncKnowledgeBase(name, **kwargs)
