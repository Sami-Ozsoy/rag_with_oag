from __future__ import annotations

from collections.abc import Sequence

from rag_with_oag.config import RetrievalConfig
from rag_with_oag.embed.base import Embedder
from rag_with_oag.retrieval.context import assemble, pack_chunks
from rag_with_oag.store.repository import GraphRepository
from rag_with_oag.types import RetrievedContext


class RAGRetriever:
    """Chunk vector search, then neighbour expansion inside the token budget."""

    def __init__(self, repo: GraphRepository, embedder: Embedder, config: RetrievalConfig) -> None:
        self.repo = repo
        self.embedder = embedder
        self.config = config

    async def retrieve(
        self, question: str, *, k: int | None = None, vector: Sequence[float] | None = None
    ) -> RetrievedContext:
        if vector is None:
            [vector] = await self.embedder.embed([question])
        hits = await self.repo.search_chunks(vector, k or self.config.k)
        packed = pack_chunks(
            hits,
            budget=self.config.context_token_budget,
            expand_neighbors=self.config.expand_neighbors,
        )
        return assemble(question, "rag", chunks=packed)
