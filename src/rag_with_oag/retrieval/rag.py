"""Rank chunks by vector search with optional BM25 fusion and reranking.

BM25 keeps one token list per chunk in memory, rebuilt lazily after writes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace

from rag_with_oag.config import RetrievalConfig
from rag_with_oag.embed.base import Embedder, embed_queries
from rag_with_oag.rerank import Reranker
from rag_with_oag.retrieval.context import pack_context
from rag_with_oag.retrieval.linking import rrf
from rag_with_oag.store.repository import GraphRepository
from rag_with_oag.types import ChunkHit, RetrievedContext


async def search_rag_hits(
    repo: GraphRepository,
    question: str,
    vector: Sequence[float],
    k: int,
    config: RetrievalConfig,
    reranker: Reranker | None,
) -> list[ChunkHit]:
    candidate_k = max(k, config.rerank_candidates) if reranker is not None else k
    if config.rag_lexical:
        by_text, by_vector = await asyncio.gather(
            repo.search_chunks_text(question, candidate_k * 3),
            repo.search_chunks(vector, candidate_k * 3),
        )
        fused = rrf(
            [by_text, by_vector],
            key=lambda h: h.chunk_id,
            k=config.rrf_k,
            weights=[config.rag_text_weight, 1.0],
        )
        hits = [replace(hit, score=score) for hit, score in fused[:candidate_k]]
    else:
        hits = await repo.search_chunks(vector, candidate_k)
    if reranker is None or not hits:
        return hits
    scores = await reranker.rerank(question, [hit.text for hit in hits])
    ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [replace(hit, score=score) for hit, score in ranked[:k]]


class RAGRetriever:
    """Rank chunk hits and expand neighbours inside the token budget."""

    def __init__(
        self,
        repo: GraphRepository,
        embedder: Embedder,
        config: RetrievalConfig,
        *,
        reranker: Reranker | None = None,
    ) -> None:
        self.repo = repo
        self.embedder = embedder
        self.config = config
        self.reranker = reranker

    async def retrieve(
        self, question: str, *, k: int | None = None, vector: Sequence[float] | None = None
    ) -> RetrievedContext:
        if vector is None:
            [vector] = await embed_queries(self.embedder, [question])
        hits = await search_rag_hits(
            self.repo, question, vector, k or self.config.k, self.config, self.reranker
        )
        return pack_context(
            question,
            "rag",
            budget=self.config.context_token_budget,
            hits=hits,
            expand_neighbors=self.config.expand_neighbors,
        )
