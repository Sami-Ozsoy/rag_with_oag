"""Hybrid: ontology facts and the source text behind them, in one context.

1. Entity linking (OAG seeds) and chunk vector search (RAG seeds) run in parallel.
2. The ontology subgraph around the seeds becomes facts.
3. Chunks that mention the subgraph's entities are ranked by similarity to the question.
4. RAG hits and these evidence chunks are fused with Reciprocal Rank Fusion.
5. The token budget is split between facts and texts.

If no entity can be linked, hybrid runs as rag and says so in ``mode_used``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from rag_with_oag.config import RetrievalConfig
from rag_with_oag.embed.base import Embedder
from rag_with_oag.ontology.model import Ontology
from rag_with_oag.retrieval.context import assemble, pack_chunks, pack_facts
from rag_with_oag.retrieval.linking import rrf
from rag_with_oag.retrieval.oag import OAGRetriever
from rag_with_oag.store.repository import GraphRepository
from rag_with_oag.types import RetrievedContext


class HybridRetriever:
    def __init__(
        self,
        repo: GraphRepository,
        embedder: Embedder,
        ontology: Ontology,
        config: RetrievalConfig,
    ) -> None:
        self.repo = repo
        self.embedder = embedder
        self.config = config
        self.oag = OAGRetriever(repo, embedder, ontology, config)

    async def retrieve(
        self,
        question: str,
        *,
        k: int | None = None,
        hops: int | None = None,
        vector: Sequence[float] | None = None,
    ) -> RetrievedContext:
        cfg = self.config
        k = k or cfg.k
        if vector is None:
            [vector] = await self.embedder.embed([question])
        seeds, rag_hits = await asyncio.gather(
            self.oag.linker.link(question, vector), self.repo.search_chunks(vector, k)
        )
        if not seeds:
            packed = pack_chunks(
                rag_hits, budget=cfg.context_token_budget, expand_neighbors=cfg.expand_neighbors
            )
            return assemble(question, "rag", chunks=packed)

        sub = await self.oag.subgraph(seeds, hops=hops)
        evidence = await self.repo.evidence_chunks(sub.entity_ids, vector, k)
        fused = [
            hit for hit, _ in rrf([rag_hits, evidence], key=lambda h: h.chunk_id, k=cfg.rrf_k)
        ][:k]

        fact_budget = int(cfg.context_token_budget * cfg.fact_budget_ratio)
        facts = pack_facts(sub.facts, budget=fact_budget)
        chunks = pack_chunks(
            fused,
            budget=cfg.context_token_budget - facts.token_count,
            expand_neighbors=cfg.expand_neighbors,
        )
        return assemble(question, "hybrid", facts=facts, chunks=chunks)
