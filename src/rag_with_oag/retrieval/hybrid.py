"""Hybrid: ontology facts and the source text behind them, in one context.

1. Entity linking and RAG retrieval run in parallel. RAG hits can use lexical fusion and reranking.
2. The ontology walk produces seed facts, then relation and entity facts by hop.
3. Rank chunks from documents that mention seeds and evidence chunks that mention subgraph entities.
4. Reserve up to ``hybrid_text_reserve`` unique hits, alternating seed-document and RAG hits.
5. Fuse seed-document, RAG and evidence hits with Reciprocal Rank Fusion.
6. Pack reserved hits before allocating up to ``fact_budget_ratio`` to facts after section overhead.
   Pack remaining fused hits and neighbours within the remaining budget.

If no entity can be linked, hybrid runs as rag and says so in ``mode_used``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from itertools import chain, zip_longest

from rag_with_oag.config import RetrievalConfig
from rag_with_oag.embed.base import Embedder, embed_queries
from rag_with_oag.ontology.model import Ontology
from rag_with_oag.rerank import Reranker
from rag_with_oag.retrieval.context import pack_context
from rag_with_oag.retrieval.linking import rrf
from rag_with_oag.retrieval.oag import OAGRetriever
from rag_with_oag.retrieval.rag import search_rag_hits
from rag_with_oag.store.repository import GraphRepository
from rag_with_oag.types import ChunkHit, RetrievedContext


class HybridRetriever:
    def __init__(
        self,
        repo: GraphRepository,
        embedder: Embedder,
        ontology: Ontology,
        config: RetrievalConfig,
        *,
        reranker: Reranker | None = None,
    ) -> None:
        self.repo = repo
        self.embedder = embedder
        self.config = config
        self.reranker = reranker
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
            [vector] = await embed_queries(self.embedder, [question])
        seeds, rag_hits = await asyncio.gather(
            self.oag.linker.link(question, vector),
            search_rag_hits(self.repo, question, vector, k, cfg, self.reranker),
        )
        if not seeds:
            return pack_context(
                question,
                "rag",
                budget=cfg.context_token_budget,
                hits=rag_hits,
                expand_neighbors=cfg.expand_neighbors,
            )

        sub = await self.oag.subgraph(seeds, hops=hops)
        seed_hits, evidence = await asyncio.gather(
            self.repo.seed_document_chunks([s.id for s in seeds], vector, k),
            self.repo.evidence_chunks(sub.entity_ids, vector, k),
        )
        prioritized: dict[str, ChunkHit] = {}
        for hit in chain.from_iterable(zip_longest(seed_hits, rag_hits)):
            if len(prioritized) == cfg.hybrid_text_reserve:
                break
            if hit is not None:
                prioritized.setdefault(hit.chunk_id, hit)
        reserve = len(prioritized)
        for hit, _ in rrf([seed_hits, rag_hits, evidence], key=lambda h: h.chunk_id, k=cfg.rrf_k):
            prioritized.setdefault(hit.chunk_id, hit)

        return pack_context(
            question,
            "hybrid",
            budget=cfg.context_token_budget,
            hits=list(prioritized.values())[: max(k, reserve)],
            facts=sub.facts,
            reserve=reserve,
            fact_budget_ratio=cfg.fact_budget_ratio,
            expand_neighbors=cfg.expand_neighbors,
        )
