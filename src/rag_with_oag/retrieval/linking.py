"""Question -> entities that serve as entry points into the graph.

Two signals are fused with weighted Reciprocal Rank Fusion: full-text matches
on folded entity names (prefix terms, so Turkish suffixes like "Atlas
Lojistik'in" still match) and vector similarity between the question and
entity embeddings. A name written in the question is the stronger evidence,
so text ranks weigh more, and text hits scoring under half the best one (a
single generic word such as "sözleşme" matching every contract) are dropped.
Vector matches below ``link_min_similarity`` are ignored, so an unrelated
question links nothing and hybrid mode can fall back to rag honestly.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence

from rag_with_oag._text import fulltext_query
from rag_with_oag.config import RetrievalConfig
from rag_with_oag.store.repository import GraphRepository
from rag_with_oag.types import EntityHit

TEXT_WEIGHT = 1.0
VECTOR_WEIGHT = 0.5
TEXT_MIN_RATIO = 0.5


def rrf[T](
    rankings: Sequence[Sequence[T]],
    *,
    key: Callable[[T], Hashable],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[T, float]]:
    """Reciprocal Rank Fusion: sum of weight/(k + rank) over the lists an item appears in."""
    scores: dict[Hashable, float] = {}
    first: dict[Hashable, T] = {}
    for i, ranking in enumerate(rankings):
        weight = weights[i] if weights is not None else 1.0
        for rank, item in enumerate(ranking, start=1):
            ident = key(item)
            scores[ident] = scores.get(ident, 0.0) + weight / (k + rank)
            first.setdefault(ident, item)
    ordered = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [(first[i], scores[i]) for i in ordered]


class EntityLinker:
    def __init__(self, repo: GraphRepository, config: RetrievalConfig) -> None:
        self.repo = repo
        self.config = config

    async def link(self, question: str, vector: Sequence[float]) -> list[EntityHit]:
        k = self.config.link_k
        # Over-fetch: vector queries cannot be combined with filters inside the index.
        by_vector = [
            hit
            for hit in await self.repo.search_entities_vector(vector, k * 3)
            if hit.score >= self.config.link_min_similarity
        ]
        by_text = await self.repo.search_entities_text(fulltext_query(question), k * 3)
        if by_text:
            best = by_text[0].score
            by_text = [hit for hit in by_text if hit.score >= best * TEXT_MIN_RATIO]
        fused = rrf(
            [by_text, by_vector],
            key=lambda h: h.id,
            k=self.config.rrf_k,
            weights=[TEXT_WEIGHT, VECTOR_WEIGHT],
        )
        return [
            EntityHit(id=hit.id, name=hit.name, type=hit.type, score=score)
            for hit, score in fused[:k]
        ]
