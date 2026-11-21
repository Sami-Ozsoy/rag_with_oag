"""Question -> entities that serve as entry points into the graph.

By default, literal keys and whole-token matches of stored normalized names
are the only seeds when present. A question that writes one entity in full
and another only by a short name links only the first if the short name
does not match the second entity's stored normalized name.

Without anchors, or with anchors disabled, weighted Reciprocal Rank Fusion
combines prefix full-text matches and vector similarity. Text ranks weigh
more, with hits below half the best text score discarded. Vector hits below
``link_min_similarity`` are ignored.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import replace

from rag_with_oag._text import fold, fulltext_query
from rag_with_oag.config import RetrievalConfig
from rag_with_oag.store.repository import GraphRepository
from rag_with_oag.types import EntityHit

TEXT_WEIGHT = 1.0
VECTOR_WEIGHT = 0.5
TEXT_MIN_RATIO = 0.5
_KEY_PUNCTUATION = "“”\"'‘’()[]{},;:?!."
_APOSTROPHE = re.compile(r"['‘’]")


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
    def __init__(
        self,
        repo: GraphRepository,
        config: RetrievalConfig,
        *,
        keyed_types: Sequence[tuple[str, str]] = (),
    ) -> None:
        self.repo = repo
        self.config = config
        self.keyed_types = tuple(keyed_types)

    async def link(self, question: str, vector: Sequence[float]) -> list[EntityHit]:
        k = self.config.link_k
        by_text = None
        if self.config.link_exact_anchors:
            by_text = await self.repo.search_entities_text(fulltext_query(question), k * 3)
            anchors = await self._exact_anchors(question, by_text)
            if anchors:
                return anchors[:k]

        # Over-fetch: vector queries cannot be combined with filters inside the index.
        by_vector = [
            hit
            for hit in await self.repo.search_entities_vector(vector, k * 3)
            if hit.score >= self.config.link_min_similarity
        ]
        if by_text is None:
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

    async def _exact_anchors(self, question: str, by_text: Sequence[EntityHit]) -> list[EntityHit]:
        positions: dict[str, int] = {}
        for position, token in enumerate(question.split()):
            value = _APOSTROPHE.split(token.strip(_KEY_PUNCTUATION), maxsplit=1)[0]
            if value:
                positions.setdefault(value, position)
                positions.setdefault(value.upper(), position)

        keyed = []
        values = list(positions)
        for label, key in self.keyed_types:
            keyed.extend(await self.repo.find_entities_by_keys(label, key, values))
        keyed.sort(key=lambda match: (positions[match[0]], match[1].id))

        names = await self.repo.entity_names_norm([hit.id for hit in by_text])
        named = _name_anchors(question, by_text, names)
        unique = {hit.id: hit for _, hit in keyed}
        for hit in named:
            unique.setdefault(hit.id, replace(hit, score=1.0))
        return list(unique.values())


def _name_anchors(
    question: str, candidates: Sequence[EntityHit], names: Mapping[str, str]
) -> list[EntityHit]:
    padded_question = f" {fold(question)} "
    spans: list[tuple[int, int, EntityHit]] = []
    for hit in candidates:
        name = names.get(hit.id)
        if not name:
            continue
        phrase = f" {name} "
        start = padded_question.find(phrase)
        while start != -1:
            spans.append((start, start + len(phrase), hit))
            start = padded_question.find(phrase, start + 1)

    spans.sort(key=lambda span: (span[0], -span[1]))
    names_in_order = []
    outer_start, outer_end = -1, -1
    for start, end, hit in spans:
        if end <= outer_end and (start, end) != (outer_start, outer_end):
            continue
        names_in_order.append(hit)
        outer_start, outer_end = start, end
    return names_in_order
