"""OAG walks ontology relations by hop from exact anchors or ranked entity matches.

The walk caps each node's fan-out per relation type and direction. Seed facts
come first, followed by relations in walk order and each new entity's
properties beside the relation that first reaches it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_with_oag.config import RetrievalConfig
from rag_with_oag.embed.base import Embedder, embed_queries
from rag_with_oag.ontology.model import Ontology
from rag_with_oag.retrieval.context import assemble, pack_context
from rag_with_oag.retrieval.linking import EntityLinker
from rag_with_oag.store.repository import EntityDetails, GraphRepository
from rag_with_oag.types import EntityHit, Fact, RetrievedContext


@dataclass(frozen=True, slots=True)
class Subgraph:
    seeds: list[EntityHit]
    facts: list[Fact]
    entity_ids: list[str]  # seeds first, then entities reached by expansion


class OAGRetriever:
    def __init__(
        self,
        repo: GraphRepository,
        embedder: Embedder,
        ontology: Ontology,
        config: RetrievalConfig,
    ) -> None:
        self.repo = repo
        self.embedder = embedder
        self.ontology = ontology
        self.config = config
        keyed_types = [(entity.name, entity.key) for entity in ontology.entities if entity.key]
        self.linker = EntityLinker(repo, config, keyed_types=keyed_types)

    async def subgraph(self, seeds: Sequence[EntityHit], *, hops: int | None = None) -> Subgraph:
        cfg = self.config
        seed_ids = [s.id for s in seeds]
        edges = await self.repo.expand(
            seed_ids,
            rel_types=self.ontology.relation_names,
            hops=hops or cfg.hops,
            max_fanout=cfg.max_fanout,
            max_facts=cfg.max_facts,
        )
        entity_ids = list(
            dict.fromkeys([*seed_ids, *(i for e in edges for i in (e.source, e.target))])
        )
        details = await self.repo.entity_details(entity_ids)
        seen = set(seed_ids)

        def name(eid: str) -> str:
            return details[eid].name if eid in details else eid

        facts: list[Fact] = []
        # Seeds describe themselves first: their properties often are the answer.
        for eid in seed_ids:
            if eid in details:
                facts.append(_entity_fact(details[eid]))
        for edge in edges:
            facts.append(
                Fact(
                    subject=name(edge.source),
                    predicate=edge.type,
                    object=name(edge.target),
                    properties=edge.props,
                    evidence=edge.evidence,
                    kind="relation",
                    subject_id=edge.source,
                    object_id=edge.target,
                )
            )
            for eid in (edge.source, edge.target):
                if eid in seen:
                    continue
                seen.add(eid)
                if eid in details and details[eid].props:
                    facts.append(_entity_fact(details[eid]))
        return Subgraph(seeds=list(seeds), facts=facts, entity_ids=entity_ids)

    async def retrieve(
        self, question: str, *, hops: int | None = None, vector: Sequence[float] | None = None
    ) -> RetrievedContext:
        if vector is None:
            [vector] = await embed_queries(self.embedder, [question])
        seeds = await self.linker.link(question, vector)
        if not seeds:
            return assemble(question, "oag")
        sub = await self.subgraph(seeds, hops=hops)
        return pack_context(
            question, "oag", budget=self.config.context_token_budget, facts=sub.facts
        )


def _entity_fact(d: EntityDetails) -> Fact:
    props = dict(d.props)
    if d.description:
        props = {"description": d.description, **props}
    return Fact(
        subject=d.name,
        predicate="is_a",
        object=d.type,
        properties=props,
        kind="entity",
        subject_id=d.id,
    )
