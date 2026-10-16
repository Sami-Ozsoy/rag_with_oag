"""OAG: link the question to entities, walk the ontology's relations, state facts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_with_oag.config import RetrievalConfig
from rag_with_oag.embed.base import Embedder
from rag_with_oag.ontology.model import Ontology
from rag_with_oag.retrieval.context import assemble, pack_facts
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
        self.linker = EntityLinker(repo, config)

    async def subgraph(self, seeds: Sequence[EntityHit], *, hops: int | None = None) -> Subgraph:
        cfg = self.config
        seed_ids = [s.id for s in seeds]
        edges = await self.repo.expand(
            seed_ids,
            rel_types=self.ontology.relation_names,
            hops=hops or cfg.hops,
            max_paths=cfg.max_paths,
            max_facts=cfg.max_facts,
        )
        entity_ids = list(
            dict.fromkeys([*seed_ids, *(i for e in edges for i in (e.source, e.target))])
        )
        details = await self.repo.entity_details(entity_ids)
        seed_set = set(seed_ids)

        def name(eid: str) -> str:
            return details[eid].name if eid in details else eid

        facts: list[Fact] = []
        # Seeds describe themselves first: their properties often are the answer.
        for eid in seed_ids:
            if eid in details:
                facts.append(_entity_fact(details[eid]))
        relation_facts = [
            Fact(
                subject=name(e.source),
                predicate=e.type,
                object=name(e.target),
                properties=e.props,
                evidence=e.evidence,
                kind="relation",
                subject_id=e.source,
                object_id=e.target,
            )
            for e in edges
        ]
        # Relations touching a seed before those further out.
        relation_facts.sort(key=lambda f: not ({f.subject_id, f.object_id} & seed_set))
        facts.extend(relation_facts)
        facts.extend(
            _entity_fact(details[eid])
            for eid in entity_ids
            if eid not in seed_set and eid in details and details[eid].props
        )
        return Subgraph(seeds=list(seeds), facts=facts, entity_ids=entity_ids)

    async def retrieve(
        self, question: str, *, hops: int | None = None, vector: Sequence[float] | None = None
    ) -> RetrievedContext:
        if vector is None:
            [vector] = await self.embedder.embed([question])
        seeds = await self.linker.link(question, vector)
        if not seeds:
            return assemble(question, "oag")
        sub = await self.subgraph(seeds, hops=hops)
        return assemble(
            question, "oag", facts=pack_facts(sub.facts, budget=self.config.context_token_budget)
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
