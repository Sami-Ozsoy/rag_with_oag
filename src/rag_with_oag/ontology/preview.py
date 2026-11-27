"""Sample documents and report what an ontology extracts before ingest."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import batched

from rag_with_oag.config import ExtractionConfig, LLMConfig, RetrievalConfig
from rag_with_oag.errors import MODEL_ENDPOINT_ERRORS
from rag_with_oag.ingest.chunker import AdaptiveChunker, Chunker
from rag_with_oag.ingest.extractor import BatchResult, EntityMention, Extractor, RelationMention
from rag_with_oag.ingest.loaders import Source
from rag_with_oag.ingest.pipeline import build_chunks
from rag_with_oag.llm.client import ChatModel, LLMClient
from rag_with_oag.ontology.model import Ontology, Property
from rag_with_oag.ontology.suggest import load_documents
from rag_with_oag.types import Chunk

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EntityTypeStats:
    name: str
    mentions: int
    distinct: int
    key: str | None
    key_fill_rate: float | None
    property_fill_rates: dict[str, float]
    examples: list[str]


@dataclass(slots=True)
class RelationTypeStats:
    name: str
    source: str
    target: str
    count: int
    property_fill_rates: dict[str, float]
    examples: list[str]


@dataclass(slots=True)
class DroppedItems:
    kind: str
    type_name: str | None
    count: int
    observed: dict[str, int]
    examples: list[str]


@dataclass(slots=True)
class PreviewReport:
    ontology: str
    documents_sampled: int
    documents_available: int
    chunks_sampled: int
    batches: int
    failed_batches: list[str]
    chunks_with_entities: int
    chunks_with_relations: int
    entity_types: list[EntityTypeStats]
    relation_types: list[RelationTypeStats]
    dropped: list[DroppedItems]
    unmapped_concepts: list[tuple[str, int]]
    hints: list[str]
    token_usage: dict[str, int]

    @property
    def unused_entity_types(self) -> list[str]:
        return [stats.name for stats in self.entity_types if not stats.mentions]

    @property
    def unused_relation_types(self) -> list[str]:
        return [stats.name for stats in self.relation_types if not stats.count]


async def preview_ontology(
    ontology: Ontology,
    sources: Iterable[Source] | Source,
    llm: LLMConfig | ChatModel,
    *,
    documents: int = 5,
    chunks_per_doc: int = 6,
    concurrency: int = 4,
    config: ExtractionConfig | None = None,
    chunker: Chunker | None = None,
    retrieval: RetrievalConfig | None = None,
) -> PreviewReport:
    """Run the ingest chunker and extractor without opening a knowledge base."""
    for name, value in (
        ("documents", documents),
        ("chunks_per_doc", chunks_per_doc),
        ("concurrency", concurrency),
    ):
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
    loaded = sorted(
        (doc for doc in load_documents(sources) if doc.text.strip()), key=lambda doc: doc.source
    )
    if not loaded:
        raise ValueError("preview needs documents with text")

    selected = _spread(loaded, documents)
    config = config or ExtractionConfig()
    chunker = chunker or AdaptiveChunker(retrieval or RetrievalConfig())
    batches = [
        (doc.source, batch)
        for doc in selected
        for batch in batched(_spread(build_chunks(doc, chunker), chunks_per_doc), config.batch_size)
    ]
    if not batches:
        raise ValueError("preview needs at least one chunk")
    report = PreviewReport(
        ontology=ontology.name,
        documents_sampled=len(selected),
        documents_available=len(loaded),
        chunks_sampled=sum(len(batch) for _, batch in batches),
        batches=len(batches),
        failed_batches=[],
        chunks_with_entities=0,
        chunks_with_relations=0,
        entity_types=[],
        relation_types=[],
        dropped=[],
        unmapped_concepts=[],
        hints=[],
        token_usage={},
    )
    logger.info(
        "Sampled %d of %d documents in %d batches", len(selected), len(loaded), len(batches)
    )
    client = None
    if isinstance(llm, LLMConfig):
        llm = client = LLMClient(llm)
    try:
        await _extract(ontology, llm, config, batches, concurrency, report)
    finally:
        if client is not None:
            await client.aclose()
    report.hints = _hints(report, ontology)
    return report


def _spread[T](items: Sequence[T], limit: int) -> list[T]:
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    return [items[i * (len(items) - 1) // (limit - 1)] for i in range(limit)]


async def _extract(
    ontology: Ontology,
    llm: ChatModel,
    config: ExtractionConfig,
    batches: list[tuple[str, tuple[Chunk, ...]]],
    concurrency: int,
    report: PreviewReport,
) -> None:
    extractor = Extractor(
        llm, ontology, mode=config.structured_mode, validation_retries=config.validation_retries
    )
    usage_before = Counter(llm.usage)
    semaphore = asyncio.Semaphore(concurrency)

    async def extract(source: str, batch: tuple[Chunk, ...]) -> BatchResult | str:
        async with semaphore:
            try:
                return await extractor.extract(batch)
            except MODEL_ENDPOINT_ERRORS:
                raise
            except Exception as exc:
                return (
                    f"{source}: chunks {batch[0].idx}-{batch[-1].idx}: {type(exc).__name__}: {exc}"
                )

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(extract(source, batch)) for source, batch in batches]
    except ExceptionGroup as group_error:
        first = group_error.exceptions[0]
        raise first from first.__cause__

    # Read results in sample order, regardless of which request finished first.
    _summarize(report, ontology, (task.result() for task in tasks))
    report.token_usage = dict(Counter(llm.usage) - usage_before)


def _summarize(
    report: PreviewReport, ontology: Ontology, results: Iterable[BatchResult | str]
) -> None:
    entities: defaultdict[str, list[EntityMention]] = defaultdict(list)
    relations: defaultdict[str, list[tuple[RelationMention, str]]] = defaultdict(list)
    dropped: dict[tuple[str, str | None], DroppedItems] = {}
    unmapped: Counter[str] = Counter()
    for result in results:
        if isinstance(result, str):
            report.failed_batches.append(result)
            continue
        mentions: dict[str, EntityMention] = {}
        for graph in result.graphs.values():
            present = [entity for entity in graph.entities if entity.mentioned]
            report.chunks_with_entities += bool(present)
            report.chunks_with_relations += bool(graph.relations)
            for entity in present:
                mentions.setdefault(entity.ref, entity)
            for term, count in graph.unmapped:
                unmapped[term] += count
        for entity in mentions.values():
            entities[entity.type].append(entity)
        for relation in result.relations:
            source, target = mentions[relation.source].name, mentions[relation.target].name
            relations[relation.type].append((relation, f"{source} -[{relation.type}]-> {target}"))
        for issue in result.issues:
            key = (issue.kind, issue.type_name)
            if key not in dropped:
                dropped[key] = DroppedItems(issue.kind, issue.type_name, 0, {}, [])
            group = dropped[key]
            group.count += 1
            if issue.observed is not None:
                observed = " -> ".join(issue.observed)
                group.observed[observed] = group.observed.get(observed, 0) + 1
            if len(group.examples) < 3 and issue.message not in group.examples:
                group.examples.append(issue.message)

    for etype in ontology.entities:
        items = entities[etype.name]
        rates = _fill_rates(etype.properties, [entity.properties for entity in items])
        identities = set()
        for entity in items:
            value = entity.properties.get(etype.key) if etype.key is not None else None
            identity = (
                ("key", value)
                if value not in (None, "")
                else ("name", ontology.match_name(entity.name))
            )
            identities.add(identity)
        report.entity_types.append(
            EntityTypeStats(
                name=etype.name,
                mentions=len(items),
                distinct=len(identities),
                key=etype.key,
                key_fill_rate=rates.get(etype.key, 0.0) if etype.key is not None else None,
                property_fill_rates=rates,
                examples=list(dict.fromkeys(entity.name for entity in items))[:3],
            )
        )
    for rtype in ontology.relations:
        edges = relations[rtype.name]
        report.relation_types.append(
            RelationTypeStats(
                name=rtype.name,
                source=rtype.source,
                target=rtype.target,
                count=len(edges),
                property_fill_rates=_fill_rates(
                    rtype.properties, [edge.properties for edge, _ in edges]
                ),
                examples=list(dict.fromkeys(example for _, example in edges))[:3],
            )
        )
    report.dropped = sorted(dropped.values(), key=lambda group: -group.count)
    report.unmapped_concepts = unmapped.most_common()


def _fill_rates(
    properties: Sequence[Property], values: Sequence[Mapping[str, object]]
) -> dict[str, float]:
    if not values:
        return {}
    return {
        prop.name: sum(row.get(prop.name) not in (None, "") for row in values) / len(values)
        for prop in properties
    }


def _hints(report: PreviewReport, ontology: Ontology) -> list[str]:
    hints = [
        f"Entity type {name} was never extracted from the sample."
        for name in report.unused_entity_types
    ]
    hints.extend(
        f"Relation type {name} was never extracted from the sample."
        for name in report.unused_relation_types
    )
    for group in report.dropped:
        if group.kind == "endpoint_mismatch" and group.type_name is not None:
            relation = ontology.relation(group.type_name)
            assert relation is not None  # endpoint checks only run for declared types
            observed = ", ".join(f"{pair} ({count})" for pair, count in group.observed.items())
            hints.append(
                f"{group.type_name} was dropped {group.count} times because its endpoints "
                "did not match: "
                f"got {observed}; the ontology allows {relation.source} -> {relation.target}."
            )
    for group in report.dropped:
        if group.kind != "endpoint_mismatch":
            type_name = f" for {group.type_name}" if group.type_name is not None else ""
            kind = group.kind.replace("_", " ")
            issues = "issue" if group.count == 1 else "issues"
            hints.append(f"{group.count} {kind} {issues}{type_name}.")
    for entity in report.entity_types:
        rate = entity.key_fill_rate
        if entity.mentions and rate is not None and rate < 0.5:
            hints.append(
                f"Key {entity.key} of {entity.name} is filled in only {rate:.0%} of mentions; "
                "instances without it are matched by name and may split into separate nodes."
            )
    stats_by_type: tuple[EntityTypeStats | RelationTypeStats, ...] = (
        *report.entity_types,
        *report.relation_types,
    )
    for stats in stats_by_type:
        hints.extend(
            f"Property {name} of {stats.name} was never filled."
            for name, rate in stats.property_fill_rates.items()
            if rate == 0
        )
    if report.unmapped_concepts:
        top = ", ".join(f"{term} ({count})" for term, count in report.unmapped_concepts[:10])
        hints.append(f"Concepts the ontology cannot express: {top}")
    if report.failed_batches:
        hints.append(
            f"{len(report.failed_batches)} of {report.batches} batches failed; "
            "the report covers only the rest."
        )
    return hints
