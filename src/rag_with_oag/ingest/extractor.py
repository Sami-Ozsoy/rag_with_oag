"""Ontology-constrained extraction of entities and relations from chunks.

Chunks go to the model in small batches, labelled K1, K2… The JSON is parsed
structurally (``llm.structured``), then checked against the ontology: unknown
types, relations between the wrong entity types, missing required properties
and bad enum values. Invalid output is sent back once with the error list;
whatever is still invalid after that is dropped and reported.

The result is split per chunk, so each chunk's extraction can be cached under
(chunk text, ontology, model, prompt) and reused when the same text is seen again.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from rag_with_oag.generation import prompts
from rag_with_oag.llm import structured
from rag_with_oag.llm.client import ChatModel, Message
from rag_with_oag.ontology.model import Ontology
from rag_with_oag.ontology.schema_gen import RawExtraction, extraction_schema
from rag_with_oag.ontology.validator import coerce_properties
from rag_with_oag.types import Chunk


@dataclass(slots=True)
class EntityMention:
    ref: str
    type: str
    name: str
    description: str | None
    properties: dict[str, Any]
    mentioned: bool = True  # False: carried along only as the endpoint of a relation


@dataclass(slots=True)
class RelationMention:
    type: str
    source: str  # entity ref within the same ChunkGraph
    target: str
    properties: dict[str, Any]
    confidence: float | None = None


@dataclass(slots=True)
class ChunkGraph:
    """What one chunk says, self-contained so it can be cached on its own."""

    entities: list[EntityMention] = field(default_factory=list)
    relations: list[RelationMention] = field(default_factory=list)
    unmapped: list[tuple[str, int]] = field(default_factory=list)

    def entity(self, ref: str) -> EntityMention | None:
        return next((e for e in self.entities if e.ref == ref), None)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> ChunkGraph:
        raw = json.loads(data)
        return cls(
            entities=[EntityMention(**e) for e in raw.get("entities", [])],
            relations=[RelationMention(**r) for r in raw.get("relations", [])],
            unmapped=[(t, int(c)) for t, c in raw.get("unmapped", [])],
        )


@dataclass(slots=True)
class BatchResult:
    graphs: dict[str, ChunkGraph]  # chunk id -> graph
    errors: list[str]


class Extractor:
    def __init__(
        self,
        llm: ChatModel,
        ontology: Ontology,
        *,
        mode: structured.StructuredMode = "auto",
        validation_retries: int = 1,
    ) -> None:
        self.llm = llm
        self.ontology = ontology
        self.mode = mode
        self.validation_retries = validation_retries
        self.schema = extraction_schema(ontology)
        self._system = prompts.EXTRACTION_SYSTEM.format(ontology=ontology.describe())
        # Part of the extraction cache key: a reworded prompt must not reuse old answers.
        self.fingerprint = hashlib.sha256(
            json.dumps(
                [self._system, prompts.EXTRACTION_USER, self.schema],
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()[:16]

    async def extract(self, chunks: Sequence[Chunk]) -> BatchResult:
        labels = {f"K{i + 1}": chunk.id for i, chunk in enumerate(chunks)}
        passages = "\n\n".join(
            f"[{label}] {c.text}" for label, c in zip(labels, chunks, strict=True)
        )
        messages: list[Message] = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": prompts.EXTRACTION_USER.format(passages=passages)},
        ]
        result = await structured.generate(
            self.llm,
            messages,
            schema=self.schema,
            model=RawExtraction,
            name="extraction",
            mode=self.mode,
        )
        graphs, errors = validate(result.value, labels, self.ontology)
        for _ in range(self.validation_retries):
            if not errors:
                break
            retry = await structured.generate(
                self.llm,
                [*messages, *structured.feedback_messages(result.raw, errors)],
                schema=self.schema,
                model=RawExtraction,
                name="extraction",
                mode=self.mode,
            )
            result = retry
            graphs, errors = validate(result.value, labels, self.ontology)
        return BatchResult(graphs=graphs, errors=errors)


def validate(
    raw: RawExtraction, labels: dict[str, str], ontology: Ontology
) -> tuple[dict[str, ChunkGraph], list[str]]:
    """Keep what obeys the ontology, split it per chunk, and list what does not."""
    errors: list[str] = []
    entities: dict[str, tuple[EntityMention, list[str]]] = {}

    for item in raw.entities:
        where = f"entity {item.ref!r} ({item.type} {item.name!r})"
        etype = ontology.entity(item.type)
        if etype is None:
            errors.append(
                f"{where}: unknown entity type {item.type!r}; use one of {ontology.entity_names}"
            )
            continue
        if not item.name.strip():
            errors.append(f"{where}: name is empty")
            continue
        if item.ref in entities:
            errors.append(f"{where}: ref {item.ref!r} is used twice")
            continue
        props, prop_errors = coerce_properties(etype.properties, item.properties, where)
        chunk_ids, chunk_errors = _chunk_ids(item.chunks, labels, where)
        if prop_errors or chunk_errors:
            errors.extend(prop_errors + chunk_errors)
            continue
        mention = EntityMention(
            item.ref, etype.name, item.name.strip(), (item.description or "").strip() or None, props
        )
        entities[item.ref] = (mention, chunk_ids)

    graphs: dict[str, ChunkGraph] = {cid: ChunkGraph() for cid in labels.values()}
    for mention, chunk_ids in entities.values():
        for cid in chunk_ids:
            graphs[cid].entities.append(mention)

    for rel in raw.relations:
        where = f"relation {rel.type}({rel.source} -> {rel.target})"
        rtype = ontology.relation(rel.type)
        if rtype is None:
            errors.append(
                f"{where}: unknown relation type {rel.type!r}; use one of {ontology.relation_names}"
            )
            continue
        src_entry, tgt_entry = entities.get(rel.source), entities.get(rel.target)
        if src_entry is None or tgt_entry is None:
            errors.append(f"{where}: source and target must be refs of valid entities")
            continue
        (src, src_chunks), (tgt, tgt_chunks) = src_entry, tgt_entry
        if (src.type, tgt.type) != (rtype.source, rtype.target):
            errors.append(
                f"{where}: {rel.type} connects {rtype.source} -> {rtype.target}, "
                f"not {src.type} -> {tgt.type}"
            )
            continue
        props, prop_errors = coerce_properties(rtype.properties, rel.properties, where)
        if prop_errors:
            errors.extend(prop_errors)
            continue
        chunk_ids, chunk_errors = _chunk_ids(rel.chunks, labels, where, allow_empty=True)
        if chunk_errors:
            errors.extend(chunk_errors)
            continue
        # No passage given: the passages that mention both ends are the evidence.
        chunk_ids = chunk_ids or [c for c in src_chunks if c in tgt_chunks] or src_chunks
        confidence = rel.confidence if rel.confidence is None or 0 <= rel.confidence <= 1 else None
        for cid in chunk_ids:
            graph = graphs[cid]
            for end in (src, tgt):
                if graph.entity(end.ref) is None:
                    graph.entities.append(
                        EntityMention(
                            end.ref,
                            end.type,
                            end.name,
                            end.description,
                            end.properties,
                            mentioned=False,
                        )
                    )
            graph.relations.append(RelationMention(rtype.name, src.ref, tgt.ref, props, confidence))

    if raw.unmapped and graphs:
        first = next(iter(graphs.values()))
        first.unmapped = [(u.term.strip(), max(1, u.count)) for u in raw.unmapped if u.term.strip()]
    return graphs, errors


def _chunk_ids(
    given: Sequence[str], labels: dict[str, str], where: str, *, allow_empty: bool = False
) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    errors: list[str] = []
    for label in given:
        key = label.strip().strip("[]").upper()
        if key in labels:
            if labels[key] not in ids:
                ids.append(labels[key])
        else:
            errors.append(f"{where}: unknown passage id {label!r}; use {list(labels)}")
    if not ids and not allow_empty and not errors:
        if len(labels) == 1:  # a single passage is unambiguous
            return list(labels.values()), []
        errors.append(f"{where}: 'chunks' must list the passages that mention it")
    return ids, errors
