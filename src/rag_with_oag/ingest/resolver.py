"""Entity resolution: decide which mentions are the same real-world entity.

Ids are deterministic: (type, folded key value) when the type declares a key
property and the mention has a value for it, else (type, folded name). On top
of that, in order:

1. Within one write, a key-less mention joins a keyed one of the same type and
   name ("Bulut Yazılım Ltd." without a tax number in one passage, with it in
   another).
2. Against the graph, a keyed mention joins the entity that already has that
   key value, or else a same-named entity still missing its key; a key-less
   mention joins the same-named entity of its type.
3. A key-less mention that is still new is compared by embedding with entities
   of its type, in the graph and in the same write, and joins one above the
   similarity threshold.

Keys win over names: two entities with different key values never merge.
"""

from __future__ import annotations

import math
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_with_oag._text import fold
from rag_with_oag.embed.base import Embedder, embed_documents
from rag_with_oag.ingest.extractor import ChunkGraph, EntityMention
from rag_with_oag.ontology.model import Ontology
from rag_with_oag.store.repository import EntityRow, GraphRepository, GraphWrite, RelationRow
from rag_with_oag.types import ID_NAMESPACE


def entity_id(entity_type: str, identity: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"entity:{entity_type}:{identity}"))


def entity_text(
    name: str, entity_type: str, description: str | None, key_props: Mapping[str, Any]
) -> str:
    """What an entity is embedded as: name, type, description and identifying values."""
    lines = [f"{name} ({entity_type})"]
    if description:
        lines.append(description)
    lines.extend(f"{k}: {v}" for k, v in key_props.items())
    return "\n".join(lines)


@dataclass(slots=True)
class _Merged:
    id: str
    type: str
    key: str | None  # the type's key property, if it has a value here
    names: Counter[str] = field(default_factory=Counter)
    description: str | None = None
    props: dict[str, Any] = field(default_factory=dict)

    def absorb(self, other: _Merged | EntityMention) -> None:
        if isinstance(other, EntityMention):
            self.names[other.name] += 1
            description, props = other.description, other.properties
        else:
            self.names.update(other.names)
            description, props = other.description, other.props
        self.description = self.description or description
        for k, v in props.items():
            self.props.setdefault(k, v)

    @property
    def name(self) -> str:
        return self.names.most_common(1)[0][0]

    @property
    def folded_names(self) -> set[str]:
        return {fold(n) for n in self.names}


class EntityResolver:
    def __init__(
        self, repo: GraphRepository, embedder: Embedder, ontology: Ontology, *, similarity: float
    ) -> None:
        self.repo = repo
        self.embedder = embedder
        self.ontology = ontology
        self.similarity = similarity

    def identity(self, mention: EntityMention) -> tuple[str, str | None]:
        """(id, key property name if the mention is keyed) before any merging."""
        etype = self.ontology.entity(mention.type)
        key = etype.key if etype else None
        if key and mention.properties.get(key) not in (None, ""):
            return entity_id(mention.type, "key:" + fold(str(mention.properties[key]))), key
        return entity_id(mention.type, "name:" + fold(mention.name)), None

    async def resolve(self, graphs: Mapping[str, ChunkGraph]) -> GraphWrite:
        merged: dict[str, _Merged] = {}
        ref_ids: dict[tuple[str, str], str] = {}  # (chunk id, ref) -> provisional id
        for chunk_id, graph in graphs.items():
            for mention in graph.entities:
                eid, key = self.identity(mention)
                ref_ids[(chunk_id, mention.ref)] = eid
                merged.setdefault(eid, _Merged(id=eid, type=mention.type, key=key)).absorb(mention)
        if not merged:
            return GraphWrite()

        remap = self._merge_within_write(merged)
        await self._match_graph(merged, remap)
        survivors = [e for e in merged.values() if e.id not in remap]
        vectors = await embed_documents(self.embedder, [self._text(e) for e in survivors])
        embedding = {e.id: list(v) for e, v in zip(survivors, vectors, strict=True)}
        remap.update(await self._similar_merges(survivors, embedding, remap))

        def final(eid: str) -> str:
            seen = {eid}
            while eid in remap:
                eid = remap[eid]
                if eid in seen:  # a bug, but fail the document instead of hanging the loop
                    raise RuntimeError(f"entity resolution loops through {eid}")
                seen.add(eid)
            return eid

        rows: dict[str, _Merged] = {}
        row_vec: dict[str, list[float]] = {}
        for entry in merged.values():
            target = final(entry.id)
            into = rows.setdefault(target, _Merged(id=target, type=entry.type, key=entry.key))
            into.absorb(entry)
            if entry.id in embedding:
                row_vec.setdefault(target, embedding[entry.id])
        missing = [e for e in rows.values() if e.id not in row_vec]
        if missing:  # merged into graph entities: their node exists, vector only for MERGE
            for e, v in zip(
                missing,
                await embed_documents(self.embedder, [self._text(e) for e in missing]),
                strict=True,
            ):
                row_vec[e.id] = list(v)

        write = GraphWrite(
            entities=[
                EntityRow(
                    id=e.id,
                    type=e.type,
                    name=e.name,
                    name_norm=fold(e.name),
                    description=e.description,
                    props=e.props,
                    embedding=row_vec[e.id],
                )
                for e in rows.values()
            ]
        )
        relations: dict[tuple[str, str, str], RelationRow] = {}
        for chunk_id, graph in graphs.items():
            for mention in graph.entities:
                if mention.mentioned:
                    write.mentions.append((chunk_id, final(ref_ids[(chunk_id, mention.ref)])))
            for rel in graph.relations:
                src = final(ref_ids[(chunk_id, rel.source)])
                tgt = final(ref_ids[(chunk_id, rel.target)])
                row = relations.setdefault(
                    (rel.type, src, tgt),
                    RelationRow(
                        type=rel.type,
                        source=src,
                        target=tgt,
                        props={},
                        confidence=None,
                        evidence=[],
                    ),
                )
                for k, v in rel.properties.items():
                    row.props.setdefault(k, v)
                if rel.confidence is not None:
                    row.confidence = max(row.confidence or 0.0, rel.confidence)
                if chunk_id not in row.evidence:
                    row.evidence.append(chunk_id)
        write.relations = list(relations.values())
        return write

    def _text(self, entry: _Merged) -> str:
        etype = self.ontology.entity(entry.type)
        key_props = (
            {etype.key: entry.props[etype.key]} if etype and etype.key in entry.props else {}
        )
        return entity_text(entry.name, entry.type, entry.description, key_props)

    @staticmethod
    def _merge_within_write(merged: Mapping[str, _Merged]) -> dict[str, str]:
        keyed_by_name: dict[tuple[str, str], str] = {}
        for e in merged.values():
            if e.key:
                for n in e.folded_names:
                    keyed_by_name.setdefault((e.type, n), e.id)
        remap = {}
        for e in merged.values():
            if e.key is None:
                target = next(
                    (
                        keyed_by_name[(e.type, n)]
                        for n in e.folded_names
                        if (e.type, n) in keyed_by_name
                    ),
                    None,
                )
                if target is not None:
                    remap[e.id] = target
        return remap

    async def _match_graph(self, merged: Mapping[str, _Merged], remap: dict[str, str]) -> None:
        """Point entities at the graph nodes they already are, updating ``remap`` in place."""
        claimed: set[str] = set()  # key-less graph nodes a keyed entity of this write took
        for e in merged.values():
            if e.id in remap:
                continue
            target = None
            if e.key:
                target = await self.repo.find_entity_by_key(e.type, e.key, e.props[e.key])
                if target is None:
                    named = await self.repo.find_entity_by_name(e.type, fold(e.name), keyless=e.key)
                    # Keys win over names: a second key value must not take the same node.
                    if named is not None and named not in claimed:
                        claimed.add(named)
                        target = named
            else:
                target = await self.repo.find_entity_by_name(e.type, fold(e.name))
            if target is None or target == e.id:
                continue
            # The graph node is canonical. If this write folded that node's own key-less
            # mention into ``e``, undo it, or the two would point at each other.
            if remap.get(target) == e.id:
                del remap[target]
            remap[e.id] = target

    async def _similar_merges(
        self,
        survivors: Sequence[_Merged],
        embedding: Mapping[str, Sequence[float]],
        remap: Mapping[str, str],
    ) -> dict[str, str]:
        """Map new key-less entities onto existing or earlier near-duplicates."""
        existing = await self.repo.existing_entities(e.id for e in survivors)
        found: dict[str, str] = {}
        accepted: dict[str, list[_Merged]] = defaultdict(list)  # type -> kept in this write
        for entry in survivors:
            if entry.key or entry.id in existing:
                accepted[entry.type].append(entry)
                continue
            target = None
            for hit in await self.repo.search_entities_vector(embedding[entry.id], 5):
                if hit.type == entry.type and hit.score >= self.similarity and hit.id != entry.id:
                    target = hit.id
                    break
            if target is None:
                for other in accepted[entry.type]:
                    if _cosine(embedding[entry.id], embedding[other.id]) >= self.similarity:
                        target = other.id
                        break
            if target is None:
                accepted[entry.type].append(entry)
            else:
                found[entry.id] = target
        return found


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
