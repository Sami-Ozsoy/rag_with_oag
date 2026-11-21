"""GraphRepository: the only code that talks to the graph.

Retrieval and ingest call these methods and never build Cypher themselves, so
swapping the backend means rewriting this module and ``queries.py`` only.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rag_with_oag._text import fold
from rag_with_oag.store import queries as q
from rag_with_oag.store import schema
from rag_with_oag.store.connection import GraphConnection
from rag_with_oag.types import Chunk, ChunkHit, Document, EntityHit

if TYPE_CHECKING:
    from rag_with_oag.ontology.model import Ontology

WRITE_BATCH = 128

# Properties every entity carries that are bookkeeping, not facts about it.
_INTERNAL_PROPS = frozenset({"id", "name", "name_norm", "embedding", "ontology_version"})


@dataclass(frozen=True, slots=True)
class DocumentState:
    complete: bool
    extracted_version: str | None


@dataclass(slots=True)
class EntityRow:
    id: str
    type: str
    name: str
    name_norm: str
    description: str | None
    props: dict[str, Any]
    embedding: list[float]


@dataclass(slots=True)
class RelationRow:
    type: str
    source: str
    target: str
    props: dict[str, Any]
    confidence: float | None
    evidence: list[str]


@dataclass(slots=True)
class GraphWrite:
    entities: list[EntityRow] = field(default_factory=list)
    mentions: list[tuple[str, str]] = field(default_factory=list)  # (chunk id, entity id)
    relations: list[RelationRow] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EntityDetails:
    id: str
    name: str
    type: str
    description: str | None
    props: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RelationEdge:
    source: str
    type: str
    target: str
    props: dict[str, Any]
    evidence: tuple[str, ...]
    hop: int


class GraphRepository:
    def __init__(self, connection: GraphConnection, kb: str) -> None:
        self.kb = q.safe_identifier(kb)
        self._connection = connection
        self._meta: schema.Meta | None = None
        self._meta_lock = asyncio.Lock()

    @property
    def _graph(self) -> Any:
        return self._connection.graph(self.kb)

    @property
    def meta(self) -> schema.Meta:
        if self._meta is None:
            raise RuntimeError("repository is not initialized; call init() first")
        return self._meta

    async def init(self, *, embed_model: str, ontology: Ontology | None = None) -> schema.Meta:
        meta = await schema.ensure_base_schema(self._graph, self.kb)
        schema.check_embed_model(meta, embed_model)
        if ontology is not None:
            schema.check_ontology(meta, ontology.fingerprint)
            if meta.ontology_version is None:
                meta = await schema.record_ontology(
                    self._graph,
                    meta,
                    version=ontology.fingerprint,
                    name=ontology.name,
                    yaml=ontology.to_yaml(),
                )
        if meta.embed_dim is not None:  # e.g. an ontology added to a v0.1 knowledge base
            await schema.create_vector_indexes(self._graph, meta.embed_dim)
        self._meta = meta
        return meta

    async def backfill_chunk_text_norm(self) -> None:
        graph = self._graph
        while True:
            result = await graph.ro_query(q.CHUNKS_MISSING_TEXT_NORM, {"limit": WRITE_BATCH})
            rows = [{"id": cid, "text_norm": fold(text)} for cid, text in result.result_set]
            if not rows:
                return
            await graph.query(q.SET_CHUNK_TEXT_NORM, {"rows": rows})

    async def ensure_embedding_dim(self, *, model: str, dim: int) -> None:
        if self.meta.embed_dim == dim:
            return
        async with self._meta_lock:
            self._meta = await schema.ensure_vector_index(
                self._graph, self.meta, model=model, dim=dim
            )

    # --- ingest: documents and chunks -----------------------------------------

    async def document_state(self, doc_id: str) -> DocumentState:
        result = await self._graph.ro_query(q.DOCUMENT_STATE, {"id": doc_id})
        if not result.result_set:
            return DocumentState(complete=False, extracted_version=None)
        status, extracted = result.result_set[0]
        return DocumentState(complete=status == "complete", extracted_version=extracted)

    async def write_document(
        self, doc: Document, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings differ in length")
        graph = self._graph
        await graph.query(
            q.UPSERT_DOCUMENT,
            {
                "id": doc.id,
                "source": doc.source,
                "title": doc.title,
                "content_hash": doc.content_hash,
                "metadata": json.dumps(doc.metadata, ensure_ascii=False, default=str),
            },
        )
        rows = [_chunk_row(c, e) for c, e in zip(chunks, embeddings, strict=True)]
        for batch in _batches(rows, WRITE_BATCH):
            await graph.query(q.UPSERT_CHUNKS, {"rows": batch})
        pairs = [[a.id, b.id] for a, b in itertools.pairwise(chunks)]
        for pair_batch in _batches(pairs, WRITE_BATCH * 4):
            await graph.query(q.LINK_NEXT, {"pairs": pair_batch})
        await graph.query(q.MARK_DOCUMENT_COMPLETE, {"id": doc.id, "chunk_count": len(chunks)})

    async def document_chunks(self, doc_id: str) -> list[Chunk]:
        result = await self._graph.ro_query(q.DOCUMENT_CHUNKS, {"id": doc_id})
        return [
            Chunk(
                id=cid,
                doc_id=did,
                idx=idx,
                text=text,
                token_count=tokens,
                start_char=start,
                end_char=end,
                section_path=tuple(section or ()),
                page_index=page,
                chunk_strategy=strategy or "",
            )
            for (
                cid,
                did,
                idx,
                text,
                tokens,
                start,
                end,
                section,
                page,
                strategy,
            ) in result.result_set
        ]

    async def mark_extracted(self, doc_id: str, version: str) -> None:
        await self._graph.query(q.MARK_DOCUMENT_EXTRACTED, {"id": doc_id, "version": version})

    # --- ingest: extraction cache and graph -------------------------------------

    async def cached_extractions(self, keys: Sequence[str]) -> dict[str, str]:
        if not keys:
            return {}
        result = await self._graph.ro_query(q.GET_EXTRACTIONS, {"keys": list(keys)})
        return {key: data for key, data in result.result_set}

    async def cache_extractions(self, entries: dict[str, str]) -> None:
        rows = [{"key": k, "data": v} for k, v in entries.items()]
        for batch in _batches(rows, WRITE_BATCH):
            await self._graph.query(q.PUT_EXTRACTIONS, {"rows": batch})

    async def existing_entities(self, ids: Iterable[str]) -> set[str]:
        id_list = list(ids)
        if not id_list:
            return set()
        result = await self._graph.ro_query(q.EXISTING_ENTITIES, {"ids": id_list})
        return {row[0] for row in result.result_set}

    async def find_entity_by_key(self, label: str, key: str, value: Any) -> str | None:
        result = await self._graph.ro_query(q.find_by_key(label, key), {"value": value})
        return result.result_set[0][0] if result.result_set else None

    async def find_entity_by_name(
        self, label: str, name_norm: str, *, keyless: str | None = None
    ) -> str | None:
        result = await self._graph.ro_query(
            q.find_by_name(label, keyless=keyless), {"name": name_norm}
        )
        return result.result_set[0][0] if result.result_set else None

    async def write_graph(self, write: GraphWrite, *, version: str) -> None:
        graph = self._graph
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in write.entities:
            by_label[e.type].append(
                {
                    "id": e.id,
                    "name": e.name,
                    "name_norm": e.name_norm,
                    "description": e.description,
                    "props": e.props,
                    "embedding": [float(x) for x in e.embedding],
                }
            )
        for label, rows in by_label.items():
            for batch in _batches(rows, WRITE_BATCH):
                await graph.query(q.upsert_entities(label), {"rows": batch, "version": version})
        mention_rows = [{"chunk": c, "entity": e} for c, e in dict.fromkeys(write.mentions)]
        for batch in _batches(mention_rows, WRITE_BATCH * 4):
            await graph.query(q.LINK_MENTIONS, {"rows": batch})
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in write.relations:
            by_type[r.type].append(
                {
                    "source": r.source,
                    "target": r.target,
                    "props": r.props,
                    "confidence": r.confidence,
                    "evidence": r.evidence,
                }
            )
        for rel_type, rows in by_type.items():
            for batch in _batches(rows, WRITE_BATCH):
                await graph.query(q.upsert_relations(rel_type), {"rows": batch, "version": version})

    # --- retrieval: chunks ----------------------------------------------------

    async def search_chunks(self, embedding: Sequence[float], k: int) -> list[ChunkHit]:
        if self.meta.embed_dim is None:  # nothing ingested yet, no vector index
            return []
        result = await self._graph.ro_query(q.SEARCH_CHUNKS, {"k": k, "q": list(embedding)})
        return [_chunk_hit(row) for row in result.result_set]

    async def search_chunks_text(self, query: str, k: int) -> list[ChunkHit]:
        """Search folded chunk text using ``_text.fulltext_query`` output."""
        if not query:
            return []
        result = await self._graph.ro_query(
            q.SEARCH_CHUNKS_TEXT, {"q": f"@text_norm:({query})", "k": k}
        )
        return [_chunk_hit(row, is_distance=False) for row in result.result_set]

    async def evidence_chunks(
        self, entity_ids: Sequence[str], embedding: Sequence[float], k: int
    ) -> list[ChunkHit]:
        if not entity_ids:
            return []
        result = await self._graph.ro_query(
            q.EVIDENCE_CHUNKS, {"entity_ids": list(entity_ids), "q": list(embedding), "k": k}
        )
        return [_chunk_hit(row) for row in result.result_set]

    async def seed_document_chunks(
        self, entity_ids: Sequence[str], embedding: Sequence[float], k: int
    ) -> list[ChunkHit]:
        if not entity_ids:
            return []
        result = await self._graph.ro_query(
            q.SEED_DOCUMENT_CHUNKS, {"entity_ids": list(entity_ids), "q": list(embedding), "k": k}
        )
        return [_chunk_hit(row) for row in result.result_set]

    # --- retrieval: entities --------------------------------------------------

    async def find_entities_by_keys(
        self, label: str, key: str, values: Sequence[str]
    ) -> list[tuple[str, EntityHit]]:
        """Exact key matches paired with the stored key's text."""
        if not values:
            return []
        result = await self._graph.ro_query(q.find_by_keys(label, key), {"values": list(values)})
        return [
            (value, EntityHit(id=eid, name=name, type=label, score=1.0))
            for eid, name, value in result.result_set
        ]

    async def search_entities_vector(self, embedding: Sequence[float], k: int) -> list[EntityHit]:
        """Nearest entities; ``score`` is cosine similarity."""
        if self.meta.embed_dim is None:
            return []
        result = await self._graph.ro_query(
            q.SEARCH_ENTITIES_VECTOR, {"k": k, "q": list(embedding)}
        )
        return [
            EntityHit(id=eid, name=name, type=_entity_type(labels), score=1.0 - float(dist))
            for eid, name, labels, dist in result.result_set
        ]

    async def search_entities_text(self, query: str, k: int) -> list[EntityHit]:
        """Full-text matches on folded names; ``query`` must come from ``_text.fulltext_query``."""
        if not query:
            return []
        result = await self._graph.ro_query(q.SEARCH_ENTITIES_TEXT, {"q": query, "k": k})
        return [
            EntityHit(id=eid, name=name, type=_entity_type(labels), score=float(score))
            for eid, name, labels, score in result.result_set
        ]

    async def entity_names_norm(self, ids: Sequence[str]) -> dict[str, str]:
        if not ids:
            return {}
        result = await self._graph.ro_query(q.ENTITY_NAMES_NORM, {"ids": list(ids)})
        return {eid: name_norm for eid, name_norm in result.result_set if name_norm}

    async def entity_details(self, ids: Sequence[str]) -> dict[str, EntityDetails]:
        if not ids:
            return {}
        result = await self._graph.ro_query(q.ENTITY_DETAILS, {"ids": list(ids)})
        details = {}
        for eid, name, labels, props in result.result_set:
            props = dict(props)
            description = props.pop("description", None)
            details[eid] = EntityDetails(
                id=eid,
                name=name,
                type=_entity_type(labels),
                description=description,
                props={k: v for k, v in props.items() if k not in _INTERNAL_PROPS},
            )
        return details

    async def expand(
        self,
        seeds: Sequence[str],
        *,
        rel_types: Sequence[str],
        hops: int,
        max_fanout: int,
        max_facts: int,
    ) -> list[RelationEdge]:
        """Walk ontology edges in both directions, querying one frontier per hop.

        At each hop, visit edges by originating seed rank, with seed neighbours
        first and neighbour IDs breaking ties. Cap each frontier node's
        fan-out per relation type and direction. Visit nodes and return edges
        once, up to ``hops`` and ``max_facts``.
        """
        if not seeds or not rel_types or max_facts == 0:
            return []
        if not isinstance(hops, int) or isinstance(hops, bool) or not 1 <= hops <= 5:
            raise ValueError(f"hops must be an integer between 1 and 5, got {hops!r}")

        cypher = q.expand(rel_types)
        graph = self._graph
        frontier = {seed: rank for rank, seed in enumerate(dict.fromkeys(seeds))}
        seed_ids = set(frontier)
        visited_nodes = seed_ids.copy()
        visited_edges: set[int] = set()
        edges: list[RelationEdge] = []
        for hop in range(1, hops + 1):
            if not frontier:
                break
            result = await graph.ro_query(cypher, {"frontier": list(frontier)})
            # Rows start with the frontier node, its neighbour, and the relationship ID.
            rows = sorted(
                result.result_set,
                key=lambda row: (frontier[row[0]], row[1] not in seed_ids, row[1], row[2]),
            )
            next_frontier: dict[str, int] = {}
            fanout: dict[tuple[str, str, bool], int] = defaultdict(int)
            for node, neighbour, edge_id, rel_type, raw_props, outgoing in rows:
                if edge_id in visited_edges:
                    continue
                group = (node, rel_type, outgoing)
                if fanout[group] >= max_fanout:
                    continue
                fanout[group] += 1
                visited_edges.add(edge_id)
                if neighbour not in visited_nodes:
                    visited_nodes.add(neighbour)
                    next_frontier[neighbour] = frontier[node]
                props = dict(raw_props)
                evidence = tuple(props.pop("evidence", None) or ())
                for internal in ("confidence", "ontology_version"):
                    props.pop(internal, None)
                source, target = (node, neighbour) if outgoing else (neighbour, node)
                edges.append(RelationEdge(source, rel_type, target, props, evidence, hop))
                if len(edges) == max_facts:
                    return edges
            frontier = next_frontier
        return edges

    # --- misc -----------------------------------------------------------------

    async def stats(self) -> dict[str, int]:
        result = await self._graph.ro_query(q.COUNT_NODES)
        documents, chunks, entities, relations = result.result_set[0]
        return {
            "documents": int(documents),
            "chunks": int(chunks),
            "entities": int(entities),
            "relations": int(relations),
        }

    async def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        """Escape hatch: raw Cypher. Documented, but outside the API stability promise."""
        result = await self._graph.query(cypher, params or {})
        return list(result.result_set)

    async def drop(self) -> None:
        """Delete this knowledge base's graph entirely."""
        await self._graph.delete()
        self._meta = None


def _batches[T](items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _entity_type(labels: Sequence[str]) -> str:
    return next((label for label in labels if label != "Entity"), "Entity")


def _chunk_row(chunk: Chunk, embedding: Sequence[float]) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "doc_id": chunk.doc_id,
        "idx": chunk.idx,
        "text": chunk.text,
        "text_norm": fold(chunk.text),
        "token_count": chunk.token_count,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "section_path": list(chunk.section_path),
        "page_index": chunk.page_index,
        "chunk_strategy": chunk.chunk_strategy,
        "embedding": [float(x) for x in embedding],
    }


def _chunk_hit(row: Sequence[Any], *, is_distance: bool = True) -> ChunkHit:
    cid, doc_id, source, text, score, section, page, prev_id, prev_text, next_id, next_text = row
    return ChunkHit(
        chunk_id=cid,
        doc_id=doc_id,
        source=source,
        text=text,
        score=1.0 - float(score) if is_distance else float(score),
        section_path=tuple(section or ()),
        page_index=page,
        prev_id=prev_id,
        prev_text=prev_text,
        next_id=next_id,
        next_text=next_text,
    )
