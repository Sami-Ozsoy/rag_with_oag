"""Every Cypher statement the library runs, in one place.

Values travel as parameters. Templates that interpolate labels, relationship
types or property names use the helper functions below to validate identifiers.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_identifier(name: str) -> str:
    if not IDENTIFIER.fullmatch(name):
        raise ValueError(f"unsafe Cypher identifier: {name!r}")
    return name


# --- schema -----------------------------------------------------------------

LIST_INDEXES = "CALL db.indexes()"
CREATE_RANGE_INDEX = "CREATE INDEX FOR (n:{label}) ON (n.{field})"
CREATE_FULLTEXT_INDEX = "CALL db.idx.fulltext.createNodeIndex('{label}', '{field}')"
CREATE_VECTOR_INDEX = (
    "CREATE VECTOR INDEX FOR (n:{label}) ON (n.{field}) "
    "OPTIONS {{dimension: {dim}, similarityFunction: 'cosine', efRuntime: {ef_runtime}}}"
)

GET_META = """
MATCH (m:Meta)
RETURN m.kb, m.schema_version, m.embed_model, m.embed_dim, m.ontology_version, m.ontology_name
LIMIT 1
"""
CREATE_META = """
MERGE (m:Meta {kb: $kb})
ON CREATE SET m.schema_version = $schema_version, m.created_at = timestamp()
"""
SET_META_EMBEDDING = """
MATCH (m:Meta {kb: $kb})
SET m.embed_model = $embed_model, m.embed_dim = $embed_dim
"""
SET_META_ONTOLOGY = """
MATCH (m:Meta {kb: $kb})
SET m.ontology_version = $version, m.ontology_name = $name, m.ontology_yaml = $yaml
"""

# --- ingest: documents and chunks -------------------------------------------

DOCUMENT_STATE = "MATCH (d:Document {id: $id}) RETURN d.status, d.extracted_version, d.source"
SET_DOCUMENT_SOURCE = "MATCH (d:Document {id: $id}) SET d.source = $source"
REPLACEMENT_IDS = """
MATCH (:Document {id: $id, source: $source}), (old:Document {source: $source})
WHERE old.id <> $id AND old.status = $status
RETURN old.id
ORDER BY old.id
"""

# A document is written as 'pending' first and flipped to 'complete' last, so an
# interrupted ingest is retried instead of being skipped as already present.
UPSERT_DOCUMENT = """
MERGE (d:Document {id: $id})
SET d.source = $source, d.title = $title, d.content_hash = $content_hash,
    d.metadata = $metadata, d.status = 'pending', d.updated_at = timestamp()
"""
MARK_DOCUMENT_COMPLETE = """
MATCH (d:Document {id: $id})
SET d.status = 'complete', d.chunk_count = $chunk_count
"""
MARK_DOCUMENT_EXTRACTED = "MATCH (d:Document {id: $id}) SET d.extracted_version = $version"

UPSERT_CHUNKS = """
UNWIND $rows AS r
MATCH (d:Document {id: r.doc_id})
MERGE (c:Chunk {id: r.id})
SET c.doc_id = r.doc_id, c.idx = r.idx, c.text = r.text, c.text_norm = r.text_norm,
    c.token_count = r.token_count,
    c.start_char = r.start_char, c.end_char = r.end_char, c.section_path = r.section_path,
    c.page_index = r.page_index, c.chunk_strategy = r.chunk_strategy,
    c.embedding = vecf32(r.embedding)
MERGE (d)-[:HAS_CHUNK]->(c)
"""

LINK_NEXT = """
UNWIND $pairs AS p
MATCH (a:Chunk {id: p[0]}), (b:Chunk {id: p[1]})
MERGE (a)-[:NEXT]->(b)
"""

DOCUMENT_CHUNKS = """
MATCH (:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk)
RETURN c.id, c.doc_id, c.idx, c.text, c.token_count, c.start_char, c.end_char,
       c.section_path, c.page_index, c.chunk_strategy
ORDER BY c.idx
"""

CHUNKS_MISSING_TEXT_NORM = """
MATCH (c:Chunk) WHERE c.text_norm IS NULL
RETURN c.id, c.text
LIMIT $limit
"""

SET_CHUNK_TEXT_NORM = """
UNWIND $rows AS r
MATCH (c:Chunk {id: r.id}) WHERE c.text_norm IS NULL
SET c.text_norm = r.text_norm
"""

# --- delete: documents, evidence and orphans ---------------------------------

DOCUMENT_TARGET_IDS = """
MATCH (d:Document {id: $target}) RETURN d.id AS doc_id, null AS source
UNION ALL
MATCH (d:Document) WHERE d.source IN $sources RETURN d.id AS doc_id, d.source AS source
"""

DOCUMENT_CHUNK_IDS = "MATCH (:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk) RETURN c.id"

MENTIONED_ENTITY_IDS = """
UNWIND $ids AS cid
MATCH (:Chunk {id: cid})-[:MENTIONS]->(e:Entity)
RETURN DISTINCT e.id
"""

_FILTER_EVIDENCE = """
MATCH (:Entity)-[r]->(:Entity)
WHERE any(cid IN coalesce(r.evidence, []) WHERE cid IN $ids)
WITH r, [cid IN r.evidence WHERE NOT cid IN $ids] AS remaining
"""

DELETE_UNSUPPORTED_RELATIONS = (
    _FILTER_EVIDENCE
    + """
WHERE size(remaining) = 0
WITH r, id(r) AS edge_id
DELETE r
RETURN edge_id
"""
)

TRIM_RELATION_EVIDENCE = (
    _FILTER_EVIDENCE
    + """
WHERE size(remaining) > 0
SET r.evidence = remaining
RETURN id(r)
"""
)

DELETE_CHUNKS = "UNWIND $ids AS cid MATCH (c:Chunk {id: cid}) DETACH DELETE c"
DELETE_DOCUMENT = "MATCH (d:Document {id: $id}) DETACH DELETE d"

DELETE_ORPHAN_ENTITIES = """
UNWIND $ids AS eid
MATCH (e:Entity {id: eid})
OPTIONAL MATCH (e)-[r]-()
WITH e, count(r) AS relationships
WHERE relationships = 0
DELETE e
"""

# --- ingest: extraction cache and graph --------------------------------------

GET_EXTRACTIONS = "UNWIND $keys AS k MATCH (x:ExtractionCache {key: k}) RETURN x.key, x.data"
PUT_EXTRACTIONS = "UNWIND $rows AS r MERGE (x:ExtractionCache {key: r.key}) SET x.data = r.data"

EXISTING_ENTITIES = "UNWIND $ids AS i MATCH (e:Entity {id: i}) RETURN e.id"

_FIND_BY_KEY = "MATCH (e:Entity:{label}) WHERE e.{key} = $value RETURN e.id LIMIT 1"
_FIND_BY_KEYS = """
MATCH (e:Entity:{label})
WITH e, toString(e.{key}) AS value
WHERE value IN $values
RETURN e.id, e.name, value
"""
_FIND_BY_NAME = "MATCH (e:Entity:{label} {{name_norm: $name}}) RETURN e.id LIMIT 1"
_FIND_KEYLESS_BY_NAME = (
    "MATCH (e:Entity:{label} {{name_norm: $name}}) WHERE e.{key} IS NULL RETURN e.id LIMIT 1"
)


def find_by_key(label: str, key: str) -> str:
    return _FIND_BY_KEY.format(label=safe_identifier(label), key=safe_identifier(key))


def find_by_keys(label: str, key: str) -> str:
    return _FIND_BY_KEYS.format(label=safe_identifier(label), key=safe_identifier(key))


def find_by_name(label: str, *, keyless: str | None = None) -> str:
    """Entity of ``label`` with this folded name; with ``keyless``, only one lacking that key."""
    if keyless is None:
        return _FIND_BY_NAME.format(label=safe_identifier(label))
    return _FIND_KEYLESS_BY_NAME.format(label=safe_identifier(label), key=safe_identifier(keyless))


_UPSERT_ENTITIES = """
UNWIND $rows AS r
MERGE (e:Entity:{label} {{id: r.id}})
ON CREATE SET e.name = r.name, e.name_norm = r.name_norm, e.embedding = vecf32(r.embedding)
SET e.ontology_version = $version,
    e.description = CASE WHEN r.description IS NULL THEN e.description ELSE r.description END
SET e += r.props
"""

LINK_MENTIONS = """
UNWIND $rows AS r
MATCH (c:Chunk {id: r.chunk}), (e:Entity {id: r.entity})
MERGE (c)-[:MENTIONS]->(e)
"""

# Evidence accumulates: the same relation stated in another chunk adds that chunk.
_UPSERT_RELATIONS = """
UNWIND $rows AS r
MATCH (a:Entity {{id: r.source}}), (b:Entity {{id: r.target}})
MERGE (a)-[x:{rel}]->(b)
SET x += r.props, x.ontology_version = $version,
    x.confidence = CASE WHEN x.confidence IS NULL OR r.confidence > x.confidence
                        THEN r.confidence ELSE x.confidence END,
    x.evidence = CASE WHEN x.evidence IS NULL THEN r.evidence
                      ELSE x.evidence + [v IN r.evidence WHERE NOT v IN x.evidence] END
"""


def upsert_entities(label: str) -> str:
    return _UPSERT_ENTITIES.format(label=safe_identifier(label))


def upsert_relations(rel_type: str) -> str:
    return _UPSERT_RELATIONS.format(rel=safe_identifier(rel_type))


# --- retrieval: chunks ---------------------------------------------------------

# queryNodes yields cosine *distance* (0 = identical), verified in the M0 spike.
SEARCH_CHUNKS = """
CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($q)) YIELD node AS c, score
MATCH (d:Document)-[:HAS_CHUNK]->(c)
OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(c)
OPTIONAL MATCH (c)-[:NEXT]->(next:Chunk)
RETURN c.id, c.doc_id, d.source, c.text, score, c.section_path, c.page_index,
       prev.id, prev.text, next.id, next.text
ORDER BY score ASC
"""

CHUNK_TEXT_BATCH = """
MATCH (c:Chunk)
WHERE $after IS NULL OR c.id > $after
RETURN c.id, c.text
ORDER BY c.id
LIMIT $limit
"""

CHUNKS_BY_IDS = """
UNWIND $rows AS r
MATCH (c:Chunk {id: r.id})
WITH c, r.score AS score
MATCH (d:Document)-[:HAS_CHUNK]->(c)
OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(c)
OPTIONAL MATCH (c)-[:NEXT]->(next:Chunk)
RETURN c.id, c.doc_id, d.source, c.text, score, c.section_path, c.page_index,
       prev.id, prev.text, next.id, next.text
"""

_RANK_CHUNKS = """
OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(c)
OPTIONAL MATCH (c)-[:NEXT]->(next:Chunk)
WITH c, d, prev, next, vec.cosineDistance(c.embedding, vecf32($q)) AS score
RETURN c.id, c.doc_id, d.source, c.text, score, c.section_path, c.page_index,
       prev.id, prev.text, next.id, next.text
ORDER BY score ASC
LIMIT $k
"""

EVIDENCE_CHUNKS = (
    """
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.id IN $entity_ids
WITH DISTINCT c
MATCH (d:Document)-[:HAS_CHUNK]->(c)
"""
    + _RANK_CHUNKS
)

SEED_DOCUMENT_CHUNKS = (
    """
UNWIND $entity_ids AS eid
MATCH (e:Entity {id: eid})<-[:MENTIONS]-(:Chunk)<-[:HAS_CHUNK]-(d:Document)
WITH DISTINCT d
MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
"""
    + _RANK_CHUNKS
)

# --- retrieval: entities -------------------------------------------------------

SEARCH_ENTITIES_VECTOR = """
CALL db.idx.vector.queryNodes('Entity', 'embedding', $k, vecf32($q)) YIELD node, score
RETURN node.id, node.name, labels(node), score
ORDER BY score ASC
"""

SEARCH_ENTITIES_TEXT = """
CALL db.idx.fulltext.queryNodes('Entity', $q) YIELD node, score
RETURN node.id, node.name, labels(node), score
ORDER BY score DESC
LIMIT $k
"""

FIND_ENTITIES_BY_NAME_NORMS = """
UNWIND $names AS n
MATCH (e:Entity {name_norm: n})
RETURN e.id, e.name, labels(e), e.name_norm
"""

ENTITY_NAMES_NORM = """
UNWIND $ids AS i
MATCH (e:Entity {id: i})
RETURN e.id, e.name_norm
"""

ENTITY_DETAILS = """
UNWIND $ids AS i
MATCH (e:Entity {id: i})
RETURN e.id, e.name, labels(e), properties(e)
"""

_EXPAND_HOP = """
UNWIND $frontier AS eid
MATCH (n:Entity {{id: eid}})-[r:{rels}]->(other:Entity)
RETURN n.id AS node, other.id AS neighbour, id(r) AS edge_id,
       type(r) AS rel_type, properties(r) AS props, true AS outgoing
UNION ALL
UNWIND $frontier AS eid
MATCH (n:Entity {{id: eid}})<-[r:{rels}]-(other:Entity)
RETURN n.id AS node, other.id AS neighbour, id(r) AS edge_id,
       type(r) AS rel_type, properties(r) AS props, false AS outgoing
"""


def expand(rel_types: Sequence[str]) -> str:
    """Get the frontier's incident ontology edges and their stored directions."""
    if not rel_types:
        raise ValueError("expand needs at least one relation type")
    rels = "|".join(safe_identifier(r) for r in rel_types)
    return _EXPAND_HOP.format(rels=rels)


# --- stats ------------------------------------------------------------------

COUNT_NODES = """
OPTIONAL MATCH (d:Document) WITH count(d) AS documents
OPTIONAL MATCH (c:Chunk) WITH documents, count(c) AS chunks
OPTIONAL MATCH (e:Entity) WITH documents, chunks, count(e) AS entities
OPTIONAL MATCH (:Entity)-[r]->(:Entity)
RETURN documents, chunks, entities, count(r) AS relations
"""
