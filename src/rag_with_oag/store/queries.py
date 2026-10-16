"""Every Cypher statement the library runs, in one place.

Values always travel as parameters. Labels, relationship types and hop counts
cannot be parameters in Cypher; the templates that take them are formatted
only through the helper functions below, which accept validated identifiers
and integers and nothing else.
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

DOCUMENT_STATE = "MATCH (d:Document {id: $id}) RETURN d.status, d.extracted_version"

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
SET c.doc_id = r.doc_id, c.idx = r.idx, c.text = r.text, c.token_count = r.token_count,
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

# --- ingest: extraction cache and graph --------------------------------------

GET_EXTRACTIONS = "UNWIND $keys AS k MATCH (x:ExtractionCache {key: k}) RETURN x.key, x.data"
PUT_EXTRACTIONS = "UNWIND $rows AS r MERGE (x:ExtractionCache {key: r.key}) SET x.data = r.data"

EXISTING_ENTITIES = "UNWIND $ids AS i MATCH (e:Entity {id: i}) RETURN e.id"

_FIND_BY_KEY = "MATCH (e:Entity:{label}) WHERE e.{key} = $value RETURN e.id LIMIT 1"
_FIND_BY_NAME = "MATCH (e:Entity:{label} {{name_norm: $name}}) RETURN e.id LIMIT 1"
_FIND_KEYLESS_BY_NAME = (
    "MATCH (e:Entity:{label} {{name_norm: $name}}) WHERE e.{key} IS NULL RETURN e.id LIMIT 1"
)


def find_by_key(label: str, key: str) -> str:
    return _FIND_BY_KEY.format(label=safe_identifier(label), key=safe_identifier(key))


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

# Chunks that mention any of the given entities, closest to the question first.
EVIDENCE_CHUNKS = """
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.id IN $entity_ids
WITH DISTINCT c
MATCH (d:Document)-[:HAS_CHUNK]->(c)
OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(c)
OPTIONAL MATCH (c)-[:NEXT]->(next:Chunk)
WITH c, d, prev, next, vec.cosineDistance(c.embedding, vecf32($q)) AS score
RETURN c.id, c.doc_id, d.source, c.text, score, c.section_path, c.page_index,
       prev.id, prev.text, next.id, next.text
ORDER BY score ASC
LIMIT $k
"""

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

ENTITY_DETAILS = """
UNWIND $ids AS i
MATCH (e:Entity {id: i})
RETURN e.id, e.name, labels(e), properties(e)
"""

_EXPAND = """
MATCH (s:Entity) WHERE s.id IN $seeds
MATCH p = (s)-[:{rels}*1..{hops}]-(:Entity)
WITH p LIMIT $max_paths
UNWIND relationships(p) AS r
WITH DISTINCT r
RETURN startNode(r).id, type(r), properties(r), endNode(r).id
LIMIT $max_facts
"""


def expand(rel_types: Sequence[str], hops: int) -> str:
    """Variable-length expansion over the ontology's relation types only."""
    if not rel_types:
        raise ValueError("expand needs at least one relation type")
    if not isinstance(hops, int) or isinstance(hops, bool) or not 1 <= hops <= 5:
        raise ValueError(f"hops must be an integer between 1 and 5, got {hops!r}")
    rels = "|".join(safe_identifier(r) for r in rel_types)
    return _EXPAND.format(rels=rels, hops=hops)


# --- stats ------------------------------------------------------------------

COUNT_NODES = """
OPTIONAL MATCH (d:Document) WITH count(d) AS documents
OPTIONAL MATCH (c:Chunk) WITH documents, count(c) AS chunks
OPTIONAL MATCH (e:Entity) WITH documents, chunks, count(e) AS entities
OPTIONAL MATCH (:Entity)-[r]->(:Entity)
RETURN documents, chunks, entities, count(r) AS relations
"""
