"""Index setup, the Meta node and compatibility checks for one knowledge-base graph."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from rag_with_oag.errors import EmbeddingMismatchError, OntologyMismatchError, StoreError
from rag_with_oag.store import queries as q

SCHEMA_VERSION = 1

# HNSW search breadth. FalkorDB defaults to 10, which the M0 spike measured at
# ~0.5 recall@10 on 3k clustered 384-dim vectors; 200 gives ~0.99 with p95 < 1 ms.
VECTOR_EF_RUNTIME = 200

# (label, field) pairs created when a knowledge base is opened. The vector index
# is created later, once the first embedding call reveals the dimension.
RANGE_INDEXES = (
    ("Document", "id"),
    ("Chunk", "id"),
    ("Meta", "kb"),
    ("Entity", "id"),
    ("Entity", "name_norm"),
    ("ExtractionCache", "key"),
)
# Entity names are matched in folded form; see rag_with_oag._text.fold.
FULLTEXT_INDEXES = (("Chunk", "text"), ("Entity", "name_norm"))
VECTOR_INDEXES = (("Chunk", "embedding"), ("Entity", "embedding"))


@dataclass(frozen=True, slots=True)
class Meta:
    kb: str
    schema_version: int
    embed_model: str | None
    embed_dim: int | None
    ontology_version: str | None = None
    ontology_name: str | None = None


async def existing_indexes(graph: Any) -> set[tuple[str, str, str]]:
    """Return ``{(label, field, kind)}`` with kind in RANGE / FULLTEXT / VECTOR."""
    try:
        result = await graph.ro_query(q.LIST_INDEXES)
    except Exception as exc:
        if "empty key" in str(exc).lower():  # graph does not exist yet
            return set()
        raise
    found: set[tuple[str, str, str]] = set()
    for row in result.result_set:
        label, _fields, kinds = row[0], row[1], row[2]
        for field, field_kinds in dict(kinds).items():
            for kind in field_kinds:
                found.add((label, field, kind))
    return found


async def ensure_base_schema(graph: Any, kb: str) -> Meta:
    """Create the Meta node and the non-vector indexes if missing; return Meta."""
    await graph.query(q.CREATE_META, {"kb": kb, "schema_version": SCHEMA_VERSION})
    present = await existing_indexes(graph)
    for label, field in RANGE_INDEXES:
        if (label, field, "RANGE") not in present:
            await graph.query(q.CREATE_RANGE_INDEX.format(label=label, field=field))
    for label, field in FULLTEXT_INDEXES:
        if (label, field, "FULLTEXT") not in present:
            await graph.query(q.CREATE_FULLTEXT_INDEX.format(label=label, field=field))
    meta = await read_meta(graph)
    if meta is None:
        raise StoreError("Meta node missing right after creation")
    if meta.schema_version > SCHEMA_VERSION:
        raise StoreError(
            f"knowledge base '{kb}' uses schema v{meta.schema_version}, "
            f"this library supports up to v{SCHEMA_VERSION}; upgrade rag_with_oag"
        )
    return meta


async def read_meta(graph: Any) -> Meta | None:
    result = await graph.ro_query(q.GET_META)
    if not result.result_set:
        return None
    kb, version, model, dim, ontology_version, ontology_name = result.result_set[0]
    return Meta(
        kb=kb,
        schema_version=int(version),
        embed_model=model,
        embed_dim=dim,
        ontology_version=ontology_version,
        ontology_name=ontology_name,
    )


def check_embed_model(meta: Meta, model: str) -> None:
    """Fail at open time when the configured embedder differs from the stored one."""
    if meta.embed_model is not None and meta.embed_model != model:
        raise EmbeddingMismatchError(
            f"knowledge base '{meta.kb}' was embedded with '{meta.embed_model}', "
            f"but the configured embedder is '{model}'"
        )


def check_ontology(meta: Meta, version: str) -> None:
    """Refuse to mix entities extracted under two different ontologies."""
    if meta.ontology_version is not None and meta.ontology_version != version:
        raise OntologyMismatchError(
            f"knowledge base '{meta.kb}' was extracted with ontology "
            f"'{meta.ontology_name}' ({meta.ontology_version}), but the configured ontology "
            f"is {version}; use a new knowledge base name for the changed ontology"
        )


async def record_ontology(graph: Any, meta: Meta, *, version: str, name: str, yaml: str) -> Meta:
    await graph.query(
        q.SET_META_ONTOLOGY, {"kb": meta.kb, "version": version, "name": name, "yaml": yaml}
    )
    return replace(meta, ontology_version=version, ontology_name=name)


async def ensure_vector_index(graph: Any, meta: Meta, *, model: str, dim: int) -> Meta:
    """Record the embedding model/dimension on first use, verify it afterwards."""
    if meta.embed_dim is not None and meta.embed_dim != dim:
        raise EmbeddingMismatchError(
            f"knowledge base '{meta.kb}' stores {meta.embed_dim}-dim vectors, "
            f"but '{model}' returned {dim} dimensions"
        )
    if meta.embed_dim is None:
        await graph.query(
            q.SET_META_EMBEDDING, {"kb": meta.kb, "embed_model": model, "embed_dim": dim}
        )
        meta = replace(meta, embed_model=model, embed_dim=dim)
    await create_vector_indexes(graph, dim)
    return meta


async def create_vector_indexes(graph: Any, dim: int) -> None:
    """Create whichever vector indexes are missing (idempotent)."""
    present = await existing_indexes(graph)
    for label, field in VECTOR_INDEXES:
        if (label, field, "VECTOR") not in present:
            await graph.query(
                q.CREATE_VECTOR_INDEX.format(
                    label=label, field=field, dim=dim, ef_runtime=VECTOR_EF_RUNTIME
                )
            )
