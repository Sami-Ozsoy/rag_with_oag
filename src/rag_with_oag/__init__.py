"""rag_with_oag: RAG, OAG and hybrid retrieval over one FalkorDB graph."""

from importlib.metadata import PackageNotFoundError, version

from rag_with_oag.config import EmbedConfig, ExtractionConfig, LLMConfig, RetrievalConfig
from rag_with_oag.errors import (
    ConfigError,
    EmbeddingMismatchError,
    KnowledgeBaseClosedError,
    ModelAuthError,
    ModelConnectionError,
    ModelNotFoundError,
    ModeNotAvailableError,
    OntologyError,
    OntologyMismatchError,
    OutputTruncatedError,
    RagOagError,
    StoreError,
    StructuredOutputError,
)
from rag_with_oag.kb import AsyncKnowledgeBase, KnowledgeBase
from rag_with_oag.ontology import EntityType, Ontology, Property, RelationType
from rag_with_oag.ontology.preview import (
    DroppedItems,
    EntityTypeStats,
    PreviewReport,
    RelationTypeStats,
)
from rag_with_oag.types import (
    ChunkHit,
    Citation,
    DeleteReport,
    Document,
    EntityHit,
    Fact,
    IngestReport,
    Mode,
    QueryResult,
    RetrievedContext,
)

try:
    __version__ = version("rag_with_oag")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    __version__ = "0.0.0"

__all__ = [
    "AsyncKnowledgeBase",
    "ChunkHit",
    "Citation",
    "ConfigError",
    "DeleteReport",
    "Document",
    "DroppedItems",
    "EmbedConfig",
    "EmbeddingMismatchError",
    "EntityHit",
    "EntityType",
    "EntityTypeStats",
    "ExtractionConfig",
    "Fact",
    "IngestReport",
    "KnowledgeBase",
    "KnowledgeBaseClosedError",
    "LLMConfig",
    "Mode",
    "ModeNotAvailableError",
    "ModelAuthError",
    "ModelConnectionError",
    "ModelNotFoundError",
    "Ontology",
    "OntologyError",
    "OntologyMismatchError",
    "OutputTruncatedError",
    "PreviewReport",
    "Property",
    "QueryResult",
    "RagOagError",
    "RelationType",
    "RelationTypeStats",
    "RetrievalConfig",
    "RetrievedContext",
    "StoreError",
    "StructuredOutputError",
    "__version__",
]
