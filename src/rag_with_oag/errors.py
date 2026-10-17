"""Exception hierarchy. Everything the library raises on purpose derives from RagOagError."""

from __future__ import annotations


class RagOagError(Exception):
    """Base class for all rag_with_oag errors."""


class ConfigError(RagOagError, ValueError):
    """Invalid or contradictory configuration."""


class ModeNotAvailableError(RagOagError):
    """The requested query mode cannot run on this knowledge base."""


class EmbeddingMismatchError(RagOagError):
    """The embedder does not match the one the knowledge base was built with.

    Mixing vectors from different models or dimensions silently ruins retrieval,
    so the library refuses instead. Rebuild the vectors with ``kb.reembed()`` (v0.3)
    or open the knowledge base with the original embedder.
    """


class StoreError(RagOagError):
    """The graph store returned an error or is in an unexpected state."""


class OntologyError(RagOagError):
    """The ontology definition is inconsistent."""


class OntologyMismatchError(RagOagError):
    """The knowledge base was extracted with a different ontology.

    Entities and relations from two ontologies cannot be mixed safely. Use a new
    knowledge base name for the new ontology (re-extraction lands in v0.3).
    """


class StructuredOutputError(RagOagError):
    """The model did not return valid structured output with any supported method."""


class OutputTruncatedError(RagOagError):
    """The model hit its token limit before finishing structured output.

    Raise ``LLMConfig.max_tokens`` or lower ``ExtractionConfig.batch_size``.
    """
