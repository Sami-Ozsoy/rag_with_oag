"""Exception hierarchy. Everything the library raises on purpose derives from RagOagError."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

import openai

from rag_with_oag.config import DEFAULT_BASE_URL

if TYPE_CHECKING:
    from rag_with_oag.config import EmbedConfig, LLMConfig


class RagOagError(Exception):
    """Base class for all rag_with_oag errors."""


class ConfigError(RagOagError, ValueError):
    """Invalid or contradictory configuration."""


class ModelConnectionError(RagOagError):
    """The model endpoint could not be reached or timed out."""


class ModelAuthError(RagOagError):
    """The model endpoint rejected authentication or denied access."""


class ModelNotFoundError(RagOagError):
    """The model endpoint did not find the requested model."""


MODEL_ENDPOINT_ERRORS = (ModelConnectionError, ModelAuthError, ModelNotFoundError)


class KnowledgeBaseClosedError(RagOagError, RuntimeError):
    """The knowledge base was closed; its model clients are gone. Create a new one."""


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


@contextmanager
def _translate_model_errors(
    client: openai.AsyncOpenAI,
    config: LLMConfig | EmbedConfig,
    *,
    role: Literal["chat", "embedding"],
    structured_request: bool = False,
) -> Iterator[None]:
    base_url, model = config.base_url, config.model
    try:
        yield
    except openai.APITimeoutError as exc:
        raise ModelConnectionError(
            f"The {role} model server at {base_url} timed out (model {model}). "
            f"Increase timeout= in the {role} model settings."
        ) from exc
    except openai.APIConnectionError as exc:
        raise ModelConnectionError(
            f"Cannot reach the {role} model server at {base_url} (model {model}). "
            'Is it running? For Ollama, start it with "ollama serve".'
        ) from exc
    except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
        env_var = "RAG_OAG_LLM_API_KEY" if role == "chat" else "RAG_OAG_EMBED_API_KEY"
        if urlsplit(base_url).hostname == urlsplit(DEFAULT_BASE_URL).hostname:
            env_var += " or OPENAI_API_KEY"
        if config.api_key is None and getattr(client, "api_key", None) == "not-needed":
            message = f"No API key is configured for {base_url} ({role} model {model})."
        else:
            message = f"The server at {base_url} rejected the API key for {role} model {model}."
        raise ModelAuthError(f"{message} Pass api_key= or set {env_var}.") from exc
    except openai.NotFoundError as exc:
        if structured_request:
            raise  # Structured generation also uses 404 to detect unsupported features.
        raise ModelNotFoundError(
            f"Model {model} was not found at {base_url} ({role} endpoint). "
            f'For Ollama, install it with "ollama pull {model}".'
        ) from exc
