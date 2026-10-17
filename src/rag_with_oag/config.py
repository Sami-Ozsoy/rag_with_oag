"""User-facing configuration.

LLMConfig and EmbedConfig read missing fields from the environment
(``RAG_OAG_LLM_*`` and ``RAG_OAG_EMBED_*``), so a script and the CLI can share
one ``.env``-style setup.
"""

from __future__ import annotations

import os
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CHUNK_STRATEGIES: tuple[str, ...] = (
    "markdown",
    "section-aware",
    "paragraph",
    "token-window",
)


def resolve_api_key(api_key: SecretStr | None, base_url: str) -> str:
    """The configured key, else ``OPENAI_API_KEY`` for OpenAI itself, else a placeholder.

    ``OPENAI_API_KEY`` is never sent to any other server: it would hand an OpenAI
    secret to whoever runs that endpoint. Servers such as vLLM or Ollama ignore the
    key, but the SDK requires one.
    """
    if api_key is not None:
        return api_key.get_secret_value()
    if urlsplit(base_url).hostname == urlsplit(DEFAULT_BASE_URL).hostname:
        return os.environ.get("OPENAI_API_KEY") or "not-needed"
    return "not-needed"


class LLMConfig(BaseSettings):
    """A chat model behind any OpenAI-compatible endpoint."""

    model_config = SettingsConfigDict(env_prefix="RAG_OAG_LLM_", frozen=True, extra="forbid")

    model: str
    base_url: str = DEFAULT_BASE_URL
    api_key: SecretStr | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


class EmbedConfig(BaseSettings):
    """An embedding model behind any OpenAI-compatible ``/embeddings`` endpoint."""

    model_config = SettingsConfigDict(env_prefix="RAG_OAG_EMBED_", frozen=True, extra="forbid")

    model: str
    base_url: str = DEFAULT_BASE_URL
    api_key: SecretStr | None = None
    batch_size: int = Field(default=64, ge=1, le=2048)
    dimensions: int | None = Field(default=None, ge=1)
    timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


class RetrievalConfig(BaseModel):
    """Chunking and retrieval knobs. The defaults suit most prototypes."""

    model_config = {"frozen": True, "extra": "forbid"}

    k: int = Field(default=6, ge=1, le=100)
    expand_neighbors: bool = True
    context_token_budget: int = Field(default=6000, ge=256)
    # oag / hybrid
    hops: int = Field(default=2, ge=1, le=5)
    max_paths: int = Field(default=50, ge=1)
    max_facts: int = Field(default=40, ge=1)
    link_k: int = Field(default=5, ge=1, description="entities used as graph entry points")
    link_min_similarity: float = Field(
        default=0.5,
        ge=-1,
        le=1,
        description="vector matches below this cosine similarity are not linked",
    )
    fact_budget_ratio: float = Field(
        default=0.35, gt=0, lt=1, description="share of the context budget for facts in hybrid"
    )
    rrf_k: int = Field(default=60, ge=1)
    chunk_strategies: tuple[str, ...] = DEFAULT_CHUNK_STRATEGIES
    chunk_min_tokens: int = Field(default=120, ge=1)
    chunk_max_tokens: int = Field(default=320, ge=1)
    chunk_hard_max_tokens: int = Field(
        default=1000,
        ge=1,
        description="Upper bound in model tokens; longer chunks are re-split to fit the embedder.",
    )

    @field_validator("chunk_strategies")
    @classmethod
    def _known_strategies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        from adaptive_chunking import registry

        if not value:
            raise ValueError("chunk_strategies must not be empty")
        unknown = [name for name in value if name not in registry.names()]
        if unknown:
            raise ValueError(
                f"unknown chunk strategies {unknown}; available: {', '.join(registry.names())}"
            )
        return value

    def model_post_init(self, __context: object) -> None:
        if self.chunk_max_tokens < self.chunk_min_tokens:
            raise ValueError("chunk_max_tokens must be >= chunk_min_tokens")


class ExtractionConfig(BaseModel):
    """How entities and relations are extracted from chunks (needs an ontology)."""

    model_config = {"frozen": True, "extra": "forbid"}

    batch_size: int = Field(default=3, ge=1, le=20, description="chunks per LLM call")
    structured_mode: Literal["auto", "json_schema", "tools", "json_object", "prompt"] = "auto"
    validation_retries: int = Field(default=1, ge=0, le=3)
    resolution_similarity: float = Field(
        default=0.92,
        gt=0,
        le=1,
        description="cosine similarity above which two key-less entities of one type merge",
    )
    cache: bool = True
