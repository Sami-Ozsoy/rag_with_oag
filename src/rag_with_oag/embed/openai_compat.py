from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI

from rag_with_oag.config import EmbedConfig, resolve_api_key


class OpenAIEmbedder:
    """Batched calls to an OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(self, config: EmbedConfig, *, client: AsyncOpenAI | None = None) -> None:
        self.config = config
        self.model = config.model
        self.usage: Counter[str] = Counter()
        self._client = client or AsyncOpenAI(
            base_url=config.base_url,
            api_key=resolve_api_key(config.api_key, config.base_url),
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        size = self.config.batch_size
        for i in range(0, len(texts), size):
            vectors.extend(await self._embed_batch(texts[i : i + size]))
        return vectors

    async def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        params: dict[str, Any] = {}
        if self.config.dimensions is not None:
            params["dimensions"] = self.config.dimensions
        response = await self._client.embeddings.create(
            model=self.model, input=list(batch), encoding_format="float", **params
        )
        if response.usage is not None:
            self.usage["embedding_tokens"] += response.usage.prompt_tokens
        # Servers are not required to keep input order; ``index`` is authoritative.
        items = sorted(response.data, key=lambda d: d.index)
        if len(items) != len(batch):
            raise RuntimeError(
                f"embedding endpoint returned {len(items)} vectors for {len(batch)} inputs"
            )
        return [list(item.embedding) for item in items]

    async def aclose(self) -> None:
        await self._client.close()
