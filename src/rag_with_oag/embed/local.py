"""Local CPU embeddings via fastembed (``pip install rag_with_oag[local]``)."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Sequence
from typing import Any

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbedder:
    def __init__(self, model: str = DEFAULT_LOCAL_MODEL, *, batch_size: int = 64) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "local embeddings need fastembed: pip install 'rag_with_oag[local]'"
            ) from exc
        self.model = model
        self.usage: Counter[str] = Counter()
        self._batch_size = batch_size
        self._impl: Any = TextEmbedding(model_name=model)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        def run() -> list[list[float]]:
            return [v.tolist() for v in self._impl.embed(list(texts), batch_size=self._batch_size)]

        return await asyncio.to_thread(run)
