"""Local cross-encoder reranking via the ``local`` extra."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

DEFAULT_LOCAL_MODEL = "jinaai/jina-reranker-v2-base-multilingual"


class FastEmbedReranker:
    """The default multilingual model requires FastEmbed 0.4.2 or newer."""

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL, *, batch_size: int = 64) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:
            raise ImportError(
                "local reranking needs fastembed: pip install 'rag_with_oag[local]'"
            ) from exc
        self.model = model
        self._batch_size = batch_size
        self._impl: Any = TextCrossEncoder(model_name=model)

    async def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        def run() -> list[float]:
            return [
                float(score)
                for score in self._impl.rerank(query, texts, batch_size=self._batch_size)
            ]

        return await asyncio.to_thread(run)
