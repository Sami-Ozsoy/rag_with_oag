"""Local CPU embeddings via fastembed (``pip install rag_with_oag[local]``)."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Sequence
from typing import Any

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"

# (query prefix, document prefix) the models were trained with. fastembed lists
# them in its model descriptions but does not add them, so we do.
KNOWN_PREFIXES: dict[str, tuple[str, str]] = {
    "Qwen/Qwen3-Embedding-0.6B": (
        "Instruct: Given a question, retrieve passages that answer it\nQuery:",
        "",
    ),
    "Qwen/Qwen3-Embedding-0.6B-Q": (
        "Instruct: Given a question, retrieve passages that answer it\nQuery:",
        "",
    ),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: "),
    "nomic-ai/nomic-embed-text-v1.5-Q": ("search_query: ", "search_document: "),
    "google/embeddinggemma-300m": ("task: search result | query: ", "title: none | text: "),
}


class FastEmbedEmbedder:
    """``query_prefix`` and ``document_prefix`` default to the model's own (``KNOWN_PREFIXES``)."""

    def __init__(
        self,
        model: str = DEFAULT_LOCAL_MODEL,
        *,
        batch_size: int = 64,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "local embeddings need fastembed: pip install 'rag_with_oag[local]'"
            ) from exc
        known_query, known_document = KNOWN_PREFIXES.get(model, ("", ""))
        self.model = model
        self.query_prefix = known_query if query_prefix is None else query_prefix
        self.document_prefix = known_document if document_prefix is None else document_prefix
        self.usage: Counter[str] = Counter()
        self._batch_size = batch_size
        self._impl: Any = TextEmbedding(model_name=model)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        def run() -> list[list[float]]:
            return [v.tolist() for v in self._impl.embed(list(texts), batch_size=self._batch_size)]

        return await asyncio.to_thread(run)
