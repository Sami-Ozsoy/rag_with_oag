"""Query-time scoring of retrieved passages, with an optional FastEmbed implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    """Return one relevance score per text in input order. Higher scores rank first."""

    async def rerank(self, query: str, texts: Sequence[str]) -> list[float]: ...
