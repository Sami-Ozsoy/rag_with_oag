from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into vectors. ``model`` is stored in the graph's Meta node."""

    model: str
    usage: Counter[str]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
