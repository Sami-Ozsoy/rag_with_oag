from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into vectors. ``model`` is stored in the graph's Meta node.

    An embedder may also have ``query_prefix`` and ``document_prefix`` strings for
    models trained with instructions, such as ``"search_query: "`` and
    ``"search_document: "`` for nomic-embed-text or ``"query: "`` / ``"passage: "``
    for e5. They are optional; without them texts are embedded as they are.
    """

    model: str
    usage: Counter[str]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _prefix(embedder: Embedder, kind: str) -> str:
    return str(getattr(embedder, f"{kind}_prefix", "") or "")


async def embed_queries(embedder: Embedder, texts: Sequence[str]) -> list[list[float]]:
    """Vectors for questions."""
    prefix = _prefix(embedder, "query")
    return await embedder.embed([prefix + t for t in texts])


async def embed_documents(embedder: Embedder, texts: Sequence[str]) -> list[list[float]]:
    """Vectors for what is stored and searched: chunks and entities."""
    prefix = _prefix(embedder, "document")
    return await embedder.embed([prefix + t for t in texts])


def embedder_identity(embedder: Embedder) -> str:
    """What stored vectors depend on: the model and the document prefix.

    Recorded in the graph, so reopening with another prefix is refused like
    another model. Without a prefix it is the bare model name, as before.
    """
    prefix = _prefix(embedder, "document")
    return f"{embedder.model} (document prefix {prefix!r})" if prefix else embedder.model
