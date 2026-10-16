"""Connection to FalkorDB: embedded (``path``) or remote (``url``), behind one async client.

For the embedded case, redislite's synchronous client owns the server process
(start, persistence, shutdown) and we talk to it over its unix socket with the
official async client. falkordblite's own async wrapper is not used because its
shutdown path never awaits, which can leave the embedded server unsaved.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from falkordb.asyncio import FalkorDB as AsyncFalkorDB

from rag_with_oag.errors import ConfigError

logger = logging.getLogger(__name__)


class GraphConnection:
    """One database, possibly holding several knowledge-base graphs.

    Several knowledge bases can share a connection by passing it explicitly.
    """

    def __init__(self, *, path: str | Path | None = None, url: str | None = None) -> None:
        if (path is None) == (url is None):
            raise ConfigError("pass exactly one of path (embedded) or url (FalkorDB server)")
        self.path = Path(path).expanduser().resolve() if path is not None else None
        self.url = url
        self._embedded: Any = None
        self._client: AsyncFalkorDB | None = None
        self._lock = asyncio.Lock()

    @property
    def is_open(self) -> bool:
        return self._client is not None

    async def open(self) -> None:
        async with self._lock:
            if self._client is not None:
                return
            if self.path is not None:
                self._embedded = await asyncio.to_thread(self._start_embedded, self.path)
                socket = self._embedded.client.socket_file
                self._client = AsyncFalkorDB(unix_socket_path=socket)
                logger.debug("embedded FalkorDB at %s (socket %s)", self.path, socket)
            else:
                assert self.url is not None
                self._client = AsyncFalkorDB.from_url(self.url)

    @staticmethod
    def _start_embedded(path: Path) -> Any:
        try:
            from redislite.falkordb_client import FalkorDB as EmbeddedFalkorDB
        except ImportError as exc:  # pragma: no cover - falkordblite is a core dependency
            raise ConfigError("embedded mode needs falkordblite: pip install falkordblite") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return EmbeddedFalkorDB(str(path))
        except Exception as exc:
            hint = " On macOS install OpenMP first: brew install libomp." if _is_macos() else ""
            raise ConfigError(f"could not start embedded FalkorDB at {path}: {exc}.{hint}") from exc

    def graph(self, name: str) -> Any:
        if self._client is None:
            raise RuntimeError("connection is not open; await open() first")
        return self._client.select_graph(name)

    async def close(self) -> None:
        async with self._lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None
            if self._embedded is not None:
                await asyncio.to_thread(self._embedded.close)
                self._embedded = None


def _is_macos() -> bool:
    import sys

    return sys.platform == "darwin"
