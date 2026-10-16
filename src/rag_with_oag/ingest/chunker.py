"""Chunking behind our own protocol.

The default wraps adaptive-oci-chunking, which tries several strategies per
document and keeps the best by intrinsic metrics. The package is alpha with a
single maintainer, so it is pinned, isolated here, and backed by a plain
token-window fallback.

Note: adaptive-oci-chunking measures size with an ASCII word regex, which
under-counts Turkish and other non-ASCII text. Its sizes are therefore only a
target; ``hard_max_tokens`` is enforced afterwards in real model tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from rag_with_oag._tokens import count_tokens, split_spans
from rag_with_oag.config import RetrievalConfig

logger = logging.getLogger(__name__)

PAGE_BREAK = "\f"


@dataclass(frozen=True, slots=True)
class Span:
    """A chunk before it gets an id: offsets into the document text plus structure."""

    start: int
    end: int
    text: str
    section_path: tuple[str, ...]
    strategy: str


class Chunker(Protocol):
    def split(self, text: str) -> list[Span]: ...


class TokenWindowChunker:
    """Fixed windows in model tokens, cut on whitespace. Also the fallback."""

    name = "token-window"

    def __init__(self, max_tokens: int = 320) -> None:
        self.max_tokens = max_tokens

    def split(self, text: str) -> list[Span]:
        spans = []
        for start, end in split_spans(text, self.max_tokens):
            start, end = _trim(text, start, end)
            if end > start:
                spans.append(Span(start, end, text[start:end], (), self.name))
        return spans


class AdaptiveChunker:
    def __init__(self, config: RetrievalConfig | None = None) -> None:
        from adaptive_chunking import AdaptiveChunker as _Impl
        from adaptive_chunking import ChunkingConfig

        cfg = config or RetrievalConfig()
        self.hard_max_tokens = cfg.chunk_hard_max_tokens
        self._impl = _Impl(
            config=ChunkingConfig(
                strategies=list(cfg.chunk_strategies),
                target_min_tokens=cfg.chunk_min_tokens,
                target_max_tokens=cfg.chunk_max_tokens,
            )
        )
        self._fallback = TokenWindowChunker(cfg.chunk_max_tokens)

    def split(self, text: str) -> list[Span]:
        if not text.strip():
            return []
        try:
            result = self._impl.chunk(text)
            spans = [
                Span(
                    c.start_char,
                    c.end_char,
                    text[c.start_char : c.end_char],
                    tuple(c.metadata.get("section_path") or ()),
                    result.strategy_name,
                )
                for c in result.chunks
            ]
        except Exception:
            logger.warning("adaptive chunking failed; using token-window fallback", exc_info=True)
            return self._fallback.split(text)
        if not spans or not _valid_offsets(text, spans):
            logger.warning(
                "adaptive chunking returned unusable offsets; using token-window fallback"
            )
            return self._fallback.split(text)
        return [piece for span in spans for piece in self._enforce_max(text, span)]

    def _enforce_max(self, text: str, span: Span) -> list[Span]:
        start, end = _trim(text, span.start, span.end)
        if end <= start:
            return []
        out = []
        for s, e in split_spans(text[start:end], self.hard_max_tokens):
            s, e = _trim(text, start + s, start + e)
            if e > s:
                out.append(Span(s, e, text[s:e], span.section_path, span.strategy))
        return out


def page_index(text: str, offset: int) -> int | None:
    """0-based page of ``offset`` for page-delimited text (PDF); None otherwise."""
    if PAGE_BREAK not in text:
        return None
    return text.count(PAGE_BREAK, 0, offset)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _valid_offsets(text: str, spans: list[Span]) -> bool:
    """We slice the source by offsets ourselves, so offsets must be in range and non-empty."""
    return all(0 <= s.start <= s.end <= len(text) for s in spans) and any(
        s.text.strip() for s in spans
    )


__all__ = ["AdaptiveChunker", "Chunker", "Span", "TokenWindowChunker", "count_tokens", "page_index"]
