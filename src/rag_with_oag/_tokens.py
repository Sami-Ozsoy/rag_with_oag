"""Model-token counting with an offline fallback.

tiktoken downloads its BPE table on first use; in an air-gapped environment we
fall back to a conservative character estimate rather than failing ingest.
"""

from __future__ import annotations

import logging
import re
from functools import cache
from typing import Any

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 3  # pessimistic for Turkish and other agglutinative languages


@cache
def _encoding() -> Any | None:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # network or cache failure
        logger.warning("tiktoken unavailable (%s); estimating tokens from characters", exc)
        return None


def count_tokens(text: str) -> int:
    enc = _encoding()
    if enc is None:
        return max(1, len(text) // _CHARS_PER_TOKEN) if text else 0
    return len(enc.encode(text, disallowed_special=()))


def split_spans(text: str, max_tokens: int) -> list[tuple[int, int]]:
    """Cut ``text`` into ``(start, end)`` spans of at most ``max_tokens`` tokens each.

    Cuts fall on whitespace so multi-byte characters and words stay intact; only a
    single word longer than the limit is cut mid-word.
    """
    if count_tokens(text) <= max_tokens:
        return [(0, len(text))]
    spans: list[tuple[int, int]] = []
    start = end = 0
    budget = 0
    for match in re.finditer(r"\S+\s*", text):
        piece_tokens = count_tokens(match.group(0))
        if budget and budget + piece_tokens > max_tokens:
            spans.append((start, end))
            start, budget = match.start(), 0
        if piece_tokens > max_tokens:  # one oversized word: cut it by characters
            step = max(1, len(match.group(0)) * max_tokens // piece_tokens)
            for i in range(match.start(), match.end(), step):
                spans.append((i, min(i + step, match.end())))
            start, end, budget = match.end(), match.end(), 0
            continue
        end = match.end()
        budget += piece_tokens
    if end > start:
        spans.append((start, end))
    return spans
