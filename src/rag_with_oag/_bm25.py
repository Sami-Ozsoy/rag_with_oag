"""In-process BM25 Okapi with one token list per chunk, rebuilt after writes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from heapq import nsmallest
from math import fsum, log1p

_K1 = 1.2
_B = 0.75


class BM25Index:
    def __init__(self, chunks: dict[str, list[str]]) -> None:
        self._chunks = chunks
        size = len(chunks)
        self._average_length = sum(map(len, chunks.values())) / max(size, 1)
        document_frequencies: Counter[str] = Counter()
        for tokens in chunks.values():
            document_frequencies.update(set(tokens))
        self._idf = {
            term: log1p((size - df + 0.5) / (df + 0.5)) for term, df in document_frequencies.items()
        }

    def rank(self, query_tokens: Iterable[str], k: int) -> list[tuple[str, float]]:
        terms = set(query_tokens).intersection(self._idf)
        if not terms or k <= 0:
            return []

        scores = []
        for chunk_id, tokens in self._chunks.items():
            frequencies = Counter(token for token in tokens if token in terms)
            if not frequencies:
                continue
            norm = _K1 * (1 - _B + _B * len(tokens) / self._average_length)
            score = fsum(
                self._idf[term] * (freq * (_K1 + 1)) / (freq + norm)
                for term, freq in frequencies.items()
            )
            scores.append((chunk_id, score))
        return nsmallest(k, scores, key=lambda pair: (-pair[1], pair[0]))
