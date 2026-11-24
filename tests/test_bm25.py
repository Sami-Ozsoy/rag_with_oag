from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from rag_with_oag._bm25 import BM25Index
from rag_with_oag._text import bm25_tokens
from rag_with_oag.store import queries as q
from rag_with_oag.store import repository
from rag_with_oag.store.connection import GraphConnection
from rag_with_oag.store.repository import GraphRepository


def test_bm25_tokenizer_folds_keys_numbers_and_turkish_suffixes() -> None:
    assert bm25_tokens("KS-2024-002’nin ve İÇİN A sözleşmesindeki IŞIK 000123456789") == [
        "ks2024002",
        "ks",
        "2024",
        "002",
        "nin",
        "sozles",
        "isik",
        "000123456789",
    ]


def test_rare_company_name_outranks_shared_registry_terms() -> None:
    chunks = {f"registry-{i:02d}": bm25_tokens("vergi kimlik numarası sicil") for i in range(19)}
    chunks["registry-19"] = bm25_tokens("Kervan Holding vergi kimlik numarası sicil")
    index = BM25Index(chunks)
    question = "Vergi kimlik numarası aranan şirket “Kervan Holding”. Sicilde hangi numara yazıyor?"

    hits = index.rank(bm25_tokens(question), 3)

    assert [cid for cid, _ in hits] == ["registry-19", "registry-00", "registry-01"]
    assert hits[0][1] > hits[1][1] == hits[2][1] > 0


def test_bm25_distinguishes_keys_with_shared_numbers() -> None:
    index = BM25Index(
        {
            "b": bm25_tokens("DS-2023-008"),
            "a": bm25_tokens("KS-2023-008"),
            "c": bm25_tokens("YS-2023-008"),
        }
    )

    assert [cid for cid, _ in index.rank(bm25_tokens("DS-2023-008'in"), 3)] == ["b", "a", "c"]


def test_bm25_scores_chunk_frequency_and_counts_each_query_term_once() -> None:
    index = BM25Index({"b": ["atlas", "atlas", "sicil"], "a": ["atlas"], "c": []})
    idf = math.log1p(1.5 / 2.5)
    expected_b = idf * (2 * 2.2) / (2 + 1.2 * (0.25 + 0.75 * 3 / (4 / 3)))
    expected_a = idf * 2.2 / (1 + 1.2 * (0.25 + 0.75 / (4 / 3)))

    hits = index.rank(bm25_tokens("Atlas atlas"), 3)

    assert hits == index.rank(bm25_tokens("Atlas"), 3)
    assert [cid for cid, _ in hits] == ["a", "b"]
    assert [score for _, score in hits] == pytest.approx([expected_a, expected_b])


async def test_concurrent_lexical_searches_share_one_batched_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(repository, "WRITE_BATCH", 2)
    pages = {None: [("a", "Atlas"), ("b", "Atlas")], "b": [("c", "Kervan")]}
    cursors = []

    async def read(cypher: str, params: dict[str, Any]) -> SimpleNamespace:
        if cypher == q.CHUNK_TEXT_BATCH:
            await asyncio.sleep(0)
            after = params["after"]
            cursors.append(after)
            assert params["limit"] == 2
            return SimpleNamespace(result_set=pages[after])
        assert cypher == q.CHUNKS_BY_IDS
        return SimpleNamespace(
            result_set=[
                [
                    row["id"],
                    "doc",
                    "atlas.txt",
                    "Atlas",
                    row["score"],
                    [],
                    None,
                    None,
                    None,
                    "next",
                    "Neighbour",
                ]
                for row in reversed(params["rows"])
            ]
        )

    connection = Mock(spec=GraphConnection)
    connection.graph.return_value.ro_query = AsyncMock(side_effect=read)
    repo = GraphRepository(connection, "test")

    first, second = await asyncio.gather(
        repo.search_chunks_text("Atlas", 2), repo.search_chunks_text("Atlas", 2)
    )

    assert cursors == [None, "b"]
    assert first == second
    assert [hit.chunk_id for hit in first] == ["a", "b"]
    assert [(hit.next_id, hit.next_text) for hit in first] == [("next", "Neighbour")] * 2
