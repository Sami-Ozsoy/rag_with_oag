"""Chain coverage checks over hand-built retrieval contexts."""

from __future__ import annotations

from dataclasses import replace

import pytest
from eval.chain import Chain, chain_coverage

from rag_with_oag.types import ChunkHit, Citation, Fact, RetrievedContext

NODE_KEYS = {"company": "100001", "parent": "100002", "contract": "DS-2023-008"}
PARTY = {
    "kind": "relation",
    "type": "TARAFIDIR",
    "source": "100001",
    "target": "DS-2023-008",
    "props": {"rol": "musteri"},
    "evidence": [{"doc": "contract.md", "literals": ["DS-2023-008", "Ayaz", "müşteri"]}],
}
CITY = {
    "kind": "property",
    "entity": "100001",
    "prop": "sehir",
    "value": "İzmir",
    "evidence": [{"doc": "registry.md", "literals": ["Ayaz", "İzmir"]}],
}
CLAUSE = {
    "kind": "text",
    "entity": "DS-2023-008",
    "field": "bedel",
    "value": "1234",
    "evidence": [{"doc": "contract.md", "literals": ["1.234 TL"]}],
}
PARTY_FACT = Fact(
    "Ayaz",
    "TARAFIDIR",
    "DS-2023-008",
    {"rol": "musteri"},
    subject_id="company",
    object_id="contract",
)
CITY_FACT = Fact("Ayaz", "is_a", "Sirket", {"sehir": "IZMIR"}, kind="entity", subject_id="company")


def context(*facts: Fact, packed: str = "", chunks: tuple[ChunkHit, ...] = ()) -> RetrievedContext:
    citations = [Citation(f"C{i}", "chunk", hit.chunk_id) for i, hit in enumerate(chunks, 1)]
    return RetrievedContext("question", "hybrid", list(chunks), list(facts), citations, packed, 0)


def test_graph_only_and_missing_hop() -> None:
    chain: Chain = [[PARTY, CITY]]
    assert chain_coverage(context(PARTY_FACT, CITY_FACT), chain, NODE_KEYS) == (2, 2)
    unrelated_city = replace(CITY_FACT, subject_id="parent")
    assert chain_coverage(context(PARTY_FACT, unrelated_city), chain, NODE_KEYS) == (1, 2)


@pytest.mark.parametrize(
    ("fact", "covered"),
    [
        (replace(PARTY_FACT, subject_id="contract", object_id="company"), 0),
        (replace(PARTY_FACT, properties={"rol": "saglayici"}), 0),
        (replace(PARTY_FACT, properties={"rol": "MUSTERI"}), 0),
        (replace(PARTY_FACT, properties={}), 1),
    ],
    ids=["wrong-direction", "wrong-role", "role-case", "missing-role"],
)
def test_graph_relation_matching(fact: Fact, covered: int) -> None:
    assert chain_coverage(context(fact), [[PARTY]], NODE_KEYS) == (covered, 1)


@pytest.mark.parametrize(("percent", "covered"), [(73.5, 1), (74.51, 0), ("74", 0)])
def test_ownership_tolerance(percent: float | str, covered: int) -> None:
    step = {
        "kind": "relation",
        "type": "ORTAGIDIR",
        "source": "100002",
        "target": "100001",
        "props": {"oran": 74},
        "evidence": [],
    }
    fact = Fact(
        "Parent",
        "ORTAGIDIR",
        "Ayaz",
        {"oran": percent},
        subject_id="parent",
        object_id="company",
    )
    assert chain_coverage(context(fact), [[step]], NODE_KEYS) == (covered, 1)


@pytest.mark.parametrize(("value", "covered"), [(1.0, 1), (True, 0), ("1", 0)])
def test_numeric_property(value: object, covered: int) -> None:
    step = {**CITY, "prop": "number", "value": 1}
    fact = replace(CITY_FACT, properties={"number": value})
    assert chain_coverage(context(fact), [[step]], NODE_KEYS) == (covered, 1)


def test_text_only_including_doc_id_fallback() -> None:
    packed = (
        "Source texts:\n\n[C1] /corpus/contract.md\nDS-2023-008: AYAZ, MÜŞTERİ. Bedel 1.234\nTL."
        "\n\n[C2] Legacy source name\nAYAZ için merkez ili IZMIR."
    )
    chunks = (
        ChunkHit("id1", "hash", "/corpus/contract.md", "unused", 1),
        ChunkHit("id2", "registry.md", "Legacy source name", "unused", 1),
    )
    assert chain_coverage(context(packed=packed, chunks=chunks), [[PARTY, CITY], [CLAUSE]], {}) == (
        3,
        3,
    )
    clause_fact = replace(CITY_FACT, subject_id="contract", properties={"bedel": "1234"})
    assert chain_coverage(context(clause_fact), [[CLAUSE]], NODE_KEYS) == (0, 1)


def test_literals_split_across_blocks() -> None:
    packed = "Source texts:\n\n[C1] registry.md\nAyaz\n\n[C2] registry.md\nİzmir"
    chunks = (
        ChunkHit("id1", "hash", "registry.md", "Ayaz", 1),
        ChunkHit("id2", "hash", "registry.md", "İzmir", 1),
    )
    assert chain_coverage(context(packed=packed, chunks=chunks), [[CITY]], {}) == (0, 1)


def test_neighbour_text_only_counts_when_packed() -> None:
    hit = ChunkHit("id1", "hash", "registry.md", "Ayaz", 1, next_id="id2", next_text="İzmir")
    packed = "Source texts:\n\n[C1] registry.md\nAyaz"
    assert chain_coverage(context(packed=packed, chunks=(hit,)), [[CITY]], {}) == (0, 1)
    assert chain_coverage(context(packed=packed + "\nİzmir", chunks=(hit,)), [[CITY]], {}) == (1, 1)


def test_text_from_wrong_document() -> None:
    hit = ChunkHit("id1", "hash", "/corpus/other.md", "Ayaz, İzmir", 1)
    packed = "Source texts:\n\n[C1] /corpus/other.md\nAyaz, İzmir"
    assert chain_coverage(context(packed=packed, chunks=(hit,)), [[CITY]], {}) == (0, 1)
