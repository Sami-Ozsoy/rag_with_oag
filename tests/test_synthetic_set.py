"""Offline checks for the synthetic corpus and evaluation graph scoring."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Never
from unittest.mock import AsyncMock, MagicMock

import pytest
import tiktoken
import tiktoken.load
import yaml
from eval import run_golden
from eval.graph_quality import (
    ExpectedEntity,
    ExpectedGraph,
    FoundEntity,
    FoundRelation,
    score_graph_quality,
)
from eval.synthetic import generate
from redis.exceptions import ResponseError

from rag_with_oag import Ontology


def snapshot(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_generate_tiny_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class UncachedEncoding(Exception):
        pass

    class OfflineEncoding:
        def encode_ordinary(self, text: str) -> range:
            return range(len(text) // 3)

    def block_download(path: str) -> Never:
        raise UncachedEncoding(path)

    # On a clean machine the first tiktoken use would download its BPE table.
    # Use the cached encoding when available. These assertions concern corpus
    # evidence and determinism, so an offline length estimate suffices otherwise.
    monkeypatch.setattr(tiktoken.load, "read_file", block_download)
    try:
        tiktoken.get_encoding("cl100k_base")
    except UncachedEncoding:
        monkeypatch.setattr(tiktoken, "get_encoding", lambda _: OfflineEncoding())

    args = ["--companies", "20", "--contracts", "20", "--questions", "20"]
    first = tmp_path / "first"
    second = tmp_path / "second"
    other_seed = tmp_path / "other_seed"
    generate.main(["--out", str(first), "--seed", "7", *args])
    generate.main(["--out", str(second), "--seed", "7", *args])
    generate.main(["--out", str(other_seed), "--seed", "8", *args])
    original = snapshot(first)
    assert original == snapshot(second)
    assert (first / "expected_graph.json").read_bytes() != (
        other_seed / "expected_graph.json"
    ).read_bytes()
    assert (first / "ontology.yaml").read_bytes() == (
        run_golden.GOLDEN / "ontology.yaml"
    ).read_bytes()

    questions = yaml.safe_load((first / "questions.yaml").read_text(encoding="utf-8"))
    assert list(questions) == ["single_hop", "multi_hop", "deep_hop", "free_text"]
    assert [len(items) for items in questions.values()] == [5, 5, 5, 5]
    for group in questions.values():
        for question in group:
            docs = [(first / "docs" / p).read_text(encoding="utf-8") for p in question["docs"]]
            for alternatives in question["answer"]:
                assert any(run_golden.matches(doc, [alternatives]) for doc in docs)

    graph: ExpectedGraph = json.loads((first / "expected_graph.json").read_text(encoding="utf-8"))
    assert Counter(e["type"] for e in graph["entities"]) == {"Sirket": 20, "Sozlesme": 20}
    parties = Counter(r["target"] for r in graph["relations"] if r["type"] == "TARAFIDIR")
    assert len(parties) == 20 and set(parties.values()) == {2}
    assert any(r["type"] == "ORTAGIDIR" for r in graph["relations"])


@pytest.mark.parametrize("duplicate_key", [None, "1"])
def test_graph_quality(duplicate_key: str | None) -> None:
    expected_entities: list[ExpectedEntity] = [
        {"type": "Sirket", "key": key, "name": name, "props": {"vergi_no": key}}
        for key, name in (
            ("1", "Zirve Gıda A.Ş."),
            ("2", "Toros Kimya A.Ş."),
            ("3", "Ege Enerji A.Ş."),
            ("4", "Atlas İnşaat A.Ş."),
        )
    ]
    expected_entities.append(
        {"type": "Sozlesme", "key": "HS-2024-001", "name": "Hizmet", "props": {}}
    )
    expected: ExpectedGraph = {
        "entities": expected_entities,
        "relations": [
            {"type": "ORTAGIDIR", "source": "1", "target": "2", "props": {"oran": 40}},
            {"type": "ORTAGIDIR", "source": "2", "target": "3", "props": {"oran": 60}},
            {"type": "ORTAGIDIR", "source": "4", "target": "2", "props": {"oran": 20}},
            {
                "type": "TARAFIDIR",
                "source": "1",
                "target": "HS-2024-001",
                "props": {"rol": "musteri"},
            },
            {
                "type": "TARAFIDIR",
                "source": "2",
                "target": "HS-2024-001",
                "props": {"rol": "saglayici"},
            },
        ],
    }
    entities = [
        FoundEntity(10, "Sirket", "1", "Zirve Gıda A.Ş."),
        FoundEntity(20, "Sirket", "2", "Toros Kimya"),
        FoundEntity(30, "Sirket", "3", "Ege Enerji"),
        FoundEntity(40, "Sirket", duplicate_key, "ZİRVE GIDA AŞ"),
        FoundEntity(50, "Sirket", "9", "Zirve Gıda Pazarlama Ltd. Şti."),
        FoundEntity(90, "Sozlesme", "HS-2024-001", "Hizmet"),
    ]
    relations = [
        FoundRelation("ORTAGIDIR", 10, 20, {"oran": 35}),
        FoundRelation("ORTAGIDIR", 30, 20, {"oran": 60}),
        FoundRelation("ORTAGIDIR", 50, 20, {"oran": 40}),
        FoundRelation("TARAFIDIR", 10, 90, {"rol": "musteri"}),
        FoundRelation("TARAFIDIR", 20, 90, {"rol": "Sağlayıcı"}),
    ]
    quality = score_graph_quality(
        expected, entities, relations, Ontology.from_yaml(run_golden.GOLDEN / "ontology.yaml")
    )
    assert quality["entities"]["Sirket"] == {
        "expected": 4,
        "found": 3,
        "missing": 1,
        "extra": 2 if duplicate_key is None else 1,
        "duplicates": 1,
    }
    ownership = quality["relations"]["ORTAGIDIR"]
    assert (ownership["precision"], ownership["recall"], ownership["reversed"]) == (1 / 3, 1 / 3, 1)
    assert ownership["properties"]["oran"] == {"correct": 0, "matched": 1, "accuracy": 0}
    parties = quality["relations"]["TARAFIDIR"]
    assert parties["properties"]["rol"] == {"correct": 1, "matched": 2, "accuracy": 0.5}


@pytest.mark.parametrize("graph_exists", [True, False])
async def test_fresh_drops_before_schema_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, graph_exists: bool
) -> None:
    db = tmp_path / "golden.db"
    connection = MagicMock(open=AsyncMock(), close=AsyncMock())
    repository = MagicMock(drop=AsyncMock(), init=AsyncMock())
    if not graph_exists:
        repository.drop.side_effect = ResponseError("Invalid graph operation on empty key")
    connection_type = MagicMock(return_value=connection)
    repository_type = MagicMock(return_value=repository)
    monkeypatch.setattr(run_golden, "GraphConnection", connection_type)
    monkeypatch.setattr(run_golden, "GraphRepository", repository_type)

    await run_golden.drop_graph(db, "previous_schema")

    connection_type.assert_called_once_with(path=db)
    repository_type.assert_called_once_with(connection, "previous_schema")
    connection.open.assert_awaited_once()
    repository.drop.assert_awaited_once()
    repository.init.assert_not_awaited()
    connection.close.assert_awaited_once()
