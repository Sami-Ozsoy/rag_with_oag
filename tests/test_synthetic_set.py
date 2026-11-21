"""Offline checks for the synthetic corpus and evaluation graph scoring."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

import pytest
import tiktoken
import yaml
from eval import run_golden
from eval.graph_quality import (
    ExpectedEntity,
    ExpectedGraph,
    FoundEntity,
    FoundRelation,
    score_graph_quality,
)
from eval.synthetic import generate, templates, templates_b
from redis.exceptions import ResponseError

from rag_with_oag import EmbedConfig, Ontology, OntologyMismatchError
from rag_with_oag.store.repository import GraphRepository
from rag_with_oag.store.schema import Meta
from rag_with_oag.types import IngestReport


def snapshot(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def offline_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    class OfflineEncoding:
        def encode_ordinary(self, text: str) -> range:
            return range(len(text) // 3)

    # Use the same token estimate for the stored snapshot on every machine.
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _: OfflineEncoding())


@pytest.mark.parametrize(("variant", "seed"), [("a", 7), ("b", 101)])
def test_generate_tiny_set(tmp_path: Path, offline_encoding: None, variant: str, seed: int) -> None:
    args = ["--companies", "20", "--contracts", "20", "--questions", "20"]
    first = tmp_path / "first"
    second = tmp_path / "second"
    other_seed = tmp_path / "other_seed"
    choice = ["--variant", variant]
    generate.main(
        ["--out", str(first), "--seed", str(seed), *([] if variant == "a" else choice), *args]
    )
    generate.main(["--out", str(second), "--seed", str(seed), *choice, *args])
    generate.main(["--out", str(other_seed), "--seed", str(seed + 1), *choice, *args])
    original = snapshot(first)
    assert original == snapshot(second)
    if variant == "a":
        # Path-to-SHA-256 manifest from the original generator with this fixture.
        hashes = {path.as_posix(): sha256(data).hexdigest() for path, data in original.items()}
        assert sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() == (
            "677f73bf2547b5a96f169d13b9a59374704bb90e0c044395b9cb9d803bbb9535"
        )
    assert (first / "expected_graph.json").read_bytes() != (
        other_seed / "expected_graph.json"
    ).read_bytes()
    assert (first / "ontology.yaml").read_bytes() == (
        run_golden.GOLDEN / "ontology.yaml"
    ).read_bytes()

    questions = yaml.safe_load((first / "questions.yaml").read_text(encoding="utf-8"))
    graph: ExpectedGraph = json.loads((first / "expected_graph.json").read_text(encoding="utf-8"))
    entity_props = {entity["key"]: entity["props"] for entity in graph["entities"]}
    texts = {doc.name: doc.read_text(encoding="utf-8") for doc in (first / "docs").iterdir()}
    assert list(questions) == ["single_hop", "multi_hop", "deep_hop", "free_text"]
    assert [len(items) for items in questions.values()] == [5, 5, 5, 5]
    for name, group in questions.items():
        for question in group:
            docs = [texts[p] for p in question["docs"]]
            for alternatives in question["answer"]:
                assert any(run_golden.matches(doc, [alternatives]) for doc in docs)
            assert len(question["chain"]) == len(question["answer"])
            for path in question["chain"]:
                assert len(path) == {"multi_hop": 2, "deep_hop": 3}.get(name, 1)
                for step in path:
                    evidence = step["evidence"]
                    filenames = [source["doc"] for source in evidence]
                    assert filenames == sorted(filenames)
                    assert set(filenames).intersection(question["docs"])
                    for source in evidence:
                        assert source["literals"]
                        assert all(
                            literal in texts[source["doc"]] for literal in source["literals"]
                        )
                    match step["kind"]:
                        case "relation":
                            assert {
                                k: v for k, v in step.items() if k not in {"kind", "evidence"}
                            } in (graph["relations"])
                        case "property":
                            assert entity_props[step["entity"]][step["prop"]] == step["value"]
                        case "text":
                            assert name == "free_text"
                            assert step["field"] not in entity_props[step["entity"]]
                        case kind:
                            pytest.fail(f"unexpected chain step kind: {kind}")

    assert Counter(e["type"] for e in graph["entities"]) == {"Sirket": 20, "Sozlesme": 20}
    parties = Counter(r["target"] for r in graph["relations"] if r["type"] == "TARAFIDIR")
    assert len(parties) == 20 and set(parties.values()) == {2}
    assert any(r["type"] == "ORTAGIDIR" for r in graph["relations"])


def test_variant_fact_templates_are_disjoint() -> None:
    def sentences(phrases: templates.TemplateSet) -> set[str]:
        fact_templates = (
            phrases.contract_title,
            *phrases.party_paragraphs,
            *phrases.registry_records,
            phrases.registry_name,
            *phrases.ownership_passages,
            phrases.ownership_list,
            phrases.ownership_entry,
            *(statement for clause in phrases.clauses.values() for statement in clause.statements),
        )
        return {
            re.sub(r"\{\w+\}", "{}", sentence).strip().rstrip(".!?")
            for template in fact_templates
            for sentence in re.split(r"[.!?]\s+|\n+", template)
            if sentence.strip()
        }

    assert templates.A.clauses.keys() == templates_b.B.clauses.keys()
    assert templates.A.questions.keys() == templates_b.B.questions.keys()
    assert len(templates_b.B.boilerplate) >= 25
    assert sentences(templates.A).isdisjoint(sentences(templates_b.B))


def test_generation_rejects_chain_without_evidence_in_question_docs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline_encoding: None
) -> None:
    original_chain_step = generate.chain_step

    def unsupported_step(
        fact: generate.Fact,
        shares: dict[tuple[str, str], int],
        clause_fields: set[str],
        sources: list[generate.SourceEvidence],
    ) -> dict[str, object]:
        return {**original_chain_step(fact, shares, clause_fields, sources), "evidence": []}

    monkeypatch.setattr(generate, "chain_step", unsupported_step)
    with pytest.raises(ValueError, match="chain step without evidence in question docs"):
        generate.generate(tmp_path, 7, 20, 20, 20)


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


@pytest.mark.parametrize("matches", [True, False])
async def test_read_seed_cache_checks_source_ontology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, matches: bool
) -> None:
    ontology = Ontology.from_yaml(run_golden.GOLDEN / "ontology.yaml")
    fingerprint = ontology.fingerprint if matches else "different"
    meta = Meta("source", 1, "source-embedder", None, fingerprint, ontology.name)
    connection = MagicMock(open=AsyncMock(), close=AsyncMock())
    connection.graph.return_value.ro_query = AsyncMock(return_value=MagicMock(result_set=[]))
    entries = [["key1", "data1"], ["key2", "data2"]]
    query = AsyncMock(return_value=entries)
    monkeypatch.setattr(run_golden, "GraphConnection", MagicMock(return_value=connection))
    monkeypatch.setattr(run_golden, "read_meta", AsyncMock(return_value=meta))
    monkeypatch.setattr(GraphRepository, "query", query)
    monkeypatch.setattr(
        "rag_with_oag.store.schema.ensure_base_schema", AsyncMock(return_value=meta)
    )

    db = tmp_path / "golden.db"
    if matches:
        assert await run_golden.read_seed_cache(db, "source", ontology) == dict(entries)
        query.assert_awaited_once_with("MATCH (x:ExtractionCache) RETURN x.key, x.data")
    else:
        with pytest.raises(OntologyMismatchError, match=r"knowledge base 'source'.*different"):
            await run_golden.read_seed_cache(db, "source", ontology)
        query.assert_not_awaited()
    connection.close.assert_awaited_once()


async def test_fresh_snapshots_source_and_seeds_target_before_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "ontology.yaml").write_bytes((run_golden.GOLDEN / "ontology.yaml").read_bytes())
    (tmp_path / "questions.yaml").write_text("{}", encoding="utf-8")
    args = argparse.Namespace(
        set=tmp_path,
        base_url=None,
        llm="fake-llm",
        extractor=None,
        out=tmp_path,
        kb="target",
        seed_cache_from="target",
        fresh=True,
        concurrency=2,
        modes="rag",
        equal_budget=False,
        lexical=False,
        rerank=None,
    )
    entries = {"key1": "data1", "key2": "data2"}
    steps = MagicMock(
        read_cache=AsyncMock(return_value=entries),
        drop=AsyncMock(),
        open=AsyncMock(),
        seed=AsyncMock(),
        ingest=AsyncMock(return_value=IngestReport()),
        stats=AsyncMock(return_value={}),
        close=AsyncMock(),
    )
    kb = MagicMock(
        open=steps.open,
        store=MagicMock(cache_extractions=steps.seed),
        ingest=steps.ingest,
        stats=steps.stats,
        close=steps.close,
    )
    monkeypatch.setattr(run_golden, "read_seed_cache", steps.read_cache)
    monkeypatch.setattr(run_golden, "drop_graph", steps.drop)
    monkeypatch.setattr(run_golden, "AsyncKnowledgeBase", MagicMock(return_value=kb))
    monkeypatch.setattr(run_golden, "make_embedder", lambda _: EmbedConfig(model="fake"))

    await run_golden.run(args)

    db = tmp_path / "golden.db"
    assert steps.mock_calls == [
        call.read_cache(db, "target", Ontology.from_yaml(tmp_path / "ontology.yaml")),
        call.drop(db, "target"),
        call.open(),
        call.seed(entries),
        call.ingest(tmp_path / "docs", concurrency=2),
        call.stats(),
        call.close(),
    ]
    assert "Copied 2 extraction cache entries from 'target'" in capsys.readouterr().out
