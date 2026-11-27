"""Offline extraction reports and CLI behavior."""

from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from rag_with_oag import (
    Document,
    EntityTypeStats,
    ExtractionConfig,
    LLMConfig,
    ModelAuthError,
    ModelConnectionError,
    ModelNotFoundError,
    Ontology,
    PreviewReport,
    RelationTypeStats,
    RetrievalConfig,
    cli,
)
from rag_with_oag.ingest.chunker import AdaptiveChunker, Span
from rag_with_oag.ingest.extractor import Extractor, validate
from rag_with_oag.ingest.pipeline import build_chunks
from rag_with_oag.llm.client import Message
from rag_with_oag.ontology import preview
from rag_with_oag.ontology.schema_gen import RawExtraction

from .conftest import FakeLLM

DOC = Document.from_text("Acme signs C-1.", source="example.txt")
EXTRACTION = ExtractionConfig(batch_size=3, validation_retries=0, structured_mode="json_schema")


class QueuedLLM(FakeLLM):
    def __init__(self, *replies: dict[str, Any] | Exception) -> None:
        super().__init__()
        self.model = "fake-preview"
        self.replies = deque(replies)
        self.closed = False
        self.running = 0
        self.peak = 0

    async def complete(self, messages: Sequence[Message], **kwargs: Any) -> str:
        reply = self.replies.popleft() if self.replies else {}
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            await asyncio.sleep(0)
            if isinstance(reply, Exception):
                raise reply
            self.reply = json.dumps(reply)
            return await super().complete(messages, **kwargs)
        finally:
            self.running -= 1

    async def aclose(self) -> None:
        assert self.running == 0
        self.closed = True


class LineChunker:
    def split(self, text: str) -> list[Span]:
        return [Span(m.start(), m.end(), m[0], (), "line") for m in re.finditer(r"[^\n]+", text)]


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_download(*args: Any, **kwargs: Any) -> bytes:
        raise OSError("tokenizer downloads are disabled in this test")

    monkeypatch.setattr("tiktoken.load.read_file", no_download)


@pytest.fixture
def schema() -> Ontology:
    return Ontology.from_yaml("""
name: contracts
entities:
  Company:
    key: code
    properties: {code: integer, city: string, active: boolean}
  Contract:
    properties: {number: {type: string, required: true}}
  Person:
    key: badge
    properties: {badge: string}
relations:
  SIGNS:
    source: Company
    target: Contract
    properties: {role: string, signed_on: date}
  KNOWS: {source: Person, target: Person}
""")


def _entity(ref: str, name: str, type_name: str = "Company", **fields: Any) -> dict[str, Any]:
    return {"ref": ref, "name": name, "type": type_name, "chunks": ["K1"]} | fields


def _relation(source: str, target: str, type_name: str = "SIGNS", **fields: Any) -> dict[str, Any]:
    return {"type": type_name, "source": source, "target": target} | fields


def _contract(ref: str = "c1", **fields: Any) -> dict[str, Any]:
    return _entity(ref, "C-1", "Contract", properties={"number": "C-1"}) | fields


def _invalid_reply() -> dict[str, Any]:
    return {
        "entities": [
            _entity("e1", "Acme"),
            _contract(),
            _entity("unknown", "Depot", "Facility"),
            _entity("blank", " "),
            _entity("e1", "Duplicate"),
            _entity("bad", "Bad", properties={"code": "bad"}, chunks=["K9"]),
            _contract("missing", properties={}, chunks=[]),
        ],
        "relations": [
            _relation("e1", "c1", "OWNS"),
            _relation("c1", "e1"),
            _relation("e1", "unknown"),
            _relation("e1", "c1", chunks=["K8"]),
        ],
    }


def test_validate_issue_kinds_and_unchanged_messages(schema: Ontology) -> None:
    raw = RawExtraction.model_validate(_invalid_reply())
    labels = {"K1": "a", "K2": "b"}
    result = validate(raw, labels, schema, detailed=True)
    assert result.errors == [
        "entity 'unknown' (Facility 'Depot'): unknown entity type 'Facility'; "
        "use one of ['Company', 'Contract', 'Person']",
        "entity 'blank' (Company ' '): name is empty",
        "entity 'e1' (Company 'Duplicate'): ref 'e1' is used twice",
        "entity 'bad' (Company 'Bad'): property 'code': expected a number, got 'bad'",
        "entity 'bad' (Company 'Bad'): unknown passage id 'K9'; use ['K1', 'K2']",
        "entity 'missing' (Contract 'C-1'): required property 'number' is missing",
        "entity 'missing' (Contract 'C-1'): 'chunks' must list the passages that mention it",
        "relation OWNS(e1 -> c1): unknown relation type 'OWNS'; use one of ['SIGNS', 'KNOWS']",
        "relation SIGNS(c1 -> e1): SIGNS connects Company -> Contract, not Contract -> Company",
        "relation SIGNS(e1 -> unknown): source and target must be refs of valid entities",
        "relation SIGNS(e1 -> c1): unknown passage id 'K8'; use ['K1', 'K2']",
    ]
    assert validate(raw, labels, schema) == (result.graphs, result.errors)
    assert [issue.message for issue in result.issues] == result.errors
    assert [(issue.kind, issue.type_name, issue.observed) for issue in result.issues] == [
        ("unknown_entity_type", "Facility", None),
        ("empty_name", "Company", None),
        ("duplicate_ref", "Company", None),
        ("property_error", "Company", None),
        ("bad_reference", "Company", None),
        ("missing_required", "Contract", None),
        ("bad_reference", "Contract", None),
        ("unknown_relation_type", "OWNS", None),
        ("endpoint_mismatch", "SIGNS", ("Contract", "Company")),
        ("bad_reference", "SIGNS", None),
        ("bad_reference", "SIGNS", None),
    ]


async def test_extractor_keeps_final_issues_and_sends_original_feedback(schema: Ontology) -> None:
    first = {"entities": [_entity("e1", "Depot", "Facility")]}
    last = {"entities": [_entity("e1", " ")]}
    llm = QueuedLLM(first, last)
    result = await Extractor(llm, schema, mode="json_schema").extract(
        build_chunks(DOC, LineChunker())
    )
    assert result.errors == ["entity 'e1' (Company ' '): name is empty"]
    assert [issue.kind for issue in result.issues] == ["empty_name"]
    feedback = llm.requests[1][-1]["content"]
    assert "- entity 'e1' (Facility 'Depot'): unknown entity type 'Facility';" in feedback
    assert "['Company', 'Contract', 'Person']" in feedback


async def test_sampling_spreads_docs_and_chunks_and_bounds_batches(schema: Ontology) -> None:
    docs = [
        Document.from_text("\n".join(f"doc{i}-chunk{j}" for j in range(7)), source=f"{i}.txt")
        for i in range(7)
    ]
    docs.append(Document.from_text(" \n", source="blank.txt"))
    config = EXTRACTION.model_copy(update={"batch_size": 2})
    options = dict(
        documents=3, chunks_per_doc=3, concurrency=2, config=config, chunker=LineChunker()
    )
    llm = QueuedLLM()
    report = await preview.preview_ontology(schema, reversed(docs), llm, **options)
    assert (report.documents_sampled, report.documents_available, report.chunks_sampled) == (
        3,
        7,
        9,
    )
    assert report.batches == 6
    assert llm.peak == 2
    expected = [
        [f"doc{i}-chunk0", f"doc{i}-chunk3"] if j == 0 else [f"doc{i}-chunk6"]
        for i in (0, 3, 6)
        for j in range(2)
    ]
    assert [
        re.findall(r"doc\d-chunk\d", request[1]["content"]) for request in llm.requests
    ] == expected

    again = QueuedLLM()
    repeated = await preview.preview_ontology(schema, docs, again, **options)
    assert again.requests == llm.requests
    assert asdict(repeated) == asdict(report)

    llm = QueuedLLM()
    report = await preview.preview_ontology(
        schema, docs[:2], llm, chunker=LineChunker(), chunks_per_doc=1
    )
    assert (report.documents_sampled, report.chunks_sampled) == (2, 2)
    assert [re.findall(r"doc\d-chunk\d", request[1]["content"]) for request in llm.requests] == [
        ["doc0-chunk0"],
        ["doc1-chunk0"],
    ]


@pytest.mark.parametrize(
    "retrieval", [None, RetrievalConfig(chunk_min_tokens=20, chunk_max_tokens=50)]
)
async def test_default_chunker_matches_ingest(
    schema: Ontology, retrieval: RetrievalConfig | None
) -> None:
    doc = Document.from_text(
        "# Section\n\n" + "Some text for the chunker. " * 300, source="sample.md"
    )
    chunks = build_chunks(doc, AdaptiveChunker(retrieval or RetrievalConfig()))
    llm = QueuedLLM()
    report = await preview.preview_ontology(schema, doc, llm, chunks_per_doc=2, retrieval=retrieval)
    assert len(chunks) > report.chunks_sampled == 2
    assert (
        llm.requests[0][1]["content"]
        == f"Passages:\n\n[K1] {chunks[0].text}\n\n[K2] {chunks[-1].text}"
    )


def _sample_reply() -> dict[str, Any]:
    return {
        "entities": [
            _entity("e1", "Acme", properties={"code": 0, "active": False}, chunks=["K1", "K2"]),
            _entity("e2", "Other Name", properties={"code": "0"}),
            _entity("e3", "Acme Ltd."),
            _entity("e4", "ACME"),
            _entity("e5", "Delta", properties={"city": " "}),
            _entity("e6", "Missing", "Facility"),
            _contract(),
        ],
        "relations": [
            _relation("e1", "c1", chunks=["K1", "K2", "K3"], properties={"role": "client"}),
            _relation("e1", "c1", chunks=["K3"], properties={"role": "client"}),
            _relation("e2", "c1"),
            _relation("c1", "e1"),
            _relation("c1", "e2"),
            _relation("e1", "e2"),
            _relation("e2", "e3"),
            _relation("e3", "e4"),
        ],
        "unmapped": [{"term": "warranty", "count": 3}, "deposit"],
    }


async def test_report_counts_groups_and_hints(schema: Ontology) -> None:
    first = Document.from_text("first\nmiddle\nlast", source="a.txt")
    second = Document.from_text("next", source="b.txt")
    llm = QueuedLLM(_sample_reply(), {"entities": [_contract()], "unmapped": ["deposit"]})
    llm.usage["prompt_tokens"] = 400
    report = await preview.preview_ontology(
        schema, [first, second], llm, chunker=LineChunker(), config=EXTRACTION
    )
    assert report.chunks_with_entities == 3  # the third chunk only carries relation endpoints
    assert report.chunks_with_relations == 3
    assert report.token_usage == {"prompt_tokens": 200, "completion_tokens": 20}
    assert report.entity_types == [
        EntityTypeStats(
            "Company",
            5,
            3,
            "code",
            0.4,
            {"code": 0.4, "city": 0.0, "active": 0.2},
            ["Acme", "Other Name", "Acme Ltd."],
        ),
        EntityTypeStats("Contract", 2, 1, None, None, {"number": 1.0}, ["C-1"]),
        EntityTypeStats("Person", 0, 0, "badge", 0.0, {}, []),
    ]
    assert report.relation_types == [
        RelationTypeStats(
            "SIGNS",
            "Company",
            "Contract",
            3,
            {"role": 2 / 3, "signed_on": 0.0},
            ["Acme -[SIGNS]-> C-1", "Other Name -[SIGNS]-> C-1"],
        ),
        RelationTypeStats("KNOWS", "Person", "Person", 0, {}, []),
    ]
    assert report.unused_entity_types == ["Person"]
    assert report.unused_relation_types == ["KNOWS"]
    assert [(d.kind, d.type_name, d.count) for d in report.dropped] == [
        ("endpoint_mismatch", "SIGNS", 5),
        ("unknown_entity_type", "Facility", 1),
    ]
    assert report.dropped[0].observed == {"Contract -> Company": 2, "Company -> Company": 3}
    assert len(report.dropped[0].examples) == 3
    assert not report.dropped[1].observed
    assert report.unmapped_concepts == [("warranty", 3), ("deposit", 2)]
    assert report.hints == [
        "Entity type Person was never extracted from the sample.",
        "Relation type KNOWS was never extracted from the sample.",
        "SIGNS was dropped 5 times because its endpoints did not match: "
        "got Contract -> Company (2), Company -> Company (3); "
        "the ontology allows Company -> Contract.",
        "1 unknown entity type issue for Facility.",
        "Key code of Company is filled in only 40% of mentions; "
        "instances without it are matched by name and may split into separate nodes.",
        "Property city of Company was never filled.",
        "Property signed_on of SIGNS was never filled.",
        "Concepts the ontology cannot express: warranty (3), deposit (2)",
    ]


async def test_failed_batch_does_not_hide_successful_results(schema: Ontology) -> None:
    bad = Document.from_text("bad", source="a.txt")
    llm = QueuedLLM(RuntimeError("batch unavailable"), {"entities": [_contract()]})
    report = await schema.apreview([bad, DOC], llm)
    assert report.batches == 2
    assert report.failed_batches == ["a.txt: chunks 0-0: RuntimeError: batch unavailable"]
    assert report.chunks_with_entities == 1
    assert report.entity_types[1].mentions == 1
    assert report.hints[-1] == "1 of 2 batches failed; the report covers only the rest."


@pytest.mark.parametrize("error", [ModelConnectionError, ModelAuthError, ModelNotFoundError])
async def test_endpoint_errors_propagate(schema: Ontology, error: type[Exception]) -> None:
    with pytest.raises(error, match="unavailable"):
        await schema.apreview(DOC, QueuedLLM(error("unavailable")))


@pytest.mark.parametrize("unavailable", [False, True])
async def test_llm_config_closes_client(
    schema: Ontology, monkeypatch: pytest.MonkeyPatch, unavailable: bool
) -> None:
    config = LLMConfig(model="fake")
    llm = QueuedLLM(ModelAuthError("unavailable")) if unavailable else QueuedLLM()

    def client(received: LLMConfig) -> QueuedLLM:
        assert received is config
        return llm

    monkeypatch.setattr(preview, "LLMClient", client)
    if unavailable:
        with pytest.raises(ModelAuthError, match="unavailable"):
            await schema.apreview(DOC, config)
    else:
        await schema.apreview(DOC, config)
    assert llm.closed


async def test_input_validation_and_event_loop_guard(schema: Ontology) -> None:
    llm = QueuedLLM()
    for arg in ("documents", "chunks_per_doc", "concurrency"):
        with pytest.raises(ValueError, match=rf"{arg} must be >= 1"):
            await schema.apreview(DOC, llm, **{arg: 0})
    with pytest.raises(ValueError, match="documents with text"):
        await schema.apreview(Document.from_text(" \n", source="blank.txt"), llm)
    with pytest.raises(RuntimeError, match=r"use await ontology\.apreview"):
        schema.preview(DOC, llm)
    assert not llm.requests


def test_sync_api_leaves_supplied_model_open(schema: Ontology) -> None:
    llm = QueuedLLM()
    result = schema.preview(DOC, llm)
    assert isinstance(result, PreviewReport)
    assert result.ontology == schema.name
    assert not llm.closed


@pytest.fixture
def cli_args(
    schema: Ontology, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    path = tmp_path / "ontology.yaml"
    signs, knows = schema.relations
    schema.model_copy(
        update={"relations": (signs, knows.model_copy(update={"name": "KNOWS_PERSON"}))}
    ).to_yaml(path)
    monkeypatch.setenv("RAG_OAG_LLM_MODEL", "fake")
    monkeypatch.delenv("RAG_OAG_EMBED_MODEL", raising=False)
    monkeypatch.setattr(cli, "_open", lambda args: pytest.fail("preview opened a knowledge base"))
    return ["preview", str(corpus), "--ontology", str(path)]


@pytest.mark.parametrize("output", ["text", "json"])
def test_cli_output(
    cli_args: list[str],
    output: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reply = {
        "entities": [_entity("e1", "Acme"), _contract()],
        "relations": [_relation("e1", "c1"), _relation("c1", "e1")],
        "unmapped": ["deposit"],
    }
    llm = QueuedLLM(reply, reply)
    monkeypatch.setattr(cli, "_make_llm", lambda config: llm)
    args = [*cli_args, "--docs", "1", "--chunks-per-doc", "1", "--concurrency", "1", "-v"]
    assert cli.main([*args, "--json"] if output == "json" else args) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert llm.closed
    if output == "json":
        report = json.loads(captured.out)
        assert report["ontology"] == "contracts"
        assert report["documents_available"] == 3
        assert report["entity_types"][0]["examples"] == ["Acme"]
        assert report["relation_types"][0]["examples"] == ["Acme -[SIGNS]-> C-1"]
        assert report["unmapped_concepts"] == [["deposit", 1]]
        assert "Concepts the ontology cannot express: deposit (1)" in report["hints"]
    else:
        summary, entities, relations, dropped, hints, tokens = captured.out.removesuffix(
            "\n"
        ).split("\n\n")
        assert summary.splitlines() == [
            "Previewed ontology 'contracts' on 1 of 3 documents (1 chunks, 1 batches).",
            "Coverage: 1/1 chunks with entities, 1/1 with relations.",
        ]
        assert entities.splitlines() == [
            "Entity types:",
            "  Company   1 mentions, 1 distinct, key code 0%   e.g. Acme",
            "  Contract  1 mentions, 1 distinct   e.g. C-1",
            "  Person    0 mentions",
        ]
        assert relations.splitlines() == [
            "Relation types:",
            "  SIGNS         (Company -> Contract)  1   e.g. Acme -[SIGNS]-> C-1",
            "  KNOWS_PERSON  (Person -> Person)     0",
        ]
        assert dropped == "Dropped:\n  endpoint_mismatch SIGNS  1  (Contract -> Company 1)"
        assert hints.startswith("Hints:\n")
        assert "  - Concepts the ontology cannot express: deposit (1)" in hints.splitlines()
        assert captured.out.count("deposit") == 1
        assert tokens == "Tokens: prompt 200, completion 20"


@pytest.mark.parametrize("docs,code", [(1, 1), (2, 0)])
def test_cli_batch_failure_exit_codes(
    cli_args: list[str],
    docs: int,
    code: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    llm = QueuedLLM(RuntimeError("failed"))
    monkeypatch.setattr(cli, "_make_llm", lambda config: llm)
    assert cli.main([*cli_args, "--docs", str(docs), "--chunks-per-doc", "1", "--json"]) == code
    report = json.loads(capsys.readouterr().out)
    assert len(report["failed_batches"]) == 1
    assert report["batches"] == docs
    assert llm.closed


def test_cli_invalid_ontology_and_missing_model(
    cli_args: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli, "_make_llm", lambda config: pytest.fail("invalid setup reached the model")
    )
    monkeypatch.delenv("RAG_OAG_LLM_MODEL")
    assert cli.main(cli_args) == 2
    assert (
        capsys.readouterr().err == "error: missing or invalid model settings: RAG_OAG_LLM_MODEL\n"
    )

    path = Path(cli_args[-1])
    path.write_text("entities: {}", encoding="utf-8")
    assert cli.main(cli_args) == 2
    assert "error: invalid ontology" in capsys.readouterr().err


@pytest.mark.parametrize("flags", [[], ["--ontology", "ontology.yaml", "--docs", "0"]])
def test_cli_argparse_errors(flags: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["preview", "docs/", *flags])
    assert exc.value.code == 2


def test_cli_missing_sources(
    cli_args: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    llm = QueuedLLM()
    monkeypatch.setattr(cli, "_make_llm", lambda config: llm)
    cli_args[1] = str(tmp_path / "missing")
    assert cli.main(cli_args) == 1
    assert "error: no such file or directory" in capsys.readouterr().err
    assert llm.closed
