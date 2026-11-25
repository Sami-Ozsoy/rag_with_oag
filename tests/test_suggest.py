"""Ontology suggestions and the CLI, using queued replies from an offline model."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from rag_with_oag import Document, LLMConfig, Ontology, cli
from rag_with_oag._tokens import count_tokens
from rag_with_oag.errors import OntologyError, StructuredOutputError
from rag_with_oag.llm import capabilities
from rag_with_oag.llm.client import Message
from rag_with_oag.ontology import suggest

from .conftest import FakeLLM


def _property(name: str, type_: str = "string", **fields: Any) -> dict[str, Any]:
    return {
        "name": name,
        "type": type_,
        "description": "",
        "required": False,
        "values": None,
    } | fields


def _reply() -> dict[str, Any]:
    return {
        "entities": [
            {
                "name": "Company",
                "description": "Sözleşmenin tarafı olan şirket",
                "key": "tax_number",
                "properties": [_property("tax_number", required=True)],
            },
            {"name": "Contract", "description": "Sözleşme", "key": None, "properties": []},
        ],
        "relations": [
            {
                "name": "SIGNS",
                "source": "Company",
                "target": "Contract",
                "description": "Şirket sözleşmeyi imzalar",
                "properties": [_property("signed_on", "date")],
            },
        ],
        "name_suffixes": ["s c"],
    }


class QueuedLLM(FakeLLM):
    def __init__(self, *replies: dict[str, Any]) -> None:
        super().__init__()
        self.model = "fake-suggestion"
        self.replies = deque(replies or [_reply()])
        self.closed = False
        self.detected_methods: list[capabilities.StructuredMethod | None] = []

    async def complete(self, messages: Sequence[Message], **kwargs: Any) -> str:
        self.detected_methods.append(capabilities.detected(self))
        self.reply = json.dumps(self.replies.popleft(), ensure_ascii=False)
        return await super().complete(messages, **kwargs)

    async def aclose(self) -> None:
        self.closed = True


DOC = Document.from_text("Acme imzaladı. Vergi numarası 123.", source="example.txt")


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_download(*args: Any, **kwargs: Any) -> bytes:
        raise OSError("tokenizer downloads are disabled in this test")

    monkeypatch.setattr("tiktoken.load.read_file", no_download)


@pytest.fixture
def fresh_capabilities() -> Iterator[None]:
    capabilities.forget()
    yield
    capabilities.forget()


async def test_sample_is_evenly_spaced_and_within_budget() -> None:
    documents = [
        Document.from_text(f"Start of {i}." + " text" * 900 + " END", source=f"{i:02}.txt")
        for i in range(11)
    ]
    llm = QueuedLLM(_reply(), _reply())
    await Ontology.asuggest(reversed(documents), llm, sample_tokens=650)
    await Ontology.asuggest(documents, llm, sample_tokens=650)
    assert llm.requests[0] == llm.requests[1]

    user = llm.requests[0][1]["content"]
    blocks = user.removeprefix("Sample documents:\n\n").split("\n\n")
    assert len(blocks) == 3
    lengths = []
    for label, i, block in zip(range(1, 4), (0, 5, 10), blocks, strict=True):
        heading, excerpt = block.split("\n", 1)
        assert heading == f"### D{label}: {i:02}.txt"
        assert excerpt.startswith(f"Start of {i}.")
        assert "END" not in excerpt
        lengths.append(count_tokens(excerpt))
    assert max(lengths) <= 200
    assert sum(lengths) <= 650


async def test_small_sample_includes_all_nonempty_documents(
    caplog: pytest.LogCaptureFixture,
) -> None:
    documents = [
        Document(text="Last sample", source="docs/z.txt", title=""),
        Document.from_text(" \n\t", source="docs/a.txt", title="Empty"),
        Document.from_text("First sample", source="docs/b.txt", title="Report"),
    ]
    llm = QueuedLLM()
    with caplog.at_level("INFO", logger=suggest.__name__):
        await Ontology.asuggest(documents, llm, sample_tokens=500)
    user = llm.requests[0][1]["content"]
    assert user.endswith("### D1: Report\nFirst sample\n\n### D2: z.txt\nLast sample")
    assert "Empty" not in user
    assert "Sampled 2 of 3 documents" in caplog.text


async def test_sample_caps_an_oversized_word() -> None:
    # A mix of character densities can make a span estimated by length too long.
    doc = Document.from_text("🦊" * 400 + "a" * 5000, source="word.txt")
    llm = QueuedLLM()
    await Ontology.asuggest(doc, llm, sample_tokens=500)
    excerpt = llm.requests[0][1]["content"].split("### D1: word.txt\n", 1)[1]
    assert doc.text.startswith(excerpt)
    assert 0 < count_tokens(excerpt) <= 500


def test_valid_suggestion_and_yaml_round_trip() -> None:
    llm = QueuedLLM()
    ontology = Ontology.suggest(DOC, llm, name="contracts", instructions="Focus on signatories.")
    assert ontology.name == "contracts"
    assert ontology.version == "1"
    assert ontology.entity_names == ["Company", "Contract"]
    assert ontology.entities[0].key == "tax_number"
    assert ontology.entities[0].properties[0].required
    assert ontology.relation_names == ["SIGNS"]
    relation = ontology.relations[0]
    assert (relation.source, relation.target) == ("Company", "Contract")
    assert relation.properties[0].type == "date"
    assert ontology.name_suffixes == ("s c",)
    assert Ontology.from_yaml(ontology.to_yaml()) == ontology
    assert "Focus on signatories." in llm.requests[0][1]["content"]
    assert "HAS_CHUNK" in llm.requests[0][0]["content"]
    assert not llm.closed
    assert len(llm.requests) == 1


def _invalid_reply(problem: str) -> dict[str, Any]:
    reply = _reply()
    if problem == "ontology":
        reply["entities"][0]["name"] = "Şirket"
        reply["relations"][0]["source"] = "Şirket"
        reply["relations"][0]["target"] = "Missing"
    else:
        reply["entities"][0]["properties"][0]["type"] = "text"
    return reply


@pytest.mark.usefixtures("fresh_capabilities")
@pytest.mark.parametrize(
    ("problem", "errors"),
    [
        ("ontology", ["names must be ASCII identifiers", "unknown entity type 'Missing'"]),
        ("structure", ["entities.0.properties.0.type", "Input should be"]),
    ],
)
async def test_invalid_reply_is_corrected(problem: str, errors: list[str]) -> None:
    bad = _invalid_reply(problem)
    llm = QueuedLLM(bad, _reply())
    ontology = await Ontology.asuggest(DOC, llm)
    assert ontology.name == "ontology"
    assert ontology.entity_names == ["Company", "Contract"]
    first, second = llm.requests
    assert second[:2] == first
    assert json.loads(second[2]["content"]) == bad
    for error in errors:
        assert error in second[3]["content"]
    if problem == "structure":
        assert llm.detected_methods == [None, None]
        assert capabilities.detected(llm) == capabilities.StructuredMethod.JSON_SCHEMA


async def test_invalid_twice_raises_ontology_error() -> None:
    bad = _invalid_reply("ontology")
    llm = QueuedLLM(bad, bad)
    with pytest.raises(OntologyError, match="suggested ontology is invalid after one correction"):
        await Ontology.asuggest(DOC, llm)
    assert len(llm.requests) == 2


@pytest.mark.usefixtures("fresh_capabilities")
async def test_schema_failures_do_not_cache_a_structured_method() -> None:
    llm = FakeLLM(reply=json.dumps(_invalid_reply("structure")))
    with pytest.raises(StructuredOutputError, match="no structured output"):
        await Ontology.asuggest(DOC, llm)
    assert capabilities.detected(llm) is None


async def test_extending_preserves_base_and_adds_types_and_properties() -> None:
    base = await Ontology.asuggest(DOC, QueuedLLM(), name="contracts")
    base = Ontology.model_validate({**base.model_dump(), "version": "2"})
    existing_entity = _reply()["entities"][0]
    existing_entity.update(
        description="Changed description",
        key="registration",
        properties=[
            _property("tax_number", "integer"),
            _property("registration"),
        ],
    )
    existing_relation = _reply()["relations"][0]
    existing_relation.update(
        source="Contract",
        target="Company",
        description="Changed direction",
        properties=[_property("signed_on"), _property("place")],
    )
    reply = {
        "entities": [
            {"name": "Region", "description": "", "key": None, "properties": []},
            existing_entity,
        ],
        "relations": [
            {
                "name": "COVERS",
                "source": "Contract",
                "target": "Region",
                "description": "Sözleşme bölgeyi kapsar",
                "properties": [],
            },
            existing_relation,
        ],
        "name_suffixes": ["s c", "s p", "s p"],
    }
    llm = QueuedLLM(reply)
    terms = ["region", ("region", 3), ("coverage", 2), *(f"rare{i:02}" for i in range(50))]
    extended = await Ontology.asuggest([], llm, base=base, unmapped=terms)
    assert (extended.name, extended.version) == ("contracts", "2")
    assert extended.entity_names == ["Company", "Contract", "Region"]
    assert extended.relation_names == ["SIGNS", "COVERS"]
    assert extended.entities[1] == base.entities[1]
    company = extended.entities[0]
    assert [p.name for p in company.properties] == ["tax_number", "registration"]
    assert company.model_copy(update={"properties": company.properties[:1]}) == base.entities[0]
    signs = extended.relations[0]
    assert [p.name for p in signs.properties] == ["signed_on", "place"]
    assert signs.model_copy(update={"properties": signs.properties[:1]}) == base.relations[0]
    assert (extended.relations[1].source, extended.relations[1].target) == ("Contract", "Region")
    assert extended.name_suffixes == ("s c", "s p")
    assert extended.fingerprint != base.fingerprint
    assert len(llm.requests) == 1
    user = llm.requests[0][1]["content"]
    assert base.describe() in user
    assert user.count("region (") == 1
    assert "- region (4)\n- coverage (2)\n" in user
    assert "- rare47 (1)" in user
    assert "- rare48 (1)" not in user


async def test_input_validation_and_sync_event_loop_guard() -> None:
    llm = QueuedLLM()
    with pytest.raises(ValueError, match="documents or unmapped concepts"):
        await Ontology.asuggest([], llm)
    with pytest.raises(ValueError, match="without a base ontology needs documents"):
        await Ontology.asuggest([], llm, unmapped=["coverage"])
    with pytest.raises(ValueError, match="sample_tokens must be at least 500"):
        await Ontology.asuggest(DOC, llm, sample_tokens=499)
    with pytest.raises(RuntimeError, match=r"use await Ontology\.asuggest"):
        Ontology.suggest(DOC, llm)
    assert not llm.requests


@pytest.mark.parametrize("invalid", [False, True])
async def test_llm_config_closes_its_client(monkeypatch: pytest.MonkeyPatch, invalid: bool) -> None:
    config = LLMConfig(model="fake")
    bad = _invalid_reply("ontology")
    llm = QueuedLLM(bad, bad) if invalid else QueuedLLM()

    def client(received: LLMConfig) -> QueuedLLM:
        assert received is config
        return llm

    monkeypatch.setattr(suggest, "LLMClient", client)
    if invalid:
        with pytest.raises(OntologyError):
            await Ontology.asuggest(DOC, config)
    else:
        await Ontology.asuggest(DOC, config)
    assert llm.closed


@pytest.fixture
def cli_llm(monkeypatch: pytest.MonkeyPatch) -> QueuedLLM:
    llm = QueuedLLM()
    monkeypatch.setenv("RAG_OAG_LLM_MODEL", "fake")
    monkeypatch.delenv("RAG_OAG_EMBED_MODEL", raising=False)
    monkeypatch.setattr(cli, "_make_llm", lambda config: llm)
    monkeypatch.setattr(cli, "_open", lambda args: pytest.fail("suggest opened a knowledge base"))
    return llm


HEADER = "# Suggested by rag_with_oag from 3 documents. Review before use.\n"


def test_cli_writes_yaml(
    cli_llm: QueuedLLM, corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    nested = corpus / "nested"
    nested.mkdir()
    (nested / "blank.txt").write_text(" \n", encoding="utf-8")
    output = tmp_path / "ontology.yaml"
    assert cli.main(["suggest", str(corpus), "-o", str(output)]) == 0
    assert output.read_text(encoding="utf-8").startswith(
        "# Suggested by rag_with_oag from 4 documents. Review before use.\n"
    )
    assert Ontology.from_yaml(output).entity_names == ["Company", "Contract"]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "entity types=2, relation types=1" in captured.err
    assert len(cli_llm.requests) == 1
    assert cli_llm.closed


def test_cli_requires_force_for_existing_output(
    cli_llm: QueuedLLM, corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "ontology.yaml"
    output.write_text("existing", encoding="utf-8")
    assert cli.main(["suggest", str(corpus), "-o", str(output)]) == 2
    assert "output file already exists" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "existing"
    assert not cli_llm.requests


def test_cli_extends_base_and_overwrites_with_force(
    cli_llm: QueuedLLM, corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("name: initial\nentities:\n  Company: {}\n", encoding="utf-8")
    output = tmp_path / "ontology.yaml"
    output.write_text("existing", encoding="utf-8")
    assert (
        cli.main(
            [
                "suggest",
                str(corpus),
                "-o",
                str(output),
                "--force",
                "--base",
                str(base),
                "--name",
                "new",
            ]
        )
        == 0
    )
    ontology = Ontology.from_yaml(output)
    assert ontology.name == "new"
    assert ontology.entity_names == ["Company", "Contract"]
    assert ontology.entities[0].key is None
    assert "added entity types=1, relation types=1" in capsys.readouterr().err
    assert "Existing ontology:" in cli_llm.requests[0][1]["content"]


def test_cli_prints_yaml_to_stdout(
    cli_llm: QueuedLLM, corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["suggest", str(corpus)]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith(HEADER)
    assert Ontology.from_yaml(captured.out).relation_names == ["SIGNS"]
    assert captured.err == ""
    assert len(cli_llm.requests) == 1


def test_cli_requires_llm_model(
    cli_llm: QueuedLLM,
    corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("RAG_OAG_LLM_MODEL")
    assert cli.main(["suggest", str(corpus)]) == 2
    assert (
        capsys.readouterr().err == "error: missing or invalid model settings: RAG_OAG_LLM_MODEL\n"
    )
    assert not cli_llm.requests


def test_cli_reports_loading_errors(
    cli_llm: QueuedLLM, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["suggest", str(tmp_path / "missing")]) == 1
    assert "error: no such file or directory" in capsys.readouterr().err
    assert not cli_llm.requests
