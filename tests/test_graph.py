"""Ontology extraction, resolution, oag and hybrid retrieval against embedded FalkorDB."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from rag_with_oag import (
    AsyncKnowledgeBase,
    ConfigError,
    Document,
    Ontology,
    OntologyMismatchError,
)
from rag_with_oag.store.connection import GraphConnection

from .conftest import CONTRACTS_ONTOLOGY, FakeEmbedder, FakeGraphLLM

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture(loop_scope="session")
async def graph_kb(
    connection: GraphConnection, kb_name: str, ontology: Ontology
) -> AsyncIterator[AsyncKnowledgeBase]:
    kb = AsyncKnowledgeBase(
        kb_name,
        connection=connection,
        embedder=FakeEmbedder(),
        llm=FakeGraphLLM(),
        ontology=ontology,
    )
    yield kb
    await kb.store.drop()
    await kb.close()


async def test_ingest_builds_the_graph(
    graph_kb: AsyncKnowledgeBase, graph_docs: list[Document]
) -> None:
    report = await graph_kb.ingest(graph_docs)
    assert report.ok, report.failures
    assert report.documents_processed == 3
    assert report.entities_written > 0 and report.relations_written == 4
    assert report.validation_errors == []
    assert report.unmapped_concepts == [("depozito", 1)]
    assert report.token_usage["embedding_tokens"] > 0
    stats = await graph_kb.stats()
    # Bulut appears with a tax number in one document and without it in another: one entity
    assert stats["entities"] == 6 and stats["relations"] == 4

    [[name, vergi, sehir, labels]] = await graph_kb.store.query(
        "MATCH (e:Entity) WHERE e.name_norm = 'bulut yazilim ltd' "
        "RETURN e.name, e.vergi_no, e.sehir, labels(e)"
    )
    assert (name, vergi, sehir) == ("Bulut Yazılım Ltd.", "1234567890", "Ankara")
    assert labels == ["Entity", "Sirket"]

    [[evidence, rol, mentions]] = await graph_kb.store.query(
        "MATCH (s:Sirket)-[r:TARAFIDIR]->(:Sozlesme) WHERE s.name_norm = 'atlas lojistik a s' "
        "OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(s) RETURN r.evidence, r.rol, count(c)"
    )
    assert len(evidence) == 1 and rol == "musteri" and mentions == 1

    [[version]] = await graph_kb.store.query("MATCH (m:Meta) RETURN m.ontology_version")
    assert version == graph_kb.ontology.fingerprint  # type: ignore[union-attr]


async def test_reingest_skips_and_does_not_call_the_model(
    graph_kb: AsyncKnowledgeBase, graph_docs: list[Document]
) -> None:
    await graph_kb.ingest(graph_docs)
    llm: FakeGraphLLM = graph_kb.llm  # type: ignore[assignment]
    calls = llm.extraction_calls
    again = await graph_kb.ingest(graph_docs)
    assert again.documents_skipped == 3 and llm.extraction_calls == calls


async def test_oag_answers_from_facts(
    graph_kb: AsyncKnowledgeBase, graph_docs: list[Document]
) -> None:
    await graph_kb.ingest(graph_docs)
    ctx = await graph_kb.retrieve("Atlas Lojistik'in sözleşmesi hangisi?", mode="oag")
    assert ctx.mode_used == "oag" and not ctx.chunks
    rendered = [f.render() for f in ctx.facts]
    assert rendered[0].startswith("Atlas Lojistik A.Ş. (Sirket)")
    assert any(
        "Atlas Lojistik A.Ş. —TARAFIDIR→ HS-2024-01 Hizmet Sözleşmesi" in r for r in rendered
    )
    assert ctx.prompt_context.startswith("Structured facts:\n[F1] ")
    assert all(c.kind == "fact" for c in ctx.citations)


async def test_hybrid_connects_documents(
    graph_kb: AsyncKnowledgeBase, graph_docs: list[Document]
) -> None:
    await graph_kb.ingest(graph_docs)
    question = "Atlas Lojistik'in hizmet aldığı şirketin ana ortağı kimdir?"

    one = await graph_kb.retrieve(question, hops=1)
    assert one.mode_used == "hybrid"  # the default with an ontology
    # Seeds are Atlas and the contract ("hizmet"); Delta is two hops from the contract.
    assert not any("Delta" in f.render() for f in one.facts)

    three = await graph_kb.retrieve(question)  # default: 2 hops
    facts = [f.render() for f in three.facts]
    assert any("Delta Holding A.Ş. —ORTAGIDIR→ Bulut Yazılım Ltd. (oran: 60.0)" in f for f in facts)
    sources = {h.source for h in three.chunks}
    assert "sicil.md" in sources  # evidence chunk from another document than the question's entity
    assert "Structured facts:" in three.prompt_context and "Source texts:" in three.prompt_context
    kinds = {c.kind for c in three.citations}
    assert kinds == {"fact", "chunk"}

    result = await graph_kb.query(question)
    assert result.mode_used == "hybrid"
    assert [c.id for c in result.citations] == ["F1", "C1"]
    system = graph_kb.llm.requests[-1][0]["content"]  # type: ignore[union-attr]
    assert "structured facts from a" in system


async def test_hybrid_falls_back_to_rag_without_entities(
    graph_kb: AsyncKnowledgeBase, graph_docs: list[Document]
) -> None:
    await graph_kb.ingest(graph_docs)
    ctx = await graph_kb.retrieve("Yarın hava yağmurlu olacak mı?")
    assert ctx.mode_used == "rag" and not ctx.facts
    oag = await graph_kb.retrieve("Yarın hava yağmurlu olacak mı?", mode="oag")
    assert oag.prompt_context == "" and oag.mode_used == "oag"


async def test_changed_ontology_is_refused(
    connection: GraphConnection, graph_kb: AsyncKnowledgeBase, graph_docs: list[Document]
) -> None:
    await graph_kb.ingest(graph_docs[:1])
    changed = Ontology.from_yaml(CONTRACTS_ONTOLOGY.replace("sehir: string", "il: string"))
    other = AsyncKnowledgeBase(
        graph_kb.name,
        connection=connection,
        embedder=FakeEmbedder(),
        llm=FakeGraphLLM(),
        ontology=changed,
    )
    with pytest.raises(OntologyMismatchError, match="new knowledge base name"):
        await other.open()
    # opening without an ontology is fine: rag only
    plain = AsyncKnowledgeBase(graph_kb.name, connection=connection, embedder=FakeEmbedder())
    assert (await plain.retrieve("Atlas")).mode_used == "rag"


async def test_ontology_added_later_extracts_existing_documents(
    connection: GraphConnection, kb_name: str, ontology: Ontology, graph_docs: list[Document]
) -> None:
    plain = AsyncKnowledgeBase(kb_name, connection=connection, embedder=FakeEmbedder())
    first = await plain.ingest(graph_docs)
    assert first.chunks_written > 0 and first.entities_written == 0

    llm = FakeGraphLLM()
    kb = AsyncKnowledgeBase(
        kb_name, connection=connection, embedder=FakeEmbedder(), llm=llm, ontology=ontology
    )
    try:
        report = await kb.ingest(graph_docs)
        assert report.documents_processed == 3 and report.chunks_written == 0
        assert report.relations_written == 4 and llm.extraction_calls == 3
    finally:
        await kb.store.drop()


class _FlakyLLM(FakeGraphLLM):
    """Fails every extraction call while ``down`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.down = True

    async def complete(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> str:
        if self.down and any("Passages:" in m["content"] for m in messages):
            raise ConnectionError("model server unreachable")
        return await super().complete(messages, **kwargs)


async def test_failed_extraction_keeps_chunks_and_is_retried(
    connection: GraphConnection, kb_name: str, ontology: Ontology, graph_docs: list[Document]
) -> None:
    llm = _FlakyLLM()
    kb = AsyncKnowledgeBase(
        kb_name, connection=connection, embedder=FakeEmbedder(), llm=llm, ontology=ontology
    )
    try:
        failed = await kb.ingest(graph_docs)
        assert len(failed.failures) == 3 and "extraction:" in failed.failures[0][1]
        assert failed.chunks_written > 0 and failed.entities_written == 0
        assert (await kb.retrieve("Atlas", mode="rag")).chunks  # rag still works

        llm.down = False
        retried = await kb.ingest(graph_docs)
        assert retried.ok and retried.chunks_written == 0 and retried.relations_written == 4
    finally:
        await kb.store.drop()


class _SloppyLLM(FakeGraphLLM):
    """First extraction answer uses a type the ontology does not have; the retry is clean."""

    async def complete(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> str:
        reply = await super().complete(messages, **kwargs)
        is_extraction = any("Passages:" in m["content"] for m in messages)
        if is_extraction and not any("That JSON has problems" in m["content"] for m in messages):
            data = json.loads(reply)
            data["entities"].append(
                {
                    "ref": "x1",
                    "type": "Kisi",
                    "name": "Ali",
                    "description": None,
                    "properties": {},
                    "chunks": ["K1"],
                }
            )
            return json.dumps(data)
        return reply


async def test_invalid_extraction_is_corrected_once(
    connection: GraphConnection, kb_name: str, ontology: Ontology, graph_docs: list[Document]
) -> None:
    llm = _SloppyLLM()
    kb = AsyncKnowledgeBase(
        kb_name, connection=connection, embedder=FakeEmbedder(), llm=llm, ontology=ontology
    )
    try:
        report = await kb.ingest(graph_docs[:1])
        assert report.validation_errors == []  # corrected on the retry
        assert llm.extraction_calls == 2
        feedback = llm.requests[1][-1]["content"]
        assert "unknown entity type 'Kisi'" in feedback
    finally:
        await kb.store.drop()


async def test_extraction_needs_a_model(
    connection: GraphConnection, kb_name: str, ontology: Ontology
) -> None:
    kb = AsyncKnowledgeBase(
        kb_name, connection=connection, embedder=FakeEmbedder(), ontology=ontology
    )
    with pytest.raises(ConfigError, match="extraction needs a model"):
        await kb.ingest([Document.from_text("x", source="x")])
    await kb.store.drop()


def test_cli_with_ontology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rag_with_oag import cli

    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in __import__("tests.conftest", fromlist=["GRAPH_DOCS"]).GRAPH_DOCS.items():
        (docs / name).write_text(text, encoding="utf-8")
    onto = tmp_path / "ontology.yaml"
    onto.write_text(CONTRACTS_ONTOLOGY, encoding="utf-8")
    monkeypatch.setattr(cli, "EmbedConfig", lambda: FakeEmbedder())
    monkeypatch.setattr(cli, "LLMConfig", lambda: FakeGraphLLM())
    common = ["--path", str(tmp_path / "cli.db"), "--kb", "contracts", "--ontology", str(onto)]

    assert cli.main(["ingest", str(docs), *common]) == 0
    out = capsys.readouterr().out
    assert "relations 4" in out and "depozito (1)" in out

    assert cli.main(["query", "Atlas Lojistik'in sözleşmesi", *common]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Answer [F1][C1].")
    assert "[F1] Atlas Lojistik A.Ş. (Sirket)" in out
