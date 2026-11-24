"""Document deletion, replacement and write serialization."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from rag_with_oag import AsyncKnowledgeBase, DeleteReport, Document, IngestReport, Ontology
from rag_with_oag.ingest.chunker import Span
from rag_with_oag.ingest.loaders import Source
from rag_with_oag.ingest.pipeline import IngestPipeline
from rag_with_oag.store import repository, schema
from rag_with_oag.store.connection import GraphConnection
from rag_with_oag.types import chunk_id

from .conftest import FakeEmbedder, FakeGraphLLM

ATLAS = "Atlas Lojistik A.Ş."
BULUT = "Bulut Yazılım Ltd."
DELTA = "Delta Holding A.Ş."
CONTRACT = "HS-2024-01 Hizmet Sözleşmesi"


class _LineChunker:
    def split(self, text: str) -> list[Span]:
        spans = []
        start = 0
        for line in text.splitlines(keepends=True):
            value = line.rstrip("\n")
            spans.append(Span(start, start + len(value), value, (), "line"))
            start += len(line)
        return spans


@pytest_asyncio.fixture
async def lifecycle_kb(
    connection: GraphConnection, kb_name: str, ontology: Ontology
) -> AsyncIterator[AsyncKnowledgeBase]:
    async with AsyncKnowledgeBase(
        kb_name,
        connection=connection,
        embedder=FakeEmbedder(),
        extractor_llm=FakeGraphLLM(),
        chunker=_LineChunker(),
        ontology=ontology,
    ) as kb:
        yield kb
        await kb.store.drop()


async def _relations(kb: AsyncKnowledgeBase) -> list[list[Any]]:
    return await kb.store.query(
        "MATCH (a:Entity)-[r]->(b:Entity) "
        "RETURN a.name, type(r), b.name, r.evidence ORDER BY a.name, type(r), b.name"
    )


@pytest.mark.integration
async def test_delete_source_prunes_graph_and_retrieval(
    lifecycle_kb: AsyncKnowledgeBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb = lifecycle_kb
    monkeypatch.setattr(repository, "WRITE_BATCH", 1)
    a_path = tmp_path / "a.txt"
    a_text = f"{ATLAS} {CONTRACT}.\n{BULUT} {CONTRACT} silinecek.\n{BULUT} {CONTRACT} tekrar."
    a_path.write_text(a_text, encoding="utf-8")
    a = Document.from_text(a_text, source=str(a_path))
    b = Document.from_text(
        f"{BULUT} {CONTRACT} kalacak. "
        "Güncel kayıt, şirket arşivinde ve ticaret sicilinde yer alır.",
        source="b.txt",
    )
    report = await kb.ingest([a_path, b], concurrency=1)
    assert report.ok
    b_chunk = chunk_id(b.id, 0)
    assert [r[:3] for r in await _relations(kb)] == [
        [ATLAS, "TARAFIDIR", CONTRACT],
        [BULUT, "TARAFIDIR", CONTRACT],
    ]
    lexical_question = "silinecek kalacak"
    assert [c.doc_id for c in await kb.store.search_chunks_text(lexical_question, 1)] == [a.id]
    question = f"{ATLAS} {BULUT} {CONTRACT}"
    before = await kb.retrieve(question, mode="hybrid")
    assert {f.subject for f in before.facts if f.predicate == "TARAFIDIR"} == {ATLAS, BULUT}

    deleted = await kb.delete(f"{tmp_path}/./a.txt", a.id)
    assert deleted == DeleteReport(
        documents=1, chunks=3, relations_deleted=1, relations_trimmed=1, entities_deleted=1
    )
    assert all(
        type(count) is int
        for count in (
            deleted.documents,
            deleted.chunks,
            deleted.relations_deleted,
            deleted.relations_trimmed,
            deleted.entities_deleted,
        )
    )
    assert [c.doc_id for c in await kb.store.search_chunks_text(lexical_question, 1)] == [b.id]
    assert await kb.store.search_chunks_text("silinecek", 1) == []
    assert await kb.store.query("MATCH (d:Document) RETURN d.id") == [[b.id]]
    assert await kb.store.query("MATCH (c:Chunk) RETURN c.id") == [[b_chunk]]
    assert await kb.store.query("MATCH ()-[r:NEXT]->() RETURN count(r)") == [[0]]
    assert await kb.store.query("MATCH (e:Entity) RETURN e.name ORDER BY e.name") == [
        [BULUT],
        [CONTRACT],
    ]
    assert await _relations(kb) == [[BULUT, "TARAFIDIR", CONTRACT, [b_chunk]]]
    for mode in ("rag", "hybrid"):
        after = await kb.retrieve(question, mode=mode)
        assert after.mode_used == mode
        assert {c.doc_id for c in after.chunks} == {b.id}
        assert ATLAS not in after.prompt_context and "silinecek" not in after.prompt_context
        assert all(f.subject != ATLAS for f in after.facts)

    llm = kb.extractor_llm
    assert isinstance(llm, FakeGraphLLM)
    calls = llm.extraction_calls
    assert (await kb.ingest(a_path)).ok
    assert llm.extraction_calls == calls


@pytest.mark.integration
async def test_delete_leaves_unrelated_orphans_and_untouched_evidence(
    lifecycle_kb: AsyncKnowledgeBase,
) -> None:
    kb = lifecycle_kb
    doc = Document.from_text(f"{DELTA} ve {BULUT}.", source="a.txt")
    assert (await kb.ingest(doc)).ok
    await kb.store.query(
        "MATCH (a:Entity {name: $delta}), (b:Entity {name: $bulut}) "
        "CREATE (a)-[:ORTAGIDIR {evidence: $empty}]->(b), (b)-[:ORTAGIDIR]->(a), "
        "(:Entity {name: $isolated})",
        {"delta": DELTA, "bulut": BULUT, "empty": [], "isolated": "isolated"},
    )
    assert await kb.delete(doc.id, Path("missing.txt")) == DeleteReport(
        documents=1, chunks=1, relations_deleted=1, not_found=["missing.txt"]
    )
    assert await kb.delete(doc.id) == DeleteReport(not_found=[doc.id])
    assert await _relations(kb) == [
        [BULUT, "ORTAGIDIR", DELTA, None],
        [DELTA, "ORTAGIDIR", BULUT, []],
    ]
    assert await kb.store.query("MATCH (e:Entity) RETURN count(e)") == [[3]]


@pytest.mark.integration
async def test_replace_restates_shared_evidence_and_removes_old_facts(
    lifecycle_kb: AsyncKnowledgeBase,
) -> None:
    kb = lifecycle_kb
    old = Document.from_text(f"{ATLAS} {BULUT} {CONTRACT} eskikayit.", source="a.txt")
    new = Document.from_text(f"{BULUT} {DELTA} {CONTRACT} yenikayit.", source=old.source)
    assert (await kb.ingest(old)).ok
    assert (await kb.store.search_chunks_text("eskikayit", 1))[0].doc_id == old.id

    report = await kb.ingest(new)
    assert report.ok and report.documents_replaced == 1
    assert await kb.store.search_chunks_text("eskikayit", 1) == []
    assert [c.doc_id for c in await kb.store.search_chunks_text("yenikayit", 1)] == [new.id]
    assert await kb.store.query("MATCH (d:Document) RETURN d.id, d.source") == [
        [new.id, old.source]
    ]
    evidence = [chunk_id(new.id, 0)]
    assert await _relations(kb) == [
        [BULUT, "TARAFIDIR", CONTRACT, evidence],
        [DELTA, "ORTAGIDIR", BULUT, evidence],
    ]


@pytest.mark.integration
async def test_failed_extraction_keeps_old_version_until_retry(
    lifecycle_kb: AsyncKnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb = lifecycle_kb
    old = Document.from_text(f"{ATLAS} {CONTRACT}.", source="a.txt")
    new = Document.from_text(f"{BULUT} {CONTRACT}.", source=old.source)
    assert (await kb.ingest(old)).ok
    llm = kb.extractor_llm
    assert isinstance(llm, FakeGraphLLM)

    async def fail(messages: Sequence[dict[str, Any]], **kwargs: Any) -> str:
        raise RuntimeError("extraction model unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(llm, "complete", fail)
        report = await kb.ingest(new)
    assert not report.ok and report.documents_replaced == 0
    assert "extraction model unavailable" in report.failures[0][1]
    assert await _relations(kb) == [[ATLAS, "TARAFIDIR", CONTRACT, [chunk_id(old.id, 0)]]]
    assert (await kb.stats())["documents"] == 2

    retry = await kb.ingest(new)
    assert retry.ok and retry.documents_replaced == 1 and retry.chunks_written == 0
    assert await _relations(kb) == [[BULUT, "TARAFIDIR", CONTRACT, [chunk_id(new.id, 0)]]]


@pytest.mark.integration
async def test_opt_out_keeps_versions_and_identical_content_moves_source(
    lifecycle_kb: AsyncKnowledgeBase,
) -> None:
    kb = lifecycle_kb
    old = Document.from_text(f"{ATLAS} {CONTRACT}.", source="a.txt")
    new = Document.from_text(f"{BULUT} {CONTRACT}.", source=old.source)
    report = await kb.ingest([old, new], replace=False)
    assert report.ok and report.documents_replaced == 0
    assert (await kb.stats())["documents"] == 2

    moved = Document.from_text(old.text, source="b.txt")
    report = await kb.ingest(moved)
    assert report.ok and report.documents_skipped == 1 and report.chunks_written == 0
    assert await kb.store.query("MATCH (d:Document) RETURN d.source, d.id ORDER BY d.source") == [
        [new.source, new.id],
        [moved.source, old.id],
    ]
    assert await kb.delete("b.txt") == DeleteReport(
        documents=1, chunks=1, relations_deleted=1, entities_deleted=1
    )


@pytest.mark.integration
async def test_replace_removes_all_other_complete_versions(
    lifecycle_kb: AsyncKnowledgeBase,
) -> None:
    kb = lifecycle_kb
    versions = [Document.from_text(f"Sürüm {i}.", source="a.txt") for i in range(4)]
    assert (await kb.ingest(versions[:3], replace=False)).ok
    await kb.store.query(
        "CREATE (:Document {id: $id, source: $source, status: $status})",
        {"id": versions[3].id, "source": versions[3].source, "status": "pending"},
    )
    report = await kb.ingest(versions[2])
    assert report.ok and report.documents_replaced == 2
    assert {doc_id for (doc_id,) in await kb.store.query("MATCH (d:Document) RETURN d.id")} == {
        versions[2].id,
        versions[3].id,
    }
    assert ("Document", "source", "RANGE") in await schema.existing_indexes(kb.store._graph)


async def test_delete_waits_for_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    entered_ingest, finish_ingest, entered_delete = (asyncio.Event() for _ in range(3))
    kb = AsyncKnowledgeBase("docs", embedder=FakeEmbedder())

    async def run(
        self: IngestPipeline, sources: Iterable[Source], *, concurrency: int, replace: bool
    ) -> IngestReport:
        entered_ingest.set()
        await finish_ingest.wait()
        return IngestReport()

    async def delete() -> DeleteReport:
        entered_delete.set()
        return await kb.delete("a.txt")

    delete_store = AsyncMock(return_value=DeleteReport(documents=1))
    monkeypatch.setattr(IngestPipeline, "run", run)
    monkeypatch.setattr(kb, "open", AsyncMock(return_value=kb))
    monkeypatch.setattr(kb.store, "delete", delete_store)

    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(kb.ingest(Document.from_text("text", source="a.txt")))
        await entered_ingest.wait()
        delete_task = tasks.create_task(delete())
        await entered_delete.wait()
        delete_store.assert_not_awaited()
        assert not delete_task.done()
        finish_ingest.set()
    assert delete_task.result().documents == 1
    delete_store.assert_awaited_once_with("a.txt")


@pytest.mark.integration
def test_cli_delete_and_no_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rag_with_oag import cli

    monkeypatch.setattr(cli, "EmbedConfig", FakeEmbedder)
    path = tmp_path / "a.txt"
    common = ["--kb", "docs", "--path", str(tmp_path / "cli.db")]
    path.write_text("original document", encoding="utf-8")
    assert cli.main(["ingest", str(path), *common]) == 0
    assert "replaced 0" in capsys.readouterr().out

    path.write_text("edited document", encoding="utf-8")
    assert cli.main(["ingest", str(path), "--no-replace", "--json", *common]) == 0
    assert json.loads(capsys.readouterr().out)["documents_replaced"] == 0

    assert cli.main(["delete", str(path), "--json", *common]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "documents": 2,
        "chunks": 2,
        "relations_deleted": 0,
        "relations_trimmed": 0,
        "entities_deleted": 0,
        "not_found": [],
    }
    assert cli.main(["delete", str(path), *common]) == 0
    output = capsys.readouterr().out
    assert "deleted documents 0" in output and f"not found: {path}" in output
