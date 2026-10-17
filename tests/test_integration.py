"""Real Cypher against embedded FalkorDB: schema, idempotent ingest, search, facade, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_with_oag import (
    AsyncKnowledgeBase,
    ConfigError,
    Document,
    EmbeddingMismatchError,
    KnowledgeBase,
    KnowledgeBaseClosedError,
    ModeNotAvailableError,
)
from rag_with_oag.ingest.chunker import AdaptiveChunker
from rag_with_oag.ingest.pipeline import IngestPipeline, build_chunks
from rag_with_oag.store import schema
from rag_with_oag.store.connection import GraphConnection
from rag_with_oag.store.repository import GraphRepository

from .conftest import FakeEmbedder, FakeLLM

pytestmark = pytest.mark.integration


# --- store -------------------------------------------------------------------


async def test_schema_setup_is_idempotent(repo: GraphRepository) -> None:
    graph = repo._graph
    await schema.ensure_base_schema(graph, repo.kb)  # second run must not fail
    await repo.ensure_embedding_dim(model="fake-embed", dim=8)
    await repo.ensure_embedding_dim(model="fake-embed", dim=8)
    indexes = await schema.existing_indexes(graph)
    assert {
        ("Chunk", "embedding", "VECTOR"),
        ("Chunk", "text", "FULLTEXT"),
        ("Document", "id", "RANGE"),
    } <= indexes
    assert repo.meta.embed_dim == 8
    [[options]] = await repo.query(
        "CALL db.indexes() YIELD label, options WHERE label = 'Chunk' RETURN options"
    )
    assert options["embedding"]["efRuntime"] == schema.VECTOR_EF_RUNTIME


async def test_dimension_and_model_mismatch_are_refused(
    connection: GraphConnection, repo: GraphRepository
) -> None:
    await repo.ensure_embedding_dim(model="fake-embed", dim=8)
    with pytest.raises(EmbeddingMismatchError, match="8-dim"):
        await repo.ensure_embedding_dim(model="fake-embed", dim=16)
    reopened = GraphRepository(connection, repo.kb)
    with pytest.raises(EmbeddingMismatchError, match="fake-embed"):
        await reopened.init(embed_model="another-model")


async def test_search_returns_similarity_best_first_with_neighbors(
    repo: GraphRepository, fake_embedder: FakeEmbedder
) -> None:
    doc = Document.from_text(
        "# A\n\n" + "\n\n".join(f"Paragraf {i}: konu{i} hakkında bilgi." for i in range(6)),
        source="a.md",
    )
    pipeline = IngestPipeline(repo, fake_embedder, AdaptiveChunker())
    await pipeline.run([doc])
    [q] = await fake_embedder.embed(["konu3 hakkında"])
    hits = await repo.search_chunks(q, 3)
    assert hits and hits == sorted(hits, key=lambda h: h.score, reverse=True)
    assert all(-1.0 <= h.score <= 1.0 for h in hits)
    assert hits[0].source == "a.md"
    stats = await repo.stats()
    assert stats["documents"] == 1 and stats["chunks"] >= 1


async def test_search_before_any_ingest_is_empty(repo: GraphRepository) -> None:
    assert await repo.search_chunks([0.1] * 8, 5) == []


# --- pipeline ----------------------------------------------------------------


async def test_ingest_is_idempotent_and_links_chunks(
    repo: GraphRepository, fake_embedder: FakeEmbedder, corpus: Path
) -> None:
    pipeline = IngestPipeline(repo, fake_embedder, AdaptiveChunker())
    first = await pipeline.run([corpus])
    assert first.ok, first.failures
    assert (first.documents_processed, first.documents_skipped) == (3, 0)
    assert first.token_usage["embedding_tokens"] > 0

    calls = len(fake_embedder.calls)
    second = await pipeline.run([corpus])
    assert (second.documents_processed, second.documents_skipped) == (0, 3)
    assert len(fake_embedder.calls) == calls  # nothing re-embedded

    [[docs, chunks, nexts, has]] = await repo.query(
        "MATCH (d:Document) WITH count(d) AS docs "
        "MATCH (c:Chunk) WITH docs, count(c) AS chunks "
        "OPTIONAL MATCH ()-[n:NEXT]->() WITH docs, chunks, count(n) AS nexts "
        "OPTIONAL MATCH (:Document)-[h:HAS_CHUNK]->() RETURN docs, chunks, nexts, count(h)"
    )
    assert docs == 3 and chunks == first.chunks_written == has
    assert nexts == chunks - docs  # a NEXT chain per document


async def test_interrupted_document_is_retried(
    repo: GraphRepository, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = Document.from_text("Yarım kalan ingest denemesi. " * 20, source="x.txt")
    pipeline = IngestPipeline(repo, fake_embedder, AdaptiveChunker())
    original = repo.write_document

    async def crash_before_complete(d, chunks, vectors):  # type: ignore[no-untyped-def]
        await repo.query(  # write the pending document, then fail like a dropped connection
            "MERGE (d:Document {id: $id}) SET d.status = 'pending'", {"id": d.id}
        )
        raise ConnectionError("lost connection")

    monkeypatch.setattr(repo, "write_document", crash_before_complete)
    failed = await pipeline.run([doc])
    assert failed.failures and "ConnectionError" in failed.failures[0][1]

    monkeypatch.setattr(repo, "write_document", original)
    retried = await pipeline.run([doc])
    assert retried.documents_processed == 1 and retried.ok


async def test_bad_sources_are_reported_not_raised(
    repo: GraphRepository, fake_embedder: FakeEmbedder, tmp_path: Path
) -> None:
    empty = tmp_path / "empty.md"
    empty.write_text("   \n", encoding="utf-8")
    report = await IngestPipeline(repo, fake_embedder, AdaptiveChunker()).run(
        [tmp_path / "missing.md", empty]
    )
    assert report.documents_processed == 0
    assert {Path(src).name for src, _ in report.failures} == {"missing.md", "empty.md"}


async def test_dimension_mismatch_aborts_ingest(
    repo: GraphRepository, fake_embedder: FakeEmbedder
) -> None:
    pipeline = IngestPipeline(repo, fake_embedder, AdaptiveChunker())
    await pipeline.run([Document.from_text("ilk doküman", source="a")])
    wider = FakeEmbedder(dim=128)  # same model name, different dimension
    with pytest.raises(EmbeddingMismatchError):
        await IngestPipeline(repo, wider, AdaptiveChunker()).run(
            [Document.from_text("ikinci doküman", source="b")]
        )


def test_build_chunks_assigns_pages_for_pdf_like_text() -> None:
    doc = Document.from_text("Birinci sayfa metni.\fİkinci sayfa metni.", source="p.pdf")
    pages = {c.page_index for c in build_chunks(doc, AdaptiveChunker())}
    assert pages <= {0, 1} and 0 in pages


# --- facade ------------------------------------------------------------------


async def test_async_kb_end_to_end(
    connection: GraphConnection, kb_name: str, corpus: Path, fake_llm: FakeLLM
) -> None:
    kb = AsyncKnowledgeBase(kb_name, connection=connection, embedder=FakeEmbedder(), llm=fake_llm)
    try:
        report = await kb.ingest(corpus)
        assert report.documents_processed == 3

        ctx = await kb.retrieve("Sözleşmenin fesih bildirimi nasıl yapılır?")
        assert ctx.mode_used == "rag" and ctx.chunks
        assert any("noter" in h.text for h in ctx.chunks)
        assert ctx.prompt_context.startswith("Source texts:\n\n[C1] ")

        res = await kb.query("Sözleşmenin fesih bildirimi nasıl yapılır?", k=2)
        assert res.answer == "According to the sources [C1]."
        assert [c.id for c in res.citations] == ["C1"]
        assert res.citations[0].ref == res.chunks[0].chunk_id
        assert res.usage == {"prompt_tokens": 100, "completion_tokens": 10}
        system, user = fake_llm.requests[-1]
        assert system["role"] == "system" and "Question: Sözleşmenin" in user["content"]

        with pytest.raises(ModeNotAvailableError, match="needs an ontology"):
            await kb.query("x", mode="hybrid")
        with pytest.raises(ValueError, match="mode must be"):
            await kb.retrieve("x", mode="graph")  # type: ignore[arg-type]
    finally:
        await kb.store.drop()
        await kb.close()
    assert connection.is_open  # a shared connection is not closed by the KB


async def test_query_without_llm_or_results(connection: GraphConnection, kb_name: str) -> None:
    kb = AsyncKnowledgeBase(kb_name, connection=connection, embedder=FakeEmbedder())
    try:
        with pytest.raises(ConfigError, match="needs an llm"):
            await kb.query("x")
        kb.llm = FakeLLM()
        res = await kb.query("boş bilgi tabanı")
        assert res.chunks == [] and "no relevant sources" in res.answer
        assert kb.llm.requests == []  # type: ignore[attr-defined]
    finally:
        await kb.store.drop()
        await kb.close()


def test_kb_rejects_bad_arguments() -> None:
    with pytest.raises(ConfigError, match="must match"):
        AsyncKnowledgeBase("bad-name", embedder=FakeEmbedder())
    with pytest.raises(ConfigError):
        AsyncKnowledgeBase("ok", embedder=FakeEmbedder(), path="a.db", url="redis://x")


def test_sync_kb_persists_across_reopen(tmp_path: Path, corpus: Path) -> None:
    db = tmp_path / "sync.db"
    with KnowledgeBase("contracts", path=db, embedder=FakeEmbedder(), llm=FakeLLM()) as kb:
        assert kb.ingest([corpus]).documents_processed == 3
        assert kb.query("kira bedeli artışı").citations
        assert kb.store.query("MATCH (d:Document) RETURN count(d)") == [[3]]

    with KnowledgeBase("contracts", path=db, embedder=FakeEmbedder()) as kb:
        assert kb.stats()["documents"] == 3
        assert kb.ingest(corpus).documents_skipped == 3
        with pytest.raises(EmbeddingMismatchError):
            KnowledgeBase("contracts", path=db, embedder=FakeEmbedder(model="other")).ingest(corpus)


async def test_closed_kb_refuses_work(connection: GraphConnection, kb_name: str) -> None:
    kb = AsyncKnowledgeBase(kb_name, connection=connection, embedder=FakeEmbedder())
    await kb.open()
    await kb.store.drop()
    await kb.close()
    await kb.close()  # twice is fine
    with pytest.raises(KnowledgeBaseClosedError, match="is closed"):
        await kb.retrieve("x")


def test_closed_sync_kb_refuses_work(tmp_path: Path) -> None:
    kb = KnowledgeBase("contracts", path=tmp_path / "c.db", embedder=FakeEmbedder())
    kb.close()
    kb.close()
    with pytest.raises(KnowledgeBaseClosedError):
        kb.stats()


# --- CLI ---------------------------------------------------------------------


def test_cli_ingest_and_retrieve(
    tmp_path: Path,
    corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rag_with_oag import cli

    monkeypatch.setenv("RAG_OAG_EMBED_MODEL", "fake-embed")
    monkeypatch.setattr(cli, "EmbedConfig", lambda: FakeEmbedder())
    db = str(tmp_path / "cli.db")

    assert cli.main(["ingest", str(corpus), "--path", db, "--kb", "docs"]) == 0
    assert "processed 3" in capsys.readouterr().out

    code = cli.main(
        ["query", "kira bedeli", "--path", db, "--kb", "docs", "--retrieve-only", "--json"]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["mode_used"] == "rag" and out["chunks"]

    assert (
        cli.main(["query", "x", "--path", db, "--kb", "docs", "--mode", "oag", "--retrieve-only"])
        == 1
    )
    assert "needs an ontology" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--ontology", "missing.yaml"], "No such file"),
        (["--ontology", "BROKEN"], "not valid YAML"),
        (["--kb", "bad-name"], "must match"),
    ],
)
def test_cli_reports_setup_errors_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
) -> None:
    from rag_with_oag import cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "BROKEN").write_text("entities: [unclosed", encoding="utf-8")
    monkeypatch.setattr(cli, "EmbedConfig", lambda: FakeEmbedder())
    args = ["query", "x", "--retrieve-only", "--path", str(tmp_path / "cli.db"), *extra]
    assert cli.main(args) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and message in err
