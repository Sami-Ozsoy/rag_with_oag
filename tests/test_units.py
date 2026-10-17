"""Pure-Python units: config, ids, tokens, loaders, chunker, context packing, citations."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from rag_with_oag import ConfigError, EmbedConfig, LLMConfig, RetrievalConfig
from rag_with_oag._tokens import count_tokens, split_spans
from rag_with_oag.config import resolve_api_key
from rag_with_oag.generation.answer import cited
from rag_with_oag.ingest.chunker import AdaptiveChunker, TokenWindowChunker, page_index
from rag_with_oag.ingest.loaders import iter_paths, load_path
from rag_with_oag.ingest.pipeline import build_chunks
from rag_with_oag.retrieval.context import pack_chunks
from rag_with_oag.store.connection import GraphConnection
from rag_with_oag.store.queries import safe_identifier
from rag_with_oag.types import ChunkHit, Citation, Document, chunk_id, content_hash

# --- config ------------------------------------------------------------------


def test_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_OAG_LLM_MODEL", "qwen")
    monkeypatch.setenv("RAG_OAG_LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("RAG_OAG_EMBED_MODEL", "bge")
    llm = LLMConfig()  # type: ignore[call-arg]
    assert (llm.model, llm.base_url) == ("qwen", "http://localhost:8000/v1")
    assert EmbedConfig().model == "bge"  # type: ignore[call-arg]


def test_openai_key_is_only_sent_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert resolve_api_key(None, "https://api.openai.com/v1") == "sk-openai"
    assert resolve_api_key(None, "https://openrouter.ai/api/v1") == "not-needed"
    assert resolve_api_key(None, "http://localhost:11434/v1") == "not-needed"
    assert resolve_api_key(None, "https://api.openai.com.evil.example/v1") == "not-needed"
    assert resolve_api_key(SecretStr("sk-or"), "https://openrouter.ai/api/v1") == "sk-or"
    monkeypatch.delenv("OPENAI_API_KEY")
    assert resolve_api_key(None, "https://api.openai.com/v1") == "not-needed"


def test_retrieval_config_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unknown chunk strategies"):
        RetrievalConfig(chunk_strategies=("nope",))
    with pytest.raises(ValidationError, match="chunk_max_tokens must be >= chunk_min_tokens"):
        RetrievalConfig(chunk_min_tokens=400, chunk_max_tokens=100)
    with pytest.raises(ValidationError, match="chunk_hard_max_tokens must be >= chunk_max_tokens"):
        RetrievalConfig(chunk_max_tokens=800, chunk_hard_max_tokens=500)


def test_connection_needs_exactly_one_target() -> None:
    with pytest.raises(ConfigError):
        GraphConnection()
    with pytest.raises(ConfigError):
        GraphConnection(path="a.db", url="redis://x")


# --- ids & safety ------------------------------------------------------------


def test_ids_are_deterministic_and_whitespace_insensitive() -> None:
    assert content_hash("a\r\nb  \n") == content_hash("a\nb")
    doc = Document.from_text("Merhaba dünya", source="x")
    assert doc.id == Document.from_text("Merhaba dünya", source="other").id
    assert chunk_id(doc.id, 0) == chunk_id(doc.id, 0) != chunk_id(doc.id, 1)


@pytest.mark.parametrize("bad", ["a-b", "1abc", "x) DETACH DELETE (n", "", "Şirket"])
def test_safe_identifier_rejects_injection(bad: str) -> None:
    with pytest.raises(ValueError):
        safe_identifier(bad)


# --- tokens ------------------------------------------------------------------


def test_split_spans_respects_limit_and_keeps_turkish_characters_intact() -> None:
    text = "Şirketler arasındaki sözleşmenin feshi ğüşıöç " * 60
    spans = split_spans(text, 40)
    assert len(spans) > 1
    for start, end in spans:
        piece = text[start:end]
        assert count_tokens(piece) <= 40
        assert "�" not in piece
    assert spans[0][0] == 0 and spans[-1][1] == len(text)


# --- loaders -----------------------------------------------------------------


def test_iter_paths_skips_hidden_and_unsupported(corpus: Path) -> None:
    names = sorted(p.name for p in iter_paths([corpus]))
    assert names == ["duyuru.html", "hizmet.md", "kira.txt"]


def test_loaders_extract_title_and_visible_text(corpus: Path) -> None:
    md = load_path(corpus / "hizmet.md")
    assert md.title == "Hizmet Sözleşmesi"
    html = load_path(corpus / "duyuru.html")
    assert html.title == "Duyuru"
    assert "# Bakım" in html.text and "bakıma girecektir" in html.text
    assert "alert" not in html.text and "p{}" not in html.text
    with pytest.raises(ValueError, match="unsupported"):
        load_path(corpus / "notes.bin")


# --- chunker -----------------------------------------------------------------


def test_small_document_stays_one_chunk(corpus: Path) -> None:
    doc = load_path(corpus / "hizmet.md")
    [chunk] = build_chunks(doc, AdaptiveChunker())
    assert chunk.text == doc.text.strip() and chunk.section_path == ("Hizmet Sözleşmesi",)


def test_adaptive_chunker_offsets_and_sections() -> None:
    body = "Bu madde kapsamında tarafların hak ve yükümlülükleri ayrıntılı olarak düzenlenmiştir. "
    text = (
        "# Sözleşme\n\n## Ödeme\n\n" + body * 40 + "Ücret aylık ödenir.\n\n"
        "## Fesih\n\n" + body * 40 + "Taraflar otuz gün önceden bildirerek feshedebilir.\n"
    )
    doc = Document.from_text(text, source="s.md")
    chunks = build_chunks(doc, AdaptiveChunker())
    assert len(chunks) > 2
    for c in chunks:
        assert doc.text[c.start_char : c.end_char] == c.text
        assert c.text == c.text.strip()
    fesih = [c for c in chunks if "feshedebilir" in c.text]
    assert fesih and fesih[0].section_path == ("Sözleşme", "Fesih")


def test_hard_max_is_enforced_in_model_tokens() -> None:
    text = "# Başlık\n\n" + ("Çalışanların ücretleri ödenmiştir. " * 400)
    # The adaptive target counts ASCII words, so 80 Turkish "tokens" are far more than 100.
    config = RetrievalConfig(chunk_min_tokens=40, chunk_max_tokens=80, chunk_hard_max_tokens=100)
    chunker = AdaptiveChunker(config)
    assert all(count_tokens(s.text) <= 100 for s in chunker.split(text))


def test_falls_back_to_token_window_when_adaptive_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    chunker = AdaptiveChunker()

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("upstream bug")

    monkeypatch.setattr(chunker._impl, "chunk", boom)
    spans = chunker.split("bir iki üç " * 300)
    assert spans and {s.strategy for s in spans} == {TokenWindowChunker.name}
    assert "fallback" in caplog.text


def test_page_index_counts_form_feeds() -> None:
    text = "page one\fpage two\fpage three"
    assert page_index(text, 0) == 0
    assert page_index(text, text.index("three")) == 2
    assert page_index("no pages", 3) is None


# --- context packing & citations ---------------------------------------------


def _hit(i: int, score: float, text: str = "metin", **kw: object) -> ChunkHit:
    return ChunkHit(chunk_id=f"c{i}", doc_id="d", source="s.md", text=text, score=score, **kw)  # type: ignore[arg-type]


def test_pack_keeps_ranking_and_respects_budget() -> None:
    hits = [_hit(2, 0.9), _hit(3, 0.5, text="uzun " * 2000), _hit(1, 0.2)]
    packed = pack_chunks(hits, budget=300, expand_neighbors=False)
    assert [h.chunk_id for h in packed.chunks] == ["c2", "c1"]  # c3 does not fit
    assert [c.id for c in packed.citations] == ["C1", "C2"]
    assert packed.text.startswith("[C1] s.md")
    assert packed.token_count <= 300


def test_pack_adds_neighbors_without_duplicating_hits() -> None:
    a = _hit(1, 0.9, text="ana", prev_id="c0", prev_text="önceki", next_id="c2", next_text="dup")
    b = _hit(2, 0.8, text="dup")
    packed = pack_chunks([a, b], budget=1000, expand_neighbors=True)
    first = packed.text.split("\n\n")[0]
    assert "önceki\nana" in first and "dup" not in first


def test_cited_keeps_known_ids_in_order() -> None:
    available = [Citation("C1", "chunk", "x"), Citation("C2", "chunk", "y")]
    got = cited("See [C2] and [C1], also [C2] and invented [C9].", available)
    assert [c.id for c in got] == ["C2", "C1"]
