"""Ontology extraction, resolution, oag and hybrid retrieval against embedded FalkorDB."""

from __future__ import annotations

import json
import math
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
    RetrievalConfig,
)
from rag_with_oag._tokens import count_tokens
from rag_with_oag.config import ExtractionConfig
from rag_with_oag.generation import prompts
from rag_with_oag.ingest.extractor import ChunkGraph, EntityMention
from rag_with_oag.ingest.graph_builder import GraphBuilder
from rag_with_oag.ingest.resolver import EntityResolver, entity_id
from rag_with_oag.retrieval.context import pack_facts
from rag_with_oag.retrieval.hybrid import HybridRetriever
from rag_with_oag.retrieval.oag import OAGRetriever
from rag_with_oag.store.connection import GraphConnection
from rag_with_oag.store.repository import EntityRow, GraphRepository, GraphWrite, RelationRow
from rag_with_oag.types import Chunk, EntityHit

from .conftest import CONTRACTS_ONTOLOGY, ENTITY_RULES, FakeEmbedder, FakeGraphLLM

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
        "MATCH (e:Entity) WHERE e.name_norm = 'bulut yazilim' "
        "RETURN e.name, e.vergi_no, e.sehir, labels(e)"
    )
    assert (name, vergi, sehir) == ("Bulut Yazılım Ltd.", "1234567890", "Ankara")
    assert labels == ["Entity", "Sirket"]

    [[evidence, rol, mentions]] = await graph_kb.store.query(
        "MATCH (s:Sirket)-[r:TARAFIDIR]->(:Sozlesme) WHERE s.name_norm = 'atlas lojistik' "
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

    two = await graph_kb.retrieve(question, hops=2)
    assert two.mode_used == "hybrid"
    assert not any("Delta" in f.render() for f in two.facts)

    three = await graph_kb.retrieve(question)
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
    question = "Yarın hava yağmurlu olacak mı?"
    ctx = await graph_kb.retrieve(question)
    assert ctx.mode_used == "rag" and not ctx.facts
    assert ctx == await graph_kb.retrieve(question, mode="rag")
    oag = await graph_kb.retrieve(question, mode="oag")
    assert oag.prompt_context == "" and oag.mode_used == "oag"


_CLAUSE_QUESTION = "KS-2024-002 numaralı metne göre yıllık gecikme faizi oranı nedir?"


class _ClauseEmbedder(FakeEmbedder):
    def _vector(self, text: str) -> list[float]:
        similarity = {
            _CLAUSE_QUESTION: 1.0,
            "KS-2024-002": 0.4,
            "Yıllık gecikme faizi %12'dir.": 0.8,
            "Ödemeler banka havalesiyle yapılır.": 0.2,
            "KS-2024-003": 0.6,
            "Yıllık gecikme faizi %18'dir.": 0.96,
        }[text]
        return [similarity, math.sqrt(1 - similarity**2), *([0.0] * (self.dim - 2))]


@pytest_asyncio.fixture(loop_scope="session")
async def contract_clause_corpus(repo: GraphRepository) -> FakeEmbedder:
    embedder = _ClauseEmbedder()
    await repo.ensure_embedding_dim(model=embedder.model, dim=embedder.dim)
    entities = []
    mentions = []
    documents = [
        ("KS-2024-002", "Yıllık gecikme faizi %12'dir.", "Ödemeler banka havalesiyle yapılır."),
        ("KS-2024-003", "Yıllık gecikme faizi %18'dir."),
    ]
    for parts in documents:
        number = parts[0]
        doc = Document.from_text("\n\n".join(parts), source=f"{number}.txt")
        chunks = []
        for idx, text in enumerate(parts):
            start = doc.text.index(text)
            chunks.append(
                Chunk(
                    f"{number}-{idx}",
                    doc.id,
                    idx,
                    text,
                    count_tokens(text),
                    start,
                    start + len(text),
                )
            )
        vectors = await embedder.embed(parts)
        await repo.write_document(doc, chunks, vectors)
        entities.append(
            EntityRow(
                number, "Sozlesme", number, number.lower(), None, {"numara": number}, vectors[0]
            )
        )
        mentions.extend((chunk.id, number) for chunk in chunks if chunk.idx != 1)
    await repo.write_graph(GraphWrite(entities=entities, mentions=mentions), version="v")
    return embedder


async def test_seed_document_chunks_rank_all_chunks_with_neighbors(
    repo: GraphRepository, contract_clause_corpus: FakeEmbedder
) -> None:
    [vector] = await contract_clause_corpus.embed([_CLAUSE_QUESTION])
    [rag_hit] = await repo.search_chunks(vector, 1)
    assert rag_hit.chunk_id == "KS-2024-003-1"

    hits = await repo.seed_document_chunks(["KS-2024-002"], vector, 5)

    assert [hit.chunk_id for hit in hits] == ["KS-2024-002-1", "KS-2024-002-0", "KS-2024-002-2"]
    clause, previous, following = hits
    assert [hit.score for hit in hits] == pytest.approx([0.8, 0.4, 0.2])
    assert clause.score < rag_hit.score
    assert (clause.prev_id, clause.prev_text) == (previous.chunk_id, previous.text)
    assert (clause.next_id, clause.next_text) == (following.chunk_id, following.text)
    assert await repo.seed_document_chunks(["KS-2024-002"], vector, 1) == [clause]


@pytest.mark.parametrize("reserve", [2, 0])
async def test_hybrid_retrieves_clause_from_the_linked_contract(
    repo: GraphRepository, contract_clause_corpus: FakeEmbedder, reserve: int
) -> None:
    ontology = Ontology.from_yaml("""
name: contracts
entities:
  Sozlesme:
    key: numara
    properties:
      numara: string
""")
    hybrid = HybridRetriever(
        repo, contract_clause_corpus, ontology, RetrievalConfig(k=4, hybrid_text_reserve=reserve)
    )

    context = await hybrid.retrieve(_CLAUSE_QUESTION)

    assert context.mode_used == "hybrid"
    assert [fact.subject_id for fact in context.facts] == ["KS-2024-002"]
    citations = {
        citation.id: citation.ref for citation in context.citations if citation.kind == "chunk"
    }
    if reserve:
        assert citations["C1"] == "KS-2024-002-1"
        assert citations["C2"] == "KS-2024-003-1"
    else:
        assert {"KS-2024-002-1", "KS-2024-003-1"} <= set(citations.values())


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


async def test_keyed_mention_joins_keyless_graph_entity_without_looping(
    repo: GraphRepository, ontology: Ontology
) -> None:
    embedder = FakeEmbedder()
    await repo.ensure_embedding_dim(model=embedder.model, dim=embedder.dim)
    resolver = EntityResolver(repo, embedder, ontology, similarity=0.92)

    def bulut(**props: Any) -> ChunkGraph:
        return ChunkGraph(entities=[EntityMention("e1", "Sirket", "Bulut Yazılım", None, props)])

    # The company is first seen without its tax number...
    first = await resolver.resolve({"c1": bulut()})
    await repo.write_graph(first, version="v")
    # ...then one document names it both without and with the tax number.
    write = await resolver.resolve({"c2": bulut(), "c3": bulut(vergi_no="1234567890")})
    assert [(e.id, e.props) for e in write.entities] == [
        (first.entities[0].id, {"vergi_no": "1234567890"})
    ]
    await repo.write_graph(write, version="v")
    assert await repo.query("MATCH (e:Sirket) RETURN e.vergi_no") == [["1234567890"]]


async def test_keyless_name_variant_joins_company_in_later_ingest(
    graph_kb: AsyncKnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    tax_no = "1234567890"
    monkeypatch.setitem(
        ENTITY_RULES, "Zirve Gıda A.Ş.", ("Sirket", {"vergi": ("vergi_no", tax_no)})
    )
    contract = Document.from_text(f"Zirve Gıda A.Ş. vergi numarası {tax_no}.", source="contract")
    assert (await graph_kb.ingest([contract])).ok

    monkeypatch.setitem(ENTITY_RULES, "Zirve Gıda", ("Sirket", {}))
    activity = Document.from_text("Zirve Gıda bu yıl üretimini artırdı.", source="activity")
    assert (await graph_kb.ingest([activity])).ok

    assert (await graph_kb.stats())["entities"] == 1
    rows = await graph_kb.store.query(
        "MATCH (c:Chunk)-[:MENTIONS]->(e:Sirket) RETURN e.name, e.name_norm, e.vergi_no, count(c)"
    )
    assert rows == [["Zirve Gıda A.Ş.", "zirve gida", tax_no, 2]]


async def test_keyless_name_variant_joins_company_in_one_write(
    repo: GraphRepository, ontology: Ontology, fake_embedder: FakeEmbedder
) -> None:
    await repo.ensure_embedding_dim(model=fake_embedder.model, dim=fake_embedder.dim)
    resolver = EntityResolver(repo, fake_embedder, ontology, similarity=1.0)
    keyed = EntityMention("e1", "Sirket", "Zirve Gıda A.Ş.", None, {"vergi_no": "1234567890"})
    plain = EntityMention("e1", "Sirket", "Zirve Gıda", None, {})
    keyless_variant = EntityMention("e2", "Sirket", "Zirve Gıda A.Ş.", None, {})
    name_identity = (entity_id("Sirket", "name:zirve gida"), None)
    assert resolver.identity(plain) == resolver.identity(keyless_variant) == name_identity

    write = await resolver.resolve(
        {
            "c1": ChunkGraph(entities=[keyed]),
            "c2": ChunkGraph(entities=[plain]),
            "c3": ChunkGraph(entities=[plain]),
        }
    )
    keyed_id, _ = resolver.identity(keyed)
    assert write.mentions == [("c1", keyed_id), ("c2", keyed_id), ("c3", keyed_id)]
    assert len(fake_embedder.calls) == len(fake_embedder.calls[0]) == 1
    await repo.write_graph(write, version="v")
    assert await repo.query("MATCH (e:Sirket) RETURN e.name, e.name_norm, e.vergi_no") == [
        ["Zirve Gıda", "zirve gida", "1234567890"]
    ]


async def test_normalized_name_does_not_merge_distinct_keys(
    repo: GraphRepository, ontology: Ontology, fake_embedder: FakeEmbedder
) -> None:
    await repo.ensure_embedding_dim(model=fake_embedder.model, dim=fake_embedder.dim)
    resolver = EntityResolver(repo, fake_embedder, ontology, similarity=0.92)
    for name, tax_no in [("Zirve Gıda A.Ş.", "1111111111"), ("Zirve Gıda", "2222222222")]:
        mention = EntityMention("e1", "Sirket", name, None, {"vergi_no": tax_no})
        write = await resolver.resolve({"c1": ChunkGraph(entities=[mention])})
        await repo.write_graph(write, version="v")
    rows = await repo.query("MATCH (e:Sirket) RETURN e.name_norm, e.vergi_no ORDER BY e.vergi_no")
    assert rows == [["zirve gida", "1111111111"], ["zirve gida", "2222222222"]]


async def test_two_keys_do_not_take_the_same_keyless_entity(
    repo: GraphRepository, ontology: Ontology
) -> None:
    embedder = FakeEmbedder()
    await repo.ensure_embedding_dim(model=embedder.model, dim=embedder.dim)
    resolver = EntityResolver(repo, embedder, ontology, similarity=0.92)

    def bulut(chunk: str, **props: Any) -> dict[str, ChunkGraph]:
        mention = EntityMention("e1", "Sirket", "Bulut Yazılım", None, props)
        return {chunk: ChunkGraph(entities=[mention])}

    await repo.write_graph(await resolver.resolve(bulut("c1")), version="v")
    write = await resolver.resolve(
        {**bulut("c2", vergi_no="1111111111"), **bulut("c3", vergi_no="2222222222")}
    )
    await repo.write_graph(write, version="v")
    rows = await repo.query("MATCH (e:Sirket) RETURN e.vergi_no ORDER BY e.vergi_no")
    assert rows == [["1111111111"], ["2222222222"]]


async def test_expansion_keeps_near_facts_when_relation_budget_runs_out(
    repo: GraphRepository,
) -> None:
    embedder = FakeEmbedder()
    await repo.ensure_embedding_dim(model=embedder.model, dim=embedder.dim)
    names = ["seed", "hub", *(f"far{i}" for i in range(30)), "near"]
    entities = [
        EntityRow(n, "Sirket", n, n, None, {}, (await embedder.embed([n]))[0]) for n in names
    ]
    edges = [("seed", "hub"), *(("hub", f"far{i}") for i in range(30)), ("seed", "near")]
    relations = [RelationRow("ORTAGIDIR", s, t, {}, None, []) for s, t in edges]
    await repo.write_graph(GraphWrite(entities=entities, relations=relations), version="v")

    found = await repo.expand(["seed"], rel_types=["ORTAGIDIR"], hops=2, max_fanout=8, max_facts=2)
    assert [(e.source, e.target, e.hop) for e in found] == [
        ("seed", "hub", 1),
        ("seed", "near", 1),
    ]


async def test_expansion_reaches_three_hops_past_contract_fanout(
    repo: GraphRepository, fake_embedder: FakeEmbedder, ontology: Ontology
) -> None:
    cfg = RetrievalConfig()
    await repo.ensure_embedding_dim(model=fake_embedder.model, dim=fake_embedder.dim)
    contracts = [f"contract{i:02}" for i in range(30)]
    parties = [f"party{i:02}" for i in range(30)]
    companies = ["seed", "A", "B", "C", *parties]
    vectors = await fake_embedder.embed(companies + contracts)
    entities = [
        EntityRow(eid, "Sozlesme" if eid in contracts else "Sirket", eid, eid, None, {}, vector)
        for eid, vector in zip(companies + contracts, vectors, strict=True)
    ]
    ownership = [("A", "seed"), ("B", "A"), ("C", "B")]
    relations = [RelationRow("ORTAGIDIR", s, t, {}, None, []) for s, t in ownership]
    for contract, party in reversed(list(zip(contracts, parties, strict=True))):
        relations.extend(
            [
                RelationRow("TARAFIDIR", "seed", contract, {}, None, []),
                RelationRow("TARAFIDIR", party, contract, {}, None, []),
            ]
        )
    await repo.write_graph(GraphWrite(entities=entities, relations=relations), version="v")

    found = await repo.expand(
        ["seed"],
        rel_types=ontology.relation_names,
        hops=3,
        max_fanout=cfg.max_fanout,
        max_facts=cfg.max_facts,
    )
    assert [(e.source, e.target, e.hop) for e in found if e.type == "ORTAGIDIR"] == [
        (s, t, hop) for hop, (s, t) in enumerate(ownership, start=1)
    ]
    seed_contracts = [e.target for e in found if e.type == "TARAFIDIR" and e.source == "seed"]
    assert seed_contracts == contracts[: cfg.max_fanout]
    assert [(e.source, e.target) for e in found if e.type == "TARAFIDIR" and e.hop == 2] == list(
        zip(parties[:8], contracts[:8], strict=True)
    )
    assert [e.hop for e in found] == sorted(e.hop for e in found)
    assert len(found) == len({(e.source, e.type, e.target) for e in found}) == 19


async def test_expansion_ranks_seeds_and_prefers_seed_neighbours(
    repo: GraphRepository, fake_embedder: FakeEmbedder
) -> None:
    await repo.ensure_embedding_dim(model=fake_embedder.model, dim=fake_embedder.dim)
    first, second = "seed-z", "seed-a"
    pairs = [
        (first, "dropped"),
        (first, "branch-z"),
        (first, second),
        ("branch-a", first),
        (second, "branch-b"),
        ("branch-z", "leaf-z"),
        ("branch-b", "leaf-a"),
    ]
    ids = sorted({eid for pair in pairs for eid in pair})
    entities = [
        EntityRow(eid, "Sirket", eid, eid, None, {}, vector)
        for eid, vector in zip(ids, await fake_embedder.embed(ids), strict=True)
    ]
    relations = [RelationRow("ORTAGIDIR", s, t, {}, None, []) for s, t in pairs]
    await repo.write_graph(GraphWrite(entities=entities, relations=relations), version="v")

    found = await repo.expand(
        [first, second], rel_types=["ORTAGIDIR"], hops=2, max_fanout=2, max_facts=120
    )
    assert [(e.source, e.target, e.hop) for e in found] == [
        (first, second, 1),
        ("branch-a", first, 1),
        (first, "branch-z", 1),
        (second, "branch-b", 1),
        ("branch-z", "leaf-z", 2),
        ("branch-b", "leaf-a", 2),
    ]


async def test_expansion_skips_visited_edges_before_counting_fanout(
    repo: GraphRepository, fake_embedder: FakeEmbedder
) -> None:
    await repo.ensure_embedding_dim(model=fake_embedder.model, dim=fake_embedder.dim)
    max_fanout = 2
    leaves = [f"leaf-{i}" for i in range(max_fanout)]
    ids = ["seed", "A", *leaves]
    entities = [
        EntityRow(eid, "Sirket", eid, eid, None, {}, vector)
        for eid, vector in zip(ids, await fake_embedder.embed(ids), strict=True)
    ]
    # At A, the seed edge and all leaf edges are incoming.
    relations = [RelationRow("ORTAGIDIR", eid, "A", {}, None, []) for eid in ["seed", *leaves]]
    await repo.write_graph(GraphWrite(entities=entities, relations=relations), version="v")

    found = await repo.expand(
        ["seed"], rel_types=["ORTAGIDIR"], hops=2, max_fanout=max_fanout, max_facts=120
    )
    assert [(e.source, e.target, e.hop) for e in found] == [
        ("seed", "A", 1),
        *((leaf, "A", 2) for leaf in leaves),
    ]


async def test_subgraph_packs_properties_next_to_the_relation_that_reaches_them(
    repo: GraphRepository, fake_embedder: FakeEmbedder, ontology: Ontology
) -> None:
    await repo.ensure_embedding_dim(model=fake_embedder.model, dim=fake_embedder.dim)
    ids = ["seed", "near", "far"]
    properties = [{}, {"sehir": "Izmir"}, {"sehir": "Ankara"}]
    entities = [
        EntityRow(eid, "Sirket", eid, eid, None, props, vector)
        for eid, props, vector in zip(ids, properties, await fake_embedder.embed(ids), strict=True)
    ]
    relations = [
        RelationRow("ORTAGIDIR", "near", "seed", {}, None, []),
        RelationRow("ORTAGIDIR", "far", "near", {}, None, []),
    ]
    await repo.write_graph(GraphWrite(entities=entities, relations=relations), version="v")
    oag = OAGRetriever(repo, fake_embedder, ontology, RetrievalConfig())

    subgraph = await oag.subgraph([EntityHit("seed", "seed", "Sirket", 1.0)])
    assert [(f.kind, f.subject_id, f.object_id) for f in subgraph.facts] == [
        ("entity", "seed", ""),
        ("relation", "near", "seed"),
        ("entity", "near", ""),
        ("relation", "far", "near"),
        ("entity", "far", ""),
    ]
    assert subgraph.entity_ids == ids
    near_facts = subgraph.facts[:3]
    budget = sum(count_tokens(f"[F{i}] {f.render()}") + 1 for i, f in enumerate(near_facts, 1))
    packed = pack_facts(subgraph.facts, budget=budget)
    assert packed.facts == near_facts
    assert "Izmir" in packed.text and "Ankara" not in packed.text


async def test_hybrid_keeps_facts_and_gives_texts_the_unused_allowance(
    repo: GraphRepository, fake_embedder: FakeEmbedder, ontology: Ontology
) -> None:
    cfg = RetrievalConfig(
        context_token_budget=256, link_k=1, expand_neighbors=False, hybrid_text_reserve=0
    )
    await repo.ensure_embedding_dim(model=fake_embedder.model, dim=fake_embedder.dim)
    [vector] = await fake_embedder.embed(["Atlas"])
    entity = EntityRow("seed", "Sirket", "Atlas", "atlas", "abc " * 94, {}, vector)
    await repo.write_graph(GraphWrite(entities=[entity]), version="v")
    hybrid = HybridRetriever(repo, fake_embedder, ontology, cfg)
    subgraph = await hybrid.oag.subgraph([EntityHit("seed", "Atlas", "Sirket", 1.0)])
    fact_budget = int(cfg.context_token_budget * cfg.fact_budget_ratio)
    facts = pack_facts(subgraph.facts, budget=fact_budget)
    assert len(facts.facts) == 1
    assert 0.35 * cfg.context_token_budget < facts.token_count < fact_budget

    # The chunk needs more than 40% of the total budget and fits the remainder.
    source = "source.txt"
    text = "Atlas"
    while count_tokens(f"[C1] {source}\n{text}") + 2 <= cfg.context_token_budget - fact_budget:
        text += " abc"
    chunk_cost = count_tokens(f"[C1] {source}\n{text}") + 2
    assert chunk_cost <= cfg.context_token_budget - facts.token_count
    doc = Document.from_text(text, source=source)
    chunk = Chunk("source", doc.id, 0, text, count_tokens(text), 0, len(text))
    await repo.write_document(doc, [chunk], await fake_embedder.embed([text]))
    await repo.write_graph(GraphWrite(mentions=[(chunk.id, entity.id)]), version="v")

    ctx = await hybrid.retrieve("Atlas", vector=vector)
    assert ctx.mode_used == "hybrid"
    assert ctx.facts == subgraph.facts
    assert [hit.text for hit in ctx.chunks] == [text]


def test_extraction_cache_key_follows_the_prompt(
    repo: GraphRepository, ontology: Ontology, monkeypatch: pytest.MonkeyPatch
) -> None:
    def key() -> str:
        builder = GraphBuilder(repo, FakeGraphLLM(), FakeEmbedder(), ontology, ExtractionConfig())
        return builder.cache_key(Chunk("c1", "d1", 0, "Atlas Lojistik A.Ş.", 5, 0, 19))

    before = key()
    assert key() == before
    monkeypatch.setattr(prompts, "EXTRACTION_SYSTEM", prompts.EXTRACTION_SYSTEM + "\n- Be brief.")
    assert key() != before


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
