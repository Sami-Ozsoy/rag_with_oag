"""Exact entity anchors against embedded FalkorDB."""

from __future__ import annotations

import pytest
import pytest_asyncio

from rag_with_oag import Ontology, RetrievalConfig
from rag_with_oag._text import fulltext_query
from rag_with_oag.ingest.extractor import ChunkGraph, EntityMention
from rag_with_oag.ingest.resolver import EntityResolver
from rag_with_oag.retrieval.oag import OAGRetriever
from rag_with_oag.store.repository import GraphRepository
from rag_with_oag.types import EntityHit

from .conftest import FakeEmbedder

pytestmark = pytest.mark.integration

CONTRACT_KEYS = ("DS-2023-008", "YS-2023-008", "KS-2023-008", "DS-2025-008")
LALE = "Lale Cam Ltd. Şti."
LALE_GROUP = "Lale Cam Yatırım Grubu Ltd. Şti."


@pytest_asyncio.fixture
async def retriever(repo: GraphRepository, fake_embedder: FakeEmbedder) -> OAGRetriever:
    ontology = Ontology.from_yaml(
        """
entities:
  Sirket:
    key: vergi_no
    properties:
      vergi_no: string
  Sozlesme:
    key: numara
    properties:
      numara: string
  Kayit:
    key: numara
    properties:
      numara: integer
"""
    )
    entities = [
        *(
            EntityMention(key, "Sozlesme", f"{key} Danışmanlık Sözleşmesi", None, {"numara": key})
            for key in CONTRACT_KEYS
        ),
        EntityMention("a", "Sirket", LALE, None, {"vergi_no": "5372628807"}),
        EntityMention("b", "Sirket", LALE_GROUP, None, {"vergi_no": "5372628808"}),
        EntityMention("c", "Kayit", "Ticaret Sicil Kaydı", None, {"numara": 5372628809}),
    ]
    await repo.ensure_embedding_dim(model=fake_embedder.model, dim=fake_embedder.dim)
    resolver = EntityResolver(repo, fake_embedder, ontology, similarity=0.92)
    write = await resolver.resolve({"c1": ChunkGraph(entities=entities)})
    write.mentions.clear()
    await repo.write_graph(write, version=ontology.fingerprint)
    return OAGRetriever(repo, fake_embedder, ontology, RetrievalConfig())


async def _link(
    retriever: OAGRetriever, question: str, config: RetrievalConfig | None = None
) -> list[EntityHit]:
    if config is not None:
        retriever = OAGRetriever(retriever.repo, retriever.embedder, retriever.ontology, config)
    [vector] = await retriever.embedder.embed([question])
    return await retriever.linker.link(question, vector)


@pytest.mark.parametrize(
    ("question", "name"),
    [
        ("DS-2023-008'in tarafları kim?", "DS-2023-008 Danışmanlık Sözleşmesi"),
        ("[“ds-2023-008’in”], tarafları kim?", "DS-2023-008 Danışmanlık Sözleşmesi"),
        (f"{LALE} hangi şehirde?", LALE),
        (f"{LALE_GROUP} hangi şehirde?", LALE_GROUP),
        ("Lale Cam hangi şehirde?", LALE),
        ("5372628807’nin ortakları kim?", LALE),
        ("5372628809’nin konusu nedir?", "Ticaret Sicil Kaydı"),
    ],
)
async def test_literal_key_or_stored_name_is_the_only_seed(
    retriever: OAGRetriever, question: str, name: str
) -> None:
    hits = await _link(retriever, question)
    assert [(hit.name, hit.score) for hit in hits] == [(name, 1.0)]


async def test_anchor_order_deduplication_and_limit(retriever: OAGRetriever) -> None:
    question = f"{LALE_GROUP} ve DS-2025-008, 5372628807, DS-2023-008, {LALE} ve DS-2025-008?"
    expected = [
        "DS-2025-008 Danışmanlık Sözleşmesi",
        LALE,
        "DS-2023-008 Danışmanlık Sözleşmesi",
        LALE_GROUP,
    ]
    assert [hit.name for hit in await _link(retriever, question)] == expected
    hits = await _link(retriever, question, RetrievalConfig(link_k=2))
    assert [hit.name for hit in hits] == expected[:2]


async def test_standalone_name_survives_a_nested_occurrence(retriever: OAGRetriever) -> None:
    hits = await _link(retriever, f"{LALE_GROUP} ve {LALE} ilişkili mi?")
    assert [(hit.name, hit.score) for hit in hits] == [(LALE_GROUP, 1.0), (LALE, 1.0)]


async def test_without_anchors_linking_keeps_the_fused_ranking(retriever: OAGRetriever) -> None:
    question = "Lale Camcılık ve danışmanlık hakkında neler biliyoruz?"
    hits = await _link(retriever, question)
    assert hits
    assert hits == await _link(retriever, question, RetrievalConfig(link_exact_anchors=False))


async def test_contract_phrase_ranks_first_without_anchors(retriever: OAGRetriever) -> None:
    question = "DS-2023-008'in tarafları kim?"
    expected = "DS-2023-008 Danışmanlık Sözleşmesi"
    config = RetrievalConfig(link_exact_anchors=False)

    text_hits = await retriever.repo.search_entities_text(fulltext_query(question), config.link_k)
    assert [hit.name for hit in text_hits] == [expected]

    hits = await _link(retriever, question, config)
    assert hits[0].name == expected
    assert all(hit.score < 1.0 for hit in hits)
