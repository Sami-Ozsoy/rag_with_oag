from __future__ import annotations

import pytest

from rag_with_oag import Ontology
from rag_with_oag.ingest.extractor import ChunkGraph, EntityMention
from rag_with_oag.ingest.resolver import EntityResolver
from rag_with_oag.store.repository import GraphRepository

from .conftest import FakeEmbedder


class NearIdenticalEmbedder(FakeEmbedder):
    def _vector(self, text: str) -> list[float]:
        return [1.0, 0.1 if text.startswith("Zirve Gıda A.Ş.") else 0.0]


@pytest.fixture
async def resolver(repo: GraphRepository, ontology: Ontology) -> EntityResolver:
    embedder = NearIdenticalEmbedder(dim=2)
    await repo.ensure_embedding_dim(model=embedder.model, dim=embedder.dim)
    return EntityResolver(repo, embedder, ontology, similarity=0.92)


def company(name: str, tax_no: str | None = None, *, ref: str = "e1") -> EntityMention:
    props = {"vergi_no": tax_no} if tax_no is not None else {}
    return EntityMention(ref, "Sirket", name, None, props)


@pytest.mark.parametrize(
    ("keyed_name", "keyless_name"),
    [
        ("Bahar Ambalaj A.Ş.", "Kemer Ambalaj A.Ş."),
        ("Zeytin Yazılım Ltd. Şti.", "Zeytin Yapı Ltd Şti"),
    ],
)
async def test_incompatible_graph_hit_stays_separate(
    resolver: EntityResolver, repo: GraphRepository, keyed_name: str, keyless_name: str
) -> None:
    keyed = await resolver.resolve({"registry": ChunkGraph(entities=[company(keyed_name, "1")])})
    await repo.write_graph(keyed, version="v")

    keyless = await resolver.resolve({"contract": ChunkGraph(entities=[company(keyless_name)])})
    await repo.write_graph(keyless, version="v")

    rows = await repo.query("MATCH (e:Sirket) RETURN e.name, e.vergi_no")
    assert {tuple(row) for row in rows} == {(keyed_name, "1"), (keyless_name, None)}


async def test_partial_names_join_keyed_company(
    resolver: EntityResolver, repo: GraphRepository
) -> None:
    registry = ChunkGraph(entities=[company("Zirve Gıda A.Ş.", "1")])
    keyed = await resolver.resolve({"registry": registry})
    await repo.write_graph(keyed, version="v")

    mentions = {name: ChunkGraph(entities=[company(name)]) for name in ("Zirve", "Zirve Gıda")}
    write = await resolver.resolve(mentions)
    await repo.write_graph(write, version="v")

    keyed_id = keyed.entities[0].id
    assert write.mentions == [("Zirve", keyed_id), ("Zirve Gıda", keyed_id)]
    assert await repo.query("MATCH (e:Sirket) RETURN e.name") == [["Zirve Gıda A.Ş."]]


async def test_incompatible_mentions_stay_separate_in_one_write(
    resolver: EntityResolver, repo: GraphRepository
) -> None:
    names = ("Bahar Ambalaj A.Ş.", "Kemer Ambalaj A.Ş.")
    graph = ChunkGraph(entities=[company(name, ref=f"e{i}") for i, name in enumerate(names)])
    write = await resolver.resolve({"contract": graph})
    await repo.write_graph(write, version="v")

    assert await repo.query("MATCH (e:Sirket) RETURN e.name ORDER BY e.name") == [
        [name] for name in names
    ]


async def test_compatible_lower_vector_hit_is_used(
    resolver: EntityResolver, repo: GraphRepository
) -> None:
    registry = ChunkGraph(
        entities=[company("Zirve Yazılım Ltd.", "1"), company("Zirve Gıda A.Ş.", "2", ref="e2")]
    )
    keyed = await resolver.resolve({"registry": registry})
    await repo.write_graph(keyed, version="v")

    hits = await repo.search_entities_vector([1.0, 0.0], 5)
    assert [hit.name for hit in hits] == ["Zirve Yazılım Ltd.", "Zirve Gıda A.Ş."]
    assert all(hit.score >= resolver.similarity for hit in hits)

    graph = ChunkGraph(entities=[company("Zirve Gıda Dağıtım")])
    write = await resolver.resolve({"contract": graph})

    assert write.mentions == [("contract", hits[1].id)]
