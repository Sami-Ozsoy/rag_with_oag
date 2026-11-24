"""Retrieval budget and ranking tests with fake search results."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock, create_autospec

import pytest

from rag_with_oag import Ontology, RetrievalConfig
from rag_with_oag._tokens import count_tokens
from rag_with_oag.rerank import Reranker
from rag_with_oag.retrieval.context import CHUNKS_HEADER, FACTS_HEADER, pack_facts
from rag_with_oag.retrieval.hybrid import HybridRetriever
from rag_with_oag.retrieval.oag import Subgraph
from rag_with_oag.retrieval.rag import RAGRetriever, search_rag_hits
from rag_with_oag.store.repository import GraphRepository
from rag_with_oag.types import ChunkHit, Citation, EntityHit, Fact

from .conftest import FakeEmbedder


def _hit(id_: str, tokens: int) -> ChunkHit:
    hit = ChunkHit(id_, "doc", "source.txt", "Atlas", 1.0)
    base = f"[C1] {hit.location}\n{hit.text}"
    padding = " ab" * (tokens - count_tokens(base))
    assert count_tokens(base + padding) == tokens
    return replace(hit, text=hit.text + padding)


def _fact(id_: str, tokens: int) -> Fact:
    fact = Fact("Atlas", "is_a", "Sirket", {"description": "Atlas"}, kind="entity", subject_id=id_)
    base = f"[F1] {fact.render()}"
    padding = " ab" * (tokens - count_tokens(base))
    assert count_tokens(base + padding) == tokens
    return replace(fact, properties={"description": "Atlas" + padding})


@pytest.fixture
def hybrid(ontology: Ontology, fake_embedder: FakeEmbedder) -> HybridRetriever:
    repo = create_autospec(GraphRepository, instance=True)
    config = RetrievalConfig(context_token_budget=256)
    return HybridRetriever(repo, fake_embedder, ontology, config)


def _search(
    hybrid: HybridRetriever,
    *,
    hits: list[ChunkHit],
    facts: list[Fact],
    seed_hits: list[ChunkHit] | None = None,
    evidence: list[ChunkHit] | None = None,
) -> None:
    seeds = [EntityHit("atlas", "Atlas", "Sirket", 1.0)]
    hybrid.oag.linker.link = AsyncMock(return_value=seeds)
    hybrid.oag.subgraph = AsyncMock(return_value=Subgraph(seeds, facts, ["atlas", "related"]))
    hybrid.repo.search_chunks = AsyncMock(return_value=hits)
    hybrid.repo.seed_document_chunks = AsyncMock(return_value=seed_hits or [])
    hybrid.repo.evidence_chunks = AsyncMock(return_value=evidence or [])


async def test_hybrid_reserves_seed_and_rag_hits_ahead_of_higher_rrf_evidence(
    hybrid: HybridRetriever,
) -> None:
    hybrid.config = RetrievalConfig(context_token_budget=256, hybrid_text_reserve=2)
    overhead = sum(map(count_tokens, (f"{FACTS_HEADER}\n", f"{CHUNKS_HEADER}\n\n", "\n\n")))
    fact_budget = int((256 - overhead) * hybrid.config.fact_budget_ratio)
    fact = _fact("full", fact_budget - 1)
    more_facts = [_fact("extra", 25)]
    first, second = _hit("first", 74), _hit("second", 10)
    evidence = [_hit("evidence-1", 89), _hit("evidence-2", 83)]
    _search(
        hybrid,
        hits=[second, *evidence],
        seed_hits=[first, *evidence],
        facts=[fact, *more_facts],
        evidence=evidence,
    )

    kept = await hybrid.retrieve("Atlas")
    assert kept.chunks == [first, second]
    assert kept.citations[-2:] == [
        Citation("C1", "chunk", "first"),
        Citation("C2", "chunk", "second"),
    ]
    assert kept.facts == [fact]
    assert kept.token_count <= 256

    hybrid.config = RetrievalConfig(context_token_budget=256, hybrid_text_reserve=0)
    fused_order = await hybrid.retrieve("Atlas")
    assert fused_order.chunks == evidence[:1]
    assert fused_order.citations[-1] == Citation("C1", "chunk", "evidence-1")
    assert fused_order.facts == kept.facts
    assert fused_order.token_count <= 256


@pytest.mark.parametrize(
    ("reserve", "expected"),
    [
        (0, ["shared", "seed-first", "evidence"]),
        (2, ["seed-first", "shared", "evidence"]),
        (4, ["seed-first", "shared", "rag-second", "seed-last"]),
        (6, ["seed-first", "shared", "rag-second", "seed-last", "rag-last"]),
    ],
)
async def test_hybrid_interleaves_unique_reserved_hits(
    hybrid: HybridRetriever, reserve: int, expected: list[str]
) -> None:
    first, shared, last = (_hit(id_, 15) for id_ in ("seed-first", "shared", "seed-last"))
    hybrid.config = RetrievalConfig(context_token_budget=256, hybrid_text_reserve=reserve)
    _search(
        hybrid,
        hits=[shared, _hit("rag-second", 15), _hit("rag-last", 15)],
        seed_hits=[first, shared, last],
        evidence=[_hit("evidence", 15)],
        facts=[],
    )
    [vector] = await hybrid.embedder.embed(["Atlas"])

    context = await hybrid.retrieve("Atlas", vector=vector, k=3)

    assert [hit.chunk_id for hit in context.chunks] == expected
    hybrid.repo.seed_document_chunks.assert_awaited_once_with(["atlas"], vector, 3)
    hybrid.repo.evidence_chunks.assert_awaited_once_with(["atlas", "related"], vector, 3)


async def test_hybrid_shrinks_fact_allowance_for_125_token_hit(hybrid: HybridRetriever) -> None:
    hit = replace(_hit("reserved", 125), next_id="next", next_text="Neighbour" + " ab" * 30)
    big_fact, small_fact = _fact("big", 130), _fact("small", 110)
    assert pack_facts([big_fact, small_fact], budget=131).facts == [big_fact]
    _search(hybrid, hits=[hit], facts=[big_fact, small_fact])

    context = await hybrid.retrieve("Atlas")

    assert context.chunks == [hit]
    assert context.citations[-1] == Citation("C1", "chunk", "reserved")
    assert context.facts == [small_fact]
    assert "Neighbour" not in context.prompt_context
    assert context.token_count <= 256


async def test_hybrid_reserves_in_rank_order_when_both_hits_do_not_fit(
    hybrid: HybridRetriever,
) -> None:
    first, second, third = _hit("first", 180), _hit("second", 110), _hit("third", 25)
    fact = _fact("small", 25)
    _search(hybrid, hits=[second, third], seed_hits=[first], facts=[fact])

    context = await hybrid.retrieve("Atlas")

    assert context.chunks == [first, third]
    assert context.facts == [fact]
    assert context.token_count <= 256


@pytest.mark.parametrize("mode", ["hybrid", "rag"])
async def test_source_header_counts_against_256_token_limit(
    hybrid: HybridRetriever, mode: str
) -> None:
    too_large = _hit("too-large", 254)
    boundary = _hit("boundary", 256 - count_tokens(f"{CHUNKS_HEADER}\n\n") - 2)
    _search(hybrid, hits=[too_large, boundary], facts=[_fact("candidate", 25)])
    rag = RAGRetriever(hybrid.repo, hybrid.embedder, hybrid.config)

    context = await (hybrid if mode == "hybrid" else rag).retrieve("Atlas")

    assert context.chunks == [boundary]
    assert not context.facts
    assert context.citations == [Citation("C1", "chunk", "boundary")]
    assert context.prompt_context.startswith(f"{CHUNKS_HEADER}\n")
    assert context.token_count <= 256


async def test_hybrid_uses_facts_only_budget_when_no_chunk_fits(hybrid: HybridRetriever) -> None:
    fact_budget = int((256 - count_tokens(f"{FACTS_HEADER}\n")) * hybrid.config.fact_budget_ratio)
    fact = _fact("fits-only-without-text-header", fact_budget - 1)
    _search(hybrid, hits=[_hit("too-large", 254)], facts=[fact])

    context = await hybrid.retrieve("Atlas")

    assert context.facts == [fact]
    assert not context.chunks
    assert CHUNKS_HEADER not in context.prompt_context
    assert context.token_count <= 256


async def test_oag_fact_header_counts_against_256_token_limit(hybrid: HybridRetriever) -> None:
    too_large = _fact("too-large", 255)
    boundary = _fact("boundary", 256 - count_tokens(f"{FACTS_HEADER}\n") - 1)
    _search(hybrid, hits=[], facts=[too_large, boundary])
    context = await hybrid.oag.retrieve("Atlas")

    assert context.facts == [boundary]
    assert context.citations == [Citation("F1", "fact", "boundary")]
    assert context.prompt_context.startswith(f"{FACTS_HEADER}\n")
    assert context.token_count <= 256


async def test_hybrid_no_seeds_matches_rag(hybrid: HybridRetriever) -> None:
    hit = replace(_hit("source", 25), prev_id="previous", prev_text="Previous passage")
    _search(hybrid, hits=[hit], facts=[])
    hybrid.oag.linker.link = AsyncMock(return_value=[])
    rag = RAGRetriever(hybrid.repo, hybrid.embedder, hybrid.config)

    context = await hybrid.retrieve("Atlas")

    assert context == await rag.retrieve("Atlas")
    assert context.mode_used == "rag"
    assert context.chunks == [hit]
    assert "Previous passage" in context.prompt_context
    assert context.token_count <= 256
    hybrid.oag.subgraph.assert_not_awaited()
    hybrid.repo.seed_document_chunks.assert_not_awaited()
    hybrid.repo.evidence_chunks.assert_not_awaited()


@pytest.mark.parametrize(
    ("lexical", "rerank_candidates", "candidate_ids", "ranked_ids"),
    [
        (False, 1, ["a", "b"], ["b", "a"]),
        (False, 3, ["a", "b", "c"], ["c", "b"]),
        (True, 3, ["b", "a", "lexical"], ["lexical", "b"]),
    ],
)
async def test_rag_and_hybrid_rerank_the_same_candidates(
    hybrid: HybridRetriever,
    lexical: bool,
    rerank_candidates: int,
    candidate_ids: list[str],
    ranked_ids: list[str],
) -> None:
    k = 2
    hybrid.config = RetrievalConfig(
        rag_lexical=lexical, rerank_candidates=rerank_candidates, hybrid_text_reserve=2
    )
    a, b, c, text_hit, seed, evidence = (
        ChunkHit(id_, id_, f"{id_}.txt", id_, 1.0)
        for id_ in ("a", "b", "c", "lexical", "seed", "evidence")
    )
    _search(hybrid, hits=[a, b, c], facts=[], seed_hits=[seed], evidence=[evidence])
    vector_search = AsyncMock(side_effect=lambda vector, count: [a, b, c][:count])
    hybrid.repo.search_chunks = vector_search
    text_search = AsyncMock(return_value=[b, text_hit])
    hybrid.repo.search_chunks_text = text_search
    scores = {"a": 1.0, "b": 2.0, "c": 3.0, "lexical": 4.0}
    reranker = create_autospec(Reranker, instance=True)
    reranker.rerank.side_effect = lambda query, texts: [scores[text] for text in texts]
    hybrid.reranker = reranker
    rag = RAGRetriever(hybrid.repo, hybrid.embedder, hybrid.config, reranker=reranker)
    vector = [1.0, 0.0]

    rag_ctx = await rag.retrieve("Atlas", k=k, vector=vector)
    hybrid_ctx = await hybrid.retrieve("Atlas", k=k, vector=vector)

    assert [hit.chunk_id for hit in rag_ctx.chunks] == ranked_ids
    assert [hit.score for hit in rag_ctx.chunks] == [scores[id_] for id_ in ranked_ids]
    assert hybrid_ctx.chunks == [seed, rag_ctx.chunks[0]]
    assert reranker.rerank.await_count == 2
    assert [call.args for call in reranker.rerank.await_args_list] == [("Atlas", candidate_ids)] * 2
    assert vector_search.await_count == 2
    fetch_k = max(k, rerank_candidates) * (3 if lexical else 1)
    assert [call.args for call in vector_search.await_args_list] == [(vector, fetch_k)] * 2
    assert [call.args for call in text_search.await_args_list] == (
        [("Atlas", fetch_k)] * 2 if lexical else []
    )
    hybrid.repo.seed_document_chunks.assert_awaited_once_with(["atlas"], vector, k)
    hybrid.repo.evidence_chunks.assert_awaited_once_with(["atlas", "related"], vector, k)


async def test_rag_default_keeps_original_vector_hits(hybrid: HybridRetriever) -> None:
    hit = ChunkHit("vector", "doc", "source.txt", "Atlas", 0.8)
    _search(hybrid, hits=[hit], facts=[])
    rag = RAGRetriever(hybrid.repo, hybrid.embedder, RetrievalConfig())

    assert (await rag.retrieve("Atlas", k=1, vector=[1.0])).chunks == [hit]
    hybrid.repo.search_chunks.assert_awaited_once_with([1.0], 1)
    hybrid.repo.search_chunks_text.assert_not_awaited()


async def test_rag_text_weight_controls_fusion(hybrid: HybridRetriever) -> None:
    text_hit = ChunkHit("text", "text-doc", "text.txt", "Kervan Holding", 4.0)
    vector_hit = ChunkHit("vector", "vector-doc", "vector.txt", "Atlas", 0.8)
    hybrid.repo.search_chunks_text = AsyncMock(return_value=[text_hit])
    hybrid.repo.search_chunks = AsyncMock(return_value=[vector_hit])
    config = RetrievalConfig(rag_lexical=True)
    question = "Kervan Holding"
    vector = [1.0]

    equal = await search_rag_hits(hybrid.repo, question, vector, 2, config, None)
    assert [hit.chunk_id for hit in equal] == ["text", "vector"]
    assert [hit.score for hit in equal] == pytest.approx([1 / 61, 1 / 61])

    weighted = await search_rag_hits(
        hybrid.repo, question, vector, 2, config.model_copy(update={"rag_text_weight": 0.25}), None
    )
    assert [hit.chunk_id for hit in weighted] == ["vector", "text"]
    assert [hit.score for hit in weighted] == pytest.approx([1 / 61, 0.25 / 61])
