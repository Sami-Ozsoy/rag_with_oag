"""Offline scoring and reporting checks for the golden runner."""

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest
from eval.run_golden import EvalMode, QuestionItem, Row, Summary, evaluate, render, summary_data

from rag_with_oag.retrieval.context import CHUNKS_HEADER, FACTS_HEADER
from rag_with_oag.types import ChunkHit, Citation, Fact, RetrievedContext

QUESTION = "Which film was released earlier?"


@pytest.fixture
def context() -> RetrievedContext:
    fact = Fact("First Film", "release date", "1 January 2000")
    first = ChunkHit("first", "hash-1", r"C:\corpus\first.md", "First Film.", 1)
    second = ChunkHit("second", "/corpus/second.md", "Legacy title", "İZMİR\nFESTIVAL.", 1)
    unpacked = ChunkHit("unpacked", "hash-3", "/corpus/unpacked.md", "1999", 1)
    return RetrievedContext(
        question=QUESTION,
        mode_used="hybrid",
        chunks=[first, second, unpacked],
        facts=[fact],
        citations=[
            Citation("F1", "fact", "unpacked"),
            Citation("C1", "chunk", first.chunk_id),
            Citation("C2", "chunk", second.chunk_id),
        ],
        prompt_context=(
            f"{FACTS_HEADER}\n[F1] {fact.render()}\n\n"
            f"{CHUNKS_HEADER}\n\n[C1] {first.source}\n{first.text}\n\n"
            f"[C2] {second.source}\n{second.text}"
        ),
        token_count=40,
    )


async def score(ctx: RetrievedContext, item: QuestionItem) -> Row:
    kb = MagicMock(retrieve=AsyncMock(return_value=ctx), query=AsyncMock())
    row = await evaluate(kb, "multi_hop", item, "hybrid", 5, None, {})
    kb.query.assert_not_awaited()
    return row


@pytest.mark.parametrize("include_unpacked", [False, True])
async def test_scores_packed_facts_and_cited_passages(
    context: RetrievedContext, include_unpacked: bool
) -> None:
    item: QuestionItem = {
        "q": QUESTION,
        "answer": [["First Film"]],
        "evidences": [
            ["First Film", "release date", "1 January 2000"],
            ["Second Film", "premiere", "İzmir Festival"],
            ["First Film", "premiere", "İzmir Festival"],
        ],
        "docs": ["first.md", "second.md", "first.md"],
    }
    if include_unpacked:
        item["evidences"].append(["Second Film", "release date", "1999"])
        item["docs"].append("unpacked.md")

    row = await score(context, item)

    assert row.context_hit
    assert row.evidence_hit is (not include_unpacked)
    assert row.doc_hit is (not include_unpacked)
    assert row.evidence_cov == row.doc_recall == (2 / 3 if include_unpacked else 1.0)


async def test_missing_annotations_and_empty_docs(context: RetrievedContext) -> None:
    item: QuestionItem = {"q": QUESTION, "answer": [["First Film"]]}
    missing = await score(context, item)
    empty_docs = await score(context, {**item, "docs": []})

    for row in (missing, empty_docs):
        assert row.evidence_hit is None
        assert row.evidence_cov is None
        assert row.doc_hit is None
        assert row.doc_recall is None


def test_aggregates_and_reports_available_metrics() -> None:
    hybrid = Row("multi_hop", "hybrid", QUESTION, "hybrid", True, 40, 15)
    rag = replace(hybrid, mode="rag", mode_used="rag")
    summary = Summary(
        model={"llm": "fake", "extractor": "fake", "embed": "fake"},
        ingest={"processed": 2, "seconds": 1, "stats": {}, "validation_errors": 0, "failures": []},
        rows=[
            hybrid,
            replace(hybrid, evidence_hit=True, evidence_cov=1.0, doc_hit=False, doc_recall=0.5),
            replace(hybrid, evidence_hit=False, evidence_cov=0.5),
            replace(rag, evidence_hit=False, evidence_cov=0.0),
            replace(rag, group="single_hop", doc_hit=True, doc_recall=1.0),
        ],
        groups=["multi_hop", "single_hop"],
    )
    modes: tuple[EvalMode, ...] = ("rag", "hybrid")
    report = render(summary, modes)
    data = summary_data(summary, modes)
    metrics = data["retrieval_metrics"]

    assert data["rows"][0]["doc_recall"] is None
    assert data["rows"][1]["evidence_hit"] is True
    assert data["rows"][1]["doc_recall"] == 0.5
    assert metrics["multi_hop"]["hybrid"] == {
        "evidence_hit": 50.0,
        "evidence_cov": 75.0,
        "doc_hit": 0.0,
        "doc_recall": 50.0,
    }
    assert metrics["multi_hop"]["rag"]["doc_recall"] is None
    assert metrics["all"]["rag"]["doc_recall"] == 100.0
    assert "| Evidence hit | Evidence coverage | Doc hit | Doc recall |" in report
    assert "M2 gate (evidence hit, multi_hop): hybrid 50% vs rag 0%" in report

    summary.rows[1].answer_ok = False
    assert "M2 gate (answer acc, multi_hop): hybrid 0%" in render(summary, modes)

    summary.rows = [hybrid, rag, summary.rows[-1]]
    report = render(summary, modes)
    assert "Evidence hit" not in report
    assert "| Context hit | Doc hit | Doc recall | Answer acc. |" in report
    assert "M2 gate (context hit, multi_hop)" in report

    summary.rows = [replace(hybrid, chain_hit=True, chain_cov=1.0), rag]
    report = render(summary, modes)
    assert "| Chain hit | Chain coverage | Answer acc. |" in report
    assert "Doc recall" not in report
    assert "M2 gate (chain hit, multi_hop)" in report
