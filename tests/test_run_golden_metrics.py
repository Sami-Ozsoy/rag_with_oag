"""Offline scoring and reporting checks for the golden runner."""

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest
from eval.run_golden import EvalMode, QuestionItem, Row, Summary, evaluate, render, summary_data
from eval.synthetic.text import date_forms, is_refusal, matches

from rag_with_oag import RagOagError
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


@pytest.mark.parametrize("date", ["1973-11-21", "21 November 1973", "November 21, 1973"])
async def test_evidence_date_matches_packed_graph_fact(
    context: RetrievedContext, date: str
) -> None:
    fact = Fact("First Film", "release date", date)
    ctx = replace(context, facts=[fact], prompt_context=f"{FACTS_HEADER}\n[F1] {fact.render()}")
    item: QuestionItem = {
        "q": QUESTION,
        "answer": [["First Film"]],
        "evidences": [["First Film", "release date", "November 21, 1973"]],
    }

    row = await score(ctx, item)

    assert row.evidence_hit
    assert row.evidence_cov == 1.0


@pytest.mark.parametrize("text", ["01 November 1973", "November 1, 1973"])
def test_date_forms_canonicalize_textual_dates(text: str) -> None:
    assert date_forms(text) == ["1 November 1973", "November 1, 1973", "1973-11-01"]


def test_invalid_date_has_no_forms() -> None:
    assert date_forms("31 February 1990") == []


@pytest.mark.parametrize("apostrophe", list("’‘ʼ´`"))
def test_apostrophe_folding(apostrophe: str) -> None:
    assert matches(f"Charley{apostrophe}s Aunt", [["Charley's Aunt"]])


async def test_missing_annotations_and_empty_docs(context: RetrievedContext) -> None:
    item: QuestionItem = {"q": QUESTION, "answer": [["First Film"]]}
    missing = await score(context, item)
    empty_docs = await score(context, {**item, "docs": []})

    for row in (missing, empty_docs):
        assert row.refused is None
        assert row.evidence_hit is None
        assert row.evidence_cov is None
        assert row.doc_hit is None
        assert row.doc_recall is None


@pytest.mark.parametrize(
    "answer",
    [
        "I CANNOT determine which film came first.",
        "I can't tell.",
        "I can’t answer from the supplied facts.",
        "I can not say who was born first.",
        "I'm unable to answer.",
        "The earlier film cannot be determined.",
        "It is not possible to determine the earlier date.",
        "The facts do not include both birth dates.",
        "The source does not contain the release date.",
        "The passage did not specify the date.",
        "These sources don't state the dates.",
        "The passage doesn’t have a birth date.",
        "The release date is not provided.",
        "The dates are not available.",
        "There is not enough information.",
        "There is insufficient evidence.",
        "No facts about the birth date were found.",
        "No data regarding the release date was found.",
        "There is no evidence.",
        "\n \t**In v2.1, the release date is not provided.**",
    ],
)
def test_detects_refusal(answer: str) -> None:
    assert is_refusal(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Mark Shim was born later.",
        "B was born first [F1].",
        "The director was born in Paris [C1].",
    ],
)
def test_committed_answer_is_not_a_refusal(answer: str) -> None:
    assert not is_refusal(answer)


@pytest.mark.parametrize("separator", [". ", "! ", "? ", "\n"])
def test_refusal_limits_missing_details_to_the_first_sentence(separator: str) -> None:
    opening = f"First Film came first{separator}"
    assert not is_refusal(opening + "The sources do not give its exact date.")
    assert is_refusal(opening + "I can't determine its exact date.")


@pytest.mark.parametrize(
    "ending",
    [
        "The question cannot be answered.",
        "I am unable to conclude which film came first.",
        "It is not possible to tell which film came first.",
        "The records do not allow us to conclude which film came first.",
        "The records do not establish whether the first film came out earlier.",
        "There is insufficient evidence to compare the release dates.",
        "There is no way to tell.",
    ],
)
def test_refusal_detects_verdict_forms_after_the_first_sentence(ending: str) -> None:
    assert is_refusal(f"The records list the two films. {ending}")


@pytest.mark.parametrize(
    ("answer", "ignore"),
    [
        pytest.param(
            "The sources show that *Forbidden Dreams* was directed by Karel Kachyňa, who died on "
            "12 March 2004 [C1][C3]. *If I Were Free* was directed by Elliott Nugent, but the "
            "sources do not give his death date [C2]. Therefore, the sources do not allow a "
            "determination of which director died later.",
            ["Forbidden Dreams", "If I Were Free"],
            id="sources-do-not-allow-a-determination",
        ),
        pytest.param(
            "The evidence shows *The Strange Vengeance of Rosalie* was directed by Jack Starrett, "
            "who died on March 27, 1989 [F4]. *Father O'Nine* was directed by Roy Kellino [C2], "
            "but the evidence does not give Roy Kellino's death date. Therefore, it is not "
            "possible to determine which director died later.",
            ["The Strange Vengeance of Rosalie", "Father O'Nine"],
            id="not-possible-to-determine",
        ),
        pytest.param(
            "The facts only give the release year for *Bearing Straight* (1989) [F1]. There is no "
            "fact about *Me And You And The World*, so I can’t determine which album came out "
            "earlier.",
            ["Bearing Straight", "Me And You And The World"],
            id="cannot-determine-album-order",
        ),
        pytest.param(
            "The provided evidence does not determine this. It gives **High Tension**'s director "
            "Alexandre Aja's birth date as **7 August 1978** [F5][C3], but it does not give a "
            "birth date for **The Man in the Net**'s director Michael Curtiz [C2]. So I can't "
            "establish which director was born later from the evidence alone.",
            ["High Tension", "The Man in the Net"],
            id="does-not-determine-and-cannot-establish",
        ),
        pytest.param(
            "The sources identify the directors: *Six of a Kind* was directed by Leo McCarey "
            "[C3], and *Jack the Giant Slayer* by Bryan Singer, born September 17, 1965 [C1][C5]. "
            "However, the sources do not give Leo McCarey's birth date, so I cannot determine "
            "which director was born earlier.",
            ["Six of a Kind", "Jack the Giant Slayer"],
            id="cannot-determine-director-order",
        ),
        pytest.param(
            "The evidence identifies the directors as Wallace Grissell for *A Yank in Indo-China* "
            "[F2][C1] and Sergiu Nicolaescu for *The Ring (1985)* [C2]. It gives Grissell’s birth "
            "date as September 3, 1904 [F3][C5], but provides no birth date for Nicolaescu. "
            "Therefore, the evidence is insufficient to determine which director was born first.",
            ["A Yank in Indo-China", "The Ring (1985)"],
            id="insufficient-to-determine-director-order",
        ),
    ],
)
def test_refusal_detects_reported_verdicts(answer: str, ignore: list[str]) -> None:
    assert is_refusal(answer, ignore=ignore)


def test_refusal_ignores_all_literal_option_mentions_longest_first() -> None:
    answer = "**i cannot say (Farewell)** and I CANNOT SAY (FAREWELL) are two mentions."
    ignore = iter(("(Farewell)", "I Cannot Say (Farewell)", ""))
    assert not is_refusal(answer, ignore=ignore)


def test_refusal_removes_options_before_splitting_sentences() -> None:
    answer = "**Welcome to St. Tropez** is listed, but its exact date is not given."
    assert is_refusal(answer, ignore=["Welcome to St. Tropez"])


@pytest.mark.parametrize(
    ("answer", "gold", "distractors", "answer_ok", "refused"),
    [
        pytest.param(
            "The facts do not include First Film's release date, so I cannot determine "
            "which film came first. Second Film was released in 2001.",
            ["First Film"],
            ["Second Film"],
            False,
            True,
            id="choice-refusal-mentions-gold-first",
        ),
        pytest.param(
            "First Film was released earlier [F1].",
            ["First Film"],
            ["Second Film"],
            True,
            False,
            id="choice-commitment",
        ),
        pytest.param(
            "First Film was released earlier [F1], though I can’t determine the exact date.",
            ["First Film"],
            None,
            True,
            True,
            id="hedged-open-answer",
        ),
        pytest.param(
            'I Cannot Say "Farewell" was released earlier: it is a 1982 film [C1], …',
            ["Farewell", 'I Cannot Say "Farewell"'],
            ["The Darkest Minds"],
            True,
            False,
            id="choice-title-contains-refusal-phrase",
        ),
        pytest.param(
            "**Welcome to St. Tropez** appears to have been released first: it is from … album "
            '*"2011"* [C2], while "Down in the DM" was released on October 27, 2015 [C4]. '
            'The evidence does not give an exact release date for "Welcome to St. Tropez," '
            "so this is based on the album year.",
            ["Welcome to St. Tropez"],
            ["Down in the DM"],
            True,
            False,
            id="choice-commitment-with-trailing-hedge",
        ),
        pytest.param(
            "The provided facts do not include Napoleão Laureano's birth date, "
            "so I cannot determine who was born first. Giovanni Vastola was born on "
            "1938-04-20 [F1].",
            ["Napoleão Laureano"],
            ["Giovanni Vastola"],
            False,
            True,
            id="choice-birth-date-refusal",
        ),
        pytest.param(
            'First Film was released earlier than I Cannot Say "Farewell" [F1].',
            ["First Film"],
            ['I Cannot Say "Farewell"'],
            True,
            False,
            id="distractor-title-contains-refusal-phrase",
        ),
        pytest.param(
            'I Cannot Say "Farewell" was released in 1982 [C1].',
            ['I Cannot Say "Farewell"'],
            None,
            True,
            False,
            id="open-title-contains-refusal-phrase",
        ),
    ],
)
async def test_refusal_scoring(
    context: RetrievedContext,
    answer: str,
    gold: list[str],
    distractors: list[str] | None,
    answer_ok: bool,
    refused: bool,
) -> None:
    kb = MagicMock(
        retrieve=AsyncMock(return_value=context),
        query=AsyncMock(return_value=MagicMock(answer=answer)),
    )
    item: QuestionItem = {"q": QUESTION, "answer": [gold]}
    if distractors is not None:
        item["distractors"] = distractors

    row = await evaluate(kb, "comparison", item, "hybrid", 5, None, {}, answers=True)

    assert row.answer_ok is answer_ok
    assert row.refused is refused


async def test_answer_error_is_not_a_refusal(context: RetrievedContext) -> None:
    kb = MagicMock(
        retrieve=AsyncMock(return_value=context),
        query=AsyncMock(side_effect=RagOagError("Cannot answer: context window exceeded")),
    )
    item: QuestionItem = {"q": QUESTION, "answer": [["First Film"]]}

    row = await evaluate(kb, "comparison", item, "hybrid", 5, None, {}, answers=True)

    assert row.answer_ok is False
    assert row.refused is False


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
    assert "Refusals" not in report
    assert data["rows"][0]["refused"] is None
    assert data["answer_metrics"]["all"]["hybrid"] == {"answer_acc": None, "refusals": None}

    summary.rows[1] = replace(summary.rows[1], answer_ok=False, refused=True)
    summary.rows[2] = replace(
        summary.rows[2],
        answer="<error> model context window exceeded",
        answer_ok=False,
        refused=False,
    )
    report = render(summary, modes)
    data = summary_data(summary, modes)
    assert "M2 gate (answer acc, multi_hop): hybrid 0%" in report
    assert "| Answer acc. | Refusals | Context tokens |" in report
    assert "| 0% | 50% | 40 | 15 ms |" in report
    assert data["rows"][1]["refused"] is True
    assert data["rows"][2]["refused"] is False
    answer_metrics = data["answer_metrics"]
    assert answer_metrics["multi_hop"]["hybrid"] == {"answer_acc": 0.0, "refusals": 50.0}
    assert answer_metrics["all"]["hybrid"] == answer_metrics["multi_hop"]["hybrid"]
    assert answer_metrics["multi_hop"]["rag"] == {"answer_acc": None, "refusals": None}

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
