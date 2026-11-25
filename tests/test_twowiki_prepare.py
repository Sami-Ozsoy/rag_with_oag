"""Offline checks for the 2Wiki converter and distractor-aware answer scoring."""

from __future__ import annotations

import json
import re
import shlex
from hashlib import sha1, sha256
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from eval import run_golden
from eval.external.twowiki import prepare
from eval.synthetic.text import answer_first


def raw_question(group: str = "compositional", identifier: str = "q1") -> prepare.RawQuestion:
    return {
        "_id": identifier,
        "type": group,
        "question": "What was the date?",
        "context": [("Dates", ["  The date was 1 January 2000.  "])],
        "supporting_facts": [("Dates", 0)],
        "evidences": [["Event", "publication date", "1 January 2000"]],
        "answer": "1 January 2000",
        "answer_id": None,
    }


@pytest.fixture
def comparison() -> prepare.RawQuestion:
    return {
        **raw_question("comparison", "q3"),
        "question": "Which film came out first, Blind Shaft or The Mask Of Fu Manchu?",
        "evidences": [
            ["Blind Shaft", "publication date", "2003"],
            ["The Mask of Fu Manchu", "publication date", "1932"],
            ["BLIND SHAFT", "filming date", "2002"],
        ],
        "answer": "The Mask Of Fu Manchu",
        "answer_id": "Q1",
    }


@pytest.fixture
def bridge_comparison() -> prepare.RawQuestion:
    return {
        **raw_question("bridge_comparison", "q4"),
        "question": "Which film's director was born later, El Extraño Viaje or Amour À Paris?",
        "evidences": [
            ["El extraño viaje", "director", "Álvaro Ruiz"],
            ["Amour a Paris", "director", "Paul Scott"],
            ["ALVARO RUIZ", "date of birth", "1921"],
            ["Paul Scott", "date of birth", "1899"],
        ],
        "answer": "El Extraño Viaje",
    }


@pytest.mark.parametrize("answer", ["  YeS ", "NO"])
def test_exclude_yes_no(comparison: prepare.RawQuestion, answer: str) -> None:
    comparison["answer"] = answer
    assert prepare.eligibility(comparison) == ("yes_no", None)


def test_comparison_distractors(
    comparison: prepare.RawQuestion, bridge_comparison: prepare.RawQuestion
) -> None:
    assert prepare.eligibility(comparison) == (None, "Blind Shaft")
    assert prepare.eligibility(bridge_comparison) == (None, "Amour a Paris")
    assert prepare.eligibility(raw_question()) == (None, None)
    assert prepare.eligibility(raw_question("inference")) == (None, None)


def test_comparison_root_matches_alias(comparison: prepare.RawQuestion) -> None:
    comparison["answer"] = "The 1932 film"
    aliases: prepare.AliasRecord = {
        "Q_id": "Q1",
        "aliases": ["The Mask Of Fu Manchu"],
        "demonyms": [],
    }
    assert prepare.eligibility(comparison, aliases) == (None, "Blind Shaft")
    assert prepare.eligibility(comparison) == ("answer_root_count", None)
    aliases["aliases"].append("Blind Shaft")
    assert prepare.eligibility(comparison, aliases) == ("answer_root_count", None)


@pytest.mark.parametrize("roots", [1, 3])
def test_comparison_requires_two_roots(comparison: prepare.RawQuestion, roots: int) -> None:
    if roots == 1:
        comparison["evidences"] = comparison["evidences"][:1]
    else:
        comparison["evidences"].append(["Third Film", "publication date", "1990"])
    assert prepare.eligibility(comparison) == ("root_count", None)


def test_comparison_requires_distractor_in_question(comparison: prepare.RawQuestion) -> None:
    comparison["question"] = comparison["question"].replace("Blind Shaft", "Another Film")
    assert prepare.eligibility(comparison) == ("distractor_not_in_question", None)


@pytest.mark.parametrize(("answer", "other"), [("France", "New France"), ("New France", "France")])
def test_comparison_rejects_substrings(answer: str, other: str) -> None:
    question: prepare.RawQuestion = {
        **raw_question("comparison"),
        "question": f"Which is older, {answer} or {other}?",
        "answer": answer,
        "evidences": [[answer, "inception", "1500"], [other, "inception", "1600"]],
    }
    assert prepare.eligibility(question) == ("answer_distractor_overlap", None)


def test_answer_alternatives_filter_and_deduplicate_aliases() -> None:
    aliases: prepare.AliasRecord = {
        "Q_id": "Q16",
        "aliases": ["CA", "🇨🇦🇨🇦", "1234", None, "CANADA", "Dominion of Canada"],
        "demonyms": ["Canadian", "Canádian", "Canadien"],
    }
    assert prepare.answer_alternatives("Canada", aliases) == [
        "Canada",
        "Dominion of Canada",
        "Canadian",
        "Canadien",
    ]
    assert prepare.answer_alternatives("42") == ["42"]


def test_answer_alternatives_include_date_forms() -> None:
    assert prepare.answer_alternatives("07 March 2020") == [
        "07 March 2020",
        "March 7, 2020",
        "2020-03-07",
    ]
    assert prepare.answer_alternatives("4 May 999") == ["4 May 999", "May 4, 999", "0999-05-04"]
    assert prepare.answer_alternatives("7 Mart 2020") == ["7 Mart 2020"]


def test_document_filenames() -> None:
    titles = [" El Extraño—Viaje! ", "El Extrano Viaje", "東京", "北京", "A" * 85]
    filenames = [prepare.document_filename(title) for title in titles]
    assert len(set(filenames)) == len(titles)
    for title, filename, slug in zip(
        titles,
        filenames,
        ["el_extrano_viaje", "el_extrano_viaje", "doc", "doc", "a" * 60],
        strict=True,
    ):
        assert re.fullmatch(r"[a-z0-9_]{1,60}-[a-f0-9]{6}\.md", filename)
        assert filename == f"{slug}-{sha1(title.encode()).hexdigest()[:6]}.md"


def test_build_corpus_keeps_first_and_counts_conflicts() -> None:
    first = raw_question()
    first["context"] = [("Title", [" First sentence. ", "Second sentence."]), ("Blank", [" ", ""])]
    second = raw_question()
    second["context"] = [*first["context"], ("Title", ["Different."]), ("Other", ["Another."])]
    documents, conflicts = prepare.build_corpus([first, second])
    assert list(documents) == ["Title", "Other"]
    assert documents["Title"].body == "First sentence. Second sentence."
    assert conflicts == 1


def test_sampling_is_deterministic_by_seed_and_group() -> None:
    eligible = {
        group: [prepare.Candidate(raw_question(group, f"q{i:02d}"), None) for i in range(20)]
        for group in prepare.GROUPS
    }
    sample = prepare.sample_questions(eligible, 5, 2026)
    reversed_pools = {group: list(reversed(items)) for group, items in eligible.items()}
    assert prepare.sample_questions(reversed_pools, 5, 2026) == sample
    assert prepare.sample_questions(eligible, 5, 2027) != sample
    eligible["compositional"] = eligible["compositional"][1:]
    assert prepare.sample_questions(eligible, 5, 2026)["comparison"] == sample["comparison"]
    assert list(sample) == list(prepare.GROUPS)
    for items in sample.values():
        ids = [item.raw["_id"] for item in items]
        assert len(ids) == 5 and ids == sorted(ids)
    with pytest.raises(ValueError, match="compositional: requested 20 questions, only 19 eligible"):
        prepare.sample_questions(eligible, 20, 2026)


@pytest.fixture
def data_dir(
    tmp_path: Path, comparison: prepare.RawQuestion, bridge_comparison: prepare.RawQuestion
) -> Path:
    data = tmp_path / "raw data"
    data.mkdir()
    first = raw_question()
    first["context"].extend(
        [("Histoire", [" A little history. "]), ("Noise", ["An unrelated fact."])]
    )
    first["supporting_facts"] = [("Histoire", 0), ("Dates", 0), ("Histoire", 1)]
    rejected = raw_question("comparison", "q5")
    rejected["answer"] = "yes"
    questions = [bridge_comparison, comparison, raw_question("inference", "q2"), first, rejected]
    (data / "dev.json").write_text(json.dumps(questions), encoding="utf-8")
    entries = [
        {"Q_id": "Q1", "aliases": ["The Mask of Fu-Manchu"], "demonyms": []},
        {"Q_id": "Q2", "aliases": ["An unrelated alias"], "demonyms": []},
    ]
    (data / "id_aliases.json").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    return data


@pytest.fixture
def offline_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    def count(body: str) -> int:
        assert not body.startswith("#") and not body.endswith("\n")
        return 7

    monkeypatch.setattr(prepare, "count_tokens", count)


def test_prepare_set(tmp_path: Path, data_dir: Path, offline_tokens: None) -> None:
    out = tmp_path / "set output"
    args = ["--data", str(data_dir), "--out", str(out), "--per-type", "1", "--seed", "17"]
    prepare.main(args)

    assert sorted(path.name for path in out.iterdir()) == [
        "README.md",
        "docs",
        "questions.yaml",
        "stats.json",
    ]
    source = (out / "questions.yaml").read_text(encoding="utf-8")
    assert source.startswith("# ") and "distractor before the first accepted" in source
    assert "El Extraño Viaje" in source
    questions = yaml.safe_load(source)
    assert list(questions) == ["compositional", "inference", "comparison", "bridge_comparison"]
    for group, rows in questions.items():
        assert len(rows) == 1
        row = rows[0]
        keys = ["q", "docs", "answer"]
        if group in prepare.COMPARISONS:
            keys.append("distractors")
            assert row["distractors"] == [
                "Blind Shaft" if group == "comparison" else "Amour a Paris"
            ]
        assert list(row) == [*keys, "id", "hops", "evidences"]
        assert row["hops"] == len(row["evidences"])
        assert len(row["answer"]) == 1
        assert all((out / "docs" / name).is_file() for name in row["docs"])
    assert questions["comparison"][0]["answer"] == [
        ["The Mask Of Fu Manchu", "The Mask of Fu-Manchu"]
    ]
    assert questions["compositional"][0]["docs"] == [
        prepare.document_filename("Histoire"),
        prepare.document_filename("Dates"),
    ]
    docs = out / "docs"
    assert (
        docs / prepare.document_filename("Dates")
    ).read_text() == "# Dates\n\nThe date was 1 January 2000.\n"
    assert (docs / prepare.document_filename("Noise")).is_file()

    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats == {
        "questions_per_type": dict.fromkeys(questions, 1),
        "hops_per_type": {
            "compositional": {"1": 1},
            "inference": {"1": 1},
            "comparison": {"3": 1},
            "bridge_comparison": {"4": 1},
        },
        "documents_written": 3,
        "title_conflicts": 0,
        "total_characters": len("The date was 1 January 2000.A little history.An unrelated fact."),
        "total_tokens": 21,
        "relation_frequencies": {
            "publication date": 4,
            "filming date": 1,
            "director": 2,
            "date of birth": 2,
        },
        "excluded_counts": {"yes_no": 1},
        "seed": 17,
        "per_type": 1,
        "dev_sha256": sha256((data_dir / "dev.json").read_bytes()).hexdigest(),
    }
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert shlex.join(["python", "eval/external/twowiki/prepare.py", *args]) in readme
    assert all(text in readme for text in ("Ho et al.", "Apache-2.0", "CC BY-SA 4.0"))


def test_force_keeps_ontology_and_unrelated_files(
    tmp_path: Path, data_dir: Path, offline_tokens: None
) -> None:
    out = tmp_path / "set"
    docs = out / "docs"
    docs.mkdir(parents=True)
    stale = docs / "stale.md"
    stale.write_text("old")
    for name in ("questions.yaml", "stats.json", "README.md", "ontology.yaml", "notes.txt"):
        (out / name).write_text("old")

    args = ["--data", str(data_dir), "--out", str(out), "--per-type", "1"]
    with pytest.raises(SystemExit) as exc:
        prepare.main(args)
    assert exc.value.code == 2
    assert stale.read_text() == "old"
    assert (out / "questions.yaml").read_text() == "old"

    prepare.main([*args, "--force"])
    assert not stale.exists()
    assert (out / "ontology.yaml").read_text() == "old"
    assert (out / "notes.txt").read_text() == "old"
    for name in ("questions.yaml", "stats.json", "README.md"):
        assert (out / name).read_text() != "old"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("El Extraño Viaje. Amour À Paris came earlier.", True),
        ("Amour À Paris came earlier than El Extraño Viaje.", False),
        ("EL EXTRANO VIAJE", True),
        ("Amour À Paris", False),
        ("The director was born later.", False),
    ],
)
def test_answer_first(text: str, expected: bool) -> None:
    assert answer_first(text, [["El extraño viaje"]], ["Amour a Paris"]) is expected


def test_answer_first_checks_all_groups_and_finds_earliest_alternative() -> None:
    groups = [["First film", "Alternative title"], ["1921"]]
    distractors = ["Other film", "Unmentioned title"]
    assert answer_first("1921, Other film, First film", groups, distractors)
    assert answer_first("Alternative title, Other film, First film, 1921", groups, distractors)
    assert not answer_first("Alternative title, Other film", groups, distractors)


@pytest.mark.parametrize("distractors", [None, [], ["Other film"]])
async def test_runner_only_applies_distractors_to_answer_accuracy(
    distractors: list[str] | None,
) -> None:
    reply = "Other film came later than First film."
    kb = MagicMock(
        retrieve=AsyncMock(
            return_value=MagicMock(prompt_context=reply, mode_used="rag", token_count=12)
        ),
        query=AsyncMock(return_value=MagicMock(answer=reply)),
    )
    item: run_golden.QuestionItem = {"q": "Which film came out first?", "answer": [["First film"]]}
    if distractors is not None:
        item["distractors"] = distractors
    row = await run_golden.evaluate(kb, "comparison", item, "rag", 5, None, {}, answers=True)
    assert row.context_hit
    assert row.answer_ok is (not distractors)
