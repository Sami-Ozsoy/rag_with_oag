"""Prepare a balanced 2WikiMultiHopQA evaluation set from local dev data."""

from __future__ import annotations

import argparse
import json
import random
import re
import shlex
import shutil
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from hashlib import sha1, sha256
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

import yaml

from rag_with_oag._tokens import count_tokens

HERE = Path(__file__).resolve().parent

if TYPE_CHECKING or __package__:
    from ...synthetic.text import norm
else:
    sys.path.insert(0, str(HERE.parents[1]))
    from synthetic.text import norm

GROUPS = ("compositional", "inference", "comparison", "bridge_comparison")
COMPARISONS = {"comparison", "bridge_comparison"}
MONTHS = (
    "January February March April May June July August September October November December"
).split()
DATE_PATTERN = re.compile(r"(\d{1,2}) ([A-Z][a-z]+) (\d{3,4})")
QUESTION_HEADER = """\
# q is the question. docs lists the distinct documents cited by supporting facts.
# answer contains one group of accepted alternatives matched after normalization.
# distractors lists competing options. A distractor before the first accepted
# alternative makes the answer incorrect.
# id is the source question ID. hops counts evidences, the source triples.

"""


class RawQuestion(TypedDict):
    _id: str
    type: str
    question: str
    context: list[tuple[str, list[str]]]
    supporting_facts: list[tuple[str, int]]
    evidences: list[list[str]]
    answer: str
    answer_id: str | None


class AliasRecord(TypedDict):
    Q_id: str
    aliases: list[object]
    demonyms: list[object]


class QuestionRow(TypedDict):
    q: str
    docs: list[str]
    answer: list[list[str]]
    distractors: NotRequired[list[str]]
    id: str
    hops: int
    evidences: list[list[str]]


class Stats(TypedDict):
    questions_per_type: dict[str, int]
    hops_per_type: dict[str, dict[int, int]]
    documents_written: int
    title_conflicts: int
    total_characters: int
    total_tokens: int
    relation_frequencies: dict[str, int]
    excluded_counts: dict[str, int]
    seed: int
    per_type: int
    dev_sha256: str


@dataclass(frozen=True)
class Candidate:
    raw: RawQuestion
    distractor: str | None


@dataclass(frozen=True)
class Document:
    filename: str
    body: str


def eligibility(
    question: RawQuestion, names: AliasRecord | None = None
) -> tuple[str | None, str | None]:
    """Return the exclusion reason and, for eligible comparisons, the distractor."""
    answer = question["answer"]
    if answer.strip().lower() in {"yes", "no"}:
        return "yes_no", None
    if question["type"] not in COMPARISONS:
        return None, None

    evidences = question["evidences"]
    objects = {norm(obj) for _, _, obj in evidences}
    roots: dict[str, str] = {}
    for subject, _, _ in evidences:
        key = norm(subject)
        if key not in objects:
            roots.setdefault(key, subject)
    if len(roots) != 2:
        return "root_count", None

    answer_key = norm(answer)
    answer_names = {answer_key}
    if names is not None:
        answer_names.update(norm(alias) for alias in names["aliases"] if isinstance(alias, str))
    answer_roots = answer_names.intersection(roots)
    if len(answer_roots) != 1:
        return "answer_root_count", None

    distractor_key = next(key for key in roots if key not in answer_roots)
    if distractor_key not in norm(question["question"]):
        return "distractor_not_in_question", None
    if answer_key in distractor_key or distractor_key in answer_key:
        return "answer_distractor_overlap", None
    return None, roots[distractor_key]


def eligible_questions(
    questions: Iterable[RawQuestion], aliases: dict[str, AliasRecord]
) -> tuple[dict[str, list[Candidate]], Counter[str]]:
    groups: dict[str, list[Candidate]] = {group: [] for group in GROUPS}
    excluded: Counter[str] = Counter()
    for question in questions:
        reason, distractor = eligibility(question, aliases.get(question["answer_id"] or ""))
        if reason is None:
            groups[question["type"]].append(Candidate(question, distractor))
        else:
            excluded[reason] += 1
    return groups, excluded


def sample_questions(
    eligible: dict[str, list[Candidate]], per_type: int, seed: int
) -> dict[str, list[Candidate]]:
    def by_id(candidate: Candidate) -> str:
        return candidate.raw["_id"]

    sampled = {}
    for group in GROUPS:
        candidates = sorted(eligible[group], key=by_id)
        if len(candidates) < per_type:
            raise ValueError(
                f"{group}: requested {per_type} questions, only {len(candidates)} eligible"
            )
        selected = random.Random(f"{seed}:{group}").sample(candidates, per_type)
        sampled[group] = sorted(selected, key=by_id)
    return sampled


def date_variants(answer: str) -> list[str]:
    match = DATE_PATTERN.fullmatch(answer)
    if match is None:
        return []
    day, month, year = match.groups()
    if month not in MONTHS:
        return []
    try:
        value = date(int(year), MONTHS.index(month) + 1, int(day))
    except ValueError:
        return []
    return [f"{month} {value.day}, {year}", value.isoformat()]


def answer_alternatives(answer: str, names: AliasRecord | None = None) -> list[str]:
    alternatives = {norm(answer): answer}
    if names is not None:
        for value in (*names["aliases"], *names["demonyms"]):
            if isinstance(value, str) and len(value) >= 4 and any(c.isalpha() for c in value):
                alternatives.setdefault(norm(value), value)
    for value in date_variants(answer):
        alternatives.setdefault(norm(value), value)
    return list(alternatives.values())


def document_filename(title: str) -> str:
    decomposed = unicodedata.normalize("NFKD", title)
    folded = "".join(c for c in decomposed if not unicodedata.combining(c)).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", folded).strip("_")[:60].rstrip("_") or "doc"
    return f"{slug}-{sha1(title.encode()).hexdigest()[:6]}.md"


def build_corpus(questions: Iterable[RawQuestion]) -> tuple[dict[str, Document], int]:
    documents: dict[str, Document] = {}
    first_sentences: dict[str, list[str]] = {}
    conflicts = 0
    for question in questions:
        for title, sentences in question["context"]:
            body = " ".join(sentence.strip() for sentence in sentences)
            if not body.strip():
                continue
            if title in first_sentences:
                conflicts += sentences != first_sentences[title]
                continue
            first_sentences[title] = sentences
            documents[title] = Document(document_filename(title), body)
    assert len({doc.filename for doc in documents.values()}) == len(documents), (
        "document name collision"
    )
    return documents, conflicts


def question_row(
    candidate: Candidate, documents: dict[str, Document], aliases: dict[str, AliasRecord]
) -> QuestionRow:
    raw = candidate.raw
    titles = dict.fromkeys(title for title, _ in raw["supporting_facts"])
    names = aliases.get(raw["answer_id"] or "")
    row: QuestionRow = {
        "q": raw["question"],
        "docs": [documents[title].filename for title in titles],
        "answer": [answer_alternatives(raw["answer"], names)],
        "distractors": [candidate.distractor] if candidate.distractor is not None else [],
        "id": raw["_id"],
        "hops": len(raw["evidences"]),
        "evidences": raw["evidences"],
    }
    if not row["distractors"]:
        del row["distractors"]
    return row


def summarize(
    questions: dict[str, list[QuestionRow]],
    documents: dict[str, Document],
    conflicts: int,
    excluded: Counter[str],
    *,
    seed: int,
    per_type: int,
    dev_sha256: str,
) -> Stats:
    relations = Counter(
        relation
        for rows in questions.values()
        for row in rows
        for _, relation, _ in row["evidences"]
    )
    return {
        "questions_per_type": {group: len(rows) for group, rows in questions.items()},
        "hops_per_type": {
            group: dict(sorted(Counter(row["hops"] for row in rows).items()))
            for group, rows in questions.items()
        },
        "documents_written": len(documents),
        "title_conflicts": conflicts,
        "total_characters": sum(len(doc.body) for doc in documents.values()),
        "total_tokens": sum(count_tokens(doc.body) for doc in documents.values()),
        "relation_frequencies": dict(sorted(relations.items())),
        "excluded_counts": dict(sorted(excluded.items())),
        "seed": seed,
        "per_type": per_type,
        "dev_sha256": dev_sha256,
    }


def make_set(
    dev: bytes, aliases: dict[str, AliasRecord], per_type: int, seed: int
) -> tuple[dict[str, list[QuestionRow]], dict[str, Document], Stats]:
    eligible, excluded = eligible_questions(json.loads(dev), aliases)
    sampled = sample_questions(eligible, per_type, seed)
    documents, conflicts = build_corpus(item.raw for group in sampled.values() for item in group)
    questions = {
        name: [question_row(item, documents, aliases) for item in items]
        for name, items in sampled.items()
    }
    stats = summarize(
        questions,
        documents,
        conflicts,
        excluded,
        seed=seed,
        per_type=per_type,
        dev_sha256=sha256(dev).hexdigest(),
    )
    return questions, documents, stats


def readme(command: list[str], stats: Stats) -> str:
    counts = stats["questions_per_type"]
    by_type = ", ".join(f"{count} {group}" for group, count in counts.items())
    return "\n".join(
        [
            "# 2WikiMultiHopQA development set",
            "",
            "Sampled from [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) "
            "(Ho et al., COLING 2020), `data_ids_april7.zip`.",
            "",
            "```bash",
            shlex.join(command),
            "```",
            "",
            f"- {sum(counts.values())} questions ({by_type})",
            f"- {stats['documents_written']:,} documents and {stats['title_conflicts']:,} "
            "title conflicts",
            f"- {stats['total_characters']:,} characters and {stats['total_tokens']:,} tokens "
            "in the paragraph bodies",
            f"- {sum(stats['excluded_counts'].values()):,} source questions excluded",
            "",
            "The dataset is licensed under Apache-2.0. Paragraph text comes from Wikipedia "
            "and is licensed under CC BY-SA 4.0.",
            "",
        ]
    )


def read_aliases(path: Path) -> dict[str, AliasRecord]:
    aliases = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                entry = cast(AliasRecord, json.loads(line))
                aliases[entry["Q_id"]] = entry
    return aliases


def write_set(
    out: Path,
    questions: dict[str, list[QuestionRow]],
    documents: dict[str, Document],
    stats: Stats,
    command: list[str],
    *,
    force: bool,
) -> None:
    docs = out / "docs"
    if force:
        if docs.exists() or docs.is_symlink():
            shutil.rmtree(docs)
        for name in ("questions.yaml", "stats.json", "README.md"):
            (out / name).unlink(missing_ok=True)
    docs.mkdir(parents=True)
    for title, doc in documents.items():
        (docs / doc.filename).write_text(
            f"# {title}\n\n{doc.body}\n", encoding="utf-8", newline="\n"
        )
    (out / "questions.yaml").write_text(
        QUESTION_HEADER + yaml.safe_dump(questions, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
        newline="\n",
    )
    (out / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (out / "README.md").write_text(readme(command, stats), encoding="utf-8", newline="\n")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "sets" / "dev500")
    parser.add_argument("--per-type", type=int, default=125)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    values = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(values)
    command = (
        [sys.executable, *sys.orig_argv[1:]]
        if argv is None
        else ["python", "eval/external/twowiki/prepare.py", *values]
    )

    try:
        if args.out.exists() and any(args.out.iterdir()) and not args.force:
            raise ValueError(f"output is not empty: {args.out} (use --force)")
        dev = (args.data / "dev.json").read_bytes()
        aliases = read_aliases(args.data / "id_aliases.json")
        questions, documents, stats = make_set(dev, aliases, args.per_type, args.seed)
        write_set(args.out, questions, documents, stats, command, force=args.force)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Prepared {len(documents):,} documents, {sum(map(len, questions.values())):,} questions, "
        f"{stats['total_tokens']:,} tokens in {args.out}"
    )


if __name__ == "__main__":
    main()
