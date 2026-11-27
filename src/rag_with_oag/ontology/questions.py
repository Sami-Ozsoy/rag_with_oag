"""Route natural-language questions and check proposed graph paths."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict

from rag_with_oag.config import LLMConfig
from rag_with_oag.errors import MODEL_ENDPOINT_ERRORS
from rag_with_oag.generation import prompts
from rag_with_oag.llm import structured
from rag_with_oag.llm.client import ChatModel, LLMClient, Message
from rag_with_oag.ontology.model import Ontology

QUESTION_BATCH_SIZE = 20
CoverageStatus = Literal["covered", "not_covered", "invalid_path", "too_many_hops", "text"]
QuestionRoute = Literal["graph", "text"]


@dataclass(slots=True)
class QuestionCoverage:
    question: str
    status: CoverageStatus
    path: str | None
    hops: int | None
    reason: str
    route: QuestionRoute | None


@dataclass(slots=True)
class QuestionCoverageReport:
    ontology: str
    hops_limit: int
    questions: list[QuestionCoverage]
    token_usage: dict[str, int]

    @property
    def covered(self) -> int:
        return sum(item.status == "covered" for item in self.questions)

    @property
    def uncovered(self) -> list[QuestionCoverage]:
        return [item for item in self.questions if item.status not in ("covered", "text")]

    @property
    def text_only(self) -> list[QuestionCoverage]:
        return [item for item in self.questions if item.status == "text"]

    @property
    def counts(self) -> dict[CoverageStatus, int]:
        counts: dict[CoverageStatus, int] = dict.fromkeys(get_args(CoverageStatus), 0)
        for item in self.questions:
            counts[item.status] += 1
        return counts


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PathStep(_StrictModel):
    relation: str
    direction: Literal["forward", "backward"]


class PathAnswer(_StrictModel):
    type: str
    property: str | None


class _ProposedPath(_StrictModel):
    index: int
    route: QuestionRoute
    route_reason: str
    answerable: bool
    start_type: str | None
    steps: list[PathStep]
    answer: PathAnswer | None
    missing: str


class _BatchAnswers(_StrictModel):
    answers: list[_ProposedPath]


def validate_path(
    ontology: Ontology,
    start_type: str | None,
    steps: Sequence[PathStep],
    answer: PathAnswer | None,
    hops: int,
) -> tuple[CoverageStatus, str]:
    """Check the proposed path against the declared graph schema and hop limit."""
    current = ontology.entity(start_type) if start_type is not None else None
    if current is None:
        return "invalid_path", f"unknown start type: {start_type!r}"
    for index, step in enumerate(steps, start=1):
        relation = ontology.relation(step.relation)
        if relation is None:
            return "invalid_path", f"step {index}: unknown relation {step.relation!r}"
        source, target = relation.source, relation.target
        origin, destination = (source, target) if step.direction == "forward" else (target, source)
        if current.name != origin:
            return "invalid_path", (
                f"step {index}: {relation.name} connects {source} -> {target}, "
                f"cannot go {step.direction} from {current.name}"
            )
        next_type = ontology.entity(destination)
        assert next_type is not None  # Ontology validates relation endpoints.
        current = next_type
    if answer is None:
        return "invalid_path", "path has no answer"
    if answer.type != current.name:
        return (
            "invalid_path",
            f"answer type {answer.type!r} does not match path end {current.name!r}",
        )
    if answer.property not in (None, "name") and current.property(answer.property) is None:
        return "invalid_path", f"unknown property {answer.property!r} on {current.name}"
    if len(steps) > hops:
        return "too_many_hops", f"path needs {len(steps)} hops (limit {hops})"
    return "covered", ""


def _render_path(ontology: Ontology, proposal: _ProposedPath) -> str:
    parts = [proposal.start_type or "?"]
    for step in proposal.steps:
        relation = ontology.relation(step.relation)
        if step.direction == "forward":
            parts.append(f" -[{step.relation}]-> {relation.target if relation else '?'}")
        else:
            parts.append(f" <-[{step.relation}]- {relation.source if relation else '?'}")
    if proposal.answer is not None and proposal.answer.property not in (None, "name"):
        parts.append(f" .{proposal.answer.property}")
    return "".join(parts)


def _coverage(
    ontology: Ontology, question: str, proposal: _ProposedPath | None, hops: int
) -> QuestionCoverage:
    if proposal is None:
        return QuestionCoverage(
            question, "not_covered", None, None, "no answer from the model", None
        )

    status: CoverageStatus = "not_covered"
    reason = proposal.missing
    if proposal.answerable:
        status, reason = validate_path(
            ontology, proposal.start_type, proposal.steps, proposal.answer, hops
        )
    has_path = proposal.answerable
    if proposal.route == "text":
        if status != "covered":
            status, reason = "text", proposal.route_reason or proposal.missing
        has_path = (
            proposal.start_type is not None or bool(proposal.steps) or proposal.answer is not None
        )
    return QuestionCoverage(
        question=question,
        status=status,
        path=_render_path(ontology, proposal) if has_path else None,
        hops=len(proposal.steps) if has_path else None,
        reason=reason,
        route=proposal.route,
    )


def _normalize_questions(questions: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(text for question in questions if (text := question.strip())))


async def check_questions(
    ontology: Ontology,
    questions: Iterable[str],
    llm: LLMConfig | ChatModel,
    *,
    hops: int = 3,
    concurrency: int = 4,
) -> QuestionCoverageReport:
    """Route questions in batches and validate proposed paths against the ontology."""
    prepared = _normalize_questions(questions)
    if not prepared:
        raise ValueError("coverage check needs at least one question")
    for name, value in (("hops", hops), ("concurrency", concurrency)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1")

    description = ontology.describe()
    schema = _BatchAnswers.model_json_schema()
    semaphore = asyncio.Semaphore(concurrency)
    client = None
    if isinstance(llm, LLMConfig):
        llm = client = LLMClient(llm)
    model = llm
    usage_before = Counter(model.usage)

    async def check(batch: tuple[str, ...]) -> list[QuestionCoverage]:
        numbered = "\n".join(f"{index}. {question}" for index, question in enumerate(batch, 1))
        messages: list[Message] = [
            {"role": "system", "content": prompts.QUESTIONS_SYSTEM},
            {
                "role": "user",
                "content": prompts.QUESTIONS_USER.format(ontology=description, questions=numbered),
            },
        ]
        async with semaphore:
            result = await structured.generate(
                model, messages, schema=schema, model=_BatchAnswers, name="question_paths"
            )
        proposals: dict[int, _ProposedPath] = {}
        for proposal in result.value.answers:
            proposals.setdefault(proposal.index, proposal)
        return [
            _coverage(ontology, question, proposals.get(index), hops)
            for index, question in enumerate(batch, 1)
        ]

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(check(batch)) for batch in batched(prepared, QUESTION_BATCH_SIZE)
            ]
    except ExceptionGroup as group_error:
        error = next(
            (exc for exc in group_error.exceptions if isinstance(exc, MODEL_ENDPOINT_ERRORS)),
            group_error.exceptions[0],
        )
        raise error from error.__cause__
    finally:
        if client is not None:
            await client.aclose()

    return QuestionCoverageReport(
        ontology=ontology.name,
        hops_limit=hops,
        questions=[item for task in tasks for item in task.result()],
        token_usage=dict(Counter(model.usage) - usage_before),
    )


def _yaml_questions(node: object, seen: set[int]) -> Iterator[str]:
    if id(node) in seen or not isinstance(node, list | Mapping):
        return
    seen.add(id(node))
    if isinstance(node, list):
        for item in node:
            if isinstance(item, str):
                yield item
            else:
                yield from _yaml_questions(item, seen)
    elif "q" in node or "question" in node:
        # A question record can also contain metadata, such as answer keywords.
        for key, value in node.items():
            if key in ("q", "question"):
                if not isinstance(value, str):
                    raise ValueError(f"{key!r} must contain a question string")
                yield value
    else:
        for value in node.values():
            yield from _yaml_questions(value, seen)


def load_questions(path: str | Path) -> list[str]:
    """Read one question per text line or collect question entries from YAML."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            questions = _normalize_questions(_yaml_questions(yaml.safe_load(text), set()))
        except (yaml.YAMLError, ValueError) as exc:
            raise ValueError(f"{path}: invalid questions YAML: {exc}") from exc
        if not questions:
            raise ValueError(f"{path}: expected a YAML list of questions or 'q'/'question' entries")
    else:
        questions = _normalize_questions(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        )
        if not questions:
            raise ValueError(f"{path}: no questions found")
    return questions
