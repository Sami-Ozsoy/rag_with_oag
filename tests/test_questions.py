"""Deterministic question coverage and CLI tests with offline model replies."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rag_with_oag import (
    LLMConfig,
    ModelAuthError,
    Ontology,
    QuestionCoverage,
    StructuredOutputError,
    cli,
)
from rag_with_oag.llm import capabilities
from rag_with_oag.llm.client import Message
from rag_with_oag.ontology import questions
from rag_with_oag.ontology.questions import PathAnswer, PathStep, load_questions, validate_path

from .conftest import FakeLLM


@pytest.fixture
def ontology() -> Ontology:
    return Ontology.from_yaml("""
name: contracts
entities:
  Sirket: {properties: {sehir: string}}
  Sozlesme: {properties: {numara: string}}
relations:
  ORTAGIDIR: {source: Sirket, target: Sirket}
  TARAFIDIR: {source: Sozlesme, target: Sirket}
""")


OWNER = PathStep(relation="ORTAGIDIR", direction="forward")
PARTY = PathStep(relation="TARAFIDIR", direction="forward")
CONTRACT = PathStep(relation="TARAFIDIR", direction="backward")


@pytest.mark.parametrize(
    ("start", "steps", "end", "property_", "limit", "status", "reason"),
    [
        ("Sozlesme", [PARTY], "Sirket", None, 3, "covered", ""),
        ("Sirket", [CONTRACT], "Sozlesme", "name", 3, "covered", ""),
        ("Sirket", [OWNER, CONTRACT], "Sozlesme", "numara", 2, "covered", ""),
        ("Sirket", [], "Sirket", "sehir", 3, "covered", ""),
        ("Other", [], "Other", None, 3, "invalid_path", "unknown start type: 'Other'"),
        (
            "Sirket",
            [PathStep(relation="MISSING", direction="forward")],
            "Sirket",
            None,
            3,
            "invalid_path",
            "step 1: unknown relation 'MISSING'",
        ),
        (
            "Sirket",
            [CONTRACT, OWNER],
            "Sirket",
            None,
            1,
            "invalid_path",
            "step 2: ORTAGIDIR connects Sirket -> Sirket, cannot go forward from Sozlesme",
        ),
        (
            "Sozlesme",
            [CONTRACT],
            "Sozlesme",
            None,
            3,
            "invalid_path",
            "step 1: TARAFIDIR connects Sozlesme -> Sirket, cannot go backward from Sozlesme",
        ),
        (
            "Sirket",
            [],
            "Sozlesme",
            None,
            3,
            "invalid_path",
            "answer type 'Sozlesme' does not match path end 'Sirket'",
        ),
        (
            "Sirket",
            [],
            "Sirket",
            "numara",
            3,
            "invalid_path",
            "unknown property 'numara' on Sirket",
        ),
        (
            "Sirket",
            [OWNER, CONTRACT],
            "Sozlesme",
            "numara",
            1,
            "too_many_hops",
            "path needs 2 hops (limit 1)",
        ),
    ],
)
def test_validate_path(
    ontology: Ontology,
    start: str,
    steps: list[PathStep],
    end: str,
    property_: str | None,
    limit: int,
    status: str,
    reason: str,
) -> None:
    answer = PathAnswer(type=end, property=property_)
    assert validate_path(ontology, start, steps, answer, limit) == (status, reason)


def _proposal(index: int, **fields: Any) -> dict[str, Any]:
    return {
        "index": index,
        "route": "graph",
        "route_reason": "The question needs a relation between entities.",
        "answerable": True,
        "start_type": "Sirket",
        "steps": [],
        "answer": {"type": "Sirket", "property": "name"},
        "missing": "",
    } | fields


def _missing(index: int, reason: str) -> dict[str, Any]:
    return _proposal(index, answerable=False, start_type=None, answer=None, missing=reason)


class QueuedLLM(FakeLLM):
    def __init__(self, *replies: dict[str, Any] | Exception) -> None:
        super().__init__()
        self.model = "fake-questions"
        self.replies = deque(replies)
        self.options: list[dict[str, Any]] = []
        self.closed = False

    async def complete(self, messages: Sequence[Message], **kwargs: Any) -> str:
        self.options.append(kwargs)
        reply = self.replies.popleft()
        if isinstance(reply, Exception):
            raise reply
        self.reply = json.dumps(reply)
        return await super().complete(messages, **kwargs)

    async def aclose(self) -> None:
        self.closed = True


def test_report(ontology: Ontology) -> None:
    llm = QueuedLLM(
        {
            "answers": [
                _proposal(7),
                _proposal(2, steps=[PathStep(relation="OTHER", direction="forward").model_dump()]),
                _proposal(
                    1,
                    steps=[OWNER.model_dump(), CONTRACT.model_dump()],
                    answer={"type": "Sozlesme", "property": "numara"},
                ),
                _proposal(1, start_type="Duplicate"),
                _proposal(3),
                _missing(4, "Needs a payment property"),
                _proposal(0),
            ]
        }
    )
    llm.usage.update(prompt_tokens=60, completion_tokens=30, old_tokens=1)
    report = ontology.check_questions(
        ["  Q1?", "Q2?", "", "Q1? ", "Q3?", "Q4?", "Q5?"], llm, hops=1
    )
    assert report.ontology == "contracts"
    assert report.hops_limit == 1
    assert report.questions == [
        QuestionCoverage(
            "Q1?",
            "too_many_hops",
            "Sirket -[ORTAGIDIR]-> Sirket <-[TARAFIDIR]- Sozlesme .numara",
            2,
            "path needs 2 hops (limit 1)",
            "graph",
        ),
        QuestionCoverage(
            "Q2?",
            "invalid_path",
            "Sirket -[OTHER]-> ?",
            1,
            "step 1: unknown relation 'OTHER'",
            "graph",
        ),
        QuestionCoverage("Q3?", "covered", "Sirket", 0, "", "graph"),
        QuestionCoverage("Q4?", "not_covered", None, None, "Needs a payment property", "graph"),
        QuestionCoverage("Q5?", "not_covered", None, None, "no answer from the model", None),
    ]
    assert report.covered == 1
    assert report.uncovered == report.questions[:2] + report.questions[3:]
    assert report.text_only == []
    assert report.counts == {
        "covered": 1,
        "not_covered": 2,
        "invalid_path": 1,
        "too_many_hops": 1,
        "text": 0,
    }
    assert report.token_usage == {"prompt_tokens": 100, "completion_tokens": 10}
    assert llm.requests[0][1]["content"].endswith("1. Q1?\n2. Q2?\n3. Q3?\n4. Q4?\n5. Q5?")
    assert ontology.describe() in llm.requests[0][1]["content"]
    output = llm.options[0]["response_format"]["json_schema"]
    assert output["strict"] is True
    for schema in [output["schema"], *output["schema"]["$defs"].values()]:
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
    assert not llm.closed


async def test_text_routing_keeps_valid_paths_and_excludes_text_from_gaps(
    ontology: Ontology,
) -> None:
    detail = "The answer is a single detail stated in one passage."
    llm = QueuedLLM(
        {
            "answers": [
                _proposal(
                    1,
                    route="text",
                    route_reason=detail,
                    answer={"type": "Sirket", "property": "sehir"},
                ),
                _missing(2, "Needs a deposit property") | {"route": "text", "route_reason": detail},
                _proposal(3, route="text", route_reason=detail, start_type="Unknown"),
                _proposal(
                    4,
                    route="text",
                    route_reason=detail,
                    steps=[OWNER.model_dump(), CONTRACT.model_dump()],
                    answer={"type": "Sozlesme", "property": "numara"},
                ),
                _missing(5, "Needs a duration property") | {"route": "text", "route_reason": ""},
                _proposal(6, route="text", route_reason=detail, start_type=None, answer=None),
                _proposal(7, route="text", route_reason=detail, answerable=False),
                _missing(8, "Needs a supplier relation"),
            ]
        }
    )
    report = await ontology.acheck_questions([f"Q{i}?" for i in range(1, 9)], llm, hops=1)
    assert report.questions == [
        QuestionCoverage("Q1?", "covered", "Sirket .sehir", 0, "", "text"),
        QuestionCoverage("Q2?", "text", None, None, detail, "text"),
        QuestionCoverage("Q3?", "text", "Unknown", 0, detail, "text"),
        QuestionCoverage(
            "Q4?",
            "text",
            "Sirket -[ORTAGIDIR]-> Sirket <-[TARAFIDIR]- Sozlesme .numara",
            2,
            detail,
            "text",
        ),
        QuestionCoverage("Q5?", "text", None, None, "Needs a duration property", "text"),
        QuestionCoverage("Q6?", "text", None, None, detail, "text"),
        QuestionCoverage("Q7?", "text", "Sirket", 0, detail, "text"),
        QuestionCoverage("Q8?", "not_covered", None, None, "Needs a supplier relation", "graph"),
    ]
    assert report.covered == 1
    assert report.text_only == report.questions[1:7]
    assert report.uncovered == [report.questions[7]]
    assert report.counts == {
        "covered": 1,
        "not_covered": 1,
        "invalid_path": 0,
        "too_many_hops": 0,
        "text": 6,
    }
    system = llm.requests[0][0]["content"]
    assert "Route each question as graph or text" in system
    assert "still propose a path when the ontology already expresses one" in system


async def test_batching_preserves_order_with_concurrent_replies(ontology: Ontology) -> None:
    first_can_finish = asyncio.Event()
    finished = []

    class BatchLLM(QueuedLLM):
        running = 0
        peak = 0

        async def complete(self, messages: Sequence[Message], **kwargs: Any) -> str:
            self.running += 1
            self.peak = max(self.peak, self.running)
            await asyncio.sleep(0)
            content = messages[1]["content"]
            if "1. Question 01?" in content:
                await first_can_finish.wait()
                finished.append(1)
            elif "1. Question 21?" in content:
                assert content.endswith("20. Question 40?")
                finished.append(21)
            else:
                assert content.endswith("5. Question 45?")
                finished.append(41)
                first_can_finish.set()
            self.running -= 1
            return await super().complete(messages, **kwargs)

    llm = BatchLLM(
        {"answers": [_missing(20, "second batch")]},
        {"answers": [_missing(5, "third batch")]},
        {"answers": [_missing(20, "first batch")]},
    )
    inputs = [f"Question {index:02}?" for index in range(1, 46)]
    report = await asyncio.wait_for(ontology.acheck_questions(iter(inputs), llm, concurrency=2), 5)
    assert llm.peak == 2
    assert finished == [21, 41, 1]
    assert [item.question for item in report.questions] == inputs
    assert [report.questions[index].reason for index in (19, 39, 44)] == [
        "first batch",
        "second batch",
        "third batch",
    ]
    assert report.counts == {
        "covered": 0,
        "not_covered": 45,
        "invalid_path": 0,
        "too_many_hops": 0,
        "text": 0,
    }
    assert report.token_usage == {"prompt_tokens": 300, "completion_tokens": 30}


@pytest.mark.parametrize("fail", [False, True])
async def test_config_client_closes(
    ontology: Ontology, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    config = LLMConfig(model="stub")
    failure = ModelAuthError("invalid credentials")
    llm = QueuedLLM(failure if fail else {"answers": [_proposal(1)]})

    def make_client(received: LLMConfig) -> QueuedLLM:
        assert received is config
        return llm

    monkeypatch.setattr(questions, "LLMClient", make_client)
    if fail:
        with pytest.raises(ModelAuthError) as caught:
            await ontology.acheck_questions(["Q?"], config)
        assert caught.value is failure
    else:
        assert (await ontology.acheck_questions(["Q?"], config)).covered == 1
    assert llm.closed


async def test_structured_retry_without_retrying_invalid_paths(ontology: Ontology) -> None:
    llm = QueuedLLM({"answers": None}, {"answers": [_proposal(1, answer=None)]})
    report = await ontology.acheck_questions(["Q?"], llm)
    assert report.questions[0].status == "invalid_path"
    assert report.questions[0].reason == "path has no answer"
    assert len(llm.requests) == 2
    assert "That JSON has problems:" in llm.requests[1][-1]["content"]


async def test_schema_errors_propagate(ontology: Ontology) -> None:
    llm = QueuedLLM(*[{"answers": None}] * 3)
    capabilities.remember(llm, capabilities.StructuredMethod.JSON_SCHEMA)
    try:
        with pytest.raises(StructuredOutputError):
            await ontology.acheck_questions(["Q?"], llm)
    finally:
        capabilities.forget(llm)


@pytest.mark.parametrize(
    ("items", "hops", "concurrency", "message"),
    [([], 3, 4, "at least one question"), (["Q?"], 0, 4, "hops"), (["Q?"], 3, 0, "concurrency")],
)
async def test_invalid_input(
    ontology: Ontology, items: list[str], hops: int, concurrency: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await questions.check_questions(
            ontology, items, QueuedLLM(), hops=hops, concurrency=concurrency
        )


async def test_sync_check_requires_no_running_loop(ontology: Ontology) -> None:
    with pytest.raises(RuntimeError, match="acheck_questions"):
        ontology.check_questions(["Q?"], QueuedLLM())


@pytest.mark.parametrize(
    ("suffix", "content"),
    [
        (".txt", " # Title\n Q1?  \n\nQ2?\nQ1?\n"),
        (".yaml", '[" Q1? ", "Q2?", "Q1?", ""]'),
        (".yml", 'group: {nested: [" Q1? ", {question: "Q2?"}, {q: "Q1?"}]}'),
        (".yaml", '{question: "Q1?", q: "Q2?"}'),
    ],
)
def test_load_questions(tmp_path: Path, suffix: str, content: str) -> None:
    path = tmp_path / f"questions{suffix}"
    path.write_text(content, encoding="utf-8")
    assert load_questions(str(path)) == ["Q1?", "Q2?"]


def test_load_golden_questions() -> None:
    path = Path(__file__).resolve().parents[1] / "eval/golden/questions.yaml"
    loaded = load_questions(path)
    assert len(loaded) == 33
    assert loaded[0] == "HS-2024-01 sözleşmesinin aylık bedeli nedir?"
    assert "Kuzey Yatırım'ın 2023 yılı net kârı ne kadardır?" in loaded


@pytest.mark.parametrize(
    ("suffix", "content", "message"),
    [
        ("txt", "\n# Empty\n", "no questions"),
        ("yaml", "question: [one, two]", "question string"),
        ("yaml", "question: [", "invalid questions YAML"),
        ("yaml", "metadata: 3", "expected a YAML list"),
        ("yaml", '"Q?"', "expected a YAML list"),
    ],
)
def test_invalid_question_file(tmp_path: Path, suffix: str, content: str, message: str) -> None:
    path = tmp_path / f"questions.{suffix}"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_questions(path)


@pytest.fixture
def cli_args(ontology: Ontology, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    ontology_path = tmp_path / "ontology.yaml"
    ontology.to_yaml(ontology_path)
    question_path = tmp_path / "questions.txt"
    question_path.write_text("Q1?\nQ2?\nQ3?\n", encoding="utf-8")
    monkeypatch.setenv("RAG_OAG_LLM_MODEL", "stub")
    monkeypatch.setattr(cli, "_open", lambda args: pytest.fail("opened a knowledge base"))
    return ["check-questions", "--ontology", str(ontology_path), "--questions", str(question_path)]


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_output(
    cli_args: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    reason = "The deposit is stated in one passage."
    llm = QueuedLLM(
        {
            "answers": [
                _proposal(1),
                _missing(2, "Needs a deposit property") | {"route": "text", "route_reason": reason},
                _missing(3, "Needs a City type"),
            ]
        }
    )
    monkeypatch.setattr(cli, "_make_llm", lambda config: llm)
    flags = ["--hops", "2", "--concurrency", "1", "-v"]
    if json_output:
        flags.append("--json")
    assert cli.main(cli_args + flags) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert llm.closed
    if json_output:
        assert json.loads(captured.out) == {
            "ontology": "contracts",
            "hops_limit": 2,
            "questions": [
                {
                    "question": "Q1?",
                    "status": "covered",
                    "path": "Sirket",
                    "hops": 0,
                    "reason": "",
                    "route": "graph",
                },
                {
                    "question": "Q2?",
                    "status": "text",
                    "path": None,
                    "hops": None,
                    "reason": reason,
                    "route": "text",
                },
                {
                    "question": "Q3?",
                    "status": "not_covered",
                    "path": None,
                    "hops": None,
                    "reason": "Needs a City type",
                    "route": "graph",
                },
            ],
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 10},
        }
    else:
        assert captured.out.splitlines() == [
            "Checked 3 questions against ontology 'contracts' (hops limit 2): "
            "1 covered, 1 answered from text, 1 gaps.",
            "  covered        Q1?",
            "    Sirket",
            "  text           Q2?",
            f"    {reason}",
            "  not covered    Q3?",
            "    Needs a City type",
            "Tokens: prompt 100, completion 10",
        ]


@pytest.mark.parametrize("bad_input", ["ontology", "questions", "settings"])
def test_cli_input_failure(
    cli_args: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    bad_input: str,
) -> None:
    monkeypatch.setattr(cli, "_make_llm", lambda config: pytest.fail("created a model client"))
    if bad_input == "ontology":
        Path(cli_args[2]).write_text("[", encoding="utf-8")
    elif bad_input == "questions":
        Path(cli_args[4]).unlink()
    else:
        monkeypatch.delenv("RAG_OAG_LLM_MODEL")
    assert cli.main(cli_args) == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_cli_model_failure(
    cli_args: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = QueuedLLM(ModelAuthError("invalid credentials"))
    monkeypatch.setattr(cli, "_make_llm", lambda config: llm)
    assert cli.main(cli_args) == 1
    assert capsys.readouterr().err == "error: invalid credentials\n"
    assert llm.closed


@pytest.mark.parametrize("flag", ["--hops", "--concurrency"])
def test_cli_rejects_nonpositive_limits(cli_args: list[str], flag: str) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main([*cli_args, flag, "0"])
    assert caught.value.code == 2
