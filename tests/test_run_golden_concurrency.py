"""Offline checks for the golden runner's question concurrency."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from eval import run_golden

from rag_with_oag import EmbedConfig
from rag_with_oag.types import IngestReport


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[argparse.Namespace, MagicMock]:
    (tmp_path / "ontology.yaml").write_bytes((run_golden.GOLDEN / "ontology.yaml").read_bytes())
    questions: dict[str, list[run_golden.QuestionItem]] = {
        "second": [{"q": q, "answer": [[q]]} for q in ("slow", "fast", "skip one")],
        "first": [{"q": q, "answer": [[q]]} for q in ("medium", "fourth", "skip two")],
    }
    monkeypatch.setattr(run_golden, "load_questions", lambda _: questions)
    monkeypatch.setattr(run_golden, "make_embedder", lambda _: EmbedConfig(model="fake"))
    kb = MagicMock(
        ingest=AsyncMock(return_value=IngestReport()),
        stats=AsyncMock(return_value={}),
        close=AsyncMock(),
    )
    monkeypatch.setattr(run_golden, "AsyncKnowledgeBase", MagicMock(return_value=kb))
    args = argparse.Namespace(
        set=tmp_path,
        out=tmp_path,
        kb="golden",
        llm="fake",
        extractor=None,
        base_url=None,
        seed_cache_from=None,
        fresh=False,
        concurrency=2,
        limit=2,
        modes="rag,hybrid",
        equal_budget=False,
        k=5,
        hops=None,
        answers=False,
    )
    return args, kb


@pytest.mark.parametrize("concurrency", [1, 3])
async def test_question_concurrency_preserves_row_order(
    concurrency: int,
    runner: tuple[argparse.Namespace, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    active = peak = 0
    completed: list[tuple[str, str]] = []
    delays = {"slow": 0.03, "fast": 0, "medium": 0.01, "fourth": 0}

    async def fake_evaluate_modes(
        _kb: object,
        group: str,
        item: run_golden.QuestionItem,
        modes: Sequence[run_golden.EvalMode],
        *_args: object,
        answers: bool,
    ) -> list[run_golden.Row]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(delays[item["q"]])
        finally:
            active -= 1
        completed.append((group, item["q"]))
        return [
            run_golden.Row(group, mode, item["q"], mode, True, 12, 0, chain_hit=mode == "hybrid")
            for mode in modes
        ]

    args, _kb = runner
    args.question_concurrency = concurrency
    monkeypatch.setattr(run_golden, "evaluate_modes", fake_evaluate_modes)

    summary = await run_golden.run(args)

    modes = run_golden.runner_modes(args)
    expected = [("second", "slow"), ("second", "fast"), ("first", "medium"), ("first", "fourth")]
    assert peak == concurrency
    assert [(row.group, row.question, row.mode) for row in summary.rows] == [
        (group, question, mode) for group, question in expected for mode in modes
    ]
    if concurrency == 1:
        assert completed == expected
    else:
        assert completed != expected
    assert capsys.readouterr().out.splitlines()[1:] == [
        line
        for group, question in completed
        for line in (f"· {group:10} {'rag':7} {question}", f"✓ {group:10} {'hybrid':7} {question}")
    ]
    assert run_golden.summary_data(summary, modes)["question_concurrency"] == concurrency
    assert f"Question concurrency `{concurrency}`" in run_golden.render(summary, modes)


async def test_question_failure_cancels_pending_work_before_close(
    runner: tuple[argparse.Namespace, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    args, kb = runner
    args.question_concurrency = 2
    events: list[str] = []
    error = RuntimeError("question failed")

    async def fake_evaluate_modes(
        _kb: object,
        _group: str,
        item: run_golden.QuestionItem,
        *_args: object,
        answers: bool,
    ) -> list[run_golden.Row]:
        events.append(item["q"])
        if item["q"] == "slow":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                events.append("slow cancelled")
                raise
        await asyncio.sleep(0)
        raise error

    monkeypatch.setattr(run_golden, "evaluate_modes", fake_evaluate_modes)
    kb.close.side_effect = lambda: events.append("closed")

    with pytest.raises(RuntimeError, match="question failed") as exc:
        await run_golden.run(args)

    assert exc.value is error
    assert events == ["slow", "fast", "slow cancelled", "closed"]


def test_cli_rejects_zero_question_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "unused"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_golden.py", "--llm", "fake", "--out", str(out), "--question-concurrency", "0"],
    )

    with pytest.raises(SystemExit) as exc:
        run_golden.main()

    assert exc.value.code == 2
    assert "--question-concurrency must be at least 1" in capsys.readouterr().err
    assert not out.exists()
