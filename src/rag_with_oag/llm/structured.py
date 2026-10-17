"""Structured JSON output from models whose support for it differs.

Methods are tried in order: ``json_schema`` (strict) -> tool calling ->
``json_object`` + schema in the prompt -> plain prompt + JSON extraction. A
method counts as unsupported when the server rejects the request or answers
with something that is not JSON at all. Output that is JSON but fails
validation is sent back with the errors and retried with the same method.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import openai
from pydantic import BaseModel, ValidationError

from rag_with_oag.errors import StructuredOutputError
from rag_with_oag.llm import capabilities
from rag_with_oag.llm.capabilities import FALLBACK_ORDER, StructuredMethod
from rag_with_oag.llm.client import ChatModel, Message

logger = logging.getLogger(__name__)

StructuredMode = StructuredMethod | str  # "auto" or a method name


class _Unsupported(Exception):
    pass


# Status codes with which servers say "I don't do that", as opposed to outages.
_UNSUPPORTED_ERRORS: tuple[type[Exception], ...] = (
    openai.BadRequestError,
    openai.UnprocessableEntityError,
    openai.NotFoundError,
    _Unsupported,
)


# How OpenAI, vLLM, LM Studio and others word a 400 for an over-long input.
_INPUT_TOO_LONG = re.compile(
    r"context[ _](length|window)|maximum context|too many tokens|(prompt|input) is too long",
    re.IGNORECASE,
)


def _input_too_long(exc: Exception) -> bool:
    return getattr(exc, "code", None) == "context_length_exceeded" or bool(
        _INPUT_TOO_LONG.search(str(exc))
    )


@dataclass(frozen=True, slots=True)
class StructuredResult[T: BaseModel]:
    value: T
    method: StructuredMethod
    raw: str
    attempts: int


async def generate[T: BaseModel](
    llm: ChatModel,
    messages: Sequence[Message],
    *,
    schema: dict[str, Any],
    model: type[T],
    name: str = "output",
    mode: StructuredMode = "auto",
    max_retries: int = 2,
) -> StructuredResult[T]:
    methods = _methods(llm, mode)
    failures: list[str] = []
    for method in methods:
        convo = list(messages)
        for attempt in range(max_retries + 1):
            try:
                raw = await _call(llm, method, convo, schema, name)
                data = _parse_json(raw)
            except _UNSUPPORTED_ERRORS as exc:
                if _input_too_long(exc):
                    raise  # every method would fail the same way, and no feature is missing
                failures.append(f"{method}: {type(exc).__name__}: {str(exc)[:200]}")
                logger.info("structured method %s unavailable for %s: %s", method, llm.model, exc)
                break  # next method
            try:
                value = model.model_validate(data)
            except ValidationError as exc:
                failures.append(f"{method} attempt {attempt + 1}: {exc.error_count()} errors")
                convo = [*convo, *_feedback(raw, _format_errors(exc))]
                continue
            if mode == "auto":
                capabilities.remember(llm, method)
            return StructuredResult(value=value, method=method, raw=raw, attempts=attempt + 1)
    raise StructuredOutputError(
        f"no structured output from {llm.model}; tried: " + "; ".join(failures)
    )


def feedback_messages(raw: str, errors: Sequence[str]) -> list[Message]:
    """Messages asking the model to correct its previous JSON."""
    return _feedback(raw, "\n".join(f"- {e}" for e in errors))


def _methods(llm: ChatModel, mode: StructuredMode) -> Sequence[StructuredMethod]:
    if mode != "auto":
        return (StructuredMethod(mode),)
    known = capabilities.detected(llm)
    return (known,) if known else FALLBACK_ORDER


async def _call(
    llm: ChatModel,
    method: StructuredMethod,
    messages: list[Message],
    schema: dict[str, Any],
    name: str,
) -> str:
    match method:
        case StructuredMethod.JSON_SCHEMA:
            return await llm.complete(
                messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": name, "schema": schema, "strict": True},
                },
            )
        case StructuredMethod.TOOLS:
            complete_tool = getattr(llm, "complete_tool", None)
            if complete_tool is None:
                raise _Unsupported("model client has no tool calling")
            arguments = await complete_tool(messages, name=name, schema=schema)
            if not arguments:
                raise _Unsupported("model did not call the tool")
            return str(arguments)
        case StructuredMethod.JSON_OBJECT:
            return await llm.complete(
                _with_schema(messages, schema), response_format={"type": "json_object"}
            )
        case StructuredMethod.PROMPT:
            return await llm.complete(_with_schema(messages, schema))
    raise ValueError(f"unknown structured method {method!r}")  # pragma: no cover


def _with_schema(messages: list[Message], schema: dict[str, Any]) -> list[Message]:
    instruction = (
        "Respond with a single JSON object and nothing else. It must match this JSON Schema:\n"
        + json.dumps(schema, ensure_ascii=False)
    )
    return [*messages, {"role": "user", "content": instruction}]


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _parse_json(raw: str) -> Any:
    """JSON from a reply, tolerating code fences and prose around one object."""
    text = raw.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise _Unsupported(f"reply is not JSON: {raw[:120]!r}")


def _format_errors(exc: ValidationError) -> str:
    return "\n".join(
        f"- {'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}"
        for err in exc.errors()[:20]
    )


def _feedback(raw: str, errors: str) -> list[Message]:
    return [
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": f"That JSON has problems:\n{errors}\n"
            "Return the complete corrected JSON object only.",
        },
    ]
