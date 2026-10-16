"""Which structured-output method works for which (endpoint, model).

Detected once per process by trying the methods in order; the first one that
works is remembered so later calls go straight to it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class StructuredMethod(StrEnum):
    JSON_SCHEMA = "json_schema"
    TOOLS = "tools"
    JSON_OBJECT = "json_object"
    PROMPT = "prompt"


FALLBACK_ORDER: tuple[StructuredMethod, ...] = (
    StructuredMethod.JSON_SCHEMA,
    StructuredMethod.TOOLS,
    StructuredMethod.JSON_OBJECT,
    StructuredMethod.PROMPT,
)

_detected: dict[tuple[str, str], StructuredMethod] = {}


def key_for(llm: Any) -> tuple[str, str]:
    return (str(getattr(llm, "endpoint", "")), str(llm.model))


def detected(llm: Any) -> StructuredMethod | None:
    return _detected.get(key_for(llm))


def remember(llm: Any, method: StructuredMethod) -> None:
    _detected[key_for(llm)] = method


def forget(llm: Any | None = None) -> None:
    """Drop cached detections (all of them, or one model's)."""
    if llm is None:
        _detected.clear()
    else:
        _detected.pop(key_for(llm), None)
