"""Value types shared across layers and returned to users."""

from __future__ import annotations

import hashlib
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

Mode = Literal["rag", "oag", "hybrid"]
MODES: tuple[Mode, ...] = ("rag", "oag", "hybrid")

# Fixed namespace so the same content always maps to the same IDs across machines.
ID_NAMESPACE = uuid.UUID("6f1c7d0e-2a51-4f43-9a0b-8f1b2a3c4d5e")


def normalize_text(text: str) -> str:
    """Canonical form used for hashing: NFC, unix newlines, no trailing whitespace."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def chunk_id(doc_id: str, idx: int) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"chunk:{doc_id}:{idx}"))


@dataclass(frozen=True, slots=True)
class Document:
    """A loaded source. ``id`` is the content hash, which makes ingest idempotent."""

    text: str
    source: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return content_hash(self.text)

    @property
    def id(self) -> str:
        return self.content_hash

    @classmethod
    def from_text(
        cls, text: str, *, source: str, title: str | None = None, **metadata: Any
    ) -> Document:
        return cls(text=text, source=source, title=title or source, metadata=metadata)


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    doc_id: str
    idx: int
    text: str
    token_count: int
    start_char: int
    end_char: int
    section_path: tuple[str, ...] = ()
    page_index: int | None = None
    chunk_strategy: str = ""


@dataclass(frozen=True, slots=True)
class ChunkHit:
    """A retrieved chunk. ``score`` is cosine similarity: higher is better."""

    chunk_id: str
    doc_id: str
    source: str
    text: str
    score: float
    section_path: tuple[str, ...] = ()
    page_index: int | None = None
    prev_id: str | None = None
    prev_text: str | None = None
    next_id: str | None = None
    next_text: str | None = None

    @property
    def location(self) -> str:
        """Human-readable pointer, e.g. ``contract.pdf · page 4 · Termination``."""
        parts = [self.source]
        if self.page_index is not None:
            parts.append(f"page {self.page_index + 1}")
        if self.section_path:
            parts.append(" > ".join(self.section_path))
        return " · ".join(parts)


@dataclass(frozen=True, slots=True)
class EntityHit:
    """An entity matched to a question. ``score`` is a fused rank score: higher is better."""

    id: str
    name: str
    type: str
    score: float


@dataclass(frozen=True, slots=True)
class Fact:
    """A typed statement from the ontology graph.

    ``kind="relation"``: ``subject —predicate→ object`` with relation properties.
    ``kind="entity"``: ``subject`` is an ``object``-typed entity with ``properties``.
    ``evidence`` holds the ids of the chunks the statement was extracted from.
    """

    subject: str
    predicate: str
    object: str
    properties: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()
    kind: Literal["relation", "entity"] = "relation"
    subject_id: str = ""
    object_id: str = ""

    def render(self) -> str:
        props = ", ".join(f"{k}: {v}" for k, v in sorted(self.properties.items()))
        if self.kind == "entity":
            return f"{self.subject} ({self.object})" + (f": {props}" if props else "")
        text = f"{self.subject} —{self.predicate}→ {self.object}"
        return f"{text} ({props})" if props else text


@dataclass(frozen=True, slots=True)
class Citation:
    id: str
    kind: Literal["chunk", "fact"]
    ref: str


@dataclass(frozen=True, slots=True)
class RetrievedContext:
    """What retrieval hands to generation, or to the caller of ``kb.retrieve``."""

    question: str
    mode_used: Mode
    chunks: list[ChunkHit]
    facts: list[Fact]
    citations: list[Citation]
    prompt_context: str
    token_count: int


@dataclass(frozen=True, slots=True)
class QueryResult:
    answer: str
    mode_used: Mode
    chunks: list[ChunkHit]
    facts: list[Fact]
    citations: list[Citation]
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class IngestReport:
    documents_processed: int = 0
    documents_skipped: int = 0
    chunks_written: int = 0
    entities_written: int = 0
    relations_written: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    validation_errors: list[tuple[str, str]] = field(default_factory=list)
    unmapped_concepts: list[tuple[str, int]] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures
