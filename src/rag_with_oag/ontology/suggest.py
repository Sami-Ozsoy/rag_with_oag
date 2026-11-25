"""Propose an ontology from a balanced sample of documents and optional ingest feedback."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast, get_args

from pydantic import BaseModel, ConfigDict, ValidationError

from rag_with_oag._text import DEFAULT_NAME_SUFFIXES
from rag_with_oag._tokens import count_tokens, split_spans
from rag_with_oag.config import LLMConfig
from rag_with_oag.errors import OntologyError
from rag_with_oag.generation import prompts
from rag_with_oag.ingest.loaders import Source, iter_paths, load_path
from rag_with_oag.llm import structured
from rag_with_oag.llm.client import ChatModel, LLMClient, Message
from rag_with_oag.ontology.model import (
    RESERVED_LABELS,
    RESERVED_PROPERTIES,
    RESERVED_RELATIONS,
    EntityType,
    Ontology,
    PropertyType,
    RelationType,
)
from rag_with_oag.ontology.schema_gen import _nullable
from rag_with_oag.types import Document

logger = logging.getLogger(__name__)


class SuggestedProperty(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: PropertyType
    description: str
    required: bool
    values: list[str] | None


class SuggestedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    key: str | None
    properties: list[SuggestedProperty]


class SuggestedRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source: str
    target: str
    description: str
    properties: list[SuggestedProperty]


class SuggestedOntology(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[SuggestedEntity]
    relations: list[SuggestedRelation]
    name_suffixes: list[str]


def _object_schema(**properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _suggestion_schema() -> dict[str, Any]:
    string = {"type": "string"}
    properties = {
        "type": "array",
        "items": _object_schema(
            name=string,
            type={"type": "string", "enum": list(get_args(PropertyType))},
            description=string,
            required={"type": "boolean"},
            values=_nullable({"type": "array", "items": string}),
        ),
    }
    return _object_schema(
        entities={
            "type": "array",
            "items": _object_schema(
                name=string, description=string, key=_nullable(string), properties=properties
            ),
        },
        relations={
            "type": "array",
            "items": _object_schema(
                name=string, source=string, target=string, description=string, properties=properties
            ),
        },
        name_suffixes={"type": "array", "items": string},
    )


def load_documents(sources: Iterable[Source] | Source) -> list[Document]:
    """Load files and directories as in ingest, passing through existing documents."""
    if isinstance(sources, str | Path | Document):
        sources = [sources]
    documents: list[Document] = []
    for source in sources:
        if isinstance(source, Document):
            documents.append(source)
        else:
            documents.extend(load_path(path) for path in iter_paths([source]))
    return documents


def _leading_text(text: str, max_tokens: int) -> str:
    _, end = split_spans(text, max_tokens)[0]
    excerpt = text[:end]
    if count_tokens(excerpt) <= max_tokens:
        return excerpt

    # The span for an oversized word is estimated by character length.
    start = 0
    while start + 1 < end:
        mid = (start + end) // 2
        if count_tokens(text[:mid]) <= max_tokens:
            start = mid
        else:
            end = mid
    return text[:start]


def _sample_documents(documents: Sequence[Document], sample_tokens: int) -> str:
    nonempty = sorted((doc for doc in documents if doc.text.strip()), key=lambda doc: doc.source)
    n = len(nonempty)
    blocks = []
    if n:
        per_doc = max(200, sample_tokens // n)
        limit = sample_tokens // 200
        if n > limit:
            nonempty = [nonempty[i * (n - 1) // (limit - 1)] for i in range(limit)]
        for i, doc in enumerate(nonempty, start=1):
            title = doc.title or Path(doc.source).name
            blocks.append(f"### D{i}: {title}\n{_leading_text(doc.text.strip(), per_doc)}")
    logger.info("Sampled %d of %d documents", len(blocks), len(documents))
    return "\n\n".join(blocks)


def _merge_types[T: EntityType | RelationType](
    base: Sequence[T], suggested: Sequence[T]
) -> tuple[T, ...]:
    merged = {item.name: item for item in base}
    for item in suggested:
        existing = merged.get(item.name)
        if existing is None:
            merged[item.name] = item
            continue
        added = tuple(p for p in item.properties if existing.property(p.name) is None)
        if added:
            merged[item.name] = cast(
                T, existing.model_copy(update={"properties": existing.properties + added})
            )
    return tuple(merged.values())


def _convert(suggestion: SuggestedOntology, base: Ontology | None, name: str | None) -> Ontology:
    entities = tuple(EntityType.model_validate(e.model_dump()) for e in suggestion.entities)
    relations = tuple(RelationType.model_validate(r.model_dump()) for r in suggestion.relations)
    suffixes = tuple(suggestion.name_suffixes)
    if base is None:
        return Ontology(
            name="ontology" if name is None else name,
            entities=entities,
            relations=relations,
            name_suffixes=suffixes,
        )

    # Include omitted base entities when checking the candidate's relation endpoints.
    names = {e.name for e in entities}
    candidate = Ontology(
        name=base.name if name is None else name,
        version=base.version,
        entities=entities + tuple(e for e in base.entities if e.name not in names),
        relations=relations,
        name_suffixes=suffixes,
    )
    return Ontology(
        name=candidate.name,
        version=candidate.version,
        entities=_merge_types(base.entities, candidate.entities),
        relations=_merge_types(base.relations, candidate.relations),
        name_suffixes=tuple(dict.fromkeys(base.name_suffixes + candidate.name_suffixes)),
    )


async def _generate(
    llm: ChatModel, messages: list[Message], base: Ontology | None, name: str | None
) -> Ontology:
    schema = _suggestion_schema()
    request = messages
    errors: list[str] = []
    for attempt in range(2):
        result = await structured.generate(
            llm, request, schema=schema, model=SuggestedOntology, name="ontology"
        )
        try:
            return _convert(result.value, base, name)
        except ValidationError as exc:
            errors = [
                f"{'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}"
                for err in exc.errors()
            ]
        except OntologyError as exc:
            errors = str(exc).split("\n- ")[1:] or [str(exc)]
        if attempt == 0:
            request = [*messages, *structured.feedback_messages(result.raw, errors)]
    raise OntologyError(
        "suggested ontology is invalid after one correction:\n- " + "\n- ".join(errors)
    )


async def suggest_ontology(
    sources: Iterable[Source] | Source,
    llm: LLMConfig | ChatModel,
    *,
    base: Ontology | None = None,
    unmapped: Iterable[str | tuple[str, int]] = (),
    name: str | None = None,
    instructions: str | None = None,
    sample_tokens: int = 12_000,
) -> Ontology:
    """Suggest an ontology to review, preserving existing definitions when extending a base."""
    if sample_tokens < 500:
        raise ValueError("sample_tokens must be at least 500")
    documents = load_documents(sources)
    counts: Counter[str] = Counter()
    for item in unmapped:
        term, count = (item, 1) if isinstance(item, str) else item
        counts[term] += count
    if not documents and not counts:
        raise ValueError("suggestion needs documents or unmapped concepts")
    if not documents and base is None:
        raise ValueError("suggestion without a base ontology needs documents")

    terms = "\n".join(f"- {term} ({count})" for term, count in counts.most_common(50))
    system = prompts.SUGGEST_SYSTEM.format(
        reserved=", ".join(sorted(RESERVED_LABELS | RESERVED_RELATIONS | RESERVED_PROPERTIES)),
        default_suffixes=", ".join(DEFAULT_NAME_SUFFIXES),
    )
    user = prompts.SUGGEST_USER.format(documents=_sample_documents(documents, sample_tokens))
    if base is not None:
        user += "\n\n" + prompts.SUGGEST_EXTEND.format(
            ontology=base.describe(), unmapped=terms or "(none)"
        )
    elif terms:
        user += f"\n\nUnmapped concepts (with counts):\n{terms}"
    if instructions:
        user += f"\n\nUser guidance about what matters:\n{instructions}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if isinstance(llm, LLMConfig):
        client = LLMClient(llm)
        try:
            return await _generate(client, messages, base, name)
        finally:
            await client.aclose()
    return await _generate(llm, messages, base, name)
