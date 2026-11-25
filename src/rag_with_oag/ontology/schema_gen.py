"""Ontology -> JSON Schema for the extraction output.

The schema follows the rules of OpenAI strict structured outputs (every key
required, optional values nullable, no additional properties), which the other
methods in the fallback chain accept as well. Entity and relation items are a
union with one variant per type, so each type only offers its own properties.
"""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field, field_validator

from rag_with_oag.ontology.model import Ontology, Property

_JSON_TYPES = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "date": "string",
    "enum": "string",
}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    if "enum" in schema:
        return {**schema, "type": [schema["type"], "null"], "enum": [*schema["enum"], None]}
    return {**schema, "type": [schema["type"], "null"]}


def _property_schema(p: Property) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": _JSON_TYPES[p.type]}
    if p.type == "enum":
        schema["enum"] = list(p.values or ())
    if p.description:
        schema["description"] = p.description
    return _nullable(schema)


def _properties_object(props: tuple[Property, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {p.name: _property_schema(p) for p in props},
        "required": [p.name for p in props],
        "additionalProperties": False,
    }


def _chunks() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string"},
        "description": "Passage ids (K1, K2, ...) this comes from",
    }


def _relation_ref(entity_type: str, side: str, description: str) -> dict[str, Any]:
    text = (
        f"ref of the {entity_type} on the {side} side of: {description}"
        if description
        else f"ref of a {entity_type}"
    )
    return {"type": "string", "description": text}


def extraction_schema(ontology: Ontology) -> dict[str, Any]:
    entity_variants = [
        {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Short unique id, e.g. e1"},
                "type": {"type": "string", "enum": [e.name]},
                "name": {"type": "string"},
                "description": {"type": ["string", "null"]},
                "properties": _properties_object(e.properties),
                "chunks": _chunks(),
            },
            "required": ["ref", "type", "name", "description", "properties", "chunks"],
            "additionalProperties": False,
        }
        for e in ontology.entities
    ]
    relation_variants = [
        {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": [r.name]},
                "source": _relation_ref(r.source, "source", r.description),
                "target": _relation_ref(r.target, "target", r.description),
                "properties": _properties_object(r.properties),
                "confidence": {"type": ["number", "null"]},
                "chunks": _chunks(),
            },
            "required": ["type", "source", "target", "properties", "confidence", "chunks"],
            "additionalProperties": False,
        }
        for r in ontology.relations
    ]
    unmapped = {
        "type": "object",
        "properties": {"term": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["term", "count"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "entities": {"type": "array", "items": _one_of(entity_variants)},
            "relations": {"type": "array", "items": _one_of(relation_variants)},
            "unmapped": {"type": "array", "items": unmapped},
        },
        "required": ["entities", "relations", "unmapped"],
        "additionalProperties": False,
    }


def _one_of(variants: list[dict[str, Any]]) -> dict[str, Any]:
    if not variants:  # no relation types: an empty array is the only valid value
        return {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    return variants[0] if len(variants) == 1 else {"anyOf": variants}


# --- the structural model the output is parsed into before ontology checks --------


class RawEntity(BaseModel):
    ref: str
    type: str
    name: str
    description: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    chunks: list[str] = Field(default_factory=list)


class RawRelation(BaseModel):
    type: str = Field(validation_alias=AliasChoices("type", "relation"))
    source: str
    target: str
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    chunks: list[str] = Field(default_factory=list)


class RawUnmapped(BaseModel):
    term: str
    count: int = 1


class RawExtraction(BaseModel):
    entities: list[RawEntity] = Field(default_factory=list)
    relations: list[RawRelation] = Field(default_factory=list)
    unmapped: list[RawUnmapped] = Field(default_factory=list)

    @field_validator("unmapped", mode="before")
    @classmethod
    def _normalize_unmapped(cls, value: Any) -> list[RawUnmapped]:
        if not isinstance(value, list):
            return []

        items: list[RawUnmapped] = []
        for item in value:
            if isinstance(item, str):
                items.append(RawUnmapped(term=item))
                continue
            if not isinstance(item, dict):
                continue
            term = next(
                (
                    text
                    for key in ("term", "concept", "name")
                    if isinstance(text := item.get(key), str) and text.strip()
                ),
                None,
            )
            if term is None:
                continue
            count = item.get("count", 1)
            if isinstance(count, str):
                try:
                    count = int(count)
                except ValueError:
                    count = 1
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                count = 1
            items.append(RawUnmapped(term=term, count=count))
        return items
