"""The ontology: entity types, relation types and their properties.

Type names become graph labels and relationship types, so they must be plain
ASCII identifiers (``Sirket``, ``TARAFIDIR``). Descriptions can be in any
language and are what the extraction model reads.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from rag_with_oag.errors import OntologyError

PropertyType = Literal["string", "integer", "number", "boolean", "date", "enum"]

# Names the library itself uses in the graph.
RESERVED_LABELS = frozenset({"Entity", "Chunk", "Document", "Meta", "ExtractionCache"})
RESERVED_RELATIONS = frozenset({"HAS_CHUNK", "NEXT", "MENTIONS"})
RESERVED_PROPERTIES = frozenset(
    {
        "id",
        "name",
        "name_norm",
        "description",
        "embedding",
        "ontology_version",
        "confidence",
        "evidence",
        "type",
        "ref",
        "chunks",
    }
)


class Property(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    type: PropertyType = "string"
    description: str = ""
    required: bool = False
    values: tuple[str, ...] | None = None  # enum members


class EntityType(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str = ""
    key: str | None = None  # property that identifies an instance, e.g. a tax number
    properties: tuple[Property, ...] = ()

    def property(self, name: str) -> Property | None:
        return next((p for p in self.properties if p.name == name), None)


class RelationType(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    source: str
    target: str
    description: str = ""
    properties: tuple[Property, ...] = ()

    def property(self, name: str) -> Property | None:
        return next((p for p in self.properties if p.name == name), None)


class Ontology(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "ontology"
    version: str = "1"
    entities: tuple[EntityType, ...]
    relations: tuple[RelationType, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        from rag_with_oag.ontology.validator import check_ontology

        issues = check_ontology(self)
        if issues:
            raise OntologyError("invalid ontology:\n- " + "\n- ".join(issues))
        return self

    # --- lookup ---------------------------------------------------------------

    def entity(self, name: str) -> EntityType | None:
        return next((e for e in self.entities if e.name == name), None)

    def relation(self, name: str) -> RelationType | None:
        return next((r for r in self.relations if r.name == name), None)

    @property
    def entity_names(self) -> list[str]:
        return [e.name for e in self.entities]

    @property
    def relation_names(self) -> list[str]:
        return [r.name for r in self.relations]

    @property
    def fingerprint(self) -> str:
        """Content hash; stored in the graph so a changed ontology is noticed."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def describe(self) -> str:
        """Compact, prompt-friendly rendering for the extraction model."""
        lines = ["Entity types:"]
        for e in self.entities:
            lines.append(f"- {e.name}: {e.description}".rstrip(": "))
            if e.key:
                lines.append(f"    identifying property: {e.key}")
            lines.extend(f"    {_describe_property(p)}" for p in e.properties)
        if self.relations:
            lines.append("Relation types (source -> target):")
            for r in self.relations:
                head = f"- {r.name}: {r.source} -> {r.target}"
                lines.append(f"{head}. {r.description}" if r.description else head)
                lines.extend(f"    {_describe_property(p)}" for p in r.properties)
        return "\n".join(lines)

    # --- loading --------------------------------------------------------------

    @classmethod
    def from_yaml(cls, source: str | Path) -> Ontology:
        """Load from a YAML file path or a YAML string."""
        from rag_with_oag.ontology.loader import from_yaml

        return from_yaml(source)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Ontology:
        from rag_with_oag.ontology.loader import from_dict

        return from_dict(data)

    @classmethod
    def from_classes(
        cls,
        entities: Sequence[type[BaseModel]],
        relations: Mapping[str, tuple[type[BaseModel], ...]] | None = None,
        *,
        name: str = "ontology",
        version: str = "1",
    ) -> Ontology:
        """Build from Pydantic models; see ``loader.from_classes``."""
        from rag_with_oag.ontology.loader import from_classes

        return from_classes(entities, relations or {}, name=name, version=version)

    def to_yaml(self, path: str | Path | None = None) -> str:
        from rag_with_oag.ontology.loader import to_yaml

        text = to_yaml(self)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


def _describe_property(p: Property) -> str:
    kind = f"enum: {'|'.join(p.values or ())}" if p.type == "enum" else p.type
    if p.type == "date":
        kind += ", YYYY-MM-DD"
    flags = ", required" if p.required else ""
    text = f"{p.name} ({kind}{flags})"
    return f"{text}: {p.description}" if p.description else text
