"""Two kinds of checks: is the ontology itself sound, and does an extraction obey it."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from rag_with_oag.ontology.model import (
    RESERVED_LABELS,
    RESERVED_PROPERTIES,
    RESERVED_RELATIONS,
    Property,
)
from rag_with_oag.store.queries import IDENTIFIER

if TYPE_CHECKING:
    from rag_with_oag.ontology.model import Ontology


def check_ontology(o: Ontology) -> list[str]:
    issues: list[str] = []
    if not o.entities:
        issues.append("at least one entity type is required")

    seen: set[str] = set()
    for e in o.entities:
        where = f"entity {e.name!r}"
        if not IDENTIFIER.fullmatch(e.name):
            issues.append(f"{where}: names must be ASCII identifiers ([A-Za-z_][A-Za-z0-9_]*)")
        if e.name in RESERVED_LABELS:
            issues.append(f"{where}: name is reserved by the library")
        if e.name in seen:
            issues.append(f"{where}: defined twice")
        seen.add(e.name)
        issues.extend(_check_properties(e.properties, where))
        if e.key is not None and e.property(e.key) is None:
            issues.append(f"{where}: key {e.key!r} is not one of its properties")

    seen_rel: set[str] = set()
    for r in o.relations:
        where = f"relation {r.name!r}"
        if not IDENTIFIER.fullmatch(r.name):
            issues.append(f"{where}: names must be ASCII identifiers ([A-Za-z_][A-Za-z0-9_]*)")
        if r.name in RESERVED_RELATIONS:
            issues.append(f"{where}: name is reserved by the library")
        if r.name in seen_rel:
            issues.append(f"{where}: defined twice")
        seen_rel.add(r.name)
        for end in (r.source, r.target):
            if end not in seen:
                issues.append(f"{where}: unknown entity type {end!r}")
        issues.extend(_check_properties(r.properties, where))
    return issues


def _check_properties(props: tuple[Property, ...], where: str) -> list[str]:
    issues: list[str] = []
    names: set[str] = set()
    for p in props:
        if not IDENTIFIER.fullmatch(p.name):
            issues.append(f"{where}: property {p.name!r} must be an ASCII identifier")
        if p.name in RESERVED_PROPERTIES:
            issues.append(f"{where}: property {p.name!r} is reserved by the library")
        if p.name in names:
            issues.append(f"{where}: property {p.name!r} defined twice")
        names.add(p.name)
        if p.type == "enum" and not p.values:
            issues.append(f"{where}: enum property {p.name!r} needs 'values'")
        if p.type != "enum" and p.values:
            issues.append(f"{where}: only enum properties take 'values' ({p.name!r})")
    return issues


def coerce_properties(
    defined: tuple[Property, ...], given: Mapping[str, Any], where: str
) -> tuple[dict[str, Any], list[str]]:
    """Coerce extracted values to their declared types.

    Unknown properties and nulls are dropped silently: the model was told to use
    null for anything the text does not state. Returns (clean values, errors).
    """
    by_name = {p.name: p for p in defined}
    clean: dict[str, Any] = {}
    errors: list[str] = []
    for name, value in given.items():
        prop = by_name.get(name)
        if prop is None or value is None or value == "":
            continue
        try:
            clean[name] = _coerce(prop, value)
        except ValueError as exc:
            errors.append(f"{where}: property {name!r}: {exc}")
    for prop in defined:
        if prop.required and prop.name not in clean:
            errors.append(f"{where}: required property {prop.name!r} is missing")
    return clean, errors


def _coerce(prop: Property, value: Any) -> Any:
    match prop.type:
        case "string":
            return str(value).strip()
        case "integer":
            if isinstance(value, bool):
                raise ValueError(f"expected an integer, got {value!r}")
            return int(str(value).replace(" ", "").replace(".", "").replace(",", ""))
        case "number":
            if isinstance(value, bool):
                raise ValueError(f"expected a number, got {value!r}")
            return float(value)
        case "boolean":
            if isinstance(value, bool):
                return value
            text = str(value).strip().casefold()
            if text in {"true", "yes", "evet", "1"}:
                return True
            if text in {"false", "no", "hayir", "hayır", "0"}:
                return False
            raise ValueError(f"expected a boolean, got {value!r}")
        case "date":
            try:
                return dt.date.fromisoformat(str(value).strip()[:10]).isoformat()
            except ValueError:
                raise ValueError(f"expected a date as YYYY-MM-DD, got {value!r}") from None
        case "enum":
            text = str(value).strip()
            for allowed in prop.values or ():
                if text.casefold() == allowed.casefold():
                    return allowed
            raise ValueError(f"{value!r} is not one of {list(prop.values or ())}")
    raise ValueError(f"unknown property type {prop.type!r}")  # pragma: no cover
