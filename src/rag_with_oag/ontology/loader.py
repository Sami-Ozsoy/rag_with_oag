"""Ontology from YAML or Pydantic classes, and back to YAML.

YAML format::

    name: contracts
    version: "1"
    entities:
      Sirket:
        description: A company that is party to contracts
        key: vergi_no
        properties:
          vergi_no: string                      # shorthand: just the type
          sehir: {type: string, description: City of the head office}
          tur: {type: enum, values: [anonim, limited]}
    relations:
      TARAFIDIR:
        source: Sirket
        target: Sozlesme
        properties:
          rol: {type: enum, values: [musteri, saglayici]}
"""

from __future__ import annotations

import datetime as dt
import enum
import inspect
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

import yaml
from pydantic import BaseModel

from rag_with_oag.errors import OntologyError
from rag_with_oag.ontology.model import EntityType, Ontology, Property, RelationType


def from_yaml(source: str | Path) -> Ontology:
    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and source.endswith((".yaml", ".yml"))
    ):
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = source
    data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise OntologyError("ontology YAML must be a mapping with 'entities' (and 'relations')")
    return from_dict(data)


def from_dict(data: Mapping[str, Any]) -> Ontology:
    unknown = set(data) - {"name", "version", "entities", "relations"}
    if unknown:
        raise OntologyError(f"unknown top-level ontology keys: {sorted(unknown)}")
    entities = [
        EntityType(
            name=name,
            description=(spec or {}).get("description", ""),
            key=(spec or {}).get("key"),
            properties=_properties((spec or {}).get("properties"), f"entity {name}"),
        )
        for name, spec in _mapping(data.get("entities"), "entities").items()
    ]
    relations = [
        RelationType(
            name=name,
            source=_required(spec, "source", f"relation {name}"),
            target=_required(spec, "target", f"relation {name}"),
            description=spec.get("description", ""),
            properties=_properties(spec.get("properties"), f"relation {name}"),
        )
        for name, spec in _mapping(data.get("relations"), "relations").items()
    ]
    return Ontology(
        name=str(data.get("name", "ontology")),
        version=str(data.get("version", "1")),
        entities=tuple(entities),
        relations=tuple(relations),
    )


def to_yaml(ontology: Ontology) -> str:
    def props(ps: Sequence[Property]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p in ps:
            spec: dict[str, Any] = {"type": p.type}
            if p.values:
                spec["values"] = list(p.values)
            if p.description:
                spec["description"] = p.description
            if p.required:
                spec["required"] = True
            out[p.name] = p.type if spec == {"type": p.type} else spec
        return out

    data: dict[str, Any] = {"name": ontology.name, "version": ontology.version, "entities": {}}
    for e in ontology.entities:
        spec: dict[str, Any] = {}
        if e.description:
            spec["description"] = e.description
        if e.key:
            spec["key"] = e.key
        if e.properties:
            spec["properties"] = props(e.properties)
        data["entities"][e.name] = spec
    if ontology.relations:
        data["relations"] = {}
        for r in ontology.relations:
            rspec: dict[str, Any] = {"source": r.source, "target": r.target}
            if r.description:
                rspec["description"] = r.description
            if r.properties:
                rspec["properties"] = props(r.properties)
            data["relations"][r.name] = rspec
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


def from_classes(
    entities: Sequence[type[BaseModel]],
    relations: Mapping[str, tuple[type[BaseModel], ...]],
    *,
    name: str,
    version: str,
) -> Ontology:
    """Pydantic models as entity types.

    The class name is the type name, the docstring its description, fields its
    properties (``Optional`` = not required). Set ``__ontology_key__ = "field"``
    on a class to name its identifying property. Relations map a name to
    ``(Source, Target)`` or ``(Source, Target, PropertiesModel)``.
    """
    entity_types = [
        EntityType(
            name=cls.__name__,
            description=inspect.cleandoc(cls.__doc__ or ""),
            key=getattr(cls, "__ontology_key__", None),
            properties=_class_properties(cls),
        )
        for cls in entities
    ]
    relation_types = []
    for rel_name, spec in relations.items():
        if len(spec) not in (2, 3):
            raise OntologyError(f"relation {rel_name}: expected (Source, Target[, Properties])")
        relation_types.append(
            RelationType(
                name=rel_name,
                source=spec[0].__name__,
                target=spec[1].__name__,
                description=inspect.cleandoc(spec[2].__doc__ or "") if len(spec) == 3 else "",
                properties=_class_properties(spec[2]) if len(spec) == 3 else (),
            )
        )
    return Ontology(
        name=name, version=version, entities=tuple(entity_types), relations=tuple(relation_types)
    )


# --- helpers ---------------------------------------------------------------------


def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise OntologyError(f"'{what}' must be a mapping of name -> definition")
    return value


def _required(spec: Mapping[str, Any] | None, key: str, where: str) -> str:
    if not spec or key not in spec:
        raise OntologyError(f"{where}: missing '{key}'")
    return str(spec[key])


def _properties(value: Any, where: str) -> tuple[Property, ...]:
    props = []
    for pname, spec in _mapping(value, f"{where} properties").items():
        if isinstance(spec, str):
            spec = {"type": spec}
        if not isinstance(spec, Mapping):
            raise OntologyError(f"{where}: property '{pname}' must be a type name or a mapping")
        values = spec.get("values")
        props.append(
            Property(
                name=pname,
                type=spec.get("type", "string"),
                description=spec.get("description", ""),
                required=bool(spec.get("required", False)),
                values=tuple(str(v) for v in values) if values is not None else None,
            )
        )
    return tuple(props)


_SCALARS: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dt.date: "date",
    dt.datetime: "date",
}


def _class_properties(cls: type[BaseModel]) -> tuple[Property, ...]:
    props = []
    for fname, field in cls.model_fields.items():
        annotation, optional = _unwrap_optional(field.annotation)
        ptype, values = _property_type(annotation, f"{cls.__name__}.{fname}")
        props.append(
            Property(
                name=fname,
                type=ptype,  # type: ignore[arg-type]
                description=field.description or "",
                required=field.is_required() and not optional,
                values=values,
            )
        )
    return tuple(props)


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    if get_origin(annotation) in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def _property_type(annotation: Any, where: str) -> tuple[str, tuple[str, ...] | None]:
    if annotation in _SCALARS:
        return _SCALARS[annotation], None
    if get_origin(annotation) is Literal:
        return "enum", tuple(str(v) for v in get_args(annotation))
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        return "enum", tuple(str(m.value) for m in annotation)
    raise OntologyError(f"{where}: unsupported property type {annotation!r}")
