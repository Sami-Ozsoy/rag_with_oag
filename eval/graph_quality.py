"""Compare a graph snapshot with the keyed ground truth of an evaluation set."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TypedDict

from rag_with_oag import Ontology


class ExpectedEntity(TypedDict):
    type: str
    key: str
    name: str
    props: dict[str, str]


class ExpectedRelation(TypedDict):
    type: str
    source: str
    target: str
    props: dict[str, str | int | float]


class ExpectedGraph(TypedDict):
    entities: list[ExpectedEntity]
    relations: list[ExpectedRelation]


@dataclass(frozen=True)
class FoundEntity:
    id: int
    type: str
    key: str | None
    name: str


@dataclass(frozen=True)
class FoundRelation:
    type: str
    source: int
    target: int
    props: dict[str, object]


class EntityScore(TypedDict):
    expected: int
    found: int
    missing: int
    extra: int
    duplicates: int


class PropertyScore(TypedDict):
    correct: int
    matched: int
    accuracy: float | None


class RelationScore(TypedDict):
    expected: int
    found: int
    matched: int
    precision: float | None
    recall: float | None
    reversed: int
    properties: dict[str, PropertyScore]


class GraphQuality(TypedDict):
    entities: dict[str, EntityScore]
    relations: dict[str, RelationScore]


def score_graph_quality(
    expected: ExpectedGraph,
    entities: list[FoundEntity],
    relations: list[FoundRelation],
    ontology: Ontology,
) -> GraphQuality:
    """Score distinct keyed triples. Edges with missing endpoint keys are false positives."""
    entity_scores: dict[str, EntityScore] = {}
    for entity_type in sorted(
        {e["type"] for e in expected["entities"]} | {e.type for e in entities}
    ):
        keys = {e["key"] for e in expected["entities"] if e["type"] == entity_type}
        names = {
            ontology.match_name(e["name"]) for e in expected["entities"] if e["type"] == entity_type
        }
        found = [e for e in entities if e.type == entity_type]
        keyed = Counter(e.key for e in found if e.key in keys)
        entity_scores[entity_type] = {
            "expected": len(keys),
            "found": len(keyed),
            "missing": len(keys) - len(keyed),
            "extra": len(found) - sum(keyed.values()),
            "duplicates": sum(count - 1 for count in keyed.values())
            + sum(e.key is None and ontology.match_name(e.name) in names for e in found),
        }

    keys_by_id = {e.id: e.key for e in entities}
    relation_scores: dict[str, RelationScore] = {}
    for relation_type in sorted(
        {e["type"] for e in expected["relations"]} | {r.type for r in relations}
    ):
        truth = {
            (e["source"], e["target"]): e["props"]
            for e in expected["relations"]
            if e["type"] == relation_type
        }
        found_edges: dict[tuple[str | int, str | int], list[dict[str, object]]] = defaultdict(list)
        for edge in relations:
            if edge.type == relation_type:
                # Distinguish unkeyed endpoints by their node IDs, which are integers.
                source = keys_by_id.get(edge.source)
                target = keys_by_id.get(edge.target)
                found_edges[
                    (
                        edge.source if source is None else source,
                        edge.target if target is None else target,
                    )
                ].append(edge.props)
        matched = truth.keys() & found_edges.keys()
        prop_scores: dict[str, PropertyScore] = {}
        for prop in ("rol", "oran"):
            if not any(prop in props for props in truth.values()):
                continue
            pairs = [pair for pair in matched if prop in truth[pair]]
            correct = sum(
                all(
                    _property_matches(prop, truth[pair][prop], props.get(prop))
                    for props in found_edges[pair]
                )
                for pair in pairs
            )
            prop_scores[prop] = {
                "correct": correct,
                "matched": len(pairs),
                "accuracy": correct / len(pairs) if pairs else None,
            }
        relation_scores[relation_type] = {
            "expected": len(truth),
            "found": len(found_edges),
            "matched": len(matched),
            "precision": len(matched) / len(found_edges) if found_edges else None,
            "recall": len(matched) / len(truth) if truth else None,
            "reversed": sum(
                (source, target) not in truth and (target, source) in truth
                for source, target in found_edges
            ),
            "properties": prop_scores,
        }
    return {"entities": entity_scores, "relations": relation_scores}


def _property_matches(prop: str, expected: str | int | float, found: object) -> bool:
    if prop == "rol":
        return found == expected
    return (
        isinstance(found, int | float)
        and not isinstance(found, bool)
        and isinstance(expected, int | float)
        and abs(found - expected) <= 0.5
    )
