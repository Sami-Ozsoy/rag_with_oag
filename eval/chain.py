"""Score the support for each hop against the packed retrieval context."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import PureWindowsPath
from typing import TYPE_CHECKING, Any

from rag_with_oag.retrieval.context import CHUNKS_HEADER
from rag_with_oag.types import Fact, RetrievedContext

if TYPE_CHECKING or __package__:
    from .synthetic.text import norm
else:
    from synthetic.text import norm

type Chain = list[list[dict[str, Any]]]

CHUNK_BLOCK = re.compile(r"(?:\A|\n\n)\[(C\d+)\] [^\n]*\n")


def _packed_text(ctx: RetrievedContext) -> dict[str, list[str]]:
    hits = {hit.chunk_id: hit for hit in ctx.chunks}
    citations = {c.id: hits[c.ref] for c in ctx.citations if c.kind == "chunk" and c.ref in hits}
    by_doc: dict[str, list[str]] = defaultdict(list)
    # Read each body separately, including any neighbours selected by the packer.
    section = ctx.prompt_context.partition(f"{CHUNKS_HEADER}\n\n")[-1]
    blocks = CHUNK_BLOCK.split(section)
    for citation, body in zip(blocks[1::2], blocks[2::2], strict=True):
        if (hit := citations.get(citation)) is not None:
            folded = norm(body)
            for name in {PureWindowsPath(value).name for value in (hit.source, hit.doc_id)}:
                by_doc[name].append(folded)
    return by_doc


def _number(value: object) -> int | float | None:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


def _graph_covers(step: dict[str, Any], fact: Fact, node_keys: Mapping[str, str]) -> bool:
    if step["kind"] == "relation" and fact.kind == "relation":
        return (
            fact.predicate == step["type"]
            and node_keys.get(fact.subject_id) == step["source"]
            and node_keys.get(fact.object_id) == step["target"]
            and all(
                prop not in fact.properties or _relation_prop(prop, value, fact.properties[prop])
                for prop, value in step["props"].items()
            )
        )
    if step["kind"] == "property" and fact.kind == "entity":
        if node_keys.get(fact.subject_id) != step["entity"]:
            return False
        found, expected = fact.properties.get(step["prop"]), step["value"]
        if isinstance(expected, str) and isinstance(found, str):
            return norm(found) == norm(expected)
        actual, target = _number(found), _number(expected)
        return actual is not None and target is not None and actual == target
    return False


def _relation_prop(prop: str, expected: object, found: object) -> bool:
    if prop == "oran":
        actual, target = _number(found), _number(expected)
        return actual is not None and target is not None and abs(actual - target) <= 0.5
    return found == expected


def chain_coverage(
    ctx: RetrievedContext, chain: Chain, node_keys: Mapping[str, str]
) -> tuple[int, int]:
    """Return the covered and total step counts, including repeated hops across paths."""
    packed = _packed_text(ctx)
    covered = total = 0
    for path in chain:
        for step in path:
            total += 1
            covered += any(_graph_covers(step, fact, node_keys) for fact in ctx.facts) or any(
                all(norm(literal) in body for literal in source["literals"])
                for source in step["evidence"]
                for body in packed.get(source["doc"], ())
            )
    return covered, total
