"""Pack retrieved material into prompt sections under a token budget.

Chunks get citation ids ``C1, C2…`` and facts ``F1, F2…``; answers cite them.
Items are packed in the order given (callers rank them). Neighbour chunks
(``NEXT`` edges) are added only after all primary hits, while budget remains,
so expansion never pushes out a better hit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_with_oag._tokens import count_tokens
from rag_with_oag.types import ChunkHit, Citation, Fact, Mode, RetrievedContext

FACTS_HEADER = "Structured facts:"
CHUNKS_HEADER = "Source texts:"


@dataclass(slots=True)
class _Block:
    hit: ChunkHit
    citation: str
    before: str | None = None
    after: str | None = None

    def render(self) -> str:
        body = "\n".join(t for t in (self.before, self.hit.text, self.after) if t)
        return f"[{self.citation}] {self.hit.location}\n{body}"


@dataclass(frozen=True, slots=True)
class PackedChunks:
    text: str
    chunks: list[ChunkHit]
    citations: list[Citation]
    token_count: int


@dataclass(frozen=True, slots=True)
class PackedFacts:
    text: str
    facts: list[Fact]
    citations: list[Citation]
    token_count: int


def pack_chunks(hits: Sequence[ChunkHit], *, budget: int, expand_neighbors: bool) -> PackedChunks:
    blocks: list[_Block] = []
    used = 0
    for hit in hits:
        block = _Block(hit=hit, citation=f"C{len(blocks) + 1}")
        cost = count_tokens(block.render()) + 2
        if used + cost > budget:
            continue  # a shorter, lower-ranked chunk may still fit
        blocks.append(block)
        used += cost

    if expand_neighbors:
        included = {b.hit.chunk_id for b in blocks}
        for block in blocks:
            for side in ("before", "after"):
                nid = block.hit.prev_id if side == "before" else block.hit.next_id
                ntext = block.hit.prev_text if side == "before" else block.hit.next_text
                if not nid or not ntext or nid in included:
                    continue
                cost = count_tokens(ntext) + 1
                if used + cost > budget:
                    continue
                setattr(block, side, ntext)
                included.add(nid)
                used += cost

    text = "\n\n".join(b.render() for b in blocks)
    return PackedChunks(
        text=text,
        chunks=[b.hit for b in blocks],
        citations=[Citation(id=b.citation, kind="chunk", ref=b.hit.chunk_id) for b in blocks],
        token_count=count_tokens(text) if text else 0,
    )


def pack_facts(facts: Sequence[Fact], *, budget: int) -> PackedFacts:
    lines: list[str] = []
    kept: list[Fact] = []
    citations: list[Citation] = []
    used = 0
    for fact in facts:
        cid = f"F{len(kept) + 1}"
        line = f"[{cid}] {fact.render()}"
        cost = count_tokens(line) + 1
        if used + cost > budget:
            continue
        lines.append(line)
        kept.append(fact)
        citations.append(Citation(id=cid, kind="fact", ref=fact_ref(fact)))
        used += cost
    text = "\n".join(lines)
    return PackedFacts(
        text=text, facts=kept, citations=citations, token_count=count_tokens(text) if text else 0
    )


def fact_ref(fact: Fact) -> str:
    if fact.kind == "entity":
        return fact.subject_id
    return f"{fact.subject_id}|{fact.predicate}|{fact.object_id}"


def assemble(
    question: str,
    mode: Mode,
    *,
    chunks: PackedChunks | None = None,
    facts: PackedFacts | None = None,
) -> RetrievedContext:
    """Join the packed sections into the context handed to generation."""
    sections = []
    if facts is not None and facts.text:
        sections.append(f"{FACTS_HEADER}\n{facts.text}")
    if chunks is not None and chunks.text:
        sections.append(f"{CHUNKS_HEADER}\n\n{chunks.text}")
    text = "\n\n".join(sections)
    return RetrievedContext(
        question=question,
        mode_used=mode,
        chunks=chunks.chunks if chunks else [],
        facts=facts.facts if facts else [],
        citations=[*(facts.citations if facts else []), *(chunks.citations if chunks else [])],
        prompt_context=text,
        token_count=count_tokens(text) if text else 0,
    )
