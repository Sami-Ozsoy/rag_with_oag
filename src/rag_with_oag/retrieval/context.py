"""Pack retrieved material into prompt sections under a token budget.

Chunks get citation ids ``C1, C2…`` and facts ``F1, F2…``; answers cite them.
Items are considered in the order given. OAG facts arrive by hop, with each
new entity's properties next to the relation that reaches it, so nearby facts
get first use of the budget. Neighbour chunks (``NEXT`` edges) are added after
primary hits while budget remains.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rag_with_oag._tokens import count_tokens
from rag_with_oag.types import ChunkHit, Citation, Fact, Mode, RetrievedContext

FACTS_HEADER = "Structured facts:"
CHUNKS_HEADER = "Source texts:"
_FACTS_PREFIX = f"{FACTS_HEADER}\n"
_CHUNKS_PREFIX = f"{CHUNKS_HEADER}\n\n"
_SECTION_SEPARATOR = "\n\n"


@dataclass(slots=True)
class _Block:
    hit: ChunkHit
    citation: str
    before: str | None = None
    after: str | None = None

    def render(self) -> str:
        body = "\n".join(t for t in (self.before, self.hit.text, self.after) if t)
        return f"[{self.citation}] {self.hit.location}\n{body}"

    def primary_cost(self) -> int:
        return count_tokens(self.render()) + 2


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


class _ChunkPacker:
    def __init__(self) -> None:
        self.blocks: list[_Block] = []
        self.used = 0

    def add_hits(
        self,
        hits: Sequence[ChunkHit],
        *,
        budget: int,
        fits: Callable[[str], bool] | None = None,
    ) -> None:
        for hit in hits:
            block = _Block(hit=hit, citation=f"C{len(self.blocks) + 1}")
            cost = block.primary_cost()
            if self.used + cost > budget:
                continue
            self.blocks.append(block)
            if fits is not None and not fits(self.text()):
                self.blocks.pop()
                continue
            self.used += cost

    def add_neighbors(self, *, budget: int, fits: Callable[[str], bool] | None = None) -> None:
        included = {block.hit.chunk_id for block in self.blocks}
        for block in self.blocks:
            for side in ("before", "after"):
                nid = block.hit.prev_id if side == "before" else block.hit.next_id
                ntext = block.hit.prev_text if side == "before" else block.hit.next_text
                if not nid or not ntext or nid in included:
                    continue
                cost = count_tokens(ntext) + 1
                if self.used + cost > budget:
                    continue
                setattr(block, side, ntext)
                if fits is not None and not fits(self.text()):
                    setattr(block, side, None)
                    continue
                included.add(nid)
                self.used += cost

    def text(self) -> str:
        return "\n\n".join(block.render() for block in self.blocks)

    def packed(self) -> PackedChunks:
        text = self.text()
        return PackedChunks(
            text=text,
            chunks=[block.hit for block in self.blocks],
            citations=[
                Citation(id=block.citation, kind="chunk", ref=block.hit.chunk_id)
                for block in self.blocks
            ],
            token_count=count_tokens(text),
        )


def pack_chunks(hits: Sequence[ChunkHit], *, budget: int, expand_neighbors: bool) -> PackedChunks:
    packer = _ChunkPacker()
    packer.add_hits(hits, budget=budget)
    if expand_neighbors:
        packer.add_neighbors(budget=budget)
    return packer.packed()


def pack_facts(
    facts: Sequence[Fact], *, budget: int, fits: Callable[[str], bool] | None = None
) -> PackedFacts:
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
        if fits is not None and not fits("\n".join(lines)):
            lines.pop()
            continue
        kept.append(fact)
        citations.append(Citation(id=cid, kind="fact", ref=fact_ref(fact)))
        used += cost
    text = "\n".join(lines)
    return PackedFacts(
        text=text, facts=kept, citations=citations, token_count=count_tokens(text) if text else 0
    )


def _context_text(facts: str, chunks: str) -> str:
    sections = []
    if facts:
        sections.append(_FACTS_PREFIX + facts)
    if chunks:
        sections.append(_CHUNKS_PREFIX + chunks)
    return _SECTION_SEPARATOR.join(sections)


def _pack_context_facts(
    facts: Sequence[Fact], *, budget: int, context_budget: int, chunks: str = ""
) -> PackedFacts:
    packed = pack_facts(facts, budget=budget)
    packed_context = _context_text(packed.text, chunks)
    if packed.token_count <= budget and count_tokens(packed_context) <= context_budget:
        return packed

    def fits(text: str) -> bool:
        return count_tokens(text) <= budget and (
            count_tokens(_context_text(text, chunks)) <= context_budget
        )

    return pack_facts(facts, budget=budget, fits=fits)


def pack_context(
    question: str,
    mode: Mode,
    *,
    budget: int,
    hits: Sequence[ChunkHit] = (),
    facts: Sequence[Fact] = (),
    reserve: int = 0,
    fact_budget_ratio: float = 1.0,
    expand_neighbors: bool = False,
) -> RetrievedContext:
    """Pack context with room for the sections that appear in the prompt."""
    chunks = _ChunkPacker()
    chunk_budget = budget - count_tokens(_CHUNKS_PREFIX)
    fact_header_cost = count_tokens(_FACTS_PREFIX)
    fact_text = ""

    def fits_chunks(text: str) -> bool:
        return count_tokens(_context_text(fact_text, text)) <= budget

    chunks.add_hits(hits[:reserve], budget=chunk_budget, fits=fits_chunks)
    packed_facts = None
    if facts and hits:
        shared_budget = chunk_budget - fact_header_cost - count_tokens(_SECTION_SEPARATOR)
        fact_budget = max(
            0, min(int(shared_budget * fact_budget_ratio), shared_budget - chunks.used)
        )
        packed_facts = _pack_context_facts(
            facts, budget=fact_budget, context_budget=budget, chunks=chunks.text()
        )
        if packed_facts.text:
            fact_text = packed_facts.text
            chunk_budget = shared_budget - packed_facts.token_count

    chunks.add_hits(hits[reserve:], budget=chunk_budget, fits=fits_chunks)
    if chunks.blocks:
        if expand_neighbors:
            chunks.add_neighbors(budget=chunk_budget, fits=fits_chunks)
        return assemble(question, mode, facts=packed_facts, chunks=chunks.packed())

    if not facts:
        return assemble(question, mode)

    fact_budget = max(0, int((budget - fact_header_cost) * fact_budget_ratio))
    packed_facts = _pack_context_facts(facts, budget=fact_budget, context_budget=budget)
    return assemble(question, mode, facts=packed_facts)


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
    text = _context_text(facts.text if facts else "", chunks.text if chunks else "")
    return RetrievedContext(
        question=question,
        mode_used=mode,
        chunks=chunks.chunks if chunks else [],
        facts=facts.facts if facts else [],
        citations=[*(facts.citations if facts else []), *(chunks.citations if chunks else [])],
        prompt_context=text,
        token_count=count_tokens(text) if text else 0,
    )
