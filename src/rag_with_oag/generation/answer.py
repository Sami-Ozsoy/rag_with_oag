from __future__ import annotations

import re

from rag_with_oag.generation import prompts
from rag_with_oag.llm.client import ChatModel, Message
from rag_with_oag.types import Citation, QueryResult, RetrievedContext

_CITATION = re.compile(r"\[((?:C|F)\d+)\]")


SYSTEM_PROMPTS = {
    "rag": prompts.RAG_SYSTEM,
    "oag": prompts.OAG_SYSTEM,
    "hybrid": prompts.HYBRID_SYSTEM,
}


def build_messages(ctx: RetrievedContext) -> list[Message]:
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[ctx.mode_used]},
        {
            "role": "user",
            "content": prompts.CONTEXT_USER.format(
                context=ctx.prompt_context, question=ctx.question
            ),
        },
    ]


def cited(answer: str, available: list[Citation]) -> list[Citation]:
    """Citations the answer actually uses, in first-mention order; invented ids are dropped."""
    by_id = {c.id: c for c in available}
    seen: dict[str, Citation] = {}
    for match in _CITATION.finditer(answer):
        cid = match.group(1)
        if cid in by_id and cid not in seen:
            seen[cid] = by_id[cid]
    return list(seen.values())


async def answer(llm: ChatModel, ctx: RetrievedContext) -> QueryResult:
    before = dict(llm.usage)
    if not ctx.prompt_context:
        text = prompts.NO_CONTEXT_ANSWER
    else:
        text = (await llm.complete(build_messages(ctx))).strip()
    return QueryResult(
        answer=text,
        mode_used=ctx.mode_used,
        chunks=ctx.chunks,
        facts=ctx.facts,
        citations=cited(text, ctx.citations),
        usage={k: v - before.get(k, 0) for k, v in llm.usage.items() if v - before.get(k, 0)},
    )
