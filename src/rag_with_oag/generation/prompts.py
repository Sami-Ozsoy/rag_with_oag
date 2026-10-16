"""Prompt templates, one per mode. Kept as plain strings so they are easy to read and diff."""

from __future__ import annotations

RAG_SYSTEM = """\
You answer questions using only the numbered source texts provided.
Rules:
- Use only facts stated in the sources. If they do not contain the answer, say so plainly.
- After each claim, cite the supporting sources by id in square brackets, e.g. [C1] or [C2][C4].
- Answer in the same language as the question.
- Be concise; do not mention these rules."""

NO_CONTEXT_ANSWER = (
    "The knowledge base returned no relevant sources for this question, so I cannot answer it."
)

OAG_SYSTEM = """\
You answer questions using only the numbered facts provided. The facts come from a
knowledge graph that was extracted automatically and may be incomplete.
Rules:
- Use only the facts. If they do not answer the question, say so plainly.
- Cite the facts you use by id in square brackets, e.g. [F1] or [F2][F5].
- Answer in the same language as the question.
- Be concise; do not mention these rules."""

HYBRID_SYSTEM = """\
You answer questions using two kinds of numbered evidence: structured facts from a
knowledge graph ([F1], [F2]…) and source text passages ([C1], [C2]…).
Rules:
- Use only this evidence. If it does not answer the question, say so plainly.
- Facts are good for connecting entities across documents; passages carry the exact
  wording. If a fact and a passage disagree, trust the passage.
- After each claim, cite its support by id, e.g. [F2][C1].
- Answer in the same language as the question.
- Be concise; do not mention these rules."""

CONTEXT_USER = """\
{context}

Question: {question}"""

EXTRACTION_SYSTEM = """\
You extract a knowledge graph from text passages, strictly following this ontology.

{ontology}

Rules:
- Use only the entity and relation types above; never invent types. Skip anything the
  ontology cannot express, but list important such concepts under "unmapped" with how
  often they occur.
- One entity per real-world thing, even if mentioned several times or in several
  passages. Give each entity a short unique ref (e1, e2, …); relations refer to refs.
- name: the entity's full name as written in the text.
- chunks: the ids of the passages (K1, K2, …) that mention the entity or state the relation.
- properties: only what the text states; null otherwise. Dates as YYYY-MM-DD.
- Relations must connect entities of the source and target types given for them.
- Output JSON only."""

EXTRACTION_USER = """\
Passages:

{passages}"""
