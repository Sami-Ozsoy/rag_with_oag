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
- Write numbers in the unit requested by the property description. For percentages, write
  the percent value (40% -> 40, not 0.4).
- Relations must connect entities of the source and target types given for them.
- Read each relation as "<source> <RELATION> <target>" using its description. Write it only
  if that sentence is true according to the text.
- Even if the text phrases it the other way round (passive voice, a genitive construction,
  or "Y's shareholders include X"), the source is the entity the description puts on the
  source side. Never add the reverse relation unless the text states it too.
- Example: for OWNS_SHARES: Company -> Company ("source holds shares of target"),
  "Beta's shareholders are Acme (30%) and Gamma (70%)" gives
  OWNS_SHARES(source=Acme, target=Beta, share=30) and
  OWNS_SHARES(source=Gamma, target=Beta, share=70). Do not write a relation from Beta
  to Acme or Gamma.
- Output JSON only."""

EXTRACTION_USER = """\
Passages:

{passages}"""

SUGGEST_SYSTEM = """\
You design an ontology that an extraction model will follow to build a knowledge graph
from these documents. The graph should support questions that chain facts across documents.

Rules:
- Prefer a small number (at most about 12) of general entity types that recur across the
  documents, and the relations the documents actually state. Do not model one-off details.
- A key is the property that uniquely identifies an instance (tax number, contract number,
  national ID, etc.). Set it only when the documents show such an identifier, otherwise null.
- Property types are string, integer, number, boolean, date and enum. Use enum only for
  a small closed set and list its members in values. Set values to null for other types.
  Mark a property required only when every instance states it.
- Use PascalCase for entity types, UPPER_SNAKE_CASE for relation types and lower_snake_case
  for properties. All names must be ASCII identifiers ([A-Za-z_][A-Za-z0-9_]*).
  Transliterate non-ASCII letters (ş→s, ğ→g, ı→i, ö→o, ü→u, ç→c).
- Never use these reserved names: {reserved}.
- The extraction model reads the descriptions: write them in the language of the
  documents and say what to extract. Phrase each relation description as
  "<source> <RELATION> <target>" so the direction is unambiguous.
- In name_suffixes, list only legal-form suffixes seen in entity names that are absent
  from the defaults ({default_suffixes}). Usually this is an empty list.
- Output JSON only."""

SUGGEST_USER = """\
Sample documents:

{documents}"""

SUGGEST_EXTEND = """\
Existing ontology:

{ontology}

Unmapped concepts the current ontology could not express during extraction (with counts):
{unmapped}

Return the full ontology. Keep every existing entity type, relation, key and property.
Add only what the documents or the unmapped concepts need."""
