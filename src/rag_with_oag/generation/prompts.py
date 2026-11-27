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

SUGGEST_QUESTIONS = """\
Questions to consider when designing the knowledge graph:
{questions}

Design the graph for questions that connect, combine or compare entities. Include the entity
types, correctly directed relations and properties those questions need.
Passage search answers single details stated in one passage. Do not add types or properties
just for those questions. Never add long free-text properties such as clauses, conditions,
explanations, event lists or formulas. Properties should be short facts such as identifiers,
names, dates, numbers and small enums that recur across documents.
Prefer general types over question-specific ones."""

SUGGEST_REPAIR = """\
Existing ontology:

{ontology}

Uncovered questions and coverage findings:
{questions}

Return the full ontology. Keep every existing entity type, relation, key and property.
Add only what the uncovered questions need."""

QUESTIONS_SYSTEM = """\
Route each question as graph or text and give a one-sentence route_reason.
- graph: the answer needs relations connecting two or more entities (the partner of the company
  that supplies X, the parties of contract C, where the customers of X are based), combines facts
  about several entities (lists, counts or which of several entities), or compares entities.
- text: the answer is a single detail about one entity or document, stated directly in a passage
  (amounts, dates, durations, conditions, clauses, explanations or descriptions).

Propose the shortest graph path that can answer the question using the ontology. For text
questions, still propose a path when the ontology already expresses one, such as a declared
property.
A path starts at an entity type (start_type). For each step, give the relation name and direction:
forward (source -> target) or backward (target -> source). Set answer.type to the final entity
type and answer.property to the requested property, or null when the answer is the final entity
itself (its name). A question about a property of the starting entity takes zero steps.
Use only entity, relation and property names from the ontology, or name for the entity's name.
Use the question's 1-based index from this batch. If no path can answer it, set answerable to false,
start_type and answer to null, and steps to an empty list. Describe the missing types, relations
or properties in missing. For an answerable question, set missing to an empty string.
Output JSON only."""

QUESTIONS_USER = """\
Ontology:

{ontology}

Questions:

{questions}"""
