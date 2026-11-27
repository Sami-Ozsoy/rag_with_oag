# Ontology

[← Back to README](../README.md)

The ontology tells the extraction model which entity types, properties, and
relations to look for. Write it in YAML, or build it from Pydantic classes with
`Ontology.from_classes`.

```yaml
name: contracts
entities:
  Sirket:                    # company
    description: A company that is party to contracts or owns shares in other companies
    key: vergi_no            # identifying property: same tax number = same company
    properties:
      vergi_no: string
      sehir: {type: string, description: City of the head office}
  Sozlesme:                  # contract
    key: numara
    properties:
      numara: string
      imza_tarihi: date
relations:
  TARAFIDIR:                 # company is a party to a contract
    source: Sirket
    target: Sozlesme
    properties:
      rol: {type: enum, values: [musteri, saglayici]}
  ORTAGIDIR: {source: Sirket, target: Sirket, properties: {oran: number}}  # owns shares
```

- Type names become graph labels, so they must be ASCII identifiers.
  The extraction model reads the descriptions, which can be in any language.
- A `key` is what makes two mentions the same entity. Without one, names are
  compared, ignoring legal forms such as "A.Ş.", "Ltd." or "GmbH" (extend the list
  with `name_suffixes`).
- The ontology's fingerprint is stored in the graph. Opening a knowledge base with
  a changed ontology raises `OntologyMismatchError` instead of mixing two schemas.
  Adding an ontology to a RAG-only knowledge base extracts the existing documents
  on the next `ingest`.

## Ontology studio

```bash
pip install 'rag_with_oag[ui]'
rag-with-oag ui docs/ --ontology ontology.yaml --questions questions.txt
```

- **Suggest** drafts or extends a schema from sample documents and can repair gaps
  against the questions you provide.
- **Schema** and **YAML** let you edit the ontology, with validation in Problems.
- **Preview** runs sample extraction and marks unused types, low key fill, and
  endpoint mismatches on the graph.
- **Check questions** reports which questions the graph covers. Select a result
  to highlight its path.
- **Save** writes the validated YAML. An optional `--ontology` file can be an
  existing schema or a new save target.

Studio serves the page on loopback only and accepts files inside the current
working directory. Open the full URL printed by the CLI, including its session
token in the URL fragment. Studio removes the fragment and keeps the token for
reloads in the same tab. Model actions use the same `RAG_OAG_LLM_*` settings
as the CLI, and the API key stays on the server. Use `--no-browser` to open the
URL yourself. The graph uses bundled Cytoscape.js (MIT).
Studio checks document paths before loading them and assumes local files and
symlinks remain stable between the check and the read.

## Suggesting an ontology

```python
from rag_with_oag import LLMConfig, Ontology

llm = LLMConfig(model="gpt-4o-mini")
ontology = Ontology.suggest(["docs/"], llm=llm)
ontology.to_yaml("ontology.yaml")
```

To extend an existing ontology, pass it with the unmapped concepts from an
ingest report.

```python
extended = Ontology.suggest(["docs/"], llm=llm, base=ontology, unmapped=report.unmapped_concepts)
```

```bash
rag-with-oag suggest docs/ -o ontology.yaml
```

The model sees a sample of about `sample_tokens` tokens from the start of the
documents, spread across them. Review and edit the suggestion before ingesting.
Extending an ontology changes its fingerprint. A knowledge base built with the
old ontology raises `OntologyMismatchError` when opened with the extension.
Rebuild it under a new name.

## Designing from questions

Hybrid mode uses passages for single details and the graph for questions that
connect, combine or compare entities. Give `suggest` both kinds of questions so
it can focus the ontology on graph questions. Use one question per line
in a text file. Blank lines and lines starting with `#` are ignored. YAML files
can contain a list of strings or grouped entries with a `q` or `question` field.
Questions are deduplicated in file order. `suggest` uses the first 50.

```bash
rag-with-oag suggest docs/ --questions questions.txt -o ontology.yaml
```

The model routes each question to `graph` or `text` and proposes a path where
the ontology can express one. The library checks those paths against the ontology.
`suggest` tries one repair round for coverage gaps, preserving existing definitions,
then checks the result again. Text-only questions need no ontology change. Use
`--no-repair` to skip that round. The coverage summary is printed to stderr.

Check a hand-written or edited ontology with the same questions. Use `--json`
for the full report.

```bash
rag-with-oag check-questions --ontology ontology.yaml --questions questions.txt --hops 3
```

- `covered` means the proposed path uses declared types, relations and properties
  within the hop limit. A property of the starting entity takes zero hops.
- `text` means the model routed the question to passages and no valid path fit
  the hop limit. The report shows the model's routing reason so you can spot
  misrouted questions.
- `not_covered` means the model found no path for a graph question or omitted it.
- `invalid_path` means the proposed path for a graph question violates the
  ontology, for example by traversing a relation from the wrong entity type.
- `too_many_hops` means the graph path is valid but exceeds `--hops` (default 3).

Coverage checks the schema. Use preview to see what extraction finds in documents.

```python
from rag_with_oag import LLMConfig, Ontology

llm = LLMConfig(model="gpt-4o-mini")
questions = ["Which company signed contract C-1?", "Where is the signer of contract C-1 based?"]
ontology = Ontology.suggest(["docs/"], llm=llm, questions=questions)
report = ontology.check_questions(questions, llm=llm, hops=3)
```

In async code, use `await Ontology.asuggest(...)` and
`await ontology.acheck_questions(...)`.

## Previewing an ontology

Preview runs the ingest chunker and extractor on an evenly spread sample of
documents and chunks. It needs only the chat model. It writes nothing, opens no
database, and skips embeddings, the extraction cache, and entity resolution.

```python
from pathlib import Path

from rag_with_oag import LLMConfig, Ontology

ontology = Ontology.from_yaml(Path("ontology.yaml"))
llm = LLMConfig(model="gpt-4o-mini")
report = ontology.preview(["docs/"], llm=llm, documents=5, chunks_per_doc=6)
```

In async code, use `await ontology.apreview(...)` with the same arguments.

```bash
rag-with-oag preview docs/ --ontology ontology.yaml --docs 5 --chunks-per-doc 6
```

Use `--json` for the full report, including property fill rates, validation
examples, failed batches, unmapped concepts, and token usage. `documents_available`
counts sources with text. Chunk coverage counts passages that mention an entity
or support a relation. Mentions and relations count once per extracted item even
when supported by several chunks.

- `entity_types` and `relation_types` include every declared type. Property fill
  rates cover each declared property when the type has at least one match.
- `distinct` counts instances by key value when one was extracted, otherwise by
  normalized name. Low `key_fill_rate` means more mentions rely on name matching
  and may end up as separate nodes after ingest.
- An `endpoint_mismatch` hint shows the observed source and target types alongside
  the pair the ontology allows. Check the relation direction and type descriptions
  against the sample before editing the definition.

Use the loop **suggest --questions → preview → edit → check-questions / preview → ingest**
to refine the ontology before building the knowledge base.
