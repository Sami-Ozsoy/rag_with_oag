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
