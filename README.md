# rag_with_oag

RAG, ontology-augmented generation (OAG) and a hybrid of the two, running over
**one FalkorDB graph**. Chunks, entities and ontology relations live in the same
graph, so a hybrid query can hand the LLM typed facts *and* the source text
behind them in a single context.

> **Status: v0.2 (pre-alpha).** All three modes work end to end: `rag`, `oag` and
> `hybrid`. Ontology drafting, streaming, re-ingest of changed documents and
> re-embedding are planned for v0.3.

## Install

```bash
pip install rag_with_oag            # embedded FalkorDB included, no server needed
pip install "rag_with_oag[pdf]"     # PDF loading
pip install "rag_with_oag[local]"   # local CPU embeddings (fastembed)
```

Python 3.12+. On macOS the embedded database needs OpenMP: `brew install libomp`.

## Quick start

```python
from rag_with_oag import KnowledgeBase, LLMConfig, EmbedConfig, Ontology

kb = KnowledgeBase(
    "contracts",
    path="./rag_with_oag.db",  # embedded FalkorDB (falkordblite)
    # url="redis://localhost:6379",  # FalkorDB server: change only this line
    llm=LLMConfig(base_url="http://localhost:11434/v1", model="qwen2.5:7b-instruct"),
    embedder=EmbedConfig(model="text-embedding-3-small"),
    ontology=Ontology.from_yaml("ontology.yaml"),  # omit for rag only
)

report = kb.ingest(["docs/"])  # txt, md, rst, html (+ pdf with the extra)
report.unmapped_concepts[:10]  # concepts the ontology could not express

res = kb.query("Which company owns the provider in contract HS-2024-01?")  # hybrid by default
print(res.answer)  # cites facts as [F1] and passages as [C1]
res.facts  # typed facts from the graph
res.chunks  # source passages
res.mode_used  # "hybrid", or "rag" if no entity in the question could be linked

ctx = kb.retrieve("...", mode="oag", hops=2)  # retrieval only, no LLM call
kb.close()
```

Any OpenAI-compatible endpoint works for the LLM, the extraction model
(`extractor_llm=`, defaults to `llm`) and the embedder: OpenAI, vLLM, Ollama,
LM Studio, OpenRouter, Groq. Use `AsyncKnowledgeBase` for the same API with `await`.

API keys: pass `api_key=` or set `RAG_OAG_LLM_API_KEY` / `RAG_OAG_EMBED_API_KEY`.
`OPENAI_API_KEY` is used only when `base_url` points at OpenAI itself; it is never
sent to OpenRouter, Groq or any other server.

Embedding models trained with instructions want them on every text, e.g.
`EmbedConfig(model="nomic-embed-text", query_prefix="search_query: ",
document_prefix="search_document: ")`. The document prefix is recorded with the
model, so a knowledge base built with one refuses to open without it.

For embeddings on the CPU without a server, install the `local` extra and pass
`embedder=FastEmbedEmbedder(...)` (from `rag_with_oag.embed.local`). The default
`BAAI/bge-small-en-v1.5` is small (67 MB) but English. For other languages a
multilingual model such as `Qwen/Qwen3-Embedding-0.6B-Q` (1.1 GB, Apache-2.0) is
the safer choice; its prefixes are filled in automatically.

## Ontology

Write it in YAML, or build it from Pydantic classes with `Ontology.from_classes`.
Type names become graph labels, so they must be ASCII identifiers. Descriptions
can be in any language: they are what the extraction model reads.

```yaml
name: contracts
entities:
  Sirket:
    description: A company that is party to contracts or owns shares in other companies
    key: vergi_no            # identifying property: same tax number = same entity
    properties:
      vergi_no: string
      sehir: {type: string, description: City of the head office}
  Sozlesme:
    key: numara
    properties:
      numara: string
      imza_tarihi: date
relations:
  TARAFIDIR:
    source: Sirket
    target: Sozlesme
    properties:
      rol: {type: enum, values: [musteri, saglayici]}
  ORTAGIDIR: {source: Sirket, target: Sirket, properties: {oran: number}}
```

The ontology's fingerprint is stored in the graph. Opening a knowledge base with
a changed ontology raises `OntologyMismatchError` instead of mixing two schemas.
Adding an ontology to a knowledge base that was rag-only extracts the existing
documents on the next `ingest`.

## Modes

| Mode | Entry point | Context for the LLM |
|---|---|---|
| `rag` | chunk vector search | passages, plus neighbouring chunks within the budget |
| `oag` | entity linking: names in the question + entity vectors | typed facts from walking the ontology's relations (`hops`, default 2) |
| `hybrid` | both, in parallel | facts **and** the passages that mention those entities, fused with RAG hits by Reciprocal Rank Fusion |

## CLI

```bash
export RAG_OAG_EMBED_MODEL=text-embedding-3-small
export RAG_OAG_LLM_MODEL=gpt-4o-mini       # *_BASE_URL and *_API_KEY as needed
rag-with-oag ingest docs/ --kb contracts --ontology ontology.yaml
rag-with-oag query "Who owns the provider?" --kb contracts --ontology ontology.yaml
rag-with-oag query "..." --kb contracts --mode rag --retrieve-only --json
```

## How it works

Ingest, per document:

1. **Load.** Files become `Document`s whose id is the content hash, so
   re-ingesting unchanged content is a no-op. **An edited file is a new
   document:** until v0.3 its old version stays in the graph and is still
   retrieved. After editing documents, rebuild under a new knowledge base name
   (or `await kb.store.drop()` with `AsyncKnowledgeBase`) and ingest again.
2. **Chunk.** [adaptive-oci-chunking](https://pypi.org/project/adaptive-oci-chunking/)
   picks the best strategy per document. Each chunk keeps its section path and page.
3. **Embed.** The embedding model and dimension are recorded in the graph. Opening
   the knowledge base with a different embedder is an error, not silent corruption.
4. **Store.** `Document -[:HAS_CHUNK]-> Chunk -[:NEXT]-> Chunk`. rag mode works from here.
5. **Extract.** Chunks go to the model in batches with a JSON Schema generated
   from the ontology. Structured output falls back from `json_schema` to tool
   calling, `json_object`, and finally prompt-only JSON, remembering per model what
   works.
6. **Validate.** Types, relation endpoints, required properties and enum values are
   checked. Invalid output goes back to the model once with the errors; whatever is
   still invalid is dropped and reported.
7. **Resolve.** Ids are deterministic from (type, key value) or (type, folded name).
   A mention without its key joins the keyed entity of the same name. Remaining
   key-less newcomers merge with near-duplicates by embedding similarity.
8. **Embed entities**, then 9. **write** them as `Entity:<Type>` nodes,
   `Chunk -[:MENTIONS]-> Entity`, and ontology relations whose `evidence` lists the
   source chunks.
10. **Report.** `IngestReport` covers documents, chunks, entities, relations,
    rejected extraction items, unmapped concepts and token usage.

Extraction results are cached per (chunk text, ontology, model, prompt). A document whose
extraction failed keeps its chunks and is retried on the next ingest.

`kb.store.query(cypher, params)` runs raw Cypher. It is documented but sits
outside the API stability promise. Inside the library, labels and relationship
types reach Cypher only from the validated ontology, and every other value is
passed as a parameter.

## Evaluation

`eval/golden` holds a small contract corpus (14 documents), its ontology, and 33
questions: single-hop, multi-hop and free-text. The runner compares the modes:

```bash
python eval/run_golden.py --base-url http://localhost:11434/v1 \
    --llm qwen2.5:7b-instruct --embed nomic-embed-text --answers --equal-budget
```

Hybrid hands the model more context than rag at the same `k`. `--equal-budget`
adds `rag_eq`, rag with `k` raised per question until its context is as large as
hybrid's, so the gate compares the modes at an equal token budget too.

## Development

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                 # embedded FalkorDB and fake models: no network or Docker
.venv/bin/ruff check . && .venv/bin/mypy
python spikes/m0_falkordb.py     # M0 backend spike: recall, latency, memory
```

## Roadmap

| Milestone | Scope |
|---|---|
| M0 | FalkorDB spike: ANN recall, hybrid latency, memory |
| M1 (v0.1) | Skeleton and RAG |
| **M2 (v0.2)** | Ontology, extraction, OAG and hybrid modes, golden set ← *you are here* |
| M3 (v0.3) | `Ontology.suggest()`, streaming, re-ingest and delete, `reembed()`, observability |

MIT licensed.
