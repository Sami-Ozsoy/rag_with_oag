# Configuration

[← Back to README](../README.md)

## Models

Any OpenAI-compatible endpoint works for the answer model (`llm`), the extraction
model (`extractor_llm`, defaults to `llm`) and the embedder. Examples include OpenAI,
vLLM, Ollama, LM Studio, OpenRouter and Groq.

- **API keys.** Pass `api_key=` or set `RAG_OAG_LLM_API_KEY` /
  `RAG_OAG_EMBED_API_KEY`. `OPENAI_API_KEY` is used only when `base_url` points at
  OpenAI itself.
- **Ollama.** Raise the context window. Ollama serves 4,096 tokens by default,
  which a hybrid context alone can fill (`OLLAMA_CONTEXT_LENGTH=16384`, or
  `PARAMETER num_ctx 16384` in the model file). A model that runs out before
  answering raises `OutputTruncatedError`.
  Set `LLMConfig(..., reasoning_effort="none")` to turn off thinking on Ollama reasoning models.
- **Embedding prefixes.** Models trained with instructions need them on every
  text, e.g. `EmbedConfig(model="nomic-embed-text", query_prefix="search_query: ",
  document_prefix="search_document: ")`. The prefix is recorded with the knowledge
  base, which then refuses to open without it.
- **Local embeddings.** With the `local` extra, pass
  `embedder=FastEmbedEmbedder(...)` from `rag_with_oag.embed.local`. The default
  `BAAI/bge-small-en-v1.5` (67 MB) is English only. For other languages use a
  multilingual model such as `Qwen/Qwen3-Embedding-0.6B-Q` (1.1 GB, Apache-2.0),
  whose prefixes are filled in automatically.

The graph and passages share one embedded FalkorDB file. To use a FalkorDB
server, pass `url="redis://localhost:6379"` in place of `path`.

## Tuning

Pass `retrieval=RetrievalConfig(...)` to `KnowledgeBase`. The defaults suit most
cases.

| Setting | Default | Meaning |
|---|---|---|
| `context_token_budget` | 6000 | Upper limit for the whole context, headers included |
| `hops` | 3 | How many relations away from the question's entities the graph walk goes |
| `max_fanout` | 8 | Edges kept per entity, relation type and direction, so 30 contracts cannot crowd out 2 shareholders |
| `fact_budget_ratio` | 0.6 | Maximum share of the budget for graph facts in hybrid |
| `hybrid_text_reserve` | 4 | Passages hybrid always keeps ahead of facts |
| `link_exact_anchors` | `True` | A key or full name written in the question makes that entity the only starting point |
| `rag_lexical` | `True` | Fuse BM25 keyword ranks with vector ranks for RAG passages (in `rag` and `hybrid`) |
| `rerank_candidates` | 30 | Passages a reranker rescores, when one is passed |

To try the optional cross-encoder reranker, pass
`KnowledgeBase(..., reranker=FastEmbedReranker())` using `FastEmbedReranker` from
`rag_with_oag.rerank.local` (the `local` extra). It runs on CPU and did not help
on our evaluation sets, so it is off by default.

## CLI

```bash
export RAG_OAG_EMBED_MODEL=text-embedding-3-small
export RAG_OAG_LLM_MODEL=gpt-4o-mini       # *_BASE_URL and *_API_KEY as needed
rag-with-oag suggest docs/ -o ontology.yaml
rag-with-oag ingest docs/ --kb contracts --ontology ontology.yaml
rag-with-oag query "Who owns the provider?" --kb contracts --ontology ontology.yaml
rag-with-oag query "..." --kb contracts --mode rag --retrieve-only --json
```

## Raw Cypher

`kb.store.query(cypher, params)` runs raw Cypher. It is documented but outside
the API stability promise. Inside the library, labels and relationship types
reach Cypher only from the validated ontology. Every other value is a parameter.
