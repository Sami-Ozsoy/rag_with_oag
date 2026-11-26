"""Ingest a contract and a shareholder record, then answer a question with citations.

Start Ollama with `ollama serve` in a separate terminal.
    ollama pull qwen2.5:7b-instruct && ollama pull bge-m3
    python examples/quickstart/run.py

For OpenAI, set OPENAI_API_KEY and export these before the first run.
    export RAG_OAG_LLM_BASE_URL=https://api.openai.com/v1 RAG_OAG_LLM_MODEL=gpt-4o-mini
    export RAG_OAG_EMBED_BASE_URL=https://api.openai.com/v1
    export RAG_OAG_EMBED_MODEL=text-embedding-3-small
"""

import os
from pathlib import Path

from rag_with_oag import EmbedConfig, KnowledgeBase, LLMConfig, Ontology, RagOagError
from rag_with_oag.retrieval.context import fact_ref


def main() -> None:
    here = Path(__file__).resolve().parent
    os.environ.setdefault("RAG_OAG_LLM_BASE_URL", "http://localhost:11434/v1")
    os.environ.setdefault("RAG_OAG_EMBED_BASE_URL", "http://localhost:11434/v1")
    os.environ.setdefault("RAG_OAG_LLM_MODEL", "qwen2.5:7b-instruct")
    os.environ.setdefault("RAG_OAG_EMBED_MODEL", "bge-m3")

    with KnowledgeBase(
        "contracts",
        path=here / "quickstart.db",
        ontology=Ontology.from_yaml(here / "ontology.yaml"),
        llm=LLMConfig(),  # type: ignore[call-arg]  # filled from env
        embedder=EmbedConfig(),  # type: ignore[call-arg]  # filled from env
    ) as kb:
        report = kb.ingest(here / "docs")
        print(
            f"Ingested {report.documents_processed} documents "
            f"({report.documents_skipped} unchanged, skipped)."
        )
        result = kb.query(
            "Which city is the company that owns the provider in contract HS-2024-01 based in?"
        )
        print(result.answer)
        sources = {fact_ref(fact): fact.render() for fact in result.facts}
        sources.update({hit.chunk_id: f"{hit.location}\n{hit.text}" for hit in result.chunks})
        for citation in result.citations:
            print(f"[{citation.id}] {sources[citation.ref]}")


if __name__ == "__main__":
    try:
        main()
    except RagOagError as exc:
        raise SystemExit(f"error: {exc}") from None
