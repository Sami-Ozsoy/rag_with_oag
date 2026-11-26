"""OpenAI-compatible clients against an in-process fake server.

openai>=3 sends requests through httpx2, which respx does not patch, so the
fake server is an httpx2 MockTransport handed to the SDK client.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import httpx2
import openai
import pytest
from openai import AsyncOpenAI

from rag_with_oag import (
    Document,
    EmbedConfig,
    ExtractionConfig,
    KnowledgeBase,
    LLMConfig,
    ModelAuthError,
    ModelConnectionError,
    ModelNotFoundError,
    Ontology,
    RagOagError,
    cli,
)
from rag_with_oag.config import DEFAULT_BASE_URL
from rag_with_oag.embed.openai_compat import OpenAIEmbedder
from rag_with_oag.ingest.chunker import TokenWindowChunker
from rag_with_oag.ingest.graph_builder import GraphBuilder
from rag_with_oag.ingest.pipeline import IngestPipeline
from rag_with_oag.llm.client import LLMClient
from rag_with_oag.store.repository import DocumentState, GraphRepository

from .conftest import FakeEmbedder

BASE = "http://llm.test/v1"
Handler = Callable[[httpx2.Request], httpx2.Response]
MESSAGES = [{"role": "user", "content": "hello"}]


def fake_server(
    handler: Handler, *, max_retries: int = 0, api_key: str = "test", base_url: str = BASE
) -> tuple[AsyncOpenAI, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request)

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        max_retries=max_retries,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(record)),
    )
    return client, seen


def completion(content: str, usage: dict[str, int] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
    if usage:
        body["usage"] = {**usage, "total_tokens": sum(usage.values())}
    return body


async def test_llm_client_completes_and_counts_usage() -> None:
    sdk, seen = fake_server(
        lambda r: httpx2.Response(
            200, json=completion("merhaba [C1]", {"prompt_tokens": 12, "completion_tokens": 3})
        )
    )
    client = LLMClient(LLMConfig(model="m", base_url=BASE), client=sdk)
    assert await client.complete([{"role": "user", "content": "selam"}]) == "merhaba [C1]"
    sent = json.loads(seen[-1].content)
    assert sent["model"] == "m" and sent["temperature"] == 0.0
    assert seen[-1].url.path == "/v1/chat/completions"
    assert client.usage == {"prompt_tokens": 12, "completion_tokens": 3}
    await client.aclose()


@pytest.mark.parametrize("effort", [None, "none"])
async def test_llm_client_reasoning_effort(effort: str | None) -> None:
    sdk, seen = fake_server(lambda _: httpx2.Response(200, json=completion("ok")))
    client = LLMClient(LLMConfig(model="m", base_url=BASE, reasoning_effort=effort), client=sdk)
    messages = [{"role": "user", "content": "x"}]
    await client.complete(messages)
    await client.complete(messages, reasoning_effort="high")
    default, override = (json.loads(request.content) for request in seen)
    if effort is None:
        assert "reasoning_effort" not in default
    else:
        assert default["reasoning_effort"] == effort
    assert override["reasoning_effort"] == "high"
    await client.aclose()


async def test_llm_client_retries_server_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter([httpx2.Response(503), httpx2.Response(200, json=completion("ok"))])
    sdk, seen = fake_server(lambda r: next(replies), max_retries=2)
    monkeypatch.setattr(sdk, "_calculate_retry_timeout", lambda *a, **k: 0.0)
    client = LLMClient(LLMConfig(model="m", base_url=BASE), client=sdk)
    assert await client.complete([{"role": "user", "content": "x"}]) == "ok"
    assert len(seen) == 2
    await client.aclose()


async def test_embedder_batches_and_reorders_by_index() -> None:
    def reply(request: httpx2.Request) -> httpx2.Response:
        inputs = json.loads(request.content)["input"]
        data = [
            {"object": "embedding", "index": i, "embedding": [float(len(t)), 1.0]}
            for i, t in enumerate(inputs)
        ]
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "model": "e",
                "data": list(reversed(data)),
                "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
            },
        )

    sdk, seen = fake_server(reply)
    embedder = OpenAIEmbedder(EmbedConfig(model="e", base_url=BASE, batch_size=2), client=sdk)
    vectors = await embedder.embed(["a", "bb", "ccc"])
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]
    assert len(seen) == 2
    assert embedder.usage["embedding_tokens"] == 3
    await embedder.aclose()


async def test_embedder_rejects_short_response() -> None:
    sdk, _ = fake_server(
        lambda r: httpx2.Response(
            200,
            json={
                "object": "list",
                "model": "e",
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0]}],
            },
        )
    )
    embedder = OpenAIEmbedder(EmbedConfig(model="e", base_url=BASE), client=sdk)
    with pytest.raises(RuntimeError, match="returned 1 vectors for 2"):
        await embedder.embed(["a", "b"])
    await embedder.aclose()


def endpoint_failure(failure: int | type[httpx2.TransportError]) -> Handler:
    def fail(request: httpx2.Request) -> httpx2.Response:
        if isinstance(failure, int):
            return httpx2.Response(
                failure, json={"error": {"message": request.headers.get("Authorization")}}
            )
        raise failure("server unavailable", request=request)

    return fail


@pytest.mark.parametrize("role", ["chat", "embedding"])
@pytest.mark.parametrize(
    ("failure", "cause", "error", "hint"),
    [
        (httpx2.ConnectError, openai.APIConnectionError, ModelConnectionError, "ollama serve"),
        (httpx2.ReadTimeout, openai.APITimeoutError, ModelConnectionError, "timed out"),
        (401, openai.AuthenticationError, ModelAuthError, "rejected the API key"),
        (403, openai.PermissionDeniedError, ModelAuthError, "rejected the API key"),
        (404, openai.NotFoundError, ModelNotFoundError, "ollama pull missing-model"),
    ],
)
async def test_model_endpoint_errors(
    role: str,
    failure: int | type[httpx2.TransportError],
    cause: type[openai.APIError],
    error: type[RagOagError],
    hint: str,
) -> None:
    key = "test-key-that-must-not-leak"
    model = "missing-model"
    sdk, seen = fake_server(endpoint_failure(failure), api_key=key)
    async with sdk:
        with pytest.raises(error, match=hint) as caught:
            if role == "chat":
                client = LLMClient(LLMConfig(model=model, base_url=BASE), client=sdk)
                await client.complete(MESSAGES)
            else:
                embedder = OpenAIEmbedder(EmbedConfig(model=model, base_url=BASE), client=sdk)
                await embedder.embed(["x"])
    assert isinstance(caught.value.__cause__, cause)
    message = str(caught.value)
    assert BASE in message and model in message and role in message.lower()
    assert key not in message
    if failure is httpx2.ReadTimeout:
        assert "timeout=" in message
    if error is ModelAuthError:
        prefix = "LLM" if role == "chat" else "EMBED"
        assert f"Pass api_key= or set RAG_OAG_{prefix}_API_KEY" in message
        assert "OPENAI_API_KEY" not in message
    assert len(seen) == 1


@pytest.mark.parametrize("role", ["chat", "embedding"])
@pytest.mark.parametrize("base_url", [BASE, DEFAULT_BASE_URL])
async def test_missing_model_key_message(role: str, base_url: str) -> None:
    sdk, _ = fake_server(endpoint_failure(401), base_url=base_url, api_key="not-needed")
    async with sdk:
        with pytest.raises(ModelAuthError, match="No API key is configured") as caught:
            if role == "chat":
                config = LLMConfig(model="m", base_url=base_url, api_key=None)
                await LLMClient(config, client=sdk).complete(MESSAGES)
            else:
                embed_config = EmbedConfig(model="m", base_url=base_url, api_key=None)
                await OpenAIEmbedder(embed_config, client=sdk).embed(["x"])
    message = str(caught.value)
    assert "not-needed" not in message
    assert ("OPENAI_API_KEY" in message) == (base_url == DEFAULT_BASE_URL)


@pytest.mark.parametrize("role", ["chat", "embedding"])
async def test_ingest_aborts_on_connection_error(
    role: str, ontology: Ontology, fake_embedder: FakeEmbedder
) -> None:
    sdk, seen = fake_server(endpoint_failure(httpx2.ConnectError))
    repo = Mock(spec=GraphRepository)
    repo.document_state.return_value = DocumentState(False, None, None)
    chunker = TokenWindowChunker()
    if role == "chat":
        llm = LLMClient(LLMConfig(model="m", base_url=BASE), client=sdk)
        builder = GraphBuilder(repo, llm, fake_embedder, ontology, ExtractionConfig(cache=False))
        pipeline = IngestPipeline(repo, fake_embedder, chunker, builder)
    else:
        embedder = OpenAIEmbedder(EmbedConfig(model="e", base_url=BASE), client=sdk)
        pipeline = IngestPipeline(repo, embedder, chunker)
    docs = [
        Document.from_text("First document", source="first.md"),
        Document.from_text("Second document", source="second.md"),
    ]
    async with sdk:
        with pytest.raises(ModelConnectionError, match=f"{role} model server") as caught:
            await pipeline.run(docs, concurrency=1)
    assert isinstance(caught.value.__cause__, openai.APIConnectionError)
    assert len(seen) == 1
    assert repo.write_document.await_count == (1 if role == "chat" else 0)
    repo.mark_extracted.assert_not_awaited()
    repo.replacement_ids.assert_not_awaited()


@pytest.mark.parametrize("command", ["ingest", "query", "delete"])
def test_cli_reports_model_error(
    command: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    kb = MagicMock(spec=KnowledgeBase)
    kb.__enter__.return_value = kb
    message = f"Cannot reach the embedding model server at {BASE} (model e)."
    getattr(kb, command).side_effect = ModelConnectionError(message)
    monkeypatch.setattr(cli, "_open", lambda _: kb)

    assert cli.main([command, "input"]) != 0
    assert capsys.readouterr().err == f"error: {message}\n"


def test_cli_suggest_reports_connection_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sdk, seen = fake_server(endpoint_failure(httpx2.ConnectError))
    monkeypatch.setenv("RAG_OAG_LLM_MODEL", "m")
    monkeypatch.setenv("RAG_OAG_LLM_BASE_URL", BASE)
    monkeypatch.setattr(cli, "_make_llm", lambda config: LLMClient(config, client=sdk))
    source = tmp_path / "doc.md"
    source.write_text("A service contract between two companies.", encoding="utf-8")

    assert cli.main(["suggest", str(source)]) != 0
    stderr = capsys.readouterr().err
    assert stderr.startswith(f"error: Cannot reach the chat model server at {BASE}")
    assert "Traceback" not in stderr
    assert len(seen) == 1
