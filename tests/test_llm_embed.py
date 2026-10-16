"""OpenAI-compatible clients against an in-process fake server.

openai>=3 sends requests through httpx2, which respx does not patch, so the
fake server is an httpx2 MockTransport handed to the SDK client.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from openai import AsyncOpenAI

from rag_with_oag import EmbedConfig, LLMConfig
from rag_with_oag.embed.openai_compat import OpenAIEmbedder
from rag_with_oag.llm.client import LLMClient

BASE = "http://llm.test/v1"
Handler = Callable[[httpx2.Request], httpx2.Response]


def fake_server(
    handler: Handler, *, max_retries: int = 0
) -> tuple[AsyncOpenAI, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request)

    client = AsyncOpenAI(
        base_url=BASE,
        api_key="test",
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
