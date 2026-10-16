"""Chat completions against any OpenAI-compatible endpoint.

vLLM, Ollama, LM Studio, OpenRouter, Groq and OpenAI all work through the
official SDK by changing ``base_url``. Retries and timeouts are delegated to
the SDK, which already backs off on 429 and 5xx.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from openai import AsyncOpenAI

from rag_with_oag.config import LLMConfig, resolve_api_key

Message = dict[str, Any]


@runtime_checkable
class ChatModel(Protocol):
    """What the library needs from a chat model. Tests plug in fakes through this."""

    model: str
    usage: Counter[str]

    async def complete(self, messages: Sequence[Message], **kwargs: Any) -> str: ...


class LLMClient:
    def __init__(self, config: LLMConfig, *, client: AsyncOpenAI | None = None) -> None:
        self.config = config
        self.model = config.model
        self.usage: Counter[str] = Counter()
        self._client = client or AsyncOpenAI(
            base_url=config.base_url,
            api_key=resolve_api_key(config.api_key),
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    @property
    def endpoint(self) -> str:
        return self.config.base_url

    async def complete(self, messages: Sequence[Message], **kwargs: Any) -> str:
        response = await self._create(messages, **kwargs)
        content = response.choices[0].message.content if response.choices else None
        return content or ""

    async def complete_tool(
        self, messages: Sequence[Message], *, name: str, schema: dict[str, Any]
    ) -> str | None:
        """Force one function call with ``schema`` as its parameters; return its arguments."""
        response = await self._create(
            messages,
            tools=[
                {
                    "type": "function",
                    "function": {"name": name, "parameters": schema, "strict": True},
                }
            ],
            tool_choice={"type": "function", "function": {"name": name}},
        )
        calls = response.choices[0].message.tool_calls if response.choices else None
        function = getattr(calls[0], "function", None) if calls else None  # not a custom tool
        return function.arguments if function is not None else None

    async def _create(self, messages: Sequence[Message], **kwargs: Any) -> Any:
        params: dict[str, Any] = {"temperature": self.config.temperature}
        if self.config.max_tokens is not None:
            params["max_tokens"] = self.config.max_tokens
        params.update(kwargs)
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),  # type: ignore[arg-type]  # plain dicts, SDK wants TypedDicts
            **params,
        )
        if response.usage is not None:
            self.usage["prompt_tokens"] += response.usage.prompt_tokens
            self.usage["completion_tokens"] += response.usage.completion_tokens
        return response

    async def aclose(self) -> None:
        await self._client.close()
