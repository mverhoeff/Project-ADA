"""Async streaming client for the local Ollama server.

Wraps the ``/api/chat`` endpoint with ``stream=True``. Yields typed
:class:`~services.llm.events.StreamEvent` values — :class:`TextChunk` for
spoken content and :class:`ToolCallChunk` for structured tool invocations
parsed from Ollama's ``message.tool_calls`` field. Connection errors and
non-200 responses are translated to
:class:`core.exceptions.ServiceUnavailableError` so the orchestrator can
surface a spoken error to the user.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from core.exceptions import ServiceUnavailableError
from services.llm.events import StreamEvent, TextChunk, ToolCallChunk


_USER_FACING_ERROR = "I can't reach my language model right now."


class LLMClient:
    """Thin async wrapper around Ollama's streaming chat endpoint.

    Args:
        config: The full loaded config dict. Only the ``llm`` subtree is
            consulted (``ollama_url``, ``model``, ``temperature``, and
            optionally ``request_timeout_seconds``).
        transport: Optional ``httpx`` transport, used by tests to inject
            mock responses without hitting the network.
    """

    def __init__(
        self,
        config: dict[str, Any],
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        llm_cfg = config["llm"]
        self._url = llm_cfg["ollama_url"].rstrip("/")
        self._model = llm_cfg["model"]
        self._temperature = float(llm_cfg.get("temperature", 0.7))
        self._timeout = float(llm_cfg.get("request_timeout_seconds", 60))
        self._transport = transport

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield typed events from the LLM stream.

        Args:
            messages: Chat history in Ollama format.
            tools: Optional Ollama-format tool declarations. When non-empty,
                Ollama may emit ``tool_calls`` in its response, which are
                yielded as :class:`ToolCallChunk` events.

        Yields:
            :class:`TextChunk` for spoken content tokens, in order, and
            :class:`ToolCallChunk` for any structured tool invocation
            Ollama produces.

        Raises:
            ServiceUnavailableError: If Ollama is unreachable, times out,
                or returns a non-200 status.
        """
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self._temperature},
        }
        if tools:
            body["tools"] = tools

        endpoint = f"{self._url}/api/chat"

        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=self._timeout,
            ) as client:
                async with client.stream("POST", endpoint, json=body) as resp:
                    if resp.status_code != 200:
                        raise ServiceUnavailableError(
                            f"Ollama returned HTTP {resp.status_code} at {endpoint}",
                            _USER_FACING_ERROR,
                        )
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        message = chunk.get("message", {}) or {}
                        content = message.get("content")
                        if content:
                            yield TextChunk(text=content)

                        for call in message.get("tool_calls") or []:
                            fn = call.get("function", {}) or {}
                            name = fn.get("name")
                            if not name:
                                continue
                            args = fn.get("arguments") or {}
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    args = {}
                            if not isinstance(args, dict):
                                args = {}
                            yield ToolCallChunk(name=name, arguments=args)

                        if chunk.get("done"):
                            return
        except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError) as e:
            raise ServiceUnavailableError(
                f"Cannot reach Ollama at {endpoint}: {e}",
                _USER_FACING_ERROR,
            ) from e
