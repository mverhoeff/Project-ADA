"""Async streaming client for the local Ollama server.

Wraps the ``/api/chat`` endpoint with ``stream=True``. Yields token strings
as they arrive. Connection errors and non-200 responses are translated to
:class:`core.exceptions.ServiceUnavailableError` so the orchestrator can
surface a spoken error to the user.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from core.exceptions import ServiceUnavailableError


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
    ) -> AsyncIterator[str]:
        """Yield token strings from the LLM stream.

        Args:
            messages: Chat history in Ollama format.
            tools: Optional tool declarations to include in the request.

        Yields:
            Incremental ``content`` strings from each streamed chunk.

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
                        content = chunk.get("message", {}).get("content")
                        if content:
                            yield content
                        if chunk.get("done"):
                            return
        except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError) as e:
            raise ServiceUnavailableError(
                f"Cannot reach Ollama at {endpoint}: {e}",
                _USER_FACING_ERROR,
            ) from e
