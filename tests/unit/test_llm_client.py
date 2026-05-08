"""Unit tests for :mod:`services.llm.client`."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from core.exceptions import ServiceUnavailableError
from services.llm.client import LLMClient


_BASE_CONFIG: dict[str, Any] = {
    "llm": {
        "ollama_url": "http://localhost:11434",
        "model": "qwen3:8b",
        "temperature": 0.7,
    },
}


def _streaming_body(chunks: list[dict[str, Any]]) -> bytes:
    return ("\n".join(json.dumps(c) for c in chunks) + "\n").encode()


@pytest.mark.asyncio
async def test_stream_yields_tokens() -> None:
    body = _streaming_body(
        [
            {"message": {"content": "Hello"}, "done": False},
            {"message": {"content": " "}, "done": False},
            {"message": {"content": "world"}, "done": True},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = LLMClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    out = [token async for token in client.stream_chat([{"role": "user", "content": "hi"}])]
    assert out == ["Hello", " ", "world"]


@pytest.mark.asyncio
async def test_stream_stops_on_done() -> None:
    body = _streaming_body(
        [
            {"message": {"content": "first"}, "done": False},
            {"message": {"content": "last"}, "done": True},
            {"message": {"content": "after-done"}, "done": False},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = LLMClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    out = [t async for t in client.stream_chat([])]
    assert out == ["first", "last"]


@pytest.mark.asyncio
async def test_skips_blank_lines_and_unparseable_chunks() -> None:
    body = (
        b"\n"
        + json.dumps({"message": {"content": "ok"}, "done": False}).encode()
        + b"\n"
        + b"not-json garbage\n"
        + json.dumps({"message": {"content": "end"}, "done": True}).encode()
        + b"\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = LLMClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    out = [t async for t in client.stream_chat([])]
    assert out == ["ok", "end"]


@pytest.mark.asyncio
async def test_raises_service_unavailable_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = LLMClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    with pytest.raises(ServiceUnavailableError) as exc_info:
        async for _ in client.stream_chat([]):
            pass
    assert "language model" in exc_info.value.user_message.lower()


@pytest.mark.asyncio
async def test_raises_service_unavailable_on_non_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"server error")

    client = LLMClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    with pytest.raises(ServiceUnavailableError) as exc_info:
        async for _ in client.stream_chat([]):
            pass
    assert "500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_passes_tools_in_request() -> None:
    captured: dict[str, Any] = {}
    body = _streaming_body([{"message": {"content": ""}, "done": True}])

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=body)

    client = LLMClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    tools = [{"type": "function", "function": {"name": "search"}}]
    async for _ in client.stream_chat(
        [{"role": "user", "content": "hi"}], tools=tools
    ):
        pass

    assert captured["body"]["tools"] == tools
    assert captured["body"]["model"] == "qwen3:8b"
    assert captured["body"]["stream"] is True
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_omits_tools_when_none() -> None:
    captured: dict[str, Any] = {}
    body = _streaming_body([{"message": {"content": ""}, "done": True}])

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=body)

    client = LLMClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    async for _ in client.stream_chat([]):
        pass

    assert "tools" not in captured["body"]


@pytest.mark.asyncio
async def test_url_trailing_slash_is_normalised() -> None:
    captured: dict[str, str] = {}
    body = _streaming_body([{"message": {"content": ""}, "done": True}])

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, content=body)

    config = {
        "llm": {
            "ollama_url": "http://localhost:11434/",
            "model": "qwen3:8b",
            "temperature": 0.7,
        },
    }
    client = LLMClient(config, transport=httpx.MockTransport(handler))
    async for _ in client.stream_chat([]):
        pass

    assert captured["url"] == "http://localhost:11434/api/chat"
