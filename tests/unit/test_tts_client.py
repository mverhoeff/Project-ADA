"""Unit tests for :mod:`services.tts.client`."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from core.exceptions import ServiceUnavailableError
from services.tts.client import TTSClient


_BASE_CONFIG: dict[str, Any] = {"tts": {"port": 8772}}


class _FakeStream(httpx.AsyncByteStream):
    """An ``httpx.AsyncByteStream`` returning a fixed list of chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_synthesize_yields_concatenated_body() -> None:
    chunks = [b"AAAA", b"BBBB", b"CCCC"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_FakeStream(chunks))

    client = TTSClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    out = b""
    async for chunk in client.synthesize("hello"):
        out += chunk
    assert out == b"AAAABBBBCCCC"


@pytest.mark.asyncio
async def test_synthesize_sends_text_voice_speed_in_json_body() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"")

    client = TTSClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    async for _ in client.synthesize("hi there", voice="af_sky", speed=1.2):
        pass
    assert captured["body"] == {"text": "hi there", "voice": "af_sky", "speed": 1.2}


@pytest.mark.asyncio
async def test_synthesize_omits_voice_and_speed_when_none() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"")

    client = TTSClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    async for _ in client.synthesize("hello"):
        pass
    assert captured["body"] == {"text": "hello"}


@pytest.mark.asyncio
async def test_synthesize_uses_post_method_and_correct_url() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        return httpx.Response(200, content=b"")

    client = TTSClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    async for _ in client.synthesize("hello"):
        pass
    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:8772/synthesize"


@pytest.mark.asyncio
async def test_synthesize_raises_value_error_on_empty_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP must not be called for empty input")

    client = TTSClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError):
        async for _ in client.synthesize(""):
            pass


@pytest.mark.asyncio
async def test_synthesize_raises_value_error_on_whitespace_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP must not be called for whitespace input")

    client = TTSClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError):
        async for _ in client.synthesize("   \n\t  "):
            pass


@pytest.mark.asyncio
async def test_synthesize_raises_service_unavailable_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = TTSClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    with pytest.raises(ServiceUnavailableError) as exc_info:
        async for _ in client.synthesize("hello"):
            pass
    assert "speak" in exc_info.value.user_message.lower()


@pytest.mark.asyncio
async def test_synthesize_raises_service_unavailable_on_non_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"unavailable")

    client = TTSClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    with pytest.raises(ServiceUnavailableError) as exc_info:
        async for _ in client.synthesize("hello"):
            pass
    assert "503" in str(exc_info.value)


@pytest.mark.asyncio
async def test_voices_returns_voice_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "http://127.0.0.1:8772/voices"
        return httpx.Response(200, content=json.dumps({"voices": ["a", "b"]}).encode())

    client = TTSClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    assert await client.voices() == ["a", "b"]


@pytest.mark.asyncio
async def test_voices_raises_service_unavailable_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    client = TTSClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    with pytest.raises(ServiceUnavailableError):
        await client.voices()


@pytest.mark.asyncio
async def test_health_returns_parsed_json() -> None:
    body = {"status": "ok", "model_loaded": True}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "http://127.0.0.1:8772/health"
        return httpx.Response(200, content=json.dumps(body).encode())

    client = TTSClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    assert await client.health() == body
