"""Unit tests for :mod:`services.stt.client`."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from core.exceptions import ServiceUnavailableError
from services.stt.client import STTClient


_BASE_CONFIG: dict[str, Any] = {"stt": {"port": 8771}}


@pytest.mark.asyncio
async def test_transcribe_returns_parsed_json() -> None:
    payload = {"text": "hello world", "language": "en", "duration_ms": 1234}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(payload).encode())

    client = STTClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    result = await client.transcribe(b"RIFF....WAVEfake")
    assert result == payload


@pytest.mark.asyncio
async def test_transcribe_sends_wav_bytes_as_request_body() -> None:
    captured: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, content=b"{}")

    client = STTClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    await client.transcribe(b"\x00\x01\x02wav-bytes")
    assert captured["body"] == b"\x00\x01\x02wav-bytes"


@pytest.mark.asyncio
async def test_transcribe_uses_post_method_and_correct_url() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        return httpx.Response(200, content=b"{}")

    client = STTClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    await client.transcribe(b"x")
    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:8771/transcribe"


@pytest.mark.asyncio
async def test_transcribe_raises_value_error_on_empty_bytes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP must not be called for empty input")

    client = STTClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError):
        await client.transcribe(b"")


@pytest.mark.asyncio
async def test_transcribe_raises_service_unavailable_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = STTClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    with pytest.raises(ServiceUnavailableError) as exc_info:
        await client.transcribe(b"x")
    assert "hear" in exc_info.value.user_message.lower()


@pytest.mark.asyncio
async def test_transcribe_raises_service_unavailable_on_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("read timeout")

    client = STTClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    with pytest.raises(ServiceUnavailableError) as exc_info:
        await client.transcribe(b"x")
    assert "hear" in exc_info.value.user_message.lower()


@pytest.mark.asyncio
async def test_transcribe_raises_service_unavailable_on_non_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"internal error")

    client = STTClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    with pytest.raises(ServiceUnavailableError) as exc_info:
        await client.transcribe(b"x")
    assert "500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_health_returns_parsed_json() -> None:
    body = {"status": "ok", "model_loaded": True}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "http://127.0.0.1:8771/health"
        return httpx.Response(200, content=json.dumps(body).encode())

    client = STTClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    assert await client.health() == body


@pytest.mark.asyncio
async def test_health_raises_service_unavailable_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    client = STTClient(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    with pytest.raises(ServiceUnavailableError):
        await client.health()
