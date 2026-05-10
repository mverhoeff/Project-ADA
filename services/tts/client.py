"""Async client for the local TTS FastAPI service.

Wraps ``POST /synthesize`` (chunked WAV stream), ``GET /voices``, and
``GET /health`` on ``localhost:<tts.port>``. The streaming method yields
raw response bytes — WAV header parsing is the playback layer's
responsibility (:mod:`orchestrator.audio_output`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from core.exceptions import ServiceUnavailableError


_USER_FACING_ERROR = "I can't speak right now."


class TTSClient:
    """Thin async wrapper around the TTS service.

    Args:
        config: The full loaded config dict. Only the ``tts`` subtree is
            consulted (``port`` and optionally
            ``request_timeout_seconds``).
        transport: Optional ``httpx`` transport, used by tests to inject
            mock responses without hitting the network.
    """

    def __init__(
        self,
        config: dict[str, Any],
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        tts_cfg = config["tts"]
        self._url = f"http://127.0.0.1:{tts_cfg['port']}"
        self._timeout = float(tts_cfg.get("request_timeout_seconds", 60))
        self._transport = transport

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream the synthesized audio bytes for ``text``.

        Yields the raw response body chunks (the first chunk of the
        underlying response begins with a 44-byte WAV header; chunk
        boundaries from ``httpx`` may not align with that header).

        Args:
            text: Sentence to synthesize. Must be non-empty after stripping.
            voice: Optional voice ID; the server falls back to its
                configured default when omitted.
            speed: Optional speech rate multiplier; same fallback rule.

        Yields:
            Successive ``bytes`` chunks from the response body.

        Raises:
            ValueError: If ``text`` is empty or whitespace only.
            ServiceUnavailableError: If the TTS service is unreachable,
                times out, or returns a non-200 status.
        """
        if not text.strip():
            raise ValueError("text must be non-empty")

        body: dict[str, Any] = {"text": text}
        if voice is not None:
            body["voice"] = voice
        if speed is not None:
            body["speed"] = speed

        endpoint = f"{self._url}/synthesize"
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=self._timeout,
            ) as client:
                async with client.stream("POST", endpoint, json=body) as resp:
                    if resp.status_code != 200:
                        raise ServiceUnavailableError(
                            f"TTS returned HTTP {resp.status_code} at {endpoint}",
                            _USER_FACING_ERROR,
                        )
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk
        except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError) as e:
            raise ServiceUnavailableError(
                f"Cannot reach TTS at {endpoint}: {e}",
                _USER_FACING_ERROR,
            ) from e

    async def voices(self) -> list[str]:
        """Query ``/voices`` and return the available voice IDs.

        Returns:
            The list of voice IDs from ``{"voices": [...]}``.

        Raises:
            ServiceUnavailableError: If the TTS service is unreachable,
                times out, or returns a non-200 status.
        """
        body = await self._get_json("/voices")
        return list(body.get("voices", []))

    async def health(self) -> dict[str, Any]:
        """Query ``/health`` and return the parsed JSON.

        Raises:
            ServiceUnavailableError: If the TTS service is unreachable,
                times out, or returns a non-200 status.
        """
        return await self._get_json("/health")

    async def _get_json(self, path: str) -> dict[str, Any]:
        endpoint = f"{self._url}{path}"
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=self._timeout,
            ) as client:
                resp = await client.get(endpoint)
                if resp.status_code != 200:
                    raise ServiceUnavailableError(
                        f"TTS returned HTTP {resp.status_code} at {endpoint}",
                        _USER_FACING_ERROR,
                    )
                return resp.json()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError) as e:
            raise ServiceUnavailableError(
                f"Cannot reach TTS at {endpoint}: {e}",
                _USER_FACING_ERROR,
            ) from e
