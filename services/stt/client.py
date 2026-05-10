"""Async client for the local STT FastAPI service.

Wraps ``POST /transcribe`` and ``GET /health`` on ``localhost:<stt.port>``.
Connection errors and non-200 responses are translated to
:class:`core.exceptions.ServiceUnavailableError` so the orchestrator can
surface a spoken error to the user.
"""

from __future__ import annotations

from typing import Any

import httpx

from core.exceptions import ServiceUnavailableError


_USER_FACING_ERROR = "I can't hear you right now."


class STTClient:
    """Thin async wrapper around the STT service.

    Args:
        config: The full loaded config dict. Only the ``stt`` subtree is
            consulted (``port`` and optionally ``request_timeout_seconds``).
        transport: Optional ``httpx`` transport, used by tests to inject
            mock responses without hitting the network.
    """

    def __init__(
        self,
        config: dict[str, Any],
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        stt_cfg = config["stt"]
        self._url = f"http://127.0.0.1:{stt_cfg['port']}"
        self._timeout = float(stt_cfg.get("request_timeout_seconds", 30))
        self._transport = transport

    async def transcribe(self, wav_bytes: bytes) -> dict[str, Any]:
        """Send WAV bytes to ``/transcribe`` and return the parsed JSON.

        Args:
            wav_bytes: Complete WAV file bytes (any sample rate; the server
                resamples to 16 kHz internally).

        Returns:
            The parsed response body, e.g. ``{"text": ..., "language": ...,
            "duration_ms": ...}``.

        Raises:
            ValueError: If ``wav_bytes`` is empty.
            ServiceUnavailableError: If the STT service is unreachable,
                times out, or returns a non-200 status.
        """
        if not wav_bytes:
            raise ValueError("wav_bytes must be non-empty")

        endpoint = f"{self._url}/transcribe"
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=self._timeout,
            ) as client:
                resp = await client.post(
                    endpoint,
                    content=wav_bytes,
                    headers={"content-type": "audio/wav"},
                )
                if resp.status_code != 200:
                    raise ServiceUnavailableError(
                        f"STT returned HTTP {resp.status_code} at {endpoint}",
                        _USER_FACING_ERROR,
                    )
                return resp.json()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError) as e:
            raise ServiceUnavailableError(
                f"Cannot reach STT at {endpoint}: {e}",
                _USER_FACING_ERROR,
            ) from e

    async def health(self) -> dict[str, Any]:
        """Query ``/health`` and return the parsed JSON.

        Returns:
            The parsed response body, e.g. ``{"status": "ok",
            "model_loaded": True}``.

        Raises:
            ServiceUnavailableError: If the STT service is unreachable,
                times out, or returns a non-200 status.
        """
        endpoint = f"{self._url}/health"
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=self._timeout,
            ) as client:
                resp = await client.get(endpoint)
                if resp.status_code != 200:
                    raise ServiceUnavailableError(
                        f"STT returned HTTP {resp.status_code} at {endpoint}",
                        _USER_FACING_ERROR,
                    )
                return resp.json()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError) as e:
            raise ServiceUnavailableError(
                f"Cannot reach STT at {endpoint}: {e}",
                _USER_FACING_ERROR,
            ) from e
