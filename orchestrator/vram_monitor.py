"""Background VRAM safety valve.

Polls ``nvidia-smi`` every 10 seconds. If usage crosses the configured
flush threshold, posts ``keep_alive: 0`` to Ollama's ``/api/generate``
endpoint, which unloads the model from GPU memory while keeping its
weights on disk; the next chat request reloads them automatically.

Failures (no NVIDIA driver, Ollama unreachable, parse errors) are
logged and swallowed: this is a last-resort safety net, not a critical
path, and must never crash the orchestrator.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any

import httpx

from core.logger import get_logger

_log = get_logger(__name__)

_POLL_INTERVAL_SECONDS = 10
_NVIDIA_SMI_TIMEOUT = 5
_FLUSH_HTTP_TIMEOUT = 10.0


class VramMonitor:
    """Watch GPU VRAM usage and flush Ollama's KV cache on threshold breach.

    Args:
        config: Full loaded config dict. Reads ``vram.warning_threshold_percent``,
            ``vram.flush_threshold_percent``, ``llm.ollama_url``, and ``llm.model``.
        transport: Optional ``httpx`` transport, used by tests to mock the
            Ollama flush call without hitting the network.
    """

    def __init__(
        self,
        config: dict[str, Any],
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        vram_cfg = config.get("vram", {})
        llm_cfg = config.get("llm", {})
        self._warn_pct = float(vram_cfg.get("warning_threshold_percent", 90))
        self._flush_pct = float(vram_cfg.get("flush_threshold_percent", 95))
        self._ollama_url = str(llm_cfg.get("ollama_url", "http://localhost:11434")).rstrip("/")
        self._model = str(llm_cfg.get("model", ""))
        self._transport = transport
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Spawn the background polling task. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._poll_loop(), name="vram_monitor")
        _log.debug("vram_monitor_started")

    async def stop(self) -> None:
        """Cancel the polling task and wait for it to exit."""
        if self._task is None:
            return
        if not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        _log.debug("vram_monitor_stopped")

    async def _poll_loop(self) -> None:
        """Poll VRAM usage and trigger a flush above the configured threshold."""
        while True:
            pct = await self._query_vram_pct()
            if pct is not None:
                if pct >= self._flush_pct:
                    _log.warning(
                        "vram_flush_threshold_exceeded",
                        vram_pct=pct,
                        threshold=self._flush_pct,
                    )
                    await self._flush_kv_cache()
                elif pct >= self._warn_pct:
                    _log.warning(
                        "vram_warning_threshold_exceeded",
                        vram_pct=pct,
                        threshold=self._warn_pct,
                    )
                else:
                    _log.debug("vram_ok", vram_pct=pct)
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    async def _query_vram_pct(self) -> float | None:
        """Return current GPU VRAM usage as a percentage, or ``None`` on failure."""

        def _run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=_NVIDIA_SMI_TIMEOUT,
            )

        try:
            result = await asyncio.to_thread(_run)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
            _log.debug("vram_query_failed", error=str(e))
            return None

        if result.returncode != 0:
            _log.debug(
                "vram_query_nonzero_exit",
                returncode=result.returncode,
                stderr=result.stderr.strip(),
            )
            return None

        lines = result.stdout.strip().splitlines()
        if not lines:
            return None
        try:
            used_str, total_str = lines[0].split(",")
            used = float(used_str.strip())
            total = float(total_str.strip())
        except (ValueError, IndexError) as e:
            _log.debug("vram_parse_failed", output=repr(result.stdout), error=str(e))
            return None

        if total <= 0:
            return None
        return (used / total) * 100.0

    async def _flush_kv_cache(self) -> None:
        """Tell Ollama to unload the model from GPU memory."""
        url = f"{self._ollama_url}/api/generate"
        body = {"model": self._model, "keep_alive": 0}
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=_FLUSH_HTTP_TIMEOUT,
            ) as client:
                resp = await client.post(url, json=body)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError) as e:
            _log.warning("vram_flush_request_failed", error=str(e), url=url)
            return

        if resp.status_code != 200:
            _log.warning("vram_flush_http_error", status=resp.status_code, url=url)
        else:
            _log.info("vram_flush_ok", model=self._model)
