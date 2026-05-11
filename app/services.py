"""Subprocess supervisor for the local STT and TTS FastAPI services.

Each :class:`ServiceProcess` owns one child Python process running
``python -m services.<svc>.server``. :meth:`start` launches the child,
forwards its stdout/stderr to the launcher's structured log, and polls
the service's ``/health`` endpoint until ``model_loaded`` flips to
``True`` (or the configured timeout expires). :meth:`stop` is
idempotent: it sends SIGTERM, waits up to ``shutdown_timeout_s``, then
escalates to SIGKILL.

The helper is deliberately stateless about which service it manages —
the call site supplies the module path, health URL, and human-readable
name.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import httpx

from core.exceptions import ServiceUnavailableError
from core.logger import get_logger

_log = get_logger(__name__)

_HEALTH_POLL_INTERVAL_S = 1.0
_HEALTH_REQUEST_TIMEOUT_S = 2.0


class ServiceProcess:
    """Manage the lifecycle of one local FastAPI inference service."""

    def __init__(
        self,
        name: str,
        module: str,
        health_url: str,
        startup_timeout_s: float,
        shutdown_timeout_s: float,
    ) -> None:
        self._name = name
        self._module = module
        self._health_url = health_url
        self._startup_timeout_s = startup_timeout_s
        self._shutdown_timeout_s = shutdown_timeout_s
        self._proc: asyncio.subprocess.Process | None = None
        self._log_tasks: list[asyncio.Task[None]] = []

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        """Spawn the child and wait until ``/health`` reports it is ready.

        Raises:
            ServiceUnavailableError: If the child cannot be spawned or the
                service does not report ready within ``startup_timeout_s``.
        """
        if self._proc is not None:
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        _log.info("service_starting", name=self._name, module=self._module)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                self._module,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except OSError as e:
            raise ServiceUnavailableError(
                f"Failed to spawn {self._name} service: {e}",
                f"The {self._name} service failed to start.",
            ) from e

        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._log_tasks = [
            asyncio.create_task(
                _forward_stream(self._proc.stdout, self._name, "stdout"),
                name=f"{self._name}-stdout",
            ),
            asyncio.create_task(
                _forward_stream(self._proc.stderr, self._name, "stderr"),
                name=f"{self._name}-stderr",
            ),
        ]

        try:
            await self._wait_ready()
        except ServiceUnavailableError:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Terminate the child and cancel log forwarders. Idempotent."""
        proc = self._proc
        if proc is None:
            return
        self._proc = None

        if proc.returncode is None:
            _log.info("service_stopping", name=self._name)
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=self._shutdown_timeout_s)
            except asyncio.TimeoutError:
                _log.warning("service_shutdown_timeout", name=self._name)
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()

        for task in self._log_tasks:
            if not task.done():
                task.cancel()
        if self._log_tasks:
            await asyncio.gather(*self._log_tasks, return_exceptions=True)
        self._log_tasks = []

        _log.info("service_stopped", name=self._name, returncode=proc.returncode)

    async def _wait_ready(self) -> None:
        """Poll the health endpoint until ``model_loaded`` is ``True`` or timeout."""
        deadline = asyncio.get_event_loop().time() + self._startup_timeout_s
        last_error: str = ""
        while True:
            if self._proc is not None and self._proc.returncode is not None:
                raise ServiceUnavailableError(
                    f"{self._name} service exited during startup "
                    f"(returncode={self._proc.returncode})",
                    f"The {self._name} service failed to start.",
                )

            ready, last_error = await self._poll_health_once()
            if ready:
                _log.info("service_ready", name=self._name)
                return

            if asyncio.get_event_loop().time() >= deadline:
                raise ServiceUnavailableError(
                    f"{self._name} service did not become ready within "
                    f"{self._startup_timeout_s:.0f}s (last error: {last_error})",
                    f"The {self._name} service failed to start.",
                )
            await asyncio.sleep(_HEALTH_POLL_INTERVAL_S)

    async def _poll_health_once(self) -> tuple[bool, str]:
        """Return ``(ready, error_string)`` for a single health probe."""
        try:
            async with httpx.AsyncClient(timeout=_HEALTH_REQUEST_TIMEOUT_S) as client:
                resp = await client.get(self._health_url)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError) as e:
            return False, f"{type(e).__name__}: {e}"

        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"

        try:
            body: dict[str, Any] = resp.json()
        except ValueError as e:
            return False, f"invalid JSON: {e}"

        if bool(body.get("model_loaded")):
            return True, ""
        return False, "model_loaded=false"


async def _forward_stream(
    stream: asyncio.StreamReader,
    service_name: str,
    channel: str,
) -> None:
    """Forward a child process stream line-by-line to the structured log."""
    try:
        while True:
            line = await stream.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                _log.info("service_log", name=service_name, channel=channel, line=text)
    except asyncio.CancelledError:
        return
