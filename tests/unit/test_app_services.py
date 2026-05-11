"""Unit tests for :mod:`app.services`."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app import services as svc_mod
from app.services import ServiceProcess
from core.exceptions import ServiceUnavailableError


def _fake_proc(returncode: int | None = None) -> MagicMock:
    """Build a fake :class:`asyncio.subprocess.Process` for tests."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = MagicMock()
    proc.stdout.readline = AsyncMock(return_value=b"")
    proc.stderr = MagicMock()
    proc.stderr.readline = AsyncMock(return_value=b"")
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    return proc


def _patch_spawn(
    monkeypatch: pytest.MonkeyPatch,
    proc: MagicMock,
) -> list[tuple[Any, ...]]:
    """Patch ``asyncio.create_subprocess_exec`` to return ``proc``."""
    calls: list[tuple[Any, ...]] = []

    async def fake_exec(*args: Any, **kwargs: Any) -> MagicMock:
        calls.append((args, kwargs))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return calls


def _patch_health(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[Any],
) -> None:
    """Patch ``httpx.AsyncClient`` so each ``get`` returns the next response.

    ``responses`` may contain ``httpx.Response``-like objects or exceptions
    to raise on the corresponding call.
    """
    queue = list(responses)

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def get(self, url: str) -> Any:
            if not queue:
                raise AssertionError("no more scripted health responses")
            item = queue.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

    monkeypatch.setattr(svc_mod.httpx, "AsyncClient", _FakeClient)


def _resp(status: int, body: dict[str, Any]) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json = MagicMock(return_value=body)
    return r


def _quick_process(name: str = "stt") -> ServiceProcess:
    return ServiceProcess(
        name=name,
        module=f"services.{name}.server",
        health_url=f"http://127.0.0.1:8771/health",
        startup_timeout_s=2.0,
        shutdown_timeout_s=1.0,
    )


@pytest.mark.asyncio
async def test_start_returns_when_health_reports_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _fake_proc()
    spawn_calls = _patch_spawn(monkeypatch, proc)
    _patch_health(
        monkeypatch,
        [
            httpx.ConnectError("not yet"),
            _resp(200, {"status": "ok", "model_loaded": False}),
            _resp(200, {"status": "ok", "model_loaded": True}),
        ],
    )
    # Tighten the poll interval so the test does not actually sleep 1s/iteration.
    monkeypatch.setattr(svc_mod, "_HEALTH_POLL_INTERVAL_S", 0.01)

    sp = _quick_process()
    await sp.start()

    assert len(spawn_calls) == 1
    args, _ = spawn_calls[0]
    assert args[1] == "-m"
    assert args[2] == "services.stt.server"

    await sp.stop()
    proc.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_start_times_out_when_never_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _fake_proc()
    _patch_spawn(monkeypatch, proc)
    # Always-failing health probe.
    responses: list[Any] = [httpx.ConnectError("nope") for _ in range(50)]
    _patch_health(monkeypatch, responses)
    monkeypatch.setattr(svc_mod, "_HEALTH_POLL_INTERVAL_S", 0.01)

    sp = ServiceProcess(
        name="stt",
        module="services.stt.server",
        health_url="http://127.0.0.1:8771/health",
        startup_timeout_s=0.05,
        shutdown_timeout_s=0.5,
    )

    with pytest.raises(ServiceUnavailableError):
        await sp.start()

    # The supervisor should have torn down the spawned child on failure.
    proc.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_start_aborts_when_child_exits_during_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _fake_proc(returncode=2)
    _patch_spawn(monkeypatch, proc)
    _patch_health(monkeypatch, [httpx.ConnectError("nope")])
    monkeypatch.setattr(svc_mod, "_HEALTH_POLL_INTERVAL_S", 0.01)

    sp = _quick_process()
    with pytest.raises(ServiceUnavailableError):
        await sp.start()


@pytest.mark.asyncio
async def test_stop_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _fake_proc()
    _patch_spawn(monkeypatch, proc)
    _patch_health(monkeypatch, [_resp(200, {"model_loaded": True})])
    monkeypatch.setattr(svc_mod, "_HEALTH_POLL_INTERVAL_S", 0.01)

    sp = _quick_process()
    await sp.start()

    await sp.stop()
    await sp.stop()  # second call must be a safe no-op

    proc.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_stop_without_start_is_safe() -> None:
    sp = _quick_process()
    await sp.stop()  # no proc, no error


@pytest.mark.asyncio
async def test_stop_escalates_to_kill_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _fake_proc()

    # Make wait() hang the first time so the SIGTERM path times out and we
    # escalate. The second wait() (after kill) returns immediately.
    wait_calls = {"n": 0}

    async def slow_wait() -> int:
        wait_calls["n"] += 1
        if wait_calls["n"] == 1:
            await asyncio.sleep(10)
        return 0

    proc.wait = slow_wait

    _patch_spawn(monkeypatch, proc)
    _patch_health(monkeypatch, [_resp(200, {"model_loaded": True})])
    monkeypatch.setattr(svc_mod, "_HEALTH_POLL_INTERVAL_S", 0.01)

    sp = ServiceProcess(
        name="stt",
        module="services.stt.server",
        health_url="http://127.0.0.1:8771/health",
        startup_timeout_s=2.0,
        shutdown_timeout_s=0.05,
    )
    await sp.start()
    await sp.stop()

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
