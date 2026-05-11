"""Unit tests for :mod:`app.main`."""

from __future__ import annotations

import argparse
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app import main as main_mod
from core.exceptions import ServiceUnavailableError


_BASE_CONFIG: dict[str, Any] = {
    "stt": {"port": 8771},
    "tts": {"port": 8772},
    "llm": {
        "ollama_url": "http://localhost:11434",
        "model": "qwen3:8b",
        "temperature": 0.7,
    },
    "audio": {"output_device": None},
    "agent": {
        "shell_timeout_seconds": 30,
        "allowed_paths": ["~/Documents"],
    },
    "app": {
        "service_startup_timeout_seconds": 5,
        "service_shutdown_timeout_seconds": 1,
    },
}


def _patch_ollama(monkeypatch: pytest.MonkeyPatch, response: Any) -> None:
    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...
        async def __aenter__(self) -> "_FakeClient":
            return self
        async def __aexit__(self, *args: Any) -> None:
            return None
        async def get(self, url: str) -> Any:
            if isinstance(response, BaseException):
                raise response
            return response

    monkeypatch.setattr(main_mod.httpx, "AsyncClient", _FakeClient)


def _resp(status: int, body: dict[str, Any]) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json = MagicMock(return_value=body)
    return r


# ---------------------------------------------------------------------------
# _check_ollama
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_ollama_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ollama(monkeypatch, _resp(200, {"models": [{"name": "qwen3:8b"}]}))
    await main_mod._check_ollama(_BASE_CONFIG)  # must not raise


@pytest.mark.asyncio
async def test_check_ollama_warns_when_model_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ollama(monkeypatch, _resp(200, {"models": [{"name": "other:8b"}]}))
    # Should not raise even though the configured model is absent.
    await main_mod._check_ollama(_BASE_CONFIG)


@pytest.mark.asyncio
async def test_check_ollama_raises_on_connect_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ollama(monkeypatch, httpx.ConnectError("refused"))
    with pytest.raises(ServiceUnavailableError):
        await main_mod._check_ollama(_BASE_CONFIG)


@pytest.mark.asyncio
async def test_check_ollama_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ollama(monkeypatch, _resp(500, {}))
    with pytest.raises(ServiceUnavailableError):
        await main_mod._check_ollama(_BASE_CONFIG)


# ---------------------------------------------------------------------------
# run() — full lifecycle with all heavy collaborators patched
# ---------------------------------------------------------------------------


class _RecordingService:
    """Stand-in for :class:`app.services.ServiceProcess`."""

    def __init__(self, name: str, **_: Any) -> None:
        self.name = name
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


class _RecordingVram:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


def _wire_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch every external collaborator of :func:`main.run`.

    Returns the bag of recorder objects so tests can inspect call order.
    """
    _patch_ollama(monkeypatch, _resp(200, {"models": [{"name": "qwen3:8b"}]}))

    created_services: list[_RecordingService] = []

    def fake_service(**kwargs: Any) -> _RecordingService:
        s = _RecordingService(**kwargs)
        created_services.append(s)
        return s

    vram = _RecordingVram()

    run_turn_calls = {"n": 0}

    async def fake_run_turn(*args: Any, **kwargs: Any) -> None:
        run_turn_calls["n"] += 1

    monkeypatch.setattr(main_mod, "ServiceProcess", fake_service)
    monkeypatch.setattr(main_mod, "VramMonitor", lambda *a, **kw: vram)
    monkeypatch.setattr(main_mod, "run_turn", fake_run_turn)
    monkeypatch.setattr(main_mod, "build_deps", lambda config: MagicMock())
    monkeypatch.setattr(main_mod, "build_session", lambda config: MagicMock())

    return {
        "services": created_services,
        "vram": vram,
        "run_turn_calls": run_turn_calls,
    }


@pytest.mark.asyncio
async def test_run_once_starts_and_stops_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bag = _wire_run(monkeypatch)
    args = argparse.Namespace(once=True, external_services=False, log_level="INFO")

    rc = await main_mod.run(args, _BASE_CONFIG)
    assert rc == 0

    services = bag["services"]
    assert len(services) == 2
    assert {s.name for s in services} == {"stt", "tts"}
    for s in services:
        assert s.started and s.stopped

    vram = bag["vram"]
    assert vram.started and vram.stopped

    assert bag["run_turn_calls"]["n"] == 1


@pytest.mark.asyncio
async def test_run_once_with_external_services_skips_subprocesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bag = _wire_run(monkeypatch)
    args = argparse.Namespace(once=True, external_services=True, log_level="INFO")

    rc = await main_mod.run(args, _BASE_CONFIG)
    assert rc == 0
    assert bag["services"] == []
    assert bag["vram"].started and bag["vram"].stopped
    assert bag["run_turn_calls"]["n"] == 1


@pytest.mark.asyncio
async def test_run_propagates_ollama_failure_without_starting_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ollama(monkeypatch, httpx.ConnectError("nope"))
    created: list[Any] = []
    monkeypatch.setattr(
        main_mod,
        "ServiceProcess",
        lambda **kw: created.append(kw) or MagicMock(),
    )

    args = argparse.Namespace(once=True, external_services=False, log_level="INFO")
    with pytest.raises(ServiceUnavailableError):
        await main_mod.run(args, _BASE_CONFIG)

    assert created == []  # services were never even constructed
