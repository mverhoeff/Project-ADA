"""Unit tests for :mod:`orchestrator.vram_monitor`."""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from orchestrator import vram_monitor as vm
from orchestrator.vram_monitor import VramMonitor


_BASE_CONFIG: dict[str, Any] = {
    "vram": {
        "warning_threshold_percent": 90,
        "flush_threshold_percent": 95,
    },
    "llm": {
        "ollama_url": "http://localhost:11434",
        "model": "qwen3:8b",
    },
}


def _smi_ok(used_mb: int, total_mb: int) -> MagicMock:
    """Return a fake :class:`subprocess.CompletedProcess` with nvidia-smi CSV output."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = f"{used_mb}, {total_mb}\n"
    m.stderr = ""
    return m


def _smi_fail(returncode: int = 1, stderr: str = "error") -> MagicMock:
    m = MagicMock()
    m.returncode = returncode
    m.stdout = ""
    m.stderr = stderr
    return m


@pytest.mark.asyncio
async def test_query_vram_pct_normal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vm.subprocess, "run", lambda *a, **kw: _smi_ok(5000, 12288))
    monitor = VramMonitor(_BASE_CONFIG)
    pct = await monitor._query_vram_pct()
    assert pct is not None
    assert pct == pytest.approx(5000 / 12288 * 100, abs=0.01)


@pytest.mark.asyncio
async def test_query_vram_pct_returns_none_when_nvidia_smi_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("nvidia-smi not on PATH")

    monkeypatch.setattr(vm.subprocess, "run", _raise)
    monitor = VramMonitor(_BASE_CONFIG)
    assert await monitor._query_vram_pct() is None


@pytest.mark.asyncio
async def test_query_vram_pct_returns_none_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vm.subprocess, "run", lambda *a, **kw: _smi_fail())
    monitor = VramMonitor(_BASE_CONFIG)
    assert await monitor._query_vram_pct() is None


@pytest.mark.asyncio
async def test_query_vram_pct_returns_none_on_unparseable_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = MagicMock(returncode=0, stdout="not a csv line\n", stderr="")
    monkeypatch.setattr(vm.subprocess, "run", lambda *a, **kw: bad)
    monitor = VramMonitor(_BASE_CONFIG)
    assert await monitor._query_vram_pct() is None


@pytest.mark.asyncio
async def test_query_vram_pct_takes_first_gpu_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    multi = MagicMock(returncode=0, stdout="5000, 12288\n3000, 8192\n", stderr="")
    monkeypatch.setattr(vm.subprocess, "run", lambda *a, **kw: multi)
    monitor = VramMonitor(_BASE_CONFIG)
    pct = await monitor._query_vram_pct()
    assert pct == pytest.approx(5000 / 12288 * 100, abs=0.01)


@pytest.mark.asyncio
async def test_poll_loop_flushes_at_flush_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = VramMonitor(_BASE_CONFIG)

    async def _fake_query() -> float:
        return 96.0

    flush_mock = AsyncMock()

    sleep_calls = 0

    async def _fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        raise asyncio.CancelledError

    monkeypatch.setattr(monitor, "_query_vram_pct", _fake_query)
    monkeypatch.setattr(monitor, "_flush_kv_cache", flush_mock)
    monkeypatch.setattr(vm.asyncio, "sleep", _fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await monitor._poll_loop()

    assert flush_mock.await_count == 1
    assert sleep_calls == 1


@pytest.mark.asyncio
async def test_poll_loop_does_not_flush_below_warn_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = VramMonitor(_BASE_CONFIG)

    async def _fake_query() -> float:
        return 50.0

    flush_mock = AsyncMock()

    async def _fake_sleep(_seconds: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(monitor, "_query_vram_pct", _fake_query)
    monkeypatch.setattr(monitor, "_flush_kv_cache", flush_mock)
    monkeypatch.setattr(vm.asyncio, "sleep", _fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await monitor._poll_loop()

    flush_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_poll_loop_does_not_flush_at_warn_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = VramMonitor(_BASE_CONFIG)

    async def _fake_query() -> float:
        return 92.0  # between warn (90) and flush (95)

    flush_mock = AsyncMock()

    async def _fake_sleep(_seconds: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(monitor, "_query_vram_pct", _fake_query)
    monkeypatch.setattr(monitor, "_flush_kv_cache", flush_mock)
    monkeypatch.setattr(vm.asyncio, "sleep", _fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await monitor._poll_loop()

    flush_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_flush_kv_cache_posts_keep_alive_zero() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.read()
        return httpx.Response(200, json={"done": True})

    monitor = VramMonitor(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    await monitor._flush_kv_cache()

    assert captured["url"] == "http://localhost:11434/api/generate"
    import json

    body = json.loads(captured["body"])
    assert body == {"model": "qwen3:8b", "keep_alive": 0}


@pytest.mark.asyncio
async def test_flush_kv_cache_swallows_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monitor = VramMonitor(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    await monitor._flush_kv_cache()  # must not raise


@pytest.mark.asyncio
async def test_flush_kv_cache_swallows_non_200_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"server error")

    monitor = VramMonitor(_BASE_CONFIG, transport=httpx.MockTransport(handler))
    await monitor._flush_kv_cache()  # must not raise


@pytest.mark.asyncio
async def test_start_creates_task_and_stop_cancels_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = VramMonitor(_BASE_CONFIG)

    started = asyncio.Event()

    async def _forever() -> None:
        started.set()
        await asyncio.sleep(9999)

    monkeypatch.setattr(monitor, "_poll_loop", _forever)

    await monitor.start()
    assert monitor._task is not None
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert not monitor._task.done()

    await monitor.stop()
    assert monitor._task is None


@pytest.mark.asyncio
async def test_start_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = VramMonitor(_BASE_CONFIG)

    async def _forever() -> None:
        await asyncio.sleep(9999)

    monkeypatch.setattr(monitor, "_poll_loop", _forever)

    await monitor.start()
    first_task = monitor._task
    await monitor.start()
    assert monitor._task is first_task

    await monitor.stop()


@pytest.mark.asyncio
async def test_stop_without_start_is_safe() -> None:
    monitor = VramMonitor(_BASE_CONFIG)
    await monitor.stop()
    assert monitor._task is None


def test_init_normalises_trailing_slash_in_url() -> None:
    config = {
        "vram": {"warning_threshold_percent": 90, "flush_threshold_percent": 95},
        "llm": {"ollama_url": "http://localhost:11434/", "model": "qwen3:8b"},
    }
    monitor = VramMonitor(config)
    assert monitor._ollama_url == "http://localhost:11434"


def test_init_uses_defaults_when_keys_missing() -> None:
    monitor = VramMonitor({})
    assert monitor._warn_pct == 90
    assert monitor._flush_pct == 95
    assert monitor._ollama_url == "http://localhost:11434"
    assert monitor._model == ""
