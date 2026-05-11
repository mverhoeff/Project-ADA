"""Unit tests for :mod:`app.deps`."""

from __future__ import annotations

from typing import Any

import pytest

from app.deps import build_deps, build_session
from orchestrator.pipeline import PipelineDeps
from orchestrator.session import Session

_BASE_CONFIG: dict[str, Any] = {
    "stt": {"port": 8771},
    "llm": {
        "ollama_url": "http://localhost:11434",
        "model": "qwen3:8b",
        "temperature": 0.7,
    },
    "tts": {"port": 8772},
    "audio": {"output_device": None},
    "agent": {
        "shell_timeout_seconds": 30,
        "allowed_paths": ["~/Documents"],
    },
}


def test_build_session_sets_platform() -> None:
    session = build_session(_BASE_CONFIG)
    assert isinstance(session, Session)
    assert session.platform in {"windows", "linux"}
    assert session.history == []


def test_build_deps_returns_wired_bundle() -> None:
    deps = build_deps(_BASE_CONFIG)
    assert isinstance(deps, PipelineDeps)
    # tools list and executor's registry must reference the same instances.
    executor_tools = list(deps.executor._tools.values())  # type: ignore[attr-defined]
    assert deps.tools == executor_tools
    # ShellTool, SystemInfoTool, FileReadTool — exactly three built-ins today.
    assert len(deps.tools) == 3
    names = {t.name for t in deps.tools}
    assert names == {"run_shell", "get_system_info", "read_file"}


def test_build_deps_is_pure_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """No network or filesystem I/O should happen during build_deps."""
    import httpx

    def fail(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - guard
        raise AssertionError("build_deps must not open an httpx client")

    monkeypatch.setattr(httpx, "AsyncClient", fail)
    monkeypatch.setattr(httpx, "Client", fail)

    deps = build_deps(_BASE_CONFIG)
    assert isinstance(deps, PipelineDeps)
