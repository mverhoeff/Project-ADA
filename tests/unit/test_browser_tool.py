"""Unit tests for :mod:`agent.tools.browser_tool`.

All tests are fully mocked — no Playwright import, no Chromium launch.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.tools import build_registry
from agent.tools.browser_tool import (
    WebSearchTool,
    _BrowserWorker,
    _unwrap_ddg_redirect,
    format_results,
)
from core.exceptions import ToolExecutionError

_BASE_CONFIG: dict[str, Any] = {
    "agent": {
        "shell_timeout_seconds": 30,
        "allowed_paths": [],
        "browser_enabled": True,
        "browser_headless": True,
        "browser_max_results": 5,
        "browser_navigation_timeout_ms": 15000,
    },
}


def _tool(config: dict[str, Any] | None = None) -> WebSearchTool:
    cfg = config if config is not None else _BASE_CONFIG
    worker = MagicMock(spec=_BrowserWorker)
    return WebSearchTool(worker, cfg)


def test_metadata_is_complete() -> None:
    tool = _tool()
    assert tool.name == "web_search"
    assert "search" in tool.description.lower()
    assert tool.schema["type"] == "object"


def test_schema_requires_only_query() -> None:
    tool = _tool()
    assert tool.schema["required"] == ["query"]
    props = tool.schema["properties"]
    assert props["query"]["type"] == "string"
    assert props["max_results"]["type"] == "integer"


def test_execute_delegates_to_worker_with_default_max() -> None:
    tool = _tool()
    tool._worker.call.return_value = "RESULTS"  # type: ignore[attr-defined]
    out = tool.execute({"query": "berlin weather"})
    assert out == "RESULTS"
    tool._worker.call.assert_called_once_with(  # type: ignore[attr-defined]
        "search", query="berlin weather", max_results=5
    )


def test_execute_strips_query_whitespace() -> None:
    tool = _tool()
    tool._worker.call.return_value = "ok"  # type: ignore[attr-defined]
    tool.execute({"query": "  hello world  "})
    args, kwargs = tool._worker.call.call_args  # type: ignore[attr-defined]
    assert kwargs["query"] == "hello world"


def test_execute_clamps_max_results_above_default() -> None:
    tool = _tool()
    tool._worker.call.return_value = "ok"  # type: ignore[attr-defined]
    tool.execute({"query": "q", "max_results": 999})
    _, kwargs = tool._worker.call.call_args  # type: ignore[attr-defined]
    assert kwargs["max_results"] == 5


def test_execute_clamps_max_results_below_one() -> None:
    tool = _tool()
    tool._worker.call.return_value = "ok"  # type: ignore[attr-defined]
    tool.execute({"query": "q", "max_results": 0})
    _, kwargs = tool._worker.call.call_args  # type: ignore[attr-defined]
    assert kwargs["max_results"] == 1


def test_execute_handles_non_numeric_max_results() -> None:
    tool = _tool()
    tool._worker.call.return_value = "ok"  # type: ignore[attr-defined]
    tool.execute({"query": "q", "max_results": "lots"})
    _, kwargs = tool._worker.call.call_args  # type: ignore[attr-defined]
    assert kwargs["max_results"] == 5


def test_execute_rejects_empty_query() -> None:
    tool = _tool()
    with pytest.raises(ToolExecutionError):
        tool.execute({"query": "   "})
    tool._worker.call.assert_not_called()  # type: ignore[attr-defined]


def test_execute_rejects_missing_query() -> None:
    tool = _tool()
    with pytest.raises(ToolExecutionError):
        tool.execute({})


def test_execute_propagates_worker_tool_execution_error() -> None:
    tool = _tool()
    tool._worker.call.side_effect = ToolExecutionError(  # type: ignore[attr-defined]
        "boom", "I couldn't reach the web right now."
    )
    with pytest.raises(ToolExecutionError) as exc_info:
        tool.execute({"query": "anything"})
    assert "reach the web" in exc_info.value.user_message


def test_format_results_layout() -> None:
    out = format_results(
        [
            {"title": "First", "snippet": "snippet one", "url": "https://a.example"},
            {"title": "Second", "snippet": "snippet two", "url": "https://b.example"},
        ]
    )
    assert "1. First" in out
    assert "2. Second" in out
    assert "https://a.example" in out
    assert "snippet two" in out
    # Each entry separated by a blank line
    assert "\n\n" in out


def test_format_results_empty() -> None:
    assert format_results([]) == "No results found."


def test_format_results_missing_fields() -> None:
    out = format_results([{"title": "Only title"}])
    assert "1. Only title" in out
    # No crash when snippet/url absent.


def test_unwrap_ddg_redirect_real_target() -> None:
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fwiki.example%2Fpage&rut=abc"
    assert _unwrap_ddg_redirect(href) == "https://wiki.example/page"


def test_unwrap_ddg_redirect_passthrough() -> None:
    href = "https://direct.example/page"
    assert _unwrap_ddg_redirect(href) == href


def test_unwrap_ddg_redirect_protocol_relative() -> None:
    assert _unwrap_ddg_redirect("//direct.example/page") == "https://direct.example/page"


def test_worker_start_error_is_sticky_and_raises_on_call() -> None:
    worker = _BrowserWorker(_BASE_CONFIG)
    sentinel = ToolExecutionError("nope", "Browser broken.")
    worker._start_error = sentinel

    # Ensure _ensure_started does NOT spawn a thread when there's a recorded
    # start error from a previous attempt.
    with patch("threading.Thread") as mock_thread:
        with pytest.raises(ToolExecutionError) as exc_info:
            worker.call("search", query="q", max_results=1)
        mock_thread.assert_not_called()
    assert exc_info.value is sentinel


def test_worker_call_re_raises_error_payload_from_response_queue() -> None:
    worker = _BrowserWorker(_BASE_CONFIG)
    # Pretend the worker thread is alive so _ensure_started is a no-op.
    fake_thread = MagicMock()
    fake_thread.is_alive.return_value = True
    worker._thread = fake_thread

    boom = ToolExecutionError("op failed", "Tool blew up.")

    def fake_put(item: Any) -> None:
        _, _, response_q = item
        response_q.put(("err", boom))

    worker._req_queue.put = fake_put  # type: ignore[method-assign]

    with pytest.raises(ToolExecutionError) as exc_info:
        worker.call("search", query="q", max_results=1)
    assert exc_info.value is boom


def test_worker_call_returns_payload_on_ok() -> None:
    worker = _BrowserWorker(_BASE_CONFIG)
    fake_thread = MagicMock()
    fake_thread.is_alive.return_value = True
    worker._thread = fake_thread

    def fake_put(item: Any) -> None:
        _, _, response_q = item
        response_q.put(("ok", "PAYLOAD"))

    worker._req_queue.put = fake_put  # type: ignore[method-assign]

    assert worker.call("search", query="q", max_results=1) == "PAYLOAD"


def test_registry_includes_web_search_when_enabled() -> None:
    registry = build_registry(_BASE_CONFIG)
    assert "web_search" in registry
    assert isinstance(registry["web_search"], WebSearchTool)


def test_registry_omits_web_search_when_disabled() -> None:
    config: dict[str, Any] = {
        "agent": {
            "shell_timeout_seconds": 30,
            "allowed_paths": [],
            "browser_enabled": False,
        },
    }
    registry = build_registry(config)
    assert "web_search" not in registry
    # The other built-ins are still present.
    assert set(registry.keys()) == {"run_shell", "get_system_info", "read_file", "open_app"}
