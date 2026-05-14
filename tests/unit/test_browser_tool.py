"""Unit tests for :mod:`agent.tools.browser_tool`.

All tests are fully mocked — no Playwright import, no Chromium launch, no
network. ``ddgs.DDGS`` is patched out for :class:`WebSearchTool` so the
test suite never touches DuckDuckGo.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools import build_registry
from agent.tools.browser_tool import (
    WebFetchTool,
    WebSearchTool,
    _BrowserWorker,
    extract_main_text,
    format_fetch_result,
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
        "web_fetch_max_chars": 4000,
        # 0 → no enrichment fetches, so the ddgs-only path is what these
        # tests exercise. The enrichment path has its own tests below.
        "web_search_fetch_count": 0,
    },
}


def _enriching_config(fetch_count: int = 2, excerpt_chars: int = 200) -> dict[str, Any]:
    """A config that enables web-search enrichment fetches."""
    agent = dict(_BASE_CONFIG["agent"])
    agent["web_search_fetch_count"] = fetch_count
    agent["web_search_excerpt_chars"] = excerpt_chars
    agent["web_search_fetch_timeout_s"] = 5
    return {"agent": agent}


def _tool(config: dict[str, Any] | None = None) -> WebSearchTool:
    cfg = config if config is not None else _BASE_CONFIG
    return WebSearchTool(cfg)


def _patch_ddgs(results: list[dict[str, str]] | Exception) -> Any:
    """Return a context manager that patches ``ddgs.DDGS`` for the test.

    Pass a list of ``{title, href, body}`` dicts for a successful query,
    or an Exception instance to simulate a ddgs failure.
    """
    mock_ddgs_cls = MagicMock()
    instance = mock_ddgs_cls.return_value
    if isinstance(results, Exception):
        instance.text.side_effect = results
    else:
        instance.text.return_value = iter(results)
    return patch("ddgs.DDGS", mock_ddgs_cls), mock_ddgs_cls


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


def test_execute_returns_formatted_results() -> None:
    tool = _tool()
    raw = [
        {
            "title": "Python (programming language)",
            "href": "https://en.wikipedia.org/wiki/Python",
            "body": "An interpreted, high-level language.",
        },
        {
            "title": "Python.org",
            "href": "https://www.python.org",
            "body": "Official site.",
        },
    ]
    ctx, mock_ddgs_cls = _patch_ddgs(raw)
    with ctx:
        out = tool.execute({"query": "python programming"})

    assert "1. Python (programming language)" in out
    assert "https://en.wikipedia.org/wiki/Python" in out
    assert "An interpreted, high-level language." in out
    assert "2. Python.org" in out

    mock_ddgs_cls.return_value.text.assert_called_once_with(
        "python programming", max_results=5
    )


def test_execute_strips_query_whitespace() -> None:
    tool = _tool()
    ctx, mock_ddgs_cls = _patch_ddgs([])
    with ctx:
        tool.execute({"query": "  hello world  "})
    args, kwargs = mock_ddgs_cls.return_value.text.call_args
    assert args[0] == "hello world"


def test_execute_clamps_max_results_above_default() -> None:
    tool = _tool()
    ctx, mock_ddgs_cls = _patch_ddgs([])
    with ctx:
        tool.execute({"query": "q", "max_results": 999})
    _, kwargs = mock_ddgs_cls.return_value.text.call_args
    assert kwargs["max_results"] == 5


def test_execute_clamps_max_results_below_one() -> None:
    tool = _tool()
    ctx, mock_ddgs_cls = _patch_ddgs([])
    with ctx:
        tool.execute({"query": "q", "max_results": 0})
    _, kwargs = mock_ddgs_cls.return_value.text.call_args
    assert kwargs["max_results"] == 1


def test_execute_handles_non_numeric_max_results() -> None:
    tool = _tool()
    ctx, mock_ddgs_cls = _patch_ddgs([])
    with ctx:
        tool.execute({"query": "q", "max_results": "lots"})
    _, kwargs = mock_ddgs_cls.return_value.text.call_args
    assert kwargs["max_results"] == 5


def test_execute_rejects_empty_query() -> None:
    tool = _tool()
    with pytest.raises(ToolExecutionError):
        tool.execute({"query": "   "})


def test_execute_rejects_missing_query() -> None:
    tool = _tool()
    with pytest.raises(ToolExecutionError):
        tool.execute({})


def test_execute_wraps_ddgs_exception_as_tool_error() -> None:
    tool = _tool()
    ctx, _ = _patch_ddgs(RuntimeError("network down"))
    with ctx:
        with pytest.raises(ToolExecutionError) as exc_info:
            tool.execute({"query": "anything"})
    assert "search failed" in exc_info.value.user_message.lower()


def test_execute_returns_no_results_message_when_empty() -> None:
    tool = _tool()
    ctx, _ = _patch_ddgs([])
    with ctx:
        out = tool.execute({"query": "nothing"})
    assert out == "No results found."


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


def test_format_results_prefers_content_over_snippet() -> None:
    out = format_results(
        [
            {
                "title": "T",
                "snippet": "short snippet",
                "content": "the full extracted page content",
                "url": "https://a.example",
            }
        ]
    )
    assert "the full extracted page content" in out
    assert "short snippet" not in out


# -- extract_main_text -------------------------------------------------------


def test_extract_main_text_empty_returns_none() -> None:
    assert extract_main_text("") is None


def test_extract_main_text_extracts_article_body() -> None:
    html = (
        "<html><head><title>T</title></head><body><article>"
        "<p>" + "A clear sentence of real article content. " * 12 + "</p>"
        "</article></body></html>"
    )
    text = extract_main_text(html)
    assert text is not None
    assert "real article content" in text


# -- WebSearchTool enrichment ------------------------------------------------


def _html_echo(request: httpx.Request) -> httpx.Response:
    """MockTransport handler: 200 with HTML that echoes the requested URL."""
    return httpx.Response(200, text=f"<html><body>{request.url}</body></html>")


def test_execute_enriches_results_with_fetched_content() -> None:
    tool = WebSearchTool(
        _enriching_config(fetch_count=2),
        transport=httpx.MockTransport(_html_echo),
    )
    raw = [
        {"title": "A", "href": "https://a.example", "body": "snippet a"},
        {"title": "B", "href": "https://b.example", "body": "snippet b"},
        {"title": "C", "href": "https://c.example", "body": "snippet c"},
    ]
    ctx, _ = _patch_ddgs(raw)
    with patch(
        "agent.tools.browser_tool.extract_main_text",
        side_effect=lambda html: f"CONTENT::{html}",
    ), ctx:
        out = tool.execute({"query": "q"})

    # Top 2 results show fetched content in place of the snippet.
    assert "CONTENT::" in out
    assert "snippet a" not in out
    assert "snippet b" not in out
    # The 3rd result is beyond fetch_count -> still shows its snippet.
    assert "snippet c" in out


def test_execute_falls_back_to_snippet_when_fetch_fails() -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    tool = WebSearchTool(
        _enriching_config(fetch_count=2),
        transport=httpx.MockTransport(failing),
    )
    raw = [
        {"title": "A", "href": "https://a.example", "body": "snippet a"},
        {"title": "B", "href": "https://b.example", "body": "snippet b"},
    ]
    ctx, _ = _patch_ddgs(raw)
    with ctx:
        out = tool.execute({"query": "q"})

    # Non-200 -> no content extracted -> the snippet is retained.
    assert "snippet a" in out
    assert "snippet b" in out


def test_execute_truncates_fetched_content_to_excerpt_chars() -> None:
    tool = WebSearchTool(
        _enriching_config(fetch_count=1, excerpt_chars=50),
        transport=httpx.MockTransport(_html_echo),
    )
    raw = [{"title": "A", "href": "https://a.example", "body": "snippet a"}]
    ctx, _ = _patch_ddgs(raw)
    long_text = "word " * 100  # 500 chars
    with patch(
        "agent.tools.browser_tool.extract_main_text",
        return_value=long_text,
    ), ctx:
        out = tool.execute({"query": "q"})

    assert "…" in out  # truncation marker present
    assert long_text not in out  # full text not included


def test_execute_does_not_fetch_when_fetch_count_zero() -> None:
    # _BASE_CONFIG sets web_search_fetch_count: 0 -> the ddgs snippet is used
    # verbatim and no httpx client is constructed.
    tool = _tool()
    raw = [{"title": "A", "href": "https://a.example", "body": "snippet a"}]
    ctx, _ = _patch_ddgs(raw)
    with ctx:
        out = tool.execute({"query": "q"})
    assert "snippet a" in out


def test_worker_start_error_is_sticky_and_raises_on_call() -> None:
    worker = _BrowserWorker(_BASE_CONFIG)
    sentinel = ToolExecutionError("nope", "Browser broken.")
    worker._start_error = sentinel

    # Ensure _ensure_started does NOT spawn a thread when there's a recorded
    # start error from a previous attempt.
    with patch("threading.Thread") as mock_thread:
        with pytest.raises(ToolExecutionError) as exc_info:
            worker.call("fetch", url="https://example.com", max_chars=4000)
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
        worker.call("fetch", url="https://example.com", max_chars=4000)
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

    assert worker.call("fetch", url="https://example.com", max_chars=4000) == "PAYLOAD"


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


# -- WebFetchTool ------------------------------------------------------------


def _fetch_tool(config: dict[str, Any] | None = None) -> WebFetchTool:
    cfg = config if config is not None else _BASE_CONFIG
    worker = MagicMock(spec=_BrowserWorker)
    return WebFetchTool(worker, cfg)


def test_web_fetch_metadata() -> None:
    tool = _fetch_tool()
    assert tool.name == "web_fetch"
    assert "url" in tool.description.lower() or "page" in tool.description.lower()
    assert tool.schema["type"] == "object"


def test_web_fetch_schema_requires_only_url() -> None:
    tool = _fetch_tool()
    assert tool.schema["required"] == ["url"]
    props = tool.schema["properties"]
    assert props["url"]["type"] == "string"
    assert props["max_chars"]["type"] == "integer"


def test_web_fetch_delegates_to_worker_with_default_max() -> None:
    tool = _fetch_tool()
    tool._worker.call.return_value = "TEXT"  # type: ignore[attr-defined]
    out = tool.execute({"url": "https://example.com/article"})
    assert out == "TEXT"
    tool._worker.call.assert_called_once_with(  # type: ignore[attr-defined]
        "fetch", url="https://example.com/article", max_chars=4000
    )


def test_web_fetch_clamps_max_chars_above_default() -> None:
    tool = _fetch_tool()
    tool._worker.call.return_value = "ok"  # type: ignore[attr-defined]
    tool.execute({"url": "https://example.com", "max_chars": 999_999})
    _, kwargs = tool._worker.call.call_args  # type: ignore[attr-defined]
    assert kwargs["max_chars"] == 4000


def test_web_fetch_clamps_max_chars_below_floor() -> None:
    tool = _fetch_tool()
    tool._worker.call.return_value = "ok"  # type: ignore[attr-defined]
    tool.execute({"url": "https://example.com", "max_chars": 10})
    _, kwargs = tool._worker.call.call_args  # type: ignore[attr-defined]
    assert kwargs["max_chars"] == 200


def test_web_fetch_handles_non_numeric_max_chars() -> None:
    tool = _fetch_tool()
    tool._worker.call.return_value = "ok"  # type: ignore[attr-defined]
    tool.execute({"url": "https://example.com", "max_chars": "lots"})
    _, kwargs = tool._worker.call.call_args  # type: ignore[attr-defined]
    assert kwargs["max_chars"] == 4000


@pytest.mark.parametrize(
    "bad_url",
    ["file:///etc/passwd", "ftp://example.com/file", "javascript:alert(1)", "example.com", ""],
)
def test_web_fetch_rejects_non_http_url(bad_url: str) -> None:
    tool = _fetch_tool()
    with pytest.raises(ToolExecutionError):
        tool.execute({"url": bad_url})
    tool._worker.call.assert_not_called()  # type: ignore[attr-defined]


def test_web_fetch_rejects_missing_url() -> None:
    tool = _fetch_tool()
    with pytest.raises(ToolExecutionError):
        tool.execute({})


def test_web_fetch_strips_url_whitespace() -> None:
    tool = _fetch_tool()
    tool._worker.call.return_value = "ok"  # type: ignore[attr-defined]
    tool.execute({"url": "   https://example.com/page  "})
    _, kwargs = tool._worker.call.call_args  # type: ignore[attr-defined]
    assert kwargs["url"] == "https://example.com/page"


def test_web_fetch_propagates_worker_tool_execution_error() -> None:
    tool = _fetch_tool()
    tool._worker.call.side_effect = ToolExecutionError(  # type: ignore[attr-defined]
        "timeout", "The page took too long to load."
    )
    with pytest.raises(ToolExecutionError) as exc_info:
        tool.execute({"url": "https://example.com"})
    assert "too long" in exc_info.value.user_message


# -- format_fetch_result -----------------------------------------------------


def test_format_fetch_result_layout() -> None:
    out = format_fetch_result(
        title="Article Title",
        url="https://example.com/a",
        text="Hello world.",
        max_chars=4000,
    )
    assert out.startswith("Article Title\n\n")
    assert "Hello world." in out
    assert out.endswith("[source: https://example.com/a]")


def test_format_fetch_result_truncates_at_whitespace() -> None:
    body = "word " * 100  # 500 chars, ample whitespace
    out = format_fetch_result(title="T", url="u", text=body, max_chars=50)
    assert "…[truncated]" in out
    # Truncation must land at a whitespace boundary — no half-word before the marker.
    snippet = out.split("…[truncated]")[0]
    assert not snippet.rstrip().endswith("wor")


def test_format_fetch_result_no_truncation_when_short() -> None:
    out = format_fetch_result(title="T", url="u", text="short body", max_chars=4000)
    assert "[truncated]" not in out


def test_format_fetch_result_handles_empty_title() -> None:
    out = format_fetch_result(title="", url="https://example.com", text="body", max_chars=4000)
    # No leading blank line, no crash, source still appended.
    assert out.startswith("body")
    assert out.endswith("[source: https://example.com]")


def test_format_fetch_result_handles_empty_text() -> None:
    out = format_fetch_result(title="Title", url="u", text="", max_chars=4000)
    assert "Title" in out
    assert "[source: u]" in out


# -- registry / fetch worker checks -----------------------------------------


def test_registry_includes_web_fetch_when_browser_enabled() -> None:
    registry = build_registry(_BASE_CONFIG)
    assert "web_search" in registry
    assert "web_fetch" in registry
    assert isinstance(registry["web_fetch"], WebFetchTool)


def test_registry_omits_web_fetch_when_browser_disabled() -> None:
    config: dict[str, Any] = {
        "agent": {
            "shell_timeout_seconds": 30,
            "allowed_paths": [],
            "browser_enabled": False,
        },
    }
    registry = build_registry(config)
    assert "web_fetch" not in registry
    assert "web_search" not in registry
