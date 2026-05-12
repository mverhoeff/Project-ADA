"""Agent tool that searches the web via a headless Chromium browser.

A single Chromium instance is launched lazily on a dedicated daemon thread
(:class:`_BrowserWorker`) and reused across calls — Playwright's sync API
objects are pinned to their creating thread, and the orchestrator dispatches
tool calls through :func:`asyncio.to_thread`, which uses a non-deterministic
worker pool. The dedicated thread is the simplest way to keep one browser
instance live across calls without restructuring the executor.

The public surface is :class:`WebSearchTool`; the worker class is module-private.
"""

from __future__ import annotations

import queue
import threading
import urllib.parse
from typing import Any, ClassVar

from agent.tools.base import BaseTool
from core.exceptions import ToolExecutionError
from core.logger import get_logger

_log = get_logger(__name__)


_DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_DDG_RESULT_SELECTOR = "div.result"
_DDG_TITLE_SELECTOR = "a.result__a"
_DDG_SNIPPET_SELECTOR = "a.result__snippet, .result__snippet"

_WORKER_START_TIMEOUT_S = 30.0


def _unwrap_ddg_redirect(href: str) -> str:
    """DDG wraps result hrefs in ``/l/?uddg=<encoded-url>&…``. Unwrap if present."""
    if not href:
        return href
    parsed = urllib.parse.urlparse(href)
    if parsed.path.startswith("/l/") or parsed.netloc.endswith("duckduckgo.com"):
        qs = urllib.parse.parse_qs(parsed.query)
        uddg = qs.get("uddg", [None])[0]
        if uddg:
            return uddg
    if href.startswith("//"):
        return "https:" + href
    return href


def format_results(results: list[dict[str, str]]) -> str:
    """Render a list of ``{title, snippet, url}`` dicts as plain text.

    Pure function — extracted so it can be unit-tested without Playwright.
    """
    if not results:
        return "No results found."
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r.get('title', '').strip()}")
        snippet = r.get("snippet", "").strip()
        if snippet:
            lines.append(f"   {snippet}")
        url = r.get("url", "").strip()
        if url:
            lines.append(f"   {url}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class _BrowserWorker:
    """Owns one Chromium instance on a dedicated daemon thread.

    Lazy: the thread is spawned on the first call to :meth:`call`. If
    Playwright import or browser launch fails, the error is captured and
    re-raised on every subsequent call so the LLM gets a stable, spoken
    failure each time.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        agent_cfg = config.get("agent", {})
        self._headless: bool = bool(agent_cfg.get("browser_headless", True))
        self._nav_timeout_ms: int = int(agent_cfg.get("browser_navigation_timeout_ms", 15000))
        self._req_queue: queue.Queue[Any] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._start_error: Exception | None = None

    def call(self, op: str, **kwargs: Any) -> Any:
        """Dispatch ``op`` with ``kwargs`` to the worker thread and return its result.

        Raises whatever exception the operation raised inside the worker.
        """
        self._ensure_started()
        if self._start_error is not None:
            raise self._start_error
        response: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._req_queue.put((op, kwargs, response))
        status, payload = response.get()
        if status == "err":
            raise payload  # type: ignore[misc]
        return payload

    def _ensure_started(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if self._start_error is not None:
            return
        self._started.clear()
        self._thread = threading.Thread(
            target=self._run, name="ada-browser-worker", daemon=True
        )
        self._thread.start()
        if not self._started.wait(timeout=_WORKER_START_TIMEOUT_S):
            self._start_error = ToolExecutionError(
                "Browser worker did not start within timeout.",
                "The browser took too long to start.",
            )

    def _run(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self._start_error = ToolExecutionError(
                f"Playwright not installed: {exc}",
                "The browser is not installed. Run 'pip install playwright' "
                "and 'playwright install chromium'.",
            )
            self._started.set()
            return

        try:
            pw = sync_playwright().start()
            browser = pw.chromium.launch(headless=self._headless)
            context = browser.new_context()
        except Exception as exc:
            _log.error("browser_launch_failed", error=str(exc))
            self._start_error = ToolExecutionError(
                f"Chromium launch failed: {exc}",
                "I couldn't start the browser. The Chromium runtime may not be installed.",
            )
            self._started.set()
            return

        _log.info("browser_worker_started", headless=self._headless)
        self._started.set()

        try:
            while True:
                item = self._req_queue.get()
                if item is None:
                    break
                op, kwargs, response = item
                try:
                    handler = getattr(self, f"_op_{op}", None)
                    if handler is None:
                        raise ToolExecutionError(
                            f"Unknown browser op: {op!r}",
                            "I tried an unknown browser action.",
                        )
                    result = handler(context, **kwargs)
                    response.put(("ok", result))
                except Exception as exc:
                    response.put(("err", exc))
        finally:
            try:
                context.close()
                browser.close()
                pw.stop()
            except Exception:
                pass

    def _op_search(self, context: Any, query: str, max_results: int) -> str:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        page = context.new_page()
        try:
            url = _DDG_HTML_URL + "?" + urllib.parse.urlencode({"q": query})
            try:
                page.goto(url, timeout=self._nav_timeout_ms, wait_until="domcontentloaded")
            except PlaywrightTimeoutError as exc:
                raise ToolExecutionError(
                    f"Navigation to {url!r} timed out after {self._nav_timeout_ms} ms.",
                    "The search timed out. The network may be slow or unreachable.",
                ) from exc

            elements = page.query_selector_all(_DDG_RESULT_SELECTOR)
            results: list[dict[str, str]] = []
            for el in elements:
                if len(results) >= max_results:
                    break
                title_el = el.query_selector(_DDG_TITLE_SELECTOR)
                if title_el is None:
                    continue
                title = (title_el.inner_text() or "").strip()
                href = (title_el.get_attribute("href") or "").strip()
                snippet_el = el.query_selector(_DDG_SNIPPET_SELECTOR)
                snippet = (snippet_el.inner_text().strip() if snippet_el is not None else "")
                results.append(
                    {
                        "title": title,
                        "snippet": snippet,
                        "url": _unwrap_ddg_redirect(href),
                    }
                )
            return format_results(results)
        finally:
            try:
                page.close()
            except Exception:
                pass


class WebSearchTool(BaseTool):
    """Search the web and return the top results as plain text for the LLM."""

    name = "web_search"
    description = (
        "Search the web and return the top results as text. "
        "Use this whenever the user asks about current events, things on the "
        "internet, or facts you are not certain about."
    )
    schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, worker: _BrowserWorker, config: dict[str, Any]) -> None:
        agent_cfg = config.get("agent", {})
        self._worker = worker
        self._default_max: int = int(agent_cfg.get("browser_max_results", 5))

    def execute(self, params: dict[str, Any]) -> str:
        query = params.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolExecutionError(
                "web_search called without a non-empty 'query' argument.",
                "I need something to search for.",
            )
        raw_max = params.get("max_results", self._default_max)
        try:
            requested = int(raw_max)
        except (TypeError, ValueError):
            requested = self._default_max
        max_results = max(1, min(requested, self._default_max))

        return self._worker.call("search", query=query.strip(), max_results=max_results)
