"""Agent tools for web search and single-page fetching.

:class:`WebSearchTool` queries DuckDuckGo through the maintained ``ddgs``
library, which handles DDG's evolving anti-bot countermeasures internally
— scraping ``html.duckduckgo.com`` with a headless Chromium no longer
works (DDG returns its JS homepage instead of results). It then fetches
the top few result pages concurrently over plain HTTP and replaces the
short DDG snippet with each page's extracted main text, so the model
works with real content instead of meta-descriptions. Page fetching is
best-effort — a page that times out or errors simply falls back to its
snippet, so one slow page never fails the whole search.

:class:`WebFetchTool` still drives a headless Chromium for arbitrary
URLs, since a single user-chosen page may be JS-heavy in a way the
search-result enrichment's plain-HTTP fetch cannot handle. A single
Chromium instance is launched lazily on a dedicated daemon thread
(:class:`_BrowserWorker`) and reused across calls — Playwright's sync
API objects are pinned to their creating thread, and the orchestrator
dispatches tool calls through :func:`asyncio.to_thread`, which uses a
non-deterministic worker pool.
"""

from __future__ import annotations

import concurrent.futures
import queue
import threading
from typing import Any, ClassVar

import httpx

from agent.tools.base import BaseTool
from core.exceptions import ToolExecutionError
from core.logger import get_logger

_log = get_logger(__name__)


_WORKER_START_TIMEOUT_S = 30.0

# Sent on the search-result enrichment fetches so sites don't immediately
# 403 a bare client; the headless-Chromium path sets its own UA.
_FETCH_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def extract_main_text(html: str) -> str | None:
    """Extract a page's main readable text from raw HTML via trafilatura.

    Pure function — no I/O. Returns ``None`` when trafilatura is not
    installed or finds nothing extractable, so callers can fall back to a
    search snippet or a coarser extraction.

    Args:
        html: Raw HTML markup of a page.

    Returns:
        The extracted main text, stripped, or ``None`` if nothing usable
        could be extracted.
    """
    try:
        import trafilatura
    except ImportError:
        return None
    text = trafilatura.extract(
        html,
        favor_recall=True,
        include_comments=False,
        include_tables=False,
    )
    if not text or not text.strip():
        return None
    return text.strip()


def format_results(results: list[dict[str, str]]) -> str:
    """Render a list of search-result dicts as plain text.

    Each dict carries ``title``, ``url`` and a body — the extracted page
    ``content`` when the result was fetched and enriched, otherwise the
    search engine's ``snippet``. Pure function — extracted so it can be
    unit-tested without network I/O.
    """
    if not results:
        return "No results found."
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r.get('title', '').strip()}")
        body = (r.get("content") or r.get("snippet") or "").strip()
        if body:
            lines.append(f"   {body}")
        url = r.get("url", "").strip()
        if url:
            lines.append(f"   {url}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_fetch_result(title: str, url: str, text: str, max_chars: int) -> str:
    """Render an extracted page as a title + body + source-line block.

    Truncates ``text`` at the last whitespace boundary ≤ ``max_chars`` and
    appends ``…[truncated]`` when clipping. Pure function — no I/O.
    """
    body = (text or "").strip()
    if len(body) > max_chars:
        cut = body.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        body = body[:cut].rstrip() + "…[truncated]"

    parts: list[str] = []
    title_clean = (title or "").strip()
    if title_clean:
        parts.append(title_clean)
        parts.append("")
    parts.append(body)
    parts.append("")
    parts.append(f"[source: {url}]")
    return "\n".join(parts)


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

    def _op_fetch(self, context: Any, url: str, max_chars: int) -> str:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        page = context.new_page()
        try:
            try:
                page.goto(url, timeout=self._nav_timeout_ms, wait_until="domcontentloaded")
            except PlaywrightTimeoutError as exc:
                raise ToolExecutionError(
                    f"Navigation to {url!r} timed out after {self._nav_timeout_ms} ms.",
                    "The page took too long to load.",
                ) from exc

            html = page.content()
            title = (page.title() or "").strip()

            text = extract_main_text(html)
            if not text:
                try:
                    text = page.locator("body").inner_text()
                except Exception:
                    text = None

            if not text or not text.strip():
                raise ToolExecutionError(
                    f"Page returned no extractable text: {url}",
                    "I couldn't read that page.",
                )

            return format_fetch_result(title, url, text, max_chars)
        finally:
            try:
                page.close()
            except Exception:
                pass


class WebSearchTool(BaseTool):
    """Search the web via ``ddgs`` and return the top results with content.

    The result list comes from DuckDuckGo via ``ddgs``; the top
    ``agent.web_search_fetch_count`` result pages are then fetched
    concurrently over plain HTTP and their extracted main text replaces
    the short DDG snippet. Enrichment is best-effort — a page that fails
    to fetch keeps its snippet.

    Args:
        config: Full config dict; reads the ``agent`` subtree
            (``browser_max_results``, ``web_search_fetch_count``,
            ``web_search_excerpt_chars``, ``web_search_fetch_timeout_s``).
        transport: Optional ``httpx`` transport for the enrichment
            fetches, used by tests to inject mock responses without
            hitting the network.
    """

    name = "web_search"
    description = (
        "Search the web and return the top results, each with the main "
        "readable text extracted from the result page. Use this whenever "
        "the user asks about current events, things on the internet, or "
        "facts you are not certain about."
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

    def __init__(
        self,
        config: dict[str, Any],
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        agent_cfg = config.get("agent", {})
        self._default_max: int = int(agent_cfg.get("browser_max_results", 5))
        self._fetch_count: int = int(agent_cfg.get("web_search_fetch_count", 3))
        self._excerpt_chars: int = int(
            agent_cfg.get("web_search_excerpt_chars", 1000)
        )
        self._fetch_timeout: float = float(
            agent_cfg.get("web_search_fetch_timeout_s", 5)
        )
        self._transport = transport

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

        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise ToolExecutionError(
                f"ddgs not installed: {exc}",
                "The web search library is not installed.",
            ) from exc

        try:
            raw = list(DDGS().text(query.strip(), max_results=max_results))
        except Exception as exc:
            _log.error("web_search_failed", error=str(exc))
            raise ToolExecutionError(
                f"ddgs search failed: {exc}",
                "The search failed. The network may be unreachable.",
            ) from exc

        results = [
            {
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
            }
            for r in raw
        ]
        self._enrich(results)
        return format_results(results)

    def _enrich(self, results: list[dict[str, str]]) -> None:
        """Fetch the top results' pages concurrently and attach ``content``.

        Mutates ``results`` in place: each successfully fetched entry
        gains a ``content`` key with the page's extracted main text.
        Best-effort — a fetch failure leaves the entry untouched so
        :func:`format_results` falls back to its snippet.
        """
        if self._fetch_count <= 0:
            return
        targets = [
            (i, r["url"])
            for i, r in enumerate(results[: self._fetch_count])
            if r.get("url")
        ]
        if not targets:
            return
        with httpx.Client(
            transport=self._transport,
            timeout=self._fetch_timeout,
            follow_redirects=True,
            headers={"User-Agent": _FETCH_USER_AGENT},
        ) as client, concurrent.futures.ThreadPoolExecutor(
            max_workers=len(targets)
        ) as pool:
            futures = {
                pool.submit(self._fetch_one, client, url): idx
                for idx, url in targets
            }
            for fut in concurrent.futures.as_completed(futures):
                content = fut.result()  # _fetch_one swallows its own errors
                if content:
                    results[futures[fut]]["content"] = content

    def _fetch_one(self, client: httpx.Client, url: str) -> str | None:
        """Fetch and extract one page; return ``None`` on any failure."""
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                return None
            text = extract_main_text(resp.text)
        except Exception as exc:
            _log.debug("web_search_fetch_failed", url=url, error=str(exc))
            return None
        if not text:
            return None
        if len(text) > self._excerpt_chars:
            cut = text.rfind(" ", 0, self._excerpt_chars)
            if cut <= 0:
                cut = self._excerpt_chars
            text = text[:cut].rstrip() + "…"
        return text


class WebFetchTool(BaseTool):
    """Open a single URL and return the page's main readable text."""

    name = "web_fetch"
    description = (
        "Open a single URL and return the page's main readable text. "
        "Use this to read a page in detail after web_search returns its URL."
    )
    schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute http(s) URL of the page to read.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Truncate the extracted text at this many characters.",
            },
        },
        "required": ["url"],
    }

    _MIN_MAX_CHARS: ClassVar[int] = 200

    def __init__(self, worker: _BrowserWorker, config: dict[str, Any]) -> None:
        agent_cfg = config.get("agent", {})
        self._worker = worker
        self._default_max: int = int(agent_cfg.get("web_fetch_max_chars", 4000))

    def execute(self, params: dict[str, Any]) -> str:
        url = params.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ToolExecutionError(
                "web_fetch called without a non-empty 'url' argument.",
                "I need a URL to read.",
            )
        url = url.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ToolExecutionError(
                f"web_fetch refused non-http(s) URL: {url!r}",
                "I can only read web pages.",
            )

        raw_max = params.get("max_chars", self._default_max)
        try:
            requested = int(raw_max)
        except (TypeError, ValueError):
            requested = self._default_max
        max_chars = max(self._MIN_MAX_CHARS, min(requested, self._default_max))

        return self._worker.call("fetch", url=url, max_chars=max_chars)
