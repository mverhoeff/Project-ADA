"""Tool registry for the ADA agent layer."""

from __future__ import annotations

from typing import Any

from agent.tools.app_tool import OpenAppTool
from agent.tools.base import BaseTool
from agent.tools.browser_tool import WebFetchTool, WebSearchTool, _BrowserWorker
from agent.tools.system_tool import FileReadTool, ShellTool, SystemInfoTool


def build_registry(config: dict[str, Any]) -> dict[str, BaseTool]:
    """Instantiate all built-in tools and return a name-keyed registry.

    The browser-backed tools are only included when
    ``config["agent"]["browser_enabled"]`` is true; when disabled, no
    Playwright import happens at all, so the rest of the app stays usable
    on machines without Chromium installed.

    Args:
        config: Full config dict, as returned by :func:`core.config.load_config`.

    Returns:
        Dict mapping each tool's :attr:`BaseTool.name` to its instance.
    """
    tools: list[BaseTool] = [
        ShellTool(config),
        SystemInfoTool(),
        FileReadTool(config),
        OpenAppTool(),
    ]
    if bool(config.get("agent", {}).get("browser_enabled", True)):
        tools.append(WebSearchTool(config))
        worker = _BrowserWorker(config)
        tools.append(WebFetchTool(worker, config))
    return {t.name: t for t in tools}


__all__ = [
    "BaseTool",
    "FileReadTool",
    "OpenAppTool",
    "ShellTool",
    "SystemInfoTool",
    "WebFetchTool",
    "WebSearchTool",
    "build_registry",
]
