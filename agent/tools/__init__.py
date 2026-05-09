"""Tool registry for the ADA agent layer."""

from __future__ import annotations

from typing import Any

from agent.tools.base import BaseTool
from agent.tools.system_tool import FileReadTool, ShellTool, SystemInfoTool


def build_registry(config: dict[str, Any]) -> dict[str, BaseTool]:
    """Instantiate all built-in tools and return a name-keyed registry.

    Args:
        config: Full config dict, as returned by :func:`core.config.load_config`.

    Returns:
        Dict mapping each tool's :attr:`BaseTool.name` to its instance.
    """
    tools: list[BaseTool] = [
        ShellTool(config),
        SystemInfoTool(),
        FileReadTool(config),
    ]
    return {t.name: t for t in tools}


__all__ = [
    "BaseTool",
    "FileReadTool",
    "ShellTool",
    "SystemInfoTool",
    "build_registry",
]
