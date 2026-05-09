"""Dispatches parsed tool calls to registered tools and returns results.

The executor is the thin synchronous boundary between a parsed tool-call dict
and a concrete :class:`agent.tools.base.BaseTool` implementation. It logs each
dispatch, returns a plain string for the LLM to consume, and converts any
unexpected tool exception into :class:`core.exceptions.ToolExecutionError`.
"""

from __future__ import annotations

from typing import Any

from agent.tools.base import BaseTool
from core.exceptions import ToolExecutionError
from core.logger import get_logger

_log = get_logger(__name__)


class ToolExecutor:
    """Looks up tools by name and runs them.

    Args:
        tools: Mapping of tool name to instance, as built by
            :func:`agent.tools.build_registry`.
    """

    def __init__(self, tools: dict[str, BaseTool]) -> None:
        self._tools = tools

    def execute(self, tool_call: dict[str, Any]) -> str:
        """Dispatch a parsed tool call and return its result string.

        Args:
            tool_call: Dict with at least a ``"name"`` key and an optional
                ``"arguments"`` key.

        Returns:
            The tool's result string on success, or a human-readable error
            string if the tool name is not in the registry.

        Raises:
            ToolExecutionError: If the tool itself raises any exception.
                Existing :class:`ToolExecutionError` instances are re-raised
                unchanged so a tool's own ``user_message`` is preserved.
        """
        name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}

        tool = self._tools.get(name)
        if tool is None:
            _log.warning("unknown_tool", name=name)
            return f"Error: unknown tool '{name}'."

        _log.info("tool_executing", name=name, arguments=arguments)
        try:
            result = tool.execute(arguments)
        except ToolExecutionError:
            raise
        except Exception as exc:
            _log.error("tool_exception", name=name, error=str(exc))
            raise ToolExecutionError(
                f"Tool '{name}' raised {type(exc).__name__}: {exc}",
                f"I encountered an error running {name}.",
            ) from exc

        _log.info("tool_done", name=name)
        return result
