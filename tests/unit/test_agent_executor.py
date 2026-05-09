"""Unit tests for :mod:`agent.executor`."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from agent.executor import ToolExecutor
from agent.tools.base import BaseTool
from core.exceptions import ToolExecutionError


class _RecordingTool(BaseTool):
    name = "fake"
    description = "Records the params it was called with."
    schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.last_params: dict[str, Any] | None = None

    def execute(self, params: dict[str, Any]) -> str:
        self.last_params = params
        return f"called with {params}"


class _RaisingTool(BaseTool):
    name = "boom"
    description = "Always raises."
    schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def execute(self, params: dict[str, Any]) -> str:
        raise self._exc


def test_dispatches_known_tool() -> None:
    tool = _RecordingTool()
    executor = ToolExecutor({tool.name: tool})

    result = executor.execute({"name": "fake", "arguments": {"x": 1}})

    assert result == "called with {'x': 1}"
    assert tool.last_params == {"x": 1}


def test_returns_error_string_for_unknown_tool() -> None:
    executor = ToolExecutor({})
    result = executor.execute({"name": "no_such_tool"})
    assert "unknown" in result.lower()
    assert "no_such_tool" in result


def test_wraps_generic_exception_as_tool_execution_error() -> None:
    tool = _RaisingTool(RuntimeError("boom"))
    executor = ToolExecutor({tool.name: tool})

    with pytest.raises(ToolExecutionError) as exc_info:
        executor.execute({"name": "boom", "arguments": {}})

    assert "boom" in exc_info.value.user_message
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_reraises_tool_execution_error_unchanged() -> None:
    original = ToolExecutionError("blocked", "That command is not allowed.")
    tool = _RaisingTool(original)
    executor = ToolExecutor({tool.name: tool})

    with pytest.raises(ToolExecutionError) as exc_info:
        executor.execute({"name": "boom", "arguments": {}})

    assert exc_info.value is original
    assert exc_info.value.user_message == "That command is not allowed."


def test_missing_arguments_key_defaults_to_empty_dict() -> None:
    tool = _RecordingTool()
    executor = ToolExecutor({tool.name: tool})

    executor.execute({"name": "fake"})

    assert tool.last_params == {}


def test_non_dict_arguments_defaults_to_empty_dict() -> None:
    tool = _RecordingTool()
    executor = ToolExecutor({tool.name: tool})

    executor.execute({"name": "fake", "arguments": "not a dict"})

    assert tool.last_params == {}
