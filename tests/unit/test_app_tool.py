"""Unit tests for :mod:`agent.tools.app_tool`."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.tools import build_registry
from agent.tools.app_tool import OpenAppTool
from core.exceptions import ToolExecutionError


def test_metadata_is_complete() -> None:
    tool = OpenAppTool()
    assert tool.name == "open_app"
    assert "Spotify" in tool.description or "launch" in tool.description.lower()
    assert tool.schema["required"] == ["name"]
    assert tool.schema["properties"]["name"]["type"] == "string"


def test_execute_calls_open_app_and_returns_success_string() -> None:
    tool = OpenAppTool()
    with patch("ada_platform.launcher.open_app") as mock_open:
        result = tool.execute({"name": "Spotify"})
    mock_open.assert_called_once_with("Spotify")
    assert result == "Opened Spotify."


def test_execute_propagates_tool_execution_error() -> None:
    tool = OpenAppTool()
    with patch(
        "ada_platform.launcher.open_app",
        side_effect=ToolExecutionError("not found", "I couldn't find an app called Spotify."),
    ):
        with pytest.raises(ToolExecutionError) as exc_info:
            tool.execute({"name": "Spotify"})
    assert "Spotify" in exc_info.value.user_message


def test_execute_missing_name_raises_key_error() -> None:
    tool = OpenAppTool()
    with pytest.raises(KeyError):
        tool.execute({})


def test_registry_contains_open_app() -> None:
    registry = build_registry({"agent": {"shell_timeout_seconds": 30, "allowed_paths": []}})
    assert "open_app" in registry
    assert isinstance(registry["open_app"], OpenAppTool)
