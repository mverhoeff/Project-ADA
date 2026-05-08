"""Unit tests for :mod:`agent.tools.base`."""

from __future__ import annotations

from typing import Any

import pytest

from agent.tools.base import BaseTool


def test_base_tool_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        BaseTool()  # type: ignore[abstract]


def test_subclass_without_execute_cannot_instantiate() -> None:
    class IncompleteTool(BaseTool):
        name = "incomplete"
        description = "missing execute"
        schema: dict[str, Any] = {}

    with pytest.raises(TypeError):
        IncompleteTool()  # type: ignore[abstract]


def test_concrete_subclass_works() -> None:
    class EchoTool(BaseTool):
        name = "echo"
        description = "Echoes its input."
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

        def execute(self, params: dict[str, Any]) -> str:
            return str(params["text"])

    tool = EchoTool()
    assert tool.name == "echo"
    assert tool.description == "Echoes its input."
    assert tool.schema["required"] == ["text"]
    assert tool.execute({"text": "hi"}) == "hi"
