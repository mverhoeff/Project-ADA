"""Unit tests for :mod:`services.llm.prompt_builder`."""

from __future__ import annotations

from datetime import date
from typing import Any

from agent.tools.base import BaseTool
from orchestrator.session import Session
from services.llm.prompt_builder import build_messages, build_tools


class _SearchTool(BaseTool):
    name = "search"
    description = "Search the web."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def execute(self, params: dict[str, Any]) -> str:
        return ""


class _ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a file from disk."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }

    def execute(self, params: dict[str, Any]) -> str:
        return ""


def test_system_message_is_first() -> None:
    messages = build_messages(Session(platform="linux"))
    assert messages[0]["role"] == "system"


def test_system_contains_platform() -> None:
    messages = build_messages(Session(platform="linux"))
    assert "linux" in messages[0]["content"]


def test_system_contains_today_date() -> None:
    messages = build_messages(Session(platform="linux"))
    assert date.today().isoformat() in messages[0]["content"]


def test_tools_not_declared_in_system_when_native_tool_calling_used() -> None:
    tools = [_SearchTool(), _ReadFileTool()]
    messages = build_messages(Session(platform="linux"), tools=tools)
    system_content = messages[0]["content"]
    assert "Search the web." not in system_content
    assert "read_file" not in system_content
    assert "Available tools" not in system_content
    assert '"function"' not in system_content


def test_history_appended_after_system() -> None:
    session = Session(
        platform="linux",
        history=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    )
    messages = build_messages(session)
    assert messages[1:] == session.history


def test_no_tools_omits_tool_block() -> None:
    messages = build_messages(Session(platform="linux"), tools=[])
    system_content = messages[0]["content"]
    assert "Available tools" not in system_content
    assert '"function"' not in system_content


def test_none_tools_omits_tool_block() -> None:
    messages = build_messages(Session(platform="linux"), tools=None)
    assert "Available tools" not in messages[0]["content"]


def test_unknown_platform_when_blank() -> None:
    messages = build_messages(Session(platform=""))
    assert "unknown" in messages[0]["content"].lower()


def test_history_passed_through_unchanged() -> None:
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "tool", "content": "tool result"},
    ]
    session = Session(platform="linux", history=history)
    messages = build_messages(session)
    assert messages[1:] == history


def test_persona_forbids_emojis() -> None:
    messages = build_messages(Session(platform="linux"))
    assert "emoji" in messages[0]["content"].lower()


def test_build_tools_returns_ollama_format() -> None:
    tools = [_SearchTool(), _ReadFileTool()]
    decls = build_tools(tools)
    assert decls == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web.",
                "parameters": _SearchTool.schema,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from disk.",
                "parameters": _ReadFileTool.schema,
            },
        },
    ]


def test_build_tools_none_when_empty_list() -> None:
    assert build_tools([]) is None


def test_build_tools_none_when_none() -> None:
    assert build_tools(None) is None
