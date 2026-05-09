"""Unit tests for :mod:`agent.router`."""

from __future__ import annotations

from agent.router import detect_tool_call


def test_returns_none_for_plain_text() -> None:
    assert detect_tool_call("hello world, no tool here") is None


def test_returns_none_for_empty_string() -> None:
    assert detect_tool_call("") is None


def test_detects_minimal_tool_call() -> None:
    text = '<tool_call>{"name": "foo", "arguments": {}}</tool_call>'
    assert detect_tool_call(text) == {"name": "foo", "arguments": {}}


def test_detects_tool_call_with_complex_arguments() -> None:
    text = (
        '<tool_call>{"name": "run_shell", "arguments": '
        '{"command": "ls -la", "timeout_seconds": 5}}</tool_call>'
    )
    result = detect_tool_call(text)
    assert result == {
        "name": "run_shell",
        "arguments": {"command": "ls -la", "timeout_seconds": 5},
    }


def test_ignores_text_before_tag() -> None:
    text = (
        "Searching the web now...\n"
        '<tool_call>{"name": "browser_search", "arguments": {"query": "ada"}}</tool_call>'
    )
    assert detect_tool_call(text) == {
        "name": "browser_search",
        "arguments": {"query": "ada"},
    }


def test_returns_none_for_malformed_json() -> None:
    text = "<tool_call>not json at all</tool_call>"
    assert detect_tool_call(text) is None


def test_returns_none_for_partial_tag() -> None:
    text = '<tool_call>{"name": "foo"'
    assert detect_tool_call(text) is None


def test_returns_none_when_name_key_missing() -> None:
    text = '<tool_call>{"arguments": {"x": 1}}</tool_call>'
    assert detect_tool_call(text) is None


def test_returns_none_when_payload_is_not_object() -> None:
    text = '<tool_call>"just a string"</tool_call>'
    assert detect_tool_call(text) is None


def test_multiline_json_body() -> None:
    text = (
        "<tool_call>{\n"
        '  "name": "read_file",\n'
        '  "arguments": {"path": "~/Documents/x.txt"}\n'
        "}</tool_call>"
    )
    assert detect_tool_call(text) == {
        "name": "read_file",
        "arguments": {"path": "~/Documents/x.txt"},
    }


def test_returns_first_match_when_multiple_tags() -> None:
    text = (
        '<tool_call>{"name": "first", "arguments": {}}</tool_call>'
        " and then "
        '<tool_call>{"name": "second", "arguments": {}}</tool_call>'
    )
    assert detect_tool_call(text) == {"name": "first", "arguments": {}}


def test_arguments_optional() -> None:
    text = '<tool_call>{"name": "ping"}</tool_call>'
    assert detect_tool_call(text) == {"name": "ping"}
