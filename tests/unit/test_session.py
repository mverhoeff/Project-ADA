"""Unit tests for :mod:`orchestrator.session`."""

from __future__ import annotations

import pytest

from orchestrator.session import Session


def test_initial_state() -> None:
    s = Session()
    assert s.history == []
    assert s.turn_count == 0
    assert s.active_tool is None
    assert s.is_speaking is False
    assert s.platform == ""


def test_add_message_appends() -> None:
    s = Session()
    s.add_message("user", "hello")
    s.add_message("assistant", "hi there")
    assert s.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_add_message_does_not_increment_turn_count() -> None:
    s = Session()
    s.add_message("user", "hi")
    s.add_message("assistant", "hello")
    assert s.turn_count == 0


def test_trim_noop_when_under_limit() -> None:
    s = Session(
        history=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
    )
    s.trim_to(max_turns=5)
    assert len(s.history) == 3


def test_trim_drops_oldest_non_system() -> None:
    s = Session(
        history=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]
    )
    s.trim_to(max_turns=2)
    assert s.history == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]


def test_trim_preserves_system_at_zero_turns() -> None:
    s = Session(
        history=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
    )
    s.trim_to(max_turns=0)
    assert s.history == [{"role": "system", "content": "sys"}]


def test_trim_with_no_system_message() -> None:
    s = Session(
        history=[
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
    )
    s.trim_to(max_turns=1)
    assert s.history == [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]


def test_trim_negative_raises() -> None:
    s = Session()
    with pytest.raises(ValueError):
        s.trim_to(-1)


def test_independent_sessions_have_separate_history() -> None:
    a = Session()
    b = Session()
    a.add_message("user", "hi")
    assert b.history == []
