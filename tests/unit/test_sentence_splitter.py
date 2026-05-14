"""Unit tests for :mod:`orchestrator.sentence_splitter`."""

from __future__ import annotations

import pytest

from orchestrator.sentence_splitter import SentenceSplitter


def _feed_all(splitter: SentenceSplitter, text: str) -> list[str]:
    """Feed ``text`` and then flush, returning every emitted sentence."""
    out: list[str] = []
    out.extend(splitter.feed(text))
    out.extend(splitter.flush())
    return out


def test_init_rejects_zero_max_chars() -> None:
    with pytest.raises(ValueError):
        SentenceSplitter(max_chars=0)


def test_init_rejects_negative_max_chars() -> None:
    with pytest.raises(ValueError):
        SentenceSplitter(max_chars=-1)


def test_empty_input_emits_nothing() -> None:
    s = SentenceSplitter()
    assert s.feed("") == []
    assert s.flush() == []


def test_basic_period_with_trailing_space() -> None:
    s = SentenceSplitter()
    assert s.feed("Hello world. ") == ["Hello world."]
    assert s.flush() == []


def test_basic_period_via_flush() -> None:
    s = SentenceSplitter()
    assert s.feed("Hello world.") == []
    assert s.flush() == ["Hello world."]


def test_exclamation_flushes() -> None:
    s = SentenceSplitter()
    assert s.feed("Great! ") == ["Great!"]


def test_question_flushes() -> None:
    s = SentenceSplitter()
    assert s.feed("Really? ") == ["Really?"]


def test_multiple_sentences_in_one_feed() -> None:
    s = SentenceSplitter()
    assert s.feed("Yes! No. Maybe? ") == ["Yes!", "No.", "Maybe?"]


def test_token_by_token_matches_bulk_feed() -> None:
    text = "Hello world. How are you? Fine!"
    bulk = SentenceSplitter()
    bulk_out = _feed_all(bulk, text)

    tok = SentenceSplitter()
    tok_out: list[str] = []
    for ch in text:
        tok_out.extend(tok.feed(ch))
    tok_out.extend(tok.flush())

    assert bulk_out == tok_out
    assert bulk_out == ["Hello world.", "How are you?", "Fine!"]


def test_decimal_not_flushed() -> None:
    s = SentenceSplitter()
    assert _feed_all(s, "Pi is 3.14 today. ") == ["Pi is 3.14 today."]


def test_version_number_not_flushed() -> None:
    s = SentenceSplitter()
    assert _feed_all(s, "Python 3.11 is great. ") == ["Python 3.11 is great."]


@pytest.mark.parametrize(
    "abbrev",
    ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "etc.", "vs.", "e.g.", "i.e."],
)
def test_common_abbreviations_not_flushed(abbrev: str) -> None:
    s = SentenceSplitter()
    text = f"Meet {abbrev} Smith later. "
    assert _feed_all(s, text) == [f"Meet {abbrev} Smith later."]


def test_us_a_abbreviation_not_flushed() -> None:
    s = SentenceSplitter()
    assert _feed_all(s, "I live in U.S.A. now. ") == ["I live in U.S.A. now."]


def test_us_abbreviation_not_flushed() -> None:
    s = SentenceSplitter()
    assert _feed_all(s, "Visit U.S. soon. ") == ["Visit U.S. soon."]


def test_ellipsis_not_flushed() -> None:
    s = SentenceSplitter()
    assert _feed_all(s, "Wait... what is this? ") == ["Wait... what is this?"]


def test_ellipsis_followed_by_space_does_not_flush() -> None:
    s = SentenceSplitter()
    assert _feed_all(s, "Hmm... okay then. ") == ["Hmm... okay then."]


def test_bare_newline_does_not_fragment() -> None:
    s = SentenceSplitter()
    out = s.feed("foo\nbar")
    assert out == []
    assert s.flush() == ["foo\nbar"]


def test_newline_is_noop_in_normal_state() -> None:
    s = SentenceSplitter()
    assert s.feed("word\nword") == []


def test_newline_mid_sentence_without_terminator_holds() -> None:
    s = SentenceSplitter()
    assert s.feed("an unfinished line\n") == []
    assert s.flush() == ["an unfinished line"]


def test_list_items_not_fragmented() -> None:
    """Markdown list items stay in one chunk so TTS does not pause per item."""
    s = SentenceSplitter()
    out = s.feed("- one\n- two\n- three")
    assert out == []
    assert s.flush() == ["- one\n- two\n- three"]


def test_newline_after_terminator_emits_with_terminator() -> None:
    s = SentenceSplitter()
    assert s.feed("Hi.\n") == ["Hi."]


def test_period_newline_still_splits() -> None:
    s = SentenceSplitter()
    assert _feed_all(s, "First.\nSecond.\n") == ["First.", "Second."]


def test_consecutive_newlines_do_not_emit_empty() -> None:
    s = SentenceSplitter()
    out = s.feed("Hi.\n\nBye.")
    assert out == ["Hi."]
    assert s.flush() == ["Bye."]


def test_max_chars_safety_valve() -> None:
    s = SentenceSplitter(max_chars=20)
    long_token = "a" * 25
    out = s.feed(long_token)
    assert out == ["a" * 21]
    assert s.flush() == ["a" * 4]


def test_max_chars_does_not_trigger_at_exact_limit() -> None:
    s = SentenceSplitter(max_chars=10)
    assert s.feed("a" * 10) == []
    assert s.flush() == ["a" * 10]


def test_buffer_resets_between_sentences() -> None:
    s = SentenceSplitter()
    assert s.feed("First. ") == ["First."]
    assert s.feed("Second. ") == ["Second."]
    assert s.feed("Third.") == []
    assert s.flush() == ["Third."]


def test_flush_clears_state() -> None:
    s = SentenceSplitter()
    s.feed("Hello world.")
    assert s.flush() == ["Hello world."]
    assert s.flush() == []
    assert s.feed("Next. ") == ["Next."]


def test_leading_whitespace_stripped() -> None:
    s = SentenceSplitter()
    assert s.feed("   Hi. ") == ["Hi."]


def test_tab_after_period_flushes() -> None:
    s = SentenceSplitter()
    assert s.feed("Hello.\tWorld.") == ["Hello."]
    assert s.flush() == ["World."]


def test_tool_call_acknowledgment_pattern() -> None:
    """LLM emits a spoken phrase before tool-use JSON.

    The splitter must flush the spoken phrase to TTS before the JSON arrives.
    """
    s = SentenceSplitter()
    out = s.feed('Searching the web now. {"tool":"search"}')
    assert out == ["Searching the web now."]


def test_sentence_terminator_without_following_space_holds() -> None:
    s = SentenceSplitter()
    assert s.feed("Hi!Bye?") == []
    assert s.flush() == ["Hi!Bye?"]


def test_period_then_digit_then_more_text() -> None:
    s = SentenceSplitter()
    assert _feed_all(s, "It costs 9.99 dollars. ") == ["It costs 9.99 dollars."]
