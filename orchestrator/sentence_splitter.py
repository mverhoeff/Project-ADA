"""Sentence splitter for the LLM → TTS streaming pipeline.

A pure synchronous finite-state machine. It accepts tokens from the LLM
stream one at a time and returns complete sentences as soon as their
boundaries are unambiguous. No I/O, no GPU, no async — designed for
exhaustive unit testing.

Flush triggers:
    * ``.`` ``!`` ``?`` followed by whitespace (or via :meth:`flush`)
    * a newline ``\\n``
    * the internal buffer exceeds ``max_chars`` (safety valve)

Held back (no flush):
    * decimals / version numbers (``.`` followed by a digit)
    * common abbreviations (``Dr.``, ``e.g.``, ``U.S.A.``)
    * ellipses (``...``)
"""

from __future__ import annotations

# Words (lowercased, with internal dots preserved) that should NOT trigger a
# sentence flush when followed by ``.`` + whitespace.
_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "u.s",
        "u.s.a",
        "u.k",
    }
)


class SentenceSplitter:
    """Token-stream → complete-sentence finite-state machine.

    The splitter is stateful. Feed tokens in order via :meth:`feed`, then
    call :meth:`flush` once the upstream stream ends to drain any remainder.

    Args:
        max_chars: Maximum buffer length before the safety-valve flush
            triggers. Defaults to 200.

    Raises:
        ValueError: If ``max_chars`` is not positive.
    """

    _NORMAL = 0
    _AFTER_DOT = 1
    _AFTER_DOT_DOT = 2
    _AFTER_ELLIPSIS = 3
    _AFTER_BANG = 4

    def __init__(self, max_chars: int = 200) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self._max_chars = max_chars
        self._buffer: str = ""
        self._state: int = self._NORMAL

    def feed(self, token: str) -> list[str]:
        """Feed one token into the splitter.

        Args:
            token: A chunk of text from the LLM stream. May be a single
                character, a word, or longer.

        Returns:
            Complete sentences ready for TTS, in order. May be empty.
        """
        out: list[str] = []
        for ch in token:
            self._buffer += ch
            self._step(ch, out)
            if len(self._buffer) > self._max_chars:
                trimmed = self._buffer.strip()
                if trimmed:
                    out.append(trimmed)
                self._buffer = ""
                self._state = self._NORMAL
        return out

    def flush(self) -> list[str]:
        """Drain any buffered text. Call this at end-of-stream.

        Returns:
            Up to one sentence containing whatever remained in the buffer,
            stripped of surrounding whitespace. Empty if the buffer is empty.
        """
        trimmed = self._buffer.strip()
        self._buffer = ""
        self._state = self._NORMAL
        return [trimmed] if trimmed else []

    def _step(self, ch: str, out: list[str]) -> None:
        if ch == "\n":
            sentence = self._buffer.strip()
            self._buffer = ""
            self._state = self._NORMAL
            if sentence:
                out.append(sentence)
            return

        if self._state == self._NORMAL:
            if ch == ".":
                self._state = self._AFTER_DOT
            elif ch in ("!", "?"):
                self._state = self._AFTER_BANG
            return

        if self._state == self._AFTER_DOT:
            if ch == ".":
                self._state = self._AFTER_DOT_DOT
            elif ch.isdigit():
                self._state = self._NORMAL
            elif ch.isspace():
                if self._is_abbreviation_at_end():
                    self._state = self._NORMAL
                else:
                    sentence = self._buffer[:-1].strip()
                    self._buffer = ""
                    self._state = self._NORMAL
                    if sentence:
                        out.append(sentence)
            else:
                self._state = self._NORMAL
            return

        if self._state == self._AFTER_DOT_DOT:
            self._state = self._AFTER_ELLIPSIS if ch == "." else self._NORMAL
            return

        if self._state == self._AFTER_ELLIPSIS:
            if ch == ".":
                self._state = self._AFTER_DOT
            elif ch in ("!", "?"):
                self._state = self._AFTER_BANG
            else:
                self._state = self._NORMAL
            return

        if self._state == self._AFTER_BANG:
            if ch.isspace():
                sentence = self._buffer[:-1].strip()
                self._buffer = ""
                self._state = self._NORMAL
                if sentence:
                    out.append(sentence)
            elif ch == ".":
                self._state = self._AFTER_DOT
            elif ch in ("!", "?"):
                self._state = self._AFTER_BANG
            else:
                self._state = self._NORMAL

    def _is_abbreviation_at_end(self) -> bool:
        """Return True if the buffer ends with a known abbreviation + period.

        Called only from the AFTER_DOT branch when the just-appended
        character is whitespace, so the buffer ends with ``"<word>.<ws>"``.
        """
        if len(self._buffer) < 2:
            return False
        before_dot = self._buffer[:-2]
        i = len(before_dot)
        while i > 0 and (before_dot[i - 1].isalnum() or before_dot[i - 1] == "."):
            i -= 1
        word = before_dot[i:].lower().rstrip(".")
        return word in _ABBREVIATIONS
