"""Pure-logic audio utilities for the TTS service.

Three responsibilities, all pure (no I/O, no model dependency):

1. ``preprocess_text``: strip markdown formatting and characters that the LLM
   may emit but should not be spoken aloud.
2. ``samples_to_pcm16``: convert Kokoro's float32 audio output to little-endian
   16-bit PCM bytes for transport over HTTP chunked streams.
3. ``wav_streaming_header``: build a 44-byte WAV RIFF header with the data
   sub-chunk length set to ``0xFFFFFFFF``, signalling streaming/unknown length.

Number-to-word expansion is intentionally **not** implemented here — Kokoro's
internal espeak phoneme converter handles digits natively, so re-doing it in
Python would only introduce regressions on edge cases (years, decimals, etc.).
"""

from __future__ import annotations

import re
import struct

import numpy as np

_TRIPLE_BACKTICK_BLOCK = re.compile(r"```.*?```", flags=re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_BOLD_DOUBLE_STAR = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)
_BOLD_DOUBLE_UNDERSCORE = re.compile(r"__(.+?)__", flags=re.DOTALL)
_ITALIC_SINGLE_STAR = re.compile(r"\*(.+?)\*", flags=re.DOTALL)
_ITALIC_SINGLE_UNDERSCORE = re.compile(r"(?<!\w)_(.+?)_(?!\w)", flags=re.DOTALL)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", flags=re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+", flags=re.MULTILINE)
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_WHITESPACE = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    """Strip markdown and non-spoken characters from ``text``.

    Args:
        text: Raw LLM output (one sentence at a time, in this service's flow).

    Returns:
        A whitespace-collapsed plain-text string ready for the synthesizer.
    """
    text = _TRIPLE_BACKTICK_BLOCK.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _BOLD_DOUBLE_STAR.sub(r"\1", text)
    text = _BOLD_DOUBLE_UNDERSCORE.sub(r"\1", text)
    text = _ITALIC_SINGLE_STAR.sub(r"\1", text)
    text = _ITALIC_SINGLE_UNDERSCORE.sub(r"\1", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = text.replace("&", " and ").replace("<", " ").replace(">", " ")
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def samples_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert float32 audio in ``[-1, 1]`` to little-endian 16-bit PCM bytes.

    Values outside ``[-1, 1]`` are clipped. The conversion uses ``32767`` as
    the scale factor so a sample of ``1.0`` maps to the maximum positive
    int16 value.

    Args:
        samples: Mono audio samples as a 1-D numpy array.

    Returns:
        Raw int16 little-endian PCM bytes.
    """
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    return pcm.tobytes()


def wav_streaming_header(
    sample_rate: int,
    n_channels: int = 1,
    bit_depth: int = 16,
) -> bytes:
    """Build a 44-byte WAV RIFF header for streaming audio of unknown length.

    The RIFF chunk size and the ``data`` sub-chunk size are both set to
    ``0xFFFFFFFF``. Standard WAV decoders that respect the size fields will
    treat this as an unknown-length stream and read until EOF.

    Args:
        sample_rate: Output sample rate in Hz (Kokoro: 24 000).
        n_channels: Number of channels (default mono).
        bit_depth: Bits per sample (default 16).

    Returns:
        Exactly 44 bytes forming a valid WAV header.
    """
    byte_rate = sample_rate * n_channels * bit_depth // 8
    block_align = n_channels * bit_depth // 8
    streaming_size = 0xFFFFFFFF

    header = b"RIFF"
    header += struct.pack("<I", streaming_size)
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<I", 16)  # fmt sub-chunk size (PCM)
    header += struct.pack("<H", 1)  # audio format = PCM
    header += struct.pack("<H", n_channels)
    header += struct.pack("<I", sample_rate)
    header += struct.pack("<I", byte_rate)
    header += struct.pack("<H", block_align)
    header += struct.pack("<H", bit_depth)
    header += b"data"
    header += struct.pack("<I", streaming_size)
    return header
