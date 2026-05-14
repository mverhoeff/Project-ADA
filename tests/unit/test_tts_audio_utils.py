"""Unit tests for :mod:`services.tts.audio_utils`."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from services.tts.audio_utils import (
    preprocess_text,
    samples_to_pcm16,
    wav_streaming_header,
)


# -- preprocess_text ---------------------------------------------------------


def test_preprocess_strips_triple_backtick_code_block() -> None:
    text = "Try this: ```python\nprint('hi')\n``` then continue."
    out = preprocess_text(text)
    assert "print" not in out
    assert "Try this:" in out
    assert "then continue." in out


def test_preprocess_strips_inline_code() -> None:
    assert preprocess_text("Run `npm install` first") == "Run npm install first"


def test_preprocess_strips_bold_double_star() -> None:
    assert preprocess_text("This is **important** stuff") == "This is important stuff"


def test_preprocess_strips_bold_double_underscore() -> None:
    assert preprocess_text("Use __caution__ here") == "Use caution here"


def test_preprocess_strips_italic_single_star() -> None:
    assert preprocess_text("Be *very* careful") == "Be very careful"


def test_preprocess_strips_italic_single_underscore() -> None:
    assert preprocess_text("That is _odd_ indeed") == "That is odd indeed"


def test_preprocess_does_not_strip_underscores_inside_words() -> None:
    # snake_case identifiers must survive intact
    assert preprocess_text("variable_name is set") == "variable_name is set"


def test_preprocess_strips_heading() -> None:
    assert preprocess_text("## Section Title") == "Section Title"


def test_preprocess_strips_bullet_dash() -> None:
    assert preprocess_text("- first item") == "first item"


def test_preprocess_strips_bullet_star() -> None:
    assert preprocess_text("* second item") == "second item"


def test_preprocess_strips_numbered_bullet() -> None:
    assert preprocess_text("1. ordered item") == "ordered item"


def test_preprocess_joins_multiline_bullets_with_commas() -> None:
    assert preprocess_text("- apple\n- banana\n- cherry") == "apple, banana, cherry"


def test_preprocess_joins_numbered_multiline_bullets_with_commas() -> None:
    assert preprocess_text("1. apple\n2. banana\n3. cherry") == "apple, banana, cherry"


def test_preprocess_strips_markdown_link() -> None:
    assert (
        preprocess_text("See [the docs](https://example.com) for more")
        == "See the docs for more"
    )


def test_preprocess_collapses_whitespace() -> None:
    assert preprocess_text("Hello   world\n\n\nfoo") == "Hello world foo"


def test_preprocess_handles_ampersand() -> None:
    assert preprocess_text("Tom & Jerry") == "Tom and Jerry"


def test_preprocess_strips_angle_brackets() -> None:
    assert preprocess_text("Press <enter> now") == "Press enter now"


def test_preprocess_empty_input() -> None:
    assert preprocess_text("") == ""
    assert preprocess_text("   \n  ") == ""


# -- samples_to_pcm16 --------------------------------------------------------


def test_samples_to_pcm16_zero_sample() -> None:
    out = samples_to_pcm16(np.array([0.0], dtype=np.float32))
    assert out == struct.pack("<h", 0)


def test_samples_to_pcm16_full_positive() -> None:
    out = samples_to_pcm16(np.array([1.0], dtype=np.float32))
    assert struct.unpack("<h", out)[0] == 32767


def test_samples_to_pcm16_full_negative() -> None:
    out = samples_to_pcm16(np.array([-1.0], dtype=np.float32))
    assert struct.unpack("<h", out)[0] == -32767


def test_samples_to_pcm16_clips_above_one() -> None:
    out = samples_to_pcm16(np.array([1.5, 2.0], dtype=np.float32))
    values = struct.unpack("<2h", out)
    assert values == (32767, 32767)


def test_samples_to_pcm16_clips_below_neg_one() -> None:
    out = samples_to_pcm16(np.array([-1.5, -2.0], dtype=np.float32))
    values = struct.unpack("<2h", out)
    assert values == (-32767, -32767)


def test_samples_to_pcm16_byte_count_matches_input() -> None:
    samples = np.zeros(480, dtype=np.float32)
    assert len(samples_to_pcm16(samples)) == 480 * 2


def test_samples_to_pcm16_is_little_endian() -> None:
    # Value 256 in int16 LE = b"\x00\x01"; value 1 = b"\x01\x00"
    out = samples_to_pcm16(np.array([1.0 / 32767, 256.0 / 32767], dtype=np.float32))
    assert out[:2] == b"\x01\x00"
    assert out[2:4] == b"\x00\x01"


# -- wav_streaming_header ----------------------------------------------------


def test_wav_streaming_header_length_is_44() -> None:
    assert len(wav_streaming_header(24_000)) == 44


def test_wav_streaming_header_riff_magic() -> None:
    header = wav_streaming_header(24_000)
    assert header[0:4] == b"RIFF"
    assert header[8:12] == b"WAVE"
    assert header[12:16] == b"fmt "
    assert header[36:40] == b"data"


def test_wav_streaming_header_riff_size_is_max() -> None:
    header = wav_streaming_header(24_000)
    assert struct.unpack("<I", header[4:8])[0] == 0xFFFFFFFF


def test_wav_streaming_header_data_size_is_max() -> None:
    header = wav_streaming_header(24_000)
    assert struct.unpack("<I", header[40:44])[0] == 0xFFFFFFFF


def test_wav_streaming_header_sample_rate_field() -> None:
    header = wav_streaming_header(24_000)
    assert struct.unpack("<I", header[24:28])[0] == 24_000


def test_wav_streaming_header_channels_and_bits() -> None:
    header = wav_streaming_header(24_000, n_channels=1, bit_depth=16)
    assert struct.unpack("<H", header[22:24])[0] == 1  # channels
    assert struct.unpack("<H", header[34:36])[0] == 16  # bits per sample


def test_wav_streaming_header_byte_rate_and_block_align() -> None:
    header = wav_streaming_header(24_000, n_channels=1, bit_depth=16)
    byte_rate = struct.unpack("<I", header[28:32])[0]
    block_align = struct.unpack("<H", header[32:34])[0]
    assert byte_rate == 24_000 * 1 * 16 // 8
    assert block_align == 1 * 16 // 8


def test_wav_streaming_header_pcm_format_code() -> None:
    header = wav_streaming_header(24_000)
    # bytes 16-19: fmt sub-chunk size = 16; bytes 20-21: format code = 1 (PCM)
    assert struct.unpack("<I", header[16:20])[0] == 16
    assert struct.unpack("<H", header[20:22])[0] == 1


@pytest.mark.parametrize("sample_rate", [16_000, 22_050, 24_000, 48_000])
def test_wav_streaming_header_various_sample_rates(sample_rate: int) -> None:
    header = wav_streaming_header(sample_rate)
    assert struct.unpack("<I", header[24:28])[0] == sample_rate
