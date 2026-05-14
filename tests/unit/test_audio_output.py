"""Unit tests for :mod:`orchestrator.audio_output`."""

from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from core.exceptions import ServiceUnavailableError
from orchestrator.audio_output import (
    AudioPlayer,
    WavFormat,
    parse_wav_header,
)
from services.tts.audio_utils import wav_streaming_header


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class FakeSink:
    """Records every interaction so tests can assert on it."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False
        self.aborted = False
        self.exited = False

    def write(self, pcm: bytes) -> None:
        self.writes.append(pcm)

    def close(self) -> None:
        self.closed = True

    def abort(self) -> None:
        self.aborted = True


class FakeFactory:
    """Records the WavFormat it was called with and yields a FakeSink."""

    def __init__(self) -> None:
        self.calls: list[WavFormat] = []
        self.sink: FakeSink | None = None

    def __call__(self, fmt: WavFormat) -> Any:
        self.calls.append(fmt)
        sink = FakeSink()
        self.sink = sink
        factory_self = self

        @contextmanager
        def cm() -> Iterator[FakeSink]:
            try:
                yield sink
            finally:
                assert factory_self.sink is not None
                factory_self.sink.exited = True

        return cm()


async def async_iter(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for c in chunks:
        yield c


def _finite_wav_header(
    sample_rate: int = 24_000,
    n_channels: int = 1,
    bit_depth: int = 16,
    data_size: int = 1024,
) -> bytes:
    byte_rate = sample_rate * n_channels * bit_depth // 8
    block_align = n_channels * bit_depth // 8
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<H", 1)
        + struct.pack("<H", n_channels)
        + struct.pack("<I", sample_rate)
        + struct.pack("<I", byte_rate)
        + struct.pack("<H", block_align)
        + struct.pack("<H", bit_depth)
        + b"data"
        + struct.pack("<I", data_size)
    )


# ---------------------------------------------------------------------------
# parse_wav_header
# ---------------------------------------------------------------------------


def test_parse_streaming_header_from_tts_audio_utils() -> None:
    header = wav_streaming_header(24_000)
    fmt = parse_wav_header(header)
    assert fmt == WavFormat(sample_rate=24_000, n_channels=1, bit_depth=16)


def test_parse_finite_wav_header() -> None:
    header = _finite_wav_header(sample_rate=24_000, data_size=1024)
    fmt = parse_wav_header(header)
    assert fmt == WavFormat(sample_rate=24_000, n_channels=1, bit_depth=16)


@pytest.mark.parametrize("sample_rate", [16_000, 22_050, 24_000, 48_000])
def test_parse_supports_various_sample_rates(sample_rate: int) -> None:
    fmt = parse_wav_header(wav_streaming_header(sample_rate))
    assert fmt.sample_rate == sample_rate


def test_parse_rejects_wrong_riff_magic() -> None:
    header = bytearray(wav_streaming_header(24_000))
    header[0:4] = b"XXXX"
    with pytest.raises(ValueError, match="RIFF"):
        parse_wav_header(bytes(header))


def test_parse_rejects_wrong_wave_magic() -> None:
    header = bytearray(wav_streaming_header(24_000))
    header[8:12] = b"XXXX"
    with pytest.raises(ValueError, match="WAVE"):
        parse_wav_header(bytes(header))


def test_parse_rejects_non_pcm_format() -> None:
    header = bytearray(wav_streaming_header(24_000))
    struct.pack_into("<H", header, 20, 3)  # IEEE float
    with pytest.raises(ValueError, match="Non-PCM"):
        parse_wav_header(bytes(header))


def test_parse_rejects_short_buffer() -> None:
    with pytest.raises(ValueError, match="44 bytes"):
        parse_wav_header(b"RIFF" + b"\x00" * 30)


def test_parse_rejects_missing_data_marker() -> None:
    header = bytearray(wav_streaming_header(24_000))
    header[36:40] = b"XXXX"
    with pytest.raises(ValueError, match="data"):
        parse_wav_header(bytes(header))


# ---------------------------------------------------------------------------
# AudioPlayer.session
# ---------------------------------------------------------------------------


def _config() -> dict[str, Any]:
    return {"audio": {"output_device": None}}


async def _play_session(player: AudioPlayer, *sentences: list[bytes]) -> None:
    """Open one session and play each sentence's chunk list through it."""
    async with player.session() as sess:
        for chunks in sentences:
            await sess.play(async_iter(chunks))


@pytest.mark.asyncio
async def test_play_writes_pcm_after_header_chunk() -> None:
    factory = FakeFactory()
    player = AudioPlayer(_config(), sink_factory=factory)

    pcm = b"\x00\x01" * 100
    await _play_session(player, [wav_streaming_header(24_000), pcm])

    assert factory.sink is not None
    assert b"".join(factory.sink.writes) == pcm
    assert factory.sink.exited is True


@pytest.mark.asyncio
async def test_play_handles_header_split_across_chunks() -> None:
    factory = FakeFactory()
    player = AudioPlayer(_config(), sink_factory=factory)

    header = wav_streaming_header(24_000)
    pcm = b"\xAB\xCD" * 50
    await _play_session(player, [header[:20], header[20:] + pcm])

    assert factory.sink is not None
    assert b"".join(factory.sink.writes) == pcm


@pytest.mark.asyncio
async def test_play_handles_pcm_concatenated_with_header_in_first_chunk() -> None:
    factory = FakeFactory()
    player = AudioPlayer(_config(), sink_factory=factory)

    header = wav_streaming_header(24_000)
    pcm = b"\x10\x20" * 25
    await _play_session(player, [header + pcm])

    assert factory.sink is not None
    assert b"".join(factory.sink.writes) == pcm


@pytest.mark.asyncio
async def test_play_passes_correct_wav_format_to_sink_factory() -> None:
    factory = FakeFactory()
    player = AudioPlayer(_config(), sink_factory=factory)

    await _play_session(player, [wav_streaming_header(24_000), b"\x00\x00"])

    assert factory.calls == [WavFormat(sample_rate=24_000, n_channels=1, bit_depth=16)]


@pytest.mark.asyncio
async def test_session_exits_sink_context_on_normal_completion() -> None:
    factory = FakeFactory()
    player = AudioPlayer(_config(), sink_factory=factory)

    async with player.session() as sess:
        await sess.play(async_iter([wav_streaming_header(24_000), b"\x00\x00"]))
        assert factory.sink is not None
        assert factory.sink.exited is False  # still open inside the session

    assert factory.sink.exited is True
    assert factory.sink.aborted is False


@pytest.mark.asyncio
async def test_session_reuses_sink_across_sentences() -> None:
    factory = FakeFactory()
    player = AudioPlayer(_config(), sink_factory=factory)

    pcm1 = b"\x01\x02" * 10
    pcm2 = b"\x03\x04" * 10
    await _play_session(
        player,
        [wav_streaming_header(24_000), pcm1],
        [wav_streaming_header(24_000), pcm2],
    )

    assert len(factory.calls) == 1  # sink opened exactly once for the turn
    assert factory.sink is not None
    # both 44-byte headers discarded; only PCM reaches the sink
    assert b"".join(factory.sink.writes) == pcm1 + pcm2


@pytest.mark.asyncio
async def test_session_carries_pcm_alignment_across_sentences() -> None:
    """A trailing odd byte from one sentence aligns against the next."""
    factory = FakeFactory()
    player = AudioPlayer(_config(), sink_factory=factory)

    # sentence 1: header + 3 PCM bytes -> 1 byte held back for alignment
    # sentence 2: header + 1 PCM byte -> joins the held byte into a sample
    await _play_session(
        player,
        [wav_streaming_header(24_000), b"\xAA\xBB\xCC"],
        [wav_streaming_header(24_000), b"\xDD"],
    )

    assert factory.sink is not None
    assert b"".join(factory.sink.writes) == b"\xAA\xBB\xCC\xDD"


@pytest.mark.asyncio
async def test_session_aborts_sink_on_cancellation() -> None:
    factory = FakeFactory()
    player = AudioPlayer(_config(), sink_factory=factory)

    started = asyncio.Event()
    block = asyncio.Event()

    async def slow_chunks() -> AsyncIterator[bytes]:
        yield wav_streaming_header(24_000)
        yield b"\x00\x00" * 10
        started.set()
        await block.wait()  # never completes — wait until cancelled
        yield b"unreachable"

    async def run() -> None:
        async with player.session() as sess:
            await sess.play(slow_chunks())

    task = asyncio.create_task(run())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert factory.sink is not None
    assert factory.sink.aborted is True
    assert factory.sink.exited is True


@pytest.mark.asyncio
async def test_play_raises_service_unavailable_on_malformed_header() -> None:
    factory = FakeFactory()
    player = AudioPlayer(_config(), sink_factory=factory)

    junk = b"junk" + b"\x00" * 40
    with pytest.raises(ServiceUnavailableError) as exc_info:
        await _play_session(player, [junk])

    assert "speak" in exc_info.value.user_message.lower()
    assert factory.sink is None  # sink never opened


@pytest.mark.asyncio
async def test_empty_session_opens_no_sink() -> None:
    factory = FakeFactory()
    player = AudioPlayer(_config(), sink_factory=factory)

    async with player.session():
        pass

    assert factory.sink is None  # nothing played, no sink, close is a no-op


@pytest.mark.asyncio
async def test_play_handles_empty_stream_gracefully() -> None:
    factory = FakeFactory()
    player = AudioPlayer(_config(), sink_factory=factory)

    await _play_session(player, [])

    assert factory.sink is None  # sink never opened, no exception raised


@pytest.mark.asyncio
async def test_play_skips_empty_chunks() -> None:
    factory = FakeFactory()
    player = AudioPlayer(_config(), sink_factory=factory)

    pcm = b"\x11\x22" * 10
    await _play_session(
        player, [b"", wav_streaming_header(24_000), b"", pcm, b""]
    )

    assert factory.sink is not None
    assert b"".join(factory.sink.writes) == pcm
