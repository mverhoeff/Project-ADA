"""Streaming audio playback for the TTS chunk stream.

Consumes an ``AsyncIterator[bytes]`` whose first 44 bytes are a streaming
WAV header (see :func:`services.tts.audio_utils.wav_streaming_header`)
and whose remaining bytes are raw little-endian int16 PCM. Opens a
``sounddevice.RawOutputStream`` matching the header, writes every PCM
chunk as it arrives, and closes the stream on completion or
cancellation.

The ``sounddevice`` import is deferred to the default sink factory so
unit tests can inject a fake sink without PortAudio installed.
"""

from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager, asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from core.exceptions import ServiceUnavailableError


_USER_FACING_ERROR = "I can't speak right now."
_HEADER_SIZE = 44
_PCM_FORMAT_CODE = 1


@dataclass(frozen=True)
class WavFormat:
    """Parsed audio format from a WAV header."""

    sample_rate: int
    n_channels: int
    bit_depth: int


def parse_wav_header(header: bytes) -> WavFormat:
    """Parse a 44-byte canonical PCM WAV header.

    The RIFF and ``data`` size fields are intentionally **not** validated
    — the streaming sentinel ``0xFFFFFFFF`` produced by
    :func:`services.tts.audio_utils.wav_streaming_header` is accepted
    transparently alongside finite sizes.

    Args:
        header: Bytes whose first 44 are a canonical PCM WAV header.

    Returns:
        A :class:`WavFormat` with the sample rate, channel count, and bit
        depth read from the ``fmt `` sub-chunk.

    Raises:
        ValueError: On any structural mismatch (wrong magic, fmt sub-chunk
            size, non-PCM format code, missing ``data`` marker) or if
            ``header`` is shorter than 44 bytes.
    """
    if len(header) < _HEADER_SIZE:
        raise ValueError(f"WAV header must be {_HEADER_SIZE} bytes, got {len(header)}")
    if header[0:4] != b"RIFF":
        raise ValueError("Missing RIFF magic")
    if header[8:12] != b"WAVE":
        raise ValueError("Missing WAVE magic")
    if header[12:16] != b"fmt ":
        raise ValueError("Missing fmt sub-chunk")
    fmt_size = struct.unpack("<I", header[16:20])[0]
    if fmt_size != 16:
        raise ValueError(f"Unsupported fmt sub-chunk size: {fmt_size}")
    audio_format = struct.unpack("<H", header[20:22])[0]
    if audio_format != _PCM_FORMAT_CODE:
        raise ValueError(f"Non-PCM audio format: {audio_format}")
    n_channels = struct.unpack("<H", header[22:24])[0]
    sample_rate = struct.unpack("<I", header[24:28])[0]
    bit_depth = struct.unpack("<H", header[34:36])[0]
    if header[36:40] != b"data":
        raise ValueError("Missing data sub-chunk")
    return WavFormat(
        sample_rate=sample_rate,
        n_channels=n_channels,
        bit_depth=bit_depth,
    )


class _OutputSink(Protocol):
    """Minimal interface :class:`AudioPlayer` needs from an output stream."""

    def write(self, pcm: bytes) -> None: ...
    def close(self) -> None: ...
    def abort(self) -> None: ...


SinkFactory = Callable[[WavFormat], AbstractContextManager[_OutputSink]]


class _PlayerSession:
    """One turn's worth of playback through a single persistent sink.

    Each :meth:`play` call drains one sentence's streaming-WAV chunk
    iterator. The first call that yields a parseable header opens the
    sink; every call (including the first) discards its own leading
    44-byte WAV header. The PCM alignment buffer persists across calls so
    a trailing odd byte from one sentence carries cleanly into the next.
    """

    def __init__(self, sink_factory: SinkFactory) -> None:
        self._sink_factory = sink_factory
        self._sink_cm: AbstractContextManager[_OutputSink] | None = None
        self._sink: _OutputSink | None = None
        self._pcm_buf = bytearray()

    async def play(self, chunk_iter: AsyncIterator[bytes]) -> None:
        """Drain one sentence's streaming-WAV chunk iterator into the sink.

        The leading 44 bytes are buffered and skipped; on the first
        sentence of the session they are parsed to open the sink. A PCM
        accumulation buffer ensures every ``sink.write()`` receives a
        whole number of samples (HTTP chunk boundaries may fall
        mid-sample, including across the sentence boundary).

        Raises:
            ServiceUnavailableError: If the first sentence's WAV header is
                malformed.
        """
        header_buf = bytearray()
        header_skipped = False
        try:
            async for chunk in chunk_iter:
                if not chunk:
                    continue
                if not header_skipped:
                    header_buf.extend(chunk)
                    if len(header_buf) < _HEADER_SIZE:
                        continue
                    if self._sink is None:
                        try:
                            wav_format = parse_wav_header(
                                bytes(header_buf[:_HEADER_SIZE])
                            )
                        except ValueError as e:
                            raise ServiceUnavailableError(
                                f"TTS produced malformed WAV header: {e}",
                                _USER_FACING_ERROR,
                            ) from e
                        self._sink_cm = self._sink_factory(wav_format)
                        self._sink = self._sink_cm.__enter__()
                    self._pcm_buf.extend(header_buf[_HEADER_SIZE:])
                    header_skipped = True
                else:
                    self._pcm_buf.extend(chunk)
                # Write only complete samples (2 bytes each for int16).
                aligned = len(self._pcm_buf) & ~1
                if aligned:
                    self._sink.write(bytes(self._pcm_buf[:aligned]))
                    del self._pcm_buf[:aligned]
        except asyncio.CancelledError:
            if self._sink is not None:
                self._sink.abort()
            raise

    def close(self) -> None:
        """Close the underlying sink. Idempotent; safe if no sink opened."""
        if self._sink_cm is not None:
            self._sink_cm.__exit__(None, None, None)
            self._sink_cm = None
            self._sink = None


class AudioPlayer:
    """Opens turn-scoped playback sessions over a streaming WAV chunk source.

    A :meth:`session` keeps one ``sounddevice`` output stream open for the
    whole turn, so consecutive sentences play back-to-back without the
    device restart gap of a per-sentence stream.

    Args:
        config: Loaded config dict; the ``audio.output_device`` key is
            forwarded to the default sink factory.
        sink_factory: Optional context-manager factory invoked once per
            session with the parsed :class:`WavFormat`. Defaults to a
            ``sounddevice``-backed sink. Tests inject a fake here.
    """

    def __init__(
        self,
        config: dict[str, Any],
        sink_factory: SinkFactory | None = None,
    ) -> None:
        self._config = config
        self._sink_factory = sink_factory or _make_sounddevice_sink_factory(
            device=config.get("audio", {}).get("output_device"),
        )

    @asynccontextmanager
    async def session(self) -> AsyncIterator[_PlayerSession]:
        """Yield a playback session that keeps one sink open for the turn.

        The sink is opened lazily on the first
        :meth:`_PlayerSession.play` call that produces a parseable
        header, and closed when the context exits — including on
        cancellation, after the in-flight ``play`` has aborted the sink.
        """
        sess = _PlayerSession(self._sink_factory)
        try:
            yield sess
        finally:
            sess.close()


def _make_sounddevice_sink_factory(device: int | str | None) -> SinkFactory:
    """Build a sink factory backed by ``sounddevice.RawOutputStream``.

    ``sounddevice`` is imported lazily so this module remains importable
    in environments without PortAudio.
    """

    @contextmanager
    def factory(fmt: WavFormat) -> Any:
        import sounddevice as sd  # local import — see module docstring

        if fmt.bit_depth != 16:
            raise ValueError(
                f"AudioPlayer only supports 16-bit PCM, got {fmt.bit_depth}"
            )
        stream = sd.RawOutputStream(
            samplerate=fmt.sample_rate,
            channels=fmt.n_channels,
            dtype="int16",
            device=device,
        )
        stream.start()
        try:
            yield stream
        finally:
            stream.stop()
            stream.close()

    return factory
