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
from contextlib import AbstractContextManager, contextmanager
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


class AudioPlayer:
    """Plays a streaming WAV chunk iterator to a sound sink.

    Args:
        config: Loaded config dict; the ``audio.output_device`` key is
            forwarded to the default sink factory.
        sink_factory: Optional context-manager factory invoked once with
            the parsed :class:`WavFormat`. Defaults to a
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

    async def play(self, chunk_iter: AsyncIterator[bytes]) -> None:
        """Play every chunk from ``chunk_iter``.

        The first 44 bytes are buffered, parsed as a WAV header, and used
        to open the sink; remaining bytes are written through. A PCM
        accumulation buffer ensures every sink.write() call receives a
        whole number of samples (HTTP chunk boundaries may fall mid-sample).

        Raises:
            ServiceUnavailableError: If the WAV header is malformed.
        """
        header_buf = bytearray()
        pcm_buf = bytearray()
        sink_cm: AbstractContextManager[_OutputSink] | None = None
        sink: _OutputSink | None = None
        try:
            async for chunk in chunk_iter:
                if not chunk:
                    continue
                if sink is None:
                    header_buf.extend(chunk)
                    if len(header_buf) < _HEADER_SIZE:
                        continue
                    try:
                        wav_format = parse_wav_header(bytes(header_buf[:_HEADER_SIZE]))
                    except ValueError as e:
                        raise ServiceUnavailableError(
                            f"TTS produced malformed WAV header: {e}",
                            _USER_FACING_ERROR,
                        ) from e
                    sink_cm = self._sink_factory(wav_format)
                    sink = sink_cm.__enter__()
                    pcm_buf.extend(header_buf[_HEADER_SIZE:])
                else:
                    pcm_buf.extend(chunk)
                # Write only complete samples (2 bytes each for int16).
                aligned = len(pcm_buf) & ~1
                if aligned:
                    sink.write(bytes(pcm_buf[:aligned]))
                    del pcm_buf[:aligned]
        except asyncio.CancelledError:
            if sink is not None:
                sink.abort()
            raise
        finally:
            if sink_cm is not None:
                sink_cm.__exit__(None, None, None)


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
