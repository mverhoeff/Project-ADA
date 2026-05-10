"""VAD-gated microphone capture.

Records 20 ms int16 mono PCM frames at 16 kHz, classifies each frame with
WebRTC VAD, and stops when the trailing-silence window exceeds the
configured threshold (or when the hard recording cap is reached). Returns
a complete WAV file (header + data) ready to POST to the STT service.

The function is intentionally synchronous — the orchestrator pipeline
will wrap it in :func:`asyncio.to_thread` so the audio loop runs off the
event loop. ``sounddevice`` is imported lazily inside
``_make_sounddevice_frame_source`` so the pure helpers and the
orchestration logic can be unit-tested without PortAudio installed.
"""

from __future__ import annotations

import io
import wave
from collections.abc import Callable, Iterator
from typing import Any

import webrtcvad


_FRAME_DURATION_MS = 20

FrameSource = Callable[[], Iterator[bytes]]
"""Returns an iterator yielding fixed-size 20 ms int16 mono PCM frames."""


def is_silence_frame(
    pcm_bytes: bytes,
    vad: webrtcvad.Vad,
    sample_rate: int,
) -> bool:
    """Return True if WebRTC VAD classifies ``pcm_bytes`` as non-speech.

    Args:
        pcm_bytes: Exactly one 20 ms frame of int16 mono PCM at
            ``sample_rate``. (640 bytes for 16 kHz.)
        vad: A configured ``webrtcvad.Vad`` instance.
        sample_rate: 8000, 16000, 32000, or 48000 Hz (VAD constraint).

    Returns:
        ``True`` when the frame is silence, ``False`` when it is speech.
    """
    return not vad.is_speech(pcm_bytes, sample_rate)


def frames_to_wav(frames: list[bytes], sample_rate: int = 16_000) -> bytes:
    """Pack a list of int16 mono PCM frames into a complete WAV file.

    Args:
        frames: Each entry is raw int16 little-endian PCM bytes.
        sample_rate: Sample rate in Hz to write into the WAV header.

    Returns:
        Bytes of a valid WAV file (header + concatenated frame data).
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def should_stop(
    consecutive_silence_ms: int,
    elapsed_ms: int,
    silence_threshold_ms: int,
    max_recording_ms: int,
    has_any_speech: bool,
) -> bool:
    """Decide whether to terminate the capture loop.

    Stops if either of:
    * ``elapsed_ms >= max_recording_ms`` (hard cap), or
    * The user has already spoken at least one frame and the trailing
      silence window has reached ``silence_threshold_ms``.

    The ``has_any_speech`` guard prevents the loop from immediately
    returning when the user is slow to start speaking.
    """
    if elapsed_ms >= max_recording_ms:
        return True
    if has_any_speech and consecutive_silence_ms >= silence_threshold_ms:
        return True
    return False


def capture_until_silence(
    config: dict[str, Any],
    frame_source: FrameSource | None = None,
) -> bytes:
    """Capture microphone audio until end-of-utterance, return WAV bytes.

    Args:
        config: Loaded config dict; the ``audio`` subtree drives sample
            rate, silence threshold, max duration, VAD aggressiveness, and
            input device selection.
        frame_source: Optional iterator factory that yields 20 ms PCM
            frames. Defaults to a ``sounddevice``-backed source. Tests
            inject a fake here.

    Returns:
        A complete int16 mono PCM WAV file at the configured sample rate.
    """
    audio_cfg = config["audio"]
    sample_rate = int(audio_cfg["mic_sample_rate"])
    silence_ms = int(audio_cfg["silence_threshold_ms"])
    max_ms = int(audio_cfg["max_recording_ms"])
    aggressiveness = int(audio_cfg.get("vad_aggressiveness", 2))
    frame_samples = sample_rate * _FRAME_DURATION_MS // 1000

    if frame_source is None:
        frame_source = _make_sounddevice_frame_source(
            sample_rate=sample_rate,
            frame_samples=frame_samples,
            device=audio_cfg.get("input_device"),
        )

    vad = webrtcvad.Vad(aggressiveness)
    frames: list[bytes] = []
    consecutive_silence_ms = 0
    elapsed_ms = 0
    has_any_speech = False

    for frame in frame_source():
        frames.append(frame)
        elapsed_ms += _FRAME_DURATION_MS
        if is_silence_frame(frame, vad, sample_rate):
            consecutive_silence_ms += _FRAME_DURATION_MS
        else:
            consecutive_silence_ms = 0
            has_any_speech = True
        if should_stop(
            consecutive_silence_ms,
            elapsed_ms,
            silence_ms,
            max_ms,
            has_any_speech,
        ):
            break

    return frames_to_wav(frames, sample_rate)


def _make_sounddevice_frame_source(
    sample_rate: int,
    frame_samples: int,
    device: int | str | None,
) -> FrameSource:
    """Build a frame source backed by ``sounddevice.RawInputStream``.

    ``sounddevice`` is imported lazily so this module remains importable
    in environments without PortAudio (e.g. CI test runs that inject a
    fake ``frame_source``).
    """

    def factory() -> Iterator[bytes]:
        import sounddevice as sd  # local import — see module docstring

        stream = sd.RawInputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_samples,
            device=device,
        )
        stream.start()
        try:
            frame_bytes = frame_samples * 2  # int16 = 2 bytes per sample
            while True:
                data, _overflowed = stream.read(frame_samples)
                yield bytes(data)[:frame_bytes]
        finally:
            stream.stop()
            stream.close()

    return factory
