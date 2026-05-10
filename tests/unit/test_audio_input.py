"""Unit tests for :mod:`orchestrator.audio_input`."""

from __future__ import annotations

import io
import wave
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import webrtcvad

from orchestrator import audio_input
from orchestrator.audio_input import (
    capture_until_silence,
    frames_to_wav,
    is_silence_frame,
    should_stop,
)


_SAMPLE_RATE = 16_000
_FRAME_SAMPLES = 320  # 20 ms at 16 kHz
_FRAME_BYTES = _FRAME_SAMPLES * 2  # int16

_BASE_CONFIG: dict[str, Any] = {
    "audio": {
        "mic_sample_rate": _SAMPLE_RATE,
        "silence_threshold_ms": 1500,
        "max_recording_ms": 30_000,
        "vad_aggressiveness": 2,
        "input_device": None,
    },
}


def _silence_frame() -> bytes:
    return b"\x00" * _FRAME_BYTES


def _sine_frame(freq_hz: float = 200.0, amplitude: float = 0.9) -> bytes:
    t = np.arange(_FRAME_SAMPLES, dtype=np.float32) / _SAMPLE_RATE
    samples = np.sin(2 * np.pi * freq_hz * t) * amplitude
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    return pcm.tobytes()


# ---------------------------------------------------------------------------
# frames_to_wav
# ---------------------------------------------------------------------------


def test_frames_to_wav_round_trip_via_stdlib_wave() -> None:
    frames = [_sine_frame(), _silence_frame(), _sine_frame()]
    wav_bytes = frames_to_wav(frames, _SAMPLE_RATE)

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == _SAMPLE_RATE
        assert wf.getnframes() == _FRAME_SAMPLES * 3
        assert wf.readframes(wf.getnframes()) == b"".join(frames)


def test_frames_to_wav_empty_list_produces_valid_empty_wav() -> None:
    wav_bytes = frames_to_wav([], _SAMPLE_RATE)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnframes() == 0
        assert wf.getframerate() == _SAMPLE_RATE


def test_frames_to_wav_uses_provided_sample_rate() -> None:
    wav_bytes = frames_to_wav([_silence_frame()], 8_000)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getframerate() == 8_000


# ---------------------------------------------------------------------------
# is_silence_frame
# ---------------------------------------------------------------------------


def test_is_silence_frame_returns_true_for_zero_pcm() -> None:
    vad = webrtcvad.Vad(2)
    assert is_silence_frame(_silence_frame(), vad, _SAMPLE_RATE) is True


def test_is_silence_frame_returns_false_for_loud_sine() -> None:
    vad = webrtcvad.Vad(2)
    assert is_silence_frame(_sine_frame(), vad, _SAMPLE_RATE) is False


# ---------------------------------------------------------------------------
# should_stop
# ---------------------------------------------------------------------------


def test_should_stop_false_when_no_speech_yet_even_with_long_silence() -> None:
    assert should_stop(
        consecutive_silence_ms=10_000,
        elapsed_ms=10_000,
        silence_threshold_ms=1500,
        max_recording_ms=30_000,
        has_any_speech=False,
    ) is False


def test_should_stop_true_at_silence_threshold_after_speech() -> None:
    assert should_stop(
        consecutive_silence_ms=1500,
        elapsed_ms=5_000,
        silence_threshold_ms=1500,
        max_recording_ms=30_000,
        has_any_speech=True,
    ) is True


def test_should_stop_false_just_below_silence_threshold() -> None:
    assert should_stop(
        consecutive_silence_ms=1480,
        elapsed_ms=5_000,
        silence_threshold_ms=1500,
        max_recording_ms=30_000,
        has_any_speech=True,
    ) is False


def test_should_stop_true_at_max_recording_even_without_speech() -> None:
    assert should_stop(
        consecutive_silence_ms=30_000,
        elapsed_ms=30_000,
        silence_threshold_ms=1500,
        max_recording_ms=30_000,
        has_any_speech=False,
    ) is True


# ---------------------------------------------------------------------------
# capture_until_silence orchestration (with injected frame_source)
# ---------------------------------------------------------------------------


# A speech-marker byte: any frame whose first byte is 0x55 is "speech",
# anything else (notably the silence frame, all zeros) is silence. This
# keeps the orchestration tests deterministic — WebRTC VAD adapts its
# noise floor across many identical frames and is unsuitable for tests
# that hinge on exact frame counts.
_SPEECH_MARKER = b"\x55" + b"\x00" * (_FRAME_BYTES - 1)


def _fake_is_silence_frame(
    pcm_bytes: bytes, vad: webrtcvad.Vad, sample_rate: int
) -> bool:
    return pcm_bytes[:1] != b"\x55"


@pytest.fixture
def fake_vad(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_input, "is_silence_frame", _fake_is_silence_frame)


def _make_source(frames: list[bytes]) -> audio_input.FrameSource:
    def factory() -> Iterator[bytes]:
        yield from frames
    return factory


def _read_wav_frames(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.readframes(wf.getnframes())


def test_capture_stops_after_trailing_silence_threshold(fake_vad: None) -> None:
    # 5 frames speech (100 ms) then 75 frames silence (1500 ms) → stop at 1500 ms.
    speech = [_SPEECH_MARKER for _ in range(5)]
    silence = [_silence_frame() for _ in range(75)]
    # Pad with extra silence the loop must NOT consume.
    extra = [_silence_frame() for _ in range(10)]

    wav_bytes = capture_until_silence(
        _BASE_CONFIG,
        frame_source=_make_source(speech + silence + extra),
    )
    raw = _read_wav_frames(wav_bytes)
    # Captured exactly the speech + threshold-window of silence: 80 frames.
    assert len(raw) == _FRAME_BYTES * 80


def test_capture_does_not_stop_on_leading_silence(fake_vad: None) -> None:
    # 100 frames silence (2000 ms) → must NOT stop, then speech, then trailing silence.
    leading_silence = [_silence_frame() for _ in range(100)]
    speech = [_SPEECH_MARKER for _ in range(5)]
    trailing_silence = [_silence_frame() for _ in range(75)]

    wav_bytes = capture_until_silence(
        _BASE_CONFIG,
        frame_source=_make_source(leading_silence + speech + trailing_silence),
    )
    raw = _read_wav_frames(wav_bytes)
    # All 180 frames must be captured.
    assert len(raw) == _FRAME_BYTES * 180


def test_capture_respects_max_recording_duration(fake_vad: None) -> None:
    # max_recording_ms = 30000, frame_ms = 20 → 1500 frames cap.
    # Provide infinite speech via a generator so the silence path can't trigger.
    def infinite_speech() -> Iterator[bytes]:
        while True:
            yield _SPEECH_MARKER

    wav_bytes = capture_until_silence(
        _BASE_CONFIG,
        frame_source=lambda: infinite_speech(),
    )
    raw = _read_wav_frames(wav_bytes)
    assert len(raw) == _FRAME_BYTES * 1500


def test_capture_default_frame_source_factory_is_invoked_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio_input, "is_silence_frame", _fake_is_silence_frame)
    captured: dict[str, Any] = {}

    def fake_make(
        sample_rate: int, frame_samples: int, device: int | str | None
    ) -> audio_input.FrameSource:
        captured["sample_rate"] = sample_rate
        captured["frame_samples"] = frame_samples
        captured["device"] = device
        # Return a tiny source so the loop terminates quickly.
        speech = [_SPEECH_MARKER for _ in range(2)]
        silence = [_silence_frame() for _ in range(75)]
        return _make_source(speech + silence)

    monkeypatch.setattr(audio_input, "_make_sounddevice_frame_source", fake_make)
    capture_until_silence(_BASE_CONFIG)

    assert captured["sample_rate"] == _SAMPLE_RATE
    assert captured["frame_samples"] == _FRAME_SAMPLES
    assert captured["device"] is None
