"""Unit tests for :mod:`services.stt.audio_utils`."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from services.stt.audio_utils import decode_wav, resample_to_16k, strip_silence


def _encode_wav(audio: np.ndarray, sample_rate: int, n_channels: int = 1) -> bytes:
    """Encode a float32 array in ``[-1, 1]`` as 16-bit PCM WAV bytes."""
    pcm = np.clip(np.round(audio * 32768.0), -32768, 32767).astype(np.int16)
    if n_channels > 1:
        pcm = np.tile(pcm.reshape(-1, 1), (1, n_channels)).flatten()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buffer.getvalue()


# -- decode_wav --------------------------------------------------------------


def test_decode_wav_mono_roundtrip() -> None:
    audio = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    wav = _encode_wav(audio, 16_000)

    decoded, rate = decode_wav(wav)

    assert rate == 16_000
    assert decoded.dtype == np.float32
    assert decoded.shape == (1600,)
    np.testing.assert_allclose(decoded, audio, atol=1.0 / 32767)


def test_decode_wav_stereo_downmix_to_mono() -> None:
    audio = np.full(1000, 0.25, dtype=np.float32)
    wav = _encode_wav(audio, 16_000, n_channels=2)

    decoded, rate = decode_wav(wav)

    assert rate == 16_000
    assert decoded.shape == (1000,)
    np.testing.assert_allclose(decoded, audio, atol=1.0 / 32767)


def test_decode_wav_stays_in_unit_range() -> None:
    audio = np.array([-1.0, 0.0, 0.999], dtype=np.float32)
    wav = _encode_wav(audio, 16_000)

    decoded, _ = decode_wav(wav)

    assert decoded.min() >= -1.0
    assert decoded.max() <= 1.0


def test_decode_wav_preserves_sample_rate() -> None:
    audio = np.zeros(8_000, dtype=np.float32)
    wav = _encode_wav(audio, 44_100)

    _, rate = decode_wav(wav)

    assert rate == 44_100


# -- resample_to_16k ---------------------------------------------------------


def test_resample_noop_when_already_16k() -> None:
    audio = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)

    out = resample_to_16k(audio, 16_000)

    assert out.dtype == np.float32
    assert len(out) == 1600


def test_resample_44100_to_16k_length_matches_ratio() -> None:
    audio = np.zeros(44_100, dtype=np.float32)  # 1 second

    out = resample_to_16k(audio, 44_100)

    assert abs(len(out) - 16_000) <= 1
    assert out.dtype == np.float32


def test_resample_8k_to_16k_doubles_length() -> None:
    audio = np.zeros(8_000, dtype=np.float32)

    out = resample_to_16k(audio, 8_000)

    assert abs(len(out) - 16_000) <= 1
    assert out.dtype == np.float32


# -- strip_silence -----------------------------------------------------------


def test_strip_silence_all_silent_falls_back_to_original() -> None:
    audio = np.zeros(16_000, dtype=np.float32)

    out = strip_silence(audio)

    assert len(out) == len(audio)
    assert out.dtype == np.float32


def test_strip_silence_audio_shorter_than_one_frame_returns_original() -> None:
    audio = np.zeros(100, dtype=np.float32)  # < 320 samples (one 20ms frame)

    out = strip_silence(audio)

    assert len(out) == 100
    assert out.dtype == np.float32


def test_strip_silence_rejects_unsupported_sample_rate() -> None:
    audio = np.zeros(1600, dtype=np.float32)
    with pytest.raises(ValueError):
        strip_silence(audio, sample_rate=22_050)


def test_strip_silence_keeps_only_speech_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock VAD to mark even-indexed frames as speech and verify selection."""

    class FakeVad:
        def __init__(self, _aggressiveness: int) -> None:
            self.calls = 0

        def is_speech(self, _frame_bytes: bytes, _sample_rate: int) -> bool:
            keep = self.calls % 2 == 0  # frames 0, 2 → speech; 1, 3 → silence
            self.calls += 1
            return keep

    monkeypatch.setattr("services.stt.audio_utils.webrtcvad.Vad", FakeVad)

    frame_samples = 320  # 20 ms at 16 kHz
    audio = np.linspace(-0.5, 0.5, 4 * frame_samples, dtype=np.float32)

    out = strip_silence(audio, sample_rate=16_000)

    assert len(out) == 2 * frame_samples
    np.testing.assert_array_equal(
        out[:frame_samples], audio[0:frame_samples]
    )
    np.testing.assert_array_equal(
        out[frame_samples:], audio[2 * frame_samples : 3 * frame_samples]
    )


def test_strip_silence_preserves_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    class AlwaysSpeech:
        def __init__(self, _aggressiveness: int) -> None:
            pass

        def is_speech(self, _frame_bytes: bytes, _sample_rate: int) -> bool:
            return True

    monkeypatch.setattr("services.stt.audio_utils.webrtcvad.Vad", AlwaysSpeech)
    audio = np.full(640, 0.1, dtype=np.float32)

    out = strip_silence(audio)

    assert out.dtype == np.float32
    assert len(out) == 640
