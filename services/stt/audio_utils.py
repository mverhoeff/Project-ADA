"""Pure-logic audio utilities for the STT service.

Functions here have no I/O or GPU dependency, so they can be unit-tested
exhaustively. They cover three responsibilities:

1. Decoding raw WAV bytes into a mono float32 numpy array.
2. Resampling that array to 16 kHz (Whisper's native rate).
3. Stripping silent frames using WebRTC VAD before inference.
"""

from __future__ import annotations

import io
import wave
from math import gcd

import numpy as np
import webrtcvad
from scipy.signal import resample_poly

_TARGET_SAMPLE_RATE = 16_000
_VAD_FRAME_DURATION_MS = 20
_VAD_SUPPORTED_RATES = {8_000, 16_000, 32_000, 48_000}


def decode_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes to a mono float32 array in ``[-1, 1]``.

    Multi-channel input is downmixed to mono by averaging across channels.

    Args:
        wav_bytes: The complete contents of a WAV file.

    Returns:
        A tuple ``(audio, sample_rate)`` where ``audio`` is a 1-D ``float32``
        numpy array normalised into ``[-1.0, 1.0]``.

    Raises:
        ValueError: If the sample width is not 1, 2, or 4 bytes.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    dtype_map: dict[int, type[np.signedinteger]] = {
        1: np.int8,
        2: np.int16,
        4: np.int32,
    }
    if sample_width not in dtype_map:
        raise ValueError(f"Unsupported sample width: {sample_width} bytes")

    audio = np.frombuffer(raw, dtype=dtype_map[sample_width]).astype(np.float32)
    audio /= float(2 ** (8 * sample_width - 1))

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio.astype(np.float32, copy=False), sample_rate


def resample_to_16k(audio: np.ndarray, source_rate: int) -> np.ndarray:
    """Resample ``audio`` to 16 000 Hz.

    A no-op if the source is already at 16 kHz.

    Args:
        audio: Mono float32 audio.
        source_rate: Sample rate of ``audio`` in Hz.

    Returns:
        A float32 numpy array sampled at 16 000 Hz.
    """
    if source_rate == _TARGET_SAMPLE_RATE:
        return audio.astype(np.float32, copy=False)

    g = gcd(source_rate, _TARGET_SAMPLE_RATE)
    up = _TARGET_SAMPLE_RATE // g
    down = source_rate // g
    resampled = resample_poly(audio, up, down)
    return resampled.astype(np.float32)


def strip_silence(
    audio: np.ndarray,
    sample_rate: int = _TARGET_SAMPLE_RATE,
    aggressiveness: int = 2,
) -> np.ndarray:
    """Remove silent frames using WebRTC VAD.

    Audio is processed in 20 ms frames; only frames classified as speech
    are retained. If every frame is classified as silence, the original
    audio is returned unchanged so the transcriber still has input.

    Args:
        audio: Mono float32 audio at ``sample_rate`` Hz.
        sample_rate: Sample rate of ``audio``. Must be one of 8000, 16000,
            32000, or 48000 Hz (the rates supported by WebRTC VAD).
        aggressiveness: WebRTC VAD aggressiveness mode (0 = least, 3 = most).

    Returns:
        A float32 numpy array containing only speech frames, or the original
        audio if no speech was detected.

    Raises:
        ValueError: If ``sample_rate`` is not supported by WebRTC VAD.
    """
    if sample_rate not in _VAD_SUPPORTED_RATES:
        raise ValueError(
            f"WebRTC VAD requires sample_rate in {sorted(_VAD_SUPPORTED_RATES)}, "
            f"got {sample_rate}"
        )

    vad = webrtcvad.Vad(aggressiveness)
    frame_samples = sample_rate * _VAD_FRAME_DURATION_MS // 1000
    n_frames = len(audio) // frame_samples
    if n_frames == 0:
        return audio.astype(np.float32, copy=False)

    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

    speech_frames: list[np.ndarray] = []
    for i in range(n_frames):
        start = i * frame_samples
        end = start + frame_samples
        if vad.is_speech(pcm[start:end].tobytes(), sample_rate):
            speech_frames.append(audio[start:end])

    if not speech_frames:
        return audio.astype(np.float32, copy=False)

    return np.concatenate(speech_frames).astype(np.float32, copy=False)
