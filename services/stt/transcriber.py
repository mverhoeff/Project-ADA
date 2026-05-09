"""Whisper Large V3 Turbo wrapper for the STT service.

Wraps :class:`faster_whisper.WhisperModel` so the server has a single,
synchronous :meth:`transcribe` entry point. Audio preprocessing (WAV
decode, resample, VAD) is delegated to :mod:`services.stt.audio_utils`.
"""

from __future__ import annotations

import time
from typing import Any

from faster_whisper import WhisperModel

from .audio_utils import decode_wav, resample_to_16k, strip_silence


class WhisperTranscriber:
    """Stateless transcriber backed by faster-whisper.

    The model is loaded once at construction time and stays resident for
    the lifetime of the process.

    Args:
        config: Full config dict. Reads ``stt.model``, ``stt.device``, and
            ``stt.compute_type``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        stt_cfg = config["stt"]
        self._model = WhisperModel(
            stt_cfg["model"],
            device=stt_cfg["device"],
            compute_type=stt_cfg["compute_type"],
        )

    def transcribe(self, wav_bytes: bytes) -> dict[str, Any]:
        """Decode WAV bytes and transcribe the speech.

        Args:
            wav_bytes: A complete WAV file as bytes.

        Returns:
            A dict with keys ``text``, ``language``, and ``duration_ms``.
            ``duration_ms`` is the wall-clock time spent on the full call,
            including decoding and inference.
        """
        start = time.monotonic()

        audio, sample_rate = decode_wav(wav_bytes)
        audio = resample_to_16k(audio, sample_rate)
        audio = strip_silence(audio)

        segments, info = self._model.transcribe(audio, beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()

        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "text": text,
            "language": info.language,
            "duration_ms": duration_ms,
        }
