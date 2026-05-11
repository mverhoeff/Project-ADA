"""Whisper Large V3 Turbo wrapper for the STT service.

Wraps :class:`faster_whisper.WhisperModel` so the server has a single,
synchronous :meth:`transcribe` entry point. Audio preprocessing (WAV
decode, resample, VAD) is delegated to :mod:`services.stt.audio_utils`.
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from core.logger import get_logger
from .audio_utils import decode_wav, resample_to_16k, strip_silence

_log = get_logger(__name__)

_CUBLAS_DLL = "cublas64_12.dll"
_CUDA_DLL_SEARCH_PATHS: list[Path] = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/lib/ollama/cuda_v12",
    *[
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit") / f"CUDA/v12.{minor}/bin"
        for minor in range(10)
    ],
]


def _ensure_cuda_dlls() -> None:
    """Make CUDA 12 DLLs discoverable to ctranslate2 on Windows.

    Both mechanisms are needed: os.add_dll_directory() so Python-side ctypes
    loads succeed, and a PATH prefix so the CUDA runtime's own loader (which
    ctranslate2 invokes internally) can resolve cublas64_12.dll and its
    transitive dependencies.
    """
    if platform.system() != "Windows":
        return
    for path in _CUDA_DLL_SEARCH_PATHS:
        if (path / _CUBLAS_DLL).exists():
            path_str = str(path)
            os.add_dll_directory(path_str)
            os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
            _log.info("cuda_dll_directory_added", path=path_str)
            return
    _log.warning(
        "cublas64_12_not_found",
        hint="Install the CUDA 12 Toolkit or add its bin/ to PATH.",
    )


class WhisperTranscriber:
    """Stateless transcriber backed by faster-whisper.

    The model is loaded once at construction time and stays resident for
    the lifetime of the process.

    Args:
        config: Full config dict. Reads ``stt.model``, ``stt.device``, and
            ``stt.compute_type``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        _ensure_cuda_dlls()
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
