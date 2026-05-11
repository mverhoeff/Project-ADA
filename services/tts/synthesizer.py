"""Kokoro v1.0 ONNX wrapper for the TTS service.

Wraps :class:`kokoro_onnx.Kokoro` so the server has a single, simple async
streaming entry point. Text preprocessing (markdown stripping) is delegated
to :mod:`services.tts.audio_utils`. Number expansion and grapheme-to-phoneme
conversion are handled inside Kokoro itself via espeak.

Model files are downloaded to ``~/.cache/kokoro-onnx/`` on first run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from kokoro_onnx import Kokoro

from core.logger import get_logger
from .audio_utils import preprocess_text

_log = get_logger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "kokoro-onnx"
_BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
_VOICES_FILENAME = "voices-v1.0.bin"


def _download(url: str, dest: Path) -> None:
    _log.info("kokoro_downloading", url=url, dest=str(dest))
    tmp = dest.with_suffix(".tmp")
    with httpx.Client(follow_redirects=True, timeout=None) as client:
        with client.stream("GET", url) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
    tmp.rename(dest)
    _log.info("kokoro_download_complete", dest=str(dest))


def _ensure_model_files(model_name: str) -> tuple[Path, Path]:
    """Return (model_path, voices_path), downloading from GitHub if missing."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_file = _CACHE_DIR / f"{model_name}.onnx"
    voices_file = _CACHE_DIR / _VOICES_FILENAME
    for url, dest in [
        (f"{_BASE_URL}/{model_name}.onnx", model_file),
        (f"{_BASE_URL}/{_VOICES_FILENAME}", voices_file),
    ]:
        if not dest.exists():
            _download(url, dest)
    return model_file, voices_file


class KokoroSynthesizer:
    """Stateful synthesizer backed by kokoro-onnx.

    The model and voice embeddings are loaded once at construction time
    (downloaded to ``~/.cache/kokoro-onnx/`` on first run) and stay
    resident for the lifetime of the process.

    Args:
        config: Full config dict. Reads ``tts.model``, ``tts.voice``, and
            ``tts.speed`` as defaults applied when callers omit those parameters.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        tts_cfg = config["tts"]
        self._default_voice: str = tts_cfg["voice"]
        self._default_speed: float = float(tts_cfg["speed"])
        model_path, voices_path = _ensure_model_files(tts_cfg["model"])
        self._model = Kokoro(model_path=str(model_path), voices_path=str(voices_path))

    def get_voices(self) -> list[str]:
        """Return the list of available voice IDs, sorted."""
        return sorted(self._model.get_voices())

    @property
    def default_voice(self) -> str:
        return self._default_voice

    @property
    def default_speed(self) -> float:
        return self._default_speed

    async def synthesize_stream(
        self,
        text: str,
        voice: str,
        speed: float,
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Yield ``(samples, sample_rate)`` audio chunks as Kokoro produces them.

        The text is preprocessed (markdown stripped) before synthesis. Each
        chunk is a sub-sentence of float32 samples in ``[-1, 1]``.

        Args:
            text: A sentence of text to synthesize.
            voice: Voice ID; must be one of the values in
                :meth:`get_voices`.
            speed: Speech rate multiplier (1.0 = natural).

        Yields:
            Tuples of ``(samples, sample_rate)`` where ``sample_rate`` is
            always 24 000 for Kokoro v1.0.
        """
        clean = preprocess_text(text)
        async for samples, sample_rate in self._model.create_stream(
            clean, voice=voice, speed=speed, lang="en-us"
        ):
            yield samples, sample_rate
