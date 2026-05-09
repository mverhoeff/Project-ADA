"""Kokoro v1.0 ONNX wrapper for the TTS service.

Wraps :class:`kokoro_onnx.Kokoro` so the server has a single, simple async
streaming entry point. Text preprocessing (markdown stripping) is delegated
to :mod:`services.tts.audio_utils`. Number expansion and grapheme-to-phoneme
conversion are handled inside Kokoro itself via espeak.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from kokoro_onnx import Kokoro

from .audio_utils import preprocess_text


class KokoroSynthesizer:
    """Stateful synthesizer backed by kokoro-onnx.

    The model and voice embeddings are loaded once at construction time
    (auto-downloaded to ``~/.cache/kokoro-onnx/`` on first run) and stay
    resident for the lifetime of the process.

    Args:
        config: Full config dict. Reads ``tts.voice`` and ``tts.speed`` as
            defaults applied when callers omit those parameters.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        tts_cfg = config["tts"]
        self._default_voice: str = tts_cfg["voice"]
        self._default_speed: float = float(tts_cfg["speed"])
        self._model = Kokoro(model_path=None, voices_path=None)

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
