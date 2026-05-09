"""FastAPI app for the TTS service.

Exposes ``POST /synthesize``, ``GET /voices``, and ``GET /health`` on the
port configured in ``config/default.yaml`` (default: 8772). The Kokoro
model is loaded once at startup via the FastAPI lifespan hook.

``/synthesize`` returns ``StreamingResponse`` of ``audio/wav``. The first
chunk is a 44-byte WAV header with the data length set to ``0xFFFFFFFF``;
subsequent chunks are raw little-endian 16-bit PCM bytes corresponding to
audio chunks emitted by ``Kokoro.create_stream``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.config import load_config
from core.logger import configure_logging, get_logger

from .audio_utils import samples_to_pcm16, wav_streaming_header
from .synthesizer import KokoroSynthesizer

_KOKORO_SAMPLE_RATE = 24_000

logger = get_logger(__name__)

_synthesizer: KokoroSynthesizer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the Kokoro model on startup, release on shutdown."""
    global _synthesizer
    config = load_config()
    logger.info("tts_loading_model", model=config["tts"]["model"])
    _synthesizer = KokoroSynthesizer(config)
    logger.info("tts_model_ready", default_voice=_synthesizer.default_voice)
    try:
        yield
    finally:
        _synthesizer = None


app = FastAPI(lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    """Body for ``POST /synthesize``."""

    text: str = Field(..., description="Sentence of text to synthesize.")
    voice: str | None = Field(
        default=None, description="Voice ID; falls back to the configured default."
    )
    speed: float | None = Field(
        default=None,
        gt=0.0,
        le=4.0,
        description="Speech rate multiplier; falls back to the configured default.",
    )


@app.post("/synthesize")
async def synthesize(body: SynthesizeRequest) -> StreamingResponse:
    """Synthesize speech and stream WAV audio back via chunked transfer."""
    if _synthesizer is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    voice = body.voice or _synthesizer.default_voice
    speed = body.speed if body.speed is not None else _synthesizer.default_speed

    async def generate() -> AsyncIterator[bytes]:
        yield wav_streaming_header(_KOKORO_SAMPLE_RATE)
        assert _synthesizer is not None
        async for samples, _sr in _synthesizer.synthesize_stream(
            body.text, voice, speed
        ):
            yield samples_to_pcm16(samples)

    return StreamingResponse(generate(), media_type="audio/wav")


@app.get("/voices")
async def voices() -> dict[str, Any]:
    """Return the list of available voice IDs."""
    if _synthesizer is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    return {"voices": _synthesizer.get_voices()}


@app.get("/health")
async def health() -> dict[str, Any]:
    """Return service readiness."""
    return {"status": "ok", "model_loaded": _synthesizer is not None}


def main() -> None:
    """Run the server (blocking entry point)."""
    import uvicorn

    configure_logging()
    config = load_config()
    uvicorn.run(
        "services.tts.server:app",
        host="127.0.0.1",
        port=config["tts"]["port"],
        log_level="info",
    )


if __name__ == "__main__":
    main()
