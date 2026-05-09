"""FastAPI app for the STT service.

Exposes ``POST /transcribe`` and ``GET /health`` on the port configured in
``config/default.yaml`` (default: 8771). The Whisper model is loaded once
at startup via the FastAPI lifespan hook and stays resident for the
lifetime of the process.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from core.config import load_config
from core.logger import configure_logging, get_logger

from .transcriber import WhisperTranscriber

logger = get_logger(__name__)

_transcriber: WhisperTranscriber | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the Whisper model on startup, release on shutdown."""
    global _transcriber
    config = load_config()
    logger.info("stt_loading_model", model=config["stt"]["model"])
    _transcriber = WhisperTranscriber(config)
    logger.info("stt_model_ready")
    try:
        yield
    finally:
        _transcriber = None


app = FastAPI(lifespan=lifespan)


@app.post("/transcribe")
async def transcribe(request: Request) -> dict[str, Any]:
    """Transcribe a WAV-encoded audio body."""
    if _transcriber is None:
        raise HTTPException(status_code=503, detail="STT model not loaded")
    wav_bytes = await request.body()
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="Empty request body")
    return await asyncio.to_thread(_transcriber.transcribe, wav_bytes)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Return service readiness."""
    return {"status": "ok", "model_loaded": _transcriber is not None}


def main() -> None:
    """Run the server (blocking entry point)."""
    import uvicorn

    configure_logging()
    config = load_config()
    uvicorn.run(
        "services.stt.server:app",
        host="127.0.0.1",
        port=config["stt"]["port"],
        log_level="info",
    )


if __name__ == "__main__":
    main()
