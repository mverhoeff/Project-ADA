"""Unit tests for :mod:`orchestrator.barge_in`."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from orchestrator.barge_in import BargeInListener, should_trigger


_SAMPLE_RATE = 16_000
_FRAME_SAMPLES = 320  # 20 ms at 16 kHz
_FRAME_BYTES = _FRAME_SAMPLES * 2
_FRAME_DURATION_MS = 20


def _silence_frame() -> bytes:
    return b"\x00" * _FRAME_BYTES


def _speech_frame(freq_hz: float = 220.0, amplitude: float = 0.9) -> bytes:
    """Loud sine wave — WebRTC VAD classifies it as speech."""
    t = np.arange(_FRAME_SAMPLES, dtype=np.float32) / _SAMPLE_RATE
    samples = np.sin(2 * np.pi * freq_hz * t) * amplitude
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    return pcm.tobytes()


def _config(
    *,
    enabled: bool = True,
    threshold_ms: int = 100,
    aggressiveness: int = 2,
) -> dict[str, Any]:
    return {
        "audio": {
            "mic_sample_rate": _SAMPLE_RATE,
            "barge_in_enabled": enabled,
            "barge_in_consecutive_speech_ms": threshold_ms,
            "barge_in_vad_aggressiveness": aggressiveness,
            "input_device": None,
        }
    }


def _make_frame_source(frames: list[bytes]) -> Any:
    """Build a FrameSource that yields the given frames then stops."""

    def factory() -> Iterator[bytes]:
        yield from frames

    return factory


# ---------------------------------------------------------------------------
# should_trigger
# ---------------------------------------------------------------------------


def test_should_trigger_false_below_threshold() -> None:
    assert should_trigger(consecutive_speech_ms=99, threshold_ms=100) is False


def test_should_trigger_true_at_threshold() -> None:
    assert should_trigger(consecutive_speech_ms=100, threshold_ms=100) is True


def test_should_trigger_true_above_threshold() -> None:
    assert should_trigger(consecutive_speech_ms=180, threshold_ms=100) is True


def test_should_trigger_false_at_zero() -> None:
    assert should_trigger(consecutive_speech_ms=0, threshold_ms=100) is False


# ---------------------------------------------------------------------------
# BargeInListener.enabled
# ---------------------------------------------------------------------------


def test_listener_disabled_by_default_when_key_missing() -> None:
    listener = BargeInListener({"audio": {}})
    assert listener.enabled is False


def test_listener_reflects_config_flag() -> None:
    assert BargeInListener(_config(enabled=False)).enabled is False
    assert BargeInListener(_config(enabled=True)).enabled is True


# ---------------------------------------------------------------------------
# wait_for_speech — disabled mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_speech_disabled_never_resolves() -> None:
    listener = BargeInListener(_config(enabled=False))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(listener.wait_for_speech(), timeout=0.05)


@pytest.mark.asyncio
async def test_wait_for_speech_disabled_can_be_cancelled() -> None:
    listener = BargeInListener(_config(enabled=False))
    task = asyncio.create_task(listener.wait_for_speech())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# wait_for_speech — enabled mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_speech_triggers_after_consecutive_speech_frames() -> None:
    # 100 ms threshold = 5 frames at 20 ms each. Provide 10 speech frames
    # so the listener triggers within the first 5.
    frames = [_speech_frame()] * 10
    listener = BargeInListener(
        _config(threshold_ms=100, aggressiveness=2),
        frame_source=_make_frame_source(frames),
    )
    await asyncio.wait_for(listener.wait_for_speech(), timeout=1.0)


@pytest.mark.asyncio
async def test_wait_for_speech_does_not_trigger_on_pure_silence() -> None:
    # All silence: must not trigger. Source eventually exhausts, which
    # surfaces as RuntimeError (the production sounddevice source is
    # infinite — exhaustion is a device failure, not a barge-in).
    frames = [_silence_frame()] * 50
    listener = BargeInListener(
        _config(threshold_ms=100, aggressiveness=2),
        frame_source=_make_frame_source(frames),
    )
    with pytest.raises(RuntimeError, match="exhausted"):
        await asyncio.wait_for(listener.wait_for_speech(), timeout=1.0)


@pytest.mark.asyncio
async def test_wait_for_speech_cancellation_stops_listener() -> None:
    # Infinite silence source: would otherwise block forever.
    def infinite_silence() -> Iterator[bytes]:
        while True:
            yield _silence_frame()

    listener = BargeInListener(
        _config(threshold_ms=100, aggressiveness=2),
        frame_source=infinite_silence,
    )
    task = asyncio.create_task(listener.wait_for_speech())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
