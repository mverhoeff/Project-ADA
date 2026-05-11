"""Background VAD listener for interruption (barge-in).

While the orchestrator pipeline is streaming a reply to the speaker, a
:class:`BargeInListener` runs in parallel and watches the microphone for
user speech. When the user starts talking for at least
``audio.barge_in_consecutive_speech_ms`` consecutive milliseconds, the
listener's :meth:`wait_for_speech` coroutine returns; the pipeline then
cancels its producer/consumer pair so the assistant stops mid-sentence
and a new turn can begin.

Echo from a shared microphone+speaker setup would otherwise cause
constant false-positives, so the listener is disabled by default
(``audio.barge_in_enabled: false``). When disabled,
:meth:`wait_for_speech` returns an awaitable that never resolves on its
own — that way the pipeline can race against it uniformly without
special-casing the None path.

The blocking frame loop runs inside :func:`asyncio.to_thread` and
honours a ``threading.Event`` stop flag set from
:meth:`wait_for_speech` on cancellation, so the thread exits within one
frame (~20 ms).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import webrtcvad

from core.logger import get_logger
from orchestrator.audio_input import FrameSource, is_silence_frame


_FRAME_DURATION_MS = 20
_log = get_logger(__name__)


def should_trigger(consecutive_speech_ms: int, threshold_ms: int) -> bool:
    """Return True once accumulated speech meets the trigger threshold."""
    return consecutive_speech_ms >= threshold_ms


class BargeInListener:
    """Detect that the user has started speaking during TTS playback.

    Args:
        config: Loaded config dict. Reads from the ``audio`` subtree:
            ``barge_in_enabled``, ``barge_in_consecutive_speech_ms``,
            ``barge_in_vad_aggressiveness``, ``mic_sample_rate``,
            ``input_device``.
        frame_source: Optional iterator factory yielding 20 ms int16
            mono PCM frames. Defaults to a ``sounddevice``-backed source
            built on first use. Tests inject a fake here.
    """

    def __init__(
        self,
        config: dict[str, Any],
        frame_source: FrameSource | None = None,
    ) -> None:
        audio_cfg = config.get("audio", {})
        self._enabled = bool(audio_cfg.get("barge_in_enabled", False))
        self._sample_rate = int(audio_cfg.get("mic_sample_rate", 16_000))
        self._threshold_ms = int(audio_cfg.get("barge_in_consecutive_speech_ms", 300))
        self._aggressiveness = int(audio_cfg.get("barge_in_vad_aggressiveness", 3))
        self._input_device = audio_cfg.get("input_device")
        self._frame_source = frame_source

    @property
    def enabled(self) -> bool:
        """True if the listener will actually run when awaited."""
        return self._enabled

    async def wait_for_speech(self) -> None:
        """Block until N consecutive speech frames are detected.

        If the listener is disabled in config, awaits an ``Event`` that
        is never set — the only way out is cancellation, which simply
        propagates :class:`asyncio.CancelledError`.
        """
        if not self._enabled:
            await asyncio.Event().wait()
            return

        stop_flag = threading.Event()
        try:
            await asyncio.to_thread(self._run, stop_flag)
        except asyncio.CancelledError:
            stop_flag.set()
            raise

    def _run(self, stop_flag: threading.Event) -> None:
        """Synchronous frame loop. Returns once speech is detected.

        Raises:
            RuntimeError: If the frame source exhausts before speech is
                detected. The microphone stream is expected to be
                infinite — exhaustion indicates a device failure that
                the pipeline should log rather than treat as barge-in.
        """
        frame_samples = self._sample_rate * _FRAME_DURATION_MS // 1000
        frame_source = self._frame_source
        if frame_source is None:
            frame_source = _make_sounddevice_frame_source(
                sample_rate=self._sample_rate,
                frame_samples=frame_samples,
                device=self._input_device,
            )

        vad = webrtcvad.Vad(self._aggressiveness)
        consecutive_speech_ms = 0
        for frame in frame_source():
            if stop_flag.is_set():
                return
            if is_silence_frame(frame, vad, self._sample_rate):
                consecutive_speech_ms = 0
            else:
                consecutive_speech_ms += _FRAME_DURATION_MS
            if should_trigger(consecutive_speech_ms, self._threshold_ms):
                _log.info(
                    "barge_in_triggered",
                    consecutive_speech_ms=consecutive_speech_ms,
                )
                return
        raise RuntimeError("barge-in frame source exhausted unexpectedly")


def _make_sounddevice_frame_source(
    sample_rate: int,
    frame_samples: int,
    device: int | str | None,
) -> FrameSource:
    """Build a frame source backed by ``sounddevice.RawInputStream``.

    Mirrors :func:`orchestrator.audio_input._make_sounddevice_frame_source`
    so the listener can open its own independent input stream alongside
    the playback output stream. ``sounddevice`` is imported lazily so
    this module remains importable without PortAudio.
    """

    def factory():  # type: ignore[no-untyped-def]
        import sounddevice as sd  # local import — see module docstring

        stream = sd.RawInputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_samples,
            device=device,
        )
        stream.start()
        try:
            frame_bytes = frame_samples * 2
            while True:
                data, _overflowed = stream.read(frame_samples)
                yield bytes(data)[:frame_bytes]
        finally:
            stream.stop()
            stream.close()

    return factory
