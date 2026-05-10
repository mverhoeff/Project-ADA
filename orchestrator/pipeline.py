"""Streaming Producer/Consumer turn loop.

One :func:`run_turn` call performs a full conversational round-trip:

    microphone capture → STT → LLM token stream → sentence splitter
                       ↘ asyncio.Queue ↗
                                              → TTS → audio playback

The LLM stream and TTS playback run as two concurrent ``asyncio`` tasks,
so the user starts hearing the response while the model is still
generating it. A leading "acknowledgement" sentence (PLAN.md §13) plays
immediately while a parallel scan watches for ``<tool_call>…</tool_call>``
in the accumulated text. When detected, the producer stops feeding the
sentence splitter, the consumer drains the queue (so any acknowledgement
finishes playing), the tool runs, and the LLM is re-invoked for its
final reply within the same turn.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from agent.executor import ToolExecutor
from agent.router import detect_tool_call
from agent.tools.base import BaseTool
from core.exceptions import ServiceUnavailableError, ToolExecutionError
from core.logger import get_logger
from orchestrator.audio_input import capture_until_silence
from orchestrator.audio_output import AudioPlayer
from orchestrator.sentence_splitter import SentenceSplitter
from orchestrator.session import Session
from services.llm.client import LLMClient
from services.llm.prompt_builder import build_messages
from services.stt.client import STTClient
from services.tts.client import TTSClient

_log = get_logger(__name__)

_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
_TAIL_KEEP = len(_TOOL_OPEN) - 1
_DEFAULT_QUEUE_SIZE = 5
_DEFAULT_MAX_TOOL_ITERATIONS = 10


@dataclass
class PipelineDeps:
    """Long-lived collaborators the pipeline composes each turn."""

    stt: STTClient
    llm: LLMClient
    tts: TTSClient
    player: AudioPlayer
    tools: list[BaseTool]
    executor: ToolExecutor


async def run_turn(
    session: Session,
    deps: PipelineDeps,
    config: dict[str, Any],
) -> None:
    """Run one full conversational turn from microphone to speaker.

    Captures audio, transcribes, streams an LLM reply with concurrent
    TTS playback, and loops on tool calls within the same turn. Errors
    from any service are spoken back to the user via TTS (best-effort)
    and the turn ends without raising.

    Args:
        session: In-memory conversation history. Mutated in place.
        deps: Service clients, tool list, and executor.
        config: Loaded config dict; the ``orchestrator`` subtree tunes
            queue size and the tool-iteration safety cap.
    """
    orch_cfg = config.get("orchestrator", {})
    max_tool_iters = int(
        orch_cfg.get("max_tool_iterations", _DEFAULT_MAX_TOOL_ITERATIONS)
    )

    wav = await asyncio.to_thread(capture_until_silence, config)

    try:
        stt_result = await deps.stt.transcribe(wav)
    except ServiceUnavailableError as e:
        _log.warning("turn_stt_unavailable", error=str(e))
        await _speak_user_message(deps, e.user_message)
        return

    transcript = stt_result.get("text", "").strip()
    if not transcript:
        _log.info("turn_empty_transcript")
        return
    session.add_message("user", transcript)

    for _ in range(max_tool_iters):
        messages = build_messages(session, deps.tools)
        try:
            tool_call, assistant_text = await _stream_with_tts(
                deps, messages, config
            )
        except ServiceUnavailableError as e:
            _log.warning("turn_stream_unavailable", error=str(e))
            await _speak_user_message(deps, e.user_message)
            session.add_message("assistant", e.user_message)
            return

        if assistant_text:
            session.add_message("assistant", assistant_text)

        if tool_call is None:
            return

        try:
            result = await asyncio.to_thread(deps.executor.execute, tool_call)
        except ToolExecutionError as e:
            _log.warning("tool_execution_failed", error=str(e))
            result = e.user_message
        session.add_message("tool", result)

    _log.warning("turn_max_tool_iterations_exceeded", limit=max_tool_iters)


async def _stream_with_tts(
    deps: PipelineDeps,
    messages: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Run the Producer/Consumer pair for a single LLM stream.

    Returns ``(tool_call_or_none, full_assistant_text)``. The assistant
    text is the verbatim concatenation of every token the LLM emitted,
    including any ``<tool_call>…</tool_call>`` block.
    """
    queue_size = int(
        config.get("orchestrator", {}).get("queue_size", _DEFAULT_QUEUE_SIZE)
    )
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=queue_size)
    accumulated: list[str] = []
    detected_tool: dict[str, Any] | None = None

    async def producer() -> None:
        nonlocal detected_tool
        splitter = SentenceSplitter()
        pending = ""
        in_tool_text = False
        try:
            async for token in deps.llm.stream_chat(messages):
                accumulated.append(token)
                full = "".join(accumulated)

                if in_tool_text:
                    tc = detect_tool_call(full)
                    if tc is not None:
                        detected_tool = tc
                        return
                    continue

                pending += token
                open_idx = pending.find(_TOOL_OPEN)
                if open_idx >= 0:
                    safe = pending[:open_idx]
                    if safe:
                        for sentence in splitter.feed(safe):
                            await _safe_put(queue, sentence)
                    pending = ""
                    in_tool_text = True
                    tc = detect_tool_call(full)
                    if tc is not None:
                        detected_tool = tc
                        return
                    continue

                if len(pending) > _TAIL_KEEP:
                    safe = pending[:-_TAIL_KEEP]
                    pending = pending[-_TAIL_KEEP:]
                    for sentence in splitter.feed(safe):
                        await _safe_put(queue, sentence)

            if not in_tool_text:
                if pending:
                    for sentence in splitter.feed(pending):
                        await _safe_put(queue, sentence)
                for sentence in splitter.flush():
                    await _safe_put(queue, sentence)
        finally:
            await queue.put(None)

    async def consumer() -> None:
        while True:
            sentence = await queue.get()
            if sentence is None:
                return
            await deps.player.play(deps.tts.synthesize(sentence))

    producer_task = asyncio.create_task(producer())
    consumer_task = asyncio.create_task(consumer())
    try:
        await asyncio.gather(producer_task, consumer_task)
    except BaseException:
        if not producer_task.done():
            producer_task.cancel()
        if not consumer_task.done():
            consumer_task.cancel()
        await asyncio.gather(
            producer_task, consumer_task, return_exceptions=True
        )
        raise

    return detected_tool, "".join(accumulated)


async def _safe_put(queue: asyncio.Queue[str | None], sentence: str) -> None:
    """Drop empty / tool-call-tainted sentences before queuing."""
    cleaned = sentence.strip()
    if not cleaned:
        return
    if _TOOL_OPEN in cleaned or _TOOL_CLOSE in cleaned:
        return
    await queue.put(cleaned)


async def _speak_user_message(deps: PipelineDeps, message: str) -> None:
    """Best-effort TTS of an error message; swallow further failures."""
    if not message.strip():
        return
    try:
        await deps.player.play(deps.tts.synthesize(message))
    except ServiceUnavailableError as e:
        _log.error("error_tts_failed", message=message, error=str(e))
