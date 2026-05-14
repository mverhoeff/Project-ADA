"""Streaming Producer/Consumer turn loop.

One :func:`run_turn` call performs a full conversational round-trip:

    microphone capture → STT → LLM event stream → sentence splitter
                       ↘ asyncio.Queue ↗
                                              → TTS → audio playback

The LLM stream and TTS playback run as two concurrent ``asyncio`` tasks,
so the user starts hearing the response while the model is still
generating it. A leading "acknowledgement" sentence (PLAN.md §13) plays
immediately. The LLM yields a typed event stream — :class:`TextChunk`
events feed the sentence splitter, while a :class:`ToolCallChunk` event
flips the producer into tool mode: it stops feeding the queue, the
consumer drains whatever was already enqueued (so the acknowledgement
finishes playing), the tool runs, and the LLM is re-invoked for its
final reply within the same turn.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from agent.executor import ToolExecutor
from agent.tools.base import BaseTool
from core.exceptions import ServiceUnavailableError, ToolExecutionError
from core.logger import get_logger
from orchestrator.audio_input import capture_until_silence
from orchestrator.audio_output import AudioPlayer
from orchestrator.barge_in import BargeInListener
from orchestrator.sentence_splitter import SentenceSplitter
from orchestrator.session import Session
from services.llm.client import LLMClient
from services.llm.events import TextChunk, ToolCallChunk
from services.llm.prompt_builder import build_messages, build_tools
from services.stt.client import STTClient
from services.tts.client import TTSClient

_log = get_logger(__name__)

_DEFAULT_QUEUE_SIZE = 5
_DEFAULT_MAX_TOOL_ITERATIONS = 10

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # geometric shapes ext
    "\U0001F800-\U0001F8FF"  # supplemental arrows-C
    "\U0001F900-\U0001F9FF"  # supplemental pictographs
    "\U0001FA00-\U0001FA6F"  # chess / symbols
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs ext-A
    "\U00002600-\U000026FF"  # miscellaneous symbols
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U0000FE0F"             # variation selector-16
    "\U0000200D"             # zero-width joiner
    "]+",
    flags=re.UNICODE,
)


@dataclass
class PipelineDeps:
    """Long-lived collaborators the pipeline composes each turn."""

    stt: STTClient
    llm: LLMClient
    tts: TTSClient
    player: AudioPlayer
    tools: list[BaseTool]
    executor: ToolExecutor
    barge_in: BargeInListener | None = None


async def run_turn(
    session: Session,
    deps: PipelineDeps,
    config: dict[str, Any],
) -> bool:
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

    Returns:
        ``True`` iff this turn was interrupted by a barge-in (the user
        started speaking during playback). The caller should skip any
        "wait for user trigger" step and start the next turn's capture
        immediately. ``False`` on every other exit path (normal
        completion, errors, empty transcript).
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
        return False

    transcript = stt_result.get("text", "").strip()
    if not transcript:
        _log.info("turn_empty_transcript")
        return False
    session.add_message("user", transcript)

    tools_decl = build_tools(deps.tools)

    for _ in range(max_tool_iters):
        messages = build_messages(session, deps.tools)
        try:
            tool_call, assistant_text, barge_in_occurred = await _stream_with_tts(
                deps, messages, tools_decl, config
            )
        except ServiceUnavailableError as e:
            _log.warning("turn_stream_unavailable", error=str(e))
            await _speak_user_message(deps, e.user_message)
            session.add_message("assistant", e.user_message)
            return False

        if barge_in_occurred:
            if assistant_text:
                session.add_message("assistant", assistant_text)
            return True

        if tool_call is None:
            if assistant_text:
                session.add_message("assistant", assistant_text)
            return False

        # Persist the assistant turn with its structured tool_calls so the
        # next LLM iteration sees a coherent user → (assistant+tool_call) →
        # tool exchange. Without this the model retries the same call.
        session.add_message(
            "assistant",
            assistant_text,
            tool_calls=[
                {
                    "function": {
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                    }
                }
            ],
        )

        try:
            result = await asyncio.to_thread(deps.executor.execute, tool_call)
        except ToolExecutionError as e:
            _log.warning("tool_execution_failed", error=str(e))
            result = e.user_message
        session.add_message("tool", result, name=tool_call["name"])

    _log.warning("turn_max_tool_iterations_exceeded", limit=max_tool_iters)
    return False


async def _stream_with_tts(
    deps: PipelineDeps,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str, bool]:
    """Run the three-stage streaming pipeline for a single LLM stream.

    Returns ``(tool_call_or_none, full_assistant_text, barge_in_occurred)``.
    The assistant text is the verbatim concatenation of every spoken
    token the LLM emitted before tool invocation or completion. Tool
    calls travel through the structured event channel and never appear
    in the spoken text.

    Three concurrent tasks form the pipeline:

    * **producer** — reads the LLM event stream, feeds text through the
      :class:`SentenceSplitter`, and enqueues complete sentences. A
      :class:`ToolCallChunk` flips it into tool mode and ends the stream.
    * **synthesizer** — pulls each sentence, starts its TTS request as a
      pump task that streams audio into a per-sentence queue, and keeps
      exactly one synthesis in flight (one sentence of lookahead) so the
      next sentence's request overlaps the current sentence's playback.
    * **player** — keeps a single audio sink open for the whole turn and
      drains each sentence's audio queue into it back-to-back.

    When ``deps.barge_in`` is set, a background listener races against
    the pipeline. If the listener fires first, all three stages are
    cancelled, the player aborts the open sink, and the tuple's third
    element is ``True``.
    """
    queue_size = int(
        config.get("orchestrator", {}).get("queue_size", _DEFAULT_QUEUE_SIZE)
    )
    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=queue_size)
    # maxsize=1 → one sentence of synthesis lookahead: the synthesizer may
    # work on sentence N+1 while the player is still playing sentence N.
    audio_queue: asyncio.Queue[asyncio.Queue[bytes | None] | None] = asyncio.Queue(
        maxsize=1
    )
    accumulated: list[str] = []
    detected_tool: dict[str, Any] | None = None

    async def producer() -> None:
        nonlocal detected_tool
        splitter = SentenceSplitter()
        try:
            async for event in deps.llm.stream_chat(messages, tools=tools):
                if isinstance(event, ToolCallChunk):
                    detected_tool = {
                        "name": event.name,
                        "arguments": event.arguments,
                    }
                    return
                # TextChunk
                accumulated.append(event.text)
                for sentence in splitter.feed(event.text):
                    await _safe_put(sentence_queue, sentence)
            for sentence in splitter.flush():
                await _safe_put(sentence_queue, sentence)
        finally:
            await sentence_queue.put(None)

    async def pump(text: str, out: asyncio.Queue[bytes | None]) -> None:
        """Stream one sentence's TTS audio into ``out``, then sentinel it."""
        try:
            async for chunk in deps.tts.synthesize(text):
                await out.put(chunk)
        finally:
            await out.put(None)

    async def synthesizer() -> None:
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                await audio_queue.put(None)
                return
            chunk_q: asyncio.Queue[bytes | None] = asyncio.Queue()
            pump_task = asyncio.create_task(pump(sentence, chunk_q))
            try:
                # Publish before awaiting the pump so the player can drain
                # this sentence live while the pump is still producing it;
                # awaiting the pump keeps exactly one synthesis in flight.
                await audio_queue.put(chunk_q)
                await pump_task
            finally:
                if not pump_task.done():
                    pump_task.cancel()
                    await asyncio.gather(pump_task, return_exceptions=True)

    async def player() -> None:
        async with deps.player.session() as playback:
            while True:
                chunk_q = await audio_queue.get()
                if chunk_q is None:
                    return
                await playback.play(_drain(chunk_q))

    producer_task = asyncio.create_task(producer())
    synth_task = asyncio.create_task(synthesizer())
    player_task = asyncio.create_task(player())
    stages = (producer_task, synth_task, player_task)
    barge_in_task: asyncio.Task[None] | None = None
    if deps.barge_in is not None:
        barge_in_task = asyncio.create_task(deps.barge_in.wait_for_speech())

    barge_in_occurred = False
    try:
        barge_in_occurred = await _await_streaming(stages, barge_in_task)
        if barge_in_occurred:
            for t in stages:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*stages, return_exceptions=True)
        if barge_in_task is not None and not barge_in_task.done():
            barge_in_task.cancel()
            await asyncio.gather(barge_in_task, return_exceptions=True)
    except BaseException:
        for t in stages:
            if not t.done():
                t.cancel()
        if barge_in_task is not None and not barge_in_task.done():
            barge_in_task.cancel()
        extras: list[asyncio.Task[None]] = (
            [barge_in_task] if barge_in_task is not None else []
        )
        await asyncio.gather(*stages, *extras, return_exceptions=True)
        raise

    return detected_tool, "".join(accumulated), barge_in_occurred


async def _drain(chunk_q: asyncio.Queue[bytes | None]) -> AsyncIterator[bytes]:
    """Yield audio chunks from a per-sentence queue until its sentinel."""
    while True:
        chunk = await chunk_q.get()
        if chunk is None:
            return
        yield chunk


async def _await_streaming(
    stages: tuple[asyncio.Task[None], ...],
    barge_in_task: asyncio.Task[None] | None,
) -> bool:
    """Await all pipeline stages, racing them against the barge-in listener.

    Returns ``True`` iff the listener fired before the stages finished.
    Re-raises the first exception thrown by any stage — checked before
    each wait so a failed stage tears the turn down immediately even
    while other stages are still pending. Listener failures are logged
    and otherwise ignored — they must never tear down a turn that would
    otherwise complete normally.
    """
    while True:
        if barge_in_task is not None and barge_in_task.done():
            if barge_in_task.cancelled():
                pass
            elif (exc := barge_in_task.exception()) is not None:
                _log.warning("barge_in_listener_failed", error=str(exc))
            else:
                return True

        for t in stages:
            if t.done() and not t.cancelled() and t.exception() is not None:
                t.result()  # re-raises

        if all(t.done() for t in stages):
            return False

        watch: set[asyncio.Task[None]] = {t for t in stages if not t.done()}
        if barge_in_task is not None and not barge_in_task.done():
            watch.add(barge_in_task)
        await asyncio.wait(watch, return_when=asyncio.FIRST_COMPLETED)


async def _safe_put(queue: asyncio.Queue[str | None], sentence: str) -> None:
    """Strip emojis and drop unspoken sentences before queuing for TTS.

    A sentence with no alphanumeric character left after emoji removal
    (e.g. ``"😀😀😀."`` → ``"."``) is dropped — TTS pronouncing bare
    punctuation is worse than silence.
    """
    cleaned = _EMOJI_RE.sub("", sentence).strip()
    if not cleaned or not re.search(r"\w", cleaned):
        return
    await queue.put(cleaned)


async def _speak_user_message(deps: PipelineDeps, message: str) -> None:
    """Best-effort TTS of an error message; swallow further failures."""
    if not message.strip():
        return
    try:
        async with deps.player.session() as playback:
            await playback.play(deps.tts.synthesize(message))
    except ServiceUnavailableError as e:
        _log.error("error_tts_failed", message=message, error=str(e))
