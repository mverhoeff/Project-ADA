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
    """Run the Producer/Consumer pair for a single LLM stream.

    Returns ``(tool_call_or_none, full_assistant_text, barge_in_occurred)``.
    The assistant text is the verbatim concatenation of every spoken
    token the LLM emitted before tool invocation or completion. Tool
    calls travel through the structured event channel and never appear
    in the spoken text.

    When ``deps.barge_in`` is set, a background listener races against
    the streaming pair. If the listener fires before streaming
    completes, the producer and consumer are cancelled, ``AudioPlayer``
    aborts the active sink, and the tuple's third element is ``True``.
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
    barge_in_task: asyncio.Task[None] | None = None
    if deps.barge_in is not None:
        barge_in_task = asyncio.create_task(deps.barge_in.wait_for_speech())

    barge_in_occurred = False
    try:
        barge_in_occurred = await _await_streaming(
            producer_task, consumer_task, barge_in_task
        )
        if barge_in_occurred:
            for t in (producer_task, consumer_task):
                if not t.done():
                    t.cancel()
            await asyncio.gather(
                producer_task, consumer_task, return_exceptions=True
            )
        if barge_in_task is not None and not barge_in_task.done():
            barge_in_task.cancel()
            await asyncio.gather(barge_in_task, return_exceptions=True)
    except BaseException:
        for t in (producer_task, consumer_task):
            if not t.done():
                t.cancel()
        if barge_in_task is not None and not barge_in_task.done():
            barge_in_task.cancel()
        extras: list[asyncio.Task[None]] = (
            [barge_in_task] if barge_in_task is not None else []
        )
        await asyncio.gather(
            producer_task, consumer_task, *extras, return_exceptions=True
        )
        raise

    return detected_tool, "".join(accumulated), barge_in_occurred


async def _await_streaming(
    producer_task: asyncio.Task[None],
    consumer_task: asyncio.Task[None],
    barge_in_task: asyncio.Task[None] | None,
) -> bool:
    """Await the producer + consumer, racing them against the listener.

    Returns ``True`` iff the listener fired before streaming finished.
    Re-raises any exception thrown by producer or consumer. Listener
    failures are logged and otherwise ignored — they must never tear
    down a turn that would otherwise complete normally.
    """
    while not (producer_task.done() and consumer_task.done()):
        watch: set[asyncio.Future[Any]] = {producer_task, consumer_task}
        if barge_in_task is not None and not barge_in_task.done():
            watch.add(barge_in_task)
        await asyncio.wait(watch, return_when=asyncio.FIRST_COMPLETED)

        if barge_in_task is not None and barge_in_task.done():
            if barge_in_task.cancelled():
                pass
            elif (exc := barge_in_task.exception()) is not None:
                _log.warning("barge_in_listener_failed", error=str(exc))
            else:
                return True

        for t in (producer_task, consumer_task):
            if t.done() and not t.cancelled() and t.exception() is not None:
                t.result()  # re-raises

    return False


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
        await deps.player.play(deps.tts.synthesize(message))
    except ServiceUnavailableError as e:
        _log.error("error_tts_failed", message=message, error=str(e))
