"""Unit tests for :mod:`orchestrator.pipeline`."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, ClassVar

import pytest

from agent.tools.base import BaseTool
from core.exceptions import ServiceUnavailableError, ToolExecutionError
from orchestrator import pipeline
from orchestrator.pipeline import PipelineDeps, run_turn
from orchestrator.session import Session
from services.llm.events import StreamEvent, TextChunk, ToolCallChunk


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSTT:
    """STTClient stand-in. Returns a configurable transcript or raises."""

    def __init__(
        self,
        text: str = "",
        error: ServiceUnavailableError | None = None,
    ) -> None:
        self._text = text
        self._error = error
        self.calls: list[bytes] = []

    async def transcribe(self, wav_bytes: bytes) -> dict[str, Any]:
        self.calls.append(wav_bytes)
        if self._error is not None:
            raise self._error
        return {"text": self._text}


class FakeLLM:
    """LLMClient stand-in.

    ``responses`` is a list of one response per ``stream_chat`` call. Each
    response is either a list of items or an Exception to raise. Items may
    be raw ``str`` (auto-wrapped as :class:`TextChunk` for convenience) or
    pre-built :class:`StreamEvent` instances.
    """

    def __init__(
        self,
        responses: list[list[str | StreamEvent] | BaseException],
    ) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []
        self.tools_calls: list[list[dict[str, Any]] | None] = []

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        # Snapshot messages so later mutations to session.history don't leak.
        self.calls.append(copy.deepcopy(messages))
        self.tools_calls.append(copy.deepcopy(tools))
        if not self._responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        for item in response:
            if isinstance(item, (TextChunk, ToolCallChunk)):
                yield item
            else:
                yield TextChunk(text=item)


class FakeTTS:
    """TTSClient stand-in. Records every text passed to ``synthesize``."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ) -> AsyncIterator[bytes]:
        self.spoken.append(text)
        # Yield a token marker so player.play has something to consume.
        yield b"audio:" + text.encode()


class _FakePlayerSession:
    """Playback-session stand-in. Drains each sentence's chunk iterator.

    Yields to the event loop after each sentence so concurrent tasks
    (notably the barge-in listener) get a turn between TTS calls. The
    real ``_PlayerSession.play`` has many await points; this fake
    compresses them into a single tail await without affecting test
    assertions.
    """

    def __init__(self, sink: list[bytes]) -> None:
        self._sink = sink

    async def play(self, chunk_iter: AsyncIterator[bytes]) -> None:
        async for chunk in chunk_iter:
            self._sink.append(chunk)
        await asyncio.sleep(0)


class FakePlayer:
    """AudioPlayer stand-in. Opens a session and records every played chunk."""

    def __init__(self) -> None:
        self.played: list[bytes] = []

    @asynccontextmanager
    async def session(self) -> AsyncIterator[_FakePlayerSession]:
        yield _FakePlayerSession(self.played)


class FakeExecutor:
    """ToolExecutor stand-in.

    ``results`` is a list of one item per ``execute`` call: either a
    string result or an exception to raise.
    """

    def __init__(self, results: list[str | BaseException]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def execute(self, tool_call: dict[str, Any]) -> str:
        self.calls.append(tool_call)
        if not self._results:
            raise AssertionError("FakeExecutor ran out of scripted results")
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeBargeInListener:
    """BargeInListener stand-in.

    ``wait_for_speech`` blocks on an ``asyncio.Event``; tests trigger it
    by calling :meth:`trigger` (often from inside another fake to keep
    the timing relative to streaming progress).
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def trigger(self) -> None:
        self._event.set()

    async def wait_for_speech(self) -> None:
        await self._event.wait()


def _config() -> dict[str, Any]:
    return {
        "audio": {"output_device": None},
        "orchestrator": {"queue_size": 5, "max_tool_iterations": 4},
    }


def _build_deps(
    *,
    stt: FakeSTT,
    llm: FakeLLM,
    tts: FakeTTS,
    player: FakePlayer,
    executor: FakeExecutor,
    tools: list[BaseTool] | None = None,
    barge_in: FakeBargeInListener | None = None,
) -> PipelineDeps:
    return PipelineDeps(
        stt=stt,            # type: ignore[arg-type]
        llm=llm,            # type: ignore[arg-type]
        tts=tts,            # type: ignore[arg-type]
        player=player,      # type: ignore[arg-type]
        tools=tools or [],
        executor=executor,  # type: ignore[arg-type]
        barge_in=barge_in,  # type: ignore[arg-type]
    )


@pytest.fixture
def fake_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace mic capture with a synchronous fake returning fixed WAV bytes."""

    def _capture(config: dict[str, Any]) -> bytes:
        return b"FAKE_WAV"

    monkeypatch.setattr(pipeline, "capture_until_silence", _capture)


def _tool_call_event(
    name: str = "shell",
    arguments: dict[str, Any] | None = None,
) -> ToolCallChunk:
    return ToolCallChunk(name=name, arguments=arguments or {})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plain_text_turn_plays_each_sentence_and_updates_history(
    fake_capture: None,
) -> None:
    stt = FakeSTT(text="hi there")
    llm = FakeLLM(responses=[["Hi! ", "How ", "are ", "you?"]])
    tts = FakeTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=[])
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)
    session = Session()

    await run_turn(session, deps, _config())

    assert tts.spoken == ["Hi!", "How are you?"]
    assert len(player.played) == 2
    assert session.history == [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "Hi! How are you?"},
    ]
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_empty_transcript_skips_llm(fake_capture: None) -> None:
    stt = FakeSTT(text="   \t\n  ")
    llm = FakeLLM(responses=[])
    tts = FakeTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=[])
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)
    session = Session()

    await run_turn(session, deps, _config())

    assert llm.calls == []
    assert tts.spoken == []
    assert session.history == []


@pytest.mark.asyncio
async def test_tool_call_path_executes_tool_and_resumes_stream(
    fake_capture: None,
) -> None:
    stt = FakeSTT(text="search for cats")
    llm = FakeLLM(
        responses=[
            ["Searching now. ", _tool_call_event("shell", {"cmd": "echo"})],
            ["Done."],
        ]
    )
    tts = FakeTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=["search-result"])
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)
    session = Session()

    await run_turn(session, deps, _config())

    # Acknowledgement spoken before tool ran, then "Done." spoken after.
    assert tts.spoken == ["Searching now.", "Done."]
    assert executor.calls == [{"name": "shell", "arguments": {"cmd": "echo"}}]

    # History captures both LLM legs and the tool result in between.
    # The assistant message records the structured tool_calls so the second
    # LLM call sees a coherent exchange and doesn't retry the same tool.
    assert session.history == [
        {"role": "user", "content": "search for cats"},
        {
            "role": "assistant",
            "content": "Searching now. ",
            "tool_calls": [
                {"function": {"name": "shell", "arguments": {"cmd": "echo"}}}
            ],
        },
        {"role": "tool", "content": "search-result", "name": "shell"},
        {"role": "assistant", "content": "Done."},
    ]


@pytest.mark.asyncio
async def test_second_llm_call_sees_structured_tool_exchange(
    fake_capture: None,
) -> None:
    """Regression: without the tool_calls field on the assistant message,
    the LLM cannot tell that the tool message answers its own request and
    loops on the same call until max_tool_iterations trips."""
    stt = FakeSTT(text="search please")
    llm = FakeLLM(
        responses=[
            ["Searching. ", _tool_call_event("web_search", {"query": "cats"})],
            ["Found some cats."],
        ]
    )
    tts = FakeTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=["1. cat result"])
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)
    session = Session()

    await run_turn(session, deps, _config())

    second_messages = llm.calls[1]
    # The assistant message right before the tool message must carry the
    # tool_calls field, otherwise Ollama's chat template can't pair them.
    assert second_messages[-2] == {
        "role": "assistant",
        "content": "Searching. ",
        "tool_calls": [
            {"function": {"name": "web_search", "arguments": {"query": "cats"}}}
        ],
    }
    assert second_messages[-1] == {
        "role": "tool",
        "content": "1. cat result",
        "name": "web_search",
    }


@pytest.mark.asyncio
async def test_bare_tool_call_event_produces_no_tts(fake_capture: None) -> None:
    stt = FakeSTT(text="run it")
    llm = FakeLLM(
        responses=[
            [_tool_call_event("shell", {})],
            ["All done."],
        ]
    )
    tts = FakeTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=["ok"])
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)
    session = Session()

    await run_turn(session, deps, _config())

    # The first LLM leg produced no spoken sentences (only a bare tool call).
    assert tts.spoken == ["All done."]
    assert executor.calls == [{"name": "shell", "arguments": {}}]


@pytest.mark.asyncio
async def test_acknowledgement_text_is_split_normally(fake_capture: None) -> None:
    stt = FakeSTT(text="please search")
    llm = FakeLLM(
        responses=[
            [
                "Searching",
                " the",
                " web",
                " now",
                ".",
                " ",
                _tool_call_event("shell", {}),
            ],
            ["Found nothing."],
        ]
    )
    tts = FakeTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=["ok"])
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)
    session = Session()

    await run_turn(session, deps, _config())

    assert tts.spoken == ["Searching the web now.", "Found nothing."]
    assert executor.calls == [{"name": "shell", "arguments": {}}]


@pytest.mark.asyncio
async def test_stt_error_speaks_user_message_and_aborts(fake_capture: None) -> None:
    err = ServiceUnavailableError("STT down", "I can't hear you right now.")
    stt = FakeSTT(error=err)
    llm = FakeLLM(responses=[])
    tts = FakeTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=[])
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)
    session = Session()

    await run_turn(session, deps, _config())

    assert tts.spoken == ["I can't hear you right now."]
    assert llm.calls == []
    # No transcript ever entered history, so no user/assistant message.
    assert session.history == []


@pytest.mark.asyncio
async def test_llm_error_mid_stream_speaks_user_message(fake_capture: None) -> None:
    err = ServiceUnavailableError("LLM down", "I can't reach my language model right now.")
    stt = FakeSTT(text="hello")
    llm = FakeLLM(responses=[err])
    tts = FakeTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=[])
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)
    session = Session()

    await run_turn(session, deps, _config())

    assert tts.spoken == ["I can't reach my language model right now."]
    assert session.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "I can't reach my language model right now."},
    ]


@pytest.mark.asyncio
async def test_tts_error_mid_stream_speaks_user_message(fake_capture: None) -> None:
    """A ServiceUnavailableError from the TTS stream surfaces through the
    synthesizer stage and is spoken back to the user."""
    err = ServiceUnavailableError("TTS down", "I can't speak right now.")
    stt = FakeSTT(text="hello")
    llm = FakeLLM(responses=[["Hello there."]])

    class FailingTTS(FakeTTS):
        """Raises a TTS ServiceUnavailableError as soon as synthesis starts."""

        async def synthesize(
            self,
            text: str,
            voice: str | None = None,
            speed: float | None = None,
        ) -> AsyncIterator[bytes]:
            self.spoken.append(text)
            raise err
            yield b""  # unreachable — present so this is an async generator

    tts = FailingTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=[])
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)
    session = Session()

    result = await run_turn(session, deps, _config())

    assert result is False
    assert session.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "I can't speak right now."},
    ]
    # "Hello there." was attempted; the error message was spoken afterwards.
    assert tts.spoken == ["Hello there.", "I can't speak right now."]


@pytest.mark.asyncio
async def test_tool_execution_error_feeds_user_message_back_to_llm(
    fake_capture: None,
) -> None:
    stt = FakeSTT(text="run it")
    llm = FakeLLM(
        responses=[
            [_tool_call_event("shell", {})],
            ["All set."],
        ]
    )
    tts = FakeTTS()
    player = FakePlayer()
    err = ToolExecutionError("boom", "I encountered an error running shell.")
    executor = FakeExecutor(results=[err])
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)
    session = Session()

    await run_turn(session, deps, _config())

    # The second LLM call's messages must end with the tool error message
    # so the LLM can incorporate it into its reply.
    assert len(llm.calls) == 2
    second_messages = llm.calls[1]
    assert second_messages[-1] == {
        "role": "tool",
        "content": "I encountered an error running shell.",
        "name": "shell",
    }
    assert tts.spoken == ["All set."]


@pytest.mark.asyncio
async def test_safety_cap_terminates_runaway_tool_loop(fake_capture: None) -> None:
    stt = FakeSTT(text="loop forever")
    # Every LLM response is yet another tool call.
    llm = FakeLLM(responses=[[_tool_call_event("shell", {})]] * 20)
    tts = FakeTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=["ok"] * 20)
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)
    session = Session()

    await run_turn(session, deps, _config())

    # max_tool_iterations is 4 in _config(); the LLM is invoked exactly that many times.
    assert len(llm.calls) == 4
    assert len(executor.calls) == 4


@pytest.mark.asyncio
async def test_consumer_drains_queue_before_tool_runs(fake_capture: None) -> None:
    stt = FakeSTT(text="search please")
    llm = FakeLLM(
        responses=[
            [
                "First sentence. ",
                "Second sentence. ",
                "Third sentence. ",
                _tool_call_event("shell", {}),
            ],
            ["Done."],
        ]
    )
    player = FakePlayer()

    # Record relative ordering of TTS calls vs executor calls.
    event_log: list[str] = []

    class OrderingTTS(FakeTTS):
        async def synthesize(
            self,
            text: str,
            voice: str | None = None,
            speed: float | None = None,
        ) -> AsyncIterator[bytes]:
            event_log.append(f"tts:{text}")
            async for chunk in super().synthesize(text, voice, speed):
                yield chunk

    class OrderingExecutor(FakeExecutor):
        def execute(self, tool_call: dict[str, Any]) -> str:
            event_log.append("exec")
            return super().execute(tool_call)

    tts = OrderingTTS()
    executor = OrderingExecutor(results=["ok"])
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)
    session = Session()

    await run_turn(session, deps, _config())

    # All three acknowledgement sentences must have been spoken before
    # the tool executor was invoked.
    exec_index = event_log.index("exec")
    pre_exec = event_log[:exec_index]
    assert pre_exec == [
        "tts:First sentence.",
        "tts:Second sentence.",
        "tts:Third sentence.",
    ]
    # And the post-tool reply was spoken last.
    assert event_log[-1] == "tts:Done."


# ---------------------------------------------------------------------------
# Tool declarations & emoji stripping
# ---------------------------------------------------------------------------


class _FakeSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web."
    schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def execute(self, params: dict[str, Any]) -> str:
        return ""


@pytest.mark.asyncio
async def test_tools_forwarded_to_llm(fake_capture: None) -> None:
    stt = FakeSTT(text="hi")
    llm = FakeLLM(responses=[["Hello."]])
    tts = FakeTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=[])
    deps = _build_deps(
        stt=stt,
        llm=llm,
        tts=tts,
        player=player,
        executor=executor,
        tools=[_FakeSearchTool()],
    )

    await run_turn(Session(), deps, _config())

    assert llm.tools_calls == [
        [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web.",
                    "parameters": _FakeSearchTool.schema,
                },
            }
        ]
    ]


@pytest.mark.asyncio
async def test_emoji_is_stripped_before_tts(fake_capture: None) -> None:
    stt = FakeSTT(text="hi")
    llm = FakeLLM(responses=[["Hello! \U0001F44B ", "How are you? \U0001F31F"]])
    tts = FakeTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=[])
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)

    await run_turn(Session(), deps, _config())

    assert tts.spoken == ["Hello!", "How are you?"]
    for spoken in tts.spoken:
        assert "\U0001F44B" not in spoken
        assert "\U0001F31F" not in spoken


@pytest.mark.asyncio
async def test_pure_emoji_sentence_is_dropped(fake_capture: None) -> None:
    stt = FakeSTT(text="hi")
    llm = FakeLLM(responses=[["\U0001F600\U0001F600\U0001F600. "]])
    tts = FakeTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=[])
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)

    await run_turn(Session(), deps, _config())

    assert tts.spoken == []


# ---------------------------------------------------------------------------
# Barge-in
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turn_returns_false_on_normal_completion(
    fake_capture: None,
) -> None:
    stt = FakeSTT(text="hi")
    llm = FakeLLM(responses=[["Hello."]])
    tts = FakeTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=[])
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)

    result = await run_turn(Session(), deps, _config())
    assert result is False


@pytest.mark.asyncio
async def test_barge_in_cancels_remaining_playback_and_returns_true(
    fake_capture: None,
) -> None:
    stt = FakeSTT(text="tell me a long story")
    llm = FakeLLM(
        responses=[
            ["First. ", "Second. ", "Third. ", "Fourth. ", "Fifth."],
        ]
    )
    player = FakePlayer()
    executor = FakeExecutor(results=[])
    barge_in = FakeBargeInListener()

    class TriggeringTTS(FakeTTS):
        """Fires the barge-in event when the second sentence is synthesized."""

        def __init__(self, listener: FakeBargeInListener) -> None:
            super().__init__()
            self._listener = listener

        async def synthesize(
            self,
            text: str,
            voice: str | None = None,
            speed: float | None = None,
        ) -> AsyncIterator[bytes]:
            self.spoken.append(text)
            if len(self.spoken) == 2:
                self._listener.trigger()
                # Let the event loop service barge_in_task before this
                # synthesize call finishes, so the listener wins the race
                # against the next queue.get().
                await asyncio.sleep(0)
            yield b"audio:" + text.encode()

    tts = TriggeringTTS(barge_in)
    deps = _build_deps(
        stt=stt, llm=llm, tts=tts, player=player, executor=executor, barge_in=barge_in
    )
    session = Session()

    result = await run_turn(session, deps, _config())

    assert result is True
    # The trigger fires while the 2nd sentence is being synthesized — that
    # sentence still completes (no cancellation point inside the fake
    # synthesize/play), but the 3rd onwards must never reach TTS.
    assert tts.spoken == ["First.", "Second."]
    # History captures the user message plus whatever the producer
    # accumulated before cancellation — at minimum the first two sentences.
    assert session.history[0] == {"role": "user", "content": "tell me a long story"}
    assert len(session.history) == 2
    assistant_msg = session.history[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"].startswith("First. Second.")


@pytest.mark.asyncio
async def test_barge_in_before_tool_call_skips_tool_execution(
    fake_capture: None,
) -> None:
    stt = FakeSTT(text="search")
    llm = FakeLLM(
        responses=[
            ["Working. ", _tool_call_event("shell", {})],
            ["Should never run."],
        ]
    )
    player = FakePlayer()
    executor = FakeExecutor(results=["unused"])
    barge_in = FakeBargeInListener()

    class TriggeringTTS(FakeTTS):
        def __init__(self, listener: FakeBargeInListener) -> None:
            super().__init__()
            self._listener = listener

        async def synthesize(
            self,
            text: str,
            voice: str | None = None,
            speed: float | None = None,
        ) -> AsyncIterator[bytes]:
            self.spoken.append(text)
            self._listener.trigger()
            await asyncio.sleep(0)
            yield b"audio:" + text.encode()

    tts = TriggeringTTS(barge_in)
    deps = _build_deps(
        stt=stt, llm=llm, tts=tts, player=player, executor=executor, barge_in=barge_in
    )
    session = Session()

    result = await run_turn(session, deps, _config())

    assert result is True
    # The acknowledgement spoke, but the tool was never invoked and the
    # second LLM call was never made.
    assert tts.spoken == ["Working."]
    assert executor.calls == []
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_no_barge_in_listener_preserves_existing_behavior(
    fake_capture: None,
) -> None:
    """Smoke test: with barge_in=None, run_turn returns False as before."""
    stt = FakeSTT(text="hi")
    llm = FakeLLM(responses=[["Hi! ", "How are you?"]])
    tts = FakeTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=[])
    deps = _build_deps(
        stt=stt, llm=llm, tts=tts, player=player, executor=executor, barge_in=None
    )

    result = await run_turn(Session(), deps, _config())
    assert result is False
    assert tts.spoken == ["Hi!", "How are you?"]
