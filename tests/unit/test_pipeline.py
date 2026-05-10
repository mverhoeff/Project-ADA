"""Unit tests for :mod:`orchestrator.pipeline`."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from core.exceptions import ServiceUnavailableError, ToolExecutionError
from orchestrator import pipeline
from orchestrator.pipeline import PipelineDeps, run_turn
from orchestrator.session import Session


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

    ``responses`` is a list of one response per ``stream_chat`` call.
    Each response is either a list of token strings (yielded one by one)
    or an Exception to raise.
    """

    def __init__(self, responses: list[list[str] | BaseException]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        # Snapshot messages so later mutations to session.history don't leak.
        self.calls.append(copy.deepcopy(messages))
        if not self._responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        for token in response:
            yield token


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


class FakePlayer:
    """AudioPlayer stand-in. Drains the chunk iterator and records calls."""

    def __init__(self) -> None:
        self.played: list[bytes] = []

    async def play(self, chunk_iter: AsyncIterator[bytes]) -> None:
        async for chunk in chunk_iter:
            self.played.append(chunk)


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
) -> PipelineDeps:
    return PipelineDeps(
        stt=stt,            # type: ignore[arg-type]
        llm=llm,            # type: ignore[arg-type]
        tts=tts,            # type: ignore[arg-type]
        player=player,      # type: ignore[arg-type]
        tools=[],
        executor=executor,  # type: ignore[arg-type]
    )


@pytest.fixture
def fake_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace mic capture with a synchronous fake returning fixed WAV bytes."""

    def _capture(config: dict[str, Any]) -> bytes:
        return b"FAKE_WAV"

    monkeypatch.setattr(pipeline, "capture_until_silence", _capture)


def _tool_call_token(name: str = "shell", arguments: dict[str, Any] | None = None) -> str:
    payload = {"name": name, "arguments": arguments or {}}
    return f"<tool_call>{json.dumps(payload)}</tool_call>"


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
            ["Searching now. ", _tool_call_token("shell", {"cmd": "echo"})],
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
    assert session.history == [
        {"role": "user", "content": "search for cats"},
        {
            "role": "assistant",
            "content": "Searching now. " + _tool_call_token("shell", {"cmd": "echo"}),
        },
        {"role": "tool", "content": "search-result"},
        {"role": "assistant", "content": "Done."},
    ]


@pytest.mark.asyncio
async def test_tool_call_xml_never_reaches_tts(fake_capture: None) -> None:
    stt = FakeSTT(text="run it")
    llm = FakeLLM(
        responses=[
            [_tool_call_token("shell", {})],
            ["All done."],
        ]
    )
    tts = FakeTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=["ok"])
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)
    session = Session()

    await run_turn(session, deps, _config())

    # The first LLM leg produced no spoken sentences (only a bare tool_call).
    assert tts.spoken == ["All done."]
    assert executor.calls == [{"name": "shell", "arguments": {}}]


@pytest.mark.asyncio
async def test_partial_tool_open_split_across_tokens_is_not_spoken(
    fake_capture: None,
) -> None:
    stt = FakeSTT(text="run it")
    # The opening tag is split across two tokens — the splitter must
    # never receive a partial "<tool" prefix.
    llm = FakeLLM(
        responses=[
            ["<to", "ol_call>", json.dumps({"name": "shell", "arguments": {}}), "</tool_call>"],
            ["Result is in."],
        ]
    )
    tts = FakeTTS()
    player = FakePlayer()
    executor = FakeExecutor(results=["ok"])
    deps = _build_deps(stt=stt, llm=llm, tts=tts, player=player, executor=executor)
    session = Session()

    await run_turn(session, deps, _config())

    assert tts.spoken == ["Result is in."]
    for spoken in tts.spoken:
        assert "<" not in spoken and "tool" not in spoken
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
                _tool_call_token("shell", {}),
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
async def test_tool_execution_error_feeds_user_message_back_to_llm(
    fake_capture: None,
) -> None:
    stt = FakeSTT(text="run it")
    llm = FakeLLM(
        responses=[
            [_tool_call_token("shell", {})],
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
    }
    assert tts.spoken == ["All set."]


@pytest.mark.asyncio
async def test_safety_cap_terminates_runaway_tool_loop(fake_capture: None) -> None:
    stt = FakeSTT(text="loop forever")
    # Every LLM response is yet another tool call.
    tool_token = _tool_call_token("shell", {})
    llm = FakeLLM(responses=[[tool_token]] * 20)
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
                _tool_call_token("shell", {}),
            ],
            ["Done."],
        ]
    )
    tts = FakeTTS()
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
