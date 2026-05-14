"""Typed events yielded by :meth:`LLMClient.stream_chat`.

Ollama returns both spoken content and structured tool calls in its
streaming response. The orchestrator needs to route them to different
sinks (text → sentence splitter → TTS; tool calls → executor), so the
client yields a discriminated union of small immutable dataclasses
rather than raw strings. ``isinstance`` dispatch keeps the producer
loop in :mod:`orchestrator.pipeline` short and lets the type checker
narrow event variants automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A piece of spoken assistant text from the LLM stream."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallChunk:
    """A structured tool invocation parsed from the LLM stream."""

    name: str
    arguments: dict[str, Any]


StreamEvent = TextChunk | ToolCallChunk
