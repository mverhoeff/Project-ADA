"""Conversation session state for one assistant process.

The session is held in memory only and resets when the process restarts.
It owns the message history that is sent to the LLM each turn and a few
flags the orchestrator reads to coordinate the streaming pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    """In-memory conversation state.

    Attributes:
        history: Ordered chat history as ``{"role": str, "content": str}``
            dicts, matching the shape Ollama's /api/chat expects.
        turn_count: Number of completed user/assistant exchanges. Managed
            by the orchestrator, not by :meth:`add_message`.
        active_tool: Name of the tool currently executing, or ``None``.
        is_speaking: ``True`` while the consumer task is playing audio.
        platform: Host OS as returned by :func:`ada_platform.detect.current_platform`.
    """

    history: list[dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    active_tool: str | None = None
    is_speaking: bool = False
    platform: str = ""

    def add_message(self, role: str, content: str, **extra: Any) -> None:
        """Append a message to the history.

        Extra keyword arguments (e.g. ``tool_calls=[...]``, ``name="..."``)
        are merged into the message dict so callers can record Ollama
        chat-format fields beyond the basic ``role``/``content`` pair.
        """
        msg: dict[str, Any] = {"role": role, "content": content}
        msg.update(extra)
        self.history.append(msg)

    def trim_to(self, max_turns: int) -> None:
        """Drop oldest non-system messages until at most ``max_turns`` pairs remain.

        A *turn* is one user message plus its assistant reply, so the
        non-system tail is capped at ``max_turns * 2`` entries. System
        messages at the head of the history are always preserved.

        Args:
            max_turns: Maximum number of user/assistant turn pairs to keep.

        Raises:
            ValueError: If ``max_turns`` is negative.
        """
        if max_turns < 0:
            raise ValueError("max_turns must be non-negative")

        target = max_turns * 2
        system_prefix_end = next(
            (i for i, m in enumerate(self.history) if m.get("role") != "system"),
            len(self.history),
        )
        non_system = self.history[system_prefix_end:]
        if len(non_system) > target:
            keep = non_system[-target:] if target > 0 else []
            self.history = self.history[:system_prefix_end] + keep
