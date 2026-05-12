"""Assemble the messages list sent to Ollama on each turn.

Builds the system prompt (persona + tool declarations + date + platform)
and prepends it to the session history. Pure function — no I/O, no async.
"""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.tools.base import BaseTool
    from orchestrator.session import Session


_PERSONA = (
    "You are ADA, a helpful local voice assistant. Reply concisely and "
    "naturally; your replies are spoken aloud, so avoid markdown, code "
    "fences, or long lists unless the user explicitly asks for them. "
    "When you decide to use a tool, first say in one short sentence what "
    "you are about to do (for example, \"Searching the web now...\"), "
    "then emit the tool_call JSON."
)


def build_messages(
    session: Session,
    tools: list[BaseTool] | None = None,
) -> list[dict[str, Any]]:
    """Build the full ``messages`` list to send to the LLM.

    Args:
        session: Current conversation session. Its ``history`` is appended
            verbatim after the system message.
        tools: Tools to declare to the LLM. ``None`` or empty omits the
            tool block entirely.

    Returns:
        A list whose first element is the system message and whose
        remaining elements are the session history in order.
    """
    parts = [
        _PERSONA,
        f"Today's date: {date.today().isoformat()}",
        f"Operating system: {session.platform or 'unknown'}",
    ]

    if tools:
        parts.append("Available tools (respond with a tool_use JSON block to invoke):")
        for tool in tools:
            declaration = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema,
                },
            }
            parts.append(json.dumps(declaration))

    system_content = "\n".join(parts)
    return [{"role": "system", "content": system_content}, *session.history]
