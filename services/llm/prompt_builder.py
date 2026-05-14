"""Assemble the messages list and tool declarations sent to Ollama.

:func:`build_messages` returns the system prompt (persona + date + platform)
prepended to the session history. :func:`build_tools` returns Ollama-format
function declarations to pass alongside via the ``/api/chat`` ``tools``
field; tool calls then arrive in the response's structured ``tool_calls``
field rather than embedded in content. Pure functions — no I/O, no async.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.tools.base import BaseTool
    from orchestrator.session import Session


_PERSONA = (
    "You are ADA, a helpful local voice assistant. Reply concisely and "
    "naturally; your replies are spoken aloud, so avoid markdown, code "
    "fences, emojis, or long lists unless the user explicitly asks for them. "
    "Never include emoji characters in your responses. "
    "Never read out URLs, web addresses, or file paths aloud — they are "
    "unpronounceable; name a website instead of reciting its address (say "
    "\"Wikipedia\", not its link). "
    "When you must enumerate items, speak them naturally — join them with "
    "commas and \"and\" (e.g. \"apples, oranges, and bananas\"), or use "
    "phrases like \"first\", \"second\", \"third\". Never output bulleted "
    "or numbered lines. "
    "When you need to use a tool, first say in one short sentence what you "
    "are about to do (for example, \"Searching the web now...\"), then "
    "invoke the tool via the native tool-calling interface — do not write "
    "tool calls as text. "
    "After a tool returns, answer the user's question directly from what you "
    "found — give them the result, not a tour of where it came from. Do not "
    "recite result titles, sources, or links unless the user explicitly asks "
    "where the information came from."
)


def build_messages(
    session: Session,
    tools: list[BaseTool] | None = None,  # noqa: ARG001 - kept for callsite symmetry
) -> list[dict[str, Any]]:
    """Build the full ``messages`` list to send to the LLM.

    Args:
        session: Current conversation session. Its ``history`` is appended
            verbatim after the system message.
        tools: Accepted for callsite symmetry with :func:`build_tools` but
            no longer embedded in the system prompt — tool declarations
            travel through Ollama's native ``tools`` request field.

    Returns:
        A list whose first element is the system message and whose
        remaining elements are the session history in order.
    """
    parts = [
        _PERSONA,
        f"Today's date: {date.today().isoformat()}",
        f"Operating system: {session.platform or 'unknown'}",
    ]
    system_content = "\n".join(parts)
    return [{"role": "system", "content": system_content}, *session.history]


def build_tools(
    tools: list[BaseTool] | None,
) -> list[dict[str, Any]] | None:
    """Convert ``BaseTool`` instances into Ollama-format function declarations.

    Args:
        tools: Registered tools, or ``None``/empty if the LLM should be
            invoked without tool access.

    Returns:
        A list of ``{"type": "function", "function": {...}}`` entries
        suitable for the Ollama ``/api/chat`` ``tools`` field, or ``None``
        when there are no tools (so the client omits the field entirely).
    """
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema,
            },
        }
        for t in tools
    ]
