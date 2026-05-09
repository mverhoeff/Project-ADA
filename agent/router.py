"""Tool call detection from accumulated LLM output.

Qwen3 emits tool calls as ``<tool_call>{"name": ..., "arguments": ...}</tool_call>``
blocks within its content stream when tool declarations are injected into the
system prompt. The router scans the lookahead buffer for this pattern and
returns the parsed call so the orchestrator can switch to the tool path.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def detect_tool_call(text: str) -> dict[str, Any] | None:
    """Detect and parse a Qwen3 tool call from an LLM output buffer.

    The buffer typically begins with a spoken acknowledgement phrase
    ("Searching the web now…") that the splitter flushes to TTS first; the
    tool call block follows. The function returns ``None`` for any malformed
    or partial input — the caller may accumulate more tokens and retry.

    Args:
        text: Accumulated LLM output, possibly containing leading plain text.

    Returns:
        Parsed tool call dict with at least a ``"name"`` key, or ``None`` if
        no valid tool call block is found.
    """
    match = _TOOL_CALL_RE.search(text)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if "name" not in parsed:
        return None
    return parsed
