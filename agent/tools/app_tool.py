"""Agent tool that launches desktop applications by display name.

The actual platform-specific launching is delegated to
:mod:`ada_platform.launcher` so this module stays a thin :class:`BaseTool`
adapter.
"""

from __future__ import annotations

from typing import Any, ClassVar

from agent.tools.base import BaseTool


class OpenAppTool(BaseTool):
    """Launch a desktop application by its display name.

    The LLM picks this tool when the user says things like "open Spotify"
    or "launch Firefox". Failures (app not found, launcher missing) bubble
    up as :class:`core.exceptions.ToolExecutionError` and are spoken aloud
    via the existing TTS error path.
    """

    name = "open_app"
    description = (
        "Launch a desktop application by its display name "
        "(e.g. 'Spotify', 'Firefox', 'Notepad'). "
        "Use this whenever the user asks to open, launch, or start an app."
    )
    schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Application name as it appears in the Start menu or app list.",
            },
        },
        "required": ["name"],
    }

    def execute(self, params: dict[str, Any]) -> str:
        from ada_platform.launcher import open_app

        name = params["name"]
        open_app(name)
        return f"Opened {name}."
