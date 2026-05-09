"""Concrete tools for shell execution, system info, and file reading.

Each class is a separate :class:`BaseTool` subclass. Tools that need
configuration accept the full config dict in ``__init__`` and read their own
sub-section. Tools raise :class:`ToolExecutionError` on policy violations
(deny-list, path restriction) and on filesystem or process errors that the
LLM should surface to the user.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import psutil

from agent.tools.base import BaseTool
from core.exceptions import ToolExecutionError

_DENY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"rm\s+-[^\s]*r", re.IGNORECASE),
    re.compile(r"mkfs", re.IGNORECASE),
    re.compile(r"dd\s+if=", re.IGNORECASE),
    re.compile(r":\(\)\s*\{", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\bformat\b", re.IGNORECASE),
    re.compile(r"deltree", re.IGNORECASE),
    re.compile(r"rmdir\s+/s", re.IGNORECASE),
)


class ShellTool(BaseTool):
    """Run a shell command and return combined stdout/stderr.

    Dangerous commands (recursive deletes, disk wipes, fork bombs, shutdown,
    reboot, format) are blocked by a regex deny-list before any subprocess
    call is made. Non-zero exit codes are NOT treated as errors — their
    output is returned to the LLM as-is.

    Args:
        config: Full config dict; reads ``agent.shell_timeout_seconds``.
    """

    name = "run_shell"
    description = "Run a shell command and return its stdout and stderr."
    schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Optional override for the command timeout.",
            },
        },
        "required": ["command"],
    }

    def __init__(self, config: dict[str, Any]) -> None:
        self._timeout: int = int(config.get("agent", {}).get("shell_timeout_seconds", 30))

    def execute(self, params: dict[str, Any]) -> str:
        command = params["command"]
        timeout = int(params.get("timeout_seconds", self._timeout))

        for pattern in _DENY_PATTERNS:
            if pattern.search(command):
                raise ToolExecutionError(
                    f"Command blocked by deny-list: {command!r}",
                    "That command is not allowed for safety reasons.",
                )

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise ToolExecutionError(
                f"Command timed out after {timeout}s: {command!r}",
                "The command took too long and was cancelled.",
            ) from None

        return (result.stdout + result.stderr).strip()


class SystemInfoTool(BaseTool):
    """Return current CPU, memory, and disk usage as a JSON string."""

    name = "get_system_info"
    description = "Return current CPU, memory, and disk usage percentages."
    schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, params: dict[str, Any]) -> str:
        info = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
        }
        return json.dumps(info)


class FileReadTool(BaseTool):
    """Read a UTF-8 text file from an allow-listed directory.

    The target path is fully resolved (``~`` expanded, ``..`` collapsed,
    symlinks followed) before being checked against the configured allowed
    directories. This blocks both relative-path traversal and symlink
    attacks pointing outside the allow-list.

    Args:
        config: Full config dict; reads ``agent.allowed_paths``.
    """

    name = "read_file"
    description = "Read a UTF-8 text file from an allow-listed directory."
    schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or ~-relative path to the file.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, config: dict[str, Any]) -> None:
        raw_paths = config.get("agent", {}).get("allowed_paths", []) or []
        self._allowed: list[Path] = [Path(p).expanduser().resolve() for p in raw_paths]

    def execute(self, params: dict[str, Any]) -> str:
        target = Path(params["path"]).expanduser().resolve()

        if not any(target.is_relative_to(allowed) for allowed in self._allowed):
            raise ToolExecutionError(
                f"Path {target} is outside allowed directories.",
                "I'm not allowed to read files outside approved directories.",
            )

        try:
            return target.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ToolExecutionError(
                f"File not found: {target}",
                "That file does not exist.",
            ) from exc
        except PermissionError as exc:
            raise ToolExecutionError(
                f"Permission denied: {target}",
                "I don't have permission to read that file.",
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(
                f"OS error reading {target}: {exc}",
                "I couldn't read that file.",
            ) from exc
