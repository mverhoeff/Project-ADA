"""Cross-platform desktop application launcher.

Exposes a single public function, :func:`open_app`, that locates and launches
an application by its display name. The Windows path uses PowerShell's
``Get-StartApps`` to resolve Start-menu entries (including UWP/Store apps
such as Spotify) to their AppUserModelID, then launches via
``shell:AppsFolder``. The Linux path uses ``gtk-launch`` against a normalised
name and falls back to scanning ``.desktop`` files when the direct lookup
fails.

All failures are surfaced as :class:`ToolExecutionError` so the agent layer
can speak ``user_message`` aloud.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ada_platform.detect import current_platform
from core.exceptions import ToolExecutionError
from core.logger import get_logger

_log = get_logger(__name__)

_WINDOWS_LOOKUP_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$apps = Get-StartApps | Where-Object { $_.Name -like "*$env:APP_NAME*" }
if (-not $apps) { exit 2 }
$app = $apps | Select-Object -First 1
Start-Process "shell:AppsFolder\$($app.AppID)"
"""

_LINUX_DESKTOP_DIRS: tuple[Path, ...] = (
    Path.home() / ".local/share/applications",
    Path("/usr/share/applications"),
    Path("/usr/local/share/applications"),
    Path("/var/lib/flatpak/exports/share/applications"),
    Path.home() / ".local/share/flatpak/exports/share/applications",
)


def open_app(name: str) -> None:
    """Launch the application whose display name matches ``name``.

    Args:
        name: User-facing application name (e.g. ``"Spotify"``).

    Raises:
        ToolExecutionError: If no matching application is found or the OS
            launch call fails.
    """
    name = name.strip()
    if not name:
        raise ToolExecutionError(
            "Empty application name passed to open_app.",
            "Please tell me which application to open.",
        )

    platform = current_platform()
    _log.info("open_app", name=name, platform=platform)
    if platform == "windows":
        _open_app_windows(name)
    else:
        _open_app_linux(name)


def _open_app_windows(name: str) -> None:
    env = os.environ.copy()
    env["APP_NAME"] = name
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _WINDOWS_LOOKUP_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError as exc:
        raise ToolExecutionError(
            "powershell.exe not found on PATH.",
            "I couldn't find PowerShell to launch the app.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolExecutionError(
            f"PowerShell lookup timed out for {name!r}.",
            f"Looking up {name} took too long.",
        ) from exc

    if result.returncode == 2:
        raise ToolExecutionError(
            f"No Start-menu app matched {name!r}.",
            f"I couldn't find an app called {name}.",
        )
    if result.returncode != 0:
        raise ToolExecutionError(
            f"Start-Process failed for {name!r}: {result.stderr.strip()}",
            f"I couldn't open {name}.",
        )


def _open_app_linux(name: str) -> None:
    normalised = name.strip().lower().replace(" ", "-")
    if _try_gtk_launch(normalised):
        return

    desktop_stem = _find_desktop_stem(name)
    if desktop_stem is not None and _try_gtk_launch(desktop_stem):
        return

    raise ToolExecutionError(
        f"No .desktop entry matched {name!r}.",
        f"I couldn't find an app called {name}.",
    )


def _try_gtk_launch(stem: str) -> bool:
    """Return True iff ``gtk-launch <stem>`` exits 0."""
    try:
        result = subprocess.run(
            ["gtk-launch", stem],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        raise ToolExecutionError(
            "gtk-launch not found on PATH.",
            "I need gtk-launch installed to open apps. "
            "Install it with 'sudo dnf install gtk3'.",
        ) from None
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def _find_desktop_stem(name: str) -> str | None:
    """Scan known .desktop directories for an entry whose Name= matches."""
    target = name.strip().lower()
    for directory in _LINUX_DESKTOP_DIRS:
        if not directory.is_dir():
            continue
        for entry in directory.glob("*.desktop"):
            display_name = _read_desktop_name(entry)
            if display_name is not None and display_name.strip().lower() == target:
                return entry.stem
    return None


def _read_desktop_name(path: Path) -> str | None:
    """Return the first ``Name=`` value from a .desktop file, or None."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("Name="):
            return line[len("Name="):]
    return None
