"""Unit tests for :mod:`ada_platform.launcher`."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ada_platform import launcher
from core.exceptions import ToolExecutionError


# -- shared helpers ----------------------------------------------------------


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


# -- top-level open_app ------------------------------------------------------


def test_open_app_empty_name_raises() -> None:
    with pytest.raises(ToolExecutionError) as exc_info:
        launcher.open_app("   ")
    assert "which application" in exc_info.value.user_message.lower()


def test_open_app_dispatches_windows() -> None:
    with (
        patch("ada_platform.launcher.current_platform", return_value="windows"),
        patch("ada_platform.launcher._open_app_windows") as mock_win,
        patch("ada_platform.launcher._open_app_linux") as mock_linux,
    ):
        launcher.open_app("Spotify")
    mock_win.assert_called_once_with("Spotify")
    mock_linux.assert_not_called()


def test_open_app_dispatches_linux() -> None:
    with (
        patch("ada_platform.launcher.current_platform", return_value="linux"),
        patch("ada_platform.launcher._open_app_windows") as mock_win,
        patch("ada_platform.launcher._open_app_linux") as mock_linux,
    ):
        launcher.open_app("Firefox")
    mock_linux.assert_called_once_with("Firefox")
    mock_win.assert_not_called()


# -- Windows path ------------------------------------------------------------


def test_windows_success() -> None:
    with patch(
        "ada_platform.launcher.subprocess.run",
        return_value=_completed(returncode=0),
    ) as mock_run:
        launcher._open_app_windows("Spotify")
    args, kwargs = mock_run.call_args
    assert args[0][0] == "powershell"
    assert "-NoProfile" in args[0]
    assert kwargs["env"]["APP_NAME"] == "Spotify"


def test_windows_not_found_raises_with_user_message() -> None:
    with patch(
        "ada_platform.launcher.subprocess.run",
        return_value=_completed(returncode=2),
    ):
        with pytest.raises(ToolExecutionError) as exc_info:
            launcher._open_app_windows("Nope")
    assert "couldn't find" in exc_info.value.user_message.lower()
    assert "Nope" in exc_info.value.user_message


def test_windows_generic_error_raises() -> None:
    with patch(
        "ada_platform.launcher.subprocess.run",
        return_value=_completed(returncode=1, stderr="boom"),
    ):
        with pytest.raises(ToolExecutionError) as exc_info:
            launcher._open_app_windows("Spotify")
    assert "couldn't open" in exc_info.value.user_message.lower()


def test_windows_no_powershell_raises() -> None:
    with patch(
        "ada_platform.launcher.subprocess.run",
        side_effect=FileNotFoundError,
    ):
        with pytest.raises(ToolExecutionError) as exc_info:
            launcher._open_app_windows("Spotify")
    assert "powershell" in exc_info.value.user_message.lower()


def test_windows_timeout_raises() -> None:
    with patch(
        "ada_platform.launcher.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="powershell", timeout=15),
    ):
        with pytest.raises(ToolExecutionError) as exc_info:
            launcher._open_app_windows("Spotify")
    assert "too long" in exc_info.value.user_message.lower()


# -- Linux path --------------------------------------------------------------


def test_linux_gtk_launch_success() -> None:
    with patch(
        "ada_platform.launcher.subprocess.run",
        return_value=_completed(returncode=0),
    ) as mock_run:
        launcher._open_app_linux("Firefox")
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == ["gtk-launch", "firefox"]


def test_linux_normalises_spaces_to_dashes() -> None:
    with patch(
        "ada_platform.launcher.subprocess.run",
        return_value=_completed(returncode=0),
    ) as mock_run:
        launcher._open_app_linux("Visual Studio Code")
    assert mock_run.call_args.args[0] == ["gtk-launch", "visual-studio-code"]


def test_linux_falls_back_to_desktop_scan(tmp_path: Path) -> None:
    desktop_dir = tmp_path / "apps"
    desktop_dir.mkdir()
    (desktop_dir / "com.spotify.Client.desktop").write_text(
        "[Desktop Entry]\nName=Spotify\nExec=spotify %U\n", encoding="utf-8"
    )

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> MagicMock:
        calls.append(cmd)
        # First call (direct gtk-launch spotify) fails; the second (with the
        # resolved .desktop stem) succeeds.
        return _completed(returncode=0 if len(calls) > 1 else 3)

    with (
        patch("ada_platform.launcher.subprocess.run", side_effect=fake_run),
        patch("ada_platform.launcher._LINUX_DESKTOP_DIRS", (desktop_dir,)),
    ):
        launcher._open_app_linux("Spotify")

    assert calls[0] == ["gtk-launch", "spotify"]
    assert calls[1] == ["gtk-launch", "com.spotify.Client"]


def test_linux_no_match_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "apps"
    empty_dir.mkdir()
    with (
        patch(
            "ada_platform.launcher.subprocess.run",
            return_value=_completed(returncode=3),
        ),
        patch("ada_platform.launcher._LINUX_DESKTOP_DIRS", (empty_dir,)),
    ):
        with pytest.raises(ToolExecutionError) as exc_info:
            launcher._open_app_linux("Nope")
    assert "couldn't find" in exc_info.value.user_message.lower()


def test_linux_gtk_launch_missing_raises() -> None:
    with patch(
        "ada_platform.launcher.subprocess.run",
        side_effect=FileNotFoundError,
    ):
        with pytest.raises(ToolExecutionError) as exc_info:
            launcher._open_app_linux("Firefox")
    assert "gtk-launch" in exc_info.value.user_message.lower()


def test_linux_desktop_name_matching_is_case_insensitive(tmp_path: Path) -> None:
    desktop_dir = tmp_path / "apps"
    desktop_dir.mkdir()
    (desktop_dir / "firefox.desktop").write_text(
        "[Desktop Entry]\nName=Firefox\nExec=firefox %u\n", encoding="utf-8"
    )

    with patch("ada_platform.launcher._LINUX_DESKTOP_DIRS", (desktop_dir,)):
        assert launcher._find_desktop_stem("FIREFOX") == "firefox"
        assert launcher._find_desktop_stem("firefox") == "firefox"
        assert launcher._find_desktop_stem("Mozilla") is None


def test_linux_read_desktop_name_handles_missing_name_field(tmp_path: Path) -> None:
    no_name = tmp_path / "weird.desktop"
    no_name.write_text("[Desktop Entry]\nExec=weird\n", encoding="utf-8")
    assert launcher._read_desktop_name(no_name) is None


def test_linux_read_desktop_name_returns_none_on_oserror(tmp_path: Path) -> None:
    missing = tmp_path / "missing.desktop"
    assert launcher._read_desktop_name(missing) is None
