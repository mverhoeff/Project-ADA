"""Unit tests for :mod:`ada_platform.detect`."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ada_platform.detect import current_platform


def test_returns_windows_on_win32() -> None:
    with patch("sys.platform", "win32"):
        assert current_platform() == "windows"


def test_returns_linux_on_linux() -> None:
    with patch("sys.platform", "linux"):
        assert current_platform() == "linux"


def test_returns_linux_on_linux2() -> None:
    with patch("sys.platform", "linux2"):
        assert current_platform() == "linux"


def test_raises_on_macos() -> None:
    with patch("sys.platform", "darwin"):
        with pytest.raises(RuntimeError, match="darwin"):
            current_platform()


def test_raises_on_unknown_platform() -> None:
    with patch("sys.platform", "freebsd"):
        with pytest.raises(RuntimeError):
            current_platform()
