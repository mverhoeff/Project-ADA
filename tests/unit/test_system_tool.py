"""Unit tests for :mod:`agent.tools.system_tool` and :mod:`agent.tools`."""

from __future__ import annotations

import json
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.tools import build_registry
from agent.tools.system_tool import (
    FileReadTool,
    ShellTool,
    SystemInfoTool,
    _query_cpu_temp,
    _query_gpu_stats,
)
from core.exceptions import ToolExecutionError

_ALLOWED_DIR = "/tmp/ada_test_allowed"

_FAKE_CONFIG: dict[str, Any] = {
    "agent": {
        "shell_timeout_seconds": 30,
        "allowed_paths": [_ALLOWED_DIR],
    }
}


# -- ShellTool ---------------------------------------------------------------


def _mock_run(stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    return MagicMock(stdout=stdout, stderr=stderr, returncode=returncode)


def test_shell_tool_executes_command() -> None:
    tool = ShellTool(_FAKE_CONFIG)
    with patch(
        "agent.tools.system_tool.subprocess.run",
        return_value=_mock_run(stdout="hi\n"),
    ):
        result = tool.execute({"command": "echo hi"})
    assert result == "hi"


def test_shell_tool_combines_stdout_and_stderr() -> None:
    tool = ShellTool(_FAKE_CONFIG)
    with patch(
        "agent.tools.system_tool.subprocess.run",
        return_value=_mock_run(stdout="out\n", stderr="err\n"),
    ):
        result = tool.execute({"command": "ls"})
    assert "out" in result
    assert "err" in result


def test_shell_tool_timeout_uses_config_default() -> None:
    tool = ShellTool(_FAKE_CONFIG)
    with patch(
        "agent.tools.system_tool.subprocess.run",
        return_value=_mock_run(),
    ) as mock_run:
        tool.execute({"command": "ls"})
    assert mock_run.call_args.kwargs["timeout"] == 30


def test_shell_tool_timeout_param_overrides_config() -> None:
    tool = ShellTool(_FAKE_CONFIG)
    with patch(
        "agent.tools.system_tool.subprocess.run",
        return_value=_mock_run(),
    ) as mock_run:
        tool.execute({"command": "ls", "timeout_seconds": 5})
    assert mock_run.call_args.kwargs["timeout"] == 5


def test_shell_tool_timeout_expired_raises_tool_error() -> None:
    tool = ShellTool(_FAKE_CONFIG)
    with patch(
        "agent.tools.system_tool.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="sleep 100", timeout=30),
    ):
        with pytest.raises(ToolExecutionError) as exc_info:
            tool.execute({"command": "sleep 100"})
    assert "took too long" in exc_info.value.user_message


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -r /tmp/foo",
        "RM -RF /home",
        "mkfs.ext4 /dev/sda",
        "dd if=/dev/zero of=/dev/sda",
        ":(){ :|:& };:",
        "shutdown -h now",
        "reboot",
        "format C:",
        "deltree /Y C:\\",
        "rmdir /s /q C:\\",
    ],
)
def test_shell_tool_deny_list_blocks_dangerous_commands(command: str) -> None:
    tool = ShellTool(_FAKE_CONFIG)
    with patch("agent.tools.system_tool.subprocess.run") as mock_run:
        with pytest.raises(ToolExecutionError) as exc_info:
            tool.execute({"command": command})
    mock_run.assert_not_called()
    assert "not allowed" in exc_info.value.user_message


def test_shell_tool_allows_safe_commands() -> None:
    tool = ShellTool(_FAKE_CONFIG)
    with patch(
        "agent.tools.system_tool.subprocess.run",
        return_value=_mock_run(stdout="ok\n"),
    ):
        assert tool.execute({"command": "ls -la"}) == "ok"
        assert tool.execute({"command": "echo hello"}) == "ok"


# -- SystemInfoTool ----------------------------------------------------------


def test_system_info_returns_json_with_expected_keys() -> None:
    tool = SystemInfoTool()
    with (
        patch("agent.tools.system_tool.psutil.cpu_percent", return_value=42.0),
        patch(
            "agent.tools.system_tool.psutil.virtual_memory",
            return_value=MagicMock(percent=55.5),
        ),
        patch(
            "agent.tools.system_tool.psutil.disk_usage",
            return_value=MagicMock(percent=66.6),
        ),
        patch("agent.tools.system_tool._query_cpu_temp", return_value=48.0),
        patch(
            "agent.tools.system_tool._query_gpu_stats",
            return_value={
                "vram_used_mb": 6000.0,
                "vram_total_mb": 8000.0,
                "vram_percent": 75.0,
                "temp_celsius": 61.0,
            },
        ),
    ):
        result = tool.execute({})
    parsed = json.loads(result)
    assert parsed == {
        "cpu_percent": 42.0,
        "cpu_temp_celsius": 48.0,
        "memory_percent": 55.5,
        "disk_percent": 66.6,
        "gpu_temp_celsius": 61.0,
        "vram_percent": 75.0,
        "vram_used_mb": 6000.0,
        "vram_total_mb": 8000.0,
    }


def test_system_info_reports_nulls_when_helpers_unavailable() -> None:
    tool = SystemInfoTool()
    with (
        patch("agent.tools.system_tool.psutil.cpu_percent", return_value=1.0),
        patch(
            "agent.tools.system_tool.psutil.virtual_memory",
            return_value=MagicMock(percent=1.0),
        ),
        patch(
            "agent.tools.system_tool.psutil.disk_usage",
            return_value=MagicMock(percent=1.0),
        ),
        patch("agent.tools.system_tool._query_cpu_temp", return_value=None),
        patch("agent.tools.system_tool._query_gpu_stats", return_value=None),
    ):
        result = tool.execute({})
    parsed = json.loads(result)
    assert parsed["cpu_temp_celsius"] is None
    assert parsed["gpu_temp_celsius"] is None
    assert parsed["vram_percent"] is None
    assert parsed["vram_used_mb"] is None
    assert parsed["vram_total_mb"] is None


def test_system_info_uses_non_blocking_cpu_sample() -> None:
    tool = SystemInfoTool()
    with (
        patch(
            "agent.tools.system_tool.psutil.cpu_percent", return_value=1.0
        ) as mock_cpu,
        patch(
            "agent.tools.system_tool.psutil.virtual_memory",
            return_value=MagicMock(percent=1.0),
        ),
        patch(
            "agent.tools.system_tool.psutil.disk_usage",
            return_value=MagicMock(percent=1.0),
        ),
        patch("agent.tools.system_tool._query_cpu_temp", return_value=None),
        patch("agent.tools.system_tool._query_gpu_stats", return_value=None),
    ):
        tool.execute({})
    assert mock_cpu.call_args.kwargs.get("interval") is None


# -- _query_gpu_stats --------------------------------------------------------


def test_query_gpu_stats_parses_nvidia_smi_csv() -> None:
    fake = _mock_run(stdout="6000, 8000, 61\n")
    with patch("agent.tools.system_tool.subprocess.run", return_value=fake):
        stats = _query_gpu_stats()
    assert stats == {
        "vram_used_mb": 6000.0,
        "vram_total_mb": 8000.0,
        "vram_percent": 75.0,
        "temp_celsius": 61.0,
    }


def test_query_gpu_stats_returns_none_when_nvidia_smi_missing() -> None:
    with patch(
        "agent.tools.system_tool.subprocess.run",
        side_effect=FileNotFoundError,
    ):
        assert _query_gpu_stats() is None


def test_query_gpu_stats_returns_none_on_timeout() -> None:
    with patch(
        "agent.tools.system_tool.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5),
    ):
        assert _query_gpu_stats() is None


def test_query_gpu_stats_returns_none_on_nonzero_exit() -> None:
    fake = _mock_run(stdout="", stderr="error", returncode=1)
    with patch("agent.tools.system_tool.subprocess.run", return_value=fake):
        assert _query_gpu_stats() is None


def test_query_gpu_stats_returns_none_on_parse_error() -> None:
    fake = _mock_run(stdout="garbage output\n")
    with patch("agent.tools.system_tool.subprocess.run", return_value=fake):
        assert _query_gpu_stats() is None


# -- _query_cpu_temp ---------------------------------------------------------


def test_query_cpu_temp_reads_first_known_key() -> None:
    with patch(
        "agent.tools.system_tool.psutil.sensors_temperatures",
        create=True,
        return_value={"coretemp": [MagicMock(current=48.0)]},
    ):
        assert _query_cpu_temp() == 48.0


def test_query_cpu_temp_returns_none_when_empty() -> None:
    with patch(
        "agent.tools.system_tool.psutil.sensors_temperatures",
        create=True,
        return_value={},
    ):
        assert _query_cpu_temp() is None


def test_query_cpu_temp_returns_none_on_not_implemented() -> None:
    with patch(
        "agent.tools.system_tool.psutil.sensors_temperatures",
        create=True,
        side_effect=NotImplementedError,
    ):
        assert _query_cpu_temp() is None


# -- FileReadTool ------------------------------------------------------------


def test_file_read_allowed_path_returns_content() -> None:
    tool = FileReadTool(_FAKE_CONFIG)
    with patch("pathlib.Path.read_text", return_value="hello") as mock_read:
        result = tool.execute({"path": f"{_ALLOWED_DIR}/note.txt"})
    assert result == "hello"
    mock_read.assert_called_once()


def test_file_read_blocked_outside_allowed() -> None:
    tool = FileReadTool(_FAKE_CONFIG)
    with patch("pathlib.Path.read_text") as mock_read:
        with pytest.raises(ToolExecutionError) as exc_info:
            tool.execute({"path": "/etc/passwd"})
    mock_read.assert_not_called()
    assert "outside approved directories" in exc_info.value.user_message


def test_file_read_traversal_is_blocked_after_resolve() -> None:
    tool = FileReadTool(_FAKE_CONFIG)
    with patch("pathlib.Path.read_text") as mock_read:
        with pytest.raises(ToolExecutionError):
            tool.execute({"path": f"{_ALLOWED_DIR}/../../etc/passwd"})
    mock_read.assert_not_called()


def test_file_read_not_found_raises_tool_error() -> None:
    tool = FileReadTool(_FAKE_CONFIG)
    with patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
        with pytest.raises(ToolExecutionError) as exc_info:
            tool.execute({"path": f"{_ALLOWED_DIR}/missing.txt"})
    assert "does not exist" in exc_info.value.user_message


def test_file_read_permission_error_raises_tool_error() -> None:
    tool = FileReadTool(_FAKE_CONFIG)
    with patch("pathlib.Path.read_text", side_effect=PermissionError):
        with pytest.raises(ToolExecutionError) as exc_info:
            tool.execute({"path": f"{_ALLOWED_DIR}/locked.txt"})
    assert "permission" in exc_info.value.user_message.lower()


def test_file_read_other_oserror_raises_tool_error() -> None:
    tool = FileReadTool(_FAKE_CONFIG)
    with patch("pathlib.Path.read_text", side_effect=OSError("disk gone")):
        with pytest.raises(ToolExecutionError) as exc_info:
            tool.execute({"path": f"{_ALLOWED_DIR}/x.txt"})
    assert "couldn't read" in exc_info.value.user_message.lower()


def test_file_read_empty_allowed_paths_blocks_all() -> None:
    tool = FileReadTool({"agent": {"allowed_paths": []}})
    with patch("pathlib.Path.read_text") as mock_read:
        with pytest.raises(ToolExecutionError):
            tool.execute({"path": f"{_ALLOWED_DIR}/x.txt"})
    mock_read.assert_not_called()


def test_file_read_missing_agent_section_blocks_all() -> None:
    tool = FileReadTool({})
    with patch("pathlib.Path.read_text") as mock_read:
        with pytest.raises(ToolExecutionError):
            tool.execute({"path": f"{_ALLOWED_DIR}/x.txt"})
    mock_read.assert_not_called()


# -- build_registry ----------------------------------------------------------


def test_build_registry_returns_all_tools() -> None:
    registry = build_registry(_FAKE_CONFIG)
    assert set(registry.keys()) == {
        "run_shell",
        "get_system_info",
        "read_file",
        "open_app",
        "web_search",
        "web_fetch",
    }
    assert isinstance(registry["run_shell"], ShellTool)
    assert isinstance(registry["get_system_info"], SystemInfoTool)
    assert isinstance(registry["read_file"], FileReadTool)
