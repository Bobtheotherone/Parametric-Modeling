# SPDX-License-Identifier: MIT
"""Unit tests for bridge/streaming.py.

Tests the streaming subprocess runner that handles agent process I/O
with real-time logging and prefixed output streaming.

Key function tested:
- run_cmd_with_streaming: Run subprocess with streaming output to console and log files
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.streaming import run_cmd_with_streaming

# -----------------------------------------------------------------------------
# Basic functionality tests
# -----------------------------------------------------------------------------


class TestRunCmdWithStreamingBasic:
    """Basic functionality tests for run_cmd_with_streaming."""

    def test_simple_echo_command(self, tmp_path: Path) -> None:
        """Simple echo command returns correct output."""
        call_dir = tmp_path / "call_0"
        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", "print('hello')"],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )
        assert rc == 0
        assert "hello" in stdout
        assert stderr == "" or stderr.strip() == ""

    def test_returns_nonzero_exit_code(self, tmp_path: Path) -> None:
        """Non-zero exit code is returned correctly."""
        call_dir = tmp_path / "call_0"
        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", "import sys; sys.exit(42)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )
        assert rc == 42

    def test_captures_stderr(self, tmp_path: Path) -> None:
        """Stderr is captured correctly."""
        call_dir = tmp_path / "call_0"
        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", "import sys; print('error', file=sys.stderr)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )
        assert rc == 0
        assert "error" in stderr

    def test_captures_both_stdout_and_stderr(self, tmp_path: Path) -> None:
        """Both stdout and stderr are captured."""
        call_dir = tmp_path / "call_0"
        script = "import sys; print('out'); print('err', file=sys.stderr)"
        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", script],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )
        assert rc == 0
        assert "out" in stdout
        assert "err" in stderr


# -----------------------------------------------------------------------------
# Log file creation tests
# -----------------------------------------------------------------------------


class TestLogFileCreation:
    """Tests for log file creation."""

    def test_creates_call_directory(self, tmp_path: Path) -> None:
        """Call directory is created if it doesn't exist."""
        call_dir = tmp_path / "calls" / "call_0"
        assert not call_dir.exists()

        run_cmd_with_streaming(
            cmd=[sys.executable, "-c", "print('test')"],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )

        assert call_dir.exists()
        assert call_dir.is_dir()

    def test_creates_stdout_log(self, tmp_path: Path) -> None:
        """Stdout log file is created."""
        call_dir = tmp_path / "call_0"
        run_cmd_with_streaming(
            cmd=[sys.executable, "-c", "print('logged output')"],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )

        stdout_log = call_dir / "agent_stdout.log"
        assert stdout_log.exists()
        content = stdout_log.read_text()
        assert "logged output" in content

    def test_creates_stderr_log(self, tmp_path: Path) -> None:
        """Stderr log file is created."""
        call_dir = tmp_path / "call_0"
        run_cmd_with_streaming(
            cmd=[sys.executable, "-c", "import sys; print('logged error', file=sys.stderr)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )

        stderr_log = call_dir / "agent_stderr.log"
        assert stderr_log.exists()
        content = stderr_log.read_text()
        assert "logged error" in content

    def test_overwrites_existing_logs(self, tmp_path: Path) -> None:
        """Existing log files are overwritten on new run."""
        call_dir = tmp_path / "call_0"
        call_dir.mkdir(parents=True)
        stdout_log = call_dir / "agent_stdout.log"
        stdout_log.write_text("old content")

        run_cmd_with_streaming(
            cmd=[sys.executable, "-c", "print('new content')"],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )

        content = stdout_log.read_text()
        assert "new content" in content
        assert "old content" not in content


# -----------------------------------------------------------------------------
# Stream mode tests
# -----------------------------------------------------------------------------


class TestStreamModes:
    """Tests for different streaming modes."""

    def test_stream_mode_none(self, tmp_path: Path) -> None:
        """stream_mode='none' produces no streaming output."""
        call_dir = tmp_path / "call_0"
        # Test passes if function completes without error
        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", "print('test')"],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )
        assert rc == 0
        assert "test" in stdout

    def test_stream_mode_stdout(self, tmp_path: Path) -> None:
        """stream_mode='stdout' is accepted."""
        call_dir = tmp_path / "call_0"
        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", "print('test')"],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="stdout",
            call_dir=call_dir,
        )
        assert rc == 0
        assert "test" in stdout

    def test_stream_mode_stderr(self, tmp_path: Path) -> None:
        """stream_mode='stderr' is accepted."""
        call_dir = tmp_path / "call_0"
        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", "import sys; print('err', file=sys.stderr)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="stderr",
            call_dir=call_dir,
        )
        assert rc == 0
        assert "err" in stderr

    def test_stream_mode_both(self, tmp_path: Path) -> None:
        """stream_mode='both' is accepted."""
        call_dir = tmp_path / "call_0"
        script = "import sys; print('out'); print('err', file=sys.stderr)"
        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", script],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="both",
            call_dir=call_dir,
        )
        assert rc == 0
        assert "out" in stdout
        assert "err" in stderr

    def test_stream_mode_case_insensitive(self, tmp_path: Path) -> None:
        """Stream mode is case-insensitive."""
        call_dir = tmp_path / "call_0"
        # Test with uppercase
        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", "print('test')"],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="BOTH",
            call_dir=call_dir,
        )
        assert rc == 0
        assert "test" in stdout


# -----------------------------------------------------------------------------
# Environment and working directory tests
# -----------------------------------------------------------------------------


class TestEnvironmentAndCwd:
    """Tests for environment and working directory handling."""

    def test_respects_working_directory(self, tmp_path: Path) -> None:
        """Working directory is respected by subprocess."""
        call_dir = tmp_path / "call_0"
        sub_dir = tmp_path / "subdir"
        sub_dir.mkdir()

        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", "import os; print(os.getcwd())"],
            cwd=sub_dir,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )

        assert rc == 0
        assert str(sub_dir) in stdout or sub_dir.name in stdout

    def test_custom_environment_variable(self, tmp_path: Path) -> None:
        """Custom environment variables are passed to subprocess."""
        call_dir = tmp_path / "call_0"
        env = os.environ.copy()
        env["TEST_VAR"] = "test_value_12345"

        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", "import os; print(os.environ.get('TEST_VAR', ''))"],
            cwd=tmp_path,
            env=env,
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )

        assert rc == 0
        assert "test_value_12345" in stdout


# -----------------------------------------------------------------------------
# Multiline output tests
# -----------------------------------------------------------------------------


class TestMultilineOutput:
    """Tests for multiline output handling."""

    def test_multiline_stdout(self, tmp_path: Path) -> None:
        """Multiline stdout is captured correctly."""
        call_dir = tmp_path / "call_0"
        script = """
for i in range(5):
    print(f'line {i}')
"""
        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", script],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )

        assert rc == 0
        for i in range(5):
            assert f"line {i}" in stdout

    def test_multiline_stderr(self, tmp_path: Path) -> None:
        """Multiline stderr is captured correctly."""
        call_dir = tmp_path / "call_0"
        script = """
import sys
for i in range(3):
    print(f'error {i}', file=sys.stderr)
"""
        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", script],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )

        assert rc == 0
        for i in range(3):
            assert f"error {i}" in stderr

    def test_interleaved_stdout_stderr(self, tmp_path: Path) -> None:
        """Interleaved stdout/stderr is captured (order not guaranteed)."""
        call_dir = tmp_path / "call_0"
        script = """
import sys
print('out1')
sys.stdout.flush()
print('err1', file=sys.stderr)
sys.stderr.flush()
print('out2')
sys.stdout.flush()
print('err2', file=sys.stderr)
sys.stderr.flush()
"""
        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", script],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )

        assert rc == 0
        assert "out1" in stdout
        assert "out2" in stdout
        assert "err1" in stderr
        assert "err2" in stderr


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_output(self, tmp_path: Path) -> None:
        """Command with no output returns empty strings."""
        call_dir = tmp_path / "call_0"
        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", "pass"],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )

        assert rc == 0
        assert stdout == "" or stdout.strip() == ""
        assert stderr == "" or stderr.strip() == ""

    def test_unicode_output(self, tmp_path: Path) -> None:
        """Unicode output is handled correctly."""
        call_dir = tmp_path / "call_0"
        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", "print('Hello, 世界! 🌍')"],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )

        assert rc == 0
        assert "Hello" in stdout

    def test_long_output(self, tmp_path: Path) -> None:
        """Long output is captured fully."""
        call_dir = tmp_path / "call_0"
        script = """
for i in range(100):
    print('x' * 100)
"""
        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", script],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="test_agent",
            stream_mode="none",
            call_dir=call_dir,
        )

        assert rc == 0
        # Should have ~10,000 'x' characters + newlines
        assert len(stdout) > 10000

    def test_agent_name_in_stream_prefix(self, tmp_path: Path) -> None:
        """Agent name should be used in stream prefix."""
        call_dir = tmp_path / "call_0"
        # This is a behavioral test - we verify the function accepts the agent parameter
        # The actual prefix formatting is tested via integration tests
        rc, stdout, stderr = run_cmd_with_streaming(
            cmd=[sys.executable, "-c", "print('test')"],
            cwd=tmp_path,
            env=os.environ.copy(),
            agent="custom_agent_name",
            stream_mode="none",
            call_dir=call_dir,
        )
        assert rc == 0
