# SPDX-License-Identifier: MIT
"""Unit tests for bridge/verify_repair/bootstrap.py.

Tests the environment bootstrap and dependency management module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.verify_repair.bootstrap import (
    BootstrapResult,
    _get_install_command,
    clear_bootstrap_marker,
    run_bootstrap,
)

# -----------------------------------------------------------------------------
# BootstrapResult tests
# -----------------------------------------------------------------------------


class TestBootstrapResult:
    """Tests for BootstrapResult dataclass."""

    def test_successful_result(self) -> None:
        """BootstrapResult can represent a successful operation."""
        result = BootstrapResult(
            success=True,
            command=["pip", "install", "-e", "."],
            returncode=0,
            stdout="Successfully installed",
            stderr="",
            elapsed_s=2.5,
        )
        assert result.success is True
        assert result.returncode == 0
        assert result.skipped is False

    def test_failed_result(self) -> None:
        """BootstrapResult can represent a failed operation."""
        result = BootstrapResult(
            success=False,
            command=["pip", "install", "-e", "."],
            returncode=1,
            stdout="",
            stderr="Error: some error",
            elapsed_s=1.0,
        )
        assert result.success is False
        assert result.returncode == 1

    def test_skipped_result(self) -> None:
        """BootstrapResult can represent a skipped operation."""
        result = BootstrapResult(
            success=True,
            command=[],
            returncode=0,
            stdout="",
            stderr="",
            elapsed_s=0.0,
            skipped=True,
            skip_reason="Already bootstrapped",
        )
        assert result.success is True
        assert result.skipped is True
        assert "Already" in result.skip_reason


# -----------------------------------------------------------------------------
# _get_install_command tests
# -----------------------------------------------------------------------------


class TestGetInstallCommand:
    """Tests for _get_install_command function."""

    def test_uv_lock_project(self, tmp_path: Path) -> None:
        """Projects with uv.lock use uv sync."""
        (tmp_path / "uv.lock").write_text("", encoding="utf-8")
        cmd = _get_install_command(tmp_path)
        assert cmd is not None
        assert "uv" in cmd
        assert "sync" in cmd

    def test_pyproject_with_project_section(self, tmp_path: Path) -> None:
        """Projects with pyproject.toml containing [project] use pip."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
        cmd = _get_install_command(tmp_path)
        assert cmd is not None
        assert "-m" in cmd
        assert "pip" in cmd
        assert "install" in cmd

    def test_setup_py_project(self, tmp_path: Path) -> None:
        """Projects with setup.py use pip."""
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()\n", encoding="utf-8")
        cmd = _get_install_command(tmp_path)
        assert cmd is not None
        assert "pip" in cmd

    def test_requirements_txt_project(self, tmp_path: Path) -> None:
        """Projects with requirements.txt use pip install -r."""
        (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")
        cmd = _get_install_command(tmp_path)
        assert cmd is not None
        assert "-r" in cmd
        assert "requirements.txt" in cmd

    def test_no_package_manager(self, tmp_path: Path) -> None:
        """Empty projects return None."""
        cmd = _get_install_command(tmp_path)
        assert cmd is None

    def test_priority_uv_over_pyproject(self, tmp_path: Path) -> None:
        """uv.lock takes priority over pyproject.toml."""
        (tmp_path / "uv.lock").write_text("", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
        cmd = _get_install_command(tmp_path)
        assert cmd is not None
        assert "uv" in cmd


# -----------------------------------------------------------------------------
# run_bootstrap tests
# -----------------------------------------------------------------------------


class TestRunBootstrap:
    """Tests for run_bootstrap function."""

    def test_no_package_manager_skips(self, tmp_path: Path) -> None:
        """run_bootstrap skips when no package manager is detected."""
        result = run_bootstrap(tmp_path, verbose=False)
        assert result.success is True
        assert result.skipped is True
        assert "No package manager" in result.skip_reason

    def test_respects_marker_file(self, tmp_path: Path) -> None:
        """run_bootstrap skips when recent marker exists."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
        marker = tmp_path / ".bootstrap_done"
        marker.write_text("2025-01-01T00:00:00", encoding="utf-8")
        # Set recent mtime
        import time

        os.utime(marker, (time.time(), time.time()))

        result = run_bootstrap(tmp_path, verbose=False)
        assert result.success is True
        assert result.skipped is True
        assert "marker" in result.skip_reason.lower()

    def test_force_ignores_marker(self, tmp_path: Path) -> None:
        """run_bootstrap with force=True ignores the marker."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
        marker = tmp_path / ".bootstrap_done"
        marker.write_text("2025-01-01T00:00:00", encoding="utf-8")

        # Mock subprocess.run to avoid actually running pip
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout="Success",
                stderr="",
            )
            result = run_bootstrap(tmp_path, force=True, verbose=False)
            assert result.skipped is False
            assert mock_run.called

    def test_writes_log_file(self, tmp_path: Path) -> None:
        """run_bootstrap writes to log file when provided."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
        log_path = tmp_path / "logs" / "bootstrap.log"

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout="Install complete",
                stderr="",
            )
            run_bootstrap(tmp_path, log_path=log_path, force=True, verbose=False)

        assert log_path.exists()
        log_content = log_path.read_text()
        assert "Command:" in log_content
        assert "Return code:" in log_content

    def test_handles_timeout(self, tmp_path: Path) -> None:
        """run_bootstrap handles timeout gracefully."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")

        import subprocess

        with mock.patch("bridge.verify_repair.bootstrap.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["pip"], timeout=600)
            result = run_bootstrap(tmp_path, force=True, verbose=False)

        assert result.success is False
        assert "timed out" in result.stderr.lower()

    def test_handles_exception(self, tmp_path: Path) -> None:
        """run_bootstrap handles exceptions gracefully."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")

        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = OSError("No such file")
            result = run_bootstrap(tmp_path, force=True, verbose=False)

        assert result.success is False
        assert result.returncode == -1

    def test_creates_marker_on_success(self, tmp_path: Path) -> None:
        """run_bootstrap creates marker file on success."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
        marker = tmp_path / ".bootstrap_done"

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout="Success",
                stderr="",
            )
            result = run_bootstrap(tmp_path, force=True, verbose=False)

        assert result.success is True
        assert marker.exists()


# -----------------------------------------------------------------------------
# clear_bootstrap_marker tests
# -----------------------------------------------------------------------------


class TestClearBootstrapMarker:
    """Tests for clear_bootstrap_marker function."""

    def test_removes_existing_marker(self, tmp_path: Path) -> None:
        """clear_bootstrap_marker removes existing marker."""
        marker = tmp_path / ".bootstrap_done"
        marker.write_text("2025-01-01T00:00:00", encoding="utf-8")
        assert marker.exists()

        clear_bootstrap_marker(tmp_path)
        assert not marker.exists()

    def test_handles_missing_marker(self, tmp_path: Path) -> None:
        """clear_bootstrap_marker handles non-existent marker gracefully."""
        marker = tmp_path / ".bootstrap_done"
        assert not marker.exists()

        # Should not raise
        clear_bootstrap_marker(tmp_path)
        assert not marker.exists()
