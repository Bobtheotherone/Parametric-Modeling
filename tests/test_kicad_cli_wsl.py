"""Tests for WSL Docker workdir workaround in KiCad CLI module.

These tests verify the WSL detection and Docker-accessible path handling
that allows the KiCad CLI runner to work with Docker Desktop on WSL2.

The core issue: In WSL2 with Docker Desktop, paths under /tmp are not
accessible to Docker containers because Docker Desktop runs in a separate
namespace. The workaround copies files to ~/.coupongen_docker_tmp/ which
IS accessible.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from formula_foundry.coupongen.kicad.cli import (
    _ENV_DISABLE_WSL_COPY,
    _ENV_DOCKER_TMP_BASE,
    _docker_accessible_workdir,
    _is_path_docker_accessible,
    _is_wsl,
)


class TestIsWsl:
    """Tests for _is_wsl() detection."""

    def test_wsl_detected_when_microsoft_in_proc_version(self) -> None:
        """Should return True when /proc/version contains 'microsoft'."""
        mock_content = "Linux version 5.15.0-microsoft-standard-WSL2"
        with patch("builtins.open", mock_open(read_data=mock_content)):
            assert _is_wsl() is True

    def test_wsl_not_detected_on_regular_linux(self) -> None:
        """Should return False on regular Linux (no 'microsoft' in version)."""
        mock_content = "Linux version 5.15.0-generic (buildd@lcy02-amd64-015)"
        with patch("builtins.open", mock_open(read_data=mock_content)):
            assert _is_wsl() is False

    def test_wsl_not_detected_when_file_missing(self) -> None:
        """Should return False when /proc/version doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert _is_wsl() is False

    def test_wsl_not_detected_when_permission_denied(self) -> None:
        """Should return False when /proc/version is not readable."""
        with patch("builtins.open", side_effect=PermissionError):
            assert _is_wsl() is False


class TestIsPathDockerAccessible:
    """Tests for _is_path_docker_accessible() behavior."""

    def test_all_paths_accessible_when_not_wsl(self) -> None:
        """On non-WSL systems, all paths should be considered accessible."""
        with patch("formula_foundry.coupongen.kicad.cli._is_wsl", return_value=False):
            assert _is_path_docker_accessible(Path("/tmp/foo")) is True
            assert _is_path_docker_accessible(Path("/var/lib/docker")) is True
            assert _is_path_docker_accessible(Path("/some/random/path")) is True

    def test_tmp_not_accessible_in_wsl(self) -> None:
        """In WSL, /tmp paths should not be considered accessible."""
        with patch("formula_foundry.coupongen.kicad.cli._is_wsl", return_value=True):
            # Clear env override
            env = os.environ.copy()
            env.pop(_ENV_DISABLE_WSL_COPY, None)
            with patch.dict(os.environ, env, clear=True):
                assert _is_path_docker_accessible(Path("/tmp/foo")) is False
                assert _is_path_docker_accessible(Path("/tmp/pytest-123/test")) is False

    def test_home_accessible_in_wsl(self) -> None:
        """In WSL, home directory paths should be accessible."""
        with patch("formula_foundry.coupongen.kicad.cli._is_wsl", return_value=True):
            home = Path.home()
            env = os.environ.copy()
            env.pop(_ENV_DISABLE_WSL_COPY, None)
            with patch.dict(os.environ, env, clear=True):
                assert _is_path_docker_accessible(home / "foo") is True
                assert _is_path_docker_accessible(home / "projects/test") is True

    def test_mnt_accessible_in_wsl(self) -> None:
        """In WSL, /mnt paths (Windows filesystem) should be accessible."""
        with patch("formula_foundry.coupongen.kicad.cli._is_wsl", return_value=True):
            env = os.environ.copy()
            env.pop(_ENV_DISABLE_WSL_COPY, None)
            with patch.dict(os.environ, env, clear=True):
                assert _is_path_docker_accessible(Path("/mnt/c/Users/test")) is True
                assert _is_path_docker_accessible(Path("/mnt/d/projects")) is True

    def test_disable_env_var_overrides_wsl_check(self) -> None:
        """COUPONGEN_DISABLE_WSL_WORKDIR_COPY=1 should bypass the check."""
        with (
            patch("formula_foundry.coupongen.kicad.cli._is_wsl", return_value=True),
            patch.dict(os.environ, {_ENV_DISABLE_WSL_COPY: "1"}),
        ):
            # Even /tmp should be "accessible" when override is set
            assert _is_path_docker_accessible(Path("/tmp/foo")) is True


class TestDockerAccessibleWorkdir:
    """Tests for _docker_accessible_workdir() copy-in/copy-out semantics."""

    def test_yields_original_when_already_accessible(self, tmp_path: Path) -> None:
        """When path is already accessible, should yield it directly."""
        with (
            patch(
                "formula_foundry.coupongen.kicad.cli._is_path_docker_accessible",
                return_value=True,
            ),
            _docker_accessible_workdir(tmp_path) as workdir,
        ):
            assert workdir == tmp_path

    def test_copies_files_when_not_accessible(self, tmp_path: Path) -> None:
        """When path is not accessible, should copy files to temp location."""
        # Create test files
        (tmp_path / "test.txt").write_text("hello")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.txt").write_text("world")

        # Use a custom temp base under home (which we can control)
        custom_tmp = tmp_path.parent / "docker_accessible_test"
        custom_tmp.mkdir(exist_ok=True)

        with (
            patch(
                "formula_foundry.coupongen.kicad.cli._is_path_docker_accessible",
                return_value=False,
            ),
            patch(
                "formula_foundry.coupongen.kicad.cli._get_wsl_docker_tmp",
                return_value=custom_tmp,
            ),
        ):
            with _docker_accessible_workdir(tmp_path) as workdir:
                # Should be a different directory
                assert workdir != tmp_path
                assert workdir.parent == custom_tmp

                # Files should be copied
                assert (workdir / "test.txt").read_text() == "hello"
                assert (workdir / "subdir" / "nested.txt").read_text() == "world"

                # Create a new file in the workdir
                (workdir / "output.txt").write_text("generated")

            # After context exits, new files should be copied back
            assert (tmp_path / "output.txt").read_text() == "generated"

        # Cleanup
        import shutil

        shutil.rmtree(custom_tmp, ignore_errors=True)

    def test_copies_back_modified_files(self, tmp_path: Path) -> None:
        """Modified files in the temp workdir should be copied back."""
        (tmp_path / "input.txt").write_text("original")

        custom_tmp = tmp_path.parent / "docker_accessible_test2"
        custom_tmp.mkdir(exist_ok=True)

        with (
            patch(
                "formula_foundry.coupongen.kicad.cli._is_path_docker_accessible",
                return_value=False,
            ),
            patch(
                "formula_foundry.coupongen.kicad.cli._get_wsl_docker_tmp",
                return_value=custom_tmp,
            ),
        ):
            with _docker_accessible_workdir(tmp_path) as workdir:
                # Modify the file
                (workdir / "input.txt").write_text("modified")

            # After context exits, changes should be copied back
            assert (tmp_path / "input.txt").read_text() == "modified"

        import shutil

        shutil.rmtree(custom_tmp, ignore_errors=True)

    def test_cleans_up_temp_directory(self, tmp_path: Path) -> None:
        """Temp directory should be cleaned up after context exits."""
        custom_tmp = tmp_path.parent / "docker_accessible_test3"
        custom_tmp.mkdir(exist_ok=True)

        temp_workdir = None

        with (
            patch(
                "formula_foundry.coupongen.kicad.cli._is_path_docker_accessible",
                return_value=False,
            ),
            patch(
                "formula_foundry.coupongen.kicad.cli._get_wsl_docker_tmp",
                return_value=custom_tmp,
            ),
        ):
            with _docker_accessible_workdir(tmp_path) as workdir:
                temp_workdir = workdir
                assert temp_workdir.exists()

            # After context exits, temp dir should be cleaned up
            assert not temp_workdir.exists()

        import shutil

        shutil.rmtree(custom_tmp, ignore_errors=True)


class TestEnvVarOverrides:
    """Tests for environment variable overrides."""

    def test_docker_tmp_base_env_var(self, tmp_path: Path) -> None:
        """COUPONGEN_DOCKER_TMP_BASE should override the temp directory location."""
        from formula_foundry.coupongen.kicad import cli

        # Reset the cached value
        original_cache = cli._WSL_DOCKER_TMP_BASE
        cli._WSL_DOCKER_TMP_BASE = None

        custom_base = tmp_path / "custom_docker_tmp"

        try:
            with patch.dict(os.environ, {_ENV_DOCKER_TMP_BASE: str(custom_base)}):
                result = cli._get_wsl_docker_tmp()
                assert result == custom_base
                assert custom_base.exists()
        finally:
            cli._WSL_DOCKER_TMP_BASE = original_cache
            import shutil

            shutil.rmtree(custom_base, ignore_errors=True)
