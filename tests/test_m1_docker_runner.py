"""Unit tests for DockerKicadRunner (CP-1.2).

These tests verify the DockerKicadRunner implementation without requiring
Docker to be installed. Integration tests that actually run Docker are
in test_kicad_drc_integration.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from formula_foundry.coupongen.kicad import (
    DockerKicadRunner,
    IKicadRunner,
    KicadRunResult,
)
from formula_foundry.coupongen.kicad.runners.docker import (
    load_docker_image_ref,
    parse_kicad_version,
)


class TestKicadRunResult:
    """Tests for KicadRunResult dataclass."""

    def test_success_property(self) -> None:
        result = KicadRunResult(returncode=0, stdout="", stderr="", command=["kicad-cli"])
        assert result.success is True
        assert result.has_drc_violations is False

    def test_has_drc_violations_property(self) -> None:
        result = KicadRunResult(returncode=5, stdout="", stderr="", command=["kicad-cli"])
        assert result.success is False
        assert result.has_drc_violations is True

    def test_other_error_codes(self) -> None:
        result = KicadRunResult(returncode=1, stdout="", stderr="error", command=["kicad-cli"])
        assert result.success is False
        assert result.has_drc_violations is False


class TestDockerKicadRunnerProtocol:
    """Tests that DockerKicadRunner implements IKicadRunner protocol."""

    def test_implements_protocol(self) -> None:
        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")
        assert isinstance(runner, IKicadRunner)


class TestDockerKicadRunner:
    """Tests for DockerKicadRunner class."""

    def test_init(self) -> None:
        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")
        assert runner.docker_image == "kicad/kicad:9.0.7"
        assert runner.kicad_bin == "kicad-cli"

    def test_init_custom_bin(self) -> None:
        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7", kicad_bin="custom-cli")
        assert runner.kicad_bin == "custom-cli"

    def test_build_docker_command(self, tmp_path: Path) -> None:
        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")
        cmd = runner._build_docker_command(["pcb", "drc", "--help"], tmp_path)

        assert cmd[0] == "docker"
        assert "run" in cmd
        assert "--rm" in cmd
        assert "-v" in cmd
        assert "-w" in cmd
        assert "/workspace" in cmd
        assert "kicad/kicad:9.0.7" in cmd
        assert "kicad-cli" in cmd
        assert "pcb" in cmd
        assert "drc" in cmd

    def test_build_docker_command_with_env(self, tmp_path: Path) -> None:
        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")
        cmd = runner._build_docker_command(
            ["--version"], tmp_path, env={"HOME": "/tmp", "DISPLAY": ":0"}
        )

        # Check environment variables are passed
        assert "-e" in cmd
        env_pairs = []
        for i, arg in enumerate(cmd):
            if arg == "-e" and i + 1 < len(cmd):
                env_pairs.append(cmd[i + 1])
        assert "HOME=/tmp" in env_pairs
        assert "DISPLAY=:0" in env_pairs

    def test_build_docker_command_with_digest(self, tmp_path: Path) -> None:
        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7@sha256:abc123")
        cmd = runner._build_docker_command(["--version"], tmp_path)

        assert "kicad/kicad:9.0.7@sha256:abc123" in cmd

    @patch("subprocess.run")
    def test_run_method(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="output", stderr=""
        )

        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")
        result = runner.run(["--version"], tmp_path)

        assert isinstance(result, KicadRunResult)
        assert result.returncode == 0
        assert result.stdout == "output"
        assert result.success is True

        # Verify subprocess.run was called correctly
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args.kwargs["capture_output"] is True
        assert call_args.kwargs["text"] is True

    @patch("subprocess.run")
    def test_kicad_cli_version(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="kicad-cli 9.0.7\n", stderr=""
        )

        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")
        version = runner.kicad_cli_version(tmp_path)

        assert version == "9.0.7"

    @patch("subprocess.run")
    def test_kicad_cli_version_failure(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Docker not found"
        )

        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")
        with pytest.raises(RuntimeError, match="Failed to get kicad-cli version"):
            runner.kicad_cli_version(tmp_path)


class TestLoadDockerImageRef:
    """Tests for load_docker_image_ref function."""

    def test_load_with_digest(self, tmp_path: Path) -> None:
        lock_file = tmp_path / "kicad.lock.json"
        lock_file.write_text(
            json.dumps(
                {
                    "docker_image": "kicad/kicad:9.0.7",
                    "docker_digest": "sha256:" + "a" * 64,
                }
            )
        )

        image_ref = load_docker_image_ref(lock_file)
        assert image_ref == f"kicad/kicad:9.0.7@{'sha256:' + 'a' * 64}"

    def test_load_with_docker_ref(self, tmp_path: Path) -> None:
        lock_file = tmp_path / "kicad.lock.json"
        lock_file.write_text(
            json.dumps(
                {
                    "docker_image": "kicad/kicad:9.0.7",
                    "docker_digest": "sha256:" + "b" * 64,
                    "docker_ref": "kicad/kicad:9.0.7@sha256:" + "b" * 64,
                }
            )
        )

        image_ref = load_docker_image_ref(lock_file)
        assert image_ref == f"kicad/kicad:9.0.7@{'sha256:' + 'b' * 64}"

    def test_load_placeholder_digest(self, tmp_path: Path) -> None:
        lock_file = tmp_path / "kicad.lock.json"
        lock_file.write_text(
            json.dumps(
                {
                    "docker_image": "kicad/kicad:9.0.7",
                    "docker_digest": "sha256:PLACEHOLDER",
                }
            )
        )

        with pytest.raises(ValueError, match="placeholder"):
            load_docker_image_ref(lock_file)

    def test_load_mismatched_docker_ref(self, tmp_path: Path) -> None:
        lock_file = tmp_path / "kicad.lock.json"
        lock_file.write_text(
            json.dumps(
                {
                    "docker_image": "kicad/kicad:9.0.7",
                    "docker_digest": "sha256:" + "c" * 64,
                    "docker_ref": "kicad/kicad:9.0.7@sha256:" + "d" * 64,
                }
            )
        )

        with pytest.raises(ValueError, match="docker_ref digest"):
            load_docker_image_ref(lock_file)

    def test_load_missing_digest(self, tmp_path: Path) -> None:
        lock_file = tmp_path / "kicad.lock.json"
        lock_file.write_text(json.dumps({"docker_image": "kicad/kicad:9.0.7"}))

        with pytest.raises(ValueError, match="docker_digest"):
            load_docker_image_ref(lock_file)

    def test_load_mismatched_digest(self, tmp_path: Path) -> None:
        lock_file = tmp_path / "kicad.lock.json"
        lock_file.write_text(
            json.dumps(
                {
                    "docker_image": "kicad/kicad:9.0.7@sha256:" + "b" * 64,
                    "docker_digest": "sha256:" + "c" * 64,
                }
            )
        )

        with pytest.raises(ValueError, match="does not match"):
            load_docker_image_ref(lock_file)

    def test_load_missing_file(self, tmp_path: Path) -> None:
        lock_file = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            load_docker_image_ref(lock_file)

    def test_load_missing_docker_image(self, tmp_path: Path) -> None:
        lock_file = tmp_path / "kicad.lock.json"
        lock_file.write_text(json.dumps({"kicad_version": "9.0.7"}))

        with pytest.raises(ValueError, match="docker_image not found"):
            load_docker_image_ref(lock_file)


class TestParseKicadVersion:
    """Tests for parse_kicad_version function."""

    def test_standard_version(self) -> None:
        assert parse_kicad_version("kicad-cli 9.0.7\n") == "9.0.7"

    def test_version_with_suffix(self) -> None:
        assert parse_kicad_version("kicad-cli 9.0.7-1\n") == "9.0.7-1"

    def test_version_only(self) -> None:
        assert parse_kicad_version("9.0.7") == "9.0.7"

    def test_multiline_output(self) -> None:
        output = "kicad-cli 9.0.7\nCopyright (C) KiCad Developers\n"
        assert parse_kicad_version(output) == "9.0.7"

    def test_no_version_pattern(self) -> None:
        assert parse_kicad_version("unknown output") == "unknown output"


class TestFromLockFile:
    """Tests for DockerKicadRunner.from_lock_file class method."""

    def test_from_lock_file(self, tmp_path: Path) -> None:
        lock_file = tmp_path / "kicad.lock.json"
        lock_file.write_text(
            json.dumps(
                {
                    "docker_image": "kicad/kicad:9.0.7",
                    "docker_digest": "sha256:" + "d" * 64,
                }
            )
        )

        runner = DockerKicadRunner.from_lock_file(lock_file)
        assert runner.docker_image == f"kicad/kicad:9.0.7@{'sha256:' + 'd' * 64}"


class TestDRCIntegration:
    """Tests for DRC-related functionality."""

    @patch("subprocess.run")
    def test_drc_run(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test running DRC via DockerKicadRunner."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )

        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")
        board_path = tmp_path / "board.kicad_pcb"
        report_path = tmp_path / "drc.json"

        # Build DRC args per design doc (CP-1.2, Section 13.1.2)
        drc_args = [
            "pcb",
            "drc",
            "--severity-all",
            "--exit-code-violations",
            "--format",
            "json",
            "--output",
            str(report_path.name),
            str(board_path.name),
        ]

        result = runner.run(drc_args, tmp_path)

        assert result.success is True
        assert result.returncode == 0

        # Verify the command structure
        call_args = mock_run.call_args[0][0]
        assert "docker" in call_args
        assert "--severity-all" in call_args
        assert "--exit-code-violations" in call_args
        assert "--format" in call_args
        assert "json" in call_args

    @patch("subprocess.run")
    def test_drc_violations_exit_code(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test DRC returns exit code 5 for violations."""
        mock_run.return_value = MagicMock(
            returncode=5,
            stdout="",
            stderr="DRC violations found",
        )

        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")
        result = runner.run(["pcb", "drc"], tmp_path)

        assert result.success is False
        assert result.has_drc_violations is True
        assert result.returncode == 5
