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
    DockerMountError,
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

    def test_build_docker_command_includes_user_flag(self, tmp_path: Path) -> None:
        """Docker command must include --user flag for bind-mount permissions.

        This is critical for CI environments where the host user ID (e.g., 1001
        on GitHub Actions) differs from the container user (uid 1000 in kicad/kicad).
        Without --user, the container cannot write to bind-mounted directories.
        """
        import os
        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")
        cmd = runner._build_docker_command(["--version"], tmp_path)

        # Verify --user flag is present with host UID:GID
        assert "--user" in cmd, "Docker command must include --user flag"
        user_idx = cmd.index("--user")
        user_value = cmd[user_idx + 1]

        # Verify format is uid:gid
        assert ":" in user_value, "--user value must be uid:gid format"
        uid, gid = user_value.split(":")
        assert uid.isdigit(), "UID must be numeric"
        assert gid.isdigit(), "GID must be numeric"

        # Verify it matches host user
        assert int(uid) == os.getuid(), f"UID should be {os.getuid()}, got {uid}"
        assert int(gid) == os.getgid(), f"GID should be {os.getgid()}, got {gid}"

    def test_build_docker_command_sets_home_env(self, tmp_path: Path) -> None:
        """Docker command must set HOME=/tmp for numeric UID without passwd entry.

        When running as --user uid:gid, the numeric UID may not have a passwd
        entry in the container. KiCad needs a writable HOME for config files.
        """
        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")
        cmd = runner._build_docker_command(["--version"], tmp_path)

        # Find all -e flags and their values
        env_values = []
        for i, arg in enumerate(cmd):
            if arg == "-e" and i + 1 < len(cmd):
                env_values.append(cmd[i + 1])

        assert "HOME=/tmp" in env_values, "Docker command must set HOME=/tmp"

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


class TestDockerMountError:
    """Tests for DockerMountError exception."""

    def test_error_message_includes_host_path(self) -> None:
        """Error message should include the host path."""
        host_path = Path("/some/test/path")
        error = DockerMountError(host_path)
        assert "/some/test/path" in str(error)

    def test_error_message_includes_expected_file(self) -> None:
        """Error message should include expected file if provided."""
        host_path = Path("/some/test/path")
        error = DockerMountError(host_path, "coupon.kicad_pcb")
        assert "coupon.kicad_pcb" in str(error)

    def test_error_message_includes_container_mount(self) -> None:
        """Error message should mention /workspace mount point."""
        host_path = Path("/some/test/path")
        error = DockerMountError(host_path)
        assert "/workspace" in str(error)

    def test_attributes_set_correctly(self) -> None:
        """Error should store host_path and expected_file attributes."""
        host_path = Path("/some/test/path")
        error = DockerMountError(host_path, "test.kicad_pcb")
        assert error.host_path == host_path
        assert error.expected_file == "test.kicad_pcb"


class TestDRCViolationSummary:
    """Tests for DRC violation summary on rc=5."""

    def test_summarize_empty_violations(self, tmp_path: Path) -> None:
        """Summary should handle empty violations list."""
        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")

        # Create a minimal DRC JSON
        drc_json = {
            "violations": [],
            "unconnected_items": [],
        }
        drc_path = tmp_path / "drc.json"
        drc_path.write_text(json.dumps(drc_json))

        summary = runner._summarize_drc_violations(drc_path, tmp_path)
        assert "Total violations: 0" in summary
        assert "Unconnected items: 0" in summary

    def test_summarize_with_violations(self, tmp_path: Path) -> None:
        """Summary should count and categorize violations."""
        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")

        # Create a DRC JSON with violations
        drc_json = {
            "violations": [
                {"type": "clearance", "severity": "error", "description": "Clearance violation"},
                {"type": "clearance", "severity": "error", "description": "Another clearance"},
                {"type": "track_dangling", "severity": "warning", "description": "Dangling track"},
            ],
            "unconnected_items": [
                {"type": "unconnected_items", "severity": "error", "description": "Missing connection"},
            ],
        }
        drc_path = tmp_path / "drc.json"
        drc_path.write_text(json.dumps(drc_json))

        summary = runner._summarize_drc_violations(drc_path, tmp_path)
        assert "Total violations: 3" in summary
        assert "Unconnected items: 1" in summary
        assert "error:" in summary.lower()
        assert "clearance: 2" in summary

    def test_summarize_missing_file(self, tmp_path: Path) -> None:
        """Summary should handle missing DRC file gracefully."""
        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")

        drc_path = tmp_path / "nonexistent.json"
        summary = runner._summarize_drc_violations(drc_path, tmp_path)
        assert "unavailable" in summary.lower() or "not found" in summary.lower()


class TestMountVerification:
    """Tests for Docker mount sanity verification.

    These tests verify that the mount verification logic correctly detects
    empty or inaccessible bind mounts before running kicad-cli.
    """

    @patch("subprocess.run")
    def test_verify_mount_detects_empty_workspace(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Mount verification should raise DockerMountError for empty workspace.

        Regression test: empty /workspace (only . and ..) should raise a clear
        error rather than letting kicad-cli fail with misleading rc=3.
        """
        # Simulate empty workspace (only total, ., and ..)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="total 8\ndrwxr-xr-x 2 root root 4096 Jan 21 00:00 .\ndrwxr-xr-x 3 root root 4096 Jan 21 00:00 ..\n",
            stderr="",
        )

        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")

        with pytest.raises(DockerMountError) as exc_info:
            runner._verify_mount(tmp_path, "coupon.kicad_pcb")

        # Error should mention the host path
        assert str(tmp_path) in str(exc_info.value) or "Docker" in str(exc_info.value)

    @patch("subprocess.run")
    def test_verify_mount_passes_with_files(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Mount verification should pass when workspace has files."""
        def side_effect(cmd, **kwargs):
            if "ls" in cmd:
                return MagicMock(
                    returncode=0,
                    stdout="total 16\ndrwxr-xr-x 2 root root 4096 Jan 21 00:00 .\ndrwxr-xr-x 3 root root 4096 Jan 21 00:00 ..\n-rw-r--r-- 1 root root 1234 Jan 21 00:00 coupon.kicad_pcb\n",
                    stderr="",
                )
            elif "test" in cmd:
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect
        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")

        # Should not raise
        runner._verify_mount(tmp_path, "coupon.kicad_pcb")

    @patch("subprocess.run")
    def test_verify_mount_fails_when_file_missing(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Mount verification should raise when expected file is missing."""
        def side_effect(cmd, **kwargs):
            if "ls" in cmd:
                return MagicMock(
                    returncode=0,
                    stdout="total 16\ndrwxr-xr-x 2 root root 4096 Jan 21 00:00 .\ndrwxr-xr-x 3 root root 4096 Jan 21 00:00 ..\n-rw-r--r-- 1 root root 1234 Jan 21 00:00 other_file.txt\n",
                    stderr="",
                )
            elif "test" in cmd:
                return MagicMock(returncode=1, stdout="", stderr="")  # File not found
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect
        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")

        with pytest.raises(DockerMountError) as exc_info:
            runner._verify_mount(tmp_path, "coupon.kicad_pcb")

        assert "coupon.kicad_pcb" in str(exc_info.value)


class TestDRCDebugDiagnostics:
    """Tests for rc=3 debug diagnostics."""

    @patch("subprocess.run")
    def test_diagnostics_included_on_rc3(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Debug diagnostics should be appended to stderr on rc=3."""
        # First call is the main kicad-cli command
        call_count = [0]

        def side_effect(cmd, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Main kicad-cli call fails with rc=3
                return MagicMock(
                    returncode=3,
                    stdout="",
                    stderr="Failed to load board\n",
                )
            else:
                # Diagnostic commands
                return MagicMock(returncode=0, stdout="diagnostic output\n", stderr="")

        mock_run.side_effect = side_effect
        runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")
        result = runner.run(["pcb", "drc"], tmp_path)

        assert result.returncode == 3
        # Diagnostics should be in stderr
        assert "DEBUG DIAGNOSTICS" in result.stderr or "diagnostic" in result.stderr.lower()
