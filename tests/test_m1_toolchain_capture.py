"""Unit tests for toolchain provenance capture (CP-5.3).

These tests verify that toolchain provenance is always captured correctly,
especially for docker builds where 'unknown' values are not allowed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from formula_foundry.coupongen.toolchain_capture import (
    ToolchainProvenance,
    ToolchainProvenanceError,
    capture_toolchain_provenance,
)


class TestToolchainProvenance:
    """Tests for ToolchainProvenance dataclass."""

    def test_to_metadata_local_mode(self) -> None:
        """Test metadata conversion for local mode."""
        provenance = ToolchainProvenance(
            mode="local",
            kicad_version="9.0.7",
            kicad_cli_version="9.0.7",
            docker_image_ref=None,
            generator_git_sha="abc123" + "0" * 34,
        )

        meta = provenance.to_metadata()

        assert meta["mode"] == "local"
        assert meta["kicad"]["version"] == "9.0.7"
        assert meta["kicad"]["cli_version_output"] == "9.0.7"
        assert meta["generator_git_sha"] == "abc123" + "0" * 34
        assert "docker" not in meta

    def test_to_metadata_docker_mode(self) -> None:
        """Test metadata conversion for docker mode."""
        provenance = ToolchainProvenance(
            mode="docker",
            kicad_version="9.0.7",
            kicad_cli_version="9.0.7",
            docker_image_ref="kicad/kicad:9.0.7@sha256:abc123",
            generator_git_sha="def456" + "0" * 34,
        )

        meta = provenance.to_metadata()

        assert meta["mode"] == "docker"
        assert meta["kicad"]["version"] == "9.0.7"
        assert meta["kicad"]["cli_version_output"] == "9.0.7"
        assert meta["docker"]["image_ref"] == "kicad/kicad:9.0.7@sha256:abc123"
        assert meta["generator_git_sha"] == "def456" + "0" * 34


class TestCaptureToolchainProvenanceLocal:
    """Tests for capture_toolchain_provenance in local mode."""

    @patch("formula_foundry.coupongen.toolchain_capture.get_kicad_cli_version")
    @patch("formula_foundry.coupongen.toolchain_capture.get_git_sha")
    def test_local_mode_success(
        self, mock_git_sha: MagicMock, mock_cli_version: MagicMock, tmp_path: Path
    ) -> None:
        """Test successful local mode provenance capture."""
        mock_git_sha.return_value = "abc123" + "0" * 34
        mock_cli_version.return_value = "9.0.7"

        provenance = capture_toolchain_provenance(
            mode="local",
            kicad_version="9.0.7",
            workdir=tmp_path,
        )

        assert provenance.mode == "local"
        assert provenance.kicad_version == "9.0.7"
        assert provenance.kicad_cli_version == "9.0.7"
        assert provenance.docker_image_ref is None
        assert provenance.generator_git_sha == "abc123" + "0" * 34

    @patch("formula_foundry.coupongen.toolchain_capture.get_kicad_cli_version")
    @patch("formula_foundry.coupongen.toolchain_capture.get_git_sha")
    def test_local_mode_allows_unknown(
        self, mock_git_sha: MagicMock, mock_cli_version: MagicMock, tmp_path: Path
    ) -> None:
        """Test that local mode allows 'unknown' kicad-cli version (fallback)."""
        mock_git_sha.return_value = "abc123" + "0" * 34
        mock_cli_version.side_effect = RuntimeError("kicad-cli not found")

        provenance = capture_toolchain_provenance(
            mode="local",
            kicad_version="9.0.7",
            workdir=tmp_path,
        )

        assert provenance.kicad_cli_version == "unknown"


class TestCaptureToolchainProvenanceDocker:
    """Tests for capture_toolchain_provenance in docker mode."""

    @patch("formula_foundry.coupongen.toolchain_capture.DockerKicadRunner")
    @patch("formula_foundry.coupongen.toolchain_capture.load_docker_image_ref")
    @patch("formula_foundry.coupongen.toolchain_capture.get_git_sha")
    def test_docker_mode_success(
        self,
        mock_git_sha: MagicMock,
        mock_load_ref: MagicMock,
        mock_runner_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test successful docker mode provenance capture."""
        mock_git_sha.return_value = "abc123" + "0" * 34
        mock_load_ref.return_value = "kicad/kicad:9.0.7@sha256:abc123"

        mock_runner = MagicMock()
        mock_runner.kicad_cli_version.return_value = "9.0.7"
        mock_runner_class.return_value = mock_runner

        provenance = capture_toolchain_provenance(
            mode="docker",
            kicad_version="9.0.7",
            docker_image="kicad/kicad:9.0.7",
            workdir=tmp_path,
        )

        assert provenance.mode == "docker"
        assert provenance.kicad_version == "9.0.7"
        assert provenance.kicad_cli_version == "9.0.7"
        assert provenance.docker_image_ref == "kicad/kicad:9.0.7@sha256:abc123"
        assert provenance.generator_git_sha == "abc123" + "0" * 34

    @patch("formula_foundry.coupongen.toolchain_capture.DockerKicadRunner")
    @patch("formula_foundry.coupongen.toolchain_capture.load_docker_image_ref")
    @patch("formula_foundry.coupongen.toolchain_capture.get_git_sha")
    def test_docker_mode_rejects_unknown(
        self,
        mock_git_sha: MagicMock,
        mock_load_ref: MagicMock,
        mock_runner_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that docker mode rejects 'unknown' kicad-cli version (CP-5.3)."""
        mock_git_sha.return_value = "abc123" + "0" * 34
        mock_load_ref.return_value = "kicad/kicad:9.0.7@sha256:abc123"

        mock_runner = MagicMock()
        mock_runner.kicad_cli_version.return_value = "unknown"
        mock_runner_class.return_value = mock_runner

        with pytest.raises(
            ToolchainProvenanceError,
            match="Docker builds must have valid kicad-cli version",
        ):
            capture_toolchain_provenance(
                mode="docker",
                kicad_version="9.0.7",
                docker_image="kicad/kicad:9.0.7",
                workdir=tmp_path,
            )

    @patch("formula_foundry.coupongen.toolchain_capture.DockerKicadRunner")
    @patch("formula_foundry.coupongen.toolchain_capture.load_docker_image_ref")
    @patch("formula_foundry.coupongen.toolchain_capture.get_git_sha")
    def test_docker_mode_rejects_empty_version(
        self,
        mock_git_sha: MagicMock,
        mock_load_ref: MagicMock,
        mock_runner_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that docker mode rejects empty kicad-cli version."""
        mock_git_sha.return_value = "abc123" + "0" * 34
        mock_load_ref.return_value = "kicad/kicad:9.0.7@sha256:abc123"

        mock_runner = MagicMock()
        mock_runner.kicad_cli_version.return_value = ""
        mock_runner_class.return_value = mock_runner

        with pytest.raises(
            ToolchainProvenanceError,
            match="Docker builds must have valid kicad-cli version",
        ):
            capture_toolchain_provenance(
                mode="docker",
                kicad_version="9.0.7",
                docker_image="kicad/kicad:9.0.7",
                workdir=tmp_path,
            )

    @patch("formula_foundry.coupongen.toolchain_capture.DockerKicadRunner")
    @patch("formula_foundry.coupongen.toolchain_capture.load_docker_image_ref")
    @patch("formula_foundry.coupongen.toolchain_capture.get_git_sha")
    def test_docker_mode_handles_runner_failure(
        self,
        mock_git_sha: MagicMock,
        mock_load_ref: MagicMock,
        mock_runner_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that docker mode raises error on runner failure."""
        mock_git_sha.return_value = "abc123" + "0" * 34
        mock_load_ref.return_value = "kicad/kicad:9.0.7@sha256:abc123"

        mock_runner = MagicMock()
        mock_runner.kicad_cli_version.side_effect = RuntimeError("Docker not found")
        mock_runner_class.return_value = mock_runner

        with pytest.raises(
            ToolchainProvenanceError,
            match="Failed to capture kicad-cli version in docker mode",
        ):
            capture_toolchain_provenance(
                mode="docker",
                kicad_version="9.0.7",
                docker_image="kicad/kicad:9.0.7",
                workdir=tmp_path,
            )

    @patch("formula_foundry.coupongen.toolchain_capture.get_git_sha")
    def test_docker_mode_without_lock_file_or_image(
        self, mock_git_sha: MagicMock, tmp_path: Path
    ) -> None:
        """Test that docker mode requires either lock file or docker_image."""
        mock_git_sha.return_value = "abc123" + "0" * 34

        # Create a non-existent lock file path
        fake_lock = tmp_path / "nonexistent.json"

        with pytest.raises(
            ToolchainProvenanceError,
            match="Docker mode requires toolchain lock file or docker_image parameter",
        ):
            capture_toolchain_provenance(
                mode="docker",
                kicad_version="9.0.7",
                docker_image=None,  # No docker_image provided
                workdir=tmp_path,
                lock_file=fake_lock,  # Lock file doesn't exist
            )

    @patch("formula_foundry.coupongen.toolchain_capture.DockerKicadRunner")
    @patch("formula_foundry.coupongen.toolchain_capture.get_git_sha")
    def test_docker_mode_fallback_to_docker_image_param(
        self,
        mock_git_sha: MagicMock,
        mock_runner_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that docker mode falls back to docker_image param if lock file not found."""
        mock_git_sha.return_value = "abc123" + "0" * 34

        mock_runner = MagicMock()
        mock_runner.kicad_cli_version.return_value = "9.0.7"
        mock_runner_class.return_value = mock_runner

        # Create a non-existent lock file path
        fake_lock = tmp_path / "nonexistent.json"

        provenance = capture_toolchain_provenance(
            mode="docker",
            kicad_version="9.0.7",
            docker_image="kicad/kicad:9.0.7",  # Provided as fallback
            workdir=tmp_path,
            lock_file=fake_lock,  # Lock file doesn't exist
        )

        assert provenance.docker_image_ref == "kicad/kicad:9.0.7"


class TestCaptureWithLockFile:
    """Tests for capture_toolchain_provenance with lock file."""

    @patch("formula_foundry.coupongen.toolchain_capture.DockerKicadRunner")
    @patch("formula_foundry.coupongen.toolchain_capture.get_git_sha")
    def test_loads_image_ref_with_digest(
        self,
        mock_git_sha: MagicMock,
        mock_runner_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that lock file with digest produces full image ref."""
        mock_git_sha.return_value = "abc123" + "0" * 34

        # Create lock file with digest
        lock_file = tmp_path / "kicad.lock.json"
        lock_file.write_text(
            json.dumps(
                {
                    "docker_image": "kicad/kicad:9.0.7",
                    "docker_digest": "sha256:abc123def456",
                }
            )
        )

        mock_runner = MagicMock()
        mock_runner.kicad_cli_version.return_value = "9.0.7"
        mock_runner_class.return_value = mock_runner

        provenance = capture_toolchain_provenance(
            mode="docker",
            kicad_version="9.0.7",
            workdir=tmp_path,
            lock_file=lock_file,
        )

        assert provenance.docker_image_ref == "kicad/kicad:9.0.7@sha256:abc123def456"


class TestToolchainMetadataFormat:
    """Tests for the toolchain metadata format in manifest."""

    def test_metadata_has_required_fields(self) -> None:
        """Test that metadata has all required fields per CP-5.1."""
        provenance = ToolchainProvenance(
            mode="docker",
            kicad_version="9.0.7",
            kicad_cli_version="9.0.7",
            docker_image_ref="kicad/kicad:9.0.7@sha256:abc123",
            generator_git_sha="def456" + "0" * 34,
        )

        meta = provenance.to_metadata()

        # Required fields per design doc section 13.5.1
        assert "kicad" in meta
        assert "version" in meta["kicad"]
        assert "cli_version_output" in meta["kicad"]
        assert "mode" in meta
        assert "generator_git_sha" in meta

        # Docker-specific fields
        assert "docker" in meta
        assert "image_ref" in meta["docker"]
