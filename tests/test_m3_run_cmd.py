"""Tests for the m3 run command.

These tests verify:
1. CLI argument parsing for run command
2. Dry run mode output
3. Run metadata generation and structure
4. Git info capture
5. Run type inference
6. Tag parsing
7. Integration with registry and artifact store
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from formula_foundry.m3.cli_main import (
    RunInputs,
    RunMetadata,
    RunOutputs,
    RunProvenance,
    StageExecution,
    ToolVersion,
    _get_git_info,
    _get_python_version,
    _infer_run_type,
    _parse_tags,
    build_parser,
    cmd_init,
    cmd_run,
)

if TYPE_CHECKING:
    pass


class TestBuildParserRun:
    """Tests for run command argument parser."""

    def test_parser_has_run_command(self) -> None:
        """Parser should have run subcommand."""
        parser = build_parser()
        args = parser.parse_args(["run", "generate_coupon"])
        assert args.command == "run"
        assert args.stage == "generate_coupon"

    def test_run_default_arguments(self) -> None:
        """Run command should have correct default arguments."""
        parser = build_parser()
        args = parser.parse_args(["run", "test_stage"])
        assert args.stage == "test_stage"
        assert args.root is None
        assert args.run_type is None
        assert args.dry_run is False
        assert args.force is False
        assert args.quiet is False
        assert args.tags == []

    def test_run_with_root(self, tmp_path: Path) -> None:
        """Run command should accept --root argument."""
        parser = build_parser()
        args = parser.parse_args(["run", "stage", "--root", str(tmp_path)])
        assert args.root == tmp_path

    def test_run_with_run_type(self) -> None:
        """Run command should accept --run-type argument."""
        parser = build_parser()
        args = parser.parse_args(["run", "stage", "--run-type", "em_simulation"])
        assert args.run_type == "em_simulation"

    def test_run_with_dry_run(self) -> None:
        """Run command should accept --dry-run flag."""
        parser = build_parser()
        args = parser.parse_args(["run", "stage", "--dry-run"])
        assert args.dry_run is True

    def test_run_with_force(self) -> None:
        """Run command should accept --force flag."""
        parser = build_parser()
        args = parser.parse_args(["run", "stage", "--force"])
        assert args.force is True

    def test_run_with_force_short(self) -> None:
        """Run command should accept -f flag."""
        parser = build_parser()
        args = parser.parse_args(["run", "stage", "-f"])
        assert args.force is True

    def test_run_with_quiet(self) -> None:
        """Run command should accept --quiet flag."""
        parser = build_parser()
        args = parser.parse_args(["run", "stage", "-q"])
        assert args.quiet is True

    def test_run_with_tags(self) -> None:
        """Run command should accept multiple --tag arguments."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "stage",
                "--tag",
                "env=test",
                "-t",
                "version=1.0",
            ]
        )
        assert args.tags == ["env=test", "version=1.0"]

    def test_run_stage_required(self) -> None:
        """Run command should require stage argument."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run"])


class TestRunTypeInference:
    """Tests for run type inference from stage name."""

    @pytest.mark.parametrize(
        "stage,expected",
        [
            ("generate_coupon", "coupon_generation"),
            ("coupon_gen", "coupon_generation"),
            ("run_drc", "drc_validation"),
            ("drc_check", "drc_validation"),
            ("run_simulation", "em_simulation"),
            ("em_sweep", "em_simulation"),
            ("sim_run", "em_simulation"),
            ("create_dataset", "dataset_build"),
            ("build_dataset", "dataset_build"),
            ("train_model", "model_training"),
            ("model_training", "model_training"),
            ("eval_model", "model_evaluation"),
            ("formula_discovery", "formula_discovery"),
            ("discover_formula", "formula_discovery"),
            ("validate_formula", "formula_validation"),
            ("export_gerbers", "export"),
            ("gerber_export", "export"),
            ("gc_sweep", "gc_sweep"),
            ("run_gc", "gc_sweep"),
            ("verify_data", "integrity_check"),
            ("integrity_check", "integrity_check"),
            ("random_stage", "other"),
            ("my_custom_step", "other"),
        ],
    )
    def test_infer_run_type(self, stage: str, expected: str) -> None:
        """Run type should be correctly inferred from stage name."""
        assert _infer_run_type(stage) == expected


class TestTagParsing:
    """Tests for tag argument parsing."""

    def test_parse_single_tag(self) -> None:
        """Should parse single KEY=VALUE tag."""
        tags = _parse_tags(["env=test"])
        assert tags == {"env": "test"}

    def test_parse_multiple_tags(self) -> None:
        """Should parse multiple tags."""
        tags = _parse_tags(["env=test", "version=1.0", "team=physics"])
        assert tags == {"env": "test", "version": "1.0", "team": "physics"}

    def test_parse_tag_with_equals_in_value(self) -> None:
        """Should handle values containing equals sign."""
        tags = _parse_tags(["equation=E=mc^2"])
        assert tags == {"equation": "E=mc^2"}

    def test_parse_empty_tags(self) -> None:
        """Should return empty dict for empty input."""
        tags = _parse_tags([])
        assert tags == {}

    def test_parse_invalid_tag_ignored(self) -> None:
        """Should ignore tags without equals sign."""
        tags = _parse_tags(["valid=tag", "invalid_tag", "also=valid"])
        assert tags == {"valid": "tag", "also": "valid"}


class TestGitInfo:
    """Tests for git information capture."""

    def test_git_info_in_repo(self, tmp_path: Path) -> None:
        """Should capture git info in a git repository."""
        # Create a minimal git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
        (tmp_path / "test.txt").write_text("test")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=tmp_path, check=True, capture_output=True)

        git_info = _get_git_info(tmp_path)

        assert git_info is not None
        assert len(git_info.commit) == 40
        assert all(c in "0123456789abcdef" for c in git_info.commit)
        # Branch could be main or master depending on git config
        assert git_info.branch in ["main", "master"]
        assert git_info.dirty is False

    def test_git_info_dirty_repo(self, tmp_path: Path) -> None:
        """Should detect dirty working directory."""
        # Create a minimal git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
        (tmp_path / "test.txt").write_text("test")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=tmp_path, check=True, capture_output=True)

        # Make it dirty
        (tmp_path / "dirty.txt").write_text("uncommitted")

        git_info = _get_git_info(tmp_path)

        assert git_info is not None
        assert git_info.dirty is True

    def test_git_info_not_in_repo(self, tmp_path: Path) -> None:
        """Should return None when not in a git repository."""
        git_info = _get_git_info(tmp_path)
        assert git_info is None


class TestPythonVersion:
    """Tests for Python version detection."""

    def test_get_python_version_format(self) -> None:
        """Should return version in X.Y.Z format."""
        version = _get_python_version()
        parts = version.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)


class TestRunMetadataDataclass:
    """Tests for RunMetadata dataclass serialization."""

    def test_run_metadata_to_dict(self) -> None:
        """RunMetadata should serialize to dict conforming to schema."""
        provenance = RunProvenance(
            hostname="test-host",
            git_commit="a" * 40,
            git_branch="main",
            git_dirty=False,
        )
        metadata = RunMetadata(
            run_id="run-20260120T120000-abc12345",
            run_type="em_simulation",
            status="completed",
            started_utc="2026-01-20T12:00:00Z",
            provenance=provenance,
            inputs=RunInputs(),
            outputs=RunOutputs(),
        )

        data = metadata.to_dict()

        assert data["schema_version"] == 1
        assert data["run_id"] == "run-20260120T120000-abc12345"
        assert data["run_type"] == "em_simulation"
        assert data["status"] == "completed"
        assert data["started_utc"] == "2026-01-20T12:00:00Z"
        assert data["provenance"]["hostname"] == "test-host"
        assert data["provenance"]["git_commit"] == "a" * 40
        assert data["inputs"]["artifacts"] == []
        assert data["outputs"]["artifacts"] == []

    def test_run_metadata_to_json(self) -> None:
        """RunMetadata should serialize to valid JSON."""
        provenance = RunProvenance(
            hostname="test-host",
            git_commit="a" * 40,
        )
        metadata = RunMetadata(
            run_id="run-test",
            run_type="other",
            status="completed",
            started_utc="2026-01-20T12:00:00Z",
            provenance=provenance,
            inputs=RunInputs(),
            outputs=RunOutputs(),
        )

        json_str = metadata.to_json()
        parsed = json.loads(json_str)

        assert parsed["run_id"] == "run-test"
        assert parsed["schema_version"] == 1

    def test_run_metadata_with_stages(self) -> None:
        """RunMetadata should include stage execution info."""
        provenance = RunProvenance(hostname="host", git_commit="a" * 40)
        stage = StageExecution(
            stage_name="test_stage",
            status="completed",
            started_utc="2026-01-20T12:00:00Z",
            finished_utc="2026-01-20T12:05:00Z",
            duration_seconds=300.0,
            cached=False,
        )
        metadata = RunMetadata(
            run_id="run-test",
            run_type="other",
            status="completed",
            started_utc="2026-01-20T12:00:00Z",
            provenance=provenance,
            inputs=RunInputs(),
            outputs=RunOutputs(),
            stages=[stage],
        )

        data = metadata.to_dict()

        assert len(data["stages"]) == 1
        assert data["stages"][0]["stage_name"] == "test_stage"
        assert data["stages"][0]["duration_seconds"] == 300.0

    def test_run_metadata_with_tags(self) -> None:
        """RunMetadata should include tags."""
        provenance = RunProvenance(hostname="host", git_commit="a" * 40)
        metadata = RunMetadata(
            run_id="run-test",
            run_type="other",
            status="completed",
            started_utc="2026-01-20T12:00:00Z",
            provenance=provenance,
            inputs=RunInputs(),
            outputs=RunOutputs(),
            tags={"env": "test", "version": "1.0"},
        )

        data = metadata.to_dict()

        assert data["tags"]["env"] == "test"
        assert data["tags"]["version"] == "1.0"


class TestToolVersion:
    """Tests for ToolVersion dataclass."""

    def test_tool_version_basic(self) -> None:
        """ToolVersion should serialize basic info."""
        tv = ToolVersion(name="python", version="3.11.5")
        data = tv.to_dict()
        assert data == {"name": "python", "version": "3.11.5"}

    def test_tool_version_with_commit(self) -> None:
        """ToolVersion should include commit_sha if present."""
        tv = ToolVersion(name="tool", version="1.0", commit_sha="abc123" + "0" * 34)
        data = tv.to_dict()
        assert data["commit_sha"] == "abc123" + "0" * 34

    def test_tool_version_with_docker(self) -> None:
        """ToolVersion should include docker_image if present."""
        tv = ToolVersion(
            name="kicad",
            version="9.0.7",
            docker_image="kicad/kicad:9.0.7@sha256:abc123",
        )
        data = tv.to_dict()
        assert data["docker_image"] == "kicad/kicad:9.0.7@sha256:abc123"


class TestCmdRunPrerequisites:
    """Tests for cmd_run prerequisite checks."""

    def test_run_fails_without_init(self, tmp_path: Path) -> None:
        """Run should fail if M3 is not initialized."""
        # Create git repo but no M3 init
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
        (tmp_path / "test.txt").write_text("test")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=tmp_path, check=True, capture_output=True)
        (tmp_path / "dvc.yaml").write_text("stages: {}")

        result = cmd_run(
            stage="test",
            root=tmp_path,
            run_type=None,
            dry_run=False,
            force=False,
            quiet=True,
            tags=[],
        )

        assert result == 2

    def test_run_fails_without_git(self, tmp_path: Path) -> None:
        """Run should fail if not in a git repository."""
        # Init M3 but no git
        cmd_init(root=tmp_path, force=False, quiet=True)
        (tmp_path / "dvc.yaml").write_text("stages: {}")

        result = cmd_run(
            stage="test",
            root=tmp_path,
            run_type=None,
            dry_run=False,
            force=False,
            quiet=True,
            tags=[],
        )

        assert result == 2

    def test_run_fails_without_dvc_yaml(self, tmp_path: Path) -> None:
        """Run should fail if dvc.yaml is not found."""
        # Init M3 and git but no dvc.yaml
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
        (tmp_path / "test.txt").write_text("test")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=tmp_path, check=True, capture_output=True)
        cmd_init(root=tmp_path, force=False, quiet=True)

        result = cmd_run(
            stage="test",
            root=tmp_path,
            run_type=None,
            dry_run=False,
            force=False,
            quiet=True,
            tags=[],
        )

        assert result == 2


class TestCmdRunDryRun:
    """Tests for cmd_run dry run mode."""

    def test_dry_run_returns_zero(self, tmp_path: Path) -> None:
        """Dry run should return 0 without executing."""
        # Setup minimal environment
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
        (tmp_path / "test.txt").write_text("test")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=tmp_path, check=True, capture_output=True)
        cmd_init(root=tmp_path, force=False, quiet=True)
        (tmp_path / "dvc.yaml").write_text("stages:\n  test_stage:\n    cmd: echo hello")

        result = cmd_run(
            stage="test_stage",
            root=tmp_path,
            run_type=None,
            dry_run=True,
            force=False,
            quiet=False,
            tags=["env=test"],
        )

        assert result == 0

    def test_dry_run_does_not_create_artifacts(self, tmp_path: Path) -> None:
        """Dry run should not create any artifacts."""
        # Setup minimal environment
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
        (tmp_path / "test.txt").write_text("test")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=tmp_path, check=True, capture_output=True)
        cmd_init(root=tmp_path, force=False, quiet=True)
        (tmp_path / "dvc.yaml").write_text("stages:\n  test_stage:\n    cmd: echo hello")

        # Count runs before
        conn = sqlite3.connect(str(tmp_path / "data" / "registry.db"))
        cursor = conn.execute("SELECT COUNT(*) FROM runs WHERE stage_name = 'test_stage'")
        count_before = cursor.fetchone()[0]
        conn.close()

        cmd_run(
            stage="test_stage",
            root=tmp_path,
            run_type=None,
            dry_run=True,
            force=False,
            quiet=True,
            tags=[],
        )

        # Count runs after
        conn = sqlite3.connect(str(tmp_path / "data" / "registry.db"))
        cursor = conn.execute("SELECT COUNT(*) FROM runs WHERE stage_name = 'test_stage'")
        count_after = cursor.fetchone()[0]
        conn.close()

        assert count_after == count_before


class TestCmdRunWithMockedDVC:
    """Tests for cmd_run with mocked DVC execution."""

    def _setup_test_env(self, tmp_path: Path) -> None:
        """Set up a minimal test environment with git and M3."""
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
        (tmp_path / "test.txt").write_text("test")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=tmp_path, check=True, capture_output=True)
        cmd_init(root=tmp_path, force=False, quiet=True)
        (tmp_path / "dvc.yaml").write_text("stages:\n  test_stage:\n    cmd: echo hello")

    @patch("formula_foundry.m3.cli_main._run_dvc_stage")
    @patch("formula_foundry.m3.cli_main.shutil.which")
    def test_run_success_creates_run_record(self, mock_which: MagicMock, mock_dvc: MagicMock, tmp_path: Path) -> None:
        """Successful run should create a run record in registry."""
        self._setup_test_env(tmp_path)
        mock_which.return_value = "/usr/bin/dvc"
        mock_dvc.return_value = (0, "Running stage 'test_stage'\nCompleted", "", False)

        result = cmd_run(
            stage="test_stage",
            root=tmp_path,
            run_type=None,
            dry_run=False,
            force=False,
            quiet=True,
            tags=[],
        )

        assert result == 0

        # Verify run was recorded
        conn = sqlite3.connect(str(tmp_path / "data" / "registry.db"))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM runs WHERE stage_name = 'test_stage'")
        runs = cursor.fetchall()
        conn.close()

        # Should have at least one run for this stage
        assert len(runs) >= 1
        run = runs[-1]  # Most recent
        assert run["status"] == "completed"
        assert run["generator"] == "m3_run"

    @patch("formula_foundry.m3.cli_main._run_dvc_stage")
    @patch("formula_foundry.m3.cli_main.shutil.which")
    def test_run_creates_metadata_artifact(self, mock_which: MagicMock, mock_dvc: MagicMock, tmp_path: Path) -> None:
        """Run should create a metadata artifact in the store."""
        self._setup_test_env(tmp_path)
        mock_which.return_value = "/usr/bin/dvc"
        mock_dvc.return_value = (0, "Completed", "", False)

        cmd_run(
            stage="test_stage",
            root=tmp_path,
            run_type=None,
            dry_run=False,
            force=False,
            quiet=True,
            tags=[],
        )

        # Verify artifact was created
        manifests_dir = tmp_path / "data" / "manifests"
        manifest_files = list(manifests_dir.glob("run-*-metadata.json"))
        assert len(manifest_files) >= 1

        # Verify manifest content
        manifest = json.loads(manifest_files[-1].read_text())
        assert manifest["artifact_type"] == "log"
        assert "metadata" in manifest["roles"]

    @patch("formula_foundry.m3.cli_main._run_dvc_stage")
    @patch("formula_foundry.m3.cli_main.shutil.which")
    def test_run_failure_records_error(self, mock_which: MagicMock, mock_dvc: MagicMock, tmp_path: Path) -> None:
        """Failed run should record error in metadata."""
        self._setup_test_env(tmp_path)
        mock_which.return_value = "/usr/bin/dvc"
        mock_dvc.return_value = (1, "", "Error: Stage failed", False)

        result = cmd_run(
            stage="test_stage",
            root=tmp_path,
            run_type=None,
            dry_run=False,
            force=False,
            quiet=True,
            tags=[],
        )

        assert result == 1

        # Verify run was recorded as failed
        conn = sqlite3.connect(str(tmp_path / "data" / "registry.db"))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM runs WHERE stage_name = 'test_stage' ORDER BY indexed_utc DESC")
        run = cursor.fetchone()
        conn.close()

        assert run["status"] == "failed"

    @patch("formula_foundry.m3.cli_main._run_dvc_stage")
    @patch("formula_foundry.m3.cli_main.shutil.which")
    def test_run_with_custom_run_type(self, mock_which: MagicMock, mock_dvc: MagicMock, tmp_path: Path) -> None:
        """Run should use custom run_type when provided."""
        self._setup_test_env(tmp_path)
        mock_which.return_value = "/usr/bin/dvc"
        mock_dvc.return_value = (0, "Completed", "", False)

        cmd_run(
            stage="test_stage",
            root=tmp_path,
            run_type="em_simulation",
            dry_run=False,
            force=False,
            quiet=True,
            tags=[],
        )

        # Verify config has correct run_type
        conn = sqlite3.connect(str(tmp_path / "data" / "registry.db"))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM runs WHERE stage_name = 'test_stage' ORDER BY indexed_utc DESC")
        run = cursor.fetchone()
        conn.close()

        config = json.loads(run["config"])
        assert config["run_type"] == "em_simulation"

    @patch("formula_foundry.m3.cli_main._run_dvc_stage")
    @patch("formula_foundry.m3.cli_main.shutil.which")
    def test_run_cached_stage(self, mock_which: MagicMock, mock_dvc: MagicMock, tmp_path: Path) -> None:
        """Cached stage should be recorded with cached status."""
        self._setup_test_env(tmp_path)
        mock_which.return_value = "/usr/bin/dvc"
        mock_dvc.return_value = (0, "Stage 'test_stage' didn't change, skipping", "", True)

        cmd_run(
            stage="test_stage",
            root=tmp_path,
            run_type=None,
            dry_run=False,
            force=False,
            quiet=True,
            tags=[],
        )

        # Check the metadata artifact for cached status
        manifests_dir = tmp_path / "data" / "manifests"
        manifest_files = list(manifests_dir.glob("run-*-metadata.json"))
        assert len(manifest_files) >= 1

        # Read the actual artifact content
        manifest = json.loads(manifest_files[-1].read_text())
        objects_dir = tmp_path / "data" / "objects"
        digest = manifest["content_hash"]["digest"]
        object_path = objects_dir / digest[:2] / digest
        metadata = json.loads(object_path.read_text())

        assert metadata["stages"][0]["cached"] is True
        assert metadata["stages"][0]["status"] == "cached"


class TestMainWithRunCommand:
    """Tests for main() with run command."""

    def test_main_run_with_args(self, tmp_path: Path) -> None:
        """Main should handle run command arguments."""
        # Just test that arguments are parsed correctly
        # Full execution tested in other tests
        from formula_foundry.m3.cli_main import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "my_stage",
                "--root",
                str(tmp_path),
                "--run-type",
                "em_simulation",
                "--dry-run",
                "--force",
                "-q",
                "-t",
                "env=test",
            ]
        )

        assert args.command == "run"
        assert args.stage == "my_stage"
        assert args.root == tmp_path
        assert args.run_type == "em_simulation"
        assert args.dry_run is True
        assert args.force is True
        assert args.quiet is True
        assert args.tags == ["env=test"]
