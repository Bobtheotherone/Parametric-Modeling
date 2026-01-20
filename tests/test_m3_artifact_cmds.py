"""Tests for the m3 artifact show and artifact list commands.

These tests verify:
1. CLI argument parsing for artifact show and artifact list
2. artifact show displays correct information and handles errors
3. artifact list filtering, pagination, and output formats
4. Integration with ArtifactStore and ArtifactRegistry
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from formula_foundry.m3.artifact_store import ArtifactStore
from formula_foundry.m3.cli_main import (
    build_parser,
    cmd_artifact_list,
    cmd_artifact_show,
    cmd_init,
    main,
)
from formula_foundry.m3.registry import ArtifactRegistry

if TYPE_CHECKING:
    pass


class TestBuildParserArtifact:
    """Tests for artifact command argument parser construction."""

    def test_parser_has_artifact_command(self) -> None:
        """Parser should have artifact subcommand."""
        parser = build_parser()
        args = parser.parse_args(["artifact", "show", "test-id"])
        assert args.command == "artifact"
        assert args.artifact_command == "show"

    def test_artifact_show_default_arguments(self) -> None:
        """artifact show should have correct default arguments."""
        parser = build_parser()
        args = parser.parse_args(["artifact", "show", "test-artifact-id"])
        assert args.artifact_id == "test-artifact-id"
        assert args.root is None
        assert args.output_json is False
        assert args.content is False
        assert args.verify is False
        assert args.quiet is False

    def test_artifact_show_with_all_options(self) -> None:
        """artifact show should accept all options."""
        parser = build_parser()
        args = parser.parse_args([
            "artifact", "show", "test-id",
            "--json", "--content", "--verify", "-q",
        ])
        assert args.output_json is True
        assert args.content is True
        assert args.verify is True
        assert args.quiet is True

    def test_artifact_list_default_arguments(self) -> None:
        """artifact list should have correct default arguments."""
        parser = build_parser()
        args = parser.parse_args(["artifact", "list"])
        assert args.artifact_type is None
        assert args.run_id is None
        assert args.roles == []
        assert args.created_after is None
        assert args.created_before is None
        assert args.limit is None
        assert args.offset == 0
        assert args.order_by == "created_utc"
        assert args.order_asc is False
        assert args.output_json is False
        assert args.long_format is False
        assert args.quiet is False

    def test_artifact_list_with_type_filter(self) -> None:
        """artifact list should accept --type filter."""
        parser = build_parser()
        args = parser.parse_args(["artifact", "list", "-t", "touchstone"])
        assert args.artifact_type == "touchstone"

    def test_artifact_list_with_run_filter(self) -> None:
        """artifact list should accept --run filter."""
        parser = build_parser()
        args = parser.parse_args(["artifact", "list", "-r", "run-123"])
        assert args.run_id == "run-123"

    def test_artifact_list_with_role_filter(self) -> None:
        """artifact list should accept multiple --role filters."""
        parser = build_parser()
        args = parser.parse_args([
            "artifact", "list",
            "--role", "metadata",
            "--role", "validation",
        ])
        assert args.roles == ["metadata", "validation"]

    def test_artifact_list_with_date_filters(self) -> None:
        """artifact list should accept --after and --before filters."""
        parser = build_parser()
        args = parser.parse_args([
            "artifact", "list",
            "--after", "2025-01-01T00:00:00Z",
            "--before", "2025-12-31T23:59:59Z",
        ])
        assert args.created_after == "2025-01-01T00:00:00Z"
        assert args.created_before == "2025-12-31T23:59:59Z"

    def test_artifact_list_with_pagination(self) -> None:
        """artifact list should accept pagination options."""
        parser = build_parser()
        args = parser.parse_args([
            "artifact", "list",
            "-n", "50",
            "--offset", "100",
        ])
        assert args.limit == 50
        assert args.offset == 100

    def test_artifact_list_with_ordering(self) -> None:
        """artifact list should accept ordering options."""
        parser = build_parser()
        args = parser.parse_args([
            "artifact", "list",
            "--order-by", "byte_size",
            "--asc",
        ])
        assert args.order_by == "byte_size"
        assert args.order_asc is True

    def test_artifact_list_with_output_options(self) -> None:
        """artifact list should accept output format options."""
        parser = build_parser()
        args = parser.parse_args(["artifact", "list", "--json", "-l"])
        assert args.output_json is True
        assert args.long_format is True


@pytest.fixture
def initialized_project(tmp_path: Path) -> Path:
    """Create an initialized M3 project for testing."""
    cmd_init(root=tmp_path, force=False, quiet=True)
    return tmp_path


@pytest.fixture
def project_with_artifacts(initialized_project: Path) -> Path:
    """Create an initialized project with sample artifacts."""
    data_dir = initialized_project / "data"
    store = ArtifactStore(
        root=data_dir,
        generator="test",
        generator_version="1.0.0",
    )

    # Create several artifacts with different types and tags
    artifacts = []

    # Artifact 1: touchstone file
    manifest1 = store.put(
        content=b"! Touchstone data\n# GHz S dB R 50\n1.0 -20 0\n",
        artifact_type="touchstone",
        roles=["oracle_output"],
        run_id="run-001",
        artifact_id="art-touchstone-001",
        stage_name="em_simulation",
        media_type="application/x-touchstone",
        tags={"coupon_id": "cpn-001", "sim_type": "full"},
    )
    artifacts.append(manifest1)

    # Artifact 2: coupon spec
    manifest2 = store.put(
        content=b'{"schema_version": 1, "coupon_family": "F1"}',
        artifact_type="coupon_spec",
        roles=["geometry", "config"],
        run_id="run-001",
        artifact_id="art-coupon-spec-001",
        stage_name="coupon_generation",
        media_type="application/json",
        tags={"coupon_id": "cpn-001"},
    )
    artifacts.append(manifest2)

    # Artifact 3: log file
    manifest3 = store.put(
        content=b"2025-01-20 10:00:00 INFO Starting simulation\n",
        artifact_type="log",
        roles=["metadata"],
        run_id="run-002",
        artifact_id="art-log-001",
        stage_name="em_simulation",
        media_type="text/plain",
    )
    artifacts.append(manifest3)

    # Artifact 4: validation report
    manifest4 = store.put(
        content=b'{"valid": true, "errors": []}',
        artifact_type="validation_report",
        roles=["validation"],
        run_id="run-002",
        artifact_id="art-validation-001",
        stage_name="drc_validation",
        media_type="application/json",
    )
    artifacts.append(manifest4)

    # Index all artifacts in the registry
    registry = ArtifactRegistry(data_dir / "registry.db")
    for manifest in artifacts:
        registry.index_artifact(manifest)
    registry.close()

    return initialized_project


class TestCmdArtifactShow:
    """Tests for the artifact show command implementation."""

    def test_show_nonexistent_artifact(self, initialized_project: Path) -> None:
        """artifact show should return error for nonexistent artifact."""
        result = cmd_artifact_show(
            artifact_id="nonexistent-id",
            root=initialized_project,
            output_json=False,
            show_content=False,
            verify=False,
            quiet=True,
        )
        assert result == 2

    def test_show_existing_artifact(
        self,
        project_with_artifacts: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """artifact show should display artifact information."""
        result = cmd_artifact_show(
            artifact_id="art-touchstone-001",
            root=project_with_artifacts,
            output_json=False,
            show_content=False,
            verify=False,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        assert "art-touchstone-001" in captured.out
        assert "touchstone" in captured.out
        assert "oracle_output" in captured.out

    def test_show_artifact_json_output(
        self,
        project_with_artifacts: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """artifact show --json should output valid JSON."""
        result = cmd_artifact_show(
            artifact_id="art-touchstone-001",
            root=project_with_artifacts,
            output_json=True,
            show_content=False,
            verify=False,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["artifact_id"] == "art-touchstone-001"
        assert data["artifact_type"] == "touchstone"
        assert data["schema_version"] == 1

    def test_show_artifact_with_content(
        self,
        project_with_artifacts: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """artifact show --content should display artifact content."""
        result = cmd_artifact_show(
            artifact_id="art-coupon-spec-001",
            root=project_with_artifacts,
            output_json=False,
            show_content=True,
            verify=False,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        assert "schema_version" in captured.out
        assert "coupon_family" in captured.out

    def test_show_artifact_with_verify(
        self,
        project_with_artifacts: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """artifact show --verify should verify integrity."""
        result = cmd_artifact_show(
            artifact_id="art-log-001",
            root=project_with_artifacts,
            output_json=False,
            show_content=False,
            verify=True,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        assert "VERIFIED" in captured.out

    def test_show_without_initialized_project(self, tmp_path: Path) -> None:
        """artifact show should fail if M3 not initialized."""
        result = cmd_artifact_show(
            artifact_id="test-id",
            root=tmp_path,
            output_json=False,
            show_content=False,
            verify=False,
            quiet=True,
        )
        assert result == 2


class TestCmdArtifactList:
    """Tests for the artifact list command implementation."""

    def test_list_empty_registry(
        self,
        initialized_project: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """artifact list should handle empty registry."""
        result = cmd_artifact_list(
            root=initialized_project,
            artifact_type=None,
            run_id=None,
            roles=[],
            created_after=None,
            created_before=None,
            limit=None,
            offset=0,
            order_by="created_utc",
            order_asc=False,
            output_json=False,
            long_format=False,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        assert "No artifacts found" in captured.out

    def test_list_all_artifacts(
        self,
        project_with_artifacts: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """artifact list should show all artifacts."""
        result = cmd_artifact_list(
            root=project_with_artifacts,
            artifact_type=None,
            run_id=None,
            roles=[],
            created_after=None,
            created_before=None,
            limit=None,
            offset=0,
            order_by="created_utc",
            order_asc=False,
            output_json=False,
            long_format=False,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        assert "art-touchstone-001" in captured.out
        assert "art-coupon-spec-001" in captured.out
        assert "art-log-001" in captured.out
        assert "art-validation-001" in captured.out
        assert "Total: 4 artifacts" in captured.out

    def test_list_filter_by_type(
        self,
        project_with_artifacts: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """artifact list should filter by type."""
        result = cmd_artifact_list(
            root=project_with_artifacts,
            artifact_type="touchstone",
            run_id=None,
            roles=[],
            created_after=None,
            created_before=None,
            limit=None,
            offset=0,
            order_by="created_utc",
            order_asc=False,
            output_json=False,
            long_format=False,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        assert "art-touchstone-001" in captured.out
        assert "art-coupon-spec-001" not in captured.out
        assert "Total: 1 artifacts" in captured.out

    def test_list_filter_by_run_id(
        self,
        project_with_artifacts: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """artifact list should filter by run ID."""
        result = cmd_artifact_list(
            root=project_with_artifacts,
            artifact_type=None,
            run_id="run-001",
            roles=[],
            created_after=None,
            created_before=None,
            limit=None,
            offset=0,
            order_by="created_utc",
            order_asc=False,
            output_json=False,
            long_format=False,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        assert "art-touchstone-001" in captured.out
        assert "art-coupon-spec-001" in captured.out
        assert "art-log-001" not in captured.out
        assert "Total: 2 artifacts" in captured.out

    def test_list_filter_by_role(
        self,
        project_with_artifacts: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """artifact list should filter by role."""
        result = cmd_artifact_list(
            root=project_with_artifacts,
            artifact_type=None,
            run_id=None,
            roles=["validation"],
            created_after=None,
            created_before=None,
            limit=None,
            offset=0,
            order_by="created_utc",
            order_asc=False,
            output_json=False,
            long_format=False,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        assert "art-validation-001" in captured.out
        assert "Total: 1 artifacts" in captured.out

    def test_list_with_limit(
        self,
        project_with_artifacts: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """artifact list should respect limit."""
        result = cmd_artifact_list(
            root=project_with_artifacts,
            artifact_type=None,
            run_id=None,
            roles=[],
            created_after=None,
            created_before=None,
            limit=2,
            offset=0,
            order_by="created_utc",
            order_asc=False,
            output_json=False,
            long_format=False,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        assert "showing 2" in captured.out

    def test_list_json_output(
        self,
        project_with_artifacts: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """artifact list --json should output valid JSON."""
        result = cmd_artifact_list(
            root=project_with_artifacts,
            artifact_type=None,
            run_id=None,
            roles=[],
            created_after=None,
            created_before=None,
            limit=None,
            offset=0,
            order_by="created_utc",
            order_asc=False,
            output_json=True,
            long_format=False,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["count"] == 4
        assert data["total"] == 4
        assert len(data["artifacts"]) == 4
        assert all("artifact_id" in a for a in data["artifacts"])

    def test_list_quiet_mode(
        self,
        project_with_artifacts: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """artifact list --quiet should only output IDs."""
        result = cmd_artifact_list(
            root=project_with_artifacts,
            artifact_type=None,
            run_id=None,
            roles=[],
            created_after=None,
            created_before=None,
            limit=None,
            offset=0,
            order_by="artifact_id",
            order_asc=True,
            output_json=False,
            long_format=False,
            quiet=True,
        )
        assert result == 0

        captured = capsys.readouterr()
        lines = [l for l in captured.out.strip().split("\n") if l]
        assert len(lines) == 4
        # Each line should be just an artifact ID
        assert all(l.startswith("art-") for l in lines)

    def test_list_long_format(
        self,
        project_with_artifacts: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """artifact list --long should show detailed info."""
        result = cmd_artifact_list(
            root=project_with_artifacts,
            artifact_type="touchstone",
            run_id=None,
            roles=[],
            created_after=None,
            created_before=None,
            limit=None,
            offset=0,
            order_by="created_utc",
            order_asc=False,
            output_json=False,
            long_format=True,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        assert "Artifact: art-touchstone-001" in captured.out
        assert "Type: touchstone" in captured.out
        assert "Roles:" in captured.out

    def test_list_without_initialized_project(self, tmp_path: Path) -> None:
        """artifact list should fail if M3 not initialized."""
        result = cmd_artifact_list(
            root=tmp_path,
            artifact_type=None,
            run_id=None,
            roles=[],
            created_after=None,
            created_before=None,
            limit=None,
            offset=0,
            order_by="created_utc",
            order_asc=False,
            output_json=False,
            long_format=False,
            quiet=True,
        )
        assert result == 2


class TestMainArtifact:
    """Tests for the main entry point with artifact commands."""

    def test_main_artifact_show(self, project_with_artifacts: Path) -> None:
        """main should handle artifact show command."""
        result = main([
            "artifact", "show", "art-touchstone-001",
            "--root", str(project_with_artifacts),
            "-q",
        ])
        assert result == 0

    def test_main_artifact_list(self, project_with_artifacts: Path) -> None:
        """main should handle artifact list command."""
        result = main([
            "artifact", "list",
            "--root", str(project_with_artifacts),
            "-q",
        ])
        assert result == 0

    def test_main_artifact_list_with_filters(
        self,
        project_with_artifacts: Path,
    ) -> None:
        """main should handle artifact list with filters."""
        result = main([
            "artifact", "list",
            "--root", str(project_with_artifacts),
            "-t", "touchstone",
            "-n", "10",
            "-q",
        ])
        assert result == 0

    def test_main_artifact_show_json(
        self,
        project_with_artifacts: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """main artifact show --json should output valid JSON."""
        result = main([
            "artifact", "show", "art-coupon-spec-001",
            "--root", str(project_with_artifacts),
            "--json",
        ])
        assert result == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["artifact_id"] == "art-coupon-spec-001"
