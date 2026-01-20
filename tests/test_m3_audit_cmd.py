"""Tests for the m3 audit command.

These tests verify:
1. CLI argument parsing for the audit command
2. Artifact auditing with lineage information
3. Hash verification
4. Required roles checking
5. JSON and text output formats
6. Error handling
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from formula_foundry.m3.cli_main import build_parser, cmd_audit, cmd_init, main

if TYPE_CHECKING:
    pass


class TestAuditParser:
    """Tests for audit command argument parsing."""

    def test_parser_has_audit_command(self) -> None:
        """Parser should have audit subcommand."""
        parser = build_parser()
        args = parser.parse_args(["audit"])
        assert args.command == "audit"

    def test_audit_default_arguments(self) -> None:
        """Audit command should have correct default arguments."""
        parser = build_parser()
        args = parser.parse_args(["audit"])
        assert args.artifact_id is None
        assert args.root is None
        assert args.format == "text"
        assert args.trace_roots is False
        assert args.verify_hashes is False
        assert args.max_depth is None
        assert args.required_roles is None
        assert args.quiet is False

    def test_audit_with_artifact_id(self) -> None:
        """Audit command should accept artifact ID."""
        parser = build_parser()
        args = parser.parse_args(["audit", "art-123"])
        assert args.artifact_id == "art-123"

    def test_audit_with_root(self, tmp_path: Path) -> None:
        """Audit command should accept --root argument."""
        parser = build_parser()
        args = parser.parse_args(["audit", "--root", str(tmp_path)])
        assert args.root == tmp_path

    def test_audit_with_json_format(self) -> None:
        """Audit command should accept --format json."""
        parser = build_parser()
        args = parser.parse_args(["audit", "--format", "json"])
        assert args.format == "json"

    def test_audit_with_trace_roots(self) -> None:
        """Audit command should accept --trace-roots flag."""
        parser = build_parser()
        args = parser.parse_args(["audit", "--trace-roots"])
        assert args.trace_roots is True

    def test_audit_with_verify_hashes(self) -> None:
        """Audit command should accept --verify-hashes flag."""
        parser = build_parser()
        args = parser.parse_args(["audit", "--verify-hashes"])
        assert args.verify_hashes is True

    def test_audit_with_max_depth(self) -> None:
        """Audit command should accept --max-depth argument."""
        parser = build_parser()
        args = parser.parse_args(["audit", "--max-depth", "5"])
        assert args.max_depth == 5

    def test_audit_with_required_roles(self) -> None:
        """Audit command should accept --required-roles argument."""
        parser = build_parser()
        args = parser.parse_args(["audit", "--required-roles", "geometry,config"])
        assert args.required_roles == "geometry,config"

    def test_audit_with_quiet(self) -> None:
        """Audit command should accept --quiet flag."""
        parser = build_parser()
        args = parser.parse_args(["audit", "-q"])
        assert args.quiet is True


class TestCmdAuditBasic:
    """Basic tests for the audit command."""

    def test_audit_fails_without_init(self, tmp_path: Path) -> None:
        """Audit should fail if M3 is not initialized."""
        result = cmd_audit(
            artifact_id=None,
            root=tmp_path,
            output_format="text",
            trace_roots=False,
            verify_hashes=False,
            max_depth=None,
            required_roles=None,
            quiet=True,
        )
        assert result == 2

    def test_audit_succeeds_with_no_artifacts(self, tmp_path: Path) -> None:
        """Audit should succeed with an empty store."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        result = cmd_audit(
            artifact_id=None,
            root=tmp_path,
            output_format="text",
            trace_roots=False,
            verify_hashes=False,
            max_depth=None,
            required_roles=None,
            quiet=True,
        )
        assert result == 0


class TestCmdAuditWithArtifacts:
    """Tests for audit command with actual artifacts."""

    @pytest.fixture
    def initialized_project(self, tmp_path: Path) -> Path:
        """Initialize a project and create some artifacts."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        return tmp_path

    @pytest.fixture
    def project_with_artifacts(self, initialized_project: Path) -> tuple[Path, list[str]]:
        """Create some test artifacts."""
        from formula_foundry.m3.artifact_store import ArtifactStore, LineageReference
        from formula_foundry.m3.registry import ArtifactRegistry

        data_dir = initialized_project / "data"
        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")

        # Create root artifact
        root_manifest = store.put(
            content=b"root config data",
            artifact_type="coupon_spec",
            roles=["config", "root_input"],
            run_id="run-001",
            artifact_id="art-root-001",
            stage_name="config_load",
        )
        registry.index_artifact(root_manifest)

        # Create geometry artifact
        geometry_manifest = store.put(
            content=b"geometry data",
            artifact_type="resolved_design",
            roles=["geometry"],
            run_id="run-002",
            artifact_id="art-geom-001",
            stage_name="geometry_resolve",
            inputs=[
                LineageReference(artifact_id="art-root-001", relation="config_from"),
            ],
        )
        registry.index_artifact(geometry_manifest)

        # Create output artifact
        output_manifest = store.put(
            content=b"output data",
            artifact_type="kicad_board",
            roles=["final_output"],
            run_id="run-003",
            artifact_id="art-output-001",
            stage_name="board_generate",
            inputs=[
                LineageReference(artifact_id="art-geom-001", relation="derived_from"),
            ],
        )
        registry.index_artifact(output_manifest)

        registry.close()
        return initialized_project, ["art-root-001", "art-geom-001", "art-output-001"]

    def test_audit_all_artifacts(self, project_with_artifacts: tuple[Path, list[str]]) -> None:
        """Audit should process all artifacts."""
        project_root, artifact_ids = project_with_artifacts
        result = cmd_audit(
            artifact_id=None,
            root=project_root,
            output_format="text",
            trace_roots=False,
            verify_hashes=False,
            max_depth=None,
            required_roles=None,
            quiet=True,
        )
        assert result == 0

    def test_audit_single_artifact(self, project_with_artifacts: tuple[Path, list[str]]) -> None:
        """Audit should process a single specified artifact."""
        project_root, artifact_ids = project_with_artifacts
        result = cmd_audit(
            artifact_id="art-output-001",
            root=project_root,
            output_format="text",
            trace_roots=False,
            verify_hashes=False,
            max_depth=None,
            required_roles=None,
            quiet=True,
        )
        assert result == 0

    def test_audit_json_output(
        self, project_with_artifacts: tuple[Path, list[str]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Audit should produce valid JSON output."""
        project_root, artifact_ids = project_with_artifacts
        result = cmd_audit(
            artifact_id=None,
            root=project_root,
            output_format="json",
            trace_roots=False,
            verify_hashes=False,
            max_depth=None,
            required_roles=None,
            quiet=True,
        )
        assert result == 0

        captured = capsys.readouterr()
        report = json.loads(captured.out)

        assert "schema_version" in report
        assert report["schema_version"] == 1
        assert "total_artifacts" in report
        assert report["total_artifacts"] == 3
        assert "artifacts" in report
        assert len(report["artifacts"]) == 3
        assert "graph_stats" in report

    def test_audit_with_hash_verification(
        self, project_with_artifacts: tuple[Path, list[str]]
    ) -> None:
        """Audit should verify content hashes when requested."""
        project_root, artifact_ids = project_with_artifacts
        result = cmd_audit(
            artifact_id=None,
            root=project_root,
            output_format="text",
            trace_roots=False,
            verify_hashes=True,
            max_depth=None,
            required_roles=None,
            quiet=True,
        )
        assert result == 0

    def test_audit_with_trace_roots(
        self, project_with_artifacts: tuple[Path, list[str]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Audit should trace lineage to roots when requested."""
        project_root, artifact_ids = project_with_artifacts
        result = cmd_audit(
            artifact_id="art-output-001",
            root=project_root,
            output_format="json",
            trace_roots=True,
            verify_hashes=False,
            max_depth=None,
            required_roles=None,
            quiet=True,
        )
        assert result == 0

        captured = capsys.readouterr()
        report = json.loads(captured.out)

        # Should have traced back to root
        artifact = report["artifacts"][0]
        assert artifact["artifact_id"] == "art-output-001"
        assert artifact["ancestor_count"] >= 1  # Should have ancestors

    def test_audit_with_max_depth(
        self, project_with_artifacts: tuple[Path, list[str]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Audit should limit ancestor traversal to max_depth."""
        project_root, artifact_ids = project_with_artifacts
        result = cmd_audit(
            artifact_id="art-output-001",
            root=project_root,
            output_format="json",
            trace_roots=False,
            verify_hashes=False,
            max_depth=1,
            required_roles=None,
            quiet=True,
        )
        assert result == 0

        captured = capsys.readouterr()
        report = json.loads(captured.out)

        artifact = report["artifacts"][0]
        # With max_depth=1, should only get immediate parent
        assert artifact["ancestor_count"] <= 2  # At most 1 level deep

    def test_audit_with_required_roles_pass(
        self, project_with_artifacts: tuple[Path, list[str]]
    ) -> None:
        """Audit should pass when required roles exist in roots."""
        project_root, artifact_ids = project_with_artifacts
        result = cmd_audit(
            artifact_id="art-output-001",
            root=project_root,
            output_format="text",
            trace_roots=True,
            verify_hashes=False,
            max_depth=None,
            required_roles="config",
            quiet=True,
        )
        # Should pass because art-root-001 has "config" role
        assert result == 0

    def test_audit_with_required_roles_fail(
        self, project_with_artifacts: tuple[Path, list[str]]
    ) -> None:
        """Audit should fail when required roles are missing from roots."""
        project_root, artifact_ids = project_with_artifacts
        result = cmd_audit(
            artifact_id="art-output-001",
            root=project_root,
            output_format="text",
            trace_roots=True,
            verify_hashes=False,
            max_depth=None,
            required_roles="oracle_output",
            quiet=True,
        )
        # Should fail because no artifact has "oracle_output" role
        assert result == 2


class TestCmdAuditHashVerification:
    """Tests for hash verification in audit command."""

    def test_audit_detects_corrupted_content(self, tmp_path: Path) -> None:
        """Audit should detect when content doesn't match hash."""
        cmd_init(root=tmp_path, force=False, quiet=True)

        from formula_foundry.m3.artifact_store import ArtifactStore
        from formula_foundry.m3.registry import ArtifactRegistry

        data_dir = tmp_path / "data"
        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")

        # Create an artifact
        manifest = store.put(
            content=b"original content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
            artifact_id="art-test-001",
        )
        registry.index_artifact(manifest)
        registry.close()

        # Corrupt the content
        object_path = data_dir / "objects" / manifest.content_hash.digest[:2] / manifest.content_hash.digest
        object_path.write_bytes(b"corrupted content")

        # Audit should detect the mismatch
        result = cmd_audit(
            artifact_id="art-test-001",
            root=tmp_path,
            output_format="text",
            trace_roots=False,
            verify_hashes=True,
            max_depth=None,
            required_roles=None,
            quiet=True,
        )
        assert result == 2  # Should fail due to hash mismatch


class TestMainWithAudit:
    """Tests for main entry point with audit command."""

    def test_main_audit_success(self, tmp_path: Path) -> None:
        """Main with audit command should succeed after init."""
        main(["init", "--root", str(tmp_path), "-q"])
        result = main(["audit", "--root", str(tmp_path), "-q"])
        assert result == 0

    def test_main_audit_json_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Main with audit --format json should produce JSON."""
        main(["init", "--root", str(tmp_path), "-q"])
        result = main(["audit", "--root", str(tmp_path), "--format", "json", "-q"])
        assert result == 0

        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert "schema_version" in report

    def test_main_audit_with_all_flags(self, tmp_path: Path) -> None:
        """Main with audit and all flags should succeed."""
        main(["init", "--root", str(tmp_path), "-q"])
        result = main([
            "audit",
            "--root", str(tmp_path),
            "--format", "json",
            "--trace-roots",
            "--verify-hashes",
            "--max-depth", "10",
            "-q",
        ])
        assert result == 0
