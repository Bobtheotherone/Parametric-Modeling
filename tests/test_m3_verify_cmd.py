"""Tests for the m3 verify command.

These tests verify:
1. CLI argument parsing for the verify command
2. Hash verification (content integrity)
3. Lineage consistency checks
4. Manifest schema validation
5. Registry consistency checks
6. Repair functionality
7. JSON and text output formats
8. Error handling
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from formula_foundry.m3.cli_main import build_parser, cmd_init, cmd_verify, main

if TYPE_CHECKING:
    pass


class TestVerifyParser:
    """Tests for verify command argument parsing."""

    def test_parser_has_verify_command(self) -> None:
        """Parser should have verify subcommand."""
        parser = build_parser()
        args = parser.parse_args(["verify"])
        assert args.command == "verify"

    def test_verify_default_arguments(self) -> None:
        """Verify command should have correct default arguments."""
        parser = build_parser()
        args = parser.parse_args(["verify"])
        assert args.artifact_id is None
        assert args.root is None
        assert args.format == "text"
        assert args.check_hash is False
        assert args.skip_hash is False
        assert args.check_lineage is False
        assert args.check_manifest is False
        assert args.check_registry is False
        assert args.full is False
        assert args.repair is False
        assert args.quiet is False

    def test_verify_with_artifact_id(self) -> None:
        """Verify command should accept artifact ID."""
        parser = build_parser()
        args = parser.parse_args(["verify", "art-123"])
        assert args.artifact_id == "art-123"

    def test_verify_with_root(self, tmp_path: Path) -> None:
        """Verify command should accept --root argument."""
        parser = build_parser()
        args = parser.parse_args(["verify", "--root", str(tmp_path)])
        assert args.root == tmp_path

    def test_verify_with_json_format(self) -> None:
        """Verify command should accept --format json."""
        parser = build_parser()
        args = parser.parse_args(["verify", "--format", "json"])
        assert args.format == "json"

    def test_verify_with_hash_flag(self) -> None:
        """Verify command should accept --hash flag."""
        parser = build_parser()
        args = parser.parse_args(["verify", "--hash"])
        assert args.check_hash is True

    def test_verify_with_no_hash_flag(self) -> None:
        """Verify command should accept --no-hash flag."""
        parser = build_parser()
        args = parser.parse_args(["verify", "--no-hash"])
        assert args.skip_hash is True

    def test_verify_with_lineage_flag(self) -> None:
        """Verify command should accept --lineage flag."""
        parser = build_parser()
        args = parser.parse_args(["verify", "--lineage"])
        assert args.check_lineage is True

    def test_verify_with_manifest_flag(self) -> None:
        """Verify command should accept --manifest flag."""
        parser = build_parser()
        args = parser.parse_args(["verify", "--manifest"])
        assert args.check_manifest is True

    def test_verify_with_registry_flag(self) -> None:
        """Verify command should accept --registry flag."""
        parser = build_parser()
        args = parser.parse_args(["verify", "--registry"])
        assert args.check_registry is True

    def test_verify_with_full_flag(self) -> None:
        """Verify command should accept --full flag."""
        parser = build_parser()
        args = parser.parse_args(["verify", "--full"])
        assert args.full is True

    def test_verify_with_repair_flag(self) -> None:
        """Verify command should accept --repair flag."""
        parser = build_parser()
        args = parser.parse_args(["verify", "--repair"])
        assert args.repair is True

    def test_verify_with_quiet_flag(self) -> None:
        """Verify command should accept --quiet flag."""
        parser = build_parser()
        args = parser.parse_args(["verify", "-q"])
        assert args.quiet is True

    def test_verify_with_multiple_flags(self) -> None:
        """Verify command should accept multiple flags."""
        parser = build_parser()
        args = parser.parse_args(["verify", "--hash", "--lineage", "--manifest", "--registry"])
        assert args.check_hash is True
        assert args.check_lineage is True
        assert args.check_manifest is True
        assert args.check_registry is True


class TestCmdVerifyBasic:
    """Basic tests for the verify command."""

    def test_verify_fails_without_init(self, tmp_path: Path) -> None:
        """Verify should fail if M3 is not initialized."""
        result = cmd_verify(
            artifact_id=None,
            root=tmp_path,
            output_format="text",
            check_hash=False,
            skip_hash=False,
            check_lineage=False,
            check_manifest=False,
            check_registry=False,
            full=False,
            repair=False,
            quiet=True,
        )
        assert result == 2

    def test_verify_succeeds_with_no_artifacts(self, tmp_path: Path) -> None:
        """Verify should succeed with an empty store."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        result = cmd_verify(
            artifact_id=None,
            root=tmp_path,
            output_format="text",
            check_hash=False,
            skip_hash=False,
            check_lineage=False,
            check_manifest=False,
            check_registry=False,
            full=False,
            repair=False,
            quiet=True,
        )
        assert result == 0


class TestCmdVerifyHashCheck:
    """Tests for hash verification functionality."""

    @pytest.fixture
    def initialized_project(self, tmp_path: Path) -> Path:
        """Initialize a project."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        return tmp_path

    @pytest.fixture
    def project_with_artifacts(self, initialized_project: Path) -> tuple[Path, list[str]]:
        """Create some test artifacts."""
        from formula_foundry.m3.artifact_store import ArtifactStore
        from formula_foundry.m3.registry import ArtifactRegistry

        data_dir = initialized_project / "data"
        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")

        # Create test artifact
        manifest = store.put(
            content=b"test content data",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
            artifact_id="art-test-001",
            stage_name="test_stage",
        )
        registry.index_artifact(manifest)

        registry.close()
        return initialized_project, ["art-test-001"]

    def test_verify_hash_pass(self, project_with_artifacts: tuple[Path, list[str]]) -> None:
        """Verify should pass when content hash is correct."""
        project_root, artifact_ids = project_with_artifacts
        result = cmd_verify(
            artifact_id="art-test-001",
            root=project_root,
            output_format="text",
            check_hash=True,
            skip_hash=False,
            check_lineage=False,
            check_manifest=False,
            check_registry=False,
            full=False,
            repair=False,
            quiet=True,
        )
        assert result == 0

    def test_verify_detects_corrupted_content(self, initialized_project: Path) -> None:
        """Verify should detect when content doesn't match hash."""
        from formula_foundry.m3.artifact_store import ArtifactStore
        from formula_foundry.m3.registry import ArtifactRegistry

        data_dir = initialized_project / "data"
        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")

        # Create an artifact
        manifest = store.put(
            content=b"original content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
            artifact_id="art-corrupt-001",
        )
        registry.index_artifact(manifest)
        registry.close()

        # Corrupt the content
        object_path = data_dir / "objects" / manifest.content_hash.digest[:2] / manifest.content_hash.digest
        object_path.write_bytes(b"corrupted content")

        # Verify should detect the mismatch
        result = cmd_verify(
            artifact_id="art-corrupt-001",
            root=initialized_project,
            output_format="text",
            check_hash=True,
            skip_hash=False,
            check_lineage=False,
            check_manifest=False,
            check_registry=False,
            full=False,
            repair=False,
            quiet=True,
        )
        assert result == 2  # Should fail due to hash mismatch

    def test_verify_default_runs_hash_check(self, project_with_artifacts: tuple[Path, list[str]]) -> None:
        """By default, verify should run hash check."""
        project_root, artifact_ids = project_with_artifacts
        result = cmd_verify(
            artifact_id=None,
            root=project_root,
            output_format="text",
            check_hash=False,  # Not explicitly set
            skip_hash=False,
            check_lineage=False,
            check_manifest=False,
            check_registry=False,
            full=False,
            repair=False,
            quiet=True,
        )
        assert result == 0

    def test_verify_skip_hash(self, initialized_project: Path) -> None:
        """Verify should skip hash check when --no-hash is specified."""
        from formula_foundry.m3.artifact_store import ArtifactStore
        from formula_foundry.m3.registry import ArtifactRegistry

        data_dir = initialized_project / "data"
        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")

        # Create an artifact then corrupt it
        manifest = store.put(
            content=b"original content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
            artifact_id="art-skip-hash-001",
        )
        registry.index_artifact(manifest)
        registry.close()

        # Corrupt the content
        object_path = data_dir / "objects" / manifest.content_hash.digest[:2] / manifest.content_hash.digest
        object_path.write_bytes(b"corrupted content")

        # With skip_hash, should pass despite corruption
        result = cmd_verify(
            artifact_id="art-skip-hash-001",
            root=initialized_project,
            output_format="text",
            check_hash=False,
            skip_hash=True,  # Skip hash check
            check_lineage=False,
            check_manifest=False,
            check_registry=False,
            full=False,
            repair=False,
            quiet=True,
        )
        assert result == 0  # Should pass since hash check is skipped


class TestCmdVerifyLineageCheck:
    """Tests for lineage consistency checking."""

    @pytest.fixture
    def initialized_project(self, tmp_path: Path) -> Path:
        """Initialize a project."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        return tmp_path

    def test_verify_lineage_pass(self, initialized_project: Path) -> None:
        """Verify should pass when all referenced inputs exist."""
        from formula_foundry.m3.artifact_store import ArtifactStore, LineageReference
        from formula_foundry.m3.registry import ArtifactRegistry

        data_dir = initialized_project / "data"
        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")

        # Create root artifact
        root_manifest = store.put(
            content=b"root data",
            artifact_type="coupon_spec",
            roles=["config"],
            run_id="run-001",
            artifact_id="art-root-001",
        )
        registry.index_artifact(root_manifest)

        # Create derived artifact
        derived_manifest = store.put(
            content=b"derived data",
            artifact_type="resolved_design",
            roles=["geometry"],
            run_id="run-002",
            artifact_id="art-derived-001",
            inputs=[LineageReference(artifact_id="art-root-001", relation="derived_from")],
        )
        registry.index_artifact(derived_manifest)
        registry.close()

        result = cmd_verify(
            artifact_id="art-derived-001",
            root=initialized_project,
            output_format="text",
            check_hash=False,
            skip_hash=True,
            check_lineage=True,
            check_manifest=False,
            check_registry=False,
            full=False,
            repair=False,
            quiet=True,
        )
        assert result == 0

    def test_verify_lineage_fail_missing_input(self, initialized_project: Path) -> None:
        """Verify should fail when referenced input is missing."""
        from formula_foundry.m3.artifact_store import ArtifactStore, LineageReference
        from formula_foundry.m3.registry import ArtifactRegistry

        data_dir = initialized_project / "data"
        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")

        # Create artifact referencing non-existent input
        manifest = store.put(
            content=b"orphan data",
            artifact_type="resolved_design",
            roles=["geometry"],
            run_id="run-001",
            artifact_id="art-orphan-001",
            inputs=[LineageReference(artifact_id="art-nonexistent", relation="derived_from")],
        )
        registry.index_artifact(manifest)
        registry.close()

        result = cmd_verify(
            artifact_id="art-orphan-001",
            root=initialized_project,
            output_format="text",
            check_hash=False,
            skip_hash=True,
            check_lineage=True,
            check_manifest=False,
            check_registry=False,
            full=False,
            repair=False,
            quiet=True,
        )
        assert result == 2  # Should fail due to missing input


class TestCmdVerifyManifestCheck:
    """Tests for manifest schema validation."""

    @pytest.fixture
    def initialized_project(self, tmp_path: Path) -> Path:
        """Initialize a project."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        return tmp_path

    def test_verify_manifest_pass(self, initialized_project: Path) -> None:
        """Verify should pass when manifest is valid."""
        from formula_foundry.m3.artifact_store import ArtifactStore
        from formula_foundry.m3.registry import ArtifactRegistry

        data_dir = initialized_project / "data"
        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")

        manifest = store.put(
            content=b"valid data",
            artifact_type="coupon_spec",
            roles=["config"],
            run_id="run-001",
            artifact_id="art-valid-001",
        )
        registry.index_artifact(manifest)
        registry.close()

        result = cmd_verify(
            artifact_id="art-valid-001",
            root=initialized_project,
            output_format="text",
            check_hash=False,
            skip_hash=True,
            check_lineage=False,
            check_manifest=True,
            check_registry=False,
            full=False,
            repair=False,
            quiet=True,
        )
        assert result == 0


class TestCmdVerifyRegistryCheck:
    """Tests for registry consistency checking."""

    @pytest.fixture
    def initialized_project(self, tmp_path: Path) -> Path:
        """Initialize a project."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        return tmp_path

    def test_verify_registry_pass(self, initialized_project: Path) -> None:
        """Verify should pass when registry matches manifest."""
        from formula_foundry.m3.artifact_store import ArtifactStore
        from formula_foundry.m3.registry import ArtifactRegistry

        data_dir = initialized_project / "data"
        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")

        manifest = store.put(
            content=b"registry test data",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
            artifact_id="art-registry-001",
        )
        registry.index_artifact(manifest)
        registry.close()

        result = cmd_verify(
            artifact_id="art-registry-001",
            root=initialized_project,
            output_format="text",
            check_hash=False,
            skip_hash=True,
            check_lineage=False,
            check_manifest=False,
            check_registry=True,
            full=False,
            repair=False,
            quiet=True,
        )
        assert result == 0

    def test_verify_registry_fail_missing(self, initialized_project: Path) -> None:
        """Verify should fail when artifact is not in registry."""
        from formula_foundry.m3.artifact_store import ArtifactStore

        data_dir = initialized_project / "data"
        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")

        # Create artifact but don't index in registry
        store.put(
            content=b"unindexed data",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
            artifact_id="art-unindexed-001",
        )

        result = cmd_verify(
            artifact_id="art-unindexed-001",
            root=initialized_project,
            output_format="text",
            check_hash=False,
            skip_hash=True,
            check_lineage=False,
            check_manifest=False,
            check_registry=True,
            full=False,
            repair=False,
            quiet=True,
        )
        assert result == 2  # Should fail due to missing registry entry

    def test_verify_registry_repair(self, initialized_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify should repair missing registry entries when --repair is specified."""
        from formula_foundry.m3.artifact_store import ArtifactStore
        from formula_foundry.m3.registry import ArtifactRegistry

        data_dir = initialized_project / "data"
        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")

        # Create artifact but don't index in registry
        store.put(
            content=b"unindexed data for repair",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
            artifact_id="art-repair-001",
        )

        result = cmd_verify(
            artifact_id="art-repair-001",
            root=initialized_project,
            output_format="json",
            check_hash=False,
            skip_hash=True,
            check_lineage=False,
            check_manifest=False,
            check_registry=True,
            full=False,
            repair=True,  # Enable repair
            quiet=False,
        )

        captured = capsys.readouterr()
        report = json.loads(captured.out)

        # Check that the artifact was repaired
        assert report["repaired_count"] == 1
        assert report["artifacts"][0]["checks"]["registry"].get("repaired") is True

        # Should return 1 (warning) because issue was repaired
        assert result == 1  # Warning because issue was repaired

        # Verify that the artifact is now in registry
        registry = ArtifactRegistry(data_dir / "registry.db")
        record = registry.get_artifact("art-repair-001")
        registry.close()
        assert record is not None
        assert record.artifact_id == "art-repair-001"


class TestCmdVerifyFullCheck:
    """Tests for full verification mode."""

    @pytest.fixture
    def initialized_project(self, tmp_path: Path) -> Path:
        """Initialize a project."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        return tmp_path

    def test_verify_full_pass(self, initialized_project: Path) -> None:
        """Full verification should pass when all checks pass."""
        from formula_foundry.m3.artifact_store import ArtifactStore
        from formula_foundry.m3.registry import ArtifactRegistry

        data_dir = initialized_project / "data"
        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")

        manifest = store.put(
            content=b"full check data",
            artifact_type="coupon_spec",
            roles=["config", "root_input"],
            run_id="run-001",
            artifact_id="art-full-001",
        )
        registry.index_artifact(manifest)
        registry.close()

        result = cmd_verify(
            artifact_id="art-full-001",
            root=initialized_project,
            output_format="text",
            check_hash=False,
            skip_hash=False,
            check_lineage=False,
            check_manifest=False,
            check_registry=False,
            full=True,  # Run all checks
            repair=False,
            quiet=True,
        )
        assert result == 0


class TestCmdVerifyJsonOutput:
    """Tests for JSON output format."""

    @pytest.fixture
    def initialized_project(self, tmp_path: Path) -> Path:
        """Initialize a project."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        return tmp_path

    def test_verify_json_output_empty(self, initialized_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """JSON output should be valid with no artifacts."""
        result = cmd_verify(
            artifact_id=None,
            root=initialized_project,
            output_format="json",
            check_hash=False,
            skip_hash=True,
            check_lineage=False,
            check_manifest=False,
            check_registry=False,
            full=False,
            repair=False,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        report = json.loads(captured.out)

        assert "schema_version" in report
        assert report["schema_version"] == 1
        assert report["total_artifacts"] == 0
        assert report["passed"] == 0
        assert report["errors"] == 0
        assert "artifacts" in report
        assert "checks_performed" in report

    def test_verify_json_output_with_artifacts(self, initialized_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """JSON output should contain artifact details."""
        from formula_foundry.m3.artifact_store import ArtifactStore
        from formula_foundry.m3.registry import ArtifactRegistry

        data_dir = initialized_project / "data"
        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")

        manifest = store.put(
            content=b"json output test",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
            artifact_id="art-json-001",
        )
        registry.index_artifact(manifest)
        registry.close()

        result = cmd_verify(
            artifact_id=None,
            root=initialized_project,
            output_format="json",
            check_hash=True,
            skip_hash=False,
            check_lineage=False,
            check_manifest=False,
            check_registry=False,
            full=False,
            repair=False,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        report = json.loads(captured.out)

        assert report["total_artifacts"] == 1
        assert report["passed"] == 1
        assert len(report["artifacts"]) == 1

        artifact = report["artifacts"][0]
        assert artifact["artifact_id"] == "art-json-001"
        assert artifact["status"] == "pass"
        assert "checks" in artifact
        assert artifact["checks"]["hash"]["passed"] is True


class TestMainWithVerify:
    """Tests for main entry point with verify command."""

    def test_main_verify_success(self, tmp_path: Path) -> None:
        """Main with verify command should succeed after init."""
        main(["init", "--root", str(tmp_path), "-q"])
        result = main(["verify", "--root", str(tmp_path), "-q"])
        assert result == 0

    def test_main_verify_with_full(self, tmp_path: Path) -> None:
        """Main with verify --full should run all checks."""
        main(["init", "--root", str(tmp_path), "-q"])
        result = main(["verify", "--root", str(tmp_path), "--full", "-q"])
        assert result == 0

    def test_main_verify_fails_without_init(self, tmp_path: Path) -> None:
        """Main with verify should fail without init."""
        result = main(["verify", "--root", str(tmp_path), "-q"])
        assert result == 2


class TestCmdVerifyMultipleArtifacts:
    """Tests for verifying multiple artifacts."""

    @pytest.fixture
    def project_with_multiple_artifacts(self, tmp_path: Path) -> tuple[Path, list[str]]:
        """Create a project with multiple artifacts."""
        cmd_init(root=tmp_path, force=False, quiet=True)

        from formula_foundry.m3.artifact_store import ArtifactStore, LineageReference
        from formula_foundry.m3.registry import ArtifactRegistry

        data_dir = tmp_path / "data"
        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")

        artifact_ids = []

        # Create chain of artifacts
        for i in range(5):
            art_id = f"art-multi-{i:03d}"
            inputs = []
            if i > 0:
                inputs.append(LineageReference(artifact_id=f"art-multi-{i - 1:03d}", relation="derived_from"))

            manifest = store.put(
                content=f"content for artifact {i}".encode(),
                artifact_type="other",
                roles=["intermediate"],
                run_id=f"run-{i:03d}",
                artifact_id=art_id,
                inputs=inputs,
            )
            registry.index_artifact(manifest)
            artifact_ids.append(art_id)

        registry.close()
        return tmp_path, artifact_ids

    def test_verify_all_artifacts(self, project_with_multiple_artifacts: tuple[Path, list[str]]) -> None:
        """Verify should process all artifacts when no specific ID given."""
        project_root, artifact_ids = project_with_multiple_artifacts
        result = cmd_verify(
            artifact_id=None,
            root=project_root,
            output_format="text",
            check_hash=True,
            skip_hash=False,
            check_lineage=True,
            check_manifest=True,
            check_registry=True,
            full=False,
            repair=False,
            quiet=True,
        )
        assert result == 0

    def test_verify_all_json_output(
        self,
        project_with_multiple_artifacts: tuple[Path, list[str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """JSON output should include all artifacts."""
        project_root, artifact_ids = project_with_multiple_artifacts
        result = cmd_verify(
            artifact_id=None,
            root=project_root,
            output_format="json",
            check_hash=True,
            skip_hash=False,
            check_lineage=True,
            check_manifest=True,
            check_registry=True,
            full=False,
            repair=False,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        report = json.loads(captured.out)

        assert report["total_artifacts"] == 5
        assert report["passed"] == 5
        assert len(report["artifacts"]) == 5
