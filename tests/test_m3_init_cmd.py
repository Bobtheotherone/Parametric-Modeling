"""Tests for the m3 init command.

These tests verify:
1. Directory structure creation
2. Registry initialization
3. MLflow configuration setup
4. Idempotent re-initialization with --force
5. CLI argument parsing
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from formula_foundry.m3.cli_main import build_parser, cmd_init, main

if TYPE_CHECKING:
    pass


class TestBuildParser:
    """Tests for argument parser construction."""

    def test_parser_has_init_command(self) -> None:
        """Parser should have init subcommand."""
        parser = build_parser()
        # Verify by parsing valid init command
        args = parser.parse_args(["init"])
        assert args.command == "init"

    def test_init_default_arguments(self) -> None:
        """Init command should have correct default arguments."""
        parser = build_parser()
        args = parser.parse_args(["init"])
        assert args.root is None
        assert args.force is False
        assert args.quiet is False

    def test_init_with_root(self, tmp_path: Path) -> None:
        """Init command should accept --root argument."""
        parser = build_parser()
        args = parser.parse_args(["init", "--root", str(tmp_path)])
        assert args.root == tmp_path

    def test_init_with_force(self) -> None:
        """Init command should accept --force flag."""
        parser = build_parser()
        args = parser.parse_args(["init", "--force"])
        assert args.force is True

    def test_init_with_quiet(self) -> None:
        """Init command should accept --quiet flag."""
        parser = build_parser()
        args = parser.parse_args(["init", "-q"])
        assert args.quiet is True

    def test_version_argument(self) -> None:
        """Parser should have --version argument."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0


class TestCmdInit:
    """Tests for the init command implementation."""

    def test_init_creates_data_directories(self, tmp_path: Path) -> None:
        """Init should create the expected data directory structure."""
        result = cmd_init(root=tmp_path, force=False, quiet=True)
        assert result == 0

        # Verify directories exist
        assert (tmp_path / "data" / "objects").is_dir()
        assert (tmp_path / "data" / "manifests").is_dir()
        assert (tmp_path / "data" / "mlflow" / "artifacts").is_dir()
        assert (tmp_path / "data" / "datasets").is_dir()

    def test_init_creates_registry_db(self, tmp_path: Path) -> None:
        """Init should create and initialize the registry database."""
        result = cmd_init(root=tmp_path, force=False, quiet=True)
        assert result == 0

        registry_db = tmp_path / "data" / "registry.db"
        assert registry_db.exists()

        # Verify schema was created
        conn = sqlite3.connect(str(registry_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        conn.close()

        assert "artifacts" in tables
        assert "datasets" in tables
        assert "runs" in tables
        assert "schema_version" in tables

    def test_init_records_init_run(self, tmp_path: Path) -> None:
        """Init should record itself as a run in the registry."""
        result = cmd_init(root=tmp_path, force=False, quiet=True)
        assert result == 0

        registry_db = tmp_path / "data" / "registry.db"
        conn = sqlite3.connect(str(registry_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM runs WHERE stage_name = 'm3_init';")
        runs = cursor.fetchall()
        conn.close()

        assert len(runs) == 1
        run = runs[0]
        assert run["generator"] == "m3_init"
        assert run["status"] == "completed"
        assert run["run_id"].startswith("init-")
        # Run ID format: init-YYYYMMDDTHHMMSS-xxxxxxxx
        parts = run["run_id"].split("-")
        assert len(parts) == 3

    def test_init_is_idempotent_without_force(self, tmp_path: Path) -> None:
        """Init without --force should skip if already initialized."""
        # First init
        result1 = cmd_init(root=tmp_path, force=False, quiet=True)
        assert result1 == 0

        # Second init without force
        result2 = cmd_init(root=tmp_path, force=False, quiet=True)
        assert result2 == 0

        # Should still only have one init run
        registry_db = tmp_path / "data" / "registry.db"
        conn = sqlite3.connect(str(registry_db))
        cursor = conn.execute("SELECT COUNT(*) FROM runs WHERE stage_name = 'm3_init';")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1

    def test_init_with_force_reinitializes(self, tmp_path: Path) -> None:
        """Init with --force should reinitialize even if already done."""
        # First init
        result1 = cmd_init(root=tmp_path, force=False, quiet=True)
        assert result1 == 0

        # Second init with force
        result2 = cmd_init(root=tmp_path, force=True, quiet=True)
        assert result2 == 0

        # Should have two init runs
        registry_db = tmp_path / "data" / "registry.db"
        conn = sqlite3.connect(str(registry_db))
        cursor = conn.execute("SELECT COUNT(*) FROM runs WHERE stage_name = 'm3_init';")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 2

    def test_init_creates_mlflow_directories(self, tmp_path: Path) -> None:
        """Init should create MLflow-specific directories."""
        result = cmd_init(root=tmp_path, force=False, quiet=True)
        assert result == 0

        # MLflow directories should exist
        assert (tmp_path / "data" / "mlflow").is_dir()
        assert (tmp_path / "data" / "mlflow" / "artifacts").is_dir()


class TestMain:
    """Tests for the main entry point."""

    def test_main_init_success(self, tmp_path: Path) -> None:
        """Main with init command should succeed."""
        result = main(["init", "--root", str(tmp_path), "-q"])
        assert result == 0

    def test_main_init_creates_structure(self, tmp_path: Path) -> None:
        """Main with init should create the expected structure."""
        main(["init", "--root", str(tmp_path), "-q"])

        assert (tmp_path / "data" / "objects").is_dir()
        assert (tmp_path / "data" / "registry.db").exists()

    def test_main_no_command_fails(self) -> None:
        """Main without command should fail."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0


class TestInitWithExistingProject:
    """Tests for init in projects with existing configurations."""

    def test_init_with_dvc_config(self, tmp_path: Path) -> None:
        """Init should detect existing DVC configuration."""
        # Create mock .dvc directory
        dvc_dir = tmp_path / ".dvc"
        dvc_dir.mkdir()
        (dvc_dir / "config").write_text("[core]\nautostage = true\n")

        result = cmd_init(root=tmp_path, force=False, quiet=True)
        assert result == 0

    def test_init_without_dvc_config(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Init should warn if DVC is not configured."""
        result = cmd_init(root=tmp_path, force=False, quiet=False)
        assert result == 0

        captured = capsys.readouterr()
        assert "DVC not initialized" in captured.out or "DVC configuration found" in captured.out

    def test_init_finds_project_root_from_subdirectory(self, tmp_path: Path) -> None:
        """Init should find project root when run from subdirectory."""
        # Create project marker
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

        # Create subdirectory
        subdir = tmp_path / "src" / "subpackage"
        subdir.mkdir(parents=True)

        # Run init from subdirectory (simulated via root=None behavior)
        # Since we can't change cwd in tests, we test the helper directly
        from formula_foundry.m3.cli_main import _find_project_root

        found_root = _find_project_root(subdir)
        assert found_root == tmp_path


class TestIntegrationWithRegistry:
    """Integration tests with the registry module."""

    def test_init_registry_can_be_queried(self, tmp_path: Path) -> None:
        """Initialized registry should be queryable via ArtifactRegistry."""
        cmd_init(root=tmp_path, force=False, quiet=True)

        from formula_foundry.m3.registry import ArtifactRegistry

        registry = ArtifactRegistry(tmp_path / "data" / "registry.db")
        runs = registry.query_runs(status="completed")
        registry.close()

        assert len(runs) == 1
        assert runs[0].stage_name == "m3_init"

    def test_init_registry_stats_available(self, tmp_path: Path) -> None:
        """Initialized registry should return valid storage stats."""
        cmd_init(root=tmp_path, force=False, quiet=True)

        from formula_foundry.m3.registry import ArtifactRegistry

        registry = ArtifactRegistry(tmp_path / "data" / "registry.db")
        stats = registry.get_storage_stats()
        registry.close()

        assert stats["total_artifacts"] == 0
        assert stats["total_bytes"] == 0


class TestIntegrationWithArtifactStore:
    """Integration tests with the artifact store module."""

    def test_init_creates_usable_artifact_store(self, tmp_path: Path) -> None:
        """Initialized directories should work with ArtifactStore."""
        cmd_init(root=tmp_path, force=False, quiet=True)

        from formula_foundry.m3.artifact_store import ArtifactStore

        store = ArtifactStore(
            root=tmp_path / "data",
            generator="test",
            generator_version="1.0.0",
        )

        # Should be able to put an artifact
        manifest = store.put(
            content=b"test content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="test-run-001",
        )

        assert manifest.artifact_id.startswith("art-")
        assert store.exists(manifest.content_hash.digest)
