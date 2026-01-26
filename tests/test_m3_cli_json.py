"""Tests for M3 CLI JSON output modes.

These tests verify:
1. m3 init --json output mode
2. m3 dataset list command and --json output
3. m3 run --json output mode
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from formula_foundry.m3.cli_main import (
    build_parser,
    cmd_dataset_list,
    cmd_init,
    main,
)


class TestInitJsonOutput:
    """Tests for init command JSON output mode."""

    def test_init_json_output_success(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Init with --json should output valid JSON on success."""
        result = cmd_init(root=tmp_path, force=False, quiet=False, output_json=True)
        assert result == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert data["status"] == "success"
        assert data["project_root"] == str(tmp_path)
        assert "initialized_utc" in data
        assert "init_registry" in data["steps_completed"]
        assert "create_directories" in data["steps_completed"]

    def test_init_json_output_already_initialized(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Init with --json should indicate already initialized."""
        # First init
        cmd_init(root=tmp_path, force=False, quiet=True, output_json=False)

        # Second init should return already_initialized
        result = cmd_init(root=tmp_path, force=False, quiet=False, output_json=True)
        assert result == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert data["status"] == "already_initialized"
        assert "Already initialized" in data["message"]

    def test_init_json_parser_flag(self) -> None:
        """Parser should accept --json flag for init command."""
        parser = build_parser()
        args = parser.parse_args(["init", "--json"])
        assert args.output_json is True


class TestDatasetListCommand:
    """Tests for the dataset list command."""

    def test_parser_has_dataset_list_command(self) -> None:
        """Parser should have dataset list subcommand."""
        parser = build_parser()
        args = parser.parse_args(["dataset", "list"])
        assert args.command == "dataset"
        assert args.dataset_command == "list"

    def test_dataset_list_default_arguments(self) -> None:
        """Dataset list should have correct default arguments."""
        parser = build_parser()
        args = parser.parse_args(["dataset", "list"])
        assert args.root is None
        assert args.name is None
        assert args.limit is None
        assert args.output_json is False
        assert args.long_format is False
        assert args.quiet is False

    def test_dataset_list_with_name_filter(self) -> None:
        """Dataset list should accept --name argument."""
        parser = build_parser()
        args = parser.parse_args(["dataset", "list", "--name", "test_dataset"])
        assert args.name == "test_dataset"

    def test_dataset_list_with_limit(self) -> None:
        """Dataset list should accept --limit argument."""
        parser = build_parser()
        args = parser.parse_args(["dataset", "list", "--limit", "10"])
        assert args.limit == 10

    def test_dataset_list_with_json(self) -> None:
        """Dataset list should accept --json argument."""
        parser = build_parser()
        args = parser.parse_args(["dataset", "list", "--json"])
        assert args.output_json is True

    def test_dataset_list_with_long_format(self) -> None:
        """Dataset list should accept --long argument."""
        parser = build_parser()
        args = parser.parse_args(["dataset", "list", "-l"])
        assert args.long_format is True

    def test_dataset_list_empty_store(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Dataset list with no datasets should return empty JSON list."""
        # Initialize store but don't create datasets
        cmd_init(root=tmp_path, force=False, quiet=True)

        result = cmd_dataset_list(
            root=tmp_path,
            name_filter=None,
            limit=None,
            output_json=True,
            long_format=False,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert data["count"] == 0
        assert data["datasets"] == []

    def test_dataset_list_with_datasets(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Dataset list should list existing datasets."""
        # Initialize store
        cmd_init(root=tmp_path, force=False, quiet=True)

        # Create a mock dataset manifest
        datasets_dir = tmp_path / "data" / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal dataset manifest
        dataset_manifest = {
            "schema_version": 1,
            "dataset_id": "test_dataset",
            "version": "v1",
            "name": "Test Dataset",
            "description": "A test dataset for CLI testing",
            "created_utc": "2026-01-21T00:00:00Z",
            "member_count": 5,
            "total_bytes": 1000,
            "content_hash": {
                "algorithm": "sha256",
                "digest": "abc123" + "0" * 58,
            },
            "provenance": {
                "generator": "test",
                "generator_version": "1.0.0",
            },
            "members": [],
        }

        manifest_path = datasets_dir / "test_dataset_v1.json"
        manifest_path.write_text(json.dumps(dataset_manifest))

        result = cmd_dataset_list(
            root=tmp_path,
            name_filter=None,
            limit=None,
            output_json=True,
            long_format=False,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert data["count"] == 1
        assert len(data["datasets"]) == 1
        assert data["datasets"][0]["dataset_id"] == "test_dataset"

    def test_dataset_list_name_filter(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Dataset list with --name should filter datasets."""
        # Initialize store
        cmd_init(root=tmp_path, force=False, quiet=True)

        # Create mock dataset manifests
        datasets_dir = tmp_path / "data" / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)

        for name in ["alpha_v1", "beta_v1", "alpha_v2"]:
            dataset_id = name.rsplit("_", 1)[0]
            version = name.rsplit("_", 1)[1]
            manifest = {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "version": version,
                "created_utc": "2026-01-21T00:00:00Z",
                "member_count": 1,
                "total_bytes": 100,
                "content_hash": {
                    "algorithm": "sha256",
                    "digest": "abc123" + "0" * 58,
                },
                "provenance": {
                    "generator": "test",
                    "generator_version": "1.0.0",
                },
                "members": [],
            }
            (datasets_dir / f"{name}.json").write_text(json.dumps(manifest))

        # Filter by name prefix
        result = cmd_dataset_list(
            root=tmp_path,
            name_filter="alpha",
            limit=None,
            output_json=True,
            long_format=False,
            quiet=False,
        )
        assert result == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert data["count"] == 1
        assert data["datasets"][0]["dataset_id"] == "alpha"


class TestRunJsonOutput:
    """Tests for run command JSON output mode."""

    def test_run_json_parser_flag(self) -> None:
        """Parser should accept --json flag for run command."""
        parser = build_parser()
        args = parser.parse_args(["run", "test_stage", "--json"])
        assert args.output_json is True

    def test_run_dry_run_json_output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Run --dry-run with --json should output valid JSON."""
        # Initialize store and create dvc.yaml
        cmd_init(root=tmp_path, force=False, quiet=True)
        (tmp_path / "dvc.yaml").write_text("stages:\n  test: {}\n")

        # Initialize git (required for run)
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "add", "."],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            cwd=tmp_path,
            capture_output=True,
        )

        # Run with dry-run and JSON output
        result = main(["run", "test_stage", "--dry-run", "--json", "--root", str(tmp_path)])
        assert result == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert data["mode"] == "dry_run"
        assert data["stage"] == "test_stage"
        assert "run_id" in data
        assert "git_commit" in data


class TestMainJsonFlags:
    """Tests for JSON flags through main entry point."""

    def test_main_init_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Main with init --json should output JSON."""
        result = main(["init", "--root", str(tmp_path), "--json"])
        assert result == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "success"

    def test_main_dataset_list_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Main with dataset list --json should output JSON."""
        cmd_init(root=tmp_path, force=False, quiet=True)

        result = main(["dataset", "list", "--root", str(tmp_path), "--json"])
        assert result == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "count" in data
        assert "datasets" in data
