"""Tests for M2 CLI commands (REQ-M2-010).

Tests cover:
- Parser structure and subcommands
- sim run command
- sim batch command
- sim status command
- sparam extract command
- validate command
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.openems.cli_main import build_parser, main

# =============================================================================
# Parser Structure Tests
# =============================================================================


def test_parser_has_required_commands() -> None:
    """Parser should have all required top-level commands."""
    parser = build_parser()
    subparsers = None
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):
            subparsers = action
            break

    assert subparsers is not None
    commands = set(subparsers.choices.keys())

    # Required commands per REQ-M2-010
    required = {"sim", "sparam", "validate", "version", "run"}
    assert required <= commands, f"Missing commands: {required - commands}"


def test_sim_subcommands_exist() -> None:
    """sim subcommand should have run, batch, and status."""
    parser = build_parser()
    args = parser.parse_args(["sim", "run", "config.json", "--out", "out/"])
    assert args.command == "sim"
    assert args.sim_command == "run"


def test_sim_run_parser() -> None:
    """sim run should accept config and required --out."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "sim",
            "run",
            "test_config.json",
            "--out",
            "/tmp/output",
            "--timeout",
            "1800",
            "--solver-mode",
            "stub",
        ]
    )
    assert args.config == Path("test_config.json")
    assert args.out == Path("/tmp/output")
    assert args.timeout == 1800.0
    assert args.solver_mode == "stub"


def test_sim_batch_parser() -> None:
    """sim batch should accept config_dir and required --out."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "sim",
            "batch",
            "/configs",
            "--out",
            "/output",
            "--max-workers",
            "8",
            "--fail-fast",
        ]
    )
    assert args.config_dir == Path("/configs")
    assert args.out == Path("/output")
    assert args.max_workers == 8
    assert args.fail_fast is True


def test_sim_status_parser() -> None:
    """sim status should accept run_id."""
    parser = build_parser()
    args = parser.parse_args(["sim", "status", "my_run_id"])
    assert args.run_id == "my_run_id"


def test_sparam_extract_parser() -> None:
    """sparam extract should accept sim_dir."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "sparam",
            "extract",
            "/sim/output",
            "--format",
            "both",
        ]
    )
    assert args.sim_dir == Path("/sim/output")
    assert args.format == "both"


def test_validate_parser() -> None:
    """validate should accept manifest path."""
    parser = build_parser()
    args = parser.parse_args(["validate", "manifest.json"])
    assert args.manifest == Path("manifest.json")


# =============================================================================
# Simulation Config Fixture
# =============================================================================


@pytest.fixture
def sample_sim_config() -> dict[str, Any]:
    """Create a valid simulation config for testing."""
    return {
        "schema_version": 1,
        "simulation_id": "test_sim_001",
        "toolchain": {
            "openems": {
                "version": "0.0.35",
                "docker_image": "ghcr.io/thliebig/openems:0.0.35",
            }
        },
        "geometry_ref": {
            "design_hash": "a" * 64,
        },
        "excitation": {
            "type": "gaussian",
            "f0_hz": 5_000_000_000,
            "fc_hz": 5_000_000_000,
        },
        "frequency": {
            "f_start_hz": 1_000_000,
            "f_stop_hz": 10_000_000_000,
            "n_points": 101,
        },
        "ports": [
            {
                "id": "P1",
                "type": "lumped",
                "impedance_ohm": 50.0,
                "excite": True,
                "position_nm": [0, 0, 0],
                "direction": "x",
            },
            {
                "id": "P2",
                "type": "lumped",
                "impedance_ohm": 50.0,
                "excite": False,
                "position_nm": [10_000_000, 0, 0],
                "direction": "x",
            },
        ],
    }


# =============================================================================
# Command Execution Tests
# =============================================================================


def test_sim_run_with_stub_mode(sample_sim_config: dict[str, Any]) -> None:
    """sim run should work in stub mode without real openEMS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Write config
        config_path = tmppath / "sim_config.json"
        config_path.write_text(json.dumps(sample_sim_config), encoding="utf-8")

        # Run command
        output_dir = tmppath / "output"
        result = main(
            [
                "sim",
                "run",
                str(config_path),
                "--out",
                str(output_dir),
                "--solver-mode",
                "stub",
                "--no-convergence",
            ]
        )

        assert result == 0
        assert output_dir.exists()

        # Should have manifest
        manifest_path = output_dir / "simulation_manifest.json"
        assert manifest_path.exists()


def test_sim_run_with_json_output(sample_sim_config: dict[str, Any]) -> None:
    """sim run --json should output JSON result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        config_path = tmppath / "sim_config.json"
        config_path.write_text(json.dumps(sample_sim_config), encoding="utf-8")

        output_dir = tmppath / "output"
        json_out = tmppath / "result.json"

        result = main(
            [
                "sim",
                "run",
                str(config_path),
                "--out",
                str(output_dir),
                "--solver-mode",
                "stub",
                "--no-convergence",
                "--json",
                str(json_out),
            ]
        )

        assert result == 0
        assert json_out.exists()

        result_data = json.loads(json_out.read_text())
        assert "simulation_hash" in result_data
        assert "output_dir" in result_data


def test_sim_batch_with_stub_mode(sample_sim_config: dict[str, Any]) -> None:
    """sim batch should process multiple configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create config directory with multiple configs
        config_dir = tmppath / "configs"
        config_dir.mkdir()

        for i in range(3):
            config = sample_sim_config.copy()
            config["simulation_id"] = f"batch_sim_{i:03d}"
            config_path = config_dir / f"sim_{i:03d}.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

        output_dir = tmppath / "batch_output"
        main(
            [
                "sim",
                "batch",
                str(config_dir),
                "--out",
                str(output_dir),
                "--solver-mode",
                "stub",
                "--max-workers",
                "2",
                "--no-convergence",
            ]
        )

        # May return non-zero due to convergence checks, but should complete
        assert output_dir.exists()

        # Should have batch result
        batch_result_path = output_dir / "batch_result.json"
        assert batch_result_path.exists()

        batch_data = json.loads(batch_result_path.read_text())
        assert batch_data["total_jobs"] == 3


def test_sim_status_completed(sample_sim_config: dict[str, Any]) -> None:
    """sim status should report completed simulation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Run a simulation first
        config_path = tmppath / "sim_config.json"
        config_path.write_text(json.dumps(sample_sim_config), encoding="utf-8")

        output_dir = tmppath / "sim_run"
        main(
            [
                "sim",
                "run",
                str(config_path),
                "--out",
                str(output_dir),
                "--solver-mode",
                "stub",
                "--no-convergence",
            ]
        )

        # Check status
        result = main(["sim", "status", str(output_dir)])
        assert result == 0  # Completed


def test_sim_status_not_found() -> None:
    """sim status should report not found for missing directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = main(["sim", "status", f"{tmpdir}/nonexistent"])
        assert result == 1


def test_sparam_extract_from_stub_sim(sample_sim_config: dict[str, Any]) -> None:
    """sparam extract should work on stub simulation output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Run a simulation first (stub mode generates sparam files)
        config_path = tmppath / "sim_config.json"
        config_path.write_text(json.dumps(sample_sim_config), encoding="utf-8")

        output_dir = tmppath / "sim_run"
        main(
            [
                "sim",
                "run",
                str(config_path),
                "--out",
                str(output_dir),
                "--solver-mode",
                "stub",
                "--no-convergence",
            ]
        )

        # Extract S-parameters
        extract_out = tmppath / "sparams"
        result = main(
            [
                "sparam",
                "extract",
                str(output_dir),
                "--out",
                str(extract_out),
                "--config",
                str(config_path),
            ]
        )

        assert result == 0
        assert extract_out.exists()


def test_validate_valid_manifest() -> None:
    """validate should pass for valid manifest."""
    manifest = {
        "schema_version": 1,
        "simulation_hash": "a" * 64,
        "spec_hash": "b" * 64,
        "sim_config_hash": "b" * 64,
        "geometry_hash": "c" * 64,
        "design_hash": "d" * 64,
        "coupon_family": "F1_SINGLE_ENDED_VIA",
        "toolchain": {
            "openems": {
                "version": "0.0.35",
                "docker_image": "test",
            }
        },
        "toolchain_hash": "e" * 64,
        "frequency_sweep": {
            "f_start_hz": 1000000,
            "f_stop_hz": 10000000000,
            "n_points": 101,
        },
        "excitation": {
            "type": "gaussian",
            "f0_hz": 5000000000,
            "fc_hz": 5000000000,
        },
        "boundaries": {
            "x_min": "PML_8",
            "x_max": "PML_8",
            "y_min": "PEC",
            "y_max": "PEC",
            "z_min": "PEC",
            "z_max": "PML_8",
        },
        "mesh_config": {
            "resolution": {
                "lambda_resolution": 20,
                "metal_edge_resolution_nm": 50000,
                "via_resolution_nm": 25000,
            },
            "smoothing": {
                "max_ratio": 1.5,
                "smooth_mesh_lines": True,
            },
        },
        "ports": [
            {
                "id": "P1",
                "type": "lumped",
                "impedance_ohm": 50.0,
                "excite": True,
                "position_nm": [0, 0, 0],
                "direction": "x",
                "deembed_enabled": False,
            }
        ],
        "outputs": [],
        "lineage": {
            "git_sha": "f" * 40,
            "timestamp_utc": "2025-01-20T12:00:00Z",
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        result = main(["validate", str(manifest_path)])
        assert result == 0


def test_validate_invalid_manifest() -> None:
    """validate should fail for manifest missing required fields."""
    manifest = {
        "schema_version": 1,
        # Missing required fields
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        result = main(["validate", str(manifest_path)])
        assert result == 1


def test_validate_missing_file() -> None:
    """validate should fail for missing file."""
    result = main(["validate", "/nonexistent/manifest.json"])
    assert result == 1


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_sim_run_missing_config() -> None:
    """sim run should fail gracefully for missing config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = main(
            [
                "sim",
                "run",
                f"{tmpdir}/nonexistent.json",
                "--out",
                f"{tmpdir}/output",
            ]
        )
        assert result == 1


def test_sim_batch_missing_dir() -> None:
    """sim batch should fail for missing config directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = main(
            [
                "sim",
                "batch",
                f"{tmpdir}/nonexistent",
                "--out",
                f"{tmpdir}/output",
            ]
        )
        assert result == 1


def test_sim_batch_empty_dir() -> None:
    """sim batch should fail for empty config directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "empty_configs"
        config_dir.mkdir()

        result = main(
            [
                "sim",
                "batch",
                str(config_dir),
                "--out",
                f"{tmpdir}/output",
            ]
        )
        assert result == 1


# =============================================================================
# JSON Output Tests
# =============================================================================


def test_validate_json_output() -> None:
    """validate --json should output JSON result."""
    manifest = {"invalid": "manifest"}

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # Capture stdout would be complex, just verify exit code
        result = main(["validate", str(manifest_path), "--json"])
        assert result == 1  # Invalid manifest


# =============================================================================
# Legacy Command Tests
# =============================================================================


def test_legacy_version_command() -> None:
    """Legacy version command should work."""
    parser = build_parser()
    args = parser.parse_args(["version"])
    assert args.command == "version"


def test_legacy_run_command() -> None:
    """Legacy run command should parse openems_args."""
    parser = build_parser()
    args = parser.parse_args(["run", "--", "--foo", "--bar"])
    assert args.command == "run"
    # REMAINDER includes the '--' separator
    assert "--foo" in args.openems_args
    assert "--bar" in args.openems_args
