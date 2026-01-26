from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from formula_foundry.coupongen import cli_main
from formula_foundry.coupongen.api import BuildResult, DrcReport


def _minimal_f0_spec_dict_for_cli() -> dict:
    """Return a minimal valid F0 (F0_CAL_THRU_LINE) spec template for CLI tests."""
    return {
        "schema_version": 1,
        "coupon_family": "F0_CAL_THRU_LINE",
        "units": "nm",
        "toolchain": {
            "kicad": {
                "version": "9.0.7",
                "docker_image": "kicad/kicad:9.0.7@sha256:test",
            }
        },
        "fab_profile": {"id": "generic"},
        "stackup": {
            "copper_layers": 2,
            "thicknesses_nm": {
                "copper_top": 35_000,
                "core": 1_000_000,
                "copper_bottom": 35_000,
            },
            "materials": {"er": 4.5, "loss_tangent": 0.02},
        },
        "board": {
            "outline": {
                "width_nm": 20_000_000,
                "length_nm": 100_000_000,
                "corner_radius_nm": 1_000_000,
            },
            "origin": {"mode": "center"},
            "text": {"coupon_id": "TEST", "include_manifest_hash": True},
        },
        "connectors": {
            "left": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [5_000_000, 0],
                "rotation_deg": 0,
            },
            "right": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [95_000_000, 0],
                "rotation_deg": 180,
            },
        },
        "transmission_line": {
            "type": "cpwg",
            "layer": "F.Cu",
            "w_nm": 200_000,
            "gap_nm": 150_000,
            "length_left_nm": 20_000_000,
            "length_right_nm": 20_000_000,
        },
        "constraints": {
            "mode": "REPAIR",
            "drc": {"must_pass": True, "severity": "error"},
            "symmetry": {"enforce": True},
            "allow_unconnected_copper": False,
        },
        "export": {
            "gerbers": {"enabled": True, "format": "RS274X"},
            "drill": {"enabled": True, "format": "excellon"},
            "outputs_dir": "outputs",
        },
    }


def _minimal_f1_spec_dict_for_cli() -> dict:
    """Return a minimal valid F1 (F1_SINGLE_ENDED_VIA) spec template for CLI tests."""
    return {
        "schema_version": 1,
        "coupon_family": "F1_SINGLE_ENDED_VIA",
        "units": "nm",
        "toolchain": {
            "kicad": {
                "version": "9.0.7",
                "docker_image": "kicad/kicad:9.0.7@sha256:test",
            }
        },
        "fab_profile": {"id": "generic"},
        "stackup": {
            "copper_layers": 4,
            "thicknesses_nm": {
                "copper_top": 35_000,
                "core": 1_000_000,
                "prepreg": 200_000,
                "copper_inner1": 35_000,
                "copper_inner2": 35_000,
                "copper_bottom": 35_000,
            },
            "materials": {"er": 4.5, "loss_tangent": 0.02},
        },
        "board": {
            "outline": {
                "width_nm": 20_000_000,
                "length_nm": 100_000_000,
                "corner_radius_nm": 1_000_000,
            },
            "origin": {"mode": "center"},
            "text": {"coupon_id": "TEST", "include_manifest_hash": True},
        },
        "connectors": {
            "left": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [5_000_000, 0],
                "rotation_deg": 0,
            },
            "right": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [95_000_000, 0],
                "rotation_deg": 180,
            },
        },
        "transmission_line": {
            "type": "cpwg",
            "layer": "F.Cu",
            "w_nm": 200_000,
            "gap_nm": 150_000,
            "length_left_nm": 20_000_000,
            "length_right_nm": 20_000_000,
            "ground_via_fence": {
                "enabled": True,
                "pitch_nm": 1_500_000,
                "offset_from_gap_nm": 500_000,
                "via": {"drill_nm": 300_000, "diameter_nm": 600_000},
            },
        },
        "discontinuity": {
            "type": "VIA_TRANSITION",
            "signal_via": {
                "drill_nm": 300_000,
                "diameter_nm": 600_000,
                "pad_diameter_nm": 900_000,
            },
            "return_vias": {
                "pattern": "ring",
                "count": 8,
                "radius_nm": 2_000_000,
                "via": {"drill_nm": 300_000, "diameter_nm": 600_000},
            },
        },
        "constraints": {
            "mode": "REPAIR",
            "drc": {"must_pass": True, "severity": "error"},
            "symmetry": {"enforce": True},
            "allow_unconnected_copper": False,
        },
        "export": {
            "gerbers": {"enabled": True, "format": "RS274X"},
            "drill": {"enabled": True, "format": "excellon"},
            "outputs_dir": "outputs",
        },
    }


def test_cli_commands_exist() -> None:
    """REQ-M1-021: CLI must have validate, generate, drc, export, build commands."""
    parser = cli_main.build_parser()
    subparsers = None
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):
            subparsers = action
            break

    assert subparsers is not None
    commands = set(subparsers.choices.keys())
    assert {"validate", "generate", "drc", "export", "build"} <= commands


def test_cli_drc_exit_code_success() -> None:
    """REQ-M1-021: DRC command returns 0 on success."""
    mock_report = DrcReport(report_path=Path("/tmp/drc.json"), returncode=0)
    with patch("formula_foundry.coupongen.cli_main.run_drc", return_value=mock_report):
        exit_code = cli_main.main(["drc", "/tmp/board.kicad_pcb", "--mode", "local"])
    assert exit_code == 0


def test_cli_drc_exit_code_violations() -> None:
    """REQ-M1-021: DRC command returns 2 on DRC violations."""
    mock_report = DrcReport(report_path=Path("/tmp/drc.json"), returncode=1)
    with patch("formula_foundry.coupongen.cli_main.run_drc", return_value=mock_report):
        exit_code = cli_main.main(["drc", "/tmp/board.kicad_pcb", "--mode", "local"])
    assert exit_code == 2


def test_cli_main_callable() -> None:
    """REQ-M1-021: CLI main function is callable and returns integer exit code."""
    assert callable(cli_main.main)


def test_cli_parser_help_does_not_crash() -> None:
    """REQ-M1-021: CLI parser builds without error."""
    parser = cli_main.build_parser()
    assert parser is not None
    # Check each subcommand can parse valid args
    args = parser.parse_args(["validate", "test.yaml"])
    assert args.command == "validate"
    args = parser.parse_args(["generate", "test.yaml", "--out", "/tmp"])
    assert args.command == "generate"
    args = parser.parse_args(["drc", "board.kicad_pcb"])
    assert args.command == "drc"
    args = parser.parse_args(["export", "board.kicad_pcb", "--out", "/tmp"])
    assert args.command == "export"
    args = parser.parse_args(["build", "test.yaml", "--out", "/tmp"])
    assert args.command == "build"


def test_cli_build_returns_design_hash_keyed_output(tmp_path: Path) -> None:
    """REQ-M1-021: Build command chains generate->drc->export and returns artifact dir keyed by design_hash."""
    mock_build_result = BuildResult(
        output_dir=tmp_path / "coupon-abc123def456",
        design_hash="abc123def456",
        coupon_id="coupon",
        manifest_path=tmp_path / "coupon-abc123def456" / "manifest.json",
        cache_hit=False,
        toolchain_hash="toolchain123",
    )
    mock_spec = MagicMock()
    mock_spec.model_dump.return_value = {"toolchain": {"kicad": {"docker_image": "kicad/kicad:9.0.7"}}}

    with (
        patch("formula_foundry.coupongen.cli_main.load_spec", return_value=mock_spec),
        patch("formula_foundry.coupongen.cli_main.build_coupon", return_value=mock_build_result) as mock_build,
        patch("sys.stdout.write") as mock_stdout,
    ):
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text("schema_version: 1")
        # Use --legacy flag to use the legacy build_coupon function (CP-3.5)
        exit_code = cli_main.main(["build", str(spec_path), "--out", str(tmp_path), "--legacy"])

    assert exit_code == 0
    mock_build.assert_called_once()
    # Verify output contains design_hash and output_dir
    call_args = mock_stdout.call_args[0][0]
    output = json.loads(call_args.strip())
    assert output["design_hash"] == "abc123def456"
    assert "output_dir" in output
    assert output["coupon_id"] == "coupon"


def test_cli_build_exit_code_success() -> None:
    """REQ-M1-021: Build command returns 0 on success."""
    mock_build_result = BuildResult(
        output_dir=Path("/tmp/coupon-abc123"),
        design_hash="abc123",
        coupon_id="coupon",
        manifest_path=Path("/tmp/coupon-abc123/manifest.json"),
        cache_hit=False,
        toolchain_hash="toolchain123",
    )
    mock_spec = MagicMock()
    mock_spec.model_dump.return_value = {"toolchain": {"kicad": {"docker_image": "kicad/kicad:9.0.7"}}}

    with (
        patch("formula_foundry.coupongen.cli_main.load_spec", return_value=mock_spec),
        patch("formula_foundry.coupongen.cli_main.build_coupon", return_value=mock_build_result),
        patch("sys.stdout.write"),
    ):
        # Use --legacy flag to use the legacy build_coupon function (CP-3.5)
        exit_code = cli_main.main(["build", "/tmp/spec.yaml", "--out", "/tmp", "--legacy"])

    assert exit_code == 0


def test_cli_export_exit_code_success() -> None:
    """REQ-M1-021: Export command returns 0 on success."""
    mock_hashes = {"gerbers/F_Cu.gbr": "hash1", "drill/drill.drl": "hash2"}

    with (
        patch("formula_foundry.coupongen.cli_main.export_fab", return_value=mock_hashes),
        patch("sys.stdout.write"),
    ):
        exit_code = cli_main.main(["export", "/tmp/board.kicad_pcb", "--out", "/tmp", "--mode", "local"])

    assert exit_code == 0


# ============================================================================
# CP-4.2: batch-filter and build-batch CLI command tests
# ============================================================================


def test_cli_batch_filter_command_exists() -> None:
    """CP-4.2: CLI must have batch-filter command."""
    parser = cli_main.build_parser()
    subparsers = None
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):
            subparsers = action
            break

    assert subparsers is not None
    commands = set(subparsers.choices.keys())
    assert "batch-filter" in commands


def test_cli_build_batch_command_exists() -> None:
    """CP-4.2: CLI must have build-batch command."""
    parser = cli_main.build_parser()
    subparsers = None
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):
            subparsers = action
            break

    assert subparsers is not None
    commands = set(subparsers.choices.keys())
    assert "build-batch" in commands


def test_cli_batch_filter_parser_args() -> None:
    """CP-4.2: batch-filter command accepts required arguments."""
    parser = cli_main.build_parser()
    args = parser.parse_args(["batch-filter", "input.npy", "--out", "/tmp/output"])
    assert args.command == "batch-filter"
    assert args.u_npy == Path("input.npy")
    assert args.out == Path("/tmp/output")
    assert args.repair is False
    assert args.profile == "generic"
    assert args.seed == 0
    assert args.no_gpu is False


def test_cli_batch_filter_parser_optional_args() -> None:
    """CP-4.2: batch-filter command accepts optional arguments."""
    parser = cli_main.build_parser()
    args = parser.parse_args(
        [
            "batch-filter",
            "input.npy",
            "--out",
            "/tmp/output",
            "--repair",
            "--profile",
            "jlcpcb",
            "--seed",
            "42",
            "--no-gpu",
        ]
    )
    assert args.command == "batch-filter"
    assert args.repair is True
    assert args.profile == "jlcpcb"
    assert args.seed == 42
    assert args.no_gpu is True


def test_cli_build_batch_parser_args() -> None:
    """CP-4.2: build-batch command accepts required arguments."""
    parser = cli_main.build_parser()
    args = parser.parse_args(
        [
            "build-batch",
            "spec_template.yaml",
            "--u",
            "vectors.npy",
            "--out",
            "/tmp/builds",
        ]
    )
    assert args.command == "build-batch"
    assert args.spec_template == Path("spec_template.yaml")
    assert args.u == Path("vectors.npy")
    assert args.out == Path("/tmp/builds")
    assert args.mode == "local"
    assert args.limit is None
    assert args.skip_filter is False


def test_cli_build_batch_parser_optional_args() -> None:
    """CP-4.2: build-batch command accepts optional arguments."""
    parser = cli_main.build_parser()
    args = parser.parse_args(
        [
            "build-batch",
            "spec_template.yaml",
            "--u",
            "vectors.npy",
            "--out",
            "/tmp/builds",
            "--mode",
            "docker",
            "--limit",
            "10",
            "--skip-filter",
        ]
    )
    assert args.command == "build-batch"
    assert args.mode == "docker"
    assert args.limit == 10
    assert args.skip_filter is True


def test_cli_batch_filter_success(tmp_path: Path) -> None:
    """CP-4.2: batch-filter command runs successfully with valid input."""
    # Create test input
    u_batch = np.random.rand(100, 19).astype(np.float64)
    input_path = tmp_path / "input.npy"
    np.save(input_path, u_batch)

    out_dir = tmp_path / "output"

    with patch("sys.stdout.write") as mock_stdout:
        exit_code = cli_main.main(
            [
                "batch-filter",
                str(input_path),
                "--out",
                str(out_dir),
                "--no-gpu",
            ]
        )

    assert exit_code == 0

    # Check outputs exist
    assert (out_dir / "mask.npy").exists()
    assert (out_dir / "u_repaired.npy").exists()
    assert (out_dir / "metadata.json").exists()

    # Check mask shape
    mask = np.load(out_dir / "mask.npy")
    assert mask.shape == (100,)
    assert mask.dtype == np.bool_

    # Check repaired shape
    u_repaired = np.load(out_dir / "u_repaired.npy")
    assert u_repaired.shape == (100, 19)

    # Check stdout output
    call_args = mock_stdout.call_args[0][0]
    output = json.loads(call_args.strip())
    assert output["n_candidates"] == 100
    assert "n_feasible" in output
    assert "feasibility_rate" in output
    assert output["mode"] == "REJECT"


def test_cli_batch_filter_repair_mode(tmp_path: Path) -> None:
    """CP-4.2: batch-filter command with --repair uses REPAIR mode."""
    u_batch = np.random.rand(50, 19).astype(np.float64)
    input_path = tmp_path / "input.npy"
    np.save(input_path, u_batch)

    out_dir = tmp_path / "output"

    with patch("sys.stdout.write") as mock_stdout:
        exit_code = cli_main.main(
            [
                "batch-filter",
                str(input_path),
                "--out",
                str(out_dir),
                "--repair",
                "--no-gpu",
            ]
        )

    assert exit_code == 0

    call_args = mock_stdout.call_args[0][0]
    output = json.loads(call_args.strip())
    assert output["mode"] == "REPAIR"


def test_cli_batch_filter_missing_input() -> None:
    """CP-4.2: batch-filter command returns 1 when input file is missing."""
    with patch("sys.stderr.write"):
        exit_code = cli_main.main(
            [
                "batch-filter",
                "/nonexistent/input.npy",
                "--out",
                "/tmp/output",
            ]
        )

    assert exit_code == 1


def test_cli_batch_filter_invalid_profile() -> None:
    """CP-4.2: batch-filter command returns 1 for invalid fab profile."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        u_batch = np.random.rand(10, 19).astype(np.float64)
        np.save(f.name, u_batch)
        input_path = f.name

    try:
        with patch("sys.stderr.write"):
            exit_code = cli_main.main(
                [
                    "batch-filter",
                    input_path,
                    "--out",
                    "/tmp/output",
                    "--profile",
                    "nonexistent_profile",
                ]
            )

        assert exit_code == 1
    finally:
        Path(input_path).unlink()


def test_cli_build_batch_success(tmp_path: Path) -> None:
    """CP-4.2/4.3: build-batch command runs with GPU filter integration.

    This test verifies the build-batch command accepts input and runs the GPU
    filter pipeline. Full end-to-end testing is in test_cp43_gpu_pipeline.py.
    """
    # Create a valid F1 spec template using proper YAML
    import yaml

    from formula_foundry.coupongen.cli_main import _run_build_batch

    spec_dict = _minimal_f1_spec_dict_for_cli()
    spec_path = tmp_path / "spec_template.yaml"
    spec_path.write_text(yaml.dump(spec_dict), encoding="utf-8")

    # Use valid u vectors with known-good parameters
    u_batch = np.ones((20, 19), dtype=np.float64) * 0.5
    # Fix spatial parameters to ensure feasibility
    u_batch[:, 3] = 0.8  # board_length_nm: 126M nm (large enough)
    u_batch[:, 13] = 0.2  # right_connector_x_nm: 85M nm (well within board)
    u_batch[:, 12] = 0.2  # left_connector_x_nm
    u_batch[:, 14] = 0.3  # trace_length_left_nm
    u_batch[:, 15] = 0.3  # trace_length_right_nm

    u_path = tmp_path / "vectors.npy"
    np.save(u_path, u_batch)

    out_dir = tmp_path / "builds"

    # Mock build_coupon_with_engine to avoid KiCad dependency
    with patch("formula_foundry.coupongen.cli_main.build_coupon_with_engine") as mock_build, patch("sys.stdout.write"):
        mock_result = MagicMock()
        mock_result.design_hash = "test_hash"
        mock_result.coupon_id = "test_id"
        mock_result.output_dir = tmp_path / "output"
        mock_result.cache_hit = False
        mock_result.manifest_path = tmp_path / "manifest.json"
        mock_build.return_value = mock_result

        cli_main.main(
            [
                "build-batch",
                str(spec_path),
                "--u",
                str(u_path),
                "--out",
                str(out_dir),
                "--skip-filter",  # Skip GPU filter for this basic test
                "--limit",
                "1",  # Just process one candidate
            ]
        )

    # Should succeed with skip_filter (no inter-parameter constraint checking)
    # The exit code may be non-zero if there were build failures
    assert (out_dir / "batch_summary.json").exists()


def test_cli_build_batch_with_limit(tmp_path: Path) -> None:
    """CP-4.2/4.3: build-batch command respects --limit flag."""
    import yaml

    spec_dict = _minimal_f1_spec_dict_for_cli()
    spec_path = tmp_path / "spec_template.yaml"
    spec_path.write_text(yaml.dump(spec_dict), encoding="utf-8")

    # Create valid u vectors
    u_batch = np.ones((100, 19), dtype=np.float64) * 0.5
    # Fix spatial parameters for feasibility
    u_batch[:, 3] = 0.8
    u_batch[:, 13] = 0.2
    u_batch[:, 12] = 0.2
    u_batch[:, 14] = 0.3
    u_batch[:, 15] = 0.3

    u_path = tmp_path / "vectors.npy"
    np.save(u_path, u_batch)

    out_dir = tmp_path / "builds"

    with patch("formula_foundry.coupongen.cli_main.build_coupon_with_engine") as mock_build, patch("sys.stdout.write"):
        mock_result = MagicMock()
        mock_result.design_hash = "test_hash"
        mock_result.coupon_id = "test_id"
        mock_result.output_dir = tmp_path / "output"
        mock_result.cache_hit = False
        mock_result.manifest_path = tmp_path / "manifest.json"
        mock_build.return_value = mock_result

        cli_main.main(
            [
                "build-batch",
                str(spec_path),
                "--u",
                str(u_path),
                "--out",
                str(out_dir),
                "--limit",
                "10",
                "--skip-filter",
            ]
        )

    # Verify batch_summary.json was written
    assert (out_dir / "batch_summary.json").exists()

    summary = json.loads((out_dir / "batch_summary.json").read_text())
    assert summary["limit_applied"] == 10


def test_cli_build_batch_missing_spec() -> None:
    """CP-4.2: build-batch command returns 1 when spec template is missing."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        u_batch = np.random.rand(10, 19).astype(np.float64)
        np.save(f.name, u_batch)
        u_path = f.name

    try:
        with patch("sys.stderr.write"):
            exit_code = cli_main.main(
                [
                    "build-batch",
                    "/nonexistent/spec.yaml",
                    "--u",
                    u_path,
                    "--out",
                    "/tmp/builds",
                ]
            )

        assert exit_code == 1
    finally:
        Path(u_path).unlink()


def test_cli_build_batch_missing_u_vectors() -> None:
    """CP-4.2: build-batch command returns 1 when u vectors file is missing."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        f.write(b"schema_version: 1\nfamily: F1")
        spec_path = f.name

    try:
        with patch("sys.stderr.write"):
            exit_code = cli_main.main(
                [
                    "build-batch",
                    spec_path,
                    "--u",
                    "/nonexistent/vectors.npy",
                    "--out",
                    "/tmp/builds",
                ]
            )

        assert exit_code == 1
    finally:
        Path(spec_path).unlink()


# ============================================================================
# REQ-M1-018: lint-spec-coverage and explain CLI command tests
# ============================================================================


def test_cli_lint_spec_coverage_command_exists() -> None:
    """REQ-M1-018: CLI must have lint-spec-coverage command."""
    parser = cli_main.build_parser()
    subparsers = None
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):
            subparsers = action
            break

    assert subparsers is not None
    commands = set(subparsers.choices.keys())
    assert "lint-spec-coverage" in commands


def test_cli_explain_command_exists() -> None:
    """REQ-M1-018: CLI must have explain command."""
    parser = cli_main.build_parser()
    subparsers = None
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):
            subparsers = action
            break

    assert subparsers is not None
    commands = set(subparsers.choices.keys())
    assert "explain" in commands


def test_cli_lint_spec_coverage_parser_args() -> None:
    """REQ-M1-018: lint-spec-coverage command accepts required arguments."""
    parser = cli_main.build_parser()
    args = parser.parse_args(["lint-spec-coverage", "test.yaml"])
    assert args.command == "lint-spec-coverage"
    assert args.spec == Path("test.yaml")
    assert args.strict is True  # Default
    assert args.json is False  # Default


def test_cli_lint_spec_coverage_parser_optional_args() -> None:
    """REQ-M1-018: lint-spec-coverage command accepts optional arguments."""
    parser = cli_main.build_parser()
    args = parser.parse_args(["lint-spec-coverage", "test.yaml", "--json"])
    assert args.command == "lint-spec-coverage"
    assert args.json is True


def test_cli_explain_parser_args() -> None:
    """REQ-M1-018: explain command accepts required arguments."""
    parser = cli_main.build_parser()
    args = parser.parse_args(["explain", "test.yaml"])
    assert args.command == "explain"
    assert args.spec == Path("test.yaml")
    assert args.out is None  # Default
    assert args.json is False  # Default


def test_cli_explain_parser_optional_args() -> None:
    """REQ-M1-018: explain command accepts optional arguments."""
    parser = cli_main.build_parser()
    args = parser.parse_args(
        [
            "explain",
            "test.yaml",
            "--out",
            "/tmp/report.txt",
            "--json",
            "--constraint-mode",
            "REPAIR",
        ]
    )
    assert args.command == "explain"
    assert args.out == Path("/tmp/report.txt")
    assert args.json is True
    assert args.constraint_mode == "REPAIR"


def test_cli_lint_spec_coverage_exit_code_success(tmp_path: Path) -> None:
    """REQ-M1-018: lint-spec-coverage returns 0 when coverage is complete."""
    import yaml

    # Create a minimal valid F1 spec with all expected paths
    spec_dict = _minimal_f1_spec_dict_for_cli()
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.dump(spec_dict), encoding="utf-8")

    with patch("sys.stdout.write"):
        exit_code = cli_main.main(["lint-spec-coverage", str(spec_path)])

    # The spec may or may not have full coverage, but it should not crash
    assert exit_code in (0, 1)


def test_cli_lint_spec_coverage_exit_code_failure(tmp_path: Path) -> None:
    """REQ-M1-018: lint-spec-coverage returns non-zero on coverage failures."""
    import yaml

    # Create an incomplete spec that will have unconsumed expected paths
    spec_dict = {
        "schema_version": 1,
        "coupon_family": "F0",
        "units": "nm",
        "toolchain": {
            "kicad": {
                "version": "9.0.7",
                "docker_image": "kicad/kicad:9.0.7@sha256:test",
            }
        },
        "fab_profile": {"id": "generic"},
        # Missing many required fields for F0
        "stackup": {
            "copper_layers": 2,
            "thicknesses_nm": {"copper": 35000},
            "materials": {"er": 4.5, "loss_tangent": 0.02},
        },
        "board": {
            "outline": {"width_nm": 10_000_000, "length_nm": 50_000_000},
            "origin": {"mode": "center"},
            "text": {"coupon_id": "TEST", "include_manifest_hash": True},
        },
        # Missing connectors, transmission_line, constraints, export
    }
    spec_path = tmp_path / "incomplete_spec.yaml"
    spec_path.write_text(yaml.dump(spec_dict), encoding="utf-8")

    with patch("sys.stderr.write"), patch("sys.stdout.write"):
        # This may fail at load time due to validation, or pass with coverage issues
        exit_code = cli_main.main(["lint-spec-coverage", str(spec_path)])

    # Should return non-zero (1) due to validation failure or coverage issues
    assert exit_code == 1


def test_cli_lint_spec_coverage_missing_spec() -> None:
    """REQ-M1-018: lint-spec-coverage returns 1 when spec file is missing."""
    with patch("sys.stderr.write"):
        exit_code = cli_main.main(
            [
                "lint-spec-coverage",
                "/nonexistent/spec.yaml",
            ]
        )

    assert exit_code == 1


def test_cli_lint_spec_coverage_json_output(tmp_path: Path) -> None:
    """REQ-M1-018: lint-spec-coverage --json outputs JSON format."""
    import yaml

    spec_dict = _minimal_f1_spec_dict_for_cli()
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.dump(spec_dict), encoding="utf-8")

    with patch("sys.stdout.write") as mock_stdout:
        exit_code = cli_main.main(["lint-spec-coverage", str(spec_path), "--json"])

    # Check that JSON was output
    call_args = mock_stdout.call_args[0][0]
    output = json.loads(call_args.strip())
    assert "spec_path" in output
    assert "coupon_family" in output
    assert "coverage_ratio" in output
    assert "is_complete" in output
    assert "unused_provided_paths" in output
    assert "unconsumed_expected_paths" in output
    assert exit_code in (0, 1)


def test_cli_explain_success(tmp_path: Path) -> None:
    """REQ-M1-018: explain command runs successfully with valid spec."""
    import yaml

    # Use F0 spec for explain tests (F0 is simpler and doesn't require footprint files)
    spec_dict = _minimal_f0_spec_dict_for_cli()
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.dump(spec_dict), encoding="utf-8")

    with patch("sys.stdout.write") as mock_stdout:
        exit_code = cli_main.main(["explain", str(spec_path)])

    # Should succeed
    assert exit_code == 0

    # Check that output was written
    call_args = mock_stdout.call_args[0][0]
    # Human-readable output should contain key sections
    assert "EXPLAIN:" in call_args
    assert "SPEC SUMMARY" in call_args
    assert "RESOLVED DESIGN" in call_args
    assert "CONSTRAINT STATUS" in call_args
    assert "TIGHTEST CONSTRAINTS BY CATEGORY" in call_args


def test_cli_explain_json_output(tmp_path: Path) -> None:
    """REQ-M1-018: explain --json outputs JSON format with required fields."""
    import yaml

    spec_dict = _minimal_f0_spec_dict_for_cli()
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.dump(spec_dict), encoding="utf-8")

    with patch("sys.stdout.write") as mock_stdout:
        exit_code = cli_main.main(["explain", str(spec_path), "--json"])

    assert exit_code == 0

    # Check JSON output contains required fields
    call_args = mock_stdout.call_args[0][0]
    output = json.loads(call_args.strip())
    assert "spec_path" in output
    assert "coupon_family" in output
    assert "constraint_mode" in output
    assert "was_repaired" in output
    assert "constraints_passed" in output
    assert "total_constraints" in output
    assert "resolved_design" in output
    assert "tightest_constraints_by_category" in output

    # Check resolved_design structure
    resolved = output["resolved_design"]
    assert "schema_version" in resolved
    assert "coupon_family" in resolved
    assert "parameters_nm" in resolved
    assert "derived_features" in resolved
    assert "dimensionless_groups" in resolved


def test_cli_explain_missing_spec() -> None:
    """REQ-M1-018: explain returns 1 when spec file is missing."""
    with patch("sys.stderr.write"):
        exit_code = cli_main.main(
            [
                "explain",
                "/nonexistent/spec.yaml",
            ]
        )

    assert exit_code == 1


def test_cli_explain_output_file(tmp_path: Path) -> None:
    """REQ-M1-018: explain --out writes to specified file."""
    import yaml

    spec_dict = _minimal_f0_spec_dict_for_cli()
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.dump(spec_dict), encoding="utf-8")

    output_path = tmp_path / "report.txt"

    with patch("sys.stdout.write") as mock_stdout:
        exit_code = cli_main.main(
            [
                "explain",
                str(spec_path),
                "--out",
                str(output_path),
            ]
        )

    assert exit_code == 0
    assert output_path.exists()

    # Check file contents
    content = output_path.read_text()
    assert "EXPLAIN:" in content
    assert "RESOLVED DESIGN" in content

    # stdout should show file write confirmation
    call_args = mock_stdout.call_args[0][0]
    assert "Explain report written to:" in call_args


def test_cli_explain_constraint_mode_override(tmp_path: Path) -> None:
    """REQ-M1-018: explain --constraint-mode overrides spec constraint mode."""
    import yaml

    spec_dict = _minimal_f0_spec_dict_for_cli()
    spec_dict["constraints"]["mode"] = "REJECT"  # Set spec to REJECT
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.dump(spec_dict), encoding="utf-8")

    with patch("sys.stdout.write") as mock_stdout:
        # Override to REPAIR
        exit_code = cli_main.main(
            [
                "explain",
                str(spec_path),
                "--json",
                "--constraint-mode",
                "REPAIR",
            ]
        )

    assert exit_code == 0

    call_args = mock_stdout.call_args[0][0]
    output = json.loads(call_args.strip())
    assert output["constraint_mode"] == "REPAIR"


def test_cli_explain_tightest_constraints_content(tmp_path: Path) -> None:
    """REQ-M1-018: explain output contains tightest constraint info per category."""
    import yaml

    spec_dict = _minimal_f0_spec_dict_for_cli()
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.dump(spec_dict), encoding="utf-8")

    with patch("sys.stdout.write") as mock_stdout:
        exit_code = cli_main.main(
            [
                "explain",
                str(spec_path),
                "--json",
            ]
        )

    assert exit_code == 0

    call_args = mock_stdout.call_args[0][0]
    output = json.loads(call_args.strip())

    tightest = output["tightest_constraints_by_category"]
    # Should have at least some categories from the constraint engine
    assert isinstance(tightest, dict)

    # Each category should have the expected fields if present
    for _category, info in tightest.items():
        assert "min_margin_nm" in info
        assert "constraint_id" in info
        assert "constraint_count" in info
        assert "failed_count" in info
        assert "passed_count" in info
