from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from formula_foundry.coupongen import cli_main
from formula_foundry.coupongen.api import BuildResult, DrcReport


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
    args = parser.parse_args([
        "batch-filter", "input.npy",
        "--out", "/tmp/output",
        "--repair",
        "--profile", "jlcpcb",
        "--seed", "42",
        "--no-gpu",
    ])
    assert args.command == "batch-filter"
    assert args.repair is True
    assert args.profile == "jlcpcb"
    assert args.seed == 42
    assert args.no_gpu is True


def test_cli_build_batch_parser_args() -> None:
    """CP-4.2: build-batch command accepts required arguments."""
    parser = cli_main.build_parser()
    args = parser.parse_args([
        "build-batch", "spec_template.yaml",
        "--u", "vectors.npy",
        "--out", "/tmp/builds",
    ])
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
    args = parser.parse_args([
        "build-batch", "spec_template.yaml",
        "--u", "vectors.npy",
        "--out", "/tmp/builds",
        "--mode", "docker",
        "--limit", "10",
        "--skip-filter",
    ])
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
        exit_code = cli_main.main([
            "batch-filter",
            str(input_path),
            "--out", str(out_dir),
            "--no-gpu",
        ])

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
        exit_code = cli_main.main([
            "batch-filter",
            str(input_path),
            "--out", str(out_dir),
            "--repair",
            "--no-gpu",
        ])

    assert exit_code == 0

    call_args = mock_stdout.call_args[0][0]
    output = json.loads(call_args.strip())
    assert output["mode"] == "REPAIR"


def test_cli_batch_filter_missing_input() -> None:
    """CP-4.2: batch-filter command returns 1 when input file is missing."""
    with patch("sys.stderr.write"):
        exit_code = cli_main.main([
            "batch-filter",
            "/nonexistent/input.npy",
            "--out", "/tmp/output",
        ])

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
            exit_code = cli_main.main([
                "batch-filter",
                input_path,
                "--out", "/tmp/output",
                "--profile", "nonexistent_profile",
            ])

        assert exit_code == 1
    finally:
        Path(input_path).unlink()


def test_cli_build_batch_success(tmp_path: Path) -> None:
    """CP-4.2: build-batch command runs successfully with valid input."""
    # Create test input files
    spec_path = tmp_path / "spec_template.yaml"
    spec_path.write_text("schema_version: 1\nfamily: F1", encoding="utf-8")

    u_batch = np.random.rand(20, 19).astype(np.float64)
    u_path = tmp_path / "vectors.npy"
    np.save(u_path, u_batch)

    out_dir = tmp_path / "builds"

    with patch("sys.stdout.write") as mock_stdout:
        exit_code = cli_main.main([
            "build-batch",
            str(spec_path),
            "--u", str(u_path),
            "--out", str(out_dir),
        ])

    assert exit_code == 0

    call_args = mock_stdout.call_args[0][0]
    output = json.loads(call_args.strip())
    assert output["n_input_vectors"] == 20
    assert output["status"] == "interface_ready"


def test_cli_build_batch_with_limit(tmp_path: Path) -> None:
    """CP-4.2: build-batch command respects --limit flag."""
    spec_path = tmp_path / "spec_template.yaml"
    spec_path.write_text("schema_version: 1\nfamily: F1", encoding="utf-8")

    u_batch = np.random.rand(100, 19).astype(np.float64)
    u_path = tmp_path / "vectors.npy"
    np.save(u_path, u_batch)

    out_dir = tmp_path / "builds"

    with patch("sys.stdout.write") as mock_stdout:
        exit_code = cli_main.main([
            "build-batch",
            str(spec_path),
            "--u", str(u_path),
            "--out", str(out_dir),
            "--limit", "10",
        ])

    assert exit_code == 0

    call_args = mock_stdout.call_args[0][0]
    output = json.loads(call_args.strip())
    assert output["limit"] == 10


def test_cli_build_batch_missing_spec() -> None:
    """CP-4.2: build-batch command returns 1 when spec template is missing."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        u_batch = np.random.rand(10, 19).astype(np.float64)
        np.save(f.name, u_batch)
        u_path = f.name

    try:
        with patch("sys.stderr.write"):
            exit_code = cli_main.main([
                "build-batch",
                "/nonexistent/spec.yaml",
                "--u", u_path,
                "--out", "/tmp/builds",
            ])

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
            exit_code = cli_main.main([
                "build-batch",
                spec_path,
                "--u", "/nonexistent/vectors.npy",
                "--out", "/tmp/builds",
            ])

        assert exit_code == 1
    finally:
        Path(spec_path).unlink()
