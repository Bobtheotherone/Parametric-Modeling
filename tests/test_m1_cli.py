from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    mock_spec.model_dump.return_value = {
        "toolchain": {"kicad": {"docker_image": "kicad/kicad:9.0.7"}}
    }

    with (
        patch("formula_foundry.coupongen.cli_main.load_spec", return_value=mock_spec),
        patch("formula_foundry.coupongen.cli_main.build_coupon", return_value=mock_build_result) as mock_build,
        patch("sys.stdout.write") as mock_stdout,
    ):
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text("schema_version: 1")
        exit_code = cli_main.main(["build", str(spec_path), "--out", str(tmp_path)])

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
    mock_spec.model_dump.return_value = {
        "toolchain": {"kicad": {"docker_image": "kicad/kicad:9.0.7"}}
    }

    with (
        patch("formula_foundry.coupongen.cli_main.load_spec", return_value=mock_spec),
        patch("formula_foundry.coupongen.cli_main.build_coupon", return_value=mock_build_result),
        patch("sys.stdout.write"),
    ):
        exit_code = cli_main.main(["build", "/tmp/spec.yaml", "--out", "/tmp"])

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
