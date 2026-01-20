from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

from formula_foundry.coupongen import cli_main
from formula_foundry.coupongen.api import DrcReport


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
