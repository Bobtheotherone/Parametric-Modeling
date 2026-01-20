from __future__ import annotations

import argparse

from formula_foundry.openems import cli_main


def test_openems_cli_commands_exist() -> None:
    parser = cli_main.build_parser()
    subparsers = None
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):
            subparsers = action
            break

    assert subparsers is not None
    commands = set(subparsers.choices.keys())
    assert {"version", "run"} <= commands
