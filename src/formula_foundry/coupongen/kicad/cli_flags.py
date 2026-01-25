"""CLI flag builders for KiCad DRC and export operations.

This module provides functions to build command-line arguments for
kicad-cli operations with deterministic zone policy enforcement.

Satisfies REQ-M1-006: DRC with zone refill enabled and exports with
zone checks enabled (KiCad CLI flags pinned in code).

Usage:
    from formula_foundry.coupongen.kicad.cli_flags import (
        build_drc_flags,
        build_export_gerber_flags,
        build_export_drill_flags,
    )

    # Build DRC command arguments
    drc_args = build_drc_flags(
        board_path=Path("board.kicad_pcb"),
        report_path=Path("drc.json"),
    )
    # Returns: ["pcb", "drc", "--severity-all", "--refill-zones", ...]

    # Build Gerber export arguments
    gerber_args = build_export_gerber_flags(
        board_path=Path("board.kicad_pcb"),
        output_dir=Path("gerbers"),
    )
    # Returns: ["pcb", "export", "gerbers", "--check-zones", ...]
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .policy import DEFAULT_ZONE_POLICY, ZonePolicy


SeverityLevel = Literal["all", "error", "warning"]


def build_drc_flags(
    board_path: Path,
    report_path: Path,
    *,
    severity: SeverityLevel = "all",
    refill_zones: bool | None = None,
    policy: ZonePolicy | None = None,
) -> list[str]:
    """Build kicad-cli DRC command arguments.

    Satisfies REQ-M1-006 and REQ-M1-016:
    - --severity-all: Report all violations including warnings (default for M1)
    - --format json: Output in JSON format for programmatic parsing
    - --exit-code-violations: Return non-zero exit code if violations exist
    - --refill-zones: Ensure copper zones are filled before DRC (REQ-M1-006)

    Args:
        board_path: Path to the .kicad_pcb file to check.
        report_path: Path where the JSON DRC report will be written.
        severity: Severity level to check ("error", "warning", "all").
            Default is "all" to check all violations.
        refill_zones: Whether to force zone refill before DRC. If None,
            uses the policy default (refill enabled).
        policy: Zone policy to use. Defaults to DEFAULT_ZONE_POLICY.

    Returns:
        List of command-line arguments for kicad-cli pcb drc.

    Example:
        >>> args = build_drc_flags(Path("board.kicad_pcb"), Path("drc.json"))
        >>> args[0:3]
        ['pcb', 'drc', '--severity-all']
        >>> '--refill-zones' in args
        True
    """
    effective_policy = policy or DEFAULT_ZONE_POLICY
    use_refill = refill_zones if refill_zones is not None else effective_policy.drc_refill_zones

    severity_arg = f"--severity-{severity}"
    args = ["pcb", "drc", severity_arg]

    if use_refill:
        args.append(effective_policy.drc_refill_flag)

    args.extend([
        "--exit-code-violations",
        "--format",
        "json",
        "--output",
        str(report_path),
        str(board_path),
    ])

    return args


def build_export_gerber_flags(
    board_path: Path,
    output_dir: Path,
    *,
    check_zones: bool | None = None,
    policy: ZonePolicy | None = None,
) -> list[str]:
    """Build kicad-cli Gerber export command arguments.

    Satisfies REQ-M1-006 and REQ-M1-017:
    - --check-zones: Validate zone integrity during export (REQ-M1-006)
    - --output: Specify output directory for Gerber files

    Args:
        board_path: Path to the .kicad_pcb file.
        output_dir: Output directory for Gerber files.
        check_zones: Whether to check zones during export. If None,
            uses the policy default (check enabled).
        policy: Zone policy to use. Defaults to DEFAULT_ZONE_POLICY.

    Returns:
        List of command-line arguments for kicad-cli pcb export gerbers.

    Example:
        >>> args = build_export_gerber_flags(Path("board.kicad_pcb"), Path("out"))
        >>> args[0:3]
        ['pcb', 'export', 'gerbers']
        >>> '--check-zones' in args
        True
    """
    effective_policy = policy or DEFAULT_ZONE_POLICY
    use_check = check_zones if check_zones is not None else effective_policy.export_check_zones

    args = ["pcb", "export", "gerbers"]

    if use_check:
        args.append(effective_policy.export_check_flag)

    args.extend([
        "--output",
        str(output_dir),
        str(board_path),
    ])

    return args


def build_export_drill_flags(
    board_path: Path,
    output_dir: Path,
    *,
    check_zones: bool | None = None,
    policy: ZonePolicy | None = None,
) -> list[str]:
    """Build kicad-cli drill export command arguments.

    Satisfies REQ-M1-006 and REQ-M1-017:
    - --check-zones: Validate zone integrity during export (REQ-M1-006)
    - --output: Specify output directory for drill files

    Args:
        board_path: Path to the .kicad_pcb file.
        output_dir: Output directory for drill files.
        check_zones: Whether to check zones during export. If None,
            uses the policy default (check enabled).
        policy: Zone policy to use. Defaults to DEFAULT_ZONE_POLICY.

    Returns:
        List of command-line arguments for kicad-cli pcb export drill.

    Example:
        >>> args = build_export_drill_flags(Path("board.kicad_pcb"), Path("out"))
        >>> args[0:3]
        ['pcb', 'export', 'drill']
        >>> '--check-zones' in args
        True
    """
    effective_policy = policy or DEFAULT_ZONE_POLICY
    use_check = check_zones if check_zones is not None else effective_policy.export_check_zones

    args = ["pcb", "export", "drill"]

    if use_check:
        args.append(effective_policy.export_check_flag)

    args.extend([
        "--output",
        str(output_dir),
        str(board_path),
    ])

    return args


def get_drc_refill_flag(policy: ZonePolicy | None = None) -> str:
    """Get the DRC refill zones flag.

    Args:
        policy: Zone policy to use. Defaults to DEFAULT_ZONE_POLICY.

    Returns:
        The CLI flag string for zone refill (e.g., "--refill-zones").
    """
    effective_policy = policy or DEFAULT_ZONE_POLICY
    return effective_policy.drc_refill_flag


def get_export_check_flag(policy: ZonePolicy | None = None) -> str:
    """Get the export check zones flag.

    Args:
        policy: Zone policy to use. Defaults to DEFAULT_ZONE_POLICY.

    Returns:
        The CLI flag string for zone check (e.g., "--check-zones").
    """
    effective_policy = policy or DEFAULT_ZONE_POLICY
    return effective_policy.export_check_flag


__all__ = [
    "SeverityLevel",
    "build_drc_flags",
    "build_export_gerber_flags",
    "build_export_drill_flags",
    "get_drc_refill_flag",
    "get_export_check_flag",
]
