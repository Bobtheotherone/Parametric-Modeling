"""Unit tests for KiCad CLI flags module.

Tests the CLI flag building functions used for DRC and export operations.

Satisfies REQ-M1-006: DRC with zone refill enabled and exports with zone
checks enabled (KiCad CLI flags/policy pinned in code and recorded in manifest).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from formula_foundry.coupongen.kicad import (
    DEFAULT_ZONE_POLICY,
    SeverityLevel,
    ZonePolicy,
    build_drc_flags,
    build_export_drill_flags,
    build_export_gerber_flags,
    get_drc_refill_flag,
    get_export_check_flag,
)


class TestBuildDrcFlags:
    """Tests for build_drc_flags function."""

    def test_basic_args_structure(self, tmp_path: Path) -> None:
        """Verify basic DRC args include required components."""
        board = tmp_path / "board.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_flags(board, report)

        assert args[0:2] == ["pcb", "drc"]
        assert str(board) in args
        assert "--output" in args
        assert str(report) in args

    def test_default_severity_is_all(self, tmp_path: Path) -> None:
        """Default severity should be 'all' for comprehensive checking."""
        board = tmp_path / "board.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_flags(board, report)

        assert "--severity-all" in args

    def test_severity_error_only(self, tmp_path: Path) -> None:
        """Verify severity can be set to 'error'."""
        board = tmp_path / "board.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_flags(board, report, severity="error")

        assert "--severity-error" in args
        assert "--severity-all" not in args

    def test_severity_warning_only(self, tmp_path: Path) -> None:
        """Verify severity can be set to 'warning'."""
        board = tmp_path / "board.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_flags(board, report, severity="warning")

        assert "--severity-warning" in args
        assert "--severity-all" not in args

    def test_refill_zones_enabled_by_default(self, tmp_path: Path) -> None:
        """REQ-M1-006: Zone refill should be enabled by default."""
        board = tmp_path / "board.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_flags(board, report)

        assert "--refill-zones" in args

    def test_refill_zones_can_be_disabled(self, tmp_path: Path) -> None:
        """Zone refill can be explicitly disabled."""
        board = tmp_path / "board.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_flags(board, report, refill_zones=False)

        assert "--refill-zones" not in args

    def test_refill_zones_can_be_explicitly_enabled(self, tmp_path: Path) -> None:
        """Zone refill can be explicitly enabled."""
        board = tmp_path / "board.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_flags(board, report, refill_zones=True)

        assert "--refill-zones" in args

    def test_exit_code_violations_included(self, tmp_path: Path) -> None:
        """Exit code violations flag should be included."""
        board = tmp_path / "board.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_flags(board, report)

        assert "--exit-code-violations" in args

    def test_json_format_specified(self, tmp_path: Path) -> None:
        """JSON format should be specified for programmatic parsing."""
        board = tmp_path / "board.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_flags(board, report)

        format_idx = args.index("--format")
        assert args[format_idx + 1] == "json"

    def test_custom_policy_overrides_default(self, tmp_path: Path) -> None:
        """Custom policy should override default settings."""
        board = tmp_path / "board.kicad_pcb"
        report = tmp_path / "drc.json"

        custom_policy = ZonePolicy(
            policy_id="test-policy",
            drc_refill_zones=False,
            drc_refill_flag="--refill-zones",
            export_check_zones=False,
            export_check_flag="--check-zones",
        )
        args = build_drc_flags(board, report, policy=custom_policy)

        # Custom policy has refill disabled
        assert "--refill-zones" not in args


class TestBuildExportGerberFlags:
    """Tests for build_export_gerber_flags function."""

    def test_basic_args_structure(self, tmp_path: Path) -> None:
        """Verify basic Gerber export args include required components."""
        board = tmp_path / "board.kicad_pcb"
        output = tmp_path / "gerbers"
        args = build_export_gerber_flags(board, output)

        assert args[0:3] == ["pcb", "export", "gerbers"]
        assert str(board) in args
        assert "--output" in args
        assert str(output) in args

    def test_check_zones_enabled_by_default(self, tmp_path: Path) -> None:
        """REQ-M1-006: Zone check should be enabled by default."""
        board = tmp_path / "board.kicad_pcb"
        output = tmp_path / "gerbers"
        args = build_export_gerber_flags(board, output)

        assert "--check-zones" in args

    def test_check_zones_can_be_disabled(self, tmp_path: Path) -> None:
        """Zone check can be explicitly disabled."""
        board = tmp_path / "board.kicad_pcb"
        output = tmp_path / "gerbers"
        args = build_export_gerber_flags(board, output, check_zones=False)

        assert "--check-zones" not in args

    def test_check_zones_can_be_explicitly_enabled(self, tmp_path: Path) -> None:
        """Zone check can be explicitly enabled."""
        board = tmp_path / "board.kicad_pcb"
        output = tmp_path / "gerbers"
        args = build_export_gerber_flags(board, output, check_zones=True)

        assert "--check-zones" in args

    def test_custom_policy_overrides_default(self, tmp_path: Path) -> None:
        """Custom policy should override default settings."""
        board = tmp_path / "board.kicad_pcb"
        output = tmp_path / "gerbers"

        custom_policy = ZonePolicy(
            policy_id="test-policy",
            drc_refill_zones=True,
            drc_refill_flag="--refill-zones",
            export_check_zones=False,
            export_check_flag="--check-zones",
        )
        args = build_export_gerber_flags(board, output, policy=custom_policy)

        # Custom policy has zone check disabled
        assert "--check-zones" not in args


class TestBuildExportDrillFlags:
    """Tests for build_export_drill_flags function."""

    def test_basic_args_structure(self, tmp_path: Path) -> None:
        """Verify basic drill export args include required components."""
        board = tmp_path / "board.kicad_pcb"
        output = tmp_path / "drill"
        args = build_export_drill_flags(board, output)

        assert args[0:3] == ["pcb", "export", "drill"]
        assert str(board) in args
        assert "--output" in args
        assert str(output) in args

    def test_check_zones_enabled_by_default(self, tmp_path: Path) -> None:
        """REQ-M1-006: Zone check should be enabled by default."""
        board = tmp_path / "board.kicad_pcb"
        output = tmp_path / "drill"
        args = build_export_drill_flags(board, output)

        assert "--check-zones" in args

    def test_check_zones_can_be_disabled(self, tmp_path: Path) -> None:
        """Zone check can be explicitly disabled."""
        board = tmp_path / "board.kicad_pcb"
        output = tmp_path / "drill"
        args = build_export_drill_flags(board, output, check_zones=False)

        assert "--check-zones" not in args

    def test_check_zones_can_be_explicitly_enabled(self, tmp_path: Path) -> None:
        """Zone check can be explicitly enabled."""
        board = tmp_path / "board.kicad_pcb"
        output = tmp_path / "drill"
        args = build_export_drill_flags(board, output, check_zones=True)

        assert "--check-zones" in args

    def test_custom_policy_overrides_default(self, tmp_path: Path) -> None:
        """Custom policy should override default settings."""
        board = tmp_path / "board.kicad_pcb"
        output = tmp_path / "drill"

        custom_policy = ZonePolicy(
            policy_id="test-policy",
            drc_refill_zones=True,
            drc_refill_flag="--refill-zones",
            export_check_zones=False,
            export_check_flag="--check-zones",
        )
        args = build_export_drill_flags(board, output, policy=custom_policy)

        # Custom policy has zone check disabled
        assert "--check-zones" not in args


class TestGetDrcRefillFlag:
    """Tests for get_drc_refill_flag function."""

    def test_default_policy_flag(self) -> None:
        """Default policy uses --refill-zones flag."""
        flag = get_drc_refill_flag()
        assert flag == "--refill-zones"

    def test_custom_policy_flag(self) -> None:
        """Custom policy flag is returned."""
        custom_policy = ZonePolicy(
            policy_id="custom",
            drc_refill_zones=True,
            drc_refill_flag="--custom-refill-flag",
            export_check_zones=True,
            export_check_flag="--check-zones",
        )
        flag = get_drc_refill_flag(policy=custom_policy)
        assert flag == "--custom-refill-flag"

    def test_flag_matches_default_policy(self) -> None:
        """Flag matches DEFAULT_ZONE_POLICY attribute."""
        flag = get_drc_refill_flag()
        assert flag == DEFAULT_ZONE_POLICY.drc_refill_flag


class TestGetExportCheckFlag:
    """Tests for get_export_check_flag function."""

    def test_default_policy_flag(self) -> None:
        """Default policy uses --check-zones flag."""
        flag = get_export_check_flag()
        assert flag == "--check-zones"

    def test_custom_policy_flag(self) -> None:
        """Custom policy flag is returned."""
        custom_policy = ZonePolicy(
            policy_id="custom",
            drc_refill_zones=True,
            drc_refill_flag="--refill-zones",
            export_check_zones=True,
            export_check_flag="--custom-check-flag",
        )
        flag = get_export_check_flag(policy=custom_policy)
        assert flag == "--custom-check-flag"

    def test_flag_matches_default_policy(self) -> None:
        """Flag matches DEFAULT_ZONE_POLICY attribute."""
        flag = get_export_check_flag()
        assert flag == DEFAULT_ZONE_POLICY.export_check_flag


class TestSeverityLevel:
    """Tests for SeverityLevel type."""

    def test_valid_severity_levels(self) -> None:
        """Verify valid severity level values."""
        # These are the valid literal values
        valid_levels: list[SeverityLevel] = ["all", "error", "warning"]
        for level in valid_levels:
            assert isinstance(level, str)


class TestReqM1006Compliance:
    """Integration tests for REQ-M1-006 compliance.

    REQ-M1-006: If CPWG uses zones, DRC MUST be run with zone refill enabled
    and exports MUST be run with zone checks enabled (KiCad CLI flags/policy
    pinned in code and recorded in manifest).
    """

    def test_default_drc_has_refill_zones(self, tmp_path: Path) -> None:
        """DRC commands must include zone refill by default."""
        board = tmp_path / "board.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_flags(board, report)

        assert "--refill-zones" in args

    def test_default_gerber_export_has_check_zones(self, tmp_path: Path) -> None:
        """Gerber export commands must include zone check by default."""
        board = tmp_path / "board.kicad_pcb"
        output = tmp_path / "gerbers"
        args = build_export_gerber_flags(board, output)

        assert "--check-zones" in args

    def test_default_drill_export_has_check_zones(self, tmp_path: Path) -> None:
        """Drill export commands must include zone check by default."""
        board = tmp_path / "board.kicad_pcb"
        output = tmp_path / "drill"
        args = build_export_drill_flags(board, output)

        assert "--check-zones" in args

    def test_all_exports_use_same_policy(self, tmp_path: Path) -> None:
        """All export operations should use the same default policy."""
        board = tmp_path / "board.kicad_pcb"

        gerber_args = build_export_gerber_flags(board, tmp_path / "gerbers")
        drill_args = build_export_drill_flags(board, tmp_path / "drill")

        # Both should include zone check
        assert "--check-zones" in gerber_args
        assert "--check-zones" in drill_args
