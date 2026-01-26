"""Tests for DRC runner wrapper with JSON report parsing.

REQ-M1-016: Tests for DRC wrapper that runs kicad-cli pcb drc
with --severity-all --exit-code-violations --format json.
Tests JSON report parsing, exit code handling (0=pass, 5=violations),
and Tier 4 constraint gate integration.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from formula_foundry.coupongen.constraints.drc import (
    DRCError,
    DRCExitCode,
    DRCReport,
    DRCResult,
    DRCViolation,
    Tier4DrcChecker,
    check_drc_gate,
    run_drc,
)

# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

SAMPLE_DRC_REPORT_CLEAN = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "source": "/path/to/board.kicad_pcb",
    "violations": [],
    "unconnected_items": [],
    "schematic_parity": [],
    "coordinate_units": "mm",
}

SAMPLE_DRC_REPORT_VIOLATIONS = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "source": "/path/to/board.kicad_pcb",
    "violations": [
        {
            "type": "clearance",
            "severity": "error",
            "description": "Track has insufficient clearance to via",
            "pos": {"x": 10.5, "y": 20.3},
            "items": ["track:1", "via:2"],
        },
        {
            "type": "min_width",
            "severity": "warning",
            "description": "Track width is below design rule minimum",
            "pos": {"x": 15.0, "y": 25.0},
            "items": ["track:3"],
        },
    ],
    "unconnected_items": [],
    "schematic_parity": [],
    "coordinate_units": "mm",
}

SAMPLE_DRC_REPORT_UNCONNECTED = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "source": "/path/to/board.kicad_pcb",
    "violations": [],
    "unconnected_items": [
        {"description": "Unconnected item at pad"},
    ],
    "schematic_parity": [],
    "coordinate_units": "mm",
}


# ---------------------------------------------------------------------------
# DRCViolation Tests
# ---------------------------------------------------------------------------


class TestDRCViolation:
    """Tests for DRCViolation dataclass."""

    def test_from_dict_basic(self) -> None:
        """Parse basic violation from dict."""
        data = {
            "type": "clearance",
            "severity": "error",
            "description": "Track clearance violation",
        }
        v = DRCViolation.from_dict(data)

        assert v.type == "clearance"
        assert v.severity == "error"
        assert v.description == "Track clearance violation"
        assert v.pos_x_mm is None
        assert v.pos_y_mm is None
        assert v.items == ()

    def test_from_dict_with_position(self) -> None:
        """Parse violation with position data."""
        data = {
            "type": "clearance",
            "severity": "error",
            "description": "Test",
            "pos": {"x": 10.5, "y": 20.3},
        }
        v = DRCViolation.from_dict(data)

        assert v.pos_x_mm == 10.5
        assert v.pos_y_mm == 20.3

    def test_from_dict_with_items(self) -> None:
        """Parse violation with affected items."""
        data = {
            "type": "clearance",
            "severity": "error",
            "description": "Test",
            "items": ["track:1", "via:2"],
        }
        v = DRCViolation.from_dict(data)

        assert v.items == ("track:1", "via:2")

    def test_is_error_property(self) -> None:
        """Test is_error property."""
        error_v = DRCViolation(type="clearance", severity="error", description="test")
        warning_v = DRCViolation(type="width", severity="warning", description="test")

        assert error_v.is_error is True
        assert error_v.is_warning is False
        assert warning_v.is_error is False
        assert warning_v.is_warning is True


# ---------------------------------------------------------------------------
# DRCReport Tests
# ---------------------------------------------------------------------------


class TestDRCReport:
    """Tests for DRCReport dataclass."""

    def test_from_dict_clean(self) -> None:
        """Parse clean DRC report."""
        report = DRCReport.from_dict(SAMPLE_DRC_REPORT_CLEAN, exit_code=0)

        assert report.source == "/path/to/board.kicad_pcb"
        assert report.violations == ()
        assert report.unconnected_items == ()
        assert report.coordinate_units == "mm"
        assert report.exit_code == 0

    def test_from_dict_with_violations(self) -> None:
        """Parse DRC report with violations."""
        report = DRCReport.from_dict(SAMPLE_DRC_REPORT_VIOLATIONS, exit_code=5)

        assert len(report.violations) == 2
        assert report.violations[0].type == "clearance"
        assert report.violations[0].severity == "error"
        assert report.violations[1].type == "min_width"
        assert report.violations[1].severity == "warning"
        assert report.exit_code == 5

    def test_passed_property_clean(self) -> None:
        """Clean report should pass."""
        report = DRCReport.from_dict(SAMPLE_DRC_REPORT_CLEAN, exit_code=0)
        assert report.passed is True

    def test_passed_property_violations(self) -> None:
        """Report with violations should not pass."""
        report = DRCReport.from_dict(SAMPLE_DRC_REPORT_VIOLATIONS, exit_code=5)
        assert report.passed is False

    def test_error_count(self) -> None:
        """Count error-severity violations."""
        report = DRCReport.from_dict(SAMPLE_DRC_REPORT_VIOLATIONS, exit_code=5)
        assert report.error_count == 1
        assert report.warning_count == 1
        assert report.total_violations == 2

    def test_get_errors(self) -> None:
        """Get only error-severity violations."""
        report = DRCReport.from_dict(SAMPLE_DRC_REPORT_VIOLATIONS, exit_code=5)
        errors = report.get_errors()

        assert len(errors) == 1
        assert errors[0].severity == "error"

    def test_get_warnings(self) -> None:
        """Get only warning-severity violations."""
        report = DRCReport.from_dict(SAMPLE_DRC_REPORT_VIOLATIONS, exit_code=5)
        warnings = report.get_warnings()

        assert len(warnings) == 1
        assert warnings[0].severity == "warning"

    def test_from_json_string(self) -> None:
        """Parse from JSON string."""
        json_str = json.dumps(SAMPLE_DRC_REPORT_CLEAN)
        report = DRCReport.from_json_string(json_str, exit_code=0)

        assert report.passed is True

    def test_from_json_file(self, tmp_path: Path) -> None:
        """Parse from JSON file."""
        report_path = tmp_path / "drc.json"
        report_path.write_text(json.dumps(SAMPLE_DRC_REPORT_CLEAN), encoding="utf-8")

        report = DRCReport.from_json_file(report_path, exit_code=0)

        assert report.passed is True

    def test_to_dict(self) -> None:
        """Convert report to dict for serialization."""
        report = DRCReport.from_dict(SAMPLE_DRC_REPORT_VIOLATIONS, exit_code=5)
        data = report.to_dict()

        assert data["source"] == "/path/to/board.kicad_pcb"
        assert data["exit_code"] == 5
        assert data["passed"] is False
        assert data["violation_count"] == 2
        assert data["error_count"] == 1
        assert data["warning_count"] == 1
        assert len(data["violations"]) == 2


# ---------------------------------------------------------------------------
# DRC Exit Code Tests
# ---------------------------------------------------------------------------


class TestDRCExitCode:
    """Tests for DRC exit code constants."""

    def test_exit_code_pass(self) -> None:
        """Exit code 0 means DRC passed."""
        assert DRCExitCode.PASS == 0

    def test_exit_code_violations(self) -> None:
        """Exit code 5 means DRC found violations."""
        assert DRCExitCode.VIOLATIONS == 5


# ---------------------------------------------------------------------------
# run_drc Function Tests
# ---------------------------------------------------------------------------


class TestRunDrc:
    """Tests for run_drc function."""

    def test_run_drc_clean(self, tmp_path: Path) -> None:
        """Run DRC on clean board."""
        board_path = tmp_path / "board.kicad_pcb"
        report_path = tmp_path / "drc.json"
        board_path.write_text("(kicad_pcb)", encoding="utf-8")

        # Mock runner
        mock_runner = MagicMock()
        mock_runner.run_drc.return_value = subprocess.CompletedProcess(
            args=["kicad-cli", "pcb", "drc"],
            returncode=0,
            stdout="",
            stderr="",
        )

        # Write mock report
        def write_report(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            report_path.write_text(json.dumps(SAMPLE_DRC_REPORT_CLEAN), encoding="utf-8")
            return subprocess.CompletedProcess(
                args=["kicad-cli", "pcb", "drc"],
                returncode=0,
                stdout="",
                stderr="",
            )

        mock_runner.run_drc.side_effect = write_report

        result = run_drc(mock_runner, board_path, report_path)

        assert result.passed is True
        assert result.returncode == 0
        assert result.report.total_violations == 0

    def test_run_drc_with_violations(self, tmp_path: Path) -> None:
        """Run DRC with violations."""
        board_path = tmp_path / "board.kicad_pcb"
        report_path = tmp_path / "drc.json"
        board_path.write_text("(kicad_pcb)", encoding="utf-8")

        # Mock runner with violations
        def write_report(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            report_path.write_text(json.dumps(SAMPLE_DRC_REPORT_VIOLATIONS), encoding="utf-8")
            return subprocess.CompletedProcess(
                args=["kicad-cli", "pcb", "drc"],
                returncode=5,
                stdout="",
                stderr="DRC violations found",
            )

        mock_runner = MagicMock()
        mock_runner.run_drc.side_effect = write_report

        result = run_drc(mock_runner, board_path, report_path)

        assert result.passed is False
        assert result.returncode == 5
        assert result.has_violations is True
        assert result.report.total_violations == 2

    def test_run_drc_board_not_found(self, tmp_path: Path) -> None:
        """Run DRC on non-existent board should raise."""
        board_path = tmp_path / "nonexistent.kicad_pcb"
        report_path = tmp_path / "drc.json"

        mock_runner = MagicMock()

        with pytest.raises(FileNotFoundError, match="Board file not found"):
            run_drc(mock_runner, board_path, report_path)


# ---------------------------------------------------------------------------
# check_drc_gate Function Tests
# ---------------------------------------------------------------------------


class TestCheckDrcGate:
    """Tests for check_drc_gate function."""

    def test_gate_pass_clean_report(self, tmp_path: Path) -> None:
        """Gate passes for clean DRC."""
        report = DRCReport.from_dict(SAMPLE_DRC_REPORT_CLEAN, exit_code=0)
        result = DRCResult(
            report=report,
            returncode=0,
            stdout="",
            stderr="",
            report_path=tmp_path / "drc.json",
            board_path=tmp_path / "board.kicad_pcb",
        )

        assert check_drc_gate(result, must_pass=True) is True

    def test_gate_fail_with_violations(self, tmp_path: Path) -> None:
        """Gate fails when must_pass and violations exist."""
        report = DRCReport.from_dict(SAMPLE_DRC_REPORT_VIOLATIONS, exit_code=5)
        result = DRCResult(
            report=report,
            returncode=5,
            stdout="",
            stderr="",
            report_path=tmp_path / "drc.json",
            board_path=tmp_path / "board.kicad_pcb",
        )

        with pytest.raises(DRCError, match="KiCad DRC failed"):
            check_drc_gate(result, must_pass=True)

    def test_gate_allow_warnings(self, tmp_path: Path) -> None:
        """Gate passes when only warnings and allow_warnings=True."""
        # Report with only warnings
        report_data = {
            "source": "/path/to/board.kicad_pcb",
            "violations": [
                {"type": "width", "severity": "warning", "description": "Test warning"},
            ],
            "unconnected_items": [],
            "schematic_parity": [],
            "coordinate_units": "mm",
        }
        report = DRCReport.from_dict(report_data, exit_code=5)
        result = DRCResult(
            report=report,
            returncode=5,
            stdout="",
            stderr="",
            report_path=tmp_path / "drc.json",
            board_path=tmp_path / "board.kicad_pcb",
        )

        # Should pass when allow_warnings=True
        assert check_drc_gate(result, must_pass=True, allow_warnings=True) is True

    def test_gate_no_must_pass(self, tmp_path: Path) -> None:
        """Gate doesn't raise when must_pass=False."""
        report = DRCReport.from_dict(SAMPLE_DRC_REPORT_VIOLATIONS, exit_code=5)
        result = DRCResult(
            report=report,
            returncode=5,
            stdout="",
            stderr="",
            report_path=tmp_path / "drc.json",
            board_path=tmp_path / "board.kicad_pcb",
        )

        # Should not raise
        passed = check_drc_gate(result, must_pass=False)
        assert passed is False


# ---------------------------------------------------------------------------
# Tier4DrcChecker Tests
# ---------------------------------------------------------------------------


class TestTier4DrcChecker:
    """Tests for Tier 4 DRC constraint checker."""

    def test_tier_property(self) -> None:
        """Tier 4 checker reports tier T4."""
        checker = Tier4DrcChecker()
        assert checker.tier == "T4"

    def test_no_drc_result_fails(self) -> None:
        """Checker fails when DRC not run."""
        checker = Tier4DrcChecker(drc_result=None)
        results = checker.check(None, {})

        assert len(results) == 1
        assert results[0].constraint_id == "T4_DRC_EXECUTED"
        assert results[0].passed is False
        assert "not been run" in results[0].reason

    def test_clean_drc_passes(self, tmp_path: Path) -> None:
        """Checker passes for clean DRC."""
        report = DRCReport.from_dict(SAMPLE_DRC_REPORT_CLEAN, exit_code=0)
        drc_result = DRCResult(
            report=report,
            returncode=0,
            stdout="",
            stderr="",
            report_path=tmp_path / "drc.json",
            board_path=tmp_path / "board.kicad_pcb",
        )

        checker = Tier4DrcChecker(drc_result=drc_result)
        results = checker.check(None, {})

        # All constraints should pass
        assert all(r.passed for r in results)
        assert any(r.constraint_id == "T4_DRC_EXECUTED" for r in results)
        assert any(r.constraint_id == "T4_DRC_ERROR_COUNT" for r in results)

    def test_violations_fail(self, tmp_path: Path) -> None:
        """Checker fails for DRC violations."""
        report = DRCReport.from_dict(SAMPLE_DRC_REPORT_VIOLATIONS, exit_code=5)
        drc_result = DRCResult(
            report=report,
            returncode=5,
            stdout="",
            stderr="",
            report_path=tmp_path / "drc.json",
            board_path=tmp_path / "board.kicad_pcb",
        )

        checker = Tier4DrcChecker(drc_result=drc_result)
        results = checker.check(None, {})

        # Should have failures
        error_result = next(r for r in results if r.constraint_id == "T4_DRC_ERROR_COUNT")
        assert error_result.passed is False

        warning_result = next(r for r in results if r.constraint_id == "T4_DRC_WARNING_COUNT")
        assert warning_result.passed is False

    def test_allow_warnings(self, tmp_path: Path) -> None:
        """Checker allows warnings when configured."""
        # Report with only warnings
        report_data = {
            "source": "/path/to/board.kicad_pcb",
            "violations": [
                {"type": "width", "severity": "warning", "description": "Test warning"},
            ],
            "unconnected_items": [],
            "schematic_parity": [],
            "coordinate_units": "mm",
        }
        report = DRCReport.from_dict(report_data, exit_code=5)
        drc_result = DRCResult(
            report=report,
            returncode=5,
            stdout="",
            stderr="",
            report_path=tmp_path / "drc.json",
            board_path=tmp_path / "board.kicad_pcb",
        )

        checker = Tier4DrcChecker(drc_result=drc_result, allow_warnings=True)
        results = checker.check(None, {})

        # Error count should pass (0 errors)
        error_result = next(r for r in results if r.constraint_id == "T4_DRC_ERROR_COUNT")
        assert error_result.passed is True

        # Warning check should not be present when allow_warnings=True
        warning_ids = [r.constraint_id for r in results]
        assert "T4_DRC_WARNING_COUNT" not in warning_ids

    def test_unconnected_items_fail(self, tmp_path: Path) -> None:
        """Checker fails for unconnected items."""
        report = DRCReport.from_dict(SAMPLE_DRC_REPORT_UNCONNECTED, exit_code=0)
        drc_result = DRCResult(
            report=report,
            returncode=0,
            stdout="",
            stderr="",
            report_path=tmp_path / "drc.json",
            board_path=tmp_path / "board.kicad_pcb",
        )

        checker = Tier4DrcChecker(drc_result=drc_result)
        results = checker.check(None, {})

        unconnected_result = next(r for r in results if r.constraint_id == "T4_DRC_UNCONNECTED_COUNT")
        assert unconnected_result.passed is False

    def test_set_drc_result(self, tmp_path: Path) -> None:
        """Can set DRC result after initialization."""
        checker = Tier4DrcChecker()

        # Initially fails
        results1 = checker.check(None, {})
        assert any(not r.passed for r in results1)

        # Set result
        report = DRCReport.from_dict(SAMPLE_DRC_REPORT_CLEAN, exit_code=0)
        drc_result = DRCResult(
            report=report,
            returncode=0,
            stdout="",
            stderr="",
            report_path=tmp_path / "drc.json",
            board_path=tmp_path / "board.kicad_pcb",
        )
        checker.set_drc_result(drc_result)

        # Now passes
        results2 = checker.check(None, {})
        assert all(r.passed for r in results2)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestDRCModuleExports:
    """Tests for module exports."""

    def test_exports_from_constraints(self) -> None:
        """DRC types exported from constraints package."""
        from formula_foundry.coupongen.constraints import (
            DRCError,
            DRCExitCode,
            DRCReport,
            DRCResult,
            DRCViolation,
            Tier4DrcChecker,
            check_drc_gate,
            run_drc,
        )

        # Verify they're the right types
        assert DRCExitCode.PASS == 0
        assert DRCError.__name__ == "DRCError"
        assert callable(run_drc)
        assert callable(check_drc_gate)
