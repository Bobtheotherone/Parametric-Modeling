"""DRC runner wrapper with JSON report parsing.

REQ-M1-016: Create DRC wrapper that runs kicad-cli pcb drc
with --severity-all --exit-code-violations --format json.
Parse JSON report, handle exit codes (0=pass, 5=violations),
and integrate as Tier 4 constraint gate.

This module provides:
1. DRCReport dataclass for parsing KiCad DRC JSON output
2. DRCViolation dataclass for structured violation information
3. DRCRunner class for executing and parsing DRC
4. Tier4DrcChecker for constraint system integration

Exit codes:
- 0: DRC passed (no violations)
- 5: DRC found violations (--exit-code-violations mode)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from formula_foundry.coupongen.kicad.cli import KicadCliRunner


class DRCExitCode(IntEnum):
    """KiCad DRC exit codes.

    Reference: KiCad CLI documentation.
    """

    PASS = 0
    VIOLATIONS = 5


DRCSeverity = Literal["error", "warning", "exclusion"]


@dataclass(frozen=True, slots=True)
class DRCViolation:
    """A single DRC violation from the JSON report.

    Attributes:
        type: Violation type (e.g., "clearance", "unconnected", "width").
        severity: Severity level (error, warning, exclusion).
        description: Human-readable description.
        pos_x_mm: X coordinate in mm (if available).
        pos_y_mm: Y coordinate in mm (if available).
        items: Affected items (tracks, vias, pads, etc.).
    """

    type: str
    severity: DRCSeverity
    description: str
    pos_x_mm: float | None = None
    pos_y_mm: float | None = None
    items: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DRCViolation:
        """Create DRCViolation from KiCad JSON report entry."""
        # Position extraction
        pos = data.get("pos", {})
        pos_x = pos.get("x") if isinstance(pos, dict) else None
        pos_y = pos.get("y") if isinstance(pos, dict) else None

        # Extract items as strings
        items_raw = data.get("items", [])
        items = tuple(str(item) for item in items_raw) if items_raw else ()

        return cls(
            type=data.get("type", "unknown"),
            severity=data.get("severity", "error"),
            description=data.get("description", ""),
            pos_x_mm=float(pos_x) if pos_x is not None else None,
            pos_y_mm=float(pos_y) if pos_y is not None else None,
            items=items,
        )

    @property
    def is_error(self) -> bool:
        """Return True if this is an error-severity violation."""
        return self.severity == "error"

    @property
    def is_warning(self) -> bool:
        """Return True if this is a warning-severity violation."""
        return self.severity == "warning"


@dataclass(frozen=True)
class DRCReport:
    """Parsed KiCad DRC JSON report.

    Attributes:
        source: Path to the board file that was checked.
        violations: List of DRC violations found.
        unconnected_items: List of unconnected item entries.
        schematic_parity: List of schematic parity issues.
        coordinate_units: Units used for coordinates (typically "mm").
        exit_code: Exit code from kicad-cli (0=pass, 5=violations).
    """

    source: str
    violations: tuple[DRCViolation, ...]
    unconnected_items: tuple[dict[str, Any], ...]
    schematic_parity: tuple[dict[str, Any], ...]
    coordinate_units: str
    exit_code: int

    @classmethod
    def from_json_file(cls, report_path: Path, exit_code: int = 0) -> DRCReport:
        """Parse DRC report from JSON file.

        Args:
            report_path: Path to the DRC JSON report.
            exit_code: Exit code from the kicad-cli command.

        Returns:
            Parsed DRCReport instance.

        Raises:
            FileNotFoundError: If report file doesn't exist.
            json.JSONDecodeError: If report is not valid JSON.
        """
        content = report_path.read_text(encoding="utf-8")
        return cls.from_json_string(content, exit_code)

    @classmethod
    def from_json_string(cls, json_str: str, exit_code: int = 0) -> DRCReport:
        """Parse DRC report from JSON string.

        Args:
            json_str: JSON string content.
            exit_code: Exit code from the kicad-cli command.

        Returns:
            Parsed DRCReport instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data, exit_code)

    @classmethod
    def from_dict(cls, data: dict[str, Any], exit_code: int = 0) -> DRCReport:
        """Parse DRC report from dictionary.

        Args:
            data: Parsed JSON data.
            exit_code: Exit code from the kicad-cli command.

        Returns:
            Parsed DRCReport instance.
        """
        violations_raw = data.get("violations", [])
        violations = tuple(DRCViolation.from_dict(v) for v in violations_raw)

        unconnected = tuple(data.get("unconnected_items", []))
        schematic = tuple(data.get("schematic_parity", []))

        return cls(
            source=data.get("source", ""),
            violations=violations,
            unconnected_items=unconnected,
            schematic_parity=schematic,
            coordinate_units=data.get("coordinate_units", "mm"),
            exit_code=exit_code,
        )

    @property
    def passed(self) -> bool:
        """Return True if DRC passed (no violations and exit code 0)."""
        return self.exit_code == DRCExitCode.PASS and len(self.violations) == 0

    @property
    def error_count(self) -> int:
        """Count of error-severity violations."""
        return sum(1 for v in self.violations if v.is_error)

    @property
    def warning_count(self) -> int:
        """Count of warning-severity violations."""
        return sum(1 for v in self.violations if v.is_warning)

    @property
    def total_violations(self) -> int:
        """Total number of violations."""
        return len(self.violations)

    def get_errors(self) -> list[DRCViolation]:
        """Return all error-severity violations."""
        return [v for v in self.violations if v.is_error]

    def get_warnings(self) -> list[DRCViolation]:
        """Return all warning-severity violations."""
        return [v for v in self.violations if v.is_warning]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "source": self.source,
            "exit_code": self.exit_code,
            "passed": self.passed,
            "violation_count": self.total_violations,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "violations": [
                {
                    "type": v.type,
                    "severity": v.severity,
                    "description": v.description,
                    "pos_x_mm": v.pos_x_mm,
                    "pos_y_mm": v.pos_y_mm,
                    "items": list(v.items),
                }
                for v in self.violations
            ],
            "unconnected_items": list(self.unconnected_items),
            "schematic_parity": list(self.schematic_parity),
        }


@dataclass
class DRCResult:
    """Result of DRC execution.

    Attributes:
        report: Parsed DRC report.
        returncode: Exit code from kicad-cli.
        stdout: Standard output from the command.
        stderr: Standard error from the command.
        report_path: Path to the JSON report file.
        board_path: Path to the board file that was checked.
    """

    report: DRCReport
    returncode: int
    stdout: str
    stderr: str
    report_path: Path
    board_path: Path

    @property
    def passed(self) -> bool:
        """Return True if DRC passed."""
        return self.report.passed

    @property
    def has_violations(self) -> bool:
        """Return True if DRC found violations."""
        return self.returncode == DRCExitCode.VIOLATIONS


class DRCError(Exception):
    """Raised when DRC execution fails."""

    def __init__(
        self,
        message: str,
        result: DRCResult | None = None,
    ) -> None:
        super().__init__(message)
        self.result = result


def run_drc(
    runner: KicadCliRunner,
    board_path: Path,
    report_path: Path,
    *,
    timeout: float | None = None,
    variables: Mapping[str, str] | None = None,
) -> DRCResult:
    """Run DRC on a board file and parse the JSON report.

    This is the main entry point for DRC execution with JSON parsing.
    Wraps KicadCliRunner.run_drc() and parses the resulting JSON report.

    Args:
        runner: KiCad CLI runner (local or docker mode).
        board_path: Path to the .kicad_pcb file.
        report_path: Path where the JSON report will be written.
        timeout: Optional timeout in seconds.
        variables: Optional text variables for board substitution.

    Returns:
        DRCResult containing the parsed report and execution details.

    Raises:
        DRCError: If DRC execution fails unexpectedly (not violations).
        FileNotFoundError: If board file doesn't exist.

    Example:
        >>> from formula_foundry.coupongen.kicad.cli import KicadCliRunner
        >>> runner = KicadCliRunner(mode="local")
        >>> result = run_drc(runner, Path("board.kicad_pcb"), Path("drc.json"))
        >>> if not result.passed:
        ...     print(f"DRC found {result.report.total_violations} violations")
    """
    if not board_path.exists():
        raise FileNotFoundError(f"Board file not found: {board_path}")

    # Ensure report directory exists
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Run DRC via the runner
    proc_result = runner.run_drc(
        board_path,
        report_path,
        timeout=timeout,
        variables=variables,
    )

    # Parse the report
    try:
        report = DRCReport.from_json_file(report_path, exit_code=proc_result.returncode)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise DRCError(
            f"Failed to parse DRC report: {e}",
            result=None,
        ) from e

    return DRCResult(
        report=report,
        returncode=proc_result.returncode,
        stdout=proc_result.stdout,
        stderr=proc_result.stderr,
        report_path=report_path,
        board_path=board_path,
    )


def check_drc_gate(
    result: DRCResult,
    *,
    must_pass: bool = True,
    allow_warnings: bool = False,
) -> bool:
    """Check if DRC result passes the constraint gate.

    Args:
        result: DRC execution result.
        must_pass: If True, DRC must pass to return True.
        allow_warnings: If True, allow warning-level violations.

    Returns:
        True if the DRC passes the gate criteria.

    Raises:
        DRCError: If must_pass is True and DRC failed.
    """
    if must_pass:
        if not result.passed:
            if allow_warnings and result.report.error_count == 0:
                # Only warnings, allowed
                return True

            violations = result.report.violations
            violation_summary = ", ".join(f"{v.type}({v.severity})" for v in violations[:5])
            if len(violations) > 5:
                violation_summary += f" ... and {len(violations) - 5} more"

            raise DRCError(
                f"KiCad DRC failed with {result.report.total_violations} violations: {violation_summary}",
                result=result,
            )

    return result.passed or (allow_warnings and result.report.error_count == 0)


# ---------------------------------------------------------------------------
# Tier 4 Constraint Integration
# ---------------------------------------------------------------------------

from formula_foundry.coupongen.constraints.tiers import (
    ConstraintResult,
    ConstraintTier,
    TierChecker,
    _bool_constraint,
    _max_constraint,
)


class Tier4DrcChecker(TierChecker):
    """Tier 4: KiCad DRC external validation (CP-4.1).

    This checker validates boards against KiCad's DRC engine.
    It runs kicad-cli pcb drc with --severity-all --exit-code-violations
    and parses the JSON report.

    Unlike Tiers 0-3 which use analytic constraint checks, Tier 4
    delegates to the external KiCad toolchain for authoritative validation.

    Attributes:
        drc_result: The DRC execution result to check.
        allow_warnings: If True, warnings don't fail the constraint.
    """

    def __init__(
        self,
        drc_result: DRCResult | None = None,
        *,
        allow_warnings: bool = False,
    ) -> None:
        """Initialize Tier 4 checker.

        Args:
            drc_result: Pre-computed DRC result. If None, check() will
                return empty results (DRC not yet run).
            allow_warnings: If True, warning-level violations are allowed.
        """
        self._drc_result = drc_result
        self._allow_warnings = allow_warnings

    @property
    def tier(self) -> ConstraintTier:
        return "T4"

    def set_drc_result(self, result: DRCResult) -> None:
        """Set the DRC result for checking.

        This is useful when the DRC is run separately and the result
        needs to be injected into the constraint system.
        """
        self._drc_result = result

    def check(
        self,
        spec: Any,
        fab_limits: dict[str, int],
        resolved: Any | None = None,
    ) -> list[ConstraintResult]:
        """Check Tier 4 DRC constraints.

        Args:
            spec: CouponSpec (not used - DRC result is pre-computed).
            fab_limits: Fab limits (not used for DRC).
            resolved: Resolved design (not used for DRC).

        Returns:
            List of constraint results from DRC validation.
        """
        results: list[ConstraintResult] = []

        if self._drc_result is None:
            # DRC not yet run - return a placeholder failure
            results.append(
                _bool_constraint(
                    "T4_DRC_EXECUTED",
                    "KiCad DRC must be executed",
                    tier="T4",
                    condition=False,
                    reason="DRC has not been run",
                )
            )
            return results

        report = self._drc_result.report

        # Check DRC execution succeeded (exit code 0 or 5)
        valid_exit = self._drc_result.returncode in (
            DRCExitCode.PASS,
            DRCExitCode.VIOLATIONS,
        )
        results.append(
            _bool_constraint(
                "T4_DRC_EXECUTED",
                "KiCad DRC must execute successfully",
                tier="T4",
                condition=valid_exit,
                reason=f"Unexpected exit code: {self._drc_result.returncode}",
            )
        )

        if not valid_exit:
            # Don't check further if DRC didn't run properly
            return results

        # Check for error-level violations
        results.append(
            _max_constraint(
                "T4_DRC_ERROR_COUNT",
                "KiCad DRC must find zero error-level violations",
                tier="T4",
                value=report.error_count,
                limit=0,
                reason=self._format_violations(report.get_errors(), "error"),
            )
        )

        # Check for warning-level violations (if not allowed)
        if not self._allow_warnings:
            results.append(
                _max_constraint(
                    "T4_DRC_WARNING_COUNT",
                    "KiCad DRC must find zero warning-level violations",
                    tier="T4",
                    value=report.warning_count,
                    limit=0,
                    reason=self._format_violations(report.get_warnings(), "warning"),
                )
            )

        # Check unconnected items
        results.append(
            _max_constraint(
                "T4_DRC_UNCONNECTED_COUNT",
                "KiCad DRC must find zero unconnected items",
                tier="T4",
                value=len(report.unconnected_items),
                limit=0,
                reason=f"{len(report.unconnected_items)} unconnected items found",
            )
        )

        return results

    def _format_violations(
        self,
        violations: list[DRCViolation],
        level: str,
    ) -> str:
        """Format violation list for error message."""
        if not violations:
            return ""

        summaries = [f"{v.type}: {v.description[:50]}" for v in violations[:3]]
        result = f"{len(violations)} {level}(s): " + "; ".join(summaries)

        if len(violations) > 3:
            result += f" ... and {len(violations) - 3} more"

        return result


# Re-export ConstraintTier with T4 support
ConstraintTierWithT4 = Literal["T0", "T1", "T2", "T3", "T4"]


__all__ = [
    # Exit codes
    "DRCExitCode",
    # Data types
    "DRCViolation",
    "DRCReport",
    "DRCResult",
    "DRCSeverity",
    # Exception
    "DRCError",
    # Functions
    "run_drc",
    "check_drc_gate",
    # Tier 4 checker
    "Tier4DrcChecker",
    "ConstraintTierWithT4",
]
