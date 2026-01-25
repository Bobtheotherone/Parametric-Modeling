# SPDX-License-Identifier: MIT
"""Unit tests for bridge/verify_repair/data.py.

Tests the data structures for verify repair operations.
Key classes tested:
- VerifyGateResult: Result from a single verify gate
- VerifySummary: Summary of a verify run
- RepairAttemptRecord: Record of a single repair attempt
- RepairLoopReport: Final report of the repair loop
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.verify_repair.data import (
    RepairAttemptRecord,
    RepairLoopReport,
    VerifyGateResult,
    VerifySummary,
)

# -----------------------------------------------------------------------------
# VerifyGateResult tests
# -----------------------------------------------------------------------------


class TestVerifyGateResult:
    """Tests for VerifyGateResult dataclass."""

    def test_basic_construction(self) -> None:
        """Basic construction with required fields."""
        result = VerifyGateResult(
            name="pytest",
            returncode=0,
            passed=True,
            stdout="5 passed",
            stderr="",
        )
        assert result.name == "pytest"
        assert result.returncode == 0
        assert result.passed is True
        assert result.stdout == "5 passed"
        assert result.stderr == ""

    def test_default_values(self) -> None:
        """Default values for optional fields."""
        result = VerifyGateResult(
            name="ruff",
            returncode=1,
            passed=False,
            stdout="errors",
            stderr="",
        )
        assert result.cmd is None
        assert result.note == ""

    def test_with_optional_fields(self) -> None:
        """Construction with optional fields."""
        result = VerifyGateResult(
            name="pytest",
            returncode=1,
            passed=False,
            stdout="",
            stderr="error",
            cmd=["pytest", "-v"],
            note="timeout exceeded",
        )
        assert result.cmd == ["pytest", "-v"]
        assert result.note == "timeout exceeded"

    def test_none_returncode(self) -> None:
        """Returncode can be None (e.g., process not started)."""
        result = VerifyGateResult(
            name="mypy",
            returncode=None,
            passed=False,
            stdout="",
            stderr="",
        )
        assert result.returncode is None

    def test_from_dict_basic(self) -> None:
        """Create from dict with basic fields."""
        data = {
            "name": "pytest",
            "returncode": 0,
            "passed": True,
            "stdout": "ok",
            "stderr": "",
        }
        result = VerifyGateResult.from_dict(data)
        assert result.name == "pytest"
        assert result.returncode == 0
        assert result.passed is True
        assert result.stdout == "ok"

    def test_from_dict_with_optional(self) -> None:
        """Create from dict with optional fields."""
        data = {
            "name": "ruff",
            "returncode": 1,
            "passed": False,
            "stdout": "errors",
            "stderr": "warning",
            "cmd": ["ruff", "check"],
            "note": "auto-fixable",
        }
        result = VerifyGateResult.from_dict(data)
        assert result.cmd == ["ruff", "check"]
        assert result.note == "auto-fixable"

    def test_from_dict_missing_optional(self) -> None:
        """Create from dict with missing optional fields."""
        data = {
            "name": "mypy",
            "returncode": 0,
            "passed": True,
        }
        result = VerifyGateResult.from_dict(data)
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.cmd is None
        assert result.note == ""

    def test_from_dict_empty_dict(self) -> None:
        """Create from empty dict uses defaults."""
        result = VerifyGateResult.from_dict({})
        assert result.name == ""
        assert result.returncode is None
        assert result.passed is False


# -----------------------------------------------------------------------------
# VerifySummary tests
# -----------------------------------------------------------------------------


class TestVerifySummary:
    """Tests for VerifySummary dataclass."""

    def test_basic_construction(self) -> None:
        """Basic construction with required fields."""
        summary = VerifySummary(
            ok=True,
            failed_gates=[],
            first_failed_gate="",
            results_by_gate={},
        )
        assert summary.ok is True
        assert summary.failed_gates == []
        assert summary.first_failed_gate == ""
        assert summary.results_by_gate == {}

    def test_with_failures(self) -> None:
        """Construction with failures."""
        ruff_result = VerifyGateResult(
            name="ruff",
            returncode=1,
            passed=False,
            stdout="errors",
            stderr="",
        )
        summary = VerifySummary(
            ok=False,
            failed_gates=["ruff"],
            first_failed_gate="ruff",
            results_by_gate={"ruff": ruff_result},
        )
        assert summary.ok is False
        assert summary.failed_gates == ["ruff"]
        assert summary.first_failed_gate == "ruff"
        assert "ruff" in summary.results_by_gate

    def test_multiple_failed_gates(self) -> None:
        """Construction with multiple failed gates."""
        summary = VerifySummary(
            ok=False,
            failed_gates=["ruff", "mypy", "pytest"],
            first_failed_gate="ruff",
            results_by_gate={},
        )
        assert len(summary.failed_gates) == 3
        assert summary.first_failed_gate == "ruff"

    def test_from_json_basic(self) -> None:
        """Create from JSON with basic structure."""
        data = {
            "ok": True,
            "failed_gates": [],
            "first_failed_gate": "",
            "results": [],
        }
        summary = VerifySummary.from_json(data)
        assert summary.ok is True
        assert summary.failed_gates == []

    def test_from_json_with_results(self) -> None:
        """Create from JSON with gate results."""
        data = {
            "ok": False,
            "failed_gates": ["pytest"],
            "first_failed_gate": "pytest",
            "results": [
                {
                    "name": "pytest",
                    "returncode": 1,
                    "passed": False,
                    "stdout": "FAILED",
                    "stderr": "",
                },
                {
                    "name": "ruff",
                    "returncode": 0,
                    "passed": True,
                    "stdout": "",
                    "stderr": "",
                },
            ],
        }
        summary = VerifySummary.from_json(data)
        assert summary.ok is False
        assert "pytest" in summary.results_by_gate
        assert "ruff" in summary.results_by_gate
        assert summary.results_by_gate["pytest"].passed is False
        assert summary.results_by_gate["ruff"].passed is True

    def test_from_json_empty(self) -> None:
        """Create from empty JSON uses defaults."""
        summary = VerifySummary.from_json({})
        assert summary.ok is False
        assert summary.failed_gates == []
        assert summary.first_failed_gate == ""
        assert summary.results_by_gate == {}


# -----------------------------------------------------------------------------
# RepairAttemptRecord tests
# -----------------------------------------------------------------------------


class TestRepairAttemptRecord:
    """Tests for RepairAttemptRecord dataclass."""

    def test_basic_construction(self) -> None:
        """Basic construction with required fields."""
        record = RepairAttemptRecord(
            attempt_index=0,
            detected_categories=["lint_ruff"],
            actions_taken=["ruff --fix"],
            verify_before=None,
            verify_after=None,
            diff_applied=True,
            elapsed_s=5.2,
        )
        assert record.attempt_index == 0
        assert record.detected_categories == ["lint_ruff"]
        assert record.actions_taken == ["ruff --fix"]
        assert record.diff_applied is True
        assert record.elapsed_s == 5.2

    def test_with_verify_summaries(self) -> None:
        """Construction with verify summaries."""
        before = VerifySummary(
            ok=False,
            failed_gates=["ruff"],
            first_failed_gate="ruff",
            results_by_gate={},
        )
        after = VerifySummary(
            ok=True,
            failed_gates=[],
            first_failed_gate="",
            results_by_gate={},
        )
        record = RepairAttemptRecord(
            attempt_index=1,
            detected_categories=["lint_ruff"],
            actions_taken=["ruff --fix"],
            verify_before=before,
            verify_after=after,
            diff_applied=True,
            elapsed_s=3.0,
        )
        assert record.verify_before is not None
        assert record.verify_before.ok is False
        assert record.verify_after is not None
        assert record.verify_after.ok is True

    def test_to_dict_basic(self) -> None:
        """Convert to dict with basic values."""
        record = RepairAttemptRecord(
            attempt_index=2,
            detected_categories=["typecheck_mypy"],
            actions_taken=["fix type error"],
            verify_before=None,
            verify_after=None,
            diff_applied=False,
            elapsed_s=1.5,
        )
        d = record.to_dict()
        assert d["attempt_index"] == 2
        assert d["detected_categories"] == ["typecheck_mypy"]
        assert d["actions_taken"] == ["fix type error"]
        assert d["diff_applied"] is False
        assert d["elapsed_s"] == 1.5
        assert d["verify_before_ok"] is None
        assert d["verify_before_failed"] == []
        assert d["verify_after_ok"] is None
        assert d["verify_after_failed"] == []

    def test_to_dict_with_summaries(self) -> None:
        """Convert to dict with verify summaries."""
        before = VerifySummary(
            ok=False,
            failed_gates=["ruff", "mypy"],
            first_failed_gate="ruff",
            results_by_gate={},
        )
        after = VerifySummary(
            ok=False,
            failed_gates=["mypy"],
            first_failed_gate="mypy",
            results_by_gate={},
        )
        record = RepairAttemptRecord(
            attempt_index=1,
            detected_categories=["lint_ruff"],
            actions_taken=["ruff --fix"],
            verify_before=before,
            verify_after=after,
            diff_applied=True,
            elapsed_s=2.0,
        )
        d = record.to_dict()
        assert d["verify_before_ok"] is False
        assert d["verify_before_failed"] == ["ruff", "mypy"]
        assert d["verify_after_ok"] is False
        assert d["verify_after_failed"] == ["mypy"]


# -----------------------------------------------------------------------------
# RepairLoopReport tests
# -----------------------------------------------------------------------------


class TestRepairLoopReport:
    """Tests for RepairLoopReport dataclass."""

    def test_basic_construction(self) -> None:
        """Basic construction with required fields."""
        report = RepairLoopReport(
            success=True,
            total_attempts=3,
            final_failed_gates=[],
            elapsed_s=15.0,
            stable_failure_signature_count=0,
            artifacts_written=["repair_report.json"],
        )
        assert report.success is True
        assert report.total_attempts == 3
        assert report.final_failed_gates == []
        assert report.elapsed_s == 15.0
        assert report.stable_failure_signature_count == 0
        assert report.artifacts_written == ["repair_report.json"]
        assert report.attempts == []
        assert report.early_stop_reason == ""

    def test_with_failures(self) -> None:
        """Construction with failures."""
        report = RepairLoopReport(
            success=False,
            total_attempts=5,
            final_failed_gates=["mypy"],
            elapsed_s=60.0,
            stable_failure_signature_count=3,
            artifacts_written=[],
            early_stop_reason="stable failure detected",
        )
        assert report.success is False
        assert report.final_failed_gates == ["mypy"]
        assert report.early_stop_reason == "stable failure detected"

    def test_with_attempts(self) -> None:
        """Construction with attempt records."""
        attempt1 = RepairAttemptRecord(
            attempt_index=0,
            detected_categories=["lint_ruff"],
            actions_taken=["ruff --fix"],
            verify_before=None,
            verify_after=None,
            diff_applied=True,
            elapsed_s=2.0,
        )
        attempt2 = RepairAttemptRecord(
            attempt_index=1,
            detected_categories=["typecheck_mypy"],
            actions_taken=["fix types"],
            verify_before=None,
            verify_after=None,
            diff_applied=True,
            elapsed_s=3.0,
        )
        report = RepairLoopReport(
            success=True,
            total_attempts=2,
            final_failed_gates=[],
            elapsed_s=5.0,
            stable_failure_signature_count=0,
            artifacts_written=[],
            attempts=[attempt1, attempt2],
        )
        assert len(report.attempts) == 2
        assert report.attempts[0].attempt_index == 0
        assert report.attempts[1].attempt_index == 1

    def test_to_dict_basic(self) -> None:
        """Convert to dict with basic values."""
        report = RepairLoopReport(
            success=True,
            total_attempts=1,
            final_failed_gates=[],
            elapsed_s=5.0,
            stable_failure_signature_count=0,
            artifacts_written=["report.json"],
        )
        d = report.to_dict()
        assert d["success"] is True
        assert d["total_attempts"] == 1
        assert d["final_failed_gates"] == []
        assert d["elapsed_s"] == 5.0
        assert d["stable_failure_signature_count"] == 0
        assert d["artifacts_written"] == ["report.json"]
        assert d["early_stop_reason"] == ""
        assert d["attempts"] == []

    def test_to_dict_with_attempts(self) -> None:
        """Convert to dict with attempt records."""
        attempt = RepairAttemptRecord(
            attempt_index=0,
            detected_categories=["lint_ruff"],
            actions_taken=["fix"],
            verify_before=None,
            verify_after=None,
            diff_applied=True,
            elapsed_s=1.0,
        )
        report = RepairLoopReport(
            success=True,
            total_attempts=1,
            final_failed_gates=[],
            elapsed_s=1.0,
            stable_failure_signature_count=0,
            artifacts_written=[],
            attempts=[attempt],
        )
        d = report.to_dict()
        assert len(d["attempts"]) == 1
        assert d["attempts"][0]["attempt_index"] == 0
        assert d["attempts"][0]["detected_categories"] == ["lint_ruff"]

    def test_to_dict_preserves_all_fields(self) -> None:
        """to_dict preserves all fields."""
        report = RepairLoopReport(
            success=False,
            total_attempts=10,
            final_failed_gates=["pytest", "mypy"],
            elapsed_s=120.5,
            stable_failure_signature_count=5,
            artifacts_written=["a.json", "b.json"],
            early_stop_reason="max attempts reached",
        )
        d = report.to_dict()
        assert d["success"] is False
        assert d["total_attempts"] == 10
        assert d["final_failed_gates"] == ["pytest", "mypy"]
        assert d["elapsed_s"] == 120.5
        assert d["stable_failure_signature_count"] == 5
        assert d["artifacts_written"] == ["a.json", "b.json"]
        assert d["early_stop_reason"] == "max attempts reached"


# -----------------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------------


class TestDataIntegration:
    """Integration tests for data structures."""

    def test_full_verify_workflow(self) -> None:
        """Test a full verify workflow with all data structures."""
        # Create gate results
        ruff_result = VerifyGateResult(
            name="ruff",
            returncode=1,
            passed=False,
            stdout="Found 5 errors",
            stderr="",
            cmd=["ruff", "check"],
        )
        pytest_result = VerifyGateResult(
            name="pytest",
            returncode=0,
            passed=True,
            stdout="10 passed",
            stderr="",
            cmd=["pytest", "-v"],
        )

        # Create before summary
        before_summary = VerifySummary(
            ok=False,
            failed_gates=["ruff"],
            first_failed_gate="ruff",
            results_by_gate={
                "ruff": ruff_result,
                "pytest": pytest_result,
            },
        )

        # Create after summary (ruff fixed)
        ruff_fixed = VerifyGateResult(
            name="ruff",
            returncode=0,
            passed=True,
            stdout="",
            stderr="",
        )
        after_summary = VerifySummary(
            ok=True,
            failed_gates=[],
            first_failed_gate="",
            results_by_gate={
                "ruff": ruff_fixed,
                "pytest": pytest_result,
            },
        )

        # Create repair attempt
        attempt = RepairAttemptRecord(
            attempt_index=0,
            detected_categories=["lint_ruff"],
            actions_taken=["ruff --fix ."],
            verify_before=before_summary,
            verify_after=after_summary,
            diff_applied=True,
            elapsed_s=3.5,
        )

        # Create final report
        report = RepairLoopReport(
            success=True,
            total_attempts=1,
            final_failed_gates=[],
            elapsed_s=3.5,
            stable_failure_signature_count=0,
            artifacts_written=["repair_report.json"],
            attempts=[attempt],
        )

        # Verify everything works together
        assert report.success is True
        assert len(report.attempts) == 1
        assert report.attempts[0].verify_before is not None
        assert report.attempts[0].verify_before.failed_gates == ["ruff"]
        assert report.attempts[0].verify_after is not None
        assert report.attempts[0].verify_after.ok is True

        # Verify serialization
        d = report.to_dict()
        assert d["success"] is True
        assert len(d["attempts"]) == 1
        assert d["attempts"][0]["verify_before_failed"] == ["ruff"]
        assert d["attempts"][0]["verify_after_ok"] is True
