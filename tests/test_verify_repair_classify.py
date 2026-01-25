# SPDX-License-Identifier: MIT
"""Unit tests for bridge/verify_repair/classify.py and data.py.

Tests cover:
- VerifyGateResult dataclass construction and from_dict
- VerifySummary dataclass construction and from_json
- RepairAttemptRecord and RepairLoopReport serialization
- Failure classification for different gate types
- Import error and mypy error extraction
- Failure signature computation
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.verify_repair.classify import (
    FailureCategory,
    classify_failures,
    compute_failure_signature,
    extract_import_errors,
    extract_mypy_errors,
    get_all_categories,
)
from bridge.verify_repair.data import (
    RepairAttemptRecord,
    RepairLoopReport,
    VerifyGateResult,
    VerifySummary,
)

# =============================================================================
# VerifyGateResult tests
# =============================================================================


class TestVerifyGateResult:
    """Tests for VerifyGateResult dataclass."""

    def test_from_dict_minimal(self) -> None:
        """Create VerifyGateResult from minimal dict."""
        data = {"name": "pytest", "passed": True}
        result = VerifyGateResult.from_dict(data)
        assert result.name == "pytest"
        assert result.passed is True
        assert result.returncode is None
        assert result.stdout == ""
        assert result.stderr == ""

    def test_from_dict_full(self) -> None:
        """Create VerifyGateResult from full dict."""
        data = {
            "name": "ruff",
            "returncode": 1,
            "passed": False,
            "stdout": "Found 3 errors",
            "stderr": "Error output",
            "cmd": ["ruff", "check", "."],
            "note": "Auto-fixable",
        }
        result = VerifyGateResult.from_dict(data)
        assert result.name == "ruff"
        assert result.returncode == 1
        assert result.passed is False
        assert result.stdout == "Found 3 errors"
        assert result.stderr == "Error output"
        assert result.cmd == ["ruff", "check", "."]
        assert result.note == "Auto-fixable"

    def test_from_dict_empty(self) -> None:
        """Create VerifyGateResult from empty dict uses defaults."""
        result = VerifyGateResult.from_dict({})
        assert result.name == ""
        assert result.passed is False


# =============================================================================
# VerifySummary tests
# =============================================================================


class TestVerifySummary:
    """Tests for VerifySummary dataclass."""

    def test_from_json_passing(self) -> None:
        """Create VerifySummary from JSON with all passing gates."""
        data = {
            "ok": True,
            "failed_gates": [],
            "first_failed_gate": "",
            "results": [
                {"name": "pytest", "passed": True, "returncode": 0, "stdout": "", "stderr": ""},
                {"name": "ruff", "passed": True, "returncode": 0, "stdout": "", "stderr": ""},
            ],
        }
        summary = VerifySummary.from_json(data)
        assert summary.ok is True
        assert summary.failed_gates == []
        assert summary.first_failed_gate == ""
        assert "pytest" in summary.results_by_gate
        assert "ruff" in summary.results_by_gate

    def test_from_json_failing(self) -> None:
        """Create VerifySummary from JSON with failed gates."""
        data = {
            "ok": False,
            "failed_gates": ["pytest", "mypy"],
            "first_failed_gate": "pytest",
            "results": [
                {"name": "pytest", "passed": False, "returncode": 1, "stdout": "FAILED", "stderr": ""},
                {"name": "mypy", "passed": False, "returncode": 1, "stdout": "error:", "stderr": ""},
            ],
        }
        summary = VerifySummary.from_json(data)
        assert summary.ok is False
        assert summary.failed_gates == ["pytest", "mypy"]
        assert summary.first_failed_gate == "pytest"
        assert not summary.results_by_gate["pytest"].passed
        assert not summary.results_by_gate["mypy"].passed

    def test_from_json_empty(self) -> None:
        """Create VerifySummary from empty JSON."""
        summary = VerifySummary.from_json({})
        assert summary.ok is False
        assert summary.failed_gates == []


# =============================================================================
# RepairAttemptRecord tests
# =============================================================================


class TestRepairAttemptRecord:
    """Tests for RepairAttemptRecord dataclass."""

    def test_to_dict_full(self) -> None:
        """Convert full RepairAttemptRecord to dict."""
        verify_before = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={},
        )
        verify_after = VerifySummary(
            ok=True,
            failed_gates=[],
            first_failed_gate="",
            results_by_gate={},
        )
        record = RepairAttemptRecord(
            attempt_index=1,
            detected_categories=["lint_ruff"],
            actions_taken=["ruff --fix"],
            verify_before=verify_before,
            verify_after=verify_after,
            diff_applied=True,
            elapsed_s=2.5,
        )
        result = record.to_dict()
        assert result["attempt_index"] == 1
        assert result["detected_categories"] == ["lint_ruff"]
        assert result["diff_applied"] is True
        assert result["elapsed_s"] == 2.5
        assert result["verify_before_ok"] is False
        assert result["verify_before_failed"] == ["pytest"]
        assert result["verify_after_ok"] is True

    def test_to_dict_none_verify(self) -> None:
        """Convert RepairAttemptRecord with None verify to dict."""
        record = RepairAttemptRecord(
            attempt_index=0,
            detected_categories=[],
            actions_taken=[],
            verify_before=None,
            verify_after=None,
            diff_applied=False,
            elapsed_s=0.0,
        )
        result = record.to_dict()
        assert result["verify_before_ok"] is None
        assert result["verify_after_ok"] is None


# =============================================================================
# RepairLoopReport tests
# =============================================================================


class TestRepairLoopReport:
    """Tests for RepairLoopReport dataclass."""

    def test_to_dict_success(self) -> None:
        """Convert successful RepairLoopReport to dict."""
        report = RepairLoopReport(
            success=True,
            total_attempts=2,
            final_failed_gates=[],
            elapsed_s=10.5,
            stable_failure_signature_count=0,
            artifacts_written=["report.json"],
        )
        result = report.to_dict()
        assert result["success"] is True
        assert result["total_attempts"] == 2
        assert result["final_failed_gates"] == []
        assert result["artifacts_written"] == ["report.json"]

    def test_to_dict_with_attempts(self) -> None:
        """Convert RepairLoopReport with attempts to dict."""
        attempt = RepairAttemptRecord(
            attempt_index=1,
            detected_categories=["lint_ruff"],
            actions_taken=["fix"],
            verify_before=None,
            verify_after=None,
            diff_applied=True,
            elapsed_s=1.0,
        )
        report = RepairLoopReport(
            success=False,
            total_attempts=1,
            final_failed_gates=["pytest"],
            elapsed_s=5.0,
            stable_failure_signature_count=2,
            artifacts_written=[],
            attempts=[attempt],
            early_stop_reason="max_attempts",
        )
        result = report.to_dict()
        assert len(result["attempts"]) == 1
        assert result["early_stop_reason"] == "max_attempts"


# =============================================================================
# FailureCategory tests
# =============================================================================


class TestFailureCategory:
    """Tests for FailureCategory enum."""

    def test_enum_values(self) -> None:
        """FailureCategory has expected string values."""
        assert FailureCategory.LINT_RUFF.value == "lint_ruff"
        assert FailureCategory.TYPECHECK_MYPY.value == "typecheck_mypy"
        assert FailureCategory.PYTEST_TEST_FAILURE.value == "pytest_test_failure"

    def test_string_enum(self) -> None:
        """FailureCategory is a string enum."""
        cat = FailureCategory.LINT_RUFF
        assert isinstance(cat, str)
        assert cat == "lint_ruff"


# =============================================================================
# classify_failures tests
# =============================================================================


class TestClassifyFailures:
    """Tests for classify_failures function."""

    def test_classify_ruff_failure(self) -> None:
        """Ruff failure is classified as LINT_RUFF."""
        gate = VerifyGateResult(
            name="ruff",
            returncode=1,
            passed=False,
            stdout="Found 3 errors",
            stderr="",
        )
        summary = VerifySummary(
            ok=False,
            failed_gates=["ruff"],
            first_failed_gate="ruff",
            results_by_gate={"ruff": gate},
        )
        result = classify_failures(summary)
        assert "ruff" in result
        assert FailureCategory.LINT_RUFF in result["ruff"]

    def test_classify_mypy_failure(self) -> None:
        """Mypy failure is classified as TYPECHECK_MYPY."""
        gate = VerifyGateResult(
            name="mypy",
            returncode=1,
            passed=False,
            stdout="src/foo.py:10: error: Missing return type",
            stderr="",
        )
        summary = VerifySummary(
            ok=False,
            failed_gates=["mypy"],
            first_failed_gate="mypy",
            results_by_gate={"mypy": gate},
        )
        result = classify_failures(summary)
        assert "mypy" in result
        assert FailureCategory.TYPECHECK_MYPY in result["mypy"]

    def test_classify_pytest_test_failure(self) -> None:
        """Pytest test failure is classified correctly."""
        gate = VerifyGateResult(
            name="pytest",
            returncode=1,
            passed=False,
            stdout="FAILED tests/test_foo.py::test_bar - AssertionError\n2 passed, 1 failed",
            stderr="",
        )
        summary = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={"pytest": gate},
        )
        result = classify_failures(summary)
        assert "pytest" in result
        assert FailureCategory.PYTEST_TEST_FAILURE in result["pytest"]

    def test_classify_pytest_collection_error(self) -> None:
        """Pytest collection error is classified correctly."""
        gate = VerifyGateResult(
            name="pytest",
            returncode=1,
            passed=False,
            stdout="ERROR collecting tests/test_foo.py",
            stderr="ImportError: cannot import name 'Foo' from 'bar'",
        )
        summary = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={"pytest": gate},
        )
        result = classify_failures(summary)
        assert "pytest" in result
        assert FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR in result["pytest"]

    def test_classify_module_not_found(self) -> None:
        """ModuleNotFoundError is classified correctly."""
        gate = VerifyGateResult(
            name="pytest",
            returncode=1,
            passed=False,
            stdout="ERROR collecting tests/test_foo.py\nModuleNotFoundError: No module named 'numpy'",
            stderr="",
        )
        summary = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={"pytest": gate},
        )
        result = classify_failures(summary)
        assert "pytest" in result
        assert FailureCategory.MISSING_DEPENDENCY in result["pytest"]

    def test_classify_git_guard_failure(self) -> None:
        """Git guard failure is classified as GIT_DIRTY."""
        gate = VerifyGateResult(
            name="git_guard",
            returncode=1,
            passed=False,
            stdout="Uncommitted changes detected",
            stderr="",
        )
        summary = VerifySummary(
            ok=False,
            failed_gates=["git_guard"],
            first_failed_gate="git_guard",
            results_by_gate={"git_guard": gate},
        )
        result = classify_failures(summary)
        assert "git_guard" in result
        assert FailureCategory.GIT_DIRTY in result["git_guard"]

    def test_classify_timeout(self) -> None:
        """Timeout is classified as TOOLING_TIMEOUT."""
        gate = VerifyGateResult(
            name="pytest",
            returncode=None,
            passed=False,
            stdout="",
            stderr="",
            note="timeout after 60s",
        )
        summary = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={"pytest": gate},
        )
        result = classify_failures(summary)
        assert "pytest" in result
        assert FailureCategory.TOOLING_TIMEOUT in result["pytest"]

    def test_classify_unknown_gate(self) -> None:
        """Unknown gate is classified as UNKNOWN."""
        gate = VerifyGateResult(
            name="custom_check",
            returncode=1,
            passed=False,
            stdout="failed",
            stderr="",
        )
        summary = VerifySummary(
            ok=False,
            failed_gates=["custom_check"],
            first_failed_gate="custom_check",
            results_by_gate={"custom_check": gate},
        )
        result = classify_failures(summary)
        assert "custom_check" in result
        assert FailureCategory.UNKNOWN in result["custom_check"]


# =============================================================================
# get_all_categories tests
# =============================================================================


class TestGetAllCategories:
    """Tests for get_all_categories function."""

    def test_empty_classification(self) -> None:
        """Empty classification returns empty set."""
        result = get_all_categories({})
        assert result == set()

    def test_single_category(self) -> None:
        """Single category is returned."""
        result = get_all_categories({"ruff": [FailureCategory.LINT_RUFF]})
        assert result == {FailureCategory.LINT_RUFF}

    def test_multiple_categories(self) -> None:
        """Multiple categories from different gates are combined."""
        classification = {
            "ruff": [FailureCategory.LINT_RUFF],
            "pytest": [FailureCategory.PYTEST_TEST_FAILURE, FailureCategory.MISSING_DEPENDENCY],
        }
        result = get_all_categories(classification)
        assert FailureCategory.LINT_RUFF in result
        assert FailureCategory.PYTEST_TEST_FAILURE in result
        assert FailureCategory.MISSING_DEPENDENCY in result


# =============================================================================
# extract_import_errors tests
# =============================================================================


class TestExtractImportErrors:
    """Tests for extract_import_errors function."""

    def test_no_import_errors(self) -> None:
        """No import errors returns empty list."""
        gate = VerifyGateResult(
            name="pytest",
            returncode=0,
            passed=True,
            stdout="all tests passed",
            stderr="",
        )
        summary = VerifySummary(
            ok=True,
            failed_gates=[],
            first_failed_gate="",
            results_by_gate={"pytest": gate},
        )
        result = extract_import_errors(summary)
        assert result == []

    def test_import_error_detected(self) -> None:
        """ImportError is extracted correctly."""
        gate = VerifyGateResult(
            name="pytest",
            returncode=1,
            passed=False,
            stdout="",
            stderr="ImportError: cannot import name 'Foo' from 'bar'",
        )
        summary = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={"pytest": gate},
        )
        result = extract_import_errors(summary)
        assert len(result) == 1
        assert result[0]["type"] == "import_error"
        assert result[0]["name"] == "Foo"

    def test_module_not_found_external(self) -> None:
        """External ModuleNotFoundError is marked as bootstrap installable."""
        gate = VerifyGateResult(
            name="pytest",
            returncode=1,
            passed=False,
            stdout="ModuleNotFoundError: No module named 'numpy'",
            stderr="",
        )
        summary = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={"pytest": gate},
        )
        result = extract_import_errors(summary)
        assert len(result) == 1
        assert result[0]["type"] == "module_not_found"
        assert result[0]["module"] == "numpy"
        assert result[0]["is_bootstrap_installable"] is True

    def test_module_not_found_internal(self) -> None:
        """Internal ModuleNotFoundError is marked as internal."""
        gate = VerifyGateResult(
            name="pytest",
            returncode=1,
            passed=False,
            stdout="ModuleNotFoundError: No module named 'formula_foundry.missing'",
            stderr="",
        )
        summary = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={"pytest": gate},
        )
        result = extract_import_errors(summary)
        assert len(result) == 1
        assert result[0]["type"] == "module_not_found"
        assert result[0]["is_internal"] is True
        assert result[0]["is_bootstrap_installable"] is False


# =============================================================================
# extract_mypy_errors tests
# =============================================================================


class TestExtractMypyErrors:
    """Tests for extract_mypy_errors function."""

    def test_no_mypy_gate(self) -> None:
        """No mypy gate returns empty list."""
        summary = VerifySummary(
            ok=True,
            failed_gates=[],
            first_failed_gate="",
            results_by_gate={},
        )
        result = extract_mypy_errors(summary)
        assert result == []

    def test_mypy_errors_extracted(self) -> None:
        """Mypy errors are extracted correctly."""
        gate = VerifyGateResult(
            name="mypy",
            returncode=1,
            passed=False,
            stdout="src/foo.py:10: error: Missing return type\nsrc/bar.py:20: error: Incompatible types",
            stderr="",
        )
        summary = VerifySummary(
            ok=False,
            failed_gates=["mypy"],
            first_failed_gate="mypy",
            results_by_gate={"mypy": gate},
        )
        result = extract_mypy_errors(summary)
        assert len(result) == 2
        assert result[0]["file"] == "src/foo.py"
        assert result[0]["line"] == "10"
        assert "Missing return type" in result[0]["message"]


# =============================================================================
# compute_failure_signature tests
# =============================================================================


class TestComputeFailureSignature:
    """Tests for compute_failure_signature function."""

    def test_deterministic_signature(self) -> None:
        """Same failures produce same signature."""
        gate = VerifyGateResult(
            name="pytest",
            returncode=1,
            passed=False,
            stdout="FAILED test_foo.py::test_bar",
            stderr="AssertionError",
        )
        summary = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={"pytest": gate},
        )
        sig1 = compute_failure_signature(summary)
        sig2 = compute_failure_signature(summary)
        assert sig1 == sig2
        assert len(sig1) == 16  # 16 hex chars

    def test_different_failures_different_signature(self) -> None:
        """Different failures produce different signatures."""
        gate1 = VerifyGateResult(
            name="pytest",
            returncode=1,
            passed=False,
            stdout="FAILED test_foo.py",
            stderr="",
        )
        gate2 = VerifyGateResult(
            name="pytest",
            returncode=1,
            passed=False,
            stdout="FAILED test_bar.py",
            stderr="",
        )
        summary1 = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={"pytest": gate1},
        )
        summary2 = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={"pytest": gate2},
        )
        sig1 = compute_failure_signature(summary1)
        sig2 = compute_failure_signature(summary2)
        assert sig1 != sig2

    def test_signature_normalizes_paths(self) -> None:
        """Signature normalizes path prefixes."""
        gate1 = VerifyGateResult(
            name="pytest",
            returncode=1,
            passed=False,
            stdout="FAILED /home/user/project/test_foo.py",
            stderr="",
        )
        gate2 = VerifyGateResult(
            name="pytest",
            returncode=1,
            passed=False,
            stdout="FAILED /other/path/test_foo.py",
            stderr="",
        )
        summary1 = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={"pytest": gate1},
        )
        summary2 = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={"pytest": gate2},
        )
        sig1 = compute_failure_signature(summary1)
        sig2 = compute_failure_signature(summary2)
        # Should be the same since paths are normalized
        assert sig1 == sig2
