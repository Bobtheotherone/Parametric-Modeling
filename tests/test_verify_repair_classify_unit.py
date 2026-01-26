# SPDX-License-Identifier: MIT
"""Unit tests for bridge/verify_repair/classify.py.

Tests the failure classification logic for verify repair operations.
Key functions and classes tested:
- FailureCategory: Enum of failure categories
- classify_failures: Classify verify failures into categories
- get_all_categories: Get all unique categories from classification
- compute_failure_signature: Compute signature for failure state
- extract_import_errors: Extract import error details
- extract_mypy_errors: Extract mypy error details
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.verify_repair.classify import (
    BOOTSTRAP_INSTALLABLE,
    INTERNAL_MODULE_PREFIXES,
    FailureCategory,
    classify_failures,
    compute_failure_signature,
    extract_import_errors,
    extract_mypy_errors,
    get_all_categories,
)
from bridge.verify_repair.data import VerifyGateResult, VerifySummary

# -----------------------------------------------------------------------------
# FailureCategory Enum tests
# -----------------------------------------------------------------------------


class TestFailureCategoryEnum:
    """Tests for FailureCategory enum."""

    def test_lint_categories_exist(self) -> None:
        """Lint-related categories exist."""
        assert FailureCategory.LINT_RUFF == "lint_ruff"
        assert FailureCategory.LINT_FORMAT == "lint_format"

    def test_typecheck_categories_exist(self) -> None:
        """Type checking categories exist."""
        assert FailureCategory.TYPECHECK_MYPY == "typecheck_mypy"

    def test_pytest_categories_exist(self) -> None:
        """Pytest-related categories exist."""
        assert FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR == "pytest_collection_import_error"
        assert FailureCategory.PYTEST_TEST_FAILURE == "pytest_test_failure"

    def test_dependency_categories_exist(self) -> None:
        """Dependency-related categories exist."""
        assert FailureCategory.MISSING_DEPENDENCY == "missing_dependency"
        assert FailureCategory.MISSING_MODULE_INTERNAL == "missing_module_internal"

    def test_tooling_categories_exist(self) -> None:
        """Tooling-related categories exist."""
        assert FailureCategory.TOOLING_ERROR == "tooling_error"
        assert FailureCategory.TOOLING_TIMEOUT == "tooling_timeout"

    def test_git_category_exists(self) -> None:
        """Git-related category exists."""
        assert FailureCategory.GIT_DIRTY == "git_dirty"

    def test_unknown_category_exists(self) -> None:
        """Unknown category exists as fallback."""
        assert FailureCategory.UNKNOWN == "unknown"

    def test_category_is_string_enum(self) -> None:
        """FailureCategory values are strings."""
        assert isinstance(FailureCategory.LINT_RUFF.value, str)
        assert isinstance(FailureCategory.UNKNOWN.value, str)


# -----------------------------------------------------------------------------
# Constants tests
# -----------------------------------------------------------------------------


class TestClassifyConstants:
    """Tests for module constants."""

    def test_bootstrap_installable_contains_common_packages(self) -> None:
        """BOOTSTRAP_INSTALLABLE contains common packages."""
        assert "numpy" in BOOTSTRAP_INSTALLABLE
        assert "pytest" in BOOTSTRAP_INSTALLABLE
        assert "ruff" in BOOTSTRAP_INSTALLABLE
        assert "mypy" in BOOTSTRAP_INSTALLABLE

    def test_internal_module_prefixes_contains_expected(self) -> None:
        """INTERNAL_MODULE_PREFIXES contains expected prefixes."""
        assert "formula_foundry" in INTERNAL_MODULE_PREFIXES
        assert "bridge" in INTERNAL_MODULE_PREFIXES
        assert "tests" in INTERNAL_MODULE_PREFIXES


# -----------------------------------------------------------------------------
# classify_failures tests
# -----------------------------------------------------------------------------


class TestClassifyFailures:
    """Tests for classify_failures function."""

    def _make_summary(
        self,
        failed_gates: list[str],
        gate_results: dict[str, dict],
    ) -> VerifySummary:
        """Helper to create a VerifySummary."""
        results_by_gate = {}
        for name, data in gate_results.items():
            results_by_gate[name] = VerifyGateResult(
                name=name,
                returncode=data.get("returncode", 1),
                passed=data.get("passed", False),
                stdout=data.get("stdout", ""),
                stderr=data.get("stderr", ""),
                note=data.get("note", ""),
            )
        return VerifySummary(
            ok=len(failed_gates) == 0,
            failed_gates=failed_gates,
            first_failed_gate=failed_gates[0] if failed_gates else "",
            results_by_gate=results_by_gate,
        )

    def test_empty_failures_returns_empty(self) -> None:
        """No failed gates returns empty dict."""
        summary = self._make_summary([], {})
        result = classify_failures(summary)
        assert result == {}

    def test_ruff_failure_classified_as_lint(self) -> None:
        """Ruff failure is classified as LINT_RUFF."""
        summary = self._make_summary(
            ["ruff"],
            {"ruff": {"returncode": 1, "stdout": "Found 5 errors", "stderr": ""}},
        )
        result = classify_failures(summary)
        assert "ruff" in result
        assert FailureCategory.LINT_RUFF in result["ruff"]

    def test_mypy_failure_classified_as_typecheck(self) -> None:
        """Mypy failure is classified as TYPECHECK_MYPY."""
        summary = self._make_summary(
            ["mypy"],
            {"mypy": {"returncode": 1, "stdout": "test.py:1: error: Something wrong", "stderr": ""}},
        )
        result = classify_failures(summary)
        assert "mypy" in result
        assert FailureCategory.TYPECHECK_MYPY in result["mypy"]

    def test_pytest_test_failure_classified(self) -> None:
        """Pytest test failure is classified as PYTEST_TEST_FAILURE."""
        summary = self._make_summary(
            ["pytest"],
            {
                "pytest": {
                    "returncode": 1,
                    "stdout": "FAILED tests/test_foo.py::test_bar - AssertionError\n3 passed, 1 failed",
                    "stderr": "",
                }
            },
        )
        result = classify_failures(summary)
        assert "pytest" in result
        assert FailureCategory.PYTEST_TEST_FAILURE in result["pytest"]

    def test_pytest_collection_error_classified(self) -> None:
        """Pytest collection error is classified as PYTEST_COLLECTION_IMPORT_ERROR."""
        summary = self._make_summary(
            ["pytest"],
            {
                "pytest": {
                    "returncode": 1,
                    "stdout": "ERROR collecting tests/test_foo.py\nImportError: cannot import name 'bar'",
                    "stderr": "",
                }
            },
        )
        result = classify_failures(summary)
        assert "pytest" in result
        assert FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR in result["pytest"]

    def test_pytest_module_not_found_internal_classified(self) -> None:
        """Pytest module not found (internal) is classified correctly."""
        summary = self._make_summary(
            ["pytest"],
            {
                "pytest": {
                    "returncode": 1,
                    "stdout": "ERROR collecting tests/test_foo.py\nModuleNotFoundError: No module named 'formula_foundry.missing'",
                    "stderr": "",
                }
            },
        )
        result = classify_failures(summary)
        assert "pytest" in result
        assert FailureCategory.MISSING_MODULE_INTERNAL in result["pytest"]

    def test_pytest_module_not_found_external_classified(self) -> None:
        """Pytest module not found (external) is classified as MISSING_DEPENDENCY."""
        summary = self._make_summary(
            ["pytest"],
            {
                "pytest": {
                    "returncode": 1,
                    "stdout": "ERROR collecting tests/test_foo.py\nModuleNotFoundError: No module named 'numpy'",
                    "stderr": "",
                }
            },
        )
        result = classify_failures(summary)
        assert "pytest" in result
        assert FailureCategory.MISSING_DEPENDENCY in result["pytest"]

    def test_git_guard_failure_classified(self) -> None:
        """Git guard failure is classified as GIT_DIRTY."""
        summary = self._make_summary(
            ["git_guard"],
            {"git_guard": {"returncode": 1, "stdout": "Working directory is dirty", "stderr": ""}},
        )
        result = classify_failures(summary)
        assert "git_guard" in result
        assert FailureCategory.GIT_DIRTY in result["git_guard"]

    def test_timeout_failure_classified(self) -> None:
        """Timeout failure is classified as TOOLING_TIMEOUT."""
        summary = self._make_summary(
            ["pytest"],
            {"pytest": {"returncode": 1, "stdout": "", "stderr": "", "note": "timeout exceeded"}},
        )
        result = classify_failures(summary)
        assert "pytest" in result
        assert FailureCategory.TOOLING_TIMEOUT in result["pytest"]

    def test_unknown_gate_classified_as_unknown(self) -> None:
        """Unknown gate is classified as UNKNOWN."""
        summary = self._make_summary(
            ["custom_gate"],
            {"custom_gate": {"returncode": 1, "stdout": "Some failure", "stderr": ""}},
        )
        result = classify_failures(summary)
        assert "custom_gate" in result
        assert FailureCategory.UNKNOWN in result["custom_gate"]

    def test_missing_gate_result_classified_as_unknown(self) -> None:
        """Gate in failed_gates but missing from results is classified as UNKNOWN."""
        summary = self._make_summary(["missing_gate"], {})
        result = classify_failures(summary)
        assert "missing_gate" in result
        assert FailureCategory.UNKNOWN in result["missing_gate"]


# -----------------------------------------------------------------------------
# get_all_categories tests
# -----------------------------------------------------------------------------


class TestGetAllCategories:
    """Tests for get_all_categories function."""

    def test_empty_classification_returns_empty_set(self) -> None:
        """Empty classification returns empty set."""
        result = get_all_categories({})
        assert result == set()

    def test_single_gate_single_category(self) -> None:
        """Single gate with single category."""
        classification = {"ruff": [FailureCategory.LINT_RUFF]}
        result = get_all_categories(classification)
        assert result == {FailureCategory.LINT_RUFF}

    def test_multiple_gates_multiple_categories(self) -> None:
        """Multiple gates with different categories."""
        classification = {
            "ruff": [FailureCategory.LINT_RUFF],
            "mypy": [FailureCategory.TYPECHECK_MYPY],
            "pytest": [FailureCategory.PYTEST_TEST_FAILURE],
        }
        result = get_all_categories(classification)
        assert result == {
            FailureCategory.LINT_RUFF,
            FailureCategory.TYPECHECK_MYPY,
            FailureCategory.PYTEST_TEST_FAILURE,
        }

    def test_duplicate_categories_deduplicated(self) -> None:
        """Duplicate categories are deduplicated."""
        classification = {
            "pytest": [FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR, FailureCategory.MISSING_DEPENDENCY],
            "mypy": [FailureCategory.MISSING_DEPENDENCY],
        }
        result = get_all_categories(classification)
        assert FailureCategory.MISSING_DEPENDENCY in result
        assert len(result) == 2

    def test_gate_with_multiple_categories(self) -> None:
        """Gate with multiple categories all included."""
        classification = {
            "pytest": [
                FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR,
                FailureCategory.MISSING_DEPENDENCY,
            ],
        }
        result = get_all_categories(classification)
        assert FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR in result
        assert FailureCategory.MISSING_DEPENDENCY in result


# -----------------------------------------------------------------------------
# compute_failure_signature tests
# -----------------------------------------------------------------------------


class TestComputeFailureSignature:
    """Tests for compute_failure_signature function."""

    def _make_summary(
        self,
        failed_gates: list[str],
        gate_results: dict[str, dict],
    ) -> VerifySummary:
        """Helper to create a VerifySummary."""
        results_by_gate = {}
        for name, data in gate_results.items():
            results_by_gate[name] = VerifyGateResult(
                name=name,
                returncode=data.get("returncode", 1),
                passed=data.get("passed", False),
                stdout=data.get("stdout", ""),
                stderr=data.get("stderr", ""),
            )
        return VerifySummary(
            ok=len(failed_gates) == 0,
            failed_gates=failed_gates,
            first_failed_gate=failed_gates[0] if failed_gates else "",
            results_by_gate=results_by_gate,
        )

    def test_signature_is_hex_string(self) -> None:
        """Signature is a hexadecimal string."""
        summary = self._make_summary(
            ["ruff"],
            {"ruff": {"stdout": "error", "stderr": ""}},
        )
        sig = compute_failure_signature(summary)
        assert isinstance(sig, str)
        assert all(c in "0123456789abcdef" for c in sig)

    def test_signature_is_16_chars(self) -> None:
        """Signature is 16 characters (truncated SHA256)."""
        summary = self._make_summary(
            ["ruff"],
            {"ruff": {"stdout": "error", "stderr": ""}},
        )
        sig = compute_failure_signature(summary)
        assert len(sig) == 16

    def test_same_input_same_signature(self) -> None:
        """Same failure produces same signature."""
        summary1 = self._make_summary(
            ["ruff"],
            {"ruff": {"stdout": "Found 5 errors", "stderr": ""}},
        )
        summary2 = self._make_summary(
            ["ruff"],
            {"ruff": {"stdout": "Found 5 errors", "stderr": ""}},
        )
        assert compute_failure_signature(summary1) == compute_failure_signature(summary2)

    def test_different_output_different_signature(self) -> None:
        """Different output produces different signature."""
        summary1 = self._make_summary(
            ["ruff"],
            {"ruff": {"stdout": "Error A", "stderr": ""}},
        )
        summary2 = self._make_summary(
            ["ruff"],
            {"ruff": {"stdout": "Error B", "stderr": ""}},
        )
        assert compute_failure_signature(summary1) != compute_failure_signature(summary2)

    def test_different_gates_different_signature(self) -> None:
        """Different failed gates produce different signature."""
        summary1 = self._make_summary(
            ["ruff"],
            {"ruff": {"stdout": "Error", "stderr": ""}},
        )
        summary2 = self._make_summary(
            ["mypy"],
            {"mypy": {"stdout": "Error", "stderr": ""}},
        )
        assert compute_failure_signature(summary1) != compute_failure_signature(summary2)

    def test_empty_failures_consistent_signature(self) -> None:
        """Empty failures produce consistent signature."""
        summary1 = self._make_summary([], {})
        summary2 = self._make_summary([], {})
        assert compute_failure_signature(summary1) == compute_failure_signature(summary2)

    def test_signature_normalizes_timestamps(self) -> None:
        """Signature normalizes timestamps."""
        summary1 = self._make_summary(
            ["pytest"],
            {"pytest": {"stdout": "2024-01-15T10:30:00 error", "stderr": ""}},
        )
        summary2 = self._make_summary(
            ["pytest"],
            {"pytest": {"stdout": "2024-01-16T11:45:30 error", "stderr": ""}},
        )
        # Should be same after timestamp normalization
        assert compute_failure_signature(summary1) == compute_failure_signature(summary2)


# -----------------------------------------------------------------------------
# extract_import_errors tests
# -----------------------------------------------------------------------------


class TestExtractImportErrors:
    """Tests for extract_import_errors function."""

    def _make_summary(
        self,
        gate_results: dict[str, dict],
    ) -> VerifySummary:
        """Helper to create a VerifySummary."""
        results_by_gate = {}
        failed_gates = []
        for name, data in gate_results.items():
            passed = data.get("passed", False)
            if not passed:
                failed_gates.append(name)
            results_by_gate[name] = VerifyGateResult(
                name=name,
                returncode=data.get("returncode", 0 if passed else 1),
                passed=passed,
                stdout=data.get("stdout", ""),
                stderr=data.get("stderr", ""),
            )
        return VerifySummary(
            ok=len(failed_gates) == 0,
            failed_gates=failed_gates,
            first_failed_gate=failed_gates[0] if failed_gates else "",
            results_by_gate=results_by_gate,
        )

    def test_no_errors_returns_empty(self) -> None:
        """No import errors returns empty list."""
        summary = self._make_summary({"ruff": {"stdout": "No errors", "stderr": ""}})
        errors = extract_import_errors(summary)
        assert errors == []

    def test_extracts_import_error(self) -> None:
        """ImportError is extracted."""
        summary = self._make_summary(
            {
                "pytest": {
                    "stdout": "ImportError: cannot import name 'foo' from 'bar'",
                    "stderr": "",
                },
            }
        )
        errors = extract_import_errors(summary)
        assert len(errors) == 1
        assert errors[0]["type"] == "import_error"
        assert errors[0]["name"] == "foo"

    def test_extracts_module_not_found(self) -> None:
        """ModuleNotFoundError is extracted."""
        summary = self._make_summary(
            {
                "pytest": {
                    "stdout": "ModuleNotFoundError: No module named 'missing_module'",
                    "stderr": "",
                },
            }
        )
        errors = extract_import_errors(summary)
        assert len(errors) == 1
        assert errors[0]["type"] == "module_not_found"
        assert errors[0]["module"] == "missing_module"
        assert errors[0]["root_module"] == "missing_module"

    def test_marks_internal_module(self) -> None:
        """Internal module is marked as internal."""
        summary = self._make_summary(
            {
                "pytest": {
                    "stdout": "ModuleNotFoundError: No module named 'formula_foundry.missing'",
                    "stderr": "",
                },
            }
        )
        errors = extract_import_errors(summary)
        assert len(errors) == 1
        assert errors[0]["is_internal"] is True
        assert errors[0]["root_module"] == "formula_foundry"

    def test_marks_bootstrap_installable(self) -> None:
        """Bootstrap-installable module is marked."""
        summary = self._make_summary(
            {
                "pytest": {
                    "stdout": "ModuleNotFoundError: No module named 'numpy'",
                    "stderr": "",
                },
            }
        )
        errors = extract_import_errors(summary)
        assert len(errors) == 1
        assert errors[0]["is_bootstrap_installable"] is True

    def test_deduplicates_errors(self) -> None:
        """Duplicate errors are deduplicated."""
        summary = self._make_summary(
            {
                "pytest": {
                    "stdout": "ModuleNotFoundError: No module named 'foo'\nModuleNotFoundError: No module named 'foo'",
                    "stderr": "",
                },
            }
        )
        errors = extract_import_errors(summary)
        # Should only have one entry for 'foo'
        foo_errors = [e for e in errors if e.get("module") == "foo"]
        assert len(foo_errors) == 1

    def test_extracts_from_stderr(self) -> None:
        """Errors are extracted from stderr too."""
        summary = self._make_summary(
            {
                "pytest": {
                    "stdout": "",
                    "stderr": "ImportError: cannot import name 'bar' from 'baz'",
                },
            }
        )
        errors = extract_import_errors(summary)
        assert len(errors) == 1
        assert errors[0]["name"] == "bar"


# -----------------------------------------------------------------------------
# extract_mypy_errors tests
# -----------------------------------------------------------------------------


class TestExtractMypyErrors:
    """Tests for extract_mypy_errors function."""

    def _make_summary(
        self,
        gate_results: dict[str, dict],
    ) -> VerifySummary:
        """Helper to create a VerifySummary."""
        results_by_gate = {}
        failed_gates = []
        for name, data in gate_results.items():
            passed = data.get("passed", False)
            if not passed:
                failed_gates.append(name)
            results_by_gate[name] = VerifyGateResult(
                name=name,
                returncode=data.get("returncode", 0 if passed else 1),
                passed=passed,
                stdout=data.get("stdout", ""),
                stderr=data.get("stderr", ""),
            )
        return VerifySummary(
            ok=len(failed_gates) == 0,
            failed_gates=failed_gates,
            first_failed_gate=failed_gates[0] if failed_gates else "",
            results_by_gate=results_by_gate,
        )

    def test_no_mypy_gate_returns_empty(self) -> None:
        """No mypy gate returns empty list."""
        summary = self._make_summary({"ruff": {"stdout": "No errors", "stderr": ""}})
        errors = extract_mypy_errors(summary)
        assert errors == []

    def test_extracts_single_mypy_error(self) -> None:
        """Single mypy error is extracted."""
        summary = self._make_summary(
            {
                "mypy": {
                    "stdout": "src/foo.py:42: error: Incompatible types",
                    "stderr": "",
                },
            }
        )
        errors = extract_mypy_errors(summary)
        assert len(errors) == 1
        assert errors[0]["file"] == "src/foo.py"
        assert errors[0]["line"] == "42"
        assert errors[0]["message"] == "Incompatible types"

    def test_extracts_multiple_mypy_errors(self) -> None:
        """Multiple mypy errors are extracted."""
        summary = self._make_summary(
            {
                "mypy": {
                    "stdout": "src/foo.py:10: error: Missing return\nsrc/bar.py:20: error: Type error",
                    "stderr": "",
                },
            }
        )
        errors = extract_mypy_errors(summary)
        assert len(errors) == 2
        assert errors[0]["file"] == "src/foo.py"
        assert errors[1]["file"] == "src/bar.py"

    def test_extracts_from_stderr(self) -> None:
        """Mypy errors are extracted from stderr too."""
        summary = self._make_summary(
            {
                "mypy": {
                    "stdout": "",
                    "stderr": "test.py:5: error: Something wrong",
                },
            }
        )
        errors = extract_mypy_errors(summary)
        assert len(errors) == 1
        assert errors[0]["file"] == "test.py"
