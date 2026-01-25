# SPDX-License-Identifier: MIT
"""Unit tests for bridge/verify_repair/repairs.py.

Tests cover:
- RepairAction dataclass
- get_applicable_repairs function
- apply_repair function with mocked subprocess
- Repair action selection based on failure categories
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.verify_repair.classify import FailureCategory
from bridge.verify_repair.repairs import (
    RepairAction,
    apply_repair,
    get_applicable_repairs,
    repair_isort,
    repair_ruff,
)

# =============================================================================
# RepairAction tests
# =============================================================================


class TestRepairAction:
    """Tests for RepairAction dataclass."""

    def test_minimal_creation(self) -> None:
        """Create RepairAction with minimal fields."""
        action = RepairAction(
            name="test",
            command=["test", "cmd"],
            success=True,
            output="OK",
        )
        assert action.name == "test"
        assert action.command == ["test", "cmd"]
        assert action.success is True
        assert action.output == "OK"
        assert action.files_modified == 0  # default

    def test_full_creation(self) -> None:
        """Create RepairAction with all fields."""
        action = RepairAction(
            name="ruff",
            command=["ruff", "--fix"],
            success=True,
            output="Fixed 3 files",
            files_modified=3,
        )
        assert action.files_modified == 3

    def test_none_command(self) -> None:
        """RepairAction can have None command."""
        action = RepairAction(
            name="bootstrap",
            command=None,
            success=True,
            output="Delegated",
        )
        assert action.command is None


# =============================================================================
# get_applicable_repairs tests
# =============================================================================


class TestGetApplicableRepairs:
    """Tests for get_applicable_repairs function."""

    def test_lint_ruff_triggers_ruff_autofix(self) -> None:
        """LINT_RUFF category triggers ruff_autofix repair."""
        result = get_applicable_repairs({FailureCategory.LINT_RUFF.value})
        assert "ruff_autofix" in result

    def test_lint_format_triggers_ruff_format(self) -> None:
        """LINT_FORMAT category triggers ruff_format repair."""
        result = get_applicable_repairs({FailureCategory.LINT_FORMAT.value})
        assert "ruff_format" in result

    def test_missing_dependency_triggers_bootstrap(self) -> None:
        """MISSING_DEPENDENCY category triggers bootstrap repair."""
        result = get_applicable_repairs({FailureCategory.MISSING_DEPENDENCY.value})
        assert "bootstrap" in result

    def test_multiple_categories(self) -> None:
        """Multiple categories return multiple repairs."""
        result = get_applicable_repairs(
            {
                FailureCategory.LINT_RUFF.value,
                FailureCategory.MISSING_DEPENDENCY.value,
            }
        )
        assert "ruff_autofix" in result
        assert "bootstrap" in result

    def test_empty_categories(self) -> None:
        """Empty categories return no repairs."""
        result = get_applicable_repairs(set())
        assert result == []

    def test_unknown_category_no_repair(self) -> None:
        """Unknown category doesn't trigger any repair."""
        result = get_applicable_repairs({FailureCategory.UNKNOWN.value})
        assert result == []

    def test_pytest_failure_no_layer1_repair(self) -> None:
        """Test failures don't have Layer 1 repairs."""
        result = get_applicable_repairs({FailureCategory.PYTEST_TEST_FAILURE.value})
        assert result == []


# =============================================================================
# repair_ruff tests
# =============================================================================


class TestRepairRuff:
    """Tests for repair_ruff function."""

    def test_ruff_success(self, tmp_path: Path) -> None:
        """repair_ruff returns success when ruff succeeds."""
        mock_result1 = mock.MagicMock()
        mock_result1.returncode = 0
        mock_result1.stdout = "Fixed 2 errors"
        mock_result1.stderr = ""

        mock_result2 = mock.MagicMock()
        mock_result2.returncode = 0
        mock_result2.stdout = "1 file reformatted"
        mock_result2.stderr = ""

        with mock.patch("subprocess.run", side_effect=[mock_result1, mock_result2]):
            result = repair_ruff(tmp_path, verbose=False)

        assert result.success is True
        assert result.name == "ruff_autofix"
        assert result.files_modified >= 0

    def test_ruff_not_found(self, tmp_path: Path) -> None:
        """repair_ruff handles ruff not in PATH."""
        with mock.patch("subprocess.run", side_effect=FileNotFoundError("ruff")):
            result = repair_ruff(tmp_path, verbose=False)

        assert result.success is False
        assert "not found" in result.output

    def test_ruff_timeout(self, tmp_path: Path) -> None:
        """repair_ruff handles timeout."""
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ruff", 120)):
            result = repair_ruff(tmp_path, verbose=False)

        assert result.success is False
        assert "timed out" in result.output.lower()

    def test_ruff_exception(self, tmp_path: Path) -> None:
        """repair_ruff handles other exceptions."""
        with mock.patch("subprocess.run", side_effect=OSError("Permission denied")):
            result = repair_ruff(tmp_path, verbose=False)

        assert result.success is False
        assert "Permission denied" in result.output


# =============================================================================
# repair_isort tests
# =============================================================================


class TestRepairIsort:
    """Tests for repair_isort function."""

    def test_isort_success(self, tmp_path: Path) -> None:
        """repair_isort returns success when isort succeeds."""
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Fixing imports"
        mock_result.stderr = ""

        with mock.patch("subprocess.run", return_value=mock_result):
            result = repair_isort(tmp_path, verbose=False)

        assert result.success is True
        assert result.name == "isort"

    def test_isort_not_found(self, tmp_path: Path) -> None:
        """repair_isort handles isort not in PATH."""
        with mock.patch("subprocess.run", side_effect=FileNotFoundError("isort")):
            result = repair_isort(tmp_path, verbose=False)

        assert result.success is False
        assert "not found" in result.output

    def test_isort_failure(self, tmp_path: Path) -> None:
        """repair_isort handles isort failure."""
        mock_result = mock.MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "Error"
        mock_result.stderr = ""

        with mock.patch("subprocess.run", return_value=mock_result):
            result = repair_isort(tmp_path, verbose=False)

        assert result.success is False


# =============================================================================
# apply_repair tests
# =============================================================================


class TestApplyRepair:
    """Tests for apply_repair function."""

    def test_apply_ruff_autofix(self, tmp_path: Path) -> None:
        """apply_repair handles ruff_autofix."""
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with mock.patch("subprocess.run", return_value=mock_result):
            result = apply_repair("ruff_autofix", tmp_path, verbose=False)

        assert result.name == "ruff_autofix"

    def test_apply_ruff_format(self, tmp_path: Path) -> None:
        """apply_repair handles ruff_format."""
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with mock.patch("subprocess.run", return_value=mock_result):
            result = apply_repair("ruff_format", tmp_path, verbose=False)

        assert result.name == "ruff_autofix"  # Uses same repair

    def test_apply_isort(self, tmp_path: Path) -> None:
        """apply_repair handles isort."""
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with mock.patch("subprocess.run", return_value=mock_result):
            result = apply_repair("isort", tmp_path, verbose=False)

        assert result.name == "isort"

    def test_apply_bootstrap(self, tmp_path: Path) -> None:
        """apply_repair handles bootstrap (delegated)."""
        result = apply_repair("bootstrap", tmp_path, verbose=False)

        assert result.name == "bootstrap"
        assert result.success is True
        assert "delegated" in result.output.lower()

    def test_apply_unknown_repair(self, tmp_path: Path) -> None:
        """apply_repair handles unknown repair name."""
        result = apply_repair("unknown_repair", tmp_path, verbose=False)

        assert result.success is False
        assert "Unknown repair" in result.output


# =============================================================================
# Integration tests
# =============================================================================


class TestRepairIntegration:
    """Integration tests for repair selection and application."""

    def test_lint_failure_flow(self, tmp_path: Path) -> None:
        """Test full flow from lint failure to repair selection."""
        # Simulate lint failure categories
        categories = {FailureCategory.LINT_RUFF.value}

        # Get applicable repairs
        repairs = get_applicable_repairs(categories)
        assert "ruff_autofix" in repairs

        # Apply repair (mocked)
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Fixed 1 error"
        mock_result.stderr = ""

        with mock.patch("subprocess.run", return_value=mock_result):
            for repair_name in repairs:
                result = apply_repair(repair_name, tmp_path, verbose=False)
                assert result.success is True

    def test_multiple_failures_multiple_repairs(self, tmp_path: Path) -> None:
        """Test multiple failures trigger multiple repairs."""
        categories = {
            FailureCategory.LINT_RUFF.value,
            FailureCategory.LINT_FORMAT.value,
            FailureCategory.MISSING_DEPENDENCY.value,
        }

        repairs = get_applicable_repairs(categories)

        # Should have ruff_autofix, ruff_format, and bootstrap
        assert "ruff_autofix" in repairs
        assert "ruff_format" in repairs
        assert "bootstrap" in repairs

    def test_no_repairs_for_test_failures(self, tmp_path: Path) -> None:
        """Test failures don't have automatic repairs."""
        categories = {
            FailureCategory.PYTEST_TEST_FAILURE.value,
            FailureCategory.TYPECHECK_MYPY.value,
        }

        repairs = get_applicable_repairs(categories)

        # No Layer 1 repairs for these
        assert repairs == []
