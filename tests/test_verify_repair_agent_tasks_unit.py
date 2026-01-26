# SPDX-License-Identifier: MIT
"""Unit tests for bridge/verify_repair/agent_tasks.py.

Tests the agent-driven repair task generation module (Layer 2).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.verify_repair.agent_tasks import (
    RepairTask,
    build_repair_task_prompt,
    generate_repair_tasks,
)
from bridge.verify_repair.classify import FailureCategory
from bridge.verify_repair.data import VerifyGateResult, VerifySummary

# -----------------------------------------------------------------------------
# RepairTask dataclass tests
# -----------------------------------------------------------------------------


class TestRepairTask:
    """Tests for RepairTask dataclass."""

    def test_minimal_creation(self) -> None:
        """Create RepairTask with minimal fields."""
        task = RepairTask(
            id="TEST-001",
            title="Test task",
            description="A test task description",
        )
        assert task.id == "TEST-001"
        assert task.title == "Test task"
        assert task.description == "A test task description"
        assert task.agent == "claude"  # default
        assert task.intensity == "medium"  # default
        assert task.failing_gates == []
        assert task.target_files == []
        assert task.reproduction_commands == []
        assert task.do_not_touch == []
        assert task.constraints == []

    def test_full_creation(self) -> None:
        """Create RepairTask with all fields."""
        task = RepairTask(
            id="REPAIR-IMPORT-ERRORS",
            title="Fix import errors",
            description="Fix the import errors",
            agent="codex",
            intensity="high",
            failing_gates=["pytest", "spec_lint"],
            target_files=["src/foo.py", "src/bar.py"],
            reproduction_commands=["pytest --collect-only"],
            do_not_touch=["tests/**", "bridge/**"],
            constraints=["Fix only import errors", "Do not add features"],
        )
        assert task.agent == "codex"
        assert task.intensity == "high"
        assert len(task.failing_gates) == 2
        assert len(task.target_files) == 2
        assert len(task.constraints) == 2

    def test_to_dict(self) -> None:
        """RepairTask.to_dict() returns correct dict."""
        task = RepairTask(
            id="TEST-001",
            title="Test task",
            description="A test task",
            agent="claude",
            intensity="low",
        )
        result = task.to_dict()
        assert result == {
            "id": "TEST-001",
            "title": "Test task",
            "description": "A test task",
            "agent": "claude",
            "intensity": "low",
        }

    def test_to_dict_excludes_extra_fields(self) -> None:
        """RepairTask.to_dict() only includes core scheduling fields."""
        task = RepairTask(
            id="TEST-001",
            title="Test task",
            description="A test task",
            failing_gates=["pytest"],
            target_files=["src/foo.py"],
            reproduction_commands=["pytest"],
            do_not_touch=["tests/**"],
            constraints=["Fix only bugs"],
        )
        result = task.to_dict()
        assert "failing_gates" not in result
        assert "target_files" not in result
        assert "reproduction_commands" not in result
        assert "do_not_touch" not in result
        assert "constraints" not in result


# -----------------------------------------------------------------------------
# build_repair_task_prompt tests
# -----------------------------------------------------------------------------


class TestBuildRepairTaskPrompt:
    """Tests for build_repair_task_prompt function."""

    def test_basic_prompt_structure(self) -> None:
        """build_repair_task_prompt returns structured prompt."""
        task = RepairTask(
            id="TEST-001",
            title="Fix import errors",
            description="Fix the import errors in the codebase.",
        )
        prompt = build_repair_task_prompt(task)

        assert "# AUTOMATED REPAIR TASK" in prompt
        assert "Task ID: TEST-001" in prompt
        assert "Title: Fix import errors" in prompt
        assert "Fix the import errors in the codebase." in prompt
        assert "## Expected Output" in prompt

    def test_prompt_includes_constraints(self) -> None:
        """build_repair_task_prompt includes constraints."""
        task = RepairTask(
            id="TEST-001",
            title="Fix errors",
            description="Fix errors",
            constraints=["Do not add features", "Keep changes minimal"],
        )
        prompt = build_repair_task_prompt(task)

        assert "## Constraints" in prompt
        assert "Do not add features" in prompt
        assert "Keep changes minimal" in prompt

    def test_prompt_includes_do_not_touch(self) -> None:
        """build_repair_task_prompt includes do_not_touch patterns."""
        task = RepairTask(
            id="TEST-001",
            title="Fix errors",
            description="Fix errors",
            do_not_touch=["tests/**", "bridge/**", "DESIGN_DOCUMENT.md"],
        )
        prompt = build_repair_task_prompt(task)

        assert "## DO NOT MODIFY" in prompt
        assert "`tests/**`" in prompt
        assert "`bridge/**`" in prompt
        assert "`DESIGN_DOCUMENT.md`" in prompt

    def test_prompt_without_do_not_touch(self) -> None:
        """build_repair_task_prompt omits DO NOT MODIFY when no patterns."""
        task = RepairTask(
            id="TEST-001",
            title="Fix errors",
            description="Fix errors",
            do_not_touch=[],
        )
        prompt = build_repair_task_prompt(task)

        assert "## DO NOT MODIFY" not in prompt

    def test_prompt_expected_output_section(self) -> None:
        """build_repair_task_prompt includes expected output section."""
        task = RepairTask(
            id="TEST-001",
            title="Fix errors",
            description="Fix errors",
        )
        prompt = build_repair_task_prompt(task)

        assert "brief analysis of the root cause" in prompt
        assert "specific files you will modify" in prompt
        assert "Verification that the fix works" in prompt


# -----------------------------------------------------------------------------
# generate_repair_tasks tests
# -----------------------------------------------------------------------------


def _make_summary(
    *,
    pytest_stdout: str = "",
    pytest_stderr: str = "",
    pytest_rc: int = 0,
    mypy_stdout: str = "",
    mypy_stderr: str = "",
    mypy_rc: int = 0,
) -> VerifySummary:
    """Helper to create a VerifySummary for testing."""
    gates: dict[str, VerifyGateResult] = {}
    failed: list[str] = []

    if pytest_rc != 0 or pytest_stderr or pytest_stdout:
        gates["pytest"] = VerifyGateResult(
            name="pytest",
            passed=(pytest_rc == 0),
            stdout=pytest_stdout,
            stderr=pytest_stderr,
            returncode=pytest_rc,
        )
        if pytest_rc != 0:
            failed.append("pytest")

    if mypy_rc != 0 or mypy_stderr or mypy_stdout:
        gates["mypy"] = VerifyGateResult(
            name="mypy",
            passed=(mypy_rc == 0),
            stdout=mypy_stdout,
            stderr=mypy_stderr,
            returncode=mypy_rc,
        )
        if mypy_rc != 0:
            failed.append("mypy")

    return VerifySummary(
        ok=(len(failed) == 0),
        failed_gates=failed,
        first_failed_gate=failed[0] if failed else "",
        results_by_gate=gates,
    )


class TestGenerateRepairTasks:
    """Tests for generate_repair_tasks function."""

    def test_no_failures_returns_empty(self) -> None:
        """generate_repair_tasks returns empty list when no failures."""
        summary = _make_summary()
        classification: dict[str, list[FailureCategory]] = {}

        tasks = generate_repair_tasks(summary, classification)
        assert tasks == []

    def test_import_error_generates_task(self) -> None:
        """generate_repair_tasks generates task for import errors."""
        summary = _make_summary(
            pytest_stdout="",
            pytest_stderr=(
                "ERROR collecting tests/test_foo.py\nImportError: cannot import name 'SomeThing' from 'module.name'\n"
            ),
            pytest_rc=1,
        )
        classification = {
            "pytest": [FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR],
        }

        tasks = generate_repair_tasks(summary, classification)

        assert len(tasks) == 1
        task = tasks[0]
        assert task.id == "REPAIR-IMPORT-ERRORS"
        assert "Import" in task.title or "import" in task.title
        assert "pytest" in task.failing_gates

    def test_mypy_error_generates_task(self) -> None:
        """generate_repair_tasks generates task for mypy errors."""
        summary = _make_summary(
            mypy_stdout="src/foo.py:10: error: Argument 1 has incompatible type\n",
            mypy_stderr="",
            mypy_rc=1,
        )
        classification = {
            "mypy": [FailureCategory.TYPECHECK_MYPY],
        }

        tasks = generate_repair_tasks(summary, classification)

        assert len(tasks) == 1
        task = tasks[0]
        assert task.id == "REPAIR-MYPY-ERRORS"
        assert "type" in task.title.lower() or "mypy" in task.title.lower()
        assert "mypy" in task.failing_gates

    def test_test_failure_generates_task(self) -> None:
        """generate_repair_tasks generates task for test failures."""
        summary = _make_summary(
            pytest_stdout=("FAILED tests/test_foo.py::test_bar - AssertionError\n1 failed, 5 passed\n"),
            pytest_stderr="",
            pytest_rc=1,
        )
        classification = {
            "pytest": [FailureCategory.PYTEST_TEST_FAILURE],
        }

        tasks = generate_repair_tasks(summary, classification)

        assert len(tasks) == 1
        task = tasks[0]
        assert task.id == "REPAIR-TEST-FAILURES"
        assert "test" in task.title.lower()
        assert "pytest" in task.failing_gates

    def test_multiple_failures_generates_multiple_tasks(self) -> None:
        """generate_repair_tasks can generate multiple tasks."""
        summary = _make_summary(
            pytest_stdout=("FAILED tests/test_foo.py::test_bar - AssertionError\n1 failed, 5 passed\n"),
            pytest_stderr="",
            pytest_rc=1,
            mypy_stdout="src/foo.py:10: error: Argument 1 has incompatible type\n",
            mypy_stderr="",
            mypy_rc=1,
        )
        classification = {
            "pytest": [FailureCategory.PYTEST_TEST_FAILURE],
            "mypy": [FailureCategory.TYPECHECK_MYPY],
        }

        tasks = generate_repair_tasks(summary, classification)

        assert len(tasks) == 2
        task_ids = {t.id for t in tasks}
        assert "REPAIR-MYPY-ERRORS" in task_ids
        assert "REPAIR-TEST-FAILURES" in task_ids

    def test_import_error_prioritized_over_test_failure(self) -> None:
        """Import errors are generated before test failures."""
        summary = _make_summary(
            pytest_stdout="",
            pytest_stderr=(
                "ERROR collecting tests/test_foo.py\n"
                "ImportError: cannot import name 'SomeThing' from 'module'\n"
                "FAILED tests/test_bar.py::test_baz\n"
                "1 failed, 0 passed\n"
            ),
            pytest_rc=1,
        )
        classification = {
            "pytest": [
                FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR,
                FailureCategory.PYTEST_TEST_FAILURE,
            ],
        }

        tasks = generate_repair_tasks(summary, classification)

        # Import error task should come first
        if len(tasks) >= 2:
            assert tasks[0].id == "REPAIR-IMPORT-ERRORS"

    def test_module_not_found_internal_generates_task(self) -> None:
        """Internal module not found errors generate tasks."""
        summary = _make_summary(
            pytest_stdout="",
            pytest_stderr=("ERROR collecting tests/test_foo.py\nModuleNotFoundError: No module named 'bridge.new_module'\n"),
            pytest_rc=1,
        )
        classification = {
            "pytest": [FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR],
        }

        tasks = generate_repair_tasks(summary, classification)

        assert len(tasks) == 1
        task = tasks[0]
        assert task.id == "REPAIR-IMPORT-ERRORS"


# -----------------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------------


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow_import_error(self) -> None:
        """Test full workflow for import error repair."""
        # Create summary with import error
        summary = _make_summary(
            pytest_stderr=(
                "ERROR collecting tests/test_foo.py\nImportError: cannot import name 'MissingClass' from 'formula_foundry.core'\n"
            ),
            pytest_rc=1,
        )
        classification = {
            "pytest": [FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR],
        }

        # Generate tasks
        tasks = generate_repair_tasks(summary, classification)
        assert len(tasks) == 1

        # Build prompt
        prompt = build_repair_task_prompt(tasks[0])
        assert "AUTOMATED REPAIR TASK" in prompt
        assert "REPAIR-IMPORT-ERRORS" in prompt

        # Verify prompt has actionable content
        assert "Reproduction" in prompt or "reproduction" in prompt.lower()

    def test_full_workflow_mypy_error(self) -> None:
        """Test full workflow for mypy error repair."""
        summary = _make_summary(
            mypy_stdout=(
                "src/formula_foundry/core.py:25: error: "
                'Incompatible return value type (got "str", expected "int")\n'
                "Found 1 error in 1 file\n"
            ),
            mypy_rc=1,
        )
        classification = {
            "mypy": [FailureCategory.TYPECHECK_MYPY],
        }

        tasks = generate_repair_tasks(summary, classification)
        assert len(tasks) == 1

        task = tasks[0]
        assert "src/formula_foundry/core.py" in task.target_files

        prompt = build_repair_task_prompt(task)
        assert "type" in prompt.lower()
