# SPDX-License-Identifier: MIT
"""Unit tests for bridge/verify_repair/executor.py.

Tests the repair task executor for the verify repair loop.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.verify_repair.agent_tasks import RepairTask
from bridge.verify_repair.executor import (
    RepairExecutionResult,
    RepairExecutor,
    create_repair_callback,
)

# -----------------------------------------------------------------------------
# RepairExecutionResult tests
# -----------------------------------------------------------------------------


class TestRepairExecutionResult:
    """Tests for RepairExecutionResult dataclass."""

    def test_successful_result(self) -> None:
        """Create successful RepairExecutionResult."""
        result = RepairExecutionResult(
            success=True,
            tasks_executed=2,
            tasks_succeeded=2,
            tasks_failed=0,
            artifacts_written=["repair_plan.json"],
            errors=[],
        )
        assert result.success is True
        assert result.tasks_executed == 2
        assert result.tasks_succeeded == 2
        assert result.tasks_failed == 0

    def test_failed_result(self) -> None:
        """Create failed RepairExecutionResult."""
        result = RepairExecutionResult(
            success=False,
            tasks_executed=2,
            tasks_succeeded=1,
            tasks_failed=1,
            errors=["ruff not found in PATH"],
        )
        assert result.success is False
        assert result.tasks_failed == 1
        assert "ruff not found" in result.errors[0]

    def test_to_dict(self) -> None:
        """RepairExecutionResult.to_dict() returns correct dict."""
        result = RepairExecutionResult(
            success=True,
            tasks_executed=2,
            tasks_succeeded=2,
            tasks_failed=0,
            artifacts_written=["file1.json"],
            errors=[],
        )
        d = result.to_dict()
        assert d == {
            "success": True,
            "tasks_executed": 2,
            "tasks_succeeded": 2,
            "tasks_failed": 0,
            "artifacts_written": ["file1.json"],
            "errors": [],
        }


# -----------------------------------------------------------------------------
# RepairExecutor tests
# -----------------------------------------------------------------------------


class TestRepairExecutor:
    """Tests for RepairExecutor class."""

    def test_init(self, tmp_path: Path) -> None:
        """RepairExecutor initializes correctly."""
        executor = RepairExecutor(
            project_root=tmp_path,
            runs_dir=tmp_path / "runs",
            verbose=False,
        )
        assert executor.project_root == tmp_path
        assert executor.runs_dir == tmp_path / "runs"
        assert executor.verbose is False

    def test_scope_lists(self) -> None:
        """RepairExecutor has expected scope lists."""
        assert "bridge/**" in RepairExecutor.ORCHESTRATOR_ALLOWLIST
        assert "src/**" in RepairExecutor.ORCHESTRATOR_DENYLIST
        assert "DESIGN_DOCUMENT.md" in RepairExecutor.ORCHESTRATOR_DENYLIST

    def test_execute_deterministic_repairs_success(self, tmp_path: Path) -> None:
        """execute_deterministic_repairs runs ruff commands."""
        executor = RepairExecutor(
            project_root=tmp_path,
            runs_dir=tmp_path / "runs",
            verbose=False,
        )

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout="1 file reformatted",
                stderr="",
            )
            result = executor.execute_deterministic_repairs()

        assert result.tasks_executed == 2  # ruff check + ruff format
        assert mock_run.call_count == 2

    def test_execute_deterministic_repairs_ruff_not_found(self, tmp_path: Path) -> None:
        """execute_deterministic_repairs handles ruff not found."""
        executor = RepairExecutor(
            project_root=tmp_path,
            runs_dir=tmp_path / "runs",
            verbose=False,
        )

        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("ruff not found")
            result = executor.execute_deterministic_repairs()

        assert result.success is False
        assert "ruff not found" in result.errors[0]

    def test_execute_deterministic_repairs_timeout(self, tmp_path: Path) -> None:
        """execute_deterministic_repairs handles timeout."""
        executor = RepairExecutor(
            project_root=tmp_path,
            runs_dir=tmp_path / "runs",
            verbose=False,
        )

        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ruff"], timeout=120)
            result = executor.execute_deterministic_repairs()

        assert result.success is False
        assert any("timed out" in e for e in result.errors)


class TestRepairExecutorCallback:
    """Tests for RepairExecutor.create_agent_task_callback()."""

    def test_empty_tasks_returns_true(self, tmp_path: Path) -> None:
        """Callback returns True for empty task list."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        executor = RepairExecutor(
            project_root=tmp_path,
            runs_dir=runs_dir,
            verbose=False,
        )
        callback = executor.create_agent_task_callback()

        result = callback([])
        assert result is True

    def test_writes_repair_plan_artifact(self, tmp_path: Path) -> None:
        """Callback writes repair plan JSON artifact."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        executor = RepairExecutor(
            project_root=tmp_path,
            runs_dir=runs_dir,
            verbose=False,
        )
        callback = executor.create_agent_task_callback()

        task = RepairTask(
            id="TEST-001",
            title="Test task",
            description="Test description",
        )

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            callback([task])

        # Find the repair plan file
        plan_files = list(runs_dir.glob("repair_plan_*.json"))
        assert len(plan_files) == 1

        plan_content = json.loads(plan_files[0].read_text())
        assert "timestamp" in plan_content
        assert "tasks" in plan_content
        assert len(plan_content["tasks"]) == 1
        assert plan_content["tasks"][0]["id"] == "TEST-001"

    def test_runs_deterministic_repairs(self, tmp_path: Path) -> None:
        """Callback runs deterministic repairs."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        executor = RepairExecutor(
            project_root=tmp_path,
            runs_dir=runs_dir,
            verbose=False,
        )
        callback = executor.create_agent_task_callback()

        task = RepairTask(
            id="TEST-001",
            title="Test task",
            description="Test description",
        )

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            result = callback([task])

        # Should have called ruff check and ruff format
        assert mock_run.call_count >= 2
        assert result is True

    def test_uses_scheduler_callback_when_provided(self, tmp_path: Path) -> None:
        """Callback uses scheduler_callback for orchestrator tasks."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        executor = RepairExecutor(
            project_root=tmp_path,
            runs_dir=runs_dir,
            verbose=False,
        )

        scheduler_called = []

        def mock_scheduler(tasks: list) -> bool:
            scheduler_called.append(tasks)
            return True

        callback = executor.create_agent_task_callback(scheduler_callback=mock_scheduler)

        # Task targeting orchestrator files (allowed)
        task = RepairTask(
            id="TEST-001",
            title="Test task",
            description="Test description",
            target_files=[],  # Empty means no scope check needed
        )

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            result = callback([task])

        assert result is True
        assert len(scheduler_called) == 1

    def test_filters_out_of_scope_tasks(self, tmp_path: Path) -> None:
        """Callback filters tasks targeting out-of-scope files."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        executor = RepairExecutor(
            project_root=tmp_path,
            runs_dir=runs_dir,
            verbose=False,
        )

        scheduler_called = []

        def mock_scheduler(tasks: list) -> bool:
            scheduler_called.append(tasks)
            return True

        callback = executor.create_agent_task_callback(scheduler_callback=mock_scheduler)

        # Task targeting src/ files (denied)
        task = RepairTask(
            id="TEST-001",
            title="Test task",
            description="Test description",
            target_files=["src/formula_foundry/core.py"],  # Out of scope
        )

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            callback([task])

        # Scheduler should not have been called with this task
        if scheduler_called:
            # If called, should be empty or not contain our task
            for call in scheduler_called:
                task_ids = [t.get("id") for t in call]
                assert "TEST-001" not in task_ids

    def test_writes_out_of_scope_artifact(self, tmp_path: Path) -> None:
        """Callback writes artifact for out-of-scope tasks."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        executor = RepairExecutor(
            project_root=tmp_path,
            runs_dir=runs_dir,
            verbose=False,
        )
        callback = executor.create_agent_task_callback()

        # Task targeting src/ files (denied)
        task = RepairTask(
            id="TEST-OUT-OF-SCOPE",
            title="Out of scope task",
            description="Test description",
            target_files=["src/formula_foundry/core.py"],
        )

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            callback([task])

        # Check for out-of-scope artifact
        out_of_scope_path = runs_dir / "out_of_scope_repairs.json"
        assert out_of_scope_path.exists()

        content = json.loads(out_of_scope_path.read_text())
        assert "tasks" in content
        assert len(content["tasks"]) == 1
        assert content["tasks"][0]["id"] == "TEST-OUT-OF-SCOPE"


# -----------------------------------------------------------------------------
# create_repair_callback tests
# -----------------------------------------------------------------------------


class TestCreateRepairCallback:
    """Tests for create_repair_callback factory function."""

    def test_creates_callback(self, tmp_path: Path) -> None:
        """create_repair_callback creates a callable."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        callback = create_repair_callback(
            project_root=tmp_path,
            runs_dir=runs_dir,
            verbose=False,
        )
        assert callable(callback)

    def test_callback_works(self, tmp_path: Path) -> None:
        """create_repair_callback returns working callback."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        callback = create_repair_callback(
            project_root=tmp_path,
            runs_dir=runs_dir,
            verbose=False,
        )

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            result = callback([])

        assert result is True

    def test_with_scheduler_callback(self, tmp_path: Path) -> None:
        """create_repair_callback accepts scheduler_callback."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        scheduler_called = []

        def mock_scheduler(tasks: list) -> bool:
            scheduler_called.append(tasks)
            return True

        callback = create_repair_callback(
            project_root=tmp_path,
            runs_dir=runs_dir,
            verbose=False,
            scheduler_callback=mock_scheduler,
        )

        task = RepairTask(
            id="TEST-001",
            title="Test task",
            description="Test",
            target_files=[],
        )

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            callback([task])

        assert len(scheduler_called) == 1
