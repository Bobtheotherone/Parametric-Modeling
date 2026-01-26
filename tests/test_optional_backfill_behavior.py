#!/usr/bin/env python3
"""Unit tests for optional backfill (FILLER-*) task behavior.

These tests verify that backfill tasks are treated as optional:
1. Backfill failures do NOT cause run failure or continuation prompts
2. Manual tasks always have manual_path written

Run with: pytest tests/test_optional_backfill_behavior.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def mock_primary_task():
    """Create a mock primary (non-backfill) task that completed successfully."""
    from bridge.loop import ParallelTask

    return ParallelTask(
        id="M1-T01",
        title="Primary task",
        description="A primary implementation task",
        agent="claude",
        status="done",
        work_completed=True,
        commit_sha="abc123",
    )


@pytest.fixture
def mock_backfill_failed_task():
    """Create a mock backfill task that failed."""
    from bridge.loop import ParallelTask

    return ParallelTask(
        id="FILLER-LINT-001",
        title="Backfill lint task",
        description="Optional lint improvements",
        agent="claude",
        status="failed",
        error="Lint check failed",
    )


@pytest.fixture
def mock_backfill_manual_task():
    """Create a mock backfill task marked manual."""
    from bridge.loop import ParallelTask

    return ParallelTask(
        id="FILLER-TYPE-002",
        title="Backfill type hints",
        description="Optional type hint improvements",
        agent="claude",
        status="manual",
        error="needs_manual_resolution: conflict in type hints",
    )


@pytest.fixture
def mock_primary_failed_task():
    """Create a mock primary task that failed."""
    from bridge.loop import ParallelTask

    return ParallelTask(
        id="M1-T02",
        title="Failed primary task",
        description="A primary task that failed",
        agent="claude",
        status="failed",
        error="Build failed",
    )


# -----------------------------------------------------------------------------
# _is_backfill_task_id Tests
# -----------------------------------------------------------------------------


class TestIsBackfillTaskId:
    """Tests for _is_backfill_task_id helper."""

    def test_filler_prefix_is_backfill(self):
        """FILLER- prefix should be identified as backfill."""
        from bridge.loop import _is_backfill_task_id

        assert _is_backfill_task_id("FILLER-LINT-001") is True
        assert _is_backfill_task_id("FILLER-TYPE-002") is True
        assert _is_backfill_task_id("FILLER-TEST-003") is True

    def test_non_filler_prefix_is_not_backfill(self):
        """Non-FILLER prefixes should not be identified as backfill."""
        from bridge.loop import _is_backfill_task_id

        assert _is_backfill_task_id("M1-T01") is False
        assert _is_backfill_task_id("M2-SIM-001") is False
        assert _is_backfill_task_id("TASK-001") is False
        assert _is_backfill_task_id("") is False


# -----------------------------------------------------------------------------
# _generate_run_summary Tests
# -----------------------------------------------------------------------------


class TestSummaryBackfillExclusion:
    """Tests for backfill exclusion in run summary."""

    def test_summary_success_ignores_backfill_failures(self, temp_dir, mock_primary_task, mock_backfill_failed_task):
        """Summary should report success when only backfill tasks failed."""
        from bridge.loop import _generate_run_summary

        tasks = [mock_primary_task, mock_backfill_failed_task]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=temp_dir,
            verify_exit_code=0,
        )

        # Success should be True since only backfill failed
        assert summary["success"] is True

        # root_failures should NOT include backfill task
        root_failure_ids = [t["id"] for t in summary["root_failures"]]
        assert "FILLER-LINT-001" not in root_failure_ids

        # optional_tasks should include the backfill task
        optional_task_ids = [t["id"] for t in summary["optional_tasks"]]
        assert "FILLER-LINT-001" in optional_task_ids

        # Counts should exclude backfill
        assert summary["failed"] == 0

    def test_summary_success_ignores_backfill_manual(self, temp_dir, mock_primary_task, mock_backfill_manual_task):
        """Summary should report success when only backfill tasks are manual."""
        from bridge.loop import _generate_run_summary

        tasks = [mock_primary_task, mock_backfill_manual_task]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=temp_dir,
            verify_exit_code=0,
        )

        # Success should be True since only backfill is manual
        assert summary["success"] is True

        # root_failures should NOT include backfill task
        root_failure_ids = [t["id"] for t in summary["root_failures"]]
        assert "FILLER-TYPE-002" not in root_failure_ids

        # optional_tasks should include the backfill task
        optional_task_ids = [t["id"] for t in summary["optional_tasks"]]
        assert "FILLER-TYPE-002" in optional_task_ids

    def test_summary_failure_when_primary_fails(
        self, temp_dir, mock_primary_task, mock_primary_failed_task, mock_backfill_failed_task
    ):
        """Summary should report failure when primary tasks fail."""
        from bridge.loop import _generate_run_summary

        tasks = [mock_primary_task, mock_primary_failed_task, mock_backfill_failed_task]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=temp_dir,
            verify_exit_code=0,
        )

        # Success should be False because primary task failed
        assert summary["success"] is False

        # root_failures should include primary task but NOT backfill
        root_failure_ids = [t["id"] for t in summary["root_failures"]]
        assert "M1-T02" in root_failure_ids
        assert "FILLER-LINT-001" not in root_failure_ids

        # Counts should only count primary failures
        assert summary["failed"] == 1

    def test_summary_blocked_excludes_backfill(self, temp_dir):
        """Blocked tasks list should exclude backfill tasks."""
        from bridge.loop import ParallelTask, _generate_run_summary

        primary_blocked = ParallelTask(
            id="M1-T03",
            title="Blocked primary",
            description="Blocked task",
            agent="claude",
            status="blocked",
            error="Blocked by M1-T02",
        )
        backfill_blocked = ParallelTask(
            id="FILLER-DOC-001",
            title="Blocked backfill",
            description="Blocked backfill task",
            agent="claude",
            status="blocked",
            error="Blocked by dependency",
        )

        tasks = [primary_blocked, backfill_blocked]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=temp_dir,
            verify_exit_code=0,
        )

        # blocked_tasks should NOT include backfill
        blocked_ids = [t["id"] for t in summary["blocked_tasks"]]
        assert "M1-T03" in blocked_ids
        assert "FILLER-DOC-001" not in blocked_ids

        # Blocked count should exclude backfill
        assert summary["blocked"] == 1


# -----------------------------------------------------------------------------
# _generate_continuation_prompt Tests
# -----------------------------------------------------------------------------


class TestContinuationPromptBackfillExclusion:
    """Tests for backfill exclusion in continuation prompts."""

    def test_continuation_prompt_does_not_include_backfill(self, temp_dir, mock_primary_task, mock_backfill_failed_task):
        """Continuation prompt should NOT mention backfill tasks as failures."""
        from bridge.loop import _generate_continuation_prompt, _generate_run_summary

        tasks = [mock_primary_task, mock_backfill_failed_task]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=temp_dir,
            verify_exit_code=0,
        )

        prompt = _generate_continuation_prompt(
            summary=summary,
            tasks=tasks,
            design_doc_text="Test design doc",
            runs_dir=temp_dir,
        )

        # FILLER task should NOT appear in the prompt as a root failure
        assert "FILLER-LINT-001" not in prompt

    def test_continuation_prompt_includes_primary_failures(self, temp_dir, mock_primary_failed_task, mock_backfill_failed_task):
        """Continuation prompt SHOULD mention primary task failures."""
        from bridge.loop import _generate_continuation_prompt, _generate_run_summary

        tasks = [mock_primary_failed_task, mock_backfill_failed_task]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=temp_dir,
            verify_exit_code=0,
        )

        prompt = _generate_continuation_prompt(
            summary=summary,
            tasks=tasks,
            design_doc_text="Test design doc",
            runs_dir=temp_dir,
        )

        # Primary failure should appear in prompt
        assert "M1-T02" in prompt
        # Backfill failure should NOT appear
        assert "FILLER-LINT-001" not in prompt


# -----------------------------------------------------------------------------
# _mark_task_manual Tests
# -----------------------------------------------------------------------------


class TestMarkTaskManual:
    """Tests for _mark_task_manual helper."""

    def test_mark_task_manual_writes_manual_file(self, temp_dir):
        """_mark_task_manual should set status, error, and write manual_path file."""
        from bridge.loop import ParallelTask, _mark_task_manual

        task = ParallelTask(
            id="M1-T99",
            title="Test manual task",
            description="A task for testing manual marking",
            agent="claude",
            status="running",
            worktree_path=temp_dir / "worktree",
            worker_id=1,
        )

        manual_dir = temp_dir / "manual"
        schema_path = temp_dir / "schema.json"

        # Create schema file (needed by _write_manual_task_file)
        schema_path.write_text("{}", encoding="utf-8")

        _mark_task_manual(
            task=task,
            reason="Test reason for manual intervention",
            manual_dir=manual_dir,
            schema_path=schema_path,
        )

        # Status should be manual
        assert task.status == "manual"

        # Error should be set
        assert task.error == "Test reason for manual intervention"

        # manual_path should be set and file should exist
        assert task.manual_path is not None
        assert task.manual_path.exists()

        # File content should contain task ID and reason
        content = task.manual_path.read_text()
        assert "M1-T99" in content
        assert "Test reason for manual intervention" in content

    def test_mark_task_manual_does_not_overwrite_existing_path(self, temp_dir):
        """_mark_task_manual should not overwrite existing manual_path."""
        from bridge.loop import ParallelTask, _mark_task_manual

        existing_path = temp_dir / "existing_manual.md"
        existing_path.write_text("Existing content", encoding="utf-8")

        task = ParallelTask(
            id="M1-T98",
            title="Test task with existing path",
            description="A task with pre-existing manual_path",
            agent="claude",
            status="running",
            manual_path=existing_path,
            worktree_path=temp_dir / "worktree",
            worker_id=1,
        )

        manual_dir = temp_dir / "manual"
        schema_path = temp_dir / "schema.json"
        schema_path.write_text("{}", encoding="utf-8")

        _mark_task_manual(
            task=task,
            reason="New reason",
            manual_dir=manual_dir,
            schema_path=schema_path,
        )

        # manual_path should still point to existing file
        assert task.manual_path == existing_path

        # Existing content should be preserved
        assert task.manual_path.read_text() == "Existing content"


# -----------------------------------------------------------------------------
# Integration: needs_continuation logic
# -----------------------------------------------------------------------------


class TestNeedsContinuationLogic:
    """Tests for needs_continuation calculation excluding backfill tasks."""

    def test_no_continuation_when_only_backfill_fails(self, temp_dir, mock_primary_task, mock_backfill_failed_task):
        """needs_continuation should be False when only backfill tasks fail."""
        from bridge.loop import _generate_run_summary

        tasks = [mock_primary_task, mock_backfill_failed_task]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=temp_dir,
            verify_exit_code=0,
        )

        # The counts used for needs_continuation should exclude backfill
        failed_count = summary["failed"]
        pending_rerun_count = summary["pending_rerun"]
        blocked_count = summary["blocked"]

        # All counts should be zero since only backfill failed
        assert failed_count == 0
        assert pending_rerun_count == 0
        assert blocked_count == 0

        # Therefore needs_continuation would be False
        needs_continuation = (failed_count + pending_rerun_count + blocked_count) > 0
        assert needs_continuation is False

    def test_continuation_when_primary_fails(self, temp_dir, mock_primary_failed_task, mock_backfill_failed_task):
        """needs_continuation should be True when primary tasks fail."""
        from bridge.loop import _generate_run_summary

        tasks = [mock_primary_failed_task, mock_backfill_failed_task]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=temp_dir,
            verify_exit_code=0,
        )

        failed_count = summary["failed"]

        # Primary failure should be counted
        assert failed_count == 1

        # Therefore needs_continuation would be True
        needs_continuation = failed_count > 0
        assert needs_continuation is True
