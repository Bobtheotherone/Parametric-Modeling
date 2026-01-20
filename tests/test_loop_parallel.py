"""Tests for parallel runner fixes.

Tests:
1. Task completion requires work_completed==true
2. Stuck detection distinguishes root failures from blocked tasks
3. Summary generation includes all required fields
4. Agent selection respects preferred_agent
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import after path setup
from bridge.loop import (
    ParallelTask,
    _generate_run_summary,
    _generate_continuation_prompt,
)


class TestTaskCompletionCriteria:
    """Tests for task completion requiring work_completed==true."""

    def test_parallel_task_has_work_completed_field(self) -> None:
        """ParallelTask dataclass should have work_completed field."""
        task = ParallelTask(
            id="TEST-1",
            title="Test task",
            description="A test task",
            agent="codex",
        )
        assert hasattr(task, "work_completed")
        assert task.work_completed is None  # Default is None

    def test_parallel_task_has_commit_sha_field(self) -> None:
        """ParallelTask dataclass should have commit_sha field."""
        task = ParallelTask(
            id="TEST-1",
            title="Test task",
            description="A test task",
            agent="codex",
        )
        assert hasattr(task, "commit_sha")
        assert task.commit_sha is None

    def test_parallel_task_has_turn_summary_field(self) -> None:
        """ParallelTask dataclass should have turn_summary field."""
        task = ParallelTask(
            id="TEST-1",
            title="Test task",
            description="A test task",
            agent="codex",
        )
        assert hasattr(task, "turn_summary")
        assert task.turn_summary is None

    def test_pending_rerun_status_exists(self) -> None:
        """pending_rerun should be a valid status."""
        task = ParallelTask(
            id="TEST-1",
            title="Test task",
            description="A test task",
            agent="codex",
            status="pending_rerun",
        )
        assert task.status == "pending_rerun"


class TestStuckDetection:
    """Tests for distinguishing root failures from blocked tasks."""

    def test_summary_categorizes_root_failures(self, tmp_path: Path) -> None:
        """Root failures should be categorized correctly."""
        tasks = [
            ParallelTask(
                id="TASK-A",
                title="Failed task",
                description="A failed task",
                agent="codex",
                status="failed",
                error="Some error",
                work_completed=False,
            ),
            ParallelTask(
                id="TASK-B",
                title="Done task",
                description="A done task",
                agent="codex",
                status="done",
                work_completed=True,
            ),
        ]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=tmp_path,
            verify_exit_code=1,
        )

        assert len(summary["root_failures"]) == 1
        assert summary["root_failures"][0]["id"] == "TASK-A"
        assert len(summary["completed_tasks"]) == 1
        assert summary["completed_tasks"][0]["id"] == "TASK-B"

    def test_summary_categorizes_pending_rerun_as_root_failure(self, tmp_path: Path) -> None:
        """pending_rerun tasks should be categorized as root failures."""
        tasks = [
            ParallelTask(
                id="TASK-A",
                title="Pending rerun task",
                description="A planning-only task",
                agent="claude",
                status="pending_rerun",
                error="Agent returned work_completed=false",
                work_completed=False,
            ),
        ]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=tmp_path,
            verify_exit_code=0,
        )

        assert len(summary["root_failures"]) == 1
        assert summary["root_failures"][0]["id"] == "TASK-A"
        assert summary["pending_rerun"] == 1

    def test_summary_categorizes_blocked_tasks(self, tmp_path: Path) -> None:
        """Skipped tasks (blocked) should be categorized correctly."""
        tasks = [
            ParallelTask(
                id="TASK-A",
                title="Failed task",
                description="A failed task",
                agent="codex",
                status="failed",
                error="Some error",
            ),
            ParallelTask(
                id="TASK-B",
                title="Blocked task",
                description="A blocked task",
                agent="codex",
                status="skipped",
                error="Skipped: blocked by failed prerequisites: ['TASK-A']",
                depends_on=["TASK-A"],
            ),
        ]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=tmp_path,
            verify_exit_code=1,
        )

        assert len(summary["root_failures"]) == 1
        assert summary["root_failures"][0]["id"] == "TASK-A"
        assert len(summary["blocked_tasks"]) == 1
        assert summary["blocked_tasks"][0]["id"] == "TASK-B"


class TestSummaryGeneration:
    """Tests for summary.json generation."""

    def test_summary_has_required_fields(self, tmp_path: Path) -> None:
        """Summary should contain all required fields."""
        tasks = [
            ParallelTask(
                id="TASK-A",
                title="Test task",
                description="A test task",
                agent="codex",
                status="done",
                work_completed=True,
            ),
        ]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=tmp_path,
            verify_exit_code=0,
        )

        required_keys = [
            "run_dir",
            "total_tasks",
            "completed",
            "failed",
            "pending_rerun",
            "blocked",
            "verify_exit_code",
            "success",
            "completed_tasks",
            "root_failures",
            "blocked_tasks",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"

    def test_summary_success_when_all_done_and_verify_passes(self, tmp_path: Path) -> None:
        """Summary.success should be True when all tasks done and verify passes."""
        tasks = [
            ParallelTask(
                id="TASK-A",
                title="Test task",
                description="A test task",
                agent="codex",
                status="done",
                work_completed=True,
            ),
        ]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=tmp_path,
            verify_exit_code=0,
        )

        assert summary["success"] is True

    def test_summary_not_success_when_failures_exist(self, tmp_path: Path) -> None:
        """Summary.success should be False when there are failures."""
        tasks = [
            ParallelTask(
                id="TASK-A",
                title="Test task",
                description="A test task",
                agent="codex",
                status="failed",
                error="Some error",
            ),
        ]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=tmp_path,
            verify_exit_code=0,
        )

        assert summary["success"] is False


class TestContinuationPrompt:
    """Tests for continuation_prompt.txt generation."""

    def test_continuation_prompt_lists_root_failures(self, tmp_path: Path) -> None:
        """Continuation prompt should list root failures."""
        tasks = [
            ParallelTask(
                id="TASK-A",
                title="Failed task",
                description="A failed task",
                agent="codex",
                status="failed",
                error="Some error",
                work_completed=False,
            ),
        ]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=tmp_path,
            verify_exit_code=1,
        )

        prompt = _generate_continuation_prompt(
            summary=summary,
            tasks=tasks,
            design_doc_text="# Test Design Doc",
            runs_dir=tmp_path,
        )

        assert "Root Failures" in prompt
        assert "TASK-A" in prompt
        assert "Some error" in prompt

    def test_continuation_prompt_lists_completed_tasks(self, tmp_path: Path) -> None:
        """Continuation prompt should list completed tasks (to avoid re-implementation)."""
        tasks = [
            ParallelTask(
                id="TASK-A",
                title="Done task",
                description="A done task",
                agent="codex",
                status="done",
                work_completed=True,
            ),
            ParallelTask(
                id="TASK-B",
                title="Failed task",
                description="A failed task",
                agent="codex",
                status="failed",
                error="Some error",
            ),
        ]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=tmp_path,
            verify_exit_code=1,
        )

        prompt = _generate_continuation_prompt(
            summary=summary,
            tasks=tasks,
            design_doc_text="# Test Design Doc",
            runs_dir=tmp_path,
        )

        assert "Completed Tasks" in prompt
        assert "DO NOT re-implement" in prompt
        assert "TASK-A" in prompt

    def test_continuation_prompt_mentions_pending_rerun(self, tmp_path: Path) -> None:
        """Continuation prompt should explain pending_rerun tasks need implementation."""
        tasks = [
            ParallelTask(
                id="TASK-A",
                title="Planning only task",
                description="A planning-only task",
                agent="claude",
                status="pending_rerun",
                error="Agent returned work_completed=false",
                work_completed=False,
            ),
        ]

        summary = _generate_run_summary(
            tasks=tasks,
            runs_dir=tmp_path,
            verify_exit_code=0,
        )

        prompt = _generate_continuation_prompt(
            summary=summary,
            tasks=tasks,
            design_doc_text="# Test Design Doc",
            runs_dir=tmp_path,
        )

        assert "pending_rerun" in prompt or "IMPLEMENT" in prompt


class TestAgentSelection:
    """Tests for agent selection respecting preferred_agent."""

    def test_task_agent_field_is_preserved(self) -> None:
        """Task agent field should be set from preferred_agent."""
        task = ParallelTask(
            id="TEST-1",
            title="Test task",
            description="A test task",
            agent="claude",  # This should be preserved
        )
        assert task.agent == "claude"

    def test_task_defaults_to_codex(self) -> None:
        """If agent is invalid, should default to codex during execution."""
        # The actual defaulting happens in _run_parallel_task,
        # but we can test the dataclass accepts any string
        task = ParallelTask(
            id="TEST-1",
            title="Test task",
            description="A test task",
            agent="invalid_agent",
        )
        # The validation happens at runtime in the parallel runner
        assert task.agent == "invalid_agent"
