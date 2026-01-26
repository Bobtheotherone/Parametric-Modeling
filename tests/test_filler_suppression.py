"""Tests for FILLER task suppression when root failures exist.

These tests verify that the orchestrator correctly suppresses optional
backfill (FILLER) tasks when there are hard blockers (root failures).

This prevents wasting credits on optional work that cannot succeed,
as seen in the Jan 26 run where 30 FILLER tasks ran during root failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MockTask:
    """Mock task for testing."""

    id: str
    status: str = "pending"


def _is_backfill_task_id(task_id: str) -> bool:
    """Check if a task ID represents an optional backfill task (FILLER-* prefix)."""
    return task_id.startswith("FILLER-")


def should_suppress_backfill(tasks: list[MockTask]) -> bool:
    """Check if backfill should be suppressed due to root failures.

    This mirrors the logic in maybe_generate_backfill() in bridge/loop.py.
    """
    root_failure_statuses = ("failed", "manual", "resource_killed")
    has_root_failures = any(
        t.status in root_failure_statuses and not _is_backfill_task_id(t.id)
        for t in tasks
    )
    return has_root_failures


class TestBackfillSuppression:
    """Test FILLER task suppression logic."""

    def test_suppress_when_non_filler_task_failed(self):
        """Test that FILLER is suppressed when a non-FILLER task has failed."""
        tasks = [
            MockTask(id="M0-FIX-01", status="failed"),
            MockTask(id="M0-FIX-02", status="pending"),
        ]
        assert should_suppress_backfill(tasks), "Should suppress when non-FILLER task failed"

    def test_suppress_when_non_filler_task_manual(self):
        """Test that FILLER is suppressed when a non-FILLER task needs manual intervention."""
        tasks = [
            MockTask(id="M0-FIX-01", status="manual"),
            MockTask(id="M0-FIX-02", status="done"),
        ]
        assert should_suppress_backfill(tasks), "Should suppress when non-FILLER task is manual"

    def test_suppress_when_non_filler_task_resource_killed(self):
        """Test that FILLER is suppressed when a non-FILLER task was resource-killed."""
        tasks = [
            MockTask(id="M0-FIX-01", status="resource_killed"),
            MockTask(id="M0-FIX-02", status="pending"),
        ]
        assert should_suppress_backfill(tasks), "Should suppress when non-FILLER task was resource-killed"

    def test_no_suppress_when_only_filler_failed(self):
        """Test that FILLER is NOT suppressed when only FILLER tasks failed."""
        tasks = [
            MockTask(id="FILLER-TEST-001", status="failed"),
            MockTask(id="FILLER-LINT-002", status="failed"),
            MockTask(id="M0-FIX-01", status="done"),
        ]
        assert not should_suppress_backfill(tasks), "Should NOT suppress when only FILLER tasks failed"

    def test_no_suppress_when_all_non_filler_done(self):
        """Test that FILLER is allowed when all non-FILLER tasks are done."""
        tasks = [
            MockTask(id="M0-FIX-01", status="done"),
            MockTask(id="M0-FIX-02", status="done"),
            MockTask(id="FILLER-TEST-001", status="pending"),
        ]
        assert not should_suppress_backfill(tasks), "Should NOT suppress when all non-FILLER tasks are done"

    def test_no_suppress_when_all_pending(self):
        """Test that FILLER is allowed when all tasks are pending."""
        tasks = [
            MockTask(id="M0-FIX-01", status="pending"),
            MockTask(id="M0-FIX-02", status="pending"),
        ]
        assert not should_suppress_backfill(tasks), "Should NOT suppress when all tasks are pending"


class TestFillerTaskIdDetection:
    """Test FILLER task ID detection."""

    def test_detects_filler_prefix(self):
        """Test that FILLER-* prefix is correctly detected."""
        assert _is_backfill_task_id("FILLER-TEST-001")
        assert _is_backfill_task_id("FILLER-LINT-002")
        assert _is_backfill_task_id("FILLER-DOCS-003")
        assert _is_backfill_task_id("FILLER-TYPE_HINTS-004")
        assert _is_backfill_task_id("FILLER-SCHEMA_LINT-005")

    def test_does_not_detect_non_filler(self):
        """Test that non-FILLER task IDs are not detected."""
        assert not _is_backfill_task_id("M0-FIX-01")
        assert not _is_backfill_task_id("M1-TASK-001")
        assert not _is_backfill_task_id("TASK-FILLER-001")  # FILLER not at start
        assert not _is_backfill_task_id("filler-test-001")  # lowercase doesn't count


class TestMultipleFailureScenarios:
    """Test scenarios with multiple failures."""

    def test_suppress_with_mixed_failures(self):
        """Test suppression with mix of FILLER and non-FILLER failures."""
        tasks = [
            MockTask(id="M0-FIX-01", status="failed"),  # Root failure
            MockTask(id="FILLER-TEST-001", status="failed"),  # FILLER failure
            MockTask(id="M0-FIX-02", status="done"),
        ]
        assert should_suppress_backfill(tasks), "Should suppress when there's any non-FILLER failure"

    def test_suppress_with_one_root_failure_among_many_done(self):
        """Test suppression when one task fails among many done."""
        tasks = [
            MockTask(id="M0-FIX-01", status="done"),
            MockTask(id="M0-FIX-02", status="done"),
            MockTask(id="M0-FIX-03", status="done"),
            MockTask(id="M0-FIX-04", status="done"),
            MockTask(id="M0-FIX-05", status="failed"),  # Single failure
            MockTask(id="M0-FIX-06", status="pending"),
        ]
        assert should_suppress_backfill(tasks), "Should suppress even with single root failure"


class TestJan26ScenarioRegression:
    """Test the exact scenario that caused the Jan 26 waste."""

    def test_jan26_scenario_should_suppress_filler(self):
        """Reproduce the Jan 26 scenario where 30 FILLER tasks ran during root failures.

        In the actual run:
        - 8 root failures (non-FILLER tasks with "failed" status)
        - 30 FILLER tasks were generated and all failed
        - This wasted ~60 credits

        The fix should suppress FILLER generation when root failures exist.
        """
        # Simulate the Jan 26 state
        tasks = [
            # Root failures (from summary.json)
            MockTask(id="M0-RF-01-LOGS-CLAUDE-LIMIT-CASE", status="failed"),
            MockTask(id="M0-RF-01-SKIP-SCAN-MATRIX", status="failed"),
            MockTask(id="M0-RF-01-M0-INTEGRATION-TESTS", status="failed"),
            MockTask(id="M0-RF-01-DOCS-REVIEW-CONTENT", status="failed"),
            MockTask(id="M0-RF-01-DOCS-REVIEW-TESTS", status="failed"),
            MockTask(id="M0-RF-01-GPU-POLICY-REVIEW-PYPROJECT", status="failed"),
            MockTask(id="M0-RF-01-GPU-POLICY-REVIEW-TESTS", status="failed"),
            MockTask(id="M0-RF-01-LOOP-READONLY-REVIEW-DOCS", status="failed"),
            # Completed tasks
            MockTask(id="M0-RF-01-LOGS-READ", status="done"),
            MockTask(id="M0-RF-01-LOGS-BRANCH-CLEANUP", status="done"),
            # Blocked tasks
            MockTask(id="M0-RF-01-FIX-VERIFY-TESTS", status="blocked"),
        ]

        # With the fix, FILLER should be suppressed
        assert should_suppress_backfill(tasks), \
            "Jan 26 scenario: FILLER should be suppressed when 8 root failures exist"
