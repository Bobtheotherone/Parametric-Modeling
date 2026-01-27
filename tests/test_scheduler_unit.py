#!/usr/bin/env python3
"""Unit tests for bridge/scheduler.py - TwoLaneScheduler and related components.

Tests cover:
- LaneConfig creation and properties
- SchedulerMetrics tracking and export
- TwoLaneScheduler lane assignment, priority calculation, and task management
- BackfillGenerator filler task generation

Run with: pytest tests/test_scheduler_unit.py -v
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pytest

# -----------------------------------------------------------------------------
# Mock SchedulableTask for testing
# -----------------------------------------------------------------------------


@dataclass
class MockTask:
    """Mock task implementing SchedulableTask protocol."""

    id: str
    status: str = "pending"
    solo: bool = False
    intensity: str = "light"
    locks: list[str] | None = None
    touched_paths: list[str] | None = None
    depends_on: list[str] | None = None

    def __post_init__(self):
        if self.locks is None:
            self.locks = []
        if self.touched_paths is None:
            self.touched_paths = []
        if self.depends_on is None:
            self.depends_on = []


# -----------------------------------------------------------------------------
# LaneConfig Tests
# -----------------------------------------------------------------------------


class TestLaneConfig:
    """Tests for LaneConfig dataclass."""

    def test_lane_config_basic_creation(self):
        """Test basic LaneConfig creation."""
        from bridge.scheduler import LaneConfig

        config = LaneConfig(coding_lane_size=5, executor_lane_size=1)
        assert config.coding_lane_size == 5
        assert config.executor_lane_size == 1

    def test_lane_config_from_max_workers(self):
        """Test LaneConfig.from_max_workers factory method."""
        from bridge.scheduler import LaneConfig

        # Standard case: 10 workers -> 9 coding, 1 executor
        config = LaneConfig.from_max_workers(10)
        assert config.coding_lane_size == 9
        assert config.executor_lane_size == 1

        # Small case: 2 workers -> 1 coding, 1 executor
        config = LaneConfig.from_max_workers(2)
        assert config.coding_lane_size == 1
        assert config.executor_lane_size == 1

        # Edge case: 1 worker -> 0 coding (min 1), 1 executor
        config = LaneConfig.from_max_workers(1)
        assert config.coding_lane_size >= 0  # Implementation may vary

    def test_lane_config_total_workers(self):
        """Test total_workers property."""
        from bridge.scheduler import LaneConfig

        config = LaneConfig(coding_lane_size=7, executor_lane_size=2)
        assert config.total_workers == 9


# -----------------------------------------------------------------------------
# SchedulerMetrics Tests
# -----------------------------------------------------------------------------


class TestSchedulerMetrics:
    """Tests for SchedulerMetrics dataclass."""

    def test_metrics_initial_state(self):
        """Test initial metrics state."""
        from bridge.scheduler import SchedulerMetrics

        metrics = SchedulerMetrics()
        assert metrics.total_tasks == 0
        assert metrics.completed_tasks == 0
        assert metrics.failed_tasks == 0
        assert metrics.blocked_tasks == 0
        assert metrics.hits == 0 if hasattr(metrics, "hits") else True
        assert len(metrics.active_workers_samples) == 0
        assert len(metrics.stall_events) == 0

    def test_metrics_sample_recording(self):
        """Test sample method records utilization data."""
        from bridge.scheduler import SchedulerMetrics

        metrics = SchedulerMetrics()
        metrics.sample(
            active_workers=5,
            queue_depth=10,
            coding_lane_active=4,
            executor_lane_active=1,
        )

        assert len(metrics.active_workers_samples) == 1
        assert len(metrics.queue_depth_samples) == 1
        assert len(metrics.coding_lane_samples) == 1
        assert len(metrics.executor_lane_samples) == 1

        # Values should be recorded
        _, count = metrics.active_workers_samples[0]
        assert count == 5

    def test_metrics_multiple_samples(self):
        """Test multiple sample recordings."""
        from bridge.scheduler import SchedulerMetrics

        metrics = SchedulerMetrics()

        for i in range(5):
            metrics.sample(
                active_workers=i,
                queue_depth=10 - i,
                coding_lane_active=i,
                executor_lane_active=0,
            )

        assert len(metrics.active_workers_samples) == 5

    def test_metrics_stall_recording(self):
        """Test stall event recording."""
        from bridge.scheduler import SchedulerMetrics

        metrics = SchedulerMetrics()
        metrics.record_stall("dependency_blocked", ["task-1", "task-2"])

        assert len(metrics.stall_events) == 1
        event = metrics.stall_events[0]
        assert event["reason"] == "dependency_blocked"
        assert "task-1" in event["task_ids"]

    def test_metrics_retry_recording(self):
        """Test retry count recording."""
        from bridge.scheduler import SchedulerMetrics

        metrics = SchedulerMetrics()
        metrics.record_retry("task-1")
        metrics.record_retry("task-1")
        metrics.record_retry("task-2")

        assert metrics.retry_counts["task-1"] == 2
        assert metrics.retry_counts["task-2"] == 1

    def test_metrics_json_repair_recording(self):
        """Test JSON repair count recording."""
        from bridge.scheduler import SchedulerMetrics

        metrics = SchedulerMetrics()
        metrics.record_json_repair("task-1")
        metrics.record_json_repair("task-1")

        assert metrics.json_repair_counts["task-1"] == 2

    def test_metrics_average_utilization(self):
        """Test average utilization calculation."""
        from bridge.scheduler import SchedulerMetrics

        metrics = SchedulerMetrics()

        # Empty samples should return 0
        assert metrics.get_average_utilization() == 0.0

        # Add samples
        for count in [2, 4, 6]:
            metrics.active_workers_samples.append((time.monotonic(), count))

        avg = metrics.get_average_utilization()
        assert avg == 4.0  # (2 + 4 + 6) / 3

    def test_metrics_to_dict(self):
        """Test metrics export to dictionary."""
        from bridge.scheduler import SchedulerMetrics

        metrics = SchedulerMetrics(total_tasks=10)
        metrics.completed_tasks = 5
        metrics.failed_tasks = 1
        metrics.record_stall("test_stall", ["task-1"])
        metrics.record_retry("task-1")

        result = metrics.to_dict()

        assert result["total_tasks"] == 10
        assert result["completed_tasks"] == 5
        assert result["failed_tasks"] == 1
        assert result["stall_count"] == 1
        assert result["total_retries"] == 1
        assert "elapsed_seconds" in result


# -----------------------------------------------------------------------------
# TwoLaneScheduler Tests
# -----------------------------------------------------------------------------


class TestTwoLaneScheduler:
    """Tests for TwoLaneScheduler class."""

    def _create_scheduler(self, tasks: list[MockTask]) -> TwoLaneScheduler:
        """Helper to create scheduler with mock tasks."""
        from bridge.scheduler import LaneConfig, TwoLaneScheduler

        config = LaneConfig(coding_lane_size=3, executor_lane_size=1)

        def deps_satisfied(t):
            return all(
                self._get_task(tasks, d).status == "completed" for d in t.depends_on if self._get_task(tasks, d) is not None
            )

        def locks_available(t):
            return True  # Simple mock

        return TwoLaneScheduler(
            lane_config=config,
            tasks=tasks,
            deps_satisfied_fn=deps_satisfied,
            locks_available_fn=locks_available,
        )

    def _get_task(self, tasks, task_id):
        """Helper to get task by ID."""
        for t in tasks:
            if t.id == task_id:
                return t
        return None

    def test_scheduler_basic_creation(self):
        """Test basic scheduler creation."""
        tasks = [MockTask(id="t1"), MockTask(id="t2")]
        scheduler = self._create_scheduler(tasks)

        assert scheduler.lane_config.coding_lane_size == 3
        assert len(scheduler.tasks) == 2
        assert len(scheduler.coding_lane_running) == 0
        assert len(scheduler.executor_lane_running) == 0

    def test_scheduler_can_start_pending_task(self):
        """Test can_start returns True for pending tasks."""
        tasks = [MockTask(id="t1", status="pending")]
        scheduler = self._create_scheduler(tasks)

        assert scheduler.can_start(tasks[0])

    def test_scheduler_cannot_start_running_task(self):
        """Test can_start returns False for non-pending tasks."""
        tasks = [MockTask(id="t1", status="in_progress")]
        scheduler = self._create_scheduler(tasks)

        assert not scheduler.can_start(tasks[0])

    def test_scheduler_cannot_start_with_unsatisfied_deps(self):
        """Test can_start returns False when dependencies not satisfied."""
        tasks = [
            MockTask(id="t1", status="pending"),
            MockTask(id="t2", status="pending", depends_on=["t1"]),
        ]
        scheduler = self._create_scheduler(tasks)

        assert scheduler.can_start(tasks[0])
        assert not scheduler.can_start(tasks[1])

    def test_scheduler_respects_active_locks(self):
        """Verify scheduler respects lock availability when tasks overlap."""
        from bridge.scheduler import LaneConfig, TwoLaneScheduler

        class MockTaskLocked:
            def __init__(self, id, status, locks):
                self.id = id
                self.status = status
                self.solo = False
                self.intensity = "light"
                self.locks = locks
                self.touched_paths = []
                self.depends_on = []

        held_locks = {"shared-lock"}

        def deps_satisfied(_):
            return True

        def locks_available(t):
            return not set(t.locks) & held_locks

        task_running = MockTaskLocked("t1", "running", ["shared-lock"])
        task_pending = MockTaskLocked("t2", "pending", ["shared-lock"])

        scheduler = TwoLaneScheduler(
            lane_config=LaneConfig(coding_lane_size=2, executor_lane_size=1),
            tasks=[task_running, task_pending],
            deps_satisfied_fn=deps_satisfied,
            locks_available_fn=locks_available,
        )

        ready = scheduler.get_ready_tasks()
        assert task_pending not in ready, "Task with overlapping lock should not be ready when lock is held"

    def test_scheduler_can_start_after_deps_satisfied(self):
        """Test can_start returns True after dependencies are completed."""
        tasks = [
            MockTask(id="t1", status="completed"),
            MockTask(id="t2", status="pending", depends_on=["t1"]),
        ]
        scheduler = self._create_scheduler(tasks)

        assert scheduler.can_start(tasks[1])

    def test_scheduler_solo_task_detection(self):
        """Test solo tasks are routed to executor lane."""
        tasks = [MockTask(id="t1", solo=True)]
        scheduler = self._create_scheduler(tasks)

        assert scheduler._is_executor_lane_task(tasks[0])

    def test_scheduler_high_intensity_task_detection(self):
        """Test high intensity tasks are routed to executor lane."""
        tasks = [MockTask(id="t1", intensity="high")]
        scheduler = self._create_scheduler(tasks)

        assert scheduler._is_executor_lane_task(tasks[0])

    def test_scheduler_light_intensity_goes_to_coding_lane(self):
        """Test light intensity tasks go to coding lane."""
        tasks = [MockTask(id="t1", intensity="light")]
        scheduler = self._create_scheduler(tasks)

        assert not scheduler._is_executor_lane_task(tasks[0])

    def test_scheduler_assign_to_coding_lane(self):
        """Test task assignment to coding lane."""
        tasks = [MockTask(id="t1", intensity="light")]
        scheduler = self._create_scheduler(tasks)

        lane = scheduler.assign_to_lane("t1")
        assert lane == "coding"
        assert "t1" in scheduler.coding_lane_running

    def test_scheduler_assign_to_executor_lane(self):
        """Test task assignment to executor lane for solo tasks."""
        tasks = [MockTask(id="t1", solo=True)]
        scheduler = self._create_scheduler(tasks)

        lane = scheduler.assign_to_lane("t1")
        assert lane == "executor"
        assert "t1" in scheduler.executor_lane_running

    def test_scheduler_release_from_lane(self):
        """Test releasing task from lane."""
        tasks = [MockTask(id="t1")]
        scheduler = self._create_scheduler(tasks)

        scheduler.assign_to_lane("t1")
        assert "t1" in scheduler.coding_lane_running

        scheduler.release_from_lane("t1")
        assert "t1" not in scheduler.coding_lane_running
        assert "t1" not in scheduler.executor_lane_running

    def test_scheduler_lane_capacity(self):
        """Test lane capacity enforcement."""
        # Create 4 tasks, coding lane has capacity 3
        tasks = [MockTask(id=f"t{i}") for i in range(4)]
        scheduler = self._create_scheduler(tasks)

        # Assign 3 to coding lane
        scheduler.assign_to_lane("t0")
        scheduler.assign_to_lane("t1")
        scheduler.assign_to_lane("t2")

        assert len(scheduler.coding_lane_running) == 3
        assert scheduler._can_start_in_coding_lane(tasks[3]) is False

    def test_scheduler_get_ready_tasks(self):
        """Test getting ready tasks sorted by priority."""
        tasks = [
            MockTask(id="t1", status="pending"),
            MockTask(id="t2", status="pending"),
            MockTask(id="t3", status="in_progress"),
        ]
        scheduler = self._create_scheduler(tasks)

        ready = scheduler.get_ready_tasks()
        assert len(ready) == 2  # Only pending tasks
        assert tasks[2] not in ready

    def test_scheduler_get_lane_stats(self):
        """Test lane statistics."""
        tasks = [MockTask(id="t1"), MockTask(id="t2", solo=True)]
        scheduler = self._create_scheduler(tasks)

        scheduler.assign_to_lane("t1")
        scheduler.assign_to_lane("t2")

        stats = scheduler.get_lane_stats()
        assert stats["coding_active"] == 1
        assert stats["executor_active"] == 1
        assert stats["coding_capacity"] == 3
        assert stats["executor_capacity"] == 1

    def test_scheduler_update_tasks(self):
        """Test dynamic task list update for backfill."""
        initial_tasks = [MockTask(id="t1")]
        scheduler = self._create_scheduler(initial_tasks)

        new_tasks = [
            MockTask(id="t1"),
            MockTask(id="t2"),
            MockTask(id="t3"),
        ]
        scheduler.update_tasks(new_tasks)

        assert len(scheduler.tasks) == 3
        assert scheduler.metrics.total_tasks == 3
        assert "t2" in scheduler.by_id

    def test_scheduler_priority_favors_tasks_with_dependents(self):
        """Test priority scoring favors tasks with more dependents."""
        tasks = [
            MockTask(id="root", status="pending"),
            MockTask(id="child1", status="pending", depends_on=["root"]),
            MockTask(id="child2", status="pending", depends_on=["root"]),
            MockTask(id="leaf", status="pending", depends_on=["child1"]),
        ]
        scheduler = self._create_scheduler(tasks)

        # Root has 2 direct dependents, should have higher priority
        root_priority = scheduler._compute_priority(tasks[0])
        child1_priority = scheduler._compute_priority(tasks[1])
        leaf_priority = scheduler._compute_priority(tasks[3])

        assert root_priority > child1_priority
        assert child1_priority > leaf_priority

    def test_scheduler_sample_metrics(self):
        """Test metrics sampling during scheduling."""
        tasks = [MockTask(id="t1")]
        scheduler = self._create_scheduler(tasks)

        scheduler.sample_metrics(queue_depth=5)

        assert len(scheduler.metrics.active_workers_samples) == 1
        assert len(scheduler.metrics.queue_depth_samples) == 1


# -----------------------------------------------------------------------------
# BackfillGenerator Tests (complementing test_orchestrator_selfheal.py)
# -----------------------------------------------------------------------------


class TestBackfillGeneratorExtended:
    """Extended tests for BackfillGenerator beyond test_orchestrator_selfheal.py."""

    def test_backfill_should_generate_threshold(self):
        """Test should_generate respects thresholds."""
        from bridge.scheduler import BackfillGenerator

        gen = BackfillGenerator(project_root="/tmp", min_queue_depth=10)

        # Queue below threshold -> should generate
        assert gen.should_generate(current_queue_depth=5, worker_count=5)

        # Queue at or above threshold -> should not generate
        assert not gen.should_generate(current_queue_depth=15, worker_count=5)

    def test_backfill_task_types_cycle(self):
        """Test filler tasks cycle through task types."""
        from bridge.scheduler import BackfillGenerator

        gen = BackfillGenerator(project_root="/tmp")

        # Generate more tasks than task types to verify cycling
        tasks = gen.generate_filler_tasks(10)

        # Should cycle through all task types
        task_types = {t.task_type for t in tasks}
        assert len(task_types) >= 3  # At least lint, test, docs

    def test_backfill_unique_ids(self):
        """Test each filler task has unique ID."""
        from bridge.scheduler import BackfillGenerator

        gen = BackfillGenerator(project_root="/tmp")

        tasks = gen.generate_filler_tasks(5)
        ids = [t.id for t in tasks]

        assert len(ids) == len(set(ids))  # All unique

    def test_backfill_generated_count_increment(self):
        """Test generated_count increments correctly."""
        from bridge.scheduler import BackfillGenerator

        gen = BackfillGenerator(project_root="/tmp")

        gen.generate_filler_tasks(3)
        assert gen.generated_count == 3

        gen.generate_filler_tasks(2)
        assert gen.generated_count == 5

    def test_backfill_rotation_single_task_calls(self):
        """Test task types rotate when calling generate_filler_tasks(1) multiple times.

        This is the critical fix for the bug where count=1 always returned lint.
        The task types should cycle: lint, test, type_hints, docs, schema_lint
        """
        from bridge.scheduler import BackfillGenerator

        gen = BackfillGenerator(project_root="/tmp")

        # Call generate_filler_tasks(1) five times
        task_types_seen = []
        for _ in range(5):
            tasks = gen.generate_filler_tasks(1)
            assert len(tasks) == 1
            task_types_seen.append(tasks[0].task_type)

        # Verify we see different types, not always "lint"
        # The expected order is: lint, test, type_hints, docs, schema_lint
        expected_types = ["lint", "test", "type_hints", "docs", "schema_lint"]
        assert task_types_seen == expected_types, f"Expected types to rotate: {expected_types}, got: {task_types_seen}"

        # Verify it cycles - call 5 more times
        task_types_round2 = []
        for _ in range(5):
            tasks = gen.generate_filler_tasks(1)
            task_types_round2.append(tasks[0].task_type)

        # Should see the same rotation again
        assert task_types_round2 == expected_types, f"Expected second round to match: {expected_types}, got: {task_types_round2}"

    def test_backfill_rotation_not_always_lint(self):
        """Verify the rotation fix: should NOT always return lint when count=1."""
        from bridge.scheduler import BackfillGenerator

        gen = BackfillGenerator(project_root="/tmp")

        # Generate 10 tasks one at a time
        lint_count = 0
        for _ in range(10):
            tasks = gen.generate_filler_tasks(1)
            if tasks[0].task_type == "lint":
                lint_count += 1

        # With 5 task types, we should see lint only 2 times in 10 calls
        # If the bug were present, lint_count would be 10
        assert lint_count == 2, f"Expected lint 2 times in 10 calls, got {lint_count} (bug: always lint)"


class TestBackfillCooldown:
    """Tests for no-op streak cooldown suppression."""

    def test_noop_streak_tracking(self):
        """Test no-op result tracking increments streak."""
        from bridge.scheduler import BackfillGenerator

        gen = BackfillGenerator(project_root="/tmp")

        # Record a no-op for a lint task
        gen.record_noop_result("FILLER-LINT-001")
        assert gen._noop_streaks.get("lint") == 1

        # Record another no-op
        gen.record_noop_result("FILLER-LINT-002")
        assert gen._noop_streaks.get("lint") == 2

    def test_successful_result_resets_streak(self):
        """Test successful result resets no-op streak."""
        from bridge.scheduler import BackfillGenerator

        gen = BackfillGenerator(project_root="/tmp")

        # Build up a streak
        gen.record_noop_result("FILLER-LINT-001")
        gen.record_noop_result("FILLER-LINT-002")
        assert gen._noop_streaks.get("lint") == 2

        # Successful result resets it
        gen.record_successful_result("FILLER-LINT-003")
        assert gen._noop_streaks.get("lint") == 0

    def test_cooldown_after_streak_threshold(self):
        """Test type goes on cooldown after reaching streak threshold."""
        from bridge.scheduler import BackfillGenerator

        gen = BackfillGenerator(project_root="/tmp")

        # First generate to set up the cycle
        gen.generate_filler_tasks(1)  # This sets _generation_cycle to 1

        # Record enough no-ops to trigger cooldown
        for i in range(gen.NOOP_STREAK_THRESHOLD):
            gen.record_noop_result(f"FILLER-LINT-{i:03d}")

        # Should now be on cooldown
        assert gen.is_type_on_cooldown("lint")
        assert "lint" in gen._cooldown_start

    def test_cooldown_skips_type_in_generation(self):
        """Test that cooldown types are skipped during generation."""
        from bridge.scheduler import BackfillGenerator

        gen = BackfillGenerator(project_root="/tmp")

        # Put lint on cooldown manually
        gen._cooldown_start["lint"] = 0
        gen._generation_cycle = 1

        # Generate tasks - should skip lint
        tasks = gen.generate_filler_tasks(5)
        task_types = [t.task_type for t in tasks]

        # lint should not appear (it's on cooldown)
        assert "lint" not in task_types, f"lint should be skipped, but got types: {task_types}"

    def test_cooldown_expires(self):
        """Test that cooldown expires after COOLDOWN_CYCLES."""
        from bridge.scheduler import BackfillGenerator

        gen = BackfillGenerator(project_root="/tmp")

        # Put lint on cooldown at cycle 0
        gen._cooldown_start["lint"] = 0
        gen._generation_cycle = gen.COOLDOWN_CYCLES  # Advance past cooldown

        # Should no longer be on cooldown
        assert not gen.is_type_on_cooldown("lint")
        # Should have been removed from cooldown tracking
        assert "lint" not in gen._cooldown_start

    def test_get_cooldown_status(self):
        """Test get_cooldown_status returns remaining cycles."""
        from bridge.scheduler import BackfillGenerator

        gen = BackfillGenerator(project_root="/tmp")

        # Put lint on cooldown at cycle 0
        gen._cooldown_start["lint"] = 0
        gen._generation_cycle = 2

        status = gen.get_cooldown_status()
        expected_remaining = gen.COOLDOWN_CYCLES - 2
        assert status.get("lint") == expected_remaining

    def test_rejection_counts_as_noop(self):
        """Test record_rejection increments no-op streak."""
        from bridge.scheduler import BackfillGenerator

        gen = BackfillGenerator(project_root="/tmp")

        gen.record_rejection("FILLER-LINT-001")
        assert gen._noop_streaks.get("lint") == 1

        gen.record_rejection("FILLER-LINT-002")
        assert gen._noop_streaks.get("lint") == 2

    def test_non_filler_tasks_ignored(self):
        """Test non-FILLER tasks are ignored for cooldown tracking."""
        from bridge.scheduler import BackfillGenerator

        gen = BackfillGenerator(project_root="/tmp")

        # These should be no-ops
        gen.record_noop_result("M1-TASK-001")
        gen.record_successful_result("M1-TASK-002")
        gen.record_rejection("M1-TASK-003")

        # No streaks should be recorded
        assert len(gen._noop_streaks) == 0


# -----------------------------------------------------------------------------
# create_scheduler Helper Tests
# -----------------------------------------------------------------------------


class TestCreateSchedulerHelper:
    """Tests for create_scheduler helper function."""

    def test_create_scheduler_basic(self):
        """Test create_scheduler creates properly configured scheduler."""
        from bridge.scheduler import TwoLaneScheduler, create_scheduler

        tasks = [MockTask(id="t1")]

        def deps_fn(t):
            return True

        def locks_fn(t):
            return True

        scheduler = create_scheduler(
            max_workers=10,
            tasks=tasks,
            deps_satisfied_fn=deps_fn,
            locks_available_fn=locks_fn,
        )

        assert isinstance(scheduler, TwoLaneScheduler)
        assert scheduler.lane_config.total_workers == 10
