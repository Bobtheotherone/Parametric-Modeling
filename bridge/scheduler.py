#!/usr/bin/env python3
"""Two-lane scheduler for parallel task execution.

This module implements an event-driven scheduler that maintains high worker
utilization (target: 10 concurrent workers) while respecting constraints.

Key features:
- Two-lane execution: coding lane (n-1 workers) + executor lane (1 worker)
- Priority scoring based on dependency graph analysis
- Backfilling with safe tasks when queue depth is low
- Comprehensive instrumentation for debugging utilization issues
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


class SchedulableTask(Protocol):
    """Protocol for tasks that can be scheduled."""
    id: str
    status: str
    solo: bool
    intensity: str
    locks: list[str]
    touched_paths: list[str]
    depends_on: list[str]


@dataclass
class SchedulerMetrics:
    """Metrics tracked by the scheduler for instrumentation."""
    start_time: float = field(default_factory=time.monotonic)
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    blocked_tasks: int = 0

    # Utilization tracking
    active_workers_samples: list[tuple[float, int]] = field(default_factory=list)
    queue_depth_samples: list[tuple[float, int]] = field(default_factory=list)

    # Stall tracking
    stall_events: list[dict[str, Any]] = field(default_factory=list)
    retry_counts: dict[str, int] = field(default_factory=dict)
    json_repair_counts: dict[str, int] = field(default_factory=dict)

    # Lane utilization
    coding_lane_samples: list[tuple[float, int]] = field(default_factory=list)
    executor_lane_samples: list[tuple[float, int]] = field(default_factory=list)

    def sample(
        self,
        active_workers: int,
        queue_depth: int,
        coding_lane_active: int,
        executor_lane_active: int,
    ) -> None:
        """Record a utilization sample."""
        now = time.monotonic()
        self.active_workers_samples.append((now, active_workers))
        self.queue_depth_samples.append((now, queue_depth))
        self.coding_lane_samples.append((now, coding_lane_active))
        self.executor_lane_samples.append((now, executor_lane_active))

    def record_stall(self, reason: str, task_ids: list[str]) -> None:
        """Record a stall event."""
        self.stall_events.append({
            "time": time.monotonic(),
            "reason": reason,
            "task_ids": task_ids,
        })

    def record_retry(self, task_id: str) -> None:
        """Record a task retry."""
        self.retry_counts[task_id] = self.retry_counts.get(task_id, 0) + 1

    def record_json_repair(self, task_id: str) -> None:
        """Record a JSON repair attempt."""
        self.json_repair_counts[task_id] = self.json_repair_counts.get(task_id, 0) + 1

    def get_average_utilization(self) -> float:
        """Get average worker utilization over the run."""
        if not self.active_workers_samples:
            return 0.0
        total = sum(count for _, count in self.active_workers_samples)
        return total / len(self.active_workers_samples)

    def to_dict(self) -> dict[str, Any]:
        """Export metrics to dictionary."""
        elapsed = time.monotonic() - self.start_time
        return {
            "elapsed_seconds": elapsed,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "blocked_tasks": self.blocked_tasks,
            "average_utilization": self.get_average_utilization(),
            "stall_count": len(self.stall_events),
            "total_retries": sum(self.retry_counts.values()),
            "total_json_repairs": sum(self.json_repair_counts.values()),
        }


@dataclass
class LaneConfig:
    """Configuration for two-lane execution."""
    coding_lane_size: int  # Number of workers for coding tasks
    executor_lane_size: int = 1  # Number of workers for heavy/executor tasks

    @classmethod
    def from_max_workers(cls, max_workers: int) -> LaneConfig:
        """Create lane config from total max workers."""
        # Reserve 1 worker for executor lane, rest for coding
        executor_size = 1
        coding_size = max(1, max_workers - executor_size)
        return cls(coding_lane_size=coding_size, executor_lane_size=executor_size)

    @property
    def total_workers(self) -> int:
        return self.coding_lane_size + self.executor_lane_size


class TwoLaneScheduler:
    """Two-lane scheduler for parallel task execution.

    Implements a two-lane model:
    - Coding lane: Most workers, handles regular implementation tasks
    - Executor lane: 1 worker, handles heavy/solo/benchmark tasks

    This ensures heavy tasks don't block coding progress.
    """

    def __init__(
        self,
        lane_config: LaneConfig,
        tasks: list[SchedulableTask],
        deps_satisfied_fn: Callable[[SchedulableTask], bool],
        locks_available_fn: Callable[[SchedulableTask], bool],
    ):
        self.lane_config = lane_config
        self.tasks = tasks
        self.by_id = {t.id: t for t in tasks}
        self.deps_satisfied = deps_satisfied_fn
        self.locks_available = locks_available_fn
        self.metrics = SchedulerMetrics(total_tasks=len(tasks))

        # Lane state
        self.coding_lane_running: set[str] = set()
        self.executor_lane_running: set[str] = set()

        # Priority cache
        self._priority_cache: dict[str, float] = {}
        self._dependents_cache: dict[str, set[str]] = {}
        self._build_dependency_graph()

    def _build_dependency_graph(self) -> None:
        """Build reverse dependency graph for priority scoring."""
        self._dependents_cache = {t.id: set() for t in self.tasks}
        for t in self.tasks:
            for dep_id in t.depends_on:
                if dep_id in self._dependents_cache:
                    self._dependents_cache[dep_id].add(t.id)

    def _compute_priority(self, task: SchedulableTask) -> float:
        """Compute priority score for a task.

        Higher score = higher priority. Factors:
        1. Number of dependents (unblocks more work)
        2. On critical path (chain length to root)
        3. Intensity (defer heavy tasks unless necessary)
        """
        if task.id in self._priority_cache:
            return self._priority_cache[task.id]

        # Base score: number of tasks this unblocks
        direct_dependents = len(self._dependents_cache.get(task.id, set()))

        # Transitive dependents (weighted less)
        transitive = self._count_transitive_dependents(task.id)
        dependent_score = direct_dependents * 10 + transitive

        # Intensity penalty for executor lane tasks
        intensity_penalty = 0
        if self._is_executor_lane_task(task):
            intensity_penalty = 5  # Slight preference for coding lane tasks

        # Calculate final priority
        priority = dependent_score - intensity_penalty
        self._priority_cache[task.id] = priority
        return priority

    def _count_transitive_dependents(self, task_id: str, visited: set | None = None) -> int:
        """Count transitive dependents for priority scoring."""
        if visited is None:
            visited = set()
        if task_id in visited:
            return 0
        visited.add(task_id)

        dependents = self._dependents_cache.get(task_id, set())
        count = len(dependents)
        for dep in dependents:
            count += self._count_transitive_dependents(dep, visited)
        return count

    def _is_executor_lane_task(self, task: SchedulableTask) -> bool:
        """Check if task should run in executor lane."""
        # Solo tasks always go to executor lane
        return task.solo or task.intensity == "high"

    def _can_start_in_coding_lane(self, task: SchedulableTask) -> bool:
        """Check if task can start in coding lane."""
        return (
            not self._is_executor_lane_task(task)
            and len(self.coding_lane_running) < self.lane_config.coding_lane_size
        )

    def _can_start_in_executor_lane(self, task: SchedulableTask) -> bool:
        """Check if task can start in executor lane."""
        # Solo tasks require empty executor lane AND empty coding lane
        if task.solo and (self.coding_lane_running or self.executor_lane_running):
            return False
        return len(self.executor_lane_running) < self.lane_config.executor_lane_size

    def can_start(self, task: SchedulableTask) -> bool:
        """Check if a task can start now."""
        if task.status != "pending":
            return False
        if not self.deps_satisfied(task):
            return False
        if not self.locks_available(task):
            return False

        # Check lane availability
        if self._is_executor_lane_task(task):
            return self._can_start_in_executor_lane(task)
        else:
            # Coding tasks can also overflow to executor lane if coding lane is full
            if self._can_start_in_coding_lane(task):
                return True
            # Allow coding tasks to use executor lane when coding lane is full
            # but only if it's not a solo situation
            return not task.solo and self._can_start_in_executor_lane(task)

    def get_ready_tasks(self) -> list[SchedulableTask]:
        """Get all tasks that can start now, sorted by priority."""
        ready = [t for t in self.tasks if self.can_start(t)]
        # Sort by priority (highest first)
        ready.sort(key=lambda t: self._compute_priority(t), reverse=True)
        return ready

    def get_ready_tasks_by_lane(self) -> tuple[list[SchedulableTask], list[SchedulableTask]]:
        """Get ready tasks separated by lane.

        Returns:
            Tuple of (coding_lane_tasks, executor_lane_tasks)
        """
        coding = []
        executor = []

        for task in self.get_ready_tasks():
            if self._is_executor_lane_task(task):
                executor.append(task)
            else:
                coding.append(task)

        return coding, executor

    def assign_to_lane(self, task_id: str) -> str:
        """Assign a task to a lane when it starts.

        Returns:
            Lane name ("coding" or "executor")
        """
        task = self.by_id.get(task_id)
        if not task:
            return "coding"

        if self._is_executor_lane_task(task):
            self.executor_lane_running.add(task_id)
            return "executor"
        elif len(self.coding_lane_running) < self.lane_config.coding_lane_size:
            self.coding_lane_running.add(task_id)
            return "coding"
        else:
            # Overflow to executor lane
            self.executor_lane_running.add(task_id)
            return "executor"

    def release_from_lane(self, task_id: str) -> None:
        """Release a task from its lane when it completes."""
        self.coding_lane_running.discard(task_id)
        self.executor_lane_running.discard(task_id)

    def get_lane_stats(self) -> dict[str, int]:
        """Get current lane utilization stats."""
        return {
            "coding_active": len(self.coding_lane_running),
            "coding_capacity": self.lane_config.coding_lane_size,
            "executor_active": len(self.executor_lane_running),
            "executor_capacity": self.lane_config.executor_lane_size,
        }

    def sample_metrics(self, queue_depth: int) -> None:
        """Record current utilization metrics."""
        self.metrics.sample(
            active_workers=len(self.coding_lane_running) + len(self.executor_lane_running),
            queue_depth=queue_depth,
            coding_lane_active=len(self.coding_lane_running),
            executor_lane_active=len(self.executor_lane_running),
        )


@dataclass
class FillerTask:
    """A filler task generated for backfilling."""
    id: str
    title: str
    description: str
    task_type: str  # "lint", "docs", "test", "type_hints", "schema_lint"
    priority: int = 0


class BackfillGenerator:
    """Generates filler tasks when queue depth is low.

    Filler tasks are safe, always-beneficial work that keeps workers busy:
    - Lint fixes
    - Documentation improvements
    - Unit test additions
    - Type hint additions
    - Schema validation fixes
    """

    TASK_TYPES = [
        ("lint", "Fix linting issues", "Run ruff check and fix any issues"),
        ("test", "Add unit tests", "Add missing unit tests for uncovered code"),
        ("type_hints", "Add type hints", "Add type annotations to untyped functions"),
        ("docs", "Improve documentation", "Add or improve docstrings"),
        ("schema_lint", "Fix schema issues", "Validate and fix JSON schema issues"),
    ]

    def __init__(self, project_root: str, min_queue_depth: int = 10):
        self.project_root = project_root
        self.min_queue_depth = min_queue_depth
        self.generated_count = 0

    def should_generate(self, current_queue_depth: int, worker_count: int) -> bool:
        """Check if backfill tasks should be generated."""
        target_depth = worker_count * 2
        return current_queue_depth < min(target_depth, self.min_queue_depth)

    def generate_filler_tasks(self, count: int) -> list[FillerTask]:
        """Generate filler tasks.

        Args:
            count: Number of filler tasks to generate

        Returns:
            List of FillerTask objects
        """
        tasks = []
        for i in range(count):
            task_type, title, description = self.TASK_TYPES[i % len(self.TASK_TYPES)]
            self.generated_count += 1
            task_id = f"FILLER-{task_type.upper()}-{self.generated_count:03d}"
            tasks.append(FillerTask(
                id=task_id,
                title=title,
                description=description,
                task_type=task_type,
                priority=-10,  # Low priority so real work takes precedence
            ))
        return tasks


def create_scheduler(
    max_workers: int,
    tasks: list,
    deps_satisfied_fn: Callable,
    locks_available_fn: Callable,
) -> TwoLaneScheduler:
    """Create a two-lane scheduler with default configuration.

    Args:
        max_workers: Maximum number of workers
        tasks: List of tasks to schedule
        deps_satisfied_fn: Function to check if dependencies are satisfied
        locks_available_fn: Function to check if locks are available

    Returns:
        Configured TwoLaneScheduler instance
    """
    lane_config = LaneConfig.from_max_workers(max_workers)
    return TwoLaneScheduler(
        lane_config=lane_config,
        tasks=tasks,
        deps_satisfied_fn=deps_satisfied_fn,
        locks_available_fn=locks_available_fn,
    )
