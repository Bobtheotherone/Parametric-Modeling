"""Batch simulation runner for parallel openEMS execution.

This module implements REQ-M2-009: Batch processing for openEMS FDTD
simulations with parallel execution, progress tracking, and convergence
validation.

Features:
- BatchConfig for controlling parallelism and resource limits
- Parallel execution of multiple simulations with process pool
- Progress tracking and status reporting
- Integration with convergence validation for each simulation
- Aggregated result collection and error handling

Hardware-aware throttling respects limits (e.g., 16 cores, 15GB RAM).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from formula_foundry.substrate import canonical_json_dumps, sha256_bytes

from .convergence import (
    ConvergenceConfig,
    ConvergenceReport,
    ConvergenceStatus,
    validate_simulation_convergence,
)
from .geometry import GeometrySpec
from .sim_runner import (
    SimulationError,
    SimulationExecutionError,
    SimulationResult,
    SimulationRunner,
    SimulationTimeoutError,
)
from .spec import SimulationSpec

logger = logging.getLogger(__name__)

# Default hardware limits (conservative)
DEFAULT_MAX_WORKERS = 4
DEFAULT_RAM_LIMIT_GB = 15.0
DEFAULT_CPU_CORES = 16
DEFAULT_RAM_PER_SIM_GB = 4.0
DEFAULT_TIMEOUT_SEC = 3600  # 1 hour per simulation


class SimulationStatus(str, Enum):
    """Status of an individual simulation in a batch."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class BatchConfig:
    """Configuration for batch simulation execution.

    Attributes:
        max_workers: Maximum number of parallel simulations.
        ram_limit_gb: Total RAM limit for all simulations.
        cpu_cores: Total CPU cores available.
        ram_per_sim_gb: Estimated RAM per simulation.
        timeout_per_sim_sec: Timeout for each individual simulation.
        fail_fast: Stop batch on first failure.
        validate_convergence: Run convergence validation after each simulation.
        convergence_config: Configuration for convergence checks.
        retry_failed: Number of retries for failed simulations.
    """

    max_workers: int = DEFAULT_MAX_WORKERS
    ram_limit_gb: float = DEFAULT_RAM_LIMIT_GB
    cpu_cores: int = DEFAULT_CPU_CORES
    ram_per_sim_gb: float = DEFAULT_RAM_PER_SIM_GB
    timeout_per_sim_sec: float = DEFAULT_TIMEOUT_SEC
    fail_fast: bool = False
    validate_convergence: bool = True
    convergence_config: ConvergenceConfig | None = None
    retry_failed: int = 0

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.ram_limit_gb <= 0:
            raise ValueError("ram_limit_gb must be > 0")
        if self.cpu_cores < 1:
            raise ValueError("cpu_cores must be >= 1")
        if self.ram_per_sim_gb <= 0:
            raise ValueError("ram_per_sim_gb must be > 0")
        if self.timeout_per_sim_sec <= 0:
            raise ValueError("timeout_per_sim_sec must be > 0")
        if self.retry_failed < 0:
            raise ValueError("retry_failed must be >= 0")

    @property
    def effective_max_workers(self) -> int:
        """Calculate effective max workers based on RAM constraints.

        The effective parallelism is limited by:
        - max_workers setting
        - RAM available / RAM per simulation
        """
        ram_limited = int(self.ram_limit_gb / self.ram_per_sim_gb)
        return max(1, min(self.max_workers, ram_limited))


@dataclass(slots=True)
class SimulationJob:
    """A single simulation job in a batch.

    Attributes:
        job_id: Unique identifier for this job.
        spec: Simulation specification.
        geometry: Geometry specification.
        output_dir: Directory for simulation outputs.
        openems_args: Optional arguments for openEMS.
        priority: Job priority (higher = run first).
        tags: Optional tags for filtering/grouping.
    """

    job_id: str
    spec: SimulationSpec
    geometry: GeometrySpec
    output_dir: Path
    openems_args: tuple[str, ...] | None = None
    priority: int = 0
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SimulationJobResult:
    """Result of a single simulation job.

    Attributes:
        job_id: Job identifier.
        status: Final status of the job.
        result: SimulationResult if successful.
        convergence_report: Convergence report if validation was run.
        error: Error message if failed.
        execution_time_sec: Total execution time.
        retry_count: Number of retries attempted.
        started_at: Start timestamp.
        completed_at: Completion timestamp.
    """

    job_id: str
    status: SimulationStatus
    result: SimulationResult | None = None
    convergence_report: ConvergenceReport | None = None
    error: str | None = None
    execution_time_sec: float | None = None
    retry_count: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def passed(self) -> bool:
        """Whether job completed successfully with passing convergence."""
        if self.status != SimulationStatus.COMPLETED:
            return False
        if self.convergence_report is not None:
            return self.convergence_report.all_passed
        return True


@dataclass(slots=True)
class BatchProgress:
    """Progress tracking for batch execution.

    Attributes:
        total: Total number of jobs.
        pending: Number of pending jobs.
        running: Number of currently running jobs.
        completed: Number of completed jobs.
        failed: Number of failed jobs.
        skipped: Number of skipped jobs.
        start_time: Batch start time.
        current_jobs: List of currently running job IDs.
    """

    total: int
    pending: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: float | None = None
    current_jobs: list[str] = field(default_factory=list)

    @property
    def finished(self) -> int:
        """Number of finished jobs (completed + failed + skipped)."""
        return self.completed + self.failed + self.skipped

    @property
    def percent_complete(self) -> float:
        """Percentage of jobs finished."""
        if self.total == 0:
            return 100.0
        return 100.0 * self.finished / self.total

    @property
    def elapsed_sec(self) -> float:
        """Elapsed time since batch started."""
        if self.start_time is None:
            return 0.0
        return time.monotonic() - self.start_time

    @property
    def estimated_remaining_sec(self) -> float | None:
        """Estimated remaining time based on current progress."""
        if self.finished == 0:
            return None
        avg_time_per_job = self.elapsed_sec / self.finished
        remaining_jobs = self.total - self.finished
        return avg_time_per_job * remaining_jobs


@dataclass(slots=True)
class BatchResult:
    """Aggregated result of a batch simulation run.

    Attributes:
        config: Batch configuration used.
        jobs: List of individual job results.
        total_time_sec: Total batch execution time.
        n_completed: Number of successfully completed simulations.
        n_failed: Number of failed simulations.
        n_skipped: Number of skipped simulations.
        n_convergence_passed: Number passing convergence validation.
        n_convergence_failed: Number failing convergence validation.
        canonical_hash: SHA256 hash of the batch result.
    """

    config: BatchConfig
    jobs: list[SimulationJobResult]
    total_time_sec: float
    n_completed: int = 0
    n_failed: int = 0
    n_skipped: int = 0
    n_convergence_passed: int = 0
    n_convergence_failed: int = 0
    canonical_hash: str = ""

    def __post_init__(self) -> None:
        # Recompute counts and hash
        self.n_completed = sum(1 for j in self.jobs if j.status == SimulationStatus.COMPLETED)
        self.n_failed = sum(
            1 for j in self.jobs if j.status in (SimulationStatus.FAILED, SimulationStatus.TIMEOUT)
        )
        self.n_skipped = sum(1 for j in self.jobs if j.status == SimulationStatus.SKIPPED)

        # Count convergence results
        for job in self.jobs:
            if job.convergence_report is not None:
                if job.convergence_report.all_passed:
                    self.n_convergence_passed += 1
                else:
                    self.n_convergence_failed += 1

        # Compute canonical hash
        if not self.canonical_hash:
            self.canonical_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute canonical hash of the batch result."""
        payload = {
            "n_jobs": len(self.jobs),
            "n_completed": self.n_completed,
            "n_failed": self.n_failed,
            "n_skipped": self.n_skipped,
            "job_hashes": sorted(
                (j.job_id, j.result.simulation_hash if j.result else "")
                for j in self.jobs
            ),
        }
        return sha256_bytes(canonical_json_dumps(payload).encode("utf-8"))

    @property
    def all_passed(self) -> bool:
        """Whether all jobs completed successfully with passing convergence."""
        return all(job.passed for job in self.jobs)

    @property
    def success_rate(self) -> float:
        """Percentage of jobs that completed successfully."""
        if len(self.jobs) == 0:
            return 0.0
        return 100.0 * self.n_completed / len(self.jobs)

    def get_job(self, job_id: str) -> SimulationJobResult | None:
        """Get a job result by ID."""
        for job in self.jobs:
            if job.job_id == job_id:
                return job
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_jobs": len(self.jobs),
            "n_completed": self.n_completed,
            "n_failed": self.n_failed,
            "n_skipped": self.n_skipped,
            "n_convergence_passed": self.n_convergence_passed,
            "n_convergence_failed": self.n_convergence_failed,
            "total_time_sec": self.total_time_sec,
            "canonical_hash": self.canonical_hash,
            "all_passed": self.all_passed,
            "success_rate": self.success_rate,
            "jobs": [
                {
                    "job_id": j.job_id,
                    "status": j.status.value,
                    "simulation_hash": j.result.simulation_hash if j.result else None,
                    "convergence_passed": j.passed,
                    "error": j.error,
                    "execution_time_sec": j.execution_time_sec,
                }
                for j in self.jobs
            ],
        }


# Type for progress callback
ProgressCallback = Callable[[BatchProgress], None]


class BatchSimulationRunner:
    """Runner for executing batches of openEMS simulations in parallel.

    This class manages parallel execution of multiple simulations with:
    - Resource-aware parallelism (respects RAM/CPU limits)
    - Progress tracking and callbacks
    - Convergence validation for each simulation
    - Error handling and optional retries
    - Aggregated result collection

    Example:
        >>> config = BatchConfig(max_workers=4, ram_limit_gb=15.0)
        >>> runner = BatchSimulationRunner(sim_runner, config)
        >>> jobs = [
        ...     SimulationJob("job1", spec1, geometry1, Path("./out1")),
        ...     SimulationJob("job2", spec2, geometry2, Path("./out2")),
        ... ]
        >>> result = runner.run(jobs)
        >>> print(f"Completed: {result.n_completed}/{len(jobs)}")
    """

    def __init__(
        self,
        sim_runner: SimulationRunner,
        config: BatchConfig | None = None,
    ) -> None:
        """Initialize the batch runner.

        Args:
            sim_runner: SimulationRunner instance for executing individual sims.
            config: Batch configuration (uses defaults if None).
        """
        self.sim_runner = sim_runner
        self.config = config or BatchConfig()
        self._progress: BatchProgress | None = None
        self._progress_lock = threading.Lock()
        self._progress_callbacks: list[ProgressCallback] = []
        self._stop_requested = False

    def add_progress_callback(self, callback: ProgressCallback) -> None:
        """Add a callback to be called on progress updates.

        Args:
            callback: Function taking BatchProgress as argument.
        """
        self._progress_callbacks.append(callback)

    def remove_progress_callback(self, callback: ProgressCallback) -> None:
        """Remove a progress callback.

        Args:
            callback: Previously registered callback to remove.
        """
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)

    def request_stop(self) -> None:
        """Request the batch to stop after current jobs complete."""
        self._stop_requested = True

    def run(
        self,
        jobs: list[SimulationJob],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> BatchResult:
        """Run a batch of simulation jobs.

        Args:
            jobs: List of simulation jobs to execute.
            progress_callback: Optional callback for progress updates.

        Returns:
            BatchResult with aggregated results for all jobs.
        """
        if progress_callback:
            self.add_progress_callback(progress_callback)

        try:
            return self._run_batch(jobs)
        finally:
            if progress_callback:
                self.remove_progress_callback(progress_callback)
            self._stop_requested = False

    def _run_batch(self, jobs: list[SimulationJob]) -> BatchResult:
        """Internal batch execution logic."""
        if not jobs:
            return BatchResult(
                config=self.config,
                jobs=[],
                total_time_sec=0.0,
            )

        # Sort jobs by priority (higher first)
        sorted_jobs = sorted(jobs, key=lambda j: j.priority, reverse=True)

        # Initialize progress
        self._progress = BatchProgress(
            total=len(sorted_jobs),
            pending=len(sorted_jobs),
            start_time=time.monotonic(),
        )
        self._notify_progress()

        # Execute jobs
        results: list[SimulationJobResult] = []
        effective_workers = self.config.effective_max_workers

        logger.info(
            "Starting batch of %d simulations with %d workers",
            len(sorted_jobs),
            effective_workers,
        )

        start_time = time.monotonic()

        # Use ThreadPoolExecutor for I/O-bound simulation dispatching
        # (actual openEMS runs in subprocesses managed by SimulationRunner)
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_job = {
                executor.submit(self._run_single_job, job): job
                for job in sorted_jobs
            }

            for future in as_completed(future_to_job):
                if self._stop_requested and self.config.fail_fast:
                    # Cancel remaining futures
                    for f in future_to_job:
                        f.cancel()
                    break

                job = future_to_job[future]
                try:
                    job_result = future.result()
                except Exception as e:
                    logger.exception("Job %s raised unexpected exception", job.job_id)
                    job_result = SimulationJobResult(
                        job_id=job.job_id,
                        status=SimulationStatus.FAILED,
                        error=str(e),
                    )

                results.append(job_result)

                # Update progress
                with self._progress_lock:
                    if self._progress:
                        self._progress.pending -= 1
                        if job_result.status == SimulationStatus.COMPLETED:
                            self._progress.completed += 1
                        elif job_result.status in (
                            SimulationStatus.FAILED,
                            SimulationStatus.TIMEOUT,
                        ):
                            self._progress.failed += 1
                        else:
                            self._progress.skipped += 1

                        if job.job_id in self._progress.current_jobs:
                            self._progress.current_jobs.remove(job.job_id)

                self._notify_progress()

                # Check fail_fast
                if (
                    self.config.fail_fast
                    and job_result.status != SimulationStatus.COMPLETED
                ):
                    logger.warning(
                        "Fail-fast triggered by job %s", job.job_id
                    )
                    self._stop_requested = True

        total_time = time.monotonic() - start_time

        logger.info(
            "Batch completed: %d/%d successful in %.2f seconds",
            sum(1 for r in results if r.status == SimulationStatus.COMPLETED),
            len(results),
            total_time,
        )

        return BatchResult(
            config=self.config,
            jobs=results,
            total_time_sec=total_time,
        )

    def _run_single_job(self, job: SimulationJob) -> SimulationJobResult:
        """Execute a single simulation job with retries."""
        started_at = datetime.now()
        retry_count = 0
        last_error: str | None = None
        was_timeout = False  # Track if the last failure was a timeout

        # Update progress
        with self._progress_lock:
            if self._progress:
                self._progress.running += 1
                self._progress.pending -= 1
                self._progress.current_jobs.append(job.job_id)
        self._notify_progress()

        start_time = time.monotonic()

        while retry_count <= self.config.retry_failed:
            if self._stop_requested:
                return SimulationJobResult(
                    job_id=job.job_id,
                    status=SimulationStatus.SKIPPED,
                    error="Batch stop requested",
                    retry_count=retry_count,
                    started_at=started_at,
                    completed_at=datetime.now(),
                )

            try:
                result = self.sim_runner.run(
                    job.spec,
                    job.geometry,
                    output_dir=job.output_dir,
                    openems_args=list(job.openems_args) if job.openems_args else None,
                    timeout_sec=self.config.timeout_per_sim_sec,
                )

                # Run convergence validation if enabled
                convergence_report: ConvergenceReport | None = None
                if self.config.validate_convergence:
                    try:
                        convergence_report = validate_simulation_convergence(
                            result.outputs_dir,
                            job.spec,
                            simulation_hash=result.simulation_hash,
                            config=self.config.convergence_config,
                        )
                        logger.debug(
                            "Job %s convergence: %s",
                            job.job_id,
                            convergence_report.overall_status.value,
                        )
                    except Exception as e:
                        logger.warning(
                            "Convergence validation failed for %s: %s",
                            job.job_id,
                            e,
                        )

                execution_time = time.monotonic() - start_time

                # Update progress (running -> completed)
                with self._progress_lock:
                    if self._progress:
                        self._progress.running -= 1

                return SimulationJobResult(
                    job_id=job.job_id,
                    status=SimulationStatus.COMPLETED,
                    result=result,
                    convergence_report=convergence_report,
                    execution_time_sec=execution_time,
                    retry_count=retry_count,
                    started_at=started_at,
                    completed_at=datetime.now(),
                )

            except SimulationTimeoutError as e:
                last_error = str(e)
                was_timeout = True
                logger.warning(
                    "Job %s timed out (attempt %d/%d)",
                    job.job_id,
                    retry_count + 1,
                    self.config.retry_failed + 1,
                )

            except SimulationExecutionError as e:
                last_error = str(e)
                was_timeout = False
                logger.warning(
                    "Job %s failed (attempt %d/%d): %s",
                    job.job_id,
                    retry_count + 1,
                    self.config.retry_failed + 1,
                    e,
                )

            except Exception as e:
                last_error = str(e)
                was_timeout = False
                logger.exception(
                    "Job %s unexpected error (attempt %d/%d)",
                    job.job_id,
                    retry_count + 1,
                    self.config.retry_failed + 1,
                )

            retry_count += 1

        execution_time = time.monotonic() - start_time

        # Update progress (running -> failed)
        with self._progress_lock:
            if self._progress:
                self._progress.running -= 1

        # Determine final status based on the exception type
        if was_timeout:
            status = SimulationStatus.TIMEOUT
        else:
            status = SimulationStatus.FAILED

        return SimulationJobResult(
            job_id=job.job_id,
            status=status,
            error=last_error,
            execution_time_sec=execution_time,
            retry_count=retry_count - 1,
            started_at=started_at,
            completed_at=datetime.now(),
        )

    def _notify_progress(self) -> None:
        """Notify all registered progress callbacks."""
        if self._progress is None:
            return
        for callback in self._progress_callbacks:
            try:
                callback(self._progress)
            except Exception:
                logger.exception("Progress callback raised exception")


# =============================================================================
# Batch Result I/O
# =============================================================================


def write_batch_result(result: BatchResult, output_path: Path) -> None:
    """Write batch result to JSON file.

    Args:
        result: BatchResult to write.
        output_path: Path for output JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json_dumps(result.to_dict())
    output_path.write_text(f"{text}\n", encoding="utf-8")


def load_batch_result_summary(json_path: Path) -> dict[str, Any]:
    """Load batch result summary from JSON file.

    Args:
        json_path: Path to batch result JSON.

    Returns:
        Dictionary with batch result summary.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If format is invalid.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Batch result file not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Batch result must be a JSON object")

    return data


# =============================================================================
# Utility Functions
# =============================================================================


def estimate_batch_time(
    n_jobs: int,
    config: BatchConfig,
    avg_sim_time_sec: float = 300.0,
) -> float:
    """Estimate total batch execution time.

    Args:
        n_jobs: Number of simulation jobs.
        config: Batch configuration.
        avg_sim_time_sec: Average time per simulation.

    Returns:
        Estimated total time in seconds.
    """
    effective_workers = config.effective_max_workers
    batches = (n_jobs + effective_workers - 1) // effective_workers
    return batches * avg_sim_time_sec


def create_batch_jobs(
    specs: list[tuple[SimulationSpec, GeometrySpec]],
    output_base: Path,
    *,
    prefix: str = "sim",
    openems_args: tuple[str, ...] | None = None,
) -> list[SimulationJob]:
    """Create batch jobs from spec/geometry pairs.

    Convenience function to create SimulationJob objects from a list of
    (spec, geometry) tuples.

    Args:
        specs: List of (SimulationSpec, GeometrySpec) tuples.
        output_base: Base directory for simulation outputs.
        prefix: Prefix for job IDs.
        openems_args: Common openEMS arguments for all jobs.

    Returns:
        List of SimulationJob objects.
    """
    jobs = []
    for idx, (spec, geometry) in enumerate(specs):
        job_id = f"{prefix}_{idx:04d}"
        output_dir = output_base / job_id
        jobs.append(
            SimulationJob(
                job_id=job_id,
                spec=spec,
                geometry=geometry,
                output_dir=output_dir,
                openems_args=openems_args,
            )
        )
    return jobs
