"""GPU-accelerated batch runner for openEMS FDTD simulations.

This module implements REQ-M2-010: GPU batching for openEMS FDTD solves with:
- VRAM monitoring and automatic batch sizing
- Fallback to sequential execution on OOM
- GPU utilization metrics tracking
- Multi-GPU support with device affinity

The GPU batch runner extends the standard batch runner with GPU-aware
resource management to maximize throughput while avoiding out-of-memory errors.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from formula_foundry.substrate import canonical_json_dumps, sha256_bytes

from .batch_runner import (
    BatchConfig,
    BatchProgress,
    BatchResult,
    BatchSimulationRunner,
    ProgressCallback,
    SimulationJob,
    SimulationJobResult,
    SimulationStatus,
)
from .convergence import ConvergenceConfig, ConvergenceReport, validate_simulation_convergence
from .manifest import GPUDeviceInfo
from .sim_runner import (
    SimulationExecutionError,
    SimulationResult,
    SimulationRunner,
    SimulationTimeoutError,
)
from .spec import SimulationSpec

logger = logging.getLogger(__name__)

# Default GPU resource limits
DEFAULT_VRAM_LIMIT_MB = 8192  # 8GB default
DEFAULT_VRAM_PER_SIM_MB = 2048  # 2GB per simulation estimate
DEFAULT_GPU_MEMORY_FRACTION = 0.8  # Use 80% of available VRAM
DEFAULT_UTILIZATION_SAMPLE_INTERVAL_SEC = 1.0
OOM_RETRY_DELAY_SEC = 5.0  # Delay before retrying after OOM


class GPUBatchMode(str, Enum):
    """GPU batching execution mode."""

    AUTO = "auto"  # Automatically detect and use GPU if available
    FORCE_GPU = "force_gpu"  # Require GPU, fail if unavailable
    FORCE_CPU = "force_cpu"  # Use CPU only, skip GPU
    HYBRID = "hybrid"  # Use both GPU and CPU based on availability


class GPUStatus(str, Enum):
    """Status of a GPU device."""

    AVAILABLE = "available"  # GPU is ready for use
    BUSY = "busy"  # GPU is running a simulation
    OOM = "oom"  # GPU ran out of memory
    UNAVAILABLE = "unavailable"  # GPU is not accessible
    FAILED = "failed"  # GPU encountered an error


@dataclass(frozen=True, slots=True)
class GPUDeviceState:
    """Current state of a GPU device.

    Attributes:
        device_id: CUDA device ID.
        device_name: GPU device name.
        status: Current device status.
        total_memory_mb: Total VRAM in MB.
        free_memory_mb: Available VRAM in MB.
        used_memory_mb: Used VRAM in MB.
        utilization_percent: GPU compute utilization (0-100).
        temperature_c: GPU temperature in Celsius (if available).
        power_usage_w: Power usage in Watts (if available).
        active_jobs: Number of jobs currently running on this GPU.
        last_updated: Timestamp of last status update.
    """

    device_id: int
    device_name: str
    status: GPUStatus = GPUStatus.AVAILABLE
    total_memory_mb: int = 0
    free_memory_mb: int = 0
    used_memory_mb: int = 0
    utilization_percent: float = 0.0
    temperature_c: float | None = None
    power_usage_w: float | None = None
    active_jobs: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "status": self.status.value,
            "total_memory_mb": self.total_memory_mb,
            "free_memory_mb": self.free_memory_mb,
            "used_memory_mb": self.used_memory_mb,
            "utilization_percent": self.utilization_percent,
            "active_jobs": self.active_jobs,
            "last_updated": self.last_updated.isoformat(),
        }
        if self.temperature_c is not None:
            result["temperature_c"] = self.temperature_c
        if self.power_usage_w is not None:
            result["power_usage_w"] = self.power_usage_w
        return result

    def to_gpu_device_info(self) -> GPUDeviceInfo:
        """Convert to manifest GPUDeviceInfo."""
        return GPUDeviceInfo(
            device_id=self.device_id,
            device_name=self.device_name,
            memory_total_mb=self.total_memory_mb,
        )

    def can_allocate(self, vram_mb: int) -> bool:
        """Check if GPU has enough free VRAM for allocation."""
        return (
            self.status == GPUStatus.AVAILABLE
            and self.free_memory_mb >= vram_mb
        )


@dataclass(frozen=True, slots=True)
class GPUBatchConfig:
    """Configuration for GPU-accelerated batch execution.

    Attributes:
        mode: GPU batching mode (auto, force_gpu, force_cpu, hybrid).
        device_ids: List of CUDA device IDs to use (None = auto-detect all).
        vram_limit_mb: Maximum VRAM to use per GPU.
        vram_per_sim_mb: Estimated VRAM per simulation.
        gpu_memory_fraction: Fraction of GPU memory to use (0.1-1.0).
        max_sims_per_gpu: Maximum concurrent simulations per GPU.
        fallback_to_cpu: Whether to fall back to CPU on GPU failure.
        oom_retry_count: Number of retries after OOM before fallback.
        oom_retry_delay_sec: Delay between OOM retries.
        utilization_sample_interval_sec: Interval for GPU utilization sampling.
        track_utilization: Whether to track GPU utilization metrics.
    """

    mode: GPUBatchMode = GPUBatchMode.AUTO
    device_ids: tuple[int, ...] | None = None
    vram_limit_mb: int = DEFAULT_VRAM_LIMIT_MB
    vram_per_sim_mb: int = DEFAULT_VRAM_PER_SIM_MB
    gpu_memory_fraction: float = DEFAULT_GPU_MEMORY_FRACTION
    max_sims_per_gpu: int = 4
    fallback_to_cpu: bool = True
    oom_retry_count: int = 2
    oom_retry_delay_sec: float = OOM_RETRY_DELAY_SEC
    utilization_sample_interval_sec: float = DEFAULT_UTILIZATION_SAMPLE_INTERVAL_SEC
    track_utilization: bool = True

    def __post_init__(self) -> None:
        if not (0.1 <= self.gpu_memory_fraction <= 1.0):
            raise ValueError("gpu_memory_fraction must be between 0.1 and 1.0")
        if self.vram_per_sim_mb <= 0:
            raise ValueError("vram_per_sim_mb must be > 0")
        if self.max_sims_per_gpu < 1:
            raise ValueError("max_sims_per_gpu must be >= 1")
        if self.oom_retry_count < 0:
            raise ValueError("oom_retry_count must be >= 0")

    @property
    def effective_vram_limit_mb(self) -> int:
        """Calculate effective VRAM limit based on fraction."""
        return int(self.vram_limit_mb * self.gpu_memory_fraction)

    @property
    def max_concurrent_per_gpu(self) -> int:
        """Calculate maximum concurrent simulations per GPU based on VRAM."""
        vram_based = self.effective_vram_limit_mb // self.vram_per_sim_mb
        return max(1, min(vram_based, self.max_sims_per_gpu))


@dataclass(slots=True)
class GPUUtilizationSample:
    """A single GPU utilization sample.

    Attributes:
        timestamp: Sample timestamp.
        device_id: GPU device ID.
        utilization_percent: GPU compute utilization (0-100).
        memory_used_mb: Memory used in MB.
        memory_free_mb: Memory free in MB.
    """

    timestamp: datetime
    device_id: int
    utilization_percent: float
    memory_used_mb: int
    memory_free_mb: int


@dataclass(slots=True)
class GPUUtilizationMetrics:
    """Aggregated GPU utilization metrics for a batch run.

    Attributes:
        device_id: GPU device ID.
        device_name: GPU device name.
        n_samples: Number of utilization samples collected.
        avg_utilization_percent: Average GPU utilization.
        max_utilization_percent: Peak GPU utilization.
        avg_memory_used_mb: Average memory usage.
        max_memory_used_mb: Peak memory usage.
        total_jobs_run: Total jobs run on this GPU.
        oom_events: Number of OOM events on this GPU.
        total_gpu_time_sec: Total time GPU was active.
    """

    device_id: int
    device_name: str = ""
    n_samples: int = 0
    avg_utilization_percent: float = 0.0
    max_utilization_percent: float = 0.0
    avg_memory_used_mb: float = 0.0
    max_memory_used_mb: int = 0
    total_jobs_run: int = 0
    oom_events: int = 0
    total_gpu_time_sec: float = 0.0
    _samples: list[GPUUtilizationSample] = field(default_factory=list)

    def add_sample(self, sample: GPUUtilizationSample) -> None:
        """Add a utilization sample and update aggregates."""
        self._samples.append(sample)
        self.n_samples = len(self._samples)

        # Update aggregates
        total_util = sum(s.utilization_percent for s in self._samples)
        total_mem = sum(s.memory_used_mb for s in self._samples)

        self.avg_utilization_percent = total_util / self.n_samples
        self.max_utilization_percent = max(s.utilization_percent for s in self._samples)
        self.avg_memory_used_mb = total_mem / self.n_samples
        self.max_memory_used_mb = max(s.memory_used_mb for s in self._samples)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "n_samples": self.n_samples,
            "avg_utilization_percent": round(self.avg_utilization_percent, 2),
            "max_utilization_percent": round(self.max_utilization_percent, 2),
            "avg_memory_used_mb": round(self.avg_memory_used_mb, 2),
            "max_memory_used_mb": self.max_memory_used_mb,
            "total_jobs_run": self.total_jobs_run,
            "oom_events": self.oom_events,
            "total_gpu_time_sec": round(self.total_gpu_time_sec, 2),
        }


@dataclass(slots=True)
class GPUJobResult:
    """Result of a GPU-accelerated simulation job.

    Extends SimulationJobResult with GPU-specific information.
    """

    job_result: SimulationJobResult
    gpu_device_id: int | None = None
    gpu_device_name: str | None = None
    gpu_memory_used_mb: int | None = None
    was_oom_retry: bool = False
    fell_back_to_cpu: bool = False


@dataclass(slots=True)
class GPUBatchResult:
    """Result of a GPU-accelerated batch run.

    Extends BatchResult with GPU-specific metrics.
    """

    batch_result: BatchResult
    gpu_config: GPUBatchConfig
    gpu_metrics: dict[int, GPUUtilizationMetrics] = field(default_factory=dict)
    n_gpu_jobs: int = 0
    n_cpu_fallback_jobs: int = 0
    n_oom_retries: int = 0
    n_oom_failures: int = 0
    gpu_device_info: list[GPUDeviceInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        base = self.batch_result.to_dict()
        base["gpu_batch"] = {
            "mode": self.gpu_config.mode.value,
            "n_gpu_jobs": self.n_gpu_jobs,
            "n_cpu_fallback_jobs": self.n_cpu_fallback_jobs,
            "n_oom_retries": self.n_oom_retries,
            "n_oom_failures": self.n_oom_failures,
            "gpu_metrics": {
                str(device_id): metrics.to_dict()
                for device_id, metrics in self.gpu_metrics.items()
            },
            "gpu_devices": [
                {
                    "device_id": info.device_id,
                    "device_name": info.device_name,
                    "memory_total_mb": info.memory_total_mb,
                }
                for info in self.gpu_device_info
            ],
        }
        return base


# =============================================================================
# GPU Detection and Monitoring
# =============================================================================


def detect_nvidia_gpus() -> list[GPUDeviceState]:
    """Detect available NVIDIA GPUs using nvidia-smi.

    Returns:
        List of GPUDeviceState for each detected GPU.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug("nvidia-smi not available: %s", e)
        return []

    if result.returncode != 0:
        logger.debug("nvidia-smi failed: %s", result.stderr)
        return []

    devices = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            parts = [p.strip() for p in line.split(",")]
            device_id = int(parts[0])
            device_name = parts[1]
            total_mb = int(float(parts[2]))
            free_mb = int(float(parts[3]))
            used_mb = int(float(parts[4]))
            utilization = float(parts[5]) if parts[5] != "[Not Supported]" else 0.0
            temperature = float(parts[6]) if parts[6] != "[Not Supported]" else None
            power = float(parts[7]) if parts[7] != "[Not Supported]" else None

            devices.append(
                GPUDeviceState(
                    device_id=device_id,
                    device_name=device_name,
                    status=GPUStatus.AVAILABLE,
                    total_memory_mb=total_mb,
                    free_memory_mb=free_mb,
                    used_memory_mb=used_mb,
                    utilization_percent=utilization,
                    temperature_c=temperature,
                    power_usage_w=power,
                    last_updated=datetime.now(),
                )
            )
        except (ValueError, IndexError) as e:
            logger.warning("Failed to parse nvidia-smi output line: %s (%s)", line, e)
            continue

    logger.debug("Detected %d NVIDIA GPUs", len(devices))
    return devices


def get_gpu_memory_info(device_id: int) -> tuple[int, int] | None:
    """Get current memory info for a specific GPU.

    Args:
        device_id: CUDA device ID.

    Returns:
        Tuple of (total_mb, free_mb) or None if unavailable.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        parts = result.stdout.strip().split(",")
        return int(float(parts[0])), int(float(parts[1]))
    except Exception:
        return None


def check_cuda_available() -> bool:
    """Check if CUDA is available via nvidia-smi.

    Returns:
        True if CUDA is available.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def is_oom_error(error: Exception) -> bool:
    """Check if an exception indicates GPU out-of-memory.

    Args:
        error: Exception to check.

    Returns:
        True if the error indicates OOM.
    """
    error_str = str(error).lower()
    oom_patterns = [
        "out of memory",
        "oom",
        "cuda error",
        "memory allocation failed",
        "insufficient memory",
        "cudaerror",
        "cuda_error_out_of_memory",
    ]
    return any(pattern in error_str for pattern in oom_patterns)


# =============================================================================
# GPU Batch Runner
# =============================================================================


class GPUBatchSimulationRunner:
    """GPU-accelerated batch simulation runner.

    This class extends BatchSimulationRunner with GPU-specific features:
    - VRAM monitoring and automatic batch sizing
    - Multi-GPU job distribution
    - OOM detection and fallback to sequential/CPU execution
    - GPU utilization metrics tracking

    Example:
        >>> gpu_config = GPUBatchConfig(
        ...     mode=GPUBatchMode.AUTO,
        ...     vram_per_sim_mb=2048,
        ...     fallback_to_cpu=True,
        ... )
        >>> batch_config = BatchConfig(max_workers=4)
        >>> runner = GPUBatchSimulationRunner(sim_runner, batch_config, gpu_config)
        >>> result = runner.run(jobs)
        >>> print(f"GPU jobs: {result.n_gpu_jobs}, CPU fallback: {result.n_cpu_fallback_jobs}")
    """

    def __init__(
        self,
        sim_runner: SimulationRunner,
        batch_config: BatchConfig | None = None,
        gpu_config: GPUBatchConfig | None = None,
    ) -> None:
        """Initialize the GPU batch runner.

        Args:
            sim_runner: SimulationRunner instance for executing individual sims.
            batch_config: Standard batch configuration.
            gpu_config: GPU-specific configuration.
        """
        self.sim_runner = sim_runner
        self.batch_config = batch_config or BatchConfig()
        self.gpu_config = gpu_config or GPUBatchConfig()

        self._devices: dict[int, GPUDeviceState] = {}
        self._device_locks: dict[int, threading.Lock] = {}
        self._device_job_counts: dict[int, int] = {}
        self._gpu_metrics: dict[int, GPUUtilizationMetrics] = {}
        self._utilization_thread: threading.Thread | None = None
        self._stop_utilization_sampling = threading.Event()
        self._progress: BatchProgress | None = None
        self._progress_lock = threading.Lock()
        self._progress_callbacks: list[ProgressCallback] = []
        self._stop_requested = False

        # OOM tracking
        self._oom_retries = 0
        self._oom_failures = 0
        self._cpu_fallback_count = 0
        self._gpu_job_count = 0

    def add_progress_callback(self, callback: ProgressCallback) -> None:
        """Add a callback for progress updates."""
        self._progress_callbacks.append(callback)

    def remove_progress_callback(self, callback: ProgressCallback) -> None:
        """Remove a progress callback."""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)

    def request_stop(self) -> None:
        """Request the batch to stop after current jobs complete."""
        self._stop_requested = True

    def detect_gpus(self) -> list[GPUDeviceState]:
        """Detect and configure available GPUs.

        Returns:
            List of detected GPU device states.
        """
        if self.gpu_config.mode == GPUBatchMode.FORCE_CPU:
            logger.info("GPU mode is FORCE_CPU, skipping GPU detection")
            return []

        devices = detect_nvidia_gpus()

        # Filter by configured device IDs if specified
        if self.gpu_config.device_ids is not None:
            devices = [d for d in devices if d.device_id in self.gpu_config.device_ids]

        # Initialize device tracking
        self._devices = {d.device_id: d for d in devices}
        self._device_locks = {d.device_id: threading.Lock() for d in devices}
        self._device_job_counts = {d.device_id: 0 for d in devices}
        self._gpu_metrics = {
            d.device_id: GPUUtilizationMetrics(
                device_id=d.device_id,
                device_name=d.device_name,
            )
            for d in devices
        }

        if not devices:
            if self.gpu_config.mode == GPUBatchMode.FORCE_GPU:
                raise RuntimeError("No GPUs detected but mode is FORCE_GPU")
            logger.info("No GPUs detected, will use CPU-only execution")
        else:
            logger.info(
                "Detected %d GPU(s): %s",
                len(devices),
                ", ".join(f"{d.device_name} ({d.device_id})" for d in devices),
            )

        return devices

    def run(
        self,
        jobs: list[SimulationJob],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> GPUBatchResult:
        """Run a batch of simulation jobs with GPU acceleration.

        Args:
            jobs: List of simulation jobs to execute.
            progress_callback: Optional callback for progress updates.

        Returns:
            GPUBatchResult with detailed GPU metrics.
        """
        if progress_callback:
            self.add_progress_callback(progress_callback)

        try:
            return self._run_gpu_batch(jobs)
        finally:
            if progress_callback:
                self.remove_progress_callback(progress_callback)
            self._stop_requested = False
            self._stop_utilization_sampling.set()
            if self._utilization_thread is not None:
                self._utilization_thread.join(timeout=2.0)

    def _run_gpu_batch(self, jobs: list[SimulationJob]) -> GPUBatchResult:
        """Internal GPU batch execution logic."""
        # Detect available GPUs
        devices = self.detect_gpus()
        use_gpu = len(devices) > 0 and self.gpu_config.mode != GPUBatchMode.FORCE_CPU

        if not jobs:
            return GPUBatchResult(
                batch_result=BatchResult(
                    config=self.batch_config,
                    jobs=[],
                    total_time_sec=0.0,
                ),
                gpu_config=self.gpu_config,
            )

        # Start utilization sampling if tracking enabled
        if use_gpu and self.gpu_config.track_utilization:
            self._start_utilization_sampling()

        # Sort jobs by priority
        sorted_jobs = sorted(jobs, key=lambda j: j.priority, reverse=True)

        # Initialize progress
        self._progress = BatchProgress(
            total=len(sorted_jobs),
            pending=len(sorted_jobs),
            start_time=time.monotonic(),
        )
        self._notify_progress()

        # Calculate effective parallelism
        if use_gpu:
            # GPU-aware parallelism: max simulations across all GPUs
            total_gpu_slots = sum(
                self.gpu_config.max_concurrent_per_gpu
                for _ in devices
            )
            effective_workers = min(
                self.batch_config.effective_max_workers,
                total_gpu_slots,
            )
        else:
            effective_workers = self.batch_config.effective_max_workers

        logger.info(
            "Starting GPU batch of %d simulations with %d workers (GPU mode: %s)",
            len(sorted_jobs),
            effective_workers,
            "enabled" if use_gpu else "disabled",
        )

        start_time = time.monotonic()
        results: list[SimulationJobResult] = []

        # Use ThreadPoolExecutor with GPU-aware job dispatch
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_job = {
                executor.submit(
                    self._run_single_gpu_job,
                    job,
                    use_gpu,
                ): job
                for job in sorted_jobs
            }

            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    gpu_job_result = future.result()
                    job_result = gpu_job_result.job_result
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
                if self.batch_config.fail_fast and job_result.status != SimulationStatus.COMPLETED:
                    logger.warning("Fail-fast triggered by job %s", job.job_id)
                    self._stop_requested = True

                # Check stop request AFTER processing current result
                if self._stop_requested and self.batch_config.fail_fast:
                    for f in future_to_job:
                        f.cancel()
                    break

            # Add SKIPPED results for any jobs that weren't processed
            processed_job_ids = {r.job_id for r in results}
            for job in sorted_jobs:
                if job.job_id not in processed_job_ids:
                    results.append(SimulationJobResult(
                        job_id=job.job_id,
                        status=SimulationStatus.SKIPPED,
                        error="Batch stopped (fail_fast triggered)",
                    ))
                    # Update progress for skipped jobs
                    with self._progress_lock:
                        if self._progress:
                            self._progress.skipped += 1
                            self._progress.pending -= 1

        total_time = time.monotonic() - start_time

        # Stop utilization sampling
        self._stop_utilization_sampling.set()
        if self._utilization_thread is not None:
            self._utilization_thread.join(timeout=2.0)

        logger.info(
            "GPU batch completed: %d/%d successful in %.2f seconds "
            "(GPU: %d, CPU fallback: %d, OOM retries: %d)",
            sum(1 for r in results if r.status == SimulationStatus.COMPLETED),
            len(results),
            total_time,
            self._gpu_job_count,
            self._cpu_fallback_count,
            self._oom_retries,
        )

        batch_result = BatchResult(
            config=self.batch_config,
            jobs=results,
            total_time_sec=total_time,
        )

        return GPUBatchResult(
            batch_result=batch_result,
            gpu_config=self.gpu_config,
            gpu_metrics=dict(self._gpu_metrics),
            n_gpu_jobs=self._gpu_job_count,
            n_cpu_fallback_jobs=self._cpu_fallback_count,
            n_oom_retries=self._oom_retries,
            n_oom_failures=self._oom_failures,
            gpu_device_info=[d.to_gpu_device_info() for d in devices],
        )

    def _run_single_gpu_job(
        self,
        job: SimulationJob,
        use_gpu: bool,
    ) -> GPUJobResult:
        """Execute a single simulation job with GPU support.

        Handles GPU device allocation, OOM detection, and fallback to CPU.
        """
        started_at = datetime.now()
        retry_count = 0
        last_error: str | None = None
        was_timeout = False
        gpu_device_id: int | None = None
        gpu_device_name: str | None = None
        fell_back_to_cpu = False
        was_oom_retry = False

        # Update progress
        with self._progress_lock:
            if self._progress:
                self._progress.running += 1
                self._progress.pending -= 1
                self._progress.current_jobs.append(job.job_id)
        self._notify_progress()

        start_time = time.monotonic()

        # Try GPU execution with OOM retry logic
        if use_gpu:
            gpu_device_id, gpu_device_name = self._allocate_gpu()

            oom_retry_count = 0
            while oom_retry_count <= self.gpu_config.oom_retry_count:
                if self._stop_requested:
                    self._release_gpu(gpu_device_id)
                    return GPUJobResult(
                        job_result=SimulationJobResult(
                            job_id=job.job_id,
                            status=SimulationStatus.SKIPPED,
                            error="Batch stop requested",
                            retry_count=retry_count,
                            started_at=started_at,
                            completed_at=datetime.now(),
                        ),
                        gpu_device_id=gpu_device_id,
                        gpu_device_name=gpu_device_name,
                    )

                try:
                    # Set GPU environment for this job
                    env_device_id = gpu_device_id if gpu_device_id is not None else 0
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(env_device_id)

                    # Enable GPU in the simulation spec
                    gpu_spec = self._enable_gpu_in_spec(job.spec, env_device_id)

                    result = self.sim_runner.run(
                        gpu_spec,
                        job.geometry,
                        output_dir=job.output_dir,
                        openems_args=list(job.openems_args) if job.openems_args else None,
                        timeout_sec=self.batch_config.timeout_per_sim_sec,
                    )

                    # Successful GPU execution
                    self._gpu_job_count += 1
                    if gpu_device_id is not None:
                        self._gpu_metrics[gpu_device_id].total_jobs_run += 1

                    convergence_report = self._run_convergence_check(job, result)
                    execution_time = time.monotonic() - start_time

                    # Release GPU
                    self._release_gpu(gpu_device_id)

                    # Update progress
                    with self._progress_lock:
                        if self._progress:
                            self._progress.running -= 1

                    return GPUJobResult(
                        job_result=SimulationJobResult(
                            job_id=job.job_id,
                            status=SimulationStatus.COMPLETED,
                            result=result,
                            convergence_report=convergence_report,
                            execution_time_sec=execution_time,
                            retry_count=retry_count,
                            started_at=started_at,
                            completed_at=datetime.now(),
                        ),
                        gpu_device_id=gpu_device_id,
                        gpu_device_name=gpu_device_name,
                        was_oom_retry=was_oom_retry,
                    )

                except SimulationTimeoutError as e:
                    last_error = str(e)
                    was_timeout = True
                    logger.warning("Job %s timed out on GPU %d", job.job_id, env_device_id)
                    break  # Don't retry timeouts

                except (SimulationExecutionError, Exception) as e:
                    if is_oom_error(e):
                        oom_retry_count += 1
                        self._oom_retries += 1
                        was_oom_retry = True

                        if gpu_device_id is not None:
                            self._gpu_metrics[gpu_device_id].oom_events += 1

                        if oom_retry_count <= self.gpu_config.oom_retry_count:
                            logger.warning(
                                "Job %s OOM on GPU %d, retry %d/%d",
                                job.job_id,
                                env_device_id,
                                oom_retry_count,
                                self.gpu_config.oom_retry_count,
                            )
                            time.sleep(self.gpu_config.oom_retry_delay_sec)
                            continue
                        else:
                            logger.warning(
                                "Job %s OOM exhausted retries on GPU %d",
                                job.job_id,
                                env_device_id,
                            )
                            self._oom_failures += 1
                            # Fall through to CPU fallback
                            break
                    else:
                        last_error = str(e)
                        logger.warning(
                            "Job %s failed on GPU %d: %s",
                            job.job_id,
                            env_device_id,
                            e,
                        )
                        break

            # Release GPU before potential CPU fallback
            self._release_gpu(gpu_device_id)

            # Try CPU fallback if enabled
            if self.gpu_config.fallback_to_cpu and not was_timeout:
                logger.info("Falling back to CPU for job %s", job.job_id)
                fell_back_to_cpu = True
                self._cpu_fallback_count += 1

                # Clear GPU environment
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

                try:
                    result = self.sim_runner.run(
                        job.spec,  # Original spec without GPU
                        job.geometry,
                        output_dir=job.output_dir,
                        openems_args=list(job.openems_args) if job.openems_args else None,
                        timeout_sec=self.batch_config.timeout_per_sim_sec,
                    )

                    convergence_report = self._run_convergence_check(job, result)
                    execution_time = time.monotonic() - start_time

                    with self._progress_lock:
                        if self._progress:
                            self._progress.running -= 1

                    return GPUJobResult(
                        job_result=SimulationJobResult(
                            job_id=job.job_id,
                            status=SimulationStatus.COMPLETED,
                            result=result,
                            convergence_report=convergence_report,
                            execution_time_sec=execution_time,
                            retry_count=retry_count,
                            started_at=started_at,
                            completed_at=datetime.now(),
                        ),
                        gpu_device_id=gpu_device_id,
                        gpu_device_name=gpu_device_name,
                        was_oom_retry=was_oom_retry,
                        fell_back_to_cpu=True,
                    )

                except SimulationTimeoutError as e:
                    last_error = str(e)
                    was_timeout = True

                except Exception as e:
                    last_error = str(e)
                    logger.warning("Job %s CPU fallback failed: %s", job.job_id, e)

        else:
            # CPU-only execution (no GPU available or mode is FORCE_CPU)
            while retry_count <= self.batch_config.retry_failed:
                if self._stop_requested:
                    return GPUJobResult(
                        job_result=SimulationJobResult(
                            job_id=job.job_id,
                            status=SimulationStatus.SKIPPED,
                            error="Batch stop requested",
                            retry_count=retry_count,
                            started_at=started_at,
                            completed_at=datetime.now(),
                        ),
                    )

                try:
                    result = self.sim_runner.run(
                        job.spec,
                        job.geometry,
                        output_dir=job.output_dir,
                        openems_args=list(job.openems_args) if job.openems_args else None,
                        timeout_sec=self.batch_config.timeout_per_sim_sec,
                    )

                    convergence_report = self._run_convergence_check(job, result)
                    execution_time = time.monotonic() - start_time

                    with self._progress_lock:
                        if self._progress:
                            self._progress.running -= 1

                    return GPUJobResult(
                        job_result=SimulationJobResult(
                            job_id=job.job_id,
                            status=SimulationStatus.COMPLETED,
                            result=result,
                            convergence_report=convergence_report,
                            execution_time_sec=execution_time,
                            retry_count=retry_count,
                            started_at=started_at,
                            completed_at=datetime.now(),
                        ),
                    )

                except SimulationTimeoutError as e:
                    last_error = str(e)
                    was_timeout = True
                    break

                except Exception as e:
                    last_error = str(e)
                    retry_count += 1
                    if retry_count <= self.batch_config.retry_failed:
                        logger.warning(
                            "Job %s failed (attempt %d/%d): %s",
                            job.job_id,
                            retry_count,
                            self.batch_config.retry_failed + 1,
                            e,
                        )

        # Job failed
        execution_time = time.monotonic() - start_time

        with self._progress_lock:
            if self._progress:
                self._progress.running -= 1

        status = SimulationStatus.TIMEOUT if was_timeout else SimulationStatus.FAILED

        return GPUJobResult(
            job_result=SimulationJobResult(
                job_id=job.job_id,
                status=status,
                error=last_error,
                execution_time_sec=execution_time,
                retry_count=retry_count,
                started_at=started_at,
                completed_at=datetime.now(),
            ),
            gpu_device_id=gpu_device_id,
            gpu_device_name=gpu_device_name,
            was_oom_retry=was_oom_retry,
            fell_back_to_cpu=fell_back_to_cpu,
        )

    def _allocate_gpu(self) -> tuple[int | None, str | None]:
        """Allocate an available GPU for a job.

        Returns:
            Tuple of (device_id, device_name) or (None, None) if no GPU available.
        """
        # Find GPU with least active jobs
        best_device: int | None = None
        best_job_count = float("inf")

        for device_id, lock in self._device_locks.items():
            with lock:
                job_count = self._device_job_counts[device_id]
                if job_count < self.gpu_config.max_concurrent_per_gpu:
                    if job_count < best_job_count:
                        best_job_count = job_count
                        best_device = device_id

        if best_device is not None:
            with self._device_locks[best_device]:
                self._device_job_counts[best_device] += 1
            device_name = self._devices.get(best_device)
            return best_device, device_name.device_name if device_name else None

        return None, None

    def _release_gpu(self, device_id: int | None) -> None:
        """Release a GPU after job completion."""
        if device_id is not None and device_id in self._device_locks:
            with self._device_locks[device_id]:
                self._device_job_counts[device_id] = max(
                    0, self._device_job_counts[device_id] - 1
                )

    def _enable_gpu_in_spec(self, spec: SimulationSpec, device_id: int) -> SimulationSpec:
        """Create a modified spec with GPU enabled.

        Args:
            spec: Original simulation spec.
            device_id: GPU device ID to use.

        Returns:
            Modified spec with GPU settings.
        """
        # Create a modified spec dict with GPU enabled
        spec_dict = spec.model_dump(mode="json")
        spec_dict["control"]["engine"]["use_gpu"] = True
        spec_dict["control"]["engine"]["gpu_device_id"] = device_id
        spec_dict["control"]["engine"]["gpu_memory_fraction"] = self.gpu_config.gpu_memory_fraction

        return SimulationSpec.model_validate(spec_dict)

    def _run_convergence_check(
        self,
        job: SimulationJob,
        result: SimulationResult,
    ) -> ConvergenceReport | None:
        """Run convergence validation if enabled."""
        if not self.batch_config.validate_convergence:
            return None

        try:
            return validate_simulation_convergence(
                result.outputs_dir,
                job.spec,
                simulation_hash=result.simulation_hash,
                config=self.batch_config.convergence_config,
            )
        except Exception as e:
            logger.warning(
                "Convergence validation failed for %s: %s",
                job.job_id,
                e,
            )
            return None

    def _start_utilization_sampling(self) -> None:
        """Start background thread for GPU utilization sampling."""
        self._stop_utilization_sampling.clear()
        self._utilization_thread = threading.Thread(
            target=self._sample_utilization_loop,
            daemon=True,
        )
        self._utilization_thread.start()

    def _sample_utilization_loop(self) -> None:
        """Background loop for sampling GPU utilization."""
        while not self._stop_utilization_sampling.is_set():
            try:
                devices = detect_nvidia_gpus()
                for device in devices:
                    if device.device_id in self._gpu_metrics:
                        sample = GPUUtilizationSample(
                            timestamp=datetime.now(),
                            device_id=device.device_id,
                            utilization_percent=device.utilization_percent,
                            memory_used_mb=device.used_memory_mb,
                            memory_free_mb=device.free_memory_mb,
                        )
                        self._gpu_metrics[device.device_id].add_sample(sample)
            except Exception as e:
                logger.debug("Error sampling GPU utilization: %s", e)

            self._stop_utilization_sampling.wait(
                timeout=self.gpu_config.utilization_sample_interval_sec
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
# Utility Functions
# =============================================================================


def estimate_gpu_batch_time(
    n_jobs: int,
    gpu_config: GPUBatchConfig,
    batch_config: BatchConfig,
    n_gpus: int = 1,
    avg_sim_time_sec: float = 300.0,
) -> float:
    """Estimate total GPU batch execution time.

    Args:
        n_jobs: Number of simulation jobs.
        gpu_config: GPU configuration.
        batch_config: Batch configuration (unused for GPU estimates).
        n_gpus: Number of available GPUs.
        avg_sim_time_sec: Average time per simulation.

    Returns:
        Estimated total time in seconds.
    """
    # For GPU estimation, use only GPU slots (not RAM-limited batch_config workers)
    # as GPU execution is limited by VRAM, not system RAM
    total_gpu_slots = n_gpus * gpu_config.max_concurrent_per_gpu
    effective_workers = max(1, total_gpu_slots)
    batches = (n_jobs + effective_workers - 1) // effective_workers
    return batches * avg_sim_time_sec


def write_gpu_batch_result(result: GPUBatchResult, output_path: Path) -> None:
    """Write GPU batch result to JSON file.

    Args:
        result: GPUBatchResult to write.
        output_path: Path for output JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json_dumps(result.to_dict())
    output_path.write_text(f"{text}\n", encoding="utf-8")
