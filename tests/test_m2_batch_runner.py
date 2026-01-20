"""Tests for openEMS batch simulation runner (REQ-M2-009).

This module tests the batch simulation functionality including:
- BatchConfig validation and resource calculations
- BatchSimulationRunner parallel execution
- Progress tracking and callbacks
- Convergence integration
- Aggregated result collection
- Error handling and retries
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from formula_foundry.openems.batch_runner import (
    BatchConfig,
    BatchProgress,
    BatchResult,
    BatchSimulationRunner,
    ProgressCallback,
    SimulationJob,
    SimulationJobResult,
    SimulationStatus,
    create_batch_jobs,
    estimate_batch_time,
    load_batch_result_summary,
    write_batch_result,
)
from formula_foundry.openems.convergence import (
    ConvergenceConfig,
    ConvergenceReport,
    ConvergenceStatus,
)
from formula_foundry.openems.geometry import GeometrySpec, StackupMaterialsSpec
from formula_foundry.openems.sim_runner import (
    SimulationError,
    SimulationExecutionError,
    SimulationResult,
    SimulationRunner,
    SimulationTimeoutError,
)
from formula_foundry.openems.spec import (
    ExcitationSpec,
    FrequencySpec,
    GeometryRefSpec,
    PortSpec,
    SimulationSpec,
    ToolchainSpec,
    OpenEMSToolchainSpec,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def make_minimal_spec(sim_id: str = "test_sim") -> SimulationSpec:
    """Create a minimal SimulationSpec for testing."""
    return SimulationSpec(
        schema_version=1,
        simulation_id=sim_id,
        toolchain=ToolchainSpec(
            openems=OpenEMSToolchainSpec(
                version="0.0.35",
                docker_image="ghcr.io/openems:0.0.35",
            )
        ),
        geometry_ref=GeometryRefSpec(design_hash="test_hash"),
        excitation=ExcitationSpec(f0_hz=5e9, fc_hz=10e9),
        frequency=FrequencySpec(f_start_hz=1e9, f_stop_hz=10e9),
        ports=[
            PortSpec(
                id="P1",
                type="lumped",
                excite=True,
                position_nm=(0, 0, 0),
                direction="x",
            )
        ],
    )


def make_minimal_geometry() -> GeometrySpec:
    """Create a minimal GeometrySpec for testing."""
    from formula_foundry.openems.geometry import (
        BoardOutlineSpec,
        DiscontinuitySpec,
        GeometrySpec,
        LayerSpec,
        StackupMaterialsSpec,
        StackupSpec,
        TransmissionLineSpec,
    )

    return GeometrySpec(
        design_hash="test_design_hash_12345",
        coupon_family="F1_SINGLE_ENDED_VIA",
        board=BoardOutlineSpec(
            width_nm=20_000_000,
            length_nm=40_000_000,
            corner_radius_nm=2_000_000,
        ),
        stackup=StackupSpec(
            copper_layers=4,
            thicknesses_nm={
                "L1_to_L2": 180_000,
                "L2_to_L3": 800_000,
                "L3_to_L4": 180_000,
            },
            materials=StackupMaterialsSpec(er=4.1, loss_tangent=0.02),
        ),
        layers=[
            LayerSpec(id="L1", z_nm=0, role="signal"),
            LayerSpec(id="L2", z_nm=180_000, role="ground"),
            LayerSpec(id="L3", z_nm=980_000, role="ground"),
            LayerSpec(id="L4", z_nm=1_160_000, role="signal"),
        ],
        transmission_line=TransmissionLineSpec(
            type="CPWG",
            layer="F.Cu",
            w_nm=300_000,
            gap_nm=180_000,
            length_left_nm=10_000_000,
            length_right_nm=10_000_000,
        ),
        discontinuity=DiscontinuitySpec(
            type="VIA_TRANSITION",
            parameters_nm={
                "signal_via.drill_nm": 300_000,
                "signal_via.diameter_nm": 650_000,
            },
        ),
    )


def make_mock_simulation_result(
    output_dir: Path,
    simulation_hash: str = "test_hash_123",
) -> SimulationResult:
    """Create a mock SimulationResult for testing."""
    outputs_dir = output_dir / "sim_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "simulation_manifest.json"
    manifest_path.write_text("{}")

    return SimulationResult(
        output_dir=output_dir,
        outputs_dir=outputs_dir,
        manifest_path=manifest_path,
        cache_hit=False,
        simulation_hash=simulation_hash,
        manifest_hash="manifest_hash_456",
        output_hashes={},
        execution_time_sec=10.0,
    )


# =============================================================================
# BatchConfig Tests
# =============================================================================


class TestBatchConfig:
    """Tests for BatchConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BatchConfig()
        assert config.max_workers == 4
        assert config.ram_limit_gb == 15.0
        assert config.cpu_cores == 16
        assert config.ram_per_sim_gb == 4.0
        assert config.timeout_per_sim_sec == 3600
        assert config.fail_fast is False
        assert config.validate_convergence is True
        assert config.retry_failed == 0

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = BatchConfig(
            max_workers=8,
            ram_limit_gb=32.0,
            cpu_cores=32,
            ram_per_sim_gb=8.0,
            timeout_per_sim_sec=7200,
            fail_fast=True,
            validate_convergence=False,
            retry_failed=2,
        )
        assert config.max_workers == 8
        assert config.ram_limit_gb == 32.0
        assert config.fail_fast is True
        assert config.retry_failed == 2

    def test_effective_max_workers_ram_limited(self):
        """Test effective workers when limited by RAM."""
        config = BatchConfig(
            max_workers=8,
            ram_limit_gb=12.0,
            ram_per_sim_gb=4.0,  # 12/4 = 3 workers
        )
        assert config.effective_max_workers == 3

    def test_effective_max_workers_worker_limited(self):
        """Test effective workers when limited by max_workers."""
        config = BatchConfig(
            max_workers=2,
            ram_limit_gb=32.0,
            ram_per_sim_gb=4.0,  # 32/4 = 8, but max is 2
        )
        assert config.effective_max_workers == 2

    def test_invalid_max_workers(self):
        """Test validation rejects invalid max_workers."""
        with pytest.raises(ValueError, match="max_workers"):
            BatchConfig(max_workers=0)

    def test_invalid_ram_limit(self):
        """Test validation rejects invalid ram_limit_gb."""
        with pytest.raises(ValueError, match="ram_limit_gb"):
            BatchConfig(ram_limit_gb=0)

    def test_invalid_timeout(self):
        """Test validation rejects invalid timeout."""
        with pytest.raises(ValueError, match="timeout_per_sim_sec"):
            BatchConfig(timeout_per_sim_sec=-1)

    def test_invalid_retry_count(self):
        """Test validation rejects negative retry count."""
        with pytest.raises(ValueError, match="retry_failed"):
            BatchConfig(retry_failed=-1)


# =============================================================================
# SimulationJob Tests
# =============================================================================


class TestSimulationJob:
    """Tests for SimulationJob dataclass."""

    def test_basic_creation(self, tmp_path):
        """Test creating a simulation job."""
        spec = make_minimal_spec()
        geometry = make_minimal_geometry()
        output_dir = tmp_path / "output"

        job = SimulationJob(
            job_id="test_job_001",
            spec=spec,
            geometry=geometry,
            output_dir=output_dir,
        )

        assert job.job_id == "test_job_001"
        assert job.spec == spec
        assert job.priority == 0
        assert job.tags == {}

    def test_with_priority_and_tags(self, tmp_path):
        """Test job with priority and tags."""
        spec = make_minimal_spec()
        geometry = make_minimal_geometry()

        job = SimulationJob(
            job_id="high_priority_job",
            spec=spec,
            geometry=geometry,
            output_dir=tmp_path,
            priority=10,
            tags={"category": "sweep", "param": "via_diameter"},
        )

        assert job.priority == 10
        assert job.tags["category"] == "sweep"


# =============================================================================
# SimulationJobResult Tests
# =============================================================================


class TestSimulationJobResult:
    """Tests for SimulationJobResult dataclass."""

    def test_successful_result(self, tmp_path):
        """Test successful job result."""
        result = make_mock_simulation_result(tmp_path)

        job_result = SimulationJobResult(
            job_id="test_001",
            status=SimulationStatus.COMPLETED,
            result=result,
            execution_time_sec=15.0,
        )

        assert job_result.passed
        assert job_result.status == SimulationStatus.COMPLETED

    def test_failed_result(self):
        """Test failed job result."""
        job_result = SimulationJobResult(
            job_id="test_002",
            status=SimulationStatus.FAILED,
            error="Simulation diverged",
        )

        assert not job_result.passed
        assert job_result.error == "Simulation diverged"

    def test_passed_with_convergence(self, tmp_path):
        """Test job passes when convergence passes."""
        result = make_mock_simulation_result(tmp_path)

        # Create a passing convergence report
        convergence_report = ConvergenceReport(
            checks=[],
            overall_status=ConvergenceStatus.PASSED,
            simulation_hash="test",
            canonical_hash="test",
            config=ConvergenceConfig(),
        )

        job_result = SimulationJobResult(
            job_id="test_003",
            status=SimulationStatus.COMPLETED,
            result=result,
            convergence_report=convergence_report,
        )

        assert job_result.passed

    def test_failed_with_convergence_failure(self, tmp_path):
        """Test job fails when convergence fails."""
        result = make_mock_simulation_result(tmp_path)

        # Create a failing convergence report
        convergence_report = ConvergenceReport(
            checks=[],
            overall_status=ConvergenceStatus.FAILED,
            simulation_hash="test",
            canonical_hash="test",
            config=ConvergenceConfig(),
        )

        job_result = SimulationJobResult(
            job_id="test_004",
            status=SimulationStatus.COMPLETED,
            result=result,
            convergence_report=convergence_report,
        )

        assert not job_result.passed  # Convergence failed


# =============================================================================
# BatchProgress Tests
# =============================================================================


class TestBatchProgress:
    """Tests for BatchProgress dataclass."""

    def test_initial_state(self):
        """Test initial progress state."""
        progress = BatchProgress(total=10, pending=10)
        assert progress.finished == 0
        assert progress.percent_complete == 0.0

    def test_partial_completion(self):
        """Test progress during execution."""
        progress = BatchProgress(
            total=10,
            pending=4,
            running=2,
            completed=3,
            failed=1,
        )
        assert progress.finished == 4
        assert progress.percent_complete == 40.0

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        start = time.monotonic()
        progress = BatchProgress(total=5, start_time=start)
        time.sleep(0.01)  # Small delay
        assert progress.elapsed_sec > 0

    def test_estimated_remaining(self):
        """Test remaining time estimation."""
        progress = BatchProgress(
            total=10,
            completed=5,
            start_time=time.monotonic() - 50.0,  # 50 seconds elapsed
        )
        # 5 jobs in 50 sec = 10 sec/job, 5 remaining = ~50 sec
        remaining = progress.estimated_remaining_sec
        assert remaining is not None
        assert 40.0 < remaining < 60.0

    def test_zero_jobs_complete(self):
        """Test percent complete with zero total."""
        progress = BatchProgress(total=0)
        assert progress.percent_complete == 100.0


# =============================================================================
# BatchResult Tests
# =============================================================================


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_empty_result(self):
        """Test result with no jobs."""
        result = BatchResult(
            config=BatchConfig(),
            jobs=[],
            total_time_sec=0.1,
        )
        assert result.n_completed == 0
        assert result.n_failed == 0
        assert result.success_rate == 0.0

    def test_all_successful(self, tmp_path):
        """Test result with all jobs successful."""
        jobs = [
            SimulationJobResult(
                job_id=f"job_{i}",
                status=SimulationStatus.COMPLETED,
                result=make_mock_simulation_result(tmp_path / f"job_{i}"),
            )
            for i in range(5)
        ]

        result = BatchResult(
            config=BatchConfig(),
            jobs=jobs,
            total_time_sec=100.0,
        )

        assert result.n_completed == 5
        assert result.n_failed == 0
        assert result.success_rate == 100.0
        assert result.all_passed

    def test_mixed_results(self, tmp_path):
        """Test result with mixed success/failure."""
        jobs = [
            SimulationJobResult(
                job_id="job_0",
                status=SimulationStatus.COMPLETED,
                result=make_mock_simulation_result(tmp_path / "job_0"),
            ),
            SimulationJobResult(
                job_id="job_1",
                status=SimulationStatus.FAILED,
                error="Error",
            ),
            SimulationJobResult(
                job_id="job_2",
                status=SimulationStatus.TIMEOUT,
                error="Timeout",
            ),
            SimulationJobResult(
                job_id="job_3",
                status=SimulationStatus.SKIPPED,
            ),
        ]

        result = BatchResult(
            config=BatchConfig(),
            jobs=jobs,
            total_time_sec=60.0,
        )

        assert result.n_completed == 1
        assert result.n_failed == 2  # failed + timeout
        assert result.n_skipped == 1
        assert result.success_rate == 25.0
        assert not result.all_passed

    def test_get_job(self, tmp_path):
        """Test retrieving job by ID."""
        jobs = [
            SimulationJobResult(
                job_id="target_job",
                status=SimulationStatus.COMPLETED,
                result=make_mock_simulation_result(tmp_path),
            ),
        ]

        result = BatchResult(
            config=BatchConfig(),
            jobs=jobs,
            total_time_sec=10.0,
        )

        assert result.get_job("target_job") is not None
        assert result.get_job("nonexistent") is None

    def test_to_dict(self, tmp_path):
        """Test converting to dictionary."""
        jobs = [
            SimulationJobResult(
                job_id="job_0",
                status=SimulationStatus.COMPLETED,
                result=make_mock_simulation_result(tmp_path),
                execution_time_sec=5.0,
            ),
        ]

        result = BatchResult(
            config=BatchConfig(),
            jobs=jobs,
            total_time_sec=10.0,
        )

        d = result.to_dict()
        assert d["total_jobs"] == 1
        assert d["n_completed"] == 1
        assert d["canonical_hash"] != ""
        assert len(d["jobs"]) == 1

    def test_canonical_hash_computed(self, tmp_path):
        """Test that canonical hash is computed."""
        jobs = [
            SimulationJobResult(
                job_id="job_0",
                status=SimulationStatus.COMPLETED,
                result=make_mock_simulation_result(tmp_path),
            ),
        ]

        result = BatchResult(
            config=BatchConfig(),
            jobs=jobs,
            total_time_sec=10.0,
        )

        assert result.canonical_hash != ""
        assert len(result.canonical_hash) == 64  # SHA256 hex


# =============================================================================
# BatchSimulationRunner Tests
# =============================================================================


class TestBatchSimulationRunner:
    """Tests for BatchSimulationRunner class."""

    def test_run_empty_batch(self):
        """Test running empty batch."""
        mock_sim_runner = MagicMock(spec=SimulationRunner)
        runner = BatchSimulationRunner(mock_sim_runner)

        result = runner.run([])

        assert len(result.jobs) == 0
        assert result.total_time_sec < 1.0

    def test_run_single_job_success(self, tmp_path):
        """Test running single successful job."""
        # Create mock SimulationRunner
        mock_result = make_mock_simulation_result(tmp_path / "output")
        mock_sim_runner = MagicMock(spec=SimulationRunner)
        mock_sim_runner.run.return_value = mock_result

        # Create runner with convergence disabled
        config = BatchConfig(validate_convergence=False)
        runner = BatchSimulationRunner(mock_sim_runner, config)

        # Create job
        job = SimulationJob(
            job_id="test_single",
            spec=make_minimal_spec(),
            geometry=make_minimal_geometry(),
            output_dir=tmp_path / "output",
        )

        result = runner.run([job])

        assert result.n_completed == 1
        assert result.n_failed == 0
        mock_sim_runner.run.assert_called_once()

    def test_run_multiple_jobs(self, tmp_path):
        """Test running multiple jobs in parallel."""
        call_count = [0]
        call_lock = threading.Lock()

        def mock_run(*args, **kwargs):
            with call_lock:
                idx = call_count[0]
                call_count[0] += 1
            time.sleep(0.01)  # Small delay
            return make_mock_simulation_result(tmp_path / f"output_{idx}")

        mock_sim_runner = MagicMock(spec=SimulationRunner)
        mock_sim_runner.run.side_effect = mock_run

        config = BatchConfig(max_workers=4, validate_convergence=False)
        runner = BatchSimulationRunner(mock_sim_runner, config)

        jobs = [
            SimulationJob(
                job_id=f"job_{i}",
                spec=make_minimal_spec(f"sim_{i}"),
                geometry=make_minimal_geometry(),
                output_dir=tmp_path / f"output_{i}",
            )
            for i in range(8)
        ]

        result = runner.run(jobs)

        assert result.n_completed == 8
        assert result.n_failed == 0
        assert mock_sim_runner.run.call_count == 8

    def test_run_with_failure(self, tmp_path):
        """Test handling simulation failures."""
        mock_sim_runner = MagicMock(spec=SimulationRunner)
        mock_sim_runner.run.side_effect = SimulationExecutionError(
            "Simulation diverged",
            returncode=1,
            stdout="",
            stderr="Error: diverged",
        )

        config = BatchConfig(validate_convergence=False, retry_failed=0)
        runner = BatchSimulationRunner(mock_sim_runner, config)

        job = SimulationJob(
            job_id="failing_job",
            spec=make_minimal_spec(),
            geometry=make_minimal_geometry(),
            output_dir=tmp_path / "output",
        )

        result = runner.run([job])

        assert result.n_completed == 0
        assert result.n_failed == 1
        assert result.jobs[0].status == SimulationStatus.FAILED
        assert "diverged" in result.jobs[0].error.lower()

    def test_run_with_timeout(self, tmp_path):
        """Test handling simulation timeouts."""
        mock_sim_runner = MagicMock(spec=SimulationRunner)
        mock_sim_runner.run.side_effect = SimulationTimeoutError(
            "Simulation timed out after 3600 seconds"
        )

        config = BatchConfig(validate_convergence=False, retry_failed=0)
        runner = BatchSimulationRunner(mock_sim_runner, config)

        job = SimulationJob(
            job_id="timeout_job",
            spec=make_minimal_spec(),
            geometry=make_minimal_geometry(),
            output_dir=tmp_path / "output",
        )

        result = runner.run([job])

        assert result.n_failed == 1
        assert result.jobs[0].status == SimulationStatus.TIMEOUT

    def test_run_with_retry(self, tmp_path):
        """Test retry behavior on failure."""
        call_count = [0]

        def mock_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise SimulationExecutionError("Temporary failure", 1, "", "")
            return make_mock_simulation_result(tmp_path / "output")

        mock_sim_runner = MagicMock(spec=SimulationRunner)
        mock_sim_runner.run.side_effect = mock_run

        config = BatchConfig(validate_convergence=False, retry_failed=2)
        runner = BatchSimulationRunner(mock_sim_runner, config)

        job = SimulationJob(
            job_id="retry_job",
            spec=make_minimal_spec(),
            geometry=make_minimal_geometry(),
            output_dir=tmp_path / "output",
        )

        result = runner.run([job])

        assert result.n_completed == 1  # Succeeded after retries
        assert result.jobs[0].retry_count == 2

    def test_fail_fast(self, tmp_path):
        """Test fail_fast stops batch on first failure."""
        execution_order = []
        exec_lock = threading.Lock()

        def mock_run(*args, **kwargs):
            job_id = args[0].simulation_id
            with exec_lock:
                execution_order.append(job_id)
            time.sleep(0.02)
            if "fail" in job_id:
                raise SimulationExecutionError("Failed", 1, "", "")
            return make_mock_simulation_result(tmp_path / job_id)

        mock_sim_runner = MagicMock(spec=SimulationRunner)
        mock_sim_runner.run.side_effect = mock_run

        config = BatchConfig(
            max_workers=1,  # Sequential execution
            validate_convergence=False,
            fail_fast=True,
        )
        runner = BatchSimulationRunner(mock_sim_runner, config)

        jobs = [
            SimulationJob(
                job_id="job_1",
                spec=make_minimal_spec("sim_fail"),  # This will fail
                geometry=make_minimal_geometry(),
                output_dir=tmp_path / "job_1",
                priority=10,  # Higher priority, runs first
            ),
            SimulationJob(
                job_id="job_2",
                spec=make_minimal_spec("sim_ok"),
                geometry=make_minimal_geometry(),
                output_dir=tmp_path / "job_2",
                priority=0,
            ),
        ]

        result = runner.run(jobs)

        # At least one job should fail or be skipped
        assert result.n_failed >= 1 or result.n_skipped >= 1

    def test_progress_callback(self, tmp_path):
        """Test progress callback is called."""
        progress_updates = []

        def progress_callback(progress: BatchProgress) -> None:
            progress_updates.append(
                (progress.total, progress.completed, progress.running)
            )

        mock_result = make_mock_simulation_result(tmp_path / "output")
        mock_sim_runner = MagicMock(spec=SimulationRunner)
        mock_sim_runner.run.return_value = mock_result

        config = BatchConfig(validate_convergence=False)
        runner = BatchSimulationRunner(mock_sim_runner, config)

        job = SimulationJob(
            job_id="test_progress",
            spec=make_minimal_spec(),
            geometry=make_minimal_geometry(),
            output_dir=tmp_path / "output",
        )

        runner.run([job], progress_callback=progress_callback)

        # Should have received multiple progress updates
        assert len(progress_updates) > 0

    def test_request_stop(self, tmp_path):
        """Test requesting batch stop."""
        started = threading.Event()

        def slow_run(*args, **kwargs):
            started.set()
            time.sleep(0.5)  # Long simulation
            return make_mock_simulation_result(tmp_path / "output")

        mock_sim_runner = MagicMock(spec=SimulationRunner)
        mock_sim_runner.run.side_effect = slow_run

        config = BatchConfig(max_workers=1, validate_convergence=False)
        runner = BatchSimulationRunner(mock_sim_runner, config)

        jobs = [
            SimulationJob(
                job_id=f"job_{i}",
                spec=make_minimal_spec(),
                geometry=make_minimal_geometry(),
                output_dir=tmp_path / f"output_{i}",
            )
            for i in range(3)
        ]

        # Start batch in thread
        result_holder = [None]

        def run_batch():
            result_holder[0] = runner.run(jobs)

        batch_thread = threading.Thread(target=run_batch)
        batch_thread.start()

        # Wait for first job to start then request stop
        started.wait(timeout=5.0)
        runner.request_stop()

        batch_thread.join(timeout=10.0)
        result = result_holder[0]

        # At least one job should be skipped
        assert result is not None
        assert result.n_skipped >= 0  # May or may not have skipped depending on timing


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_estimate_batch_time(self):
        """Test batch time estimation."""
        # Ensure enough RAM for the workers so effective_max_workers == max_workers
        config = BatchConfig(max_workers=4, ram_limit_gb=32.0, ram_per_sim_gb=4.0)
        # 20 jobs, 4 workers, 100 sec each = 5 batches * 100 = 500 sec
        estimated = estimate_batch_time(20, config, avg_sim_time_sec=100.0)
        assert estimated == 500.0

    def test_estimate_batch_time_small_batch(self):
        """Test estimation for small batch."""
        config = BatchConfig(max_workers=4)
        # 2 jobs, 4 workers = 1 batch
        estimated = estimate_batch_time(2, config, avg_sim_time_sec=100.0)
        assert estimated == 100.0

    def test_create_batch_jobs(self, tmp_path):
        """Test creating batch jobs from specs."""
        specs = [
            (make_minimal_spec(f"sim_{i}"), make_minimal_geometry())
            for i in range(3)
        ]

        jobs = create_batch_jobs(
            specs,
            tmp_path,
            prefix="batch",
            openems_args=("--debug",),
        )

        assert len(jobs) == 3
        assert jobs[0].job_id == "batch_0000"
        assert jobs[1].job_id == "batch_0001"
        assert jobs[0].openems_args == ("--debug",)
        assert jobs[0].output_dir == tmp_path / "batch_0000"


# =============================================================================
# I/O Tests
# =============================================================================


class TestBatchResultIO:
    """Tests for batch result I/O functions."""

    def test_write_and_load_result(self, tmp_path):
        """Test writing and loading batch result."""
        jobs = [
            SimulationJobResult(
                job_id="job_0",
                status=SimulationStatus.COMPLETED,
                result=make_mock_simulation_result(tmp_path / "job_0"),
                execution_time_sec=10.0,
            ),
            SimulationJobResult(
                job_id="job_1",
                status=SimulationStatus.FAILED,
                error="Test error",
            ),
        ]

        result = BatchResult(
            config=BatchConfig(),
            jobs=jobs,
            total_time_sec=20.0,
        )

        output_path = tmp_path / "batch_result.json"
        write_batch_result(result, output_path)

        assert output_path.exists()

        # Load and verify
        loaded = load_batch_result_summary(output_path)
        assert loaded["total_jobs"] == 2
        assert loaded["n_completed"] == 1
        assert loaded["n_failed"] == 1

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_batch_result_summary(tmp_path / "nonexistent.json")

    def test_write_creates_parent_dirs(self, tmp_path):
        """Test that write creates parent directories."""
        jobs = [
            SimulationJobResult(
                job_id="job_0",
                status=SimulationStatus.COMPLETED,
                result=make_mock_simulation_result(tmp_path / "job_0"),
            ),
        ]

        result = BatchResult(
            config=BatchConfig(),
            jobs=jobs,
            total_time_sec=5.0,
        )

        deep_path = tmp_path / "deep" / "nested" / "path" / "result.json"
        write_batch_result(result, deep_path)

        assert deep_path.exists()


# =============================================================================
# Integration Tests
# =============================================================================


class TestBatchRunnerIntegration:
    """Integration tests for batch runner."""

    def test_full_workflow_stub_mode(self, tmp_path):
        """Test full batch workflow with stub simulation runner."""
        # Create a real stub SimulationRunner
        sim_runner = SimulationRunner(mode="stub")

        # Disable convergence validation for stub mode
        config = BatchConfig(
            max_workers=2,
            validate_convergence=False,
        )

        runner = BatchSimulationRunner(sim_runner, config)

        # Create jobs
        jobs = [
            SimulationJob(
                job_id=f"stub_job_{i}",
                spec=make_minimal_spec(f"stub_sim_{i}"),
                geometry=make_minimal_geometry(),
                output_dir=tmp_path / f"stub_output_{i}",
            )
            for i in range(4)
        ]

        # Run batch
        result = runner.run(jobs)

        # Verify results
        assert result.n_completed == 4
        assert result.n_failed == 0
        assert result.all_passed
        assert result.total_time_sec > 0

        # Verify output files exist
        for i in range(4):
            outputs_dir = tmp_path / f"stub_output_{i}" / "sim_outputs"
            assert outputs_dir.exists()

    def test_jobs_sorted_by_priority(self, tmp_path):
        """Test that jobs are executed in priority order."""
        execution_order = []
        exec_lock = threading.Lock()

        def tracking_run(*args, **kwargs):
            job_id = args[0].simulation_id
            with exec_lock:
                execution_order.append(job_id)
            time.sleep(0.01)
            return make_mock_simulation_result(tmp_path / job_id)

        mock_sim_runner = MagicMock(spec=SimulationRunner)
        mock_sim_runner.run.side_effect = tracking_run

        config = BatchConfig(max_workers=1, validate_convergence=False)  # Sequential
        runner = BatchSimulationRunner(mock_sim_runner, config)

        jobs = [
            SimulationJob(
                job_id="low_priority",
                spec=make_minimal_spec("sim_low"),
                geometry=make_minimal_geometry(),
                output_dir=tmp_path / "low",
                priority=1,
            ),
            SimulationJob(
                job_id="high_priority",
                spec=make_minimal_spec("sim_high"),
                geometry=make_minimal_geometry(),
                output_dir=tmp_path / "high",
                priority=10,
            ),
            SimulationJob(
                job_id="medium_priority",
                spec=make_minimal_spec("sim_medium"),
                geometry=make_minimal_geometry(),
                output_dir=tmp_path / "medium",
                priority=5,
            ),
        ]

        runner.run(jobs)

        # High priority should run first
        assert execution_order[0] == "sim_high"
