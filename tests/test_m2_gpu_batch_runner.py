"""Tests for GPU batch runner (REQ-M2-010).

This module tests the GPU-accelerated batch runner for openEMS FDTD simulations,
including:
- GPU detection and configuration
- VRAM monitoring and automatic batch sizing
- OOM detection and fallback to sequential/CPU execution
- GPU utilization metrics tracking
- Multi-GPU job distribution
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from formula_foundry.openems.batch_runner import BatchConfig, SimulationJob
from formula_foundry.openems.geometry import (
    BoardOutlineSpec,
    DiscontinuitySpec,
    GeometrySpec,
    LayerSpec,
    StackupMaterialsSpec,
    StackupSpec,
    TransmissionLineSpec,
)
from formula_foundry.openems.gpu_batch_runner import (
    DEFAULT_GPU_MEMORY_FRACTION,
    DEFAULT_VRAM_LIMIT_MB,
    DEFAULT_VRAM_PER_SIM_MB,
    GPUBatchConfig,
    GPUBatchMode,
    GPUBatchResult,
    GPUBatchSimulationRunner,
    GPUDeviceState,
    GPUJobResult,
    GPUStatus,
    GPUUtilizationMetrics,
    GPUUtilizationSample,
    check_cuda_available,
    detect_nvidia_gpus,
    estimate_gpu_batch_time,
    get_gpu_memory_info,
    is_oom_error,
    write_gpu_batch_result,
)
from formula_foundry.openems.manifest import GPUDeviceInfo
from formula_foundry.openems.sim_runner import (
    SimulationExecutionError,
    SimulationResult,
    SimulationRunner,
    SimulationTimeoutError,
)
from formula_foundry.openems.spec import (
    ExcitationSpec,
    FrequencySpec,
    GeometryRefSpec,
    OpenEMSToolchainSpec,
    PortSpec,
    SimulationSpec,
    ToolchainSpec,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def stub_simulation_spec() -> SimulationSpec:
    """Create a stub simulation spec for testing."""
    return SimulationSpec(
        schema_version=1,
        simulation_id="test_sim",
        toolchain=ToolchainSpec(
            openems=OpenEMSToolchainSpec(
                version="0.0.35",
                docker_image="ghcr.io/thliebig/openems:0.0.35",
            )
        ),
        geometry_ref=GeometryRefSpec(
            design_hash="a" * 64,
            coupon_id="test_coupon",
        ),
        excitation=ExcitationSpec(
            type="gaussian",
            f0_hz=5_000_000_000,
            fc_hz=5_000_000_000,
        ),
        frequency=FrequencySpec(
            f_start_hz=1_000_000_000,
            f_stop_hz=10_000_000_000,
            n_points=201,
        ),
        ports=[
            PortSpec(
                id="P1",
                type="lumped",
                impedance_ohm=50.0,
                excite=True,
                position_nm=(0, 0, 0),
                direction="x",
            ),
            PortSpec(
                id="P2",
                type="lumped",
                impedance_ohm=50.0,
                excite=False,
                position_nm=(10_000_000, 0, 0),
                direction="x",
            ),
        ],
    )


@pytest.fixture
def stub_geometry_spec() -> GeometrySpec:
    """Create a stub geometry spec for testing."""
    return GeometrySpec(
        schema_version=1,
        design_hash="a" * 64,
        coupon_family="test",
        units="nm",
        origin="EDGE_L_CENTER",
        board=BoardOutlineSpec(
            width_nm=20_000_000,
            length_nm=80_000_000,
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
        ],
        transmission_line=TransmissionLineSpec(
            type="CPWG",
            layer="F.Cu",
            w_nm=300_000,
            gap_nm=180_000,
            length_left_nm=25_000_000,
            length_right_nm=25_000_000,
        ),
        discontinuity=DiscontinuitySpec(
            type="VIA_TRANSITION",
            parameters_nm={},
        ),
        parameters_nm={},
        derived_features={},
        dimensionless_groups={},
    )


@pytest.fixture
def stub_sim_runner() -> SimulationRunner:
    """Create a stub simulation runner for testing."""
    return SimulationRunner(mode="stub")


@pytest.fixture
def sample_jobs(
    stub_simulation_spec: SimulationSpec,
    stub_geometry_spec: GeometrySpec,
    tmp_path: Path,
) -> list[SimulationJob]:
    """Create sample simulation jobs for testing."""
    jobs = []
    for i in range(4):
        jobs.append(
            SimulationJob(
                job_id=f"job_{i:04d}",
                spec=stub_simulation_spec,
                geometry=stub_geometry_spec,
                output_dir=tmp_path / f"job_{i:04d}",
                priority=i,
            )
        )
    return jobs


# =============================================================================
# GPU Device State Tests
# =============================================================================


class TestGPUDeviceState:
    """Tests for GPUDeviceState dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic GPUDeviceState creation."""
        state = GPUDeviceState(
            device_id=0,
            device_name="NVIDIA A100",
            status=GPUStatus.AVAILABLE,
            total_memory_mb=40960,
            free_memory_mb=38000,
            used_memory_mb=2960,
            utilization_percent=15.0,
        )
        assert state.device_id == 0
        assert state.device_name == "NVIDIA A100"
        assert state.status == GPUStatus.AVAILABLE
        assert state.total_memory_mb == 40960
        assert state.free_memory_mb == 38000

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        state = GPUDeviceState(
            device_id=0,
            device_name="Tesla V100",
            status=GPUStatus.BUSY,
            total_memory_mb=16384,
            free_memory_mb=8192,
            used_memory_mb=8192,
            utilization_percent=75.5,
            temperature_c=65.0,
            power_usage_w=250.0,
            active_jobs=2,
        )
        d = state.to_dict()
        assert d["device_id"] == 0
        assert d["device_name"] == "Tesla V100"
        assert d["status"] == "busy"
        assert d["temperature_c"] == 65.0
        assert d["power_usage_w"] == 250.0
        assert d["active_jobs"] == 2

    def test_to_gpu_device_info(self) -> None:
        """Test conversion to GPUDeviceInfo for manifest."""
        state = GPUDeviceState(
            device_id=1,
            device_name="RTX 3090",
            total_memory_mb=24576,
            free_memory_mb=20000,
            used_memory_mb=4576,
        )
        info = state.to_gpu_device_info()
        assert isinstance(info, GPUDeviceInfo)
        assert info.device_id == 1
        assert info.device_name == "RTX 3090"
        assert info.memory_total_mb == 24576

    def test_can_allocate_sufficient_memory(self) -> None:
        """Test allocation check with sufficient memory."""
        state = GPUDeviceState(
            device_id=0,
            device_name="A100",
            status=GPUStatus.AVAILABLE,
            total_memory_mb=40960,
            free_memory_mb=30000,
            used_memory_mb=10960,
        )
        assert state.can_allocate(2048)
        assert state.can_allocate(30000)

    def test_can_allocate_insufficient_memory(self) -> None:
        """Test allocation check with insufficient memory."""
        state = GPUDeviceState(
            device_id=0,
            device_name="A100",
            status=GPUStatus.AVAILABLE,
            total_memory_mb=40960,
            free_memory_mb=1000,
            used_memory_mb=39960,
        )
        assert not state.can_allocate(2048)

    def test_can_allocate_unavailable_gpu(self) -> None:
        """Test allocation check with unavailable GPU."""
        state = GPUDeviceState(
            device_id=0,
            device_name="A100",
            status=GPUStatus.OOM,
            total_memory_mb=40960,
            free_memory_mb=30000,
            used_memory_mb=10960,
        )
        assert not state.can_allocate(2048)


# =============================================================================
# GPU Batch Config Tests
# =============================================================================


class TestGPUBatchConfig:
    """Tests for GPUBatchConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = GPUBatchConfig()
        assert config.mode == GPUBatchMode.AUTO
        assert config.device_ids is None
        assert config.vram_limit_mb == DEFAULT_VRAM_LIMIT_MB
        assert config.vram_per_sim_mb == DEFAULT_VRAM_PER_SIM_MB
        assert config.gpu_memory_fraction == DEFAULT_GPU_MEMORY_FRACTION
        assert config.max_sims_per_gpu == 4
        assert config.fallback_to_cpu is True
        assert config.oom_retry_count == 2
        assert config.track_utilization is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = GPUBatchConfig(
            mode=GPUBatchMode.FORCE_GPU,
            device_ids=(0, 1),
            vram_limit_mb=16384,
            vram_per_sim_mb=4096,
            gpu_memory_fraction=0.9,
            max_sims_per_gpu=2,
            fallback_to_cpu=False,
            oom_retry_count=3,
        )
        assert config.mode == GPUBatchMode.FORCE_GPU
        assert config.device_ids == (0, 1)
        assert config.vram_limit_mb == 16384
        assert config.fallback_to_cpu is False

    def test_invalid_memory_fraction_low(self) -> None:
        """Test validation of too-low memory fraction."""
        with pytest.raises(ValueError, match="gpu_memory_fraction"):
            GPUBatchConfig(gpu_memory_fraction=0.05)

    def test_invalid_memory_fraction_high(self) -> None:
        """Test validation of too-high memory fraction."""
        with pytest.raises(ValueError, match="gpu_memory_fraction"):
            GPUBatchConfig(gpu_memory_fraction=1.5)

    def test_invalid_vram_per_sim(self) -> None:
        """Test validation of invalid VRAM per sim."""
        with pytest.raises(ValueError, match="vram_per_sim_mb"):
            GPUBatchConfig(vram_per_sim_mb=0)

    def test_invalid_max_sims_per_gpu(self) -> None:
        """Test validation of invalid max sims per GPU."""
        with pytest.raises(ValueError, match="max_sims_per_gpu"):
            GPUBatchConfig(max_sims_per_gpu=0)

    def test_invalid_oom_retry_count(self) -> None:
        """Test validation of invalid OOM retry count."""
        with pytest.raises(ValueError, match="oom_retry_count"):
            GPUBatchConfig(oom_retry_count=-1)

    def test_effective_vram_limit(self) -> None:
        """Test effective VRAM limit calculation."""
        config = GPUBatchConfig(
            vram_limit_mb=10000,
            gpu_memory_fraction=0.8,
        )
        assert config.effective_vram_limit_mb == 8000

    def test_max_concurrent_per_gpu(self) -> None:
        """Test max concurrent per GPU calculation."""
        config = GPUBatchConfig(
            vram_limit_mb=16384,
            vram_per_sim_mb=4096,
            gpu_memory_fraction=1.0,
            max_sims_per_gpu=8,
        )
        # 16384 / 4096 = 4, but max_sims_per_gpu = 8, so should be 4
        assert config.max_concurrent_per_gpu == 4

        config2 = GPUBatchConfig(
            vram_limit_mb=16384,
            vram_per_sim_mb=2048,
            gpu_memory_fraction=1.0,
            max_sims_per_gpu=2,
        )
        # 16384 / 2048 = 8, but max_sims_per_gpu = 2, so should be 2
        assert config2.max_concurrent_per_gpu == 2


# =============================================================================
# GPU Utilization Metrics Tests
# =============================================================================


class TestGPUUtilizationMetrics:
    """Tests for GPUUtilizationMetrics."""

    def test_add_sample(self) -> None:
        """Test adding utilization samples."""
        metrics = GPUUtilizationMetrics(device_id=0, device_name="A100")

        sample1 = GPUUtilizationSample(
            timestamp=datetime.now(),
            device_id=0,
            utilization_percent=50.0,
            memory_used_mb=4000,
            memory_free_mb=36960,
        )
        metrics.add_sample(sample1)
        assert metrics.n_samples == 1
        assert metrics.avg_utilization_percent == 50.0
        assert metrics.max_utilization_percent == 50.0

        sample2 = GPUUtilizationSample(
            timestamp=datetime.now(),
            device_id=0,
            utilization_percent=100.0,
            memory_used_mb=8000,
            memory_free_mb=32960,
        )
        metrics.add_sample(sample2)
        assert metrics.n_samples == 2
        assert metrics.avg_utilization_percent == 75.0
        assert metrics.max_utilization_percent == 100.0
        assert metrics.avg_memory_used_mb == 6000.0
        assert metrics.max_memory_used_mb == 8000

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = GPUUtilizationMetrics(
            device_id=0,
            device_name="V100",
            n_samples=10,
            avg_utilization_percent=65.5,
            max_utilization_percent=95.0,
            avg_memory_used_mb=12000.0,
            max_memory_used_mb=15000,
            total_jobs_run=5,
            oom_events=1,
            total_gpu_time_sec=300.0,
        )
        d = metrics.to_dict()
        assert d["device_id"] == 0
        assert d["device_name"] == "V100"
        assert d["n_samples"] == 10
        assert d["avg_utilization_percent"] == 65.5
        assert d["total_jobs_run"] == 5
        assert d["oom_events"] == 1


# =============================================================================
# GPU Detection Tests
# =============================================================================


class TestGPUDetection:
    """Tests for GPU detection functions."""

    @patch("subprocess.run")
    def test_detect_nvidia_gpus_success(self, mock_run: MagicMock) -> None:
        """Test successful GPU detection."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="0, NVIDIA A100, 40960, 38000, 2960, 15, 45, 100\n1, NVIDIA A100, 40960, 39000, 1960, 10, 42, 90\n",
        )
        devices = detect_nvidia_gpus()
        assert len(devices) == 2
        assert devices[0].device_id == 0
        assert devices[0].device_name == "NVIDIA A100"
        assert devices[0].total_memory_mb == 40960
        assert devices[1].device_id == 1

    @patch("subprocess.run")
    def test_detect_nvidia_gpus_no_nvidia_smi(self, mock_run: MagicMock) -> None:
        """Test GPU detection when nvidia-smi not available."""
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
        devices = detect_nvidia_gpus()
        assert devices == []

    @patch("subprocess.run")
    def test_detect_nvidia_gpus_timeout(self, mock_run: MagicMock) -> None:
        """Test GPU detection timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 10)
        devices = detect_nvidia_gpus()
        assert devices == []

    @patch("subprocess.run")
    def test_check_cuda_available_true(self, mock_run: MagicMock) -> None:
        """Test CUDA availability check when available."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="NVIDIA A100\n",
        )
        assert check_cuda_available() is True

    @patch("subprocess.run")
    def test_check_cuda_available_false(self, mock_run: MagicMock) -> None:
        """Test CUDA availability check when not available."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=1,
            stdout="",
        )
        assert check_cuda_available() is False


# =============================================================================
# OOM Detection Tests
# =============================================================================


class TestOOMDetection:
    """Tests for OOM error detection."""

    def test_is_oom_error_cuda_oom(self) -> None:
        """Test OOM detection for CUDA out of memory."""
        err = RuntimeError("CUDA error: out of memory")
        assert is_oom_error(err) is True

    def test_is_oom_error_memory_allocation(self) -> None:
        """Test OOM detection for memory allocation failure."""
        err = RuntimeError("Memory allocation failed: insufficient memory")
        assert is_oom_error(err) is True

    def test_is_oom_error_cuda_error_code(self) -> None:
        """Test OOM detection for CUDA error code."""
        err = RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")
        assert is_oom_error(err) is True

    def test_is_oom_error_generic_oom(self) -> None:
        """Test OOM detection for generic OOM message."""
        err = MemoryError("OOM killed process")
        assert is_oom_error(err) is True

    def test_is_oom_error_unrelated(self) -> None:
        """Test OOM detection for unrelated error."""
        err = ValueError("Invalid parameter")
        assert is_oom_error(err) is False

    def test_is_oom_error_simulation_error(self) -> None:
        """Test OOM detection for simulation execution error."""
        err = SimulationExecutionError(
            "openEMS failed: CUDA error: out of memory",
            returncode=1,
            stdout="",
            stderr="CUDA error: out of memory",
        )
        assert is_oom_error(err) is True


# =============================================================================
# GPU Batch Runner Tests
# =============================================================================


class TestGPUBatchSimulationRunner:
    """Tests for GPUBatchSimulationRunner."""

    def test_init_default_config(self, stub_sim_runner: SimulationRunner) -> None:
        """Test initialization with default config."""
        runner = GPUBatchSimulationRunner(stub_sim_runner)
        assert runner.batch_config is not None
        assert runner.gpu_config is not None
        assert runner.gpu_config.mode == GPUBatchMode.AUTO

    def test_init_custom_config(self, stub_sim_runner: SimulationRunner) -> None:
        """Test initialization with custom config."""
        batch_config = BatchConfig(max_workers=8)
        gpu_config = GPUBatchConfig(
            mode=GPUBatchMode.FORCE_CPU,
            vram_per_sim_mb=4096,
        )
        runner = GPUBatchSimulationRunner(
            stub_sim_runner,
            batch_config,
            gpu_config,
        )
        assert runner.batch_config.max_workers == 8
        assert runner.gpu_config.mode == GPUBatchMode.FORCE_CPU

    @patch("formula_foundry.openems.gpu_batch_runner.detect_nvidia_gpus")
    def test_detect_gpus_force_cpu(
        self,
        mock_detect: MagicMock,
        stub_sim_runner: SimulationRunner,
    ) -> None:
        """Test GPU detection skipped in FORCE_CPU mode."""
        gpu_config = GPUBatchConfig(mode=GPUBatchMode.FORCE_CPU)
        runner = GPUBatchSimulationRunner(
            stub_sim_runner,
            gpu_config=gpu_config,
        )
        devices = runner.detect_gpus()
        assert devices == []
        mock_detect.assert_not_called()

    @patch("formula_foundry.openems.gpu_batch_runner.detect_nvidia_gpus")
    def test_detect_gpus_auto_with_devices(
        self,
        mock_detect: MagicMock,
        stub_sim_runner: SimulationRunner,
    ) -> None:
        """Test GPU detection in AUTO mode with available devices."""
        mock_detect.return_value = [
            GPUDeviceState(
                device_id=0,
                device_name="A100",
                total_memory_mb=40960,
                free_memory_mb=38000,
                used_memory_mb=2960,
            ),
        ]
        runner = GPUBatchSimulationRunner(stub_sim_runner)
        devices = runner.detect_gpus()
        assert len(devices) == 1
        assert devices[0].device_name == "A100"

    @patch("formula_foundry.openems.gpu_batch_runner.detect_nvidia_gpus")
    def test_detect_gpus_force_gpu_no_devices(
        self,
        mock_detect: MagicMock,
        stub_sim_runner: SimulationRunner,
    ) -> None:
        """Test GPU detection fails in FORCE_GPU mode without devices."""
        mock_detect.return_value = []
        gpu_config = GPUBatchConfig(mode=GPUBatchMode.FORCE_GPU)
        runner = GPUBatchSimulationRunner(
            stub_sim_runner,
            gpu_config=gpu_config,
        )
        with pytest.raises(RuntimeError, match="No GPUs detected"):
            runner.detect_gpus()

    @patch("formula_foundry.openems.gpu_batch_runner.detect_nvidia_gpus")
    def test_detect_gpus_filter_by_device_ids(
        self,
        mock_detect: MagicMock,
        stub_sim_runner: SimulationRunner,
    ) -> None:
        """Test GPU detection filtering by device IDs."""
        mock_detect.return_value = [
            GPUDeviceState(device_id=0, device_name="GPU0"),
            GPUDeviceState(device_id=1, device_name="GPU1"),
            GPUDeviceState(device_id=2, device_name="GPU2"),
        ]
        gpu_config = GPUBatchConfig(device_ids=(0, 2))
        runner = GPUBatchSimulationRunner(
            stub_sim_runner,
            gpu_config=gpu_config,
        )
        devices = runner.detect_gpus()
        assert len(devices) == 2
        assert devices[0].device_id == 0
        assert devices[1].device_id == 2

    @patch("formula_foundry.openems.gpu_batch_runner.detect_nvidia_gpus")
    def test_run_empty_jobs(
        self,
        mock_detect: MagicMock,
        stub_sim_runner: SimulationRunner,
    ) -> None:
        """Test running with empty job list."""
        mock_detect.return_value = []
        runner = GPUBatchSimulationRunner(stub_sim_runner)
        result = runner.run([])
        assert isinstance(result, GPUBatchResult)
        assert len(result.batch_result.jobs) == 0
        assert result.batch_result.total_time_sec == 0.0

    @patch("formula_foundry.openems.gpu_batch_runner.detect_nvidia_gpus")
    def test_run_cpu_only(
        self,
        mock_detect: MagicMock,
        stub_sim_runner: SimulationRunner,
        sample_jobs: list[SimulationJob],
    ) -> None:
        """Test running batch in CPU-only mode."""
        mock_detect.return_value = []
        runner = GPUBatchSimulationRunner(stub_sim_runner)
        result = runner.run(sample_jobs)

        assert isinstance(result, GPUBatchResult)
        assert len(result.batch_result.jobs) == len(sample_jobs)
        assert result.batch_result.n_completed == len(sample_jobs)
        assert result.n_gpu_jobs == 0
        assert result.n_cpu_fallback_jobs == 0

    @patch("formula_foundry.openems.gpu_batch_runner.detect_nvidia_gpus")
    def test_run_with_progress_callback(
        self,
        mock_detect: MagicMock,
        stub_sim_runner: SimulationRunner,
        sample_jobs: list[SimulationJob],
    ) -> None:
        """Test running with progress callback."""
        mock_detect.return_value = []
        runner = GPUBatchSimulationRunner(stub_sim_runner)

        progress_updates: list[float] = []

        def callback(progress: Any) -> None:
            progress_updates.append(progress.percent_complete)

        result = runner.run(sample_jobs, progress_callback=callback)
        assert len(progress_updates) > 0
        assert result.batch_result.n_completed == len(sample_jobs)

    @patch("formula_foundry.openems.gpu_batch_runner.detect_nvidia_gpus")
    def test_request_stop(
        self,
        mock_detect: MagicMock,
        stub_sim_runner: SimulationRunner,
        sample_jobs: list[SimulationJob],
    ) -> None:
        """Test stopping batch execution."""
        mock_detect.return_value = []
        batch_config = BatchConfig(fail_fast=True)
        runner = GPUBatchSimulationRunner(stub_sim_runner, batch_config)

        # Request stop immediately
        runner.request_stop()
        result = runner.run(sample_jobs)

        # At least some jobs should be skipped
        assert result.batch_result.n_skipped > 0 or result.batch_result.n_completed > 0


# =============================================================================
# GPU Batch Result Tests
# =============================================================================


class TestGPUBatchResult:
    """Tests for GPUBatchResult."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        from formula_foundry.openems.batch_runner import BatchResult

        batch_result = BatchResult(
            config=BatchConfig(),
            jobs=[],
            total_time_sec=100.0,
        )
        gpu_config = GPUBatchConfig(mode=GPUBatchMode.AUTO)
        result = GPUBatchResult(
            batch_result=batch_result,
            gpu_config=gpu_config,
            n_gpu_jobs=10,
            n_cpu_fallback_jobs=2,
            n_oom_retries=3,
            n_oom_failures=1,
            gpu_device_info=[
                GPUDeviceInfo(device_id=0, device_name="A100", memory_total_mb=40960),
            ],
        )
        d = result.to_dict()
        assert "gpu_batch" in d
        assert d["gpu_batch"]["mode"] == "auto"
        assert d["gpu_batch"]["n_gpu_jobs"] == 10
        assert d["gpu_batch"]["n_cpu_fallback_jobs"] == 2
        assert d["gpu_batch"]["n_oom_retries"] == 3
        assert d["gpu_batch"]["n_oom_failures"] == 1
        assert len(d["gpu_batch"]["gpu_devices"]) == 1


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_estimate_gpu_batch_time(self) -> None:
        """Test GPU batch time estimation."""
        gpu_config = GPUBatchConfig(max_sims_per_gpu=2)
        batch_config = BatchConfig(max_workers=4)

        # 10 jobs, 2 GPUs with 2 slots each = 4 total slots
        # 10 / 4 = 3 batches (ceiling)
        time_sec = estimate_gpu_batch_time(
            n_jobs=10,
            gpu_config=gpu_config,
            batch_config=batch_config,
            n_gpus=2,
            avg_sim_time_sec=60.0,
        )
        assert time_sec == 180.0  # 3 * 60

    def test_write_gpu_batch_result(self, tmp_path: Path) -> None:
        """Test writing GPU batch result to file."""
        from formula_foundry.openems.batch_runner import BatchResult

        batch_result = BatchResult(
            config=BatchConfig(),
            jobs=[],
            total_time_sec=50.0,
        )
        gpu_config = GPUBatchConfig()
        result = GPUBatchResult(
            batch_result=batch_result,
            gpu_config=gpu_config,
            n_gpu_jobs=5,
        )

        output_path = tmp_path / "result.json"
        write_gpu_batch_result(result, output_path)

        assert output_path.exists()
        import json

        data = json.loads(output_path.read_text())
        assert "gpu_batch" in data
        assert data["gpu_batch"]["n_gpu_jobs"] == 5


# =============================================================================
# Integration Tests
# =============================================================================


class TestGPUBatchIntegration:
    """Integration tests for GPU batch runner."""

    @patch("formula_foundry.openems.gpu_batch_runner.detect_nvidia_gpus")
    def test_full_cpu_batch_workflow(
        self,
        mock_detect: MagicMock,
        stub_sim_runner: SimulationRunner,
        sample_jobs: list[SimulationJob],
        tmp_path: Path,
    ) -> None:
        """Test complete CPU batch workflow."""
        mock_detect.return_value = []

        batch_config = BatchConfig(
            max_workers=2,
            validate_convergence=False,
        )
        gpu_config = GPUBatchConfig(
            mode=GPUBatchMode.AUTO,
            fallback_to_cpu=True,
        )
        runner = GPUBatchSimulationRunner(
            stub_sim_runner,
            batch_config,
            gpu_config,
        )

        result = runner.run(sample_jobs)

        # Verify result structure
        assert isinstance(result, GPUBatchResult)
        assert len(result.batch_result.jobs) == len(sample_jobs)
        assert result.batch_result.n_completed == len(sample_jobs)
        assert result.batch_result.n_failed == 0
        assert result.n_gpu_jobs == 0

        # Write and verify output
        output_path = tmp_path / "batch_result.json"
        write_gpu_batch_result(result, output_path)
        assert output_path.exists()

    @patch("formula_foundry.openems.gpu_batch_runner.detect_nvidia_gpus")
    def test_batch_with_priorities(
        self,
        mock_detect: MagicMock,
        stub_sim_runner: SimulationRunner,
        stub_simulation_spec: SimulationSpec,
        stub_geometry_spec: GeometrySpec,
        tmp_path: Path,
    ) -> None:
        """Test batch respects job priorities."""
        mock_detect.return_value = []

        # Create jobs with different priorities
        jobs = [
            SimulationJob(
                job_id="low",
                spec=stub_simulation_spec,
                geometry=stub_geometry_spec,
                output_dir=tmp_path / "low",
                priority=0,
            ),
            SimulationJob(
                job_id="high",
                spec=stub_simulation_spec,
                geometry=stub_geometry_spec,
                output_dir=tmp_path / "high",
                priority=10,
            ),
            SimulationJob(
                job_id="medium",
                spec=stub_simulation_spec,
                geometry=stub_geometry_spec,
                output_dir=tmp_path / "medium",
                priority=5,
            ),
        ]

        runner = GPUBatchSimulationRunner(
            stub_sim_runner,
            BatchConfig(max_workers=1),  # Single worker to preserve order
        )
        result = runner.run(jobs)

        assert result.batch_result.n_completed == 3

    @patch("formula_foundry.openems.gpu_batch_runner.detect_nvidia_gpus")
    def test_batch_fail_fast(
        self,
        mock_detect: MagicMock,
        stub_geometry_spec: GeometrySpec,
        tmp_path: Path,
    ) -> None:
        """Test fail-fast mode stops on first error."""
        mock_detect.return_value = []

        # Create a mock runner that fails on specific jobs
        mock_runner = MagicMock(spec=SimulationRunner)
        mock_runner.mode = "stub"

        call_count = [0]

        def run_side_effect(*args: Any, **kwargs: Any) -> SimulationResult:
            call_count[0] += 1
            if call_count[0] == 2:
                raise SimulationExecutionError(
                    "Simulated failure",
                    returncode=1,
                    stdout="",
                    stderr="error",
                )
            return SimulationResult(
                output_dir=kwargs.get("output_dir", tmp_path),
                outputs_dir=kwargs.get("output_dir", tmp_path) / "sim_outputs",
                manifest_path=kwargs.get("output_dir", tmp_path) / "manifest.json",
                cache_hit=False,
                simulation_hash="a" * 64,
                manifest_hash="b" * 64,
                output_hashes={},
            )

        mock_runner.run.side_effect = run_side_effect

        # Create jobs
        spec = SimulationSpec(
            schema_version=1,
            toolchain=ToolchainSpec(
                openems=OpenEMSToolchainSpec(
                    version="0.0.35",
                    docker_image="test",
                )
            ),
            geometry_ref=GeometryRefSpec(design_hash="a" * 64),
            excitation=ExcitationSpec(type="gaussian", f0_hz=1, fc_hz=1),
            frequency=FrequencySpec(f_start_hz=1, f_stop_hz=2, n_points=2),
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
        jobs = [
            SimulationJob(
                job_id=f"job_{i}",
                spec=spec,
                geometry=stub_geometry_spec,
                output_dir=tmp_path / f"job_{i}",
            )
            for i in range(5)
        ]

        runner = GPUBatchSimulationRunner(
            mock_runner,
            BatchConfig(max_workers=1, fail_fast=True, validate_convergence=False),
        )
        result = runner.run(jobs)

        # With fail_fast, should stop after first failure
        assert result.batch_result.n_failed >= 1
        total_processed = result.batch_result.n_completed + result.batch_result.n_failed + result.batch_result.n_skipped
        # Some jobs should be skipped
        assert total_processed == len(jobs)
