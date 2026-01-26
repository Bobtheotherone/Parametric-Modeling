# SPDX-License-Identifier: MIT
"""Edge case tests for substrate/runner.py module.

This module provides additional coverage for the LocalJobRunner, HardwareConfig,
ResourceRequest, and TaskSpec classes, focusing on:
- Validation edge cases (type errors, boundary values)
- HardwareConfig loading and conversion methods
- Token computation and resource allocation
- Error handling paths

Complements the core functionality tests in test_m0_runner.py.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from formula_foundry.substrate import runner


class TestResourceRequestValidation:
    """Edge case tests for ResourceRequest validation."""

    def test_cpu_threads_must_be_int(self) -> None:
        """cpu_threads must be an integer, not float."""
        with pytest.raises(TypeError, match="cpu_threads must be an int"):
            runner.ResourceRequest(cpu_threads=2.5, ram_gb=1.0, vram_gb=0.0)  # type: ignore[arg-type]

    def test_cpu_threads_must_be_positive(self) -> None:
        """cpu_threads must be at least 1."""
        with pytest.raises(ValueError, match="cpu_threads must be >= 1"):
            runner.ResourceRequest(cpu_threads=0, ram_gb=1.0, vram_gb=0.0)

    def test_ram_gb_must_be_non_negative(self) -> None:
        """ram_gb must be >= 0."""
        with pytest.raises(ValueError, match="ram_gb must be >= 0"):
            runner.ResourceRequest(cpu_threads=1, ram_gb=-0.5, vram_gb=0.0)

    def test_vram_gb_must_be_non_negative(self) -> None:
        """vram_gb must be >= 0."""
        with pytest.raises(ValueError, match="vram_gb must be >= 0"):
            runner.ResourceRequest(cpu_threads=1, ram_gb=1.0, vram_gb=-0.1)

    def test_ram_gb_coerced_to_float(self) -> None:
        """ram_gb integer values are coerced to float."""
        req = runner.ResourceRequest(cpu_threads=1, ram_gb=4, vram_gb=0)
        assert isinstance(req.ram_gb, float)
        assert req.ram_gb == 4.0

    def test_vram_gb_coerced_to_float(self) -> None:
        """vram_gb integer values are coerced to float."""
        req = runner.ResourceRequest(cpu_threads=1, ram_gb=1.0, vram_gb=2)
        assert isinstance(req.vram_gb, float)
        assert req.vram_gb == 2.0

    def test_zero_vram_allowed(self) -> None:
        """vram_gb of 0 is allowed (CPU-only task)."""
        req = runner.ResourceRequest(cpu_threads=1, ram_gb=1.0, vram_gb=0.0)
        assert req.vram_gb == 0.0

    def test_zero_ram_allowed(self) -> None:
        """ram_gb of 0 is allowed (minimal task)."""
        req = runner.ResourceRequest(cpu_threads=1, ram_gb=0.0, vram_gb=0.0)
        assert req.ram_gb == 0.0

    def test_frozen_dataclass(self) -> None:
        """ResourceRequest is immutable."""
        req = runner.ResourceRequest(cpu_threads=2, ram_gb=1.0, vram_gb=0.5)
        with pytest.raises(AttributeError):
            req.cpu_threads = 4  # type: ignore[misc]


class TestTaskSpecValidation:
    """Edge case tests for TaskSpec validation."""

    def test_task_id_must_be_non_empty(self) -> None:
        """task_id must be a non-empty string."""
        req = runner.ResourceRequest(cpu_threads=1, ram_gb=1.0, vram_gb=0.0)
        with pytest.raises(ValueError, match="task_id must be a non-empty string"):
            runner.TaskSpec(task_id="", resources=req, fn=lambda: None)

    def test_task_id_must_be_string(self) -> None:
        """task_id must be a string, not other types."""
        req = runner.ResourceRequest(cpu_threads=1, ram_gb=1.0, vram_gb=0.0)
        with pytest.raises(ValueError, match="task_id must be a non-empty string"):
            runner.TaskSpec(task_id=123, resources=req, fn=lambda: None)  # type: ignore[arg-type]

    def test_task_spec_frozen(self) -> None:
        """TaskSpec is immutable."""
        req = runner.ResourceRequest(cpu_threads=1, ram_gb=1.0, vram_gb=0.0)
        task = runner.TaskSpec(task_id="test", resources=req, fn=lambda: None)
        with pytest.raises(AttributeError):
            task.task_id = "new_id"  # type: ignore[misc]


class TestHardwareConfigValidation:
    """Edge case tests for HardwareConfig validation."""

    def test_cpu_cores_must_be_int(self) -> None:
        """cpu_cores must be an integer."""
        with pytest.raises(TypeError, match="cpu_cores must be an int"):
            runner.HardwareConfig(cpu_cores=4.5, ram_gb=8.0, vram_gb=0.0)  # type: ignore[arg-type]

    def test_cpu_cores_must_be_positive(self) -> None:
        """cpu_cores must be at least 1."""
        with pytest.raises(ValueError, match="cpu_cores must be >= 1"):
            runner.HardwareConfig(cpu_cores=0, ram_gb=8.0, vram_gb=0.0)

    def test_ram_gb_must_be_positive(self) -> None:
        """ram_gb must be > 0 for hardware config."""
        with pytest.raises(ValueError, match="ram_gb must be > 0"):
            runner.HardwareConfig(cpu_cores=4, ram_gb=0.0, vram_gb=0.0)

    def test_ram_gb_negative_rejected(self) -> None:
        """Negative ram_gb is rejected."""
        with pytest.raises(ValueError, match="ram_gb must be > 0"):
            runner.HardwareConfig(cpu_cores=4, ram_gb=-1.0, vram_gb=0.0)

    def test_vram_gb_must_be_non_negative(self) -> None:
        """vram_gb must be >= 0."""
        with pytest.raises(ValueError, match="vram_gb must be >= 0"):
            runner.HardwareConfig(cpu_cores=4, ram_gb=8.0, vram_gb=-1.0)

    def test_zero_vram_allowed(self) -> None:
        """vram_gb of 0 is allowed (no GPU)."""
        hw = runner.HardwareConfig(cpu_cores=4, ram_gb=8.0, vram_gb=0.0)
        assert hw.vram_gb == 0.0

    def test_values_coerced_to_float(self) -> None:
        """Integer ram_gb and vram_gb are coerced to float."""
        hw = runner.HardwareConfig(cpu_cores=4, ram_gb=8, vram_gb=2)
        assert isinstance(hw.ram_gb, float)
        assert isinstance(hw.vram_gb, float)
        assert hw.ram_gb == 8.0
        assert hw.vram_gb == 2.0


class TestHardwareConfigFromMapping:
    """Tests for HardwareConfig.from_mapping class method."""

    def test_from_mapping_valid(self) -> None:
        """from_mapping creates HardwareConfig from valid dict."""
        data = {"cpu_cores": 8, "ram_gb": 16.0, "vram_gb": 8.0}
        hw = runner.HardwareConfig.from_mapping(data)
        assert hw.cpu_cores == 8
        assert hw.ram_gb == 16.0
        assert hw.vram_gb == 8.0

    def test_from_mapping_missing_cpu_cores(self) -> None:
        """from_mapping fails if cpu_cores is missing."""
        data: dict[str, Any] = {"ram_gb": 16.0, "vram_gb": 8.0}
        with pytest.raises(ValueError, match="missing keys.*cpu_cores"):
            runner.HardwareConfig.from_mapping(data)

    def test_from_mapping_missing_ram_gb(self) -> None:
        """from_mapping fails if ram_gb is missing."""
        data: dict[str, Any] = {"cpu_cores": 8, "vram_gb": 8.0}
        with pytest.raises(ValueError, match="missing keys.*ram_gb"):
            runner.HardwareConfig.from_mapping(data)

    def test_from_mapping_missing_vram_gb(self) -> None:
        """from_mapping fails if vram_gb is missing."""
        data: dict[str, Any] = {"cpu_cores": 8, "ram_gb": 16.0}
        with pytest.raises(ValueError, match="missing keys.*vram_gb"):
            runner.HardwareConfig.from_mapping(data)

    def test_from_mapping_missing_multiple(self) -> None:
        """from_mapping reports all missing keys."""
        data: dict[str, Any] = {"vram_gb": 8.0}
        with pytest.raises(ValueError, match="missing keys"):
            runner.HardwareConfig.from_mapping(data)

    def test_from_mapping_extra_keys_allowed(self) -> None:
        """from_mapping allows extra keys in the mapping."""
        data = {"cpu_cores": 4, "ram_gb": 8.0, "vram_gb": 0.0, "extra_key": "ignored"}
        hw = runner.HardwareConfig.from_mapping(data)
        assert hw.cpu_cores == 4


class TestHardwareConfigTokenCapacity:
    """Tests for HardwareConfig.token_capacity method."""

    def test_token_capacity_default_granularity(self) -> None:
        """token_capacity with default 1GB granularity."""
        hw = runner.HardwareConfig(cpu_cores=4, ram_gb=8.0, vram_gb=2.0)
        tokens = hw.token_capacity()
        assert tokens.cpu == 4
        assert tokens.ram == 8  # 8GB / 1GB = 8 tokens
        assert tokens.vram == 2  # 2GB / 1GB = 2 tokens

    def test_token_capacity_custom_granularity(self) -> None:
        """token_capacity with custom granularity."""
        hw = runner.HardwareConfig(cpu_cores=4, ram_gb=8.0, vram_gb=2.5)
        tokens = hw.token_capacity(granularity_gb=0.5)
        assert tokens.cpu == 4
        assert tokens.ram == 16  # 8GB / 0.5GB = 16 tokens
        assert tokens.vram == 5  # 2.5GB / 0.5GB = 5 tokens

    def test_token_capacity_fractional_rounds_up(self) -> None:
        """Fractional tokens round up (ceiling)."""
        hw = runner.HardwareConfig(cpu_cores=2, ram_gb=2.5, vram_gb=0.1)
        tokens = hw.token_capacity(granularity_gb=1.0)
        assert tokens.ram == 3  # ceil(2.5) = 3
        assert tokens.vram == 1  # ceil(0.1) = 1

    def test_token_capacity_zero_vram(self) -> None:
        """Zero vram produces 0 vram tokens."""
        hw = runner.HardwareConfig(cpu_cores=2, ram_gb=4.0, vram_gb=0.0)
        tokens = hw.token_capacity()
        assert tokens.vram == 0


class TestLocalJobRunnerValidation:
    """Tests for LocalJobRunner initialization validation."""

    def test_memory_granularity_must_be_positive(self) -> None:
        """memory_granularity_gb must be > 0."""
        hw = runner.HardwareConfig(cpu_cores=4, ram_gb=8.0, vram_gb=0.0)
        with pytest.raises(ValueError, match="memory_granularity_gb must be > 0"):
            runner.LocalJobRunner(hw, memory_granularity_gb=0.0)

    def test_memory_granularity_negative_rejected(self) -> None:
        """Negative memory_granularity_gb is rejected."""
        hw = runner.HardwareConfig(cpu_cores=4, ram_gb=8.0, vram_gb=0.0)
        with pytest.raises(ValueError, match="memory_granularity_gb must be > 0"):
            runner.LocalJobRunner(hw, memory_granularity_gb=-0.5)


class TestLocalJobRunnerScheduleValidation:
    """Tests for LocalJobRunner.schedule resource validation."""

    def test_schedule_rejects_oversized_cpu(self) -> None:
        """Schedule rejects tasks requiring more CPU than available."""
        hw = runner.HardwareConfig(cpu_cores=2, ram_gb=8.0, vram_gb=1.0)
        local_runner = runner.LocalJobRunner(hw)
        task = runner.TaskSpec("big-cpu", runner.ResourceRequest(4, 1.0, 0.0), fn=lambda: None)
        with pytest.raises(ValueError, match="cpu"):
            local_runner.schedule([task])

    def test_schedule_rejects_oversized_ram(self) -> None:
        """Schedule rejects tasks requiring more RAM than available."""
        hw = runner.HardwareConfig(cpu_cores=4, ram_gb=4.0, vram_gb=0.0)
        local_runner = runner.LocalJobRunner(hw)
        task = runner.TaskSpec("big-ram", runner.ResourceRequest(1, 8.0, 0.0), fn=lambda: None)
        with pytest.raises(ValueError, match="RAM"):
            local_runner.schedule([task])

    def test_schedule_rejects_oversized_vram(self) -> None:
        """Schedule rejects tasks requiring more VRAM than available."""
        hw = runner.HardwareConfig(cpu_cores=4, ram_gb=8.0, vram_gb=2.0)
        local_runner = runner.LocalJobRunner(hw)
        task = runner.TaskSpec("big-vram", runner.ResourceRequest(1, 1.0, 4.0), fn=lambda: None)
        with pytest.raises(ValueError, match="VRAM"):
            local_runner.schedule([task])


class TestLocalJobRunnerRunWithLogs:
    """Tests for LocalJobRunner.run_with_logs method."""

    def test_run_with_logs_empty_tasks(self) -> None:
        """Empty task list returns empty results and logs."""
        hw = runner.HardwareConfig(cpu_cores=2, ram_gb=4.0, vram_gb=0.0)
        local_runner = runner.LocalJobRunner(hw)
        results, logs = local_runner.run_with_logs([])
        assert results == {}
        assert logs == []

    def test_run_with_logs_returns_logs(self) -> None:
        """run_with_logs returns execution logs."""
        hw = runner.HardwareConfig(cpu_cores=2, ram_gb=4.0, vram_gb=0.0)
        local_runner = runner.LocalJobRunner(hw)
        task = runner.TaskSpec("task-1", runner.ResourceRequest(1, 0.5, 0.0), fn=lambda: "done")
        results, logs = local_runner.run_with_logs([task])

        assert results["task-1"] == "done"
        assert len(logs) == 2  # start and finish events
        assert logs[0]["event"] == "start"
        assert logs[1]["event"] == "finish"
        assert logs[1]["status"] == "ok"

    def test_run_with_logs_captures_error_status(self) -> None:
        """run_with_logs captures error status in logs."""
        hw = runner.HardwareConfig(cpu_cores=2, ram_gb=4.0, vram_gb=0.0)
        local_runner = runner.LocalJobRunner(hw)

        def failing_fn() -> None:
            raise ValueError("intentional error")

        task = runner.TaskSpec("failing", runner.ResourceRequest(1, 0.5, 0.0), fn=failing_fn)

        with pytest.raises(ValueError, match="intentional error"):
            local_runner.run_with_logs([task])

        # Check last_run_log for error status
        assert len(local_runner.last_run_log) == 2
        finish_log = local_runner.last_run_log[1]
        assert finish_log["event"] == "finish"
        assert finish_log["status"] == "error"
        assert "intentional error" in finish_log["detail"]


class TestLoadHardwareConfig:
    """Tests for load_hardware_config function."""

    def test_load_hardware_config_file_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError when config file doesn't exist."""
        missing_path = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="hardware config not found"):
            runner.load_hardware_config(missing_path)

    def test_load_hardware_config_invalid_yaml_type(self, tmp_path: Path) -> None:
        """ValueError when YAML is not a mapping (e.g., a list)."""
        config_path = tmp_path / "bad_hardware.yaml"
        config_path.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            runner.load_hardware_config(config_path)

    def test_load_hardware_config_valid_yaml(self, tmp_path: Path) -> None:
        """load_hardware_config works with valid YAML file."""
        config_path = tmp_path / "hardware.yaml"
        config_path.write_text(
            "cpu_cores: 8\nram_gb: 32.0\nvram_gb: 16.0\n",
            encoding="utf-8",
        )
        hw = runner.load_hardware_config(config_path)
        assert hw.cpu_cores == 8
        assert hw.ram_gb == 32.0
        assert hw.vram_gb == 16.0


class TestGbToTokens:
    """Tests for _gb_to_tokens internal function."""

    def test_zero_amount_returns_zero(self) -> None:
        """Zero GB returns zero tokens."""
        assert runner._gb_to_tokens(0.0, 1.0) == 0

    def test_negative_amount_returns_zero(self) -> None:
        """Negative GB returns zero tokens."""
        assert runner._gb_to_tokens(-1.0, 1.0) == 0

    def test_exact_multiple(self) -> None:
        """Exact multiple returns exact token count."""
        assert runner._gb_to_tokens(4.0, 1.0) == 4
        assert runner._gb_to_tokens(2.0, 0.5) == 4

    def test_rounds_up(self) -> None:
        """Fractional tokens round up."""
        assert runner._gb_to_tokens(4.1, 1.0) == 5
        assert runner._gb_to_tokens(0.1, 1.0) == 1


class TestResourceTokens:
    """Tests for ResourceTokens dataclass."""

    def test_frozen_dataclass(self) -> None:
        """ResourceTokens is immutable."""
        tokens = runner.ResourceTokens(cpu=2, ram=4, vram=1)
        with pytest.raises(AttributeError):
            tokens.cpu = 8  # type: ignore[misc]
