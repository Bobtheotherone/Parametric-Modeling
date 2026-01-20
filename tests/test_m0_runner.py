from __future__ import annotations

import concurrent.futures
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from formula_foundry.substrate import runner


def test_runner_task_resource_schema() -> None:
    resources = runner.ResourceRequest(cpu_threads=2, ram_gb=1.5, vram_gb=0.0)
    task = runner.TaskSpec(task_id="task-1", resources=resources, fn=lambda: "ok")

    assert task.task_id == "task-1"
    assert task.resources.cpu_threads == 2
    assert task.resources.ram_gb == 1.5
    assert task.resources.vram_gb == 0.0

    with pytest.raises(ValueError, match="cpu_threads"):
        runner.ResourceRequest(cpu_threads=0, ram_gb=1.0, vram_gb=0.0)

    with pytest.raises(ValueError, match="ram_gb"):
        runner.ResourceRequest(cpu_threads=1, ram_gb=-1.0, vram_gb=0.0)


def test_runner_deterministic_schedule() -> None:
    hardware = runner.HardwareConfig(cpu_cores=2, ram_gb=4.0, vram_gb=1.0)
    local_runner = runner.LocalJobRunner(hardware)
    tasks = [
        runner.TaskSpec("task-a", runner.ResourceRequest(1, 1.0, 0.0), fn=lambda: "a"),
        runner.TaskSpec("task-b", runner.ResourceRequest(1, 0.5, 1.0), fn=lambda: "b"),
        runner.TaskSpec("task-c", runner.ResourceRequest(1, 0.5, 0.0), fn=lambda: "c"),
    ]

    schedule_one = [task.task_id for task in local_runner.schedule(tasks)]
    schedule_two = [task.task_id for task in local_runner.schedule(tasks)]

    assert schedule_one == schedule_two == ["task-a", "task-b", "task-c"]
    assert isinstance(local_runner.cpu_semaphore, threading.Semaphore)
    assert isinstance(local_runner.gpu_semaphore, threading.Semaphore)

    oversized = runner.TaskSpec("oversized", runner.ResourceRequest(3, 1.0, 0.0), fn=lambda: "boom")
    with pytest.raises(ValueError, match="cpu"):
        local_runner.schedule([oversized])


def test_hardware_yaml_contract() -> None:
    config_path = Path("config") / "hardware.yaml"
    assert config_path.exists()

    config = runner.load_hardware_config(config_path)

    assert config.cpu_cores > 0
    assert config.ram_gb > 0
    assert config.vram_gb >= 0


def test_runner_overlaps_non_overlapping_tasks() -> None:
    hardware = runner.HardwareConfig(cpu_cores=2, ram_gb=4.0, vram_gb=1.0)
    local_runner = runner.LocalJobRunner(hardware)

    def cpu_task() -> str:
        time.sleep(0.2)
        return "cpu"

    def gpu_task() -> str:
        time.sleep(0.2)
        return "gpu"

    tasks = [
        runner.TaskSpec("cpu", runner.ResourceRequest(1, 0.5, 0.0), fn=cpu_task),
        runner.TaskSpec("gpu", runner.ResourceRequest(1, 0.5, 1.0), fn=gpu_task),
    ]

    start = time.monotonic()
    results = local_runner.run(tasks)
    elapsed = time.monotonic() - start

    assert elapsed < 0.35
    assert results == {"cpu": "cpu", "gpu": "gpu"}


def test_runner_thread_pool_cap() -> None:
    """Verify that ThreadPoolExecutor max_workers is capped.

    For a large task list (e.g., 200 tasks), the runner must NOT attempt to create
    more threads than max(4, cpu_cores). This prevents thread explosion when
    scheduling thousands of tasks.
    """
    captured_max_workers: list[int] = []
    original_executor = concurrent.futures.ThreadPoolExecutor

    class CapturingExecutor(original_executor):  # type: ignore[valid-type,misc]
        def __init__(self, *args: object, max_workers: int | None = None, **kwargs: object) -> None:
            if max_workers is not None:
                captured_max_workers.append(max_workers)
            super().__init__(*args, max_workers=max_workers, **kwargs)

    hardware = runner.HardwareConfig(cpu_cores=4, ram_gb=32.0, vram_gb=0.0)
    local_runner = runner.LocalJobRunner(hardware)

    tasks = [runner.TaskSpec(f"task-{i}", runner.ResourceRequest(1, 0.1, 0.0), fn=lambda: "ok") for i in range(200)]

    with mock.patch(
        "formula_foundry.substrate.runner.concurrent.futures.ThreadPoolExecutor",
        CapturingExecutor,
    ):
        local_runner.run(tasks)

    assert len(captured_max_workers) == 1, "Expected exactly one ThreadPoolExecutor instantiation"
    actual_cap = captured_max_workers[0]
    expected_cap = max(4, hardware.cpu_cores)
    assert actual_cap == expected_cap, (
        f"ThreadPoolExecutor max_workers={actual_cap}, expected {expected_cap} (capped by max(4, cpu_cores={hardware.cpu_cores}))"
    )
    assert actual_cap < 200, "max_workers must NOT equal len(tasks) for large task lists"
