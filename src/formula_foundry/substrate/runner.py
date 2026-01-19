from __future__ import annotations

import concurrent.futures
import importlib
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MEMORY_GRANULARITY_GB = 1.0
DEFAULT_HARDWARE_PATH = Path(__file__).resolve().parents[3] / "config" / "hardware.yaml"


@dataclass(frozen=True)
class ResourceRequest:
    cpu_threads: int
    ram_gb: float
    vram_gb: float

    def __post_init__(self) -> None:
        if not isinstance(self.cpu_threads, int):
            raise TypeError("cpu_threads must be an int")
        if self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be >= 1")
        ram_gb = float(self.ram_gb)
        vram_gb = float(self.vram_gb)
        if ram_gb < 0:
            raise ValueError("ram_gb must be >= 0")
        if vram_gb < 0:
            raise ValueError("vram_gb must be >= 0")
        object.__setattr__(self, "ram_gb", ram_gb)
        object.__setattr__(self, "vram_gb", vram_gb)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    resources: ResourceRequest
    fn: Callable[[], Any]

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a non-empty string")


@dataclass(frozen=True)
class ResourceTokens:
    cpu: int
    ram: int
    vram: int


@dataclass(frozen=True)
class HardwareConfig:
    cpu_cores: int
    ram_gb: float
    vram_gb: float

    def __post_init__(self) -> None:
        if not isinstance(self.cpu_cores, int):
            raise TypeError("cpu_cores must be an int")
        if self.cpu_cores <= 0:
            raise ValueError("cpu_cores must be >= 1")
        ram_gb = float(self.ram_gb)
        vram_gb = float(self.vram_gb)
        if ram_gb <= 0:
            raise ValueError("ram_gb must be > 0")
        if vram_gb < 0:
            raise ValueError("vram_gb must be >= 0")
        object.__setattr__(self, "ram_gb", ram_gb)
        object.__setattr__(self, "vram_gb", vram_gb)

    def token_capacity(self, *, granularity_gb: float = DEFAULT_MEMORY_GRANULARITY_GB) -> ResourceTokens:
        return ResourceTokens(
            cpu=self.cpu_cores,
            ram=_gb_to_tokens(self.ram_gb, granularity_gb),
            vram=_gb_to_tokens(self.vram_gb, granularity_gb),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> HardwareConfig:
        if "cpu_cores" not in data or "ram_gb" not in data or "vram_gb" not in data:
            missing = {key for key in ("cpu_cores", "ram_gb", "vram_gb") if key not in data}
            raise ValueError(f"hardware config missing keys: {sorted(missing)}")
        return cls(
            cpu_cores=data["cpu_cores"],
            ram_gb=data["ram_gb"],
            vram_gb=data["vram_gb"],
        )

    @classmethod
    def from_yaml(cls, path: Path) -> HardwareConfig:
        payload = _load_yaml(path)
        return cls.from_mapping(payload)


class LocalJobRunner:
    def __init__(self, hardware: HardwareConfig, *, memory_granularity_gb: float = DEFAULT_MEMORY_GRANULARITY_GB) -> None:
        if memory_granularity_gb <= 0:
            raise ValueError("memory_granularity_gb must be > 0")
        self.hardware = hardware
        self.memory_granularity_gb = memory_granularity_gb
        self._capacity = hardware.token_capacity(granularity_gb=memory_granularity_gb)
        self.cpu_semaphore = threading.Semaphore(self._capacity.cpu)
        self.ram_semaphore = threading.Semaphore(self._capacity.ram)
        self.gpu_semaphore = threading.Semaphore(self._capacity.vram)
        self.last_run_log: list[dict[str, Any]] = []

    def schedule(self, tasks: Sequence[TaskSpec]) -> list[TaskSpec]:
        scheduled: list[TaskSpec] = []
        for task in tasks:
            tokens = self._tokens_for_request(task.resources)
            self._validate_tokens(tokens, task_id=task.task_id)
            scheduled.append(task)
        return scheduled

    def run(self, tasks: Sequence[TaskSpec]) -> dict[str, Any]:
        results, _ = self.run_with_logs(tasks)
        return results

    def run_with_logs(self, tasks: Sequence[TaskSpec]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        scheduled = self.schedule(tasks)
        if not scheduled:
            self.last_run_log = []
            return {}, []
        results: dict[str, Any] = {}
        logs: list[dict[str, Any]] = []
        log_lock = threading.Lock()
        seq = 0

        def log_event(task: TaskSpec, event: str, *, status: str | None = None, detail: str | None = None) -> None:
            nonlocal seq
            with log_lock:
                seq += 1
                entry: dict[str, Any] = {
                    "seq": seq,
                    "ts": time.monotonic(),
                    "event": event,
                    "task_id": task.task_id,
                    "resources": {
                        "cpu_threads": task.resources.cpu_threads,
                        "ram_gb": task.resources.ram_gb,
                        "vram_gb": task.resources.vram_gb,
                    },
                }
                if status:
                    entry["status"] = status
                if detail:
                    entry["detail"] = detail
                logs.append(entry)

        def run_task(task: TaskSpec) -> Any:
            tokens = self._tokens_for_request(task.resources)
            acquired = self._acquire_resources(tokens)
            log_event(task, "start")
            status = "ok"
            detail = None
            try:
                return task.fn()
            except Exception as exc:
                status = "error"
                detail = str(exc)
                raise
            finally:
                self._release_resources(acquired)
                log_event(task, "finish", status=status, detail=detail)

        first_error: Exception | None = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(scheduled)) as executor:
            future_map = {executor.submit(run_task, task): task for task in scheduled}
            for future in concurrent.futures.as_completed(future_map):
                task = future_map[future]
                try:
                    results[task.task_id] = future.result()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
        self.last_run_log = sorted(logs, key=lambda entry: entry["seq"])
        if first_error is not None:
            raise first_error
        return results, self.last_run_log

    def _tokens_for_request(self, request: ResourceRequest) -> ResourceTokens:
        return ResourceTokens(
            cpu=request.cpu_threads,
            ram=_gb_to_tokens(request.ram_gb, self.memory_granularity_gb),
            vram=_gb_to_tokens(request.vram_gb, self.memory_granularity_gb),
        )

    def _validate_tokens(self, tokens: ResourceTokens, *, task_id: str) -> None:
        if tokens.cpu > self._capacity.cpu:
            raise ValueError(f"Task {task_id!r} requires {tokens.cpu} cpu threads; limit is {self._capacity.cpu}")
        if tokens.ram > self._capacity.ram:
            raise ValueError(f"Task {task_id!r} requires {tokens.ram} RAM tokens; limit is {self._capacity.ram}")
        if tokens.vram > self._capacity.vram:
            raise ValueError(f"Task {task_id!r} requires {tokens.vram} VRAM tokens; limit is {self._capacity.vram}")

    def _acquire_resources(self, tokens: ResourceTokens) -> list[tuple[threading.Semaphore, int]]:
        acquired: list[tuple[threading.Semaphore, int]] = []
        for semaphore, count in (
            (self.cpu_semaphore, tokens.cpu),
            (self.ram_semaphore, tokens.ram),
            (self.gpu_semaphore, tokens.vram),
        ):
            _acquire_units(semaphore, count)
            acquired.append((semaphore, count))
        return acquired

    def _release_resources(self, acquired: list[tuple[threading.Semaphore, int]]) -> None:
        for semaphore, count in reversed(acquired):
            _release_units(semaphore, count)


def load_hardware_config(path: Path | None = None) -> HardwareConfig:
    resolved = path or DEFAULT_HARDWARE_PATH
    return HardwareConfig.from_yaml(resolved)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"hardware config not found: {path}")
    yaml_module = _import_yaml()
    payload = yaml_module.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("hardware config must be a YAML mapping")
    return payload


def _gb_to_tokens(amount_gb: float, granularity_gb: float) -> int:
    if amount_gb <= 0:
        return 0
    return int(math.ceil(amount_gb / granularity_gb))


def _acquire_units(semaphore: threading.Semaphore, count: int) -> None:
    for _ in range(count):
        semaphore.acquire()


def _release_units(semaphore: threading.Semaphore, count: int) -> None:
    for _ in range(count):
        semaphore.release()


def _import_yaml() -> Any:
    try:
        return importlib.import_module("yaml")
    except Exception as exc:  # pragma: no cover - dependency failure should be obvious
        raise RuntimeError("pyyaml is required to load hardware.yaml") from exc
