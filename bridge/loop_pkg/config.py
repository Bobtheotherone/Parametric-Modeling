"""Configuration and state classes for the orchestration loop."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

AGENTS: tuple[str, ...] = ("codex", "claude")


@dataclasses.dataclass(frozen=True)
class ParallelSettings:
    """Settings for parallel worker execution."""

    max_workers_default: int = 8
    cpu_intensive_threshold_pct: float = 40.0
    mem_intensive_threshold_pct: float = 40.0
    sample_interval_s: float = 1.0
    consecutive_samples: int = 3
    kill_grace_s: float = 8.0
    terminal_max_bytes_per_worker: int = 40000
    terminal_max_line_length: int = 600
    disable_gpu_by_default: bool = True


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Configuration for an orchestration run."""

    max_calls_per_agent: int
    quota_retry_attempts: int
    max_total_calls: int
    max_json_correction_attempts: int
    fallback_order: list[str]
    enable_agents: list[str]
    smoke_route: tuple[str, ...]

    agent_scripts: dict[str, str]
    agent_models: dict[str, str]
    quota_error_patterns: dict[str, list[str]]
    supports_write_access: dict[str, bool]
    parallel: ParallelSettings


@dataclasses.dataclass
class RunState:
    """Mutable state for a single orchestration run."""

    run_id: str
    project_root: Path
    runs_dir: Path
    schema_path: Path
    system_prompt_path: Path
    design_doc_path: Path
    smoke_route: tuple[str, ...] = tuple()

    total_calls: int = 0
    call_counts: dict[str, int] = dataclasses.field(default_factory=lambda: {a: 0 for a in AGENTS})
    quota_failures: dict[str, int] = dataclasses.field(default_factory=lambda: {a: 0 for a in AGENTS})
    disabled_by_quota: dict[str, bool] = dataclasses.field(default_factory=lambda: {a: False for a in AGENTS})
    history: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    # Dynamic write access policy (set by previous turn)
    grant_write_access: bool = False
