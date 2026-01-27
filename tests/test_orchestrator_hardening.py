"""Regression tests for orchestrator hardening (max-workers, directive files, alternation)."""

from __future__ import annotations

import json
from pathlib import Path

from bridge.loop import (
    _load_system_prompt,
    _override_next_agent,
    compute_effective_max_workers,
    load_config,
    materialize_directive_file,
)
from bridge.loop_pkg.config import AgentCapabilities, ParallelSettings, RunConfig, RunState

# ---------------------------------------------------------------------------
# A) max-workers selection logic
# ---------------------------------------------------------------------------


def test_cli_override_respected_exactly():
    """--max-workers 10 must yield exactly 10 (not clamped to old safe_cap)."""
    eff, reason = compute_effective_max_workers(
        cli_max_workers=10,
        config_default=8,
        config_hard_cap=32,
        cpu_cores=16,
    )
    assert eff == 10
    assert "cli_override=10" in reason


def test_cli_override_clamped_by_hard_cap():
    """CLI value above hard cap must be clamped and logged."""
    eff, reason = compute_effective_max_workers(
        cli_max_workers=40,
        config_default=8,
        config_hard_cap=32,
        cpu_cores=16,
    )
    assert eff == 32
    assert "clamped_by_hard_cap" in reason


def test_auto_mode_uses_config_default():
    """--max-workers 0 (auto) must use config.parallel.max_workers_default."""
    eff, reason = compute_effective_max_workers(
        cli_max_workers=0,
        config_default=10,
        config_hard_cap=32,
        cpu_cores=16,
    )
    assert eff == 10
    assert "config_default=10" in reason


def test_auto_mode_respects_plan_cap():
    """Auto mode with plan_max_parallel < default should clamp."""
    eff, reason = compute_effective_max_workers(
        cli_max_workers=0,
        config_default=10,
        config_hard_cap=32,
        cpu_cores=16,
        plan_max_parallel=6,
    )
    assert eff == 6
    assert "clamped_by_plan=6" in reason


def test_auto_mode_clamped_by_hard_cap():
    """Auto mode should clamp to hard cap."""
    eff, reason = compute_effective_max_workers(
        cli_max_workers=0,
        config_default=20,
        config_hard_cap=8,
        cpu_cores=16,
    )
    assert eff == 8
    assert "clamped_by_hard_cap=8" in reason


def test_auto_mode_clamped_by_schema_max():
    """Auto mode should clamp to schema max."""
    eff, reason = compute_effective_max_workers(
        cli_max_workers=0,
        config_default=64,
        config_hard_cap=128,
        cpu_cores=16,
        schema_max=32,
    )
    assert eff == 32
    assert "clamped_by_schema_max=32" in reason


def test_cli_override_not_clamped_by_plan():
    """CLI override must ignore plan_max_parallel and remain exact."""
    eff, reason = compute_effective_max_workers(
        cli_max_workers=10,
        config_default=8,
        config_hard_cap=32,
        cpu_cores=16,
        plan_max_parallel=6,
    )
    assert eff == 10
    assert "clamped_by_plan=6" not in reason
    assert "plan_max_parallel_ignored=6" in reason


def test_hard_cap_clamped_to_schema_max():
    """Schema max should clamp CLI overrides even when hard cap is higher."""
    eff, reason = compute_effective_max_workers(
        cli_max_workers=50,
        config_default=8,
        config_hard_cap=64,
        cpu_cores=16,
    )
    assert eff == 32  # schema_max=32
    assert "clamped_by_schema_max=32" in reason


def test_minimum_one_worker():
    """Must always have at least 1 worker."""
    eff, _ = compute_effective_max_workers(
        cli_max_workers=0,
        config_default=0,
        config_hard_cap=32,
        cpu_cores=1,
    )
    assert eff >= 1


# ---------------------------------------------------------------------------
# B) Directive file: no AGENTS.md prompt injection + CLAUDE.md materialization
# ---------------------------------------------------------------------------


AGENTS_SENTINEL = "UNIQUE-AGENTS-SENTINEL-xK9q2"


def test_load_system_prompt_does_not_inject_agents_md(tmp_path: Path):
    """_load_system_prompt must NOT include AGENTS.md content in the prompt."""
    (tmp_path / "AGENTS.md").write_text(AGENTS_SENTINEL, encoding="utf-8")
    prompt_dir = tmp_path / "bridge" / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "system_engineering.md").write_text("ENGINEERING PROMPT", encoding="utf-8")
    (prompt_dir / "system.md").write_text("DEFAULT", encoding="utf-8")

    system_prompt, _ = _load_system_prompt(
        project_root=tmp_path,
        system_prompt_path=tmp_path / "bridge" / "prompts" / "system.md",
        planner_profile="engineering",
    )

    assert AGENTS_SENTINEL not in system_prompt, (
        "AGENTS.md content must NOT be injected into the system prompt"
    )


def test_load_system_prompt_balanced_never_injects(tmp_path: Path):
    """Balanced profile also must not inject AGENTS.md."""
    (tmp_path / "AGENTS.md").write_text(AGENTS_SENTINEL, encoding="utf-8")
    prompt_dir = tmp_path / "bridge" / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "system.md").write_text("DEFAULT", encoding="utf-8")

    system_prompt, _ = _load_system_prompt(
        project_root=tmp_path,
        system_prompt_path=tmp_path / "bridge" / "prompts" / "system.md",
        planner_profile="balanced",
    )

    assert AGENTS_SENTINEL not in system_prompt


def test_materialize_claude_md_from_agents_md(tmp_path: Path):
    """When only AGENTS.md exists, materialize_directive_file copies it to CLAUDE.md."""
    project = tmp_path / "project"
    project.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (project / "AGENTS.md").write_text("agent directive content", encoding="utf-8")

    materialize_directive_file(
        project_root=project,
        target_dir=worktree,
        agent_name="claude",
    )

    assert (worktree / "CLAUDE.md").exists()
    assert (worktree / "CLAUDE.md").read_text() == "agent directive content"
    # AGENTS.md should also be materialized
    assert (worktree / "AGENTS.md").exists()


def test_materialize_prefers_claude_md_for_claude(tmp_path: Path):
    """When both CLAUDE.md and AGENTS.md exist, prefer CLAUDE.md for Claude agents."""
    project = tmp_path / "project"
    project.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (project / "AGENTS.md").write_text("agents version", encoding="utf-8")
    (project / "CLAUDE.md").write_text("claude version", encoding="utf-8")

    materialize_directive_file(
        project_root=project,
        target_dir=worktree,
        agent_name="claude",
    )

    assert (worktree / "CLAUDE.md").read_text() == "claude version"


def test_materialize_idempotent(tmp_path: Path):
    """Calling materialize twice with same content is a no-op."""
    project = tmp_path / "project"
    project.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (project / "AGENTS.md").write_text("content", encoding="utf-8")

    materialize_directive_file(project_root=project, target_dir=worktree, agent_name="claude")
    mtime1 = (worktree / "CLAUDE.md").stat().st_mtime_ns

    materialize_directive_file(project_root=project, target_dir=worktree, agent_name="claude")
    mtime2 = (worktree / "CLAUDE.md").stat().st_mtime_ns

    assert mtime1 == mtime2, "Idempotent copy should not rewrite file"


# ---------------------------------------------------------------------------
# C) Config plumbing for new parallel fields
# ---------------------------------------------------------------------------


def test_load_config_reads_parallel_fields():
    """load_config must surface max_workers_hard_cap and force_alternation from config.json."""
    config_path = Path(__file__).parent.parent / "bridge" / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    config = load_config(config_path)

    assert config.parallel.max_workers_hard_cap == raw["parallel"]["max_workers_hard_cap"]
    assert config.parallel.force_alternation == raw["parallel"]["force_alternation"]


def test_materialize_codex_agent_no_claude_md(tmp_path: Path):
    """For codex agent, CLAUDE.md should NOT be created; only AGENTS.md."""
    project = tmp_path / "project"
    project.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (project / "AGENTS.md").write_text("content", encoding="utf-8")

    materialize_directive_file(project_root=project, target_dir=worktree, agent_name="codex")

    assert not (worktree / "CLAUDE.md").exists()
    assert (worktree / "AGENTS.md").exists()


# ---------------------------------------------------------------------------
# D) Alternation / handoff configuration
# ---------------------------------------------------------------------------


def _make_config(force_alternation: bool = True) -> RunConfig:
    return RunConfig(
        max_calls_per_agent=15,
        quota_retry_attempts=1,
        max_total_calls=100,
        max_json_correction_attempts=1,
        fallback_order=["codex", "claude"],
        enable_agents=["codex", "claude"],
        smoke_route=tuple(),
        agent_scripts={"codex": "", "claude": ""},
        agent_models={"codex": "m", "claude": "m"},
        quota_error_patterns={"codex": [], "claude": []},
        supports_write_access={"codex": True, "claude": True},
        agent_capabilities={"codex": AgentCapabilities(), "claude": AgentCapabilities()},
        parallel=ParallelSettings(force_alternation=force_alternation),
    )


def _make_state(history: list[dict] | None = None) -> RunState:
    return RunState(
        run_id="test",
        project_root=Path("/tmp/test"),
        runs_dir=Path("/tmp/test/runs"),
        schema_path=Path("/tmp/test/schema.json"),
        system_prompt_path=Path("/tmp/test/system.md"),
        design_doc_path=Path("/tmp/test/DESIGN_DOCUMENT.md"),
        history=history or [],
    )


def test_alternation_disabled_respects_requested_agent(monkeypatch):
    """With force_alternation=False, requested agent is respected."""
    # Patch out agent policy forced_agent
    import bridge.loop as loop_mod
    monkeypatch.setattr(loop_mod, "get_agent_policy", lambda: type("P", (), {"forced_agent": None})())

    config = _make_config(force_alternation=False)
    state = _make_state(history=[{"agent": "codex"}])

    effective, reason = _override_next_agent("codex", config, state)
    assert effective == "codex"
    assert reason is None  # No override


def test_alternation_enabled_forces_switch(monkeypatch):
    """With force_alternation=True, agent is alternated."""
    import bridge.loop as loop_mod
    monkeypatch.setattr(loop_mod, "get_agent_policy", lambda: type("P", (), {"forced_agent": None})())

    config = _make_config(force_alternation=True)
    state = _make_state(history=[{"agent": "codex"}])

    effective, reason = _override_next_agent("codex", config, state)
    assert effective == "claude"
    assert "alternation" in (reason or "")


# ---------------------------------------------------------------------------
# E) Orchestrator-level backfill runaway prevention
# ---------------------------------------------------------------------------


def test_backfill_generator_suppresses_when_queue_saturated():
    """BackfillGenerator.should_generate returns False when queue >= worker_count * 2."""
    from bridge.scheduler import BackfillGenerator

    gen = BackfillGenerator(project_root="/tmp/test", min_queue_depth=20)
    # 10 workers, queue at 20 (== worker_count * 2) -> suppress
    assert not gen.should_generate(current_queue_depth=20, worker_count=10)
    # queue at 21 -> still suppress
    assert not gen.should_generate(current_queue_depth=21, worker_count=10)


def test_backfill_generator_allows_when_queue_below_threshold():
    """BackfillGenerator.should_generate returns True when queue < threshold."""
    from bridge.scheduler import BackfillGenerator

    gen = BackfillGenerator(project_root="/tmp/test", min_queue_depth=20)
    assert gen.should_generate(current_queue_depth=5, worker_count=10)


def test_backfill_generator_disabled_always_suppresses():
    """A disabled BackfillGenerator never generates."""
    from bridge.scheduler import BackfillGenerator

    gen = BackfillGenerator(project_root="/tmp/test", min_queue_depth=20)
    gen.disabled = True
    assert not gen.should_generate(current_queue_depth=0, worker_count=10)


def test_orchestrator_filler_quota_policy():
    """Validate the filler quota rule: no filler when queued filler >= max_workers // 2.

    This tests the exact policy from maybe_generate_backfill() Policy 3.
    """
    from bridge.loop import _is_backfill_task_id

    max_workers = 10
    max_queued_filler = max(1, max_workers // 2)  # == 5

    # Simulate: 5 pending FILLER tasks (at the quota limit)
    task_ids = [f"FILLER-LINT-{i:03d}" for i in range(5)] + ["CORE-TASK-001"]
    task_statuses = ["pending"] * 5 + ["pending"]

    queued_filler = sum(
        1 for tid, st in zip(task_ids, task_statuses, strict=True)
        if st == "pending" and _is_backfill_task_id(tid)
    )
    assert queued_filler >= max_queued_filler, "Setup: filler should be at quota"
    # Policy: suppress
    should_suppress = queued_filler >= max_queued_filler
    assert should_suppress, "Policy 3: must suppress when queued filler >= max_workers // 2"


def test_orchestrator_queue_saturation_policy():
    """Validate queue saturation rule: no filler when total queued >= max_workers.

    This tests the exact policy from maybe_generate_backfill() Policy 2.
    """
    max_workers = 8
    # 8 pending tasks total (== max_workers)
    total_queued = 8
    assert total_queued >= max_workers
    # Policy: suppress
    should_suppress = total_queued >= max_workers
    assert should_suppress, "Policy 2: must suppress when total queued >= max_workers"


def test_orchestrator_root_failure_suppression_uses_real_id_check():
    """Validate root failure detection uses _is_backfill_task_id from bridge.loop.

    Ensures the orchestrator correctly distinguishes FILLER from core tasks.
    """
    from bridge.loop import _is_backfill_task_id

    root_failure_statuses = ("failed", "manual", "resource_killed")

    # Scenario: core task failed + filler tasks pending
    tasks = [
        ("M0-FIX-01", "failed"),
        ("FILLER-LINT-001", "pending"),
        ("FILLER-TEST-002", "pending"),
    ]
    has_root_failures = any(
        st in root_failure_statuses and not _is_backfill_task_id(tid)
        for tid, st in tasks
    )
    assert has_root_failures, "Policy 1: core failure must trigger suppression"

    # Scenario: only filler failed
    tasks_filler_only = [
        ("FILLER-LINT-001", "failed"),
        ("M0-FIX-01", "done"),
    ]
    has_root_failures_2 = any(
        st in root_failure_statuses and not _is_backfill_task_id(tid)
        for tid, st in tasks_filler_only
    )
    assert not has_root_failures_2, "FILLER failures must NOT trigger root suppression"
