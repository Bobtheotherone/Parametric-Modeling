"""Tests for engineering planner profile behavior."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from bridge.loop import (
    AgentCapabilities,
    ParallelSettings,
    ParallelTask,
    RunConfig,
    RunState,
    run_parallel,
    _build_parallel_task_prompt,
    _detect_non_progress_turn,
    _load_system_prompt,
    _max_retries_for_profile,
    _maybe_write_task_report,
    _should_self_heal_task,
)


def _make_config() -> RunConfig:
    return RunConfig(
        max_calls_per_agent=1,
        quota_retry_attempts=1,
        max_total_calls=1,
        max_json_correction_attempts=1,
        fallback_order=["codex", "claude"],
        enable_agents=["codex", "claude"],
        smoke_route=tuple(),
        agent_scripts={"codex": "", "claude": ""},
        agent_models={"codex": "test-model", "claude": "test-model"},
        quota_error_patterns={"codex": [], "claude": []},
        supports_write_access={"codex": True, "claude": True},
        agent_capabilities={"codex": AgentCapabilities(), "claude": AgentCapabilities()},
        parallel=ParallelSettings(),
    )


def test_engineering_prompt_composition_includes_agents_and_task_packet(tmp_path: Path) -> None:
    agents_text = "AGENTS-PROTOCOL-TEST"
    system_text = "ENGINEERING-SYSTEM-PROMPT"

    (tmp_path / "AGENTS.md").write_text(agents_text, encoding="utf-8")
    prompt_dir = tmp_path / "bridge" / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "system_engineering.md").write_text(system_text, encoding="utf-8")
    (prompt_dir / "system.md").write_text("DEFAULT", encoding="utf-8")

    system_prompt, effective_path = _load_system_prompt(
        project_root=tmp_path,
        system_prompt_path=tmp_path / "bridge" / "prompts" / "system.md",
        planner_profile="engineering",
    )

    assert "ENGINEERING-SYSTEM-PROMPT" in system_prompt
    assert "AGENTS-PROTOCOL-TEST" in system_prompt
    assert effective_path.name == "system_engineering.md"

    task = ParallelTask(
        id="M2-TEST-001",
        title="Do thing",
        description="Implement the thing fully",
        agent="codex",
    )

    prompt_text = _build_parallel_task_prompt(
        system_prompt=system_prompt,
        task=task,
        worker_id=1,
        milestone_id="M2",
        repo_snapshot="",
        design_doc_text="",
        resource_policy={},
        planner_profile="engineering",
    )

    assert "Task Packet" in prompt_text
    assert "Repo Protocol" in prompt_text


def test_non_progress_detection_and_retry_policy() -> None:
    turn_obj = {
        "work_completed": False,
        "summary": "wrapper_status=ok",
        "requirement_progress": {"commands_run": []},
    }
    non_progress, reason = _detect_non_progress_turn(turn_obj, has_changes=False)
    assert non_progress
    assert "no commands" in reason
    assert _max_retries_for_profile("engineering") == 0


def test_engineering_self_heal_gating() -> None:
    failed_task = ParallelTask(
        id="M2-FAIL",
        title="Fail",
        description="",
        agent="codex",
        status="failed",
    )
    failed_task.error = "Some generic failure"

    allowed, reason = _should_self_heal_task(failed_task, "engineering")
    assert not allowed
    assert reason == "non_infra_failure"

    resource_task = ParallelTask(
        id="M2-RES",
        title="Res",
        description="",
        agent="codex",
        status="resource_killed",
    )
    resource_task.error = "Stopped for resources"

    allowed, reason = _should_self_heal_task(resource_task, "engineering")
    assert allowed
    assert reason == "resource_killed"


def test_task_report_written_in_engineering(tmp_path: Path) -> None:
    task = ParallelTask(
        id="M2-REPORT",
        title="Report",
        description="",
        agent="codex",
        status="done",
    )
    task.task_dir = tmp_path / "task"
    task.task_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = task.task_dir / "changed_files.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {"path": "src/app.py", "operation": "modify", "sha256": "abc"},
                    {"path": "tests/test_app.py", "operation": "add", "sha256": "def"},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    task.turn_obj = {
        "summary": "Did work",
        "work_completed": True,
        "requirement_progress": {"commands_run": ["pytest -q"], "tests_added_or_modified": ["tests/test_app.py"]},
    }
    task.turn_summary = "Did work"

    _maybe_write_task_report(
        task=task,
        runs_dir=tmp_path,
        planner_profile="engineering",
        config=_make_config(),
    )

    report_path = tmp_path / "reports" / "M2-REPORT.md"
    assert report_path.exists(), "Engineering mode should write a task report"
    content = report_path.read_text(encoding="utf-8")
    assert "Task Report" in content
    assert "Files changed" in content


def test_run_parallel_preflight_writes_run_json(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(
        run_id="RUN-PREFLIGHT",
        project_root=tmp_path,
        runs_dir=runs_dir,
        schema_path=tmp_path / "schema.json",
        system_prompt_path=tmp_path / "system.md",
        design_doc_path=tmp_path / "DESIGN_DOCUMENT.md",
    )
    config = _make_config()

    def _fake_preflight(*_args, **_kwargs):
        return False, "preflight failed", None

    monkeypatch.setattr("bridge.loop._preflight_check_repo", _fake_preflight)

    rc = run_parallel(
        args=SimpleNamespace(),
        config=config,
        state=state,
        stats_ids=[],
        stats_id_set=set(),
        system_prompt="",
    )

    assert rc == 2
    run_json_path = runs_dir / "run.json"
    assert run_json_path.exists()
    payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "preflight_failed"
    assert payload["error"] == "preflight failed"
    assert payload["run_id"] == "RUN-PREFLIGHT"
