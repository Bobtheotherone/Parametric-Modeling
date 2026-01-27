"""Reporting utilities for orchestrator task runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class TaskLike(Protocol):
    id: str
    title: str
    status: str
    agent: str
    locks: list[str]
    touched_paths: list[str]
    depends_on: list[str]
    work_completed: bool | None
    turn_summary: str | None
    error: str | None
    manual_path: Path | None
    prompt_path: Path | None
    out_path: Path | None
    raw_log_path: Path | None
    patch_path: Path | None
    patch_manifest_path: Path | None
    task_dir: Path | None
    turn_obj: dict | None
    commit_sha: str | None
    commands_run: list[str] | None
    tests_added_or_modified: list[str] | None
    covered_req_ids: list[str] | None


@dataclass
class FailureClassification:
    category: str
    next_action: str


def classify_task_failure(status: str, error: str | None) -> FailureClassification:
    if status == "done":
        return FailureClassification(category="success", next_action="")

    error_text = (error or "").lower()

    if "non_progress" in error_text or "work_completed=false" in error_text:
        return FailureClassification(
            category="non_progress",
            next_action="Re-run the task manually with full implementation; ensure commands run and changes are produced.",
        )
    if status == "resource_killed" or "stopped for resources" in error_text:
        return FailureClassification(
            category="resource_killed",
            next_action="Re-run with reduced parallelism or --allow-resource-intensive; consider setting task solo.",
        )
    if "json validation" in error_text or "invalid json" in error_text:
        return FailureClassification(
            category="invalid_json",
            next_action="Fix the agent output to match the schema and re-run the task.",
        )
    if "agent exit code" in error_text:
        return FailureClassification(
            category="agent_error",
            next_action="Inspect raw logs for invocation/environment issues, then re-run the task.",
        )
    if "scope_rejected" in error_text:
        return FailureClassification(
            category="scope_rejected",
            next_action="Restrict changes to the allowed scope or adjust the task scope before re-running.",
        )
    if "merge conflict" in error_text:
        return FailureClassification(
            category="merge_conflict",
            next_action="Resolve conflicts manually or adjust locks to avoid overlap, then re-run.",
        )
    if status == "manual":
        return FailureClassification(
            category="manual_intervention",
            next_action="Follow the manual task file instructions and re-run the task.",
        )

    return FailureClassification(
        category="unknown_failure",
        next_action="Inspect logs, determine root cause, and re-run with a targeted fix.",
    )


def _load_patch_manifest(task: TaskLike) -> dict[str, Any] | None:
    manifest_path = task.patch_manifest_path
    if manifest_path is None and task.task_dir:
        candidate = task.task_dir / "changed_files.json"
        if candidate.exists():
            manifest_path = candidate
    if manifest_path and manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _diffstat_from_manifest(manifest: dict[str, Any] | None) -> dict[str, int]:
    if not manifest:
        return {"total": 0, "add": 0, "modify": 0, "delete": 0, "binary": 0}
    counts = {"total": 0, "add": 0, "modify": 0, "delete": 0, "binary": 0}
    for f in manifest.get("files", []) or []:
        op = str(f.get("operation", ""))
        counts["total"] += 1
        if op in counts:
            counts[op] += 1
    return counts


def _fmt_list(items: list[str]) -> str:
    return ", ".join(items) if items else "(none)"


def write_task_report(
    *,
    runs_dir: Path,
    task: TaskLike,
    agent_model: str | None,
    planner_profile: str,
) -> Path:
    report_dir = runs_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    safe_id = "".join(c if c.isalnum() or c in "-_" else "-" for c in task.id)
    report_path = report_dir / f"{safe_id}.md"

    classification = classify_task_failure(task.status, task.error)
    manifest = _load_patch_manifest(task)
    diffstat = _diffstat_from_manifest(manifest)

    files_changed = []
    if manifest:
        for f in manifest.get("files", []) or []:
            path = str(f.get("path", ""))
            op = str(f.get("operation", ""))
            if path:
                files_changed.append(f"{path} ({op})")

    commands_run = task.commands_run or []
    tests_added = task.tests_added_or_modified or []
    covered = task.covered_req_ids or []

    if not commands_run and task.turn_obj:
        req = task.turn_obj.get("requirement_progress", {}) or {}
        commands_run = list(req.get("commands_run", []) or [])
        tests_added = list(req.get("tests_added_or_modified", []) or [])
        covered = list(req.get("covered_req_ids", []) or [])

    turn_json = ""
    if task.turn_obj:
        try:
            turn_json = json.dumps(task.turn_obj, indent=2, sort_keys=True)
        except Exception:
            turn_json = ""

    lines = [
        f"# Task Report: {task.id} - {task.title}",
        "",
        "## Task Metadata",
        f"- Status: {task.status}",
        f"- Planner profile: {planner_profile}",
        f"- Agent: {task.agent} (model: {agent_model or 'unknown'})",
        f"- Dependencies: {_fmt_list([str(d) for d in task.depends_on])}",
        f"- Locks: {_fmt_list([str(l) for l in task.locks])}",
        f"- Touched paths: {_fmt_list([str(p) for p in task.touched_paths])}",
        f"- Work completed: {task.work_completed}",
        f"- Commit SHA: {task.commit_sha or 'none'}",
        "",
        "## Agent Output",
        f"- Summary: {task.turn_summary or '(none)'}",
        f"- Covered requirements: {_fmt_list([str(c) for c in covered])}",
        f"- Tests added/modified: {_fmt_list([str(t) for t in tests_added])}",
        f"- Commands run: {_fmt_list([str(c) for c in commands_run])}",
    ]

    if turn_json:
        lines.extend(["", "### Full Turn JSON", "```json", turn_json, "```"])
    else:
        lines.extend(["", "### Full Turn JSON", "(no valid JSON payload)"])

    lines.extend(
        [
            "",
            "## Changes",
            f"- Files changed: {_fmt_list(files_changed) if files_changed else 'no changes'}",
            f"- Diffstat: total={diffstat['total']} add={diffstat['add']} modify={diffstat['modify']} delete={diffstat['delete']} binary={diffstat['binary']}",
            f"- Patch artifact: {task.patch_path or 'none'}",
        ]
    )

    lines.extend(
        [
            "",
            "## Execution Artifacts",
            f"- Prompt: {task.prompt_path or 'none'}",
            f"- Output JSON: {task.out_path or 'none'}",
            f"- Raw log: {task.raw_log_path or 'none'}",
        ]
    )

    if task.status != "done":
        lines.extend(
            [
                "",
                "## Failure Analysis",
                f"- Root cause: {classification.category}",
                f"- Next action: {classification.next_action}",
                f"- Error: {task.error or 'none'}",
                f"- Manual file: {task.manual_path or 'none'}",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
