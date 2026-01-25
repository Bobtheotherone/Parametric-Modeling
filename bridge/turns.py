"""Helpers for building schema-valid agent turns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

AGENTS: tuple[str, ...] = ("codex", "claude")
VALID_PHASES: tuple[str, ...] = ("plan", "implement", "verify", "finalize")
DEFAULT_STATS_REF = {"codex": "CX-1", "claude": "CL-1"}
DEFAULT_ERROR_PHASE = "plan"


def build_error_turn(
    *,
    agent: str,
    milestone_id: str,
    summary: str,
    error_detail: str | None = None,
    next_agent: str | None = None,
    next_prompt: str = "",
    delegate_rationale: str = "",
    stats_refs: Sequence[str] | None = None,
    stats_id_set: set[str] | None = None,
    phase: str | None = None,
    needs_write_access: bool = True,
    work_completed: bool = False,
    project_complete: bool = False,
    gates_passed: Sequence[str] | None = None,
    requirement_progress: Mapping[str, Sequence[str]] | None = None,
    artifacts: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Build a schema-valid error turn with safe defaults."""
    agent_id = agent if agent in AGENTS else str(agent).strip() or "codex"
    phase_val = phase if phase in VALID_PHASES else DEFAULT_ERROR_PHASE
    next_agent_val = next_agent if next_agent in AGENTS else ("claude" if agent_id == "codex" else "codex")
    stats = normalize_stats_refs(agent_id, stats_refs, stats_id_set)
    return {
        "agent": agent_id,
        "milestone_id": str(milestone_id),
        "phase": phase_val,
        "work_completed": bool(work_completed),
        "project_complete": bool(project_complete),
        "summary": _join_summary(summary, error_detail),
        "gates_passed": _to_str_list(gates_passed),
        "requirement_progress": normalize_requirement_progress(requirement_progress),
        "next_agent": next_agent_val,
        "next_prompt": str(next_prompt),
        "delegate_rationale": str(delegate_rationale),
        "stats_refs": stats,
        "needs_write_access": bool(needs_write_access),
        "artifacts": normalize_artifacts(artifacts),
    }


def error_turn(**kwargs: Any) -> dict[str, Any]:
    """Alias for build_error_turn for convenience."""
    return build_error_turn(**kwargs)


def normalize_stats_refs(
    agent: str,
    stats_refs: Sequence[str] | None,
    stats_id_set: set[str] | None = None,
) -> list[str]:
    """Normalize stats refs, ensuring at least one valid entry."""
    refs = _to_str_list(stats_refs)
    if stats_id_set:
        refs = [r for r in refs if r in stats_id_set]
    if refs:
        return refs
    fallback = DEFAULT_STATS_REF.get(agent, "CX-1")
    if stats_id_set:
        if fallback in stats_id_set:
            return [fallback]
        if stats_id_set:
            return [sorted(stats_id_set)[0]]
    return [fallback]


def normalize_requirement_progress(
    val: Mapping[str, Sequence[str]] | None,
) -> dict[str, list[str]]:
    """Normalize requirement progress entries to string lists."""
    rp = val if isinstance(val, Mapping) else {}
    return {
        "covered_req_ids": _to_str_list(rp.get("covered_req_ids")),
        "tests_added_or_modified": _to_str_list(rp.get("tests_added_or_modified")),
        "commands_run": _to_str_list(rp.get("commands_run")),
    }


def normalize_artifacts(val: Sequence[Mapping[str, str]] | None) -> list[dict[str, str]]:
    """Normalize artifact list to required schema shape."""
    if not isinstance(val, Sequence):
        return []
    out: list[dict[str, str]] = []
    for item in val:
        if not isinstance(item, Mapping):
            continue
        path = item.get("path")
        desc = item.get("description")
        if isinstance(path, str) and path.strip() and isinstance(desc, str):
            out.append({"path": path.strip(), "description": desc.strip()})
    return out


def _to_str_list(val: Any) -> list[str]:
    if not isinstance(val, Sequence) or isinstance(val, (str, bytes)):
        return []
    return [str(x).strip() for x in val if isinstance(x, str) and x.strip()]


def _join_summary(summary: str, error_detail: str | None) -> str:
    base = str(summary or "").strip()
    detail = str(error_detail or "").strip()
    if detail:
        return f"{base}\n{detail}" if base else detail
    return base


__all__ = [
    "AGENTS",
    "VALID_PHASES",
    "build_error_turn",
    "error_turn",
    "normalize_artifacts",
    "normalize_requirement_progress",
    "normalize_stats_refs",
]
