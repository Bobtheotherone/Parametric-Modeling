#!/usr/bin/env python3
"""Formula Foundry tri-agent orchestration loop.

What this loop enforces (by design):
- Milestone specs are *normative* and live in DESIGN_DOCUMENT.md.
- Every normative requirement must map to at least one pytest in the Test Matrix.
- Agents must iterate until `python -m tools.verify --strict-git` passes.
- The run is not considered complete unless the repo is committed and clean.

This loop drives 3 coding agents:
- Codex CLI (writes code)
- Gemini CLI (spec engineer / advisory)
- Claude Code (writes code)

All agent handoffs are strict JSON validated against bridge/turn.schema.json.

NOTE on prompt size:
The DESIGN_DOCUMENT.md for later milestones can become large. To keep the most
important information visible to *all* agents (including those without repo
filesystem access), the orchestrator injects a "spec excerpt" that always
includes:
- milestone header + preamble
- Normative Requirements
- Definition of Done
- Test Matrix

The excerpt is truncated only after these critical sections are included.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

# When run as `python bridge/loop.py`, Python sets sys.path[0] to the `bridge/`
# directory. We want to import sibling packages (e.g., `tools`) from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AGENTS = ("codex", "gemini", "claude")


@dataclasses.dataclass
class RunConfig:
    max_calls_per_agent: int
    quota_retry_attempts: int
    max_total_calls: int
    max_json_correction_attempts: int
    fallback_order: list[str]
    agent_scripts: dict[str, str]
    quota_error_patterns: dict[str, list[str]]
    supports_write_access: dict[str, bool]


@dataclasses.dataclass
class RunState:
    run_id: str
    project_root: Path
    runs_dir: Path
    schema_path: Path
    system_prompt_path: Path
    design_doc_path: Path

    total_calls: int = 0
    call_counts: dict[str, int] = dataclasses.field(default_factory=lambda: {a: 0 for a in AGENTS})
    quota_failures: dict[str, int] = dataclasses.field(default_factory=lambda: {a: 0 for a in AGENTS})
    disabled_by_quota: dict[str, bool] = dataclasses.field(default_factory=lambda: {a: False for a in AGENTS})
    history: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    # dynamic write access policy (set by previous turn)
    grant_write_access: bool = False


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(config_path: Path) -> RunConfig:
    data = _load_json(config_path)
    limits = data["limits"]
    agents = data["agents"]

    agent_scripts: dict[str, str] = {}
    quota_pats: dict[str, list[str]] = {}
    supports_write: dict[str, bool] = {}

    for a in AGENTS:
        if a not in agents:
            raise ValueError(f"Missing agent in config: {a}")
        agent_scripts[a] = agents[a]["script"]
        quota_pats[a] = list(agents[a].get("quota_error_patterns", []))
        supports_write[a] = bool(agents[a].get("supports_write_access", False))

    return RunConfig(
        max_calls_per_agent=int(limits["max_calls_per_agent"]),
        quota_retry_attempts=int(limits["quota_retry_attempts"]),
        max_total_calls=int(limits["max_total_calls"]),
        max_json_correction_attempts=int(limits.get("max_json_correction_attempts", 2)),
        fallback_order=list(data["fallback_order"]),
        agent_scripts=agent_scripts,
        quota_error_patterns=quota_pats,
        supports_write_access=supports_write,
    )


def _run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def _git_try(cmd: list[str], cwd: Path) -> str:
    try:
        rc, out, err = _run_cmd(cmd, cwd=cwd, env=os.environ.copy())
        if rc != 0:
            return f"$ {' '.join(cmd)}\n<nonzero exit {rc}>\n{err.strip()}\n"
        return f"$ {' '.join(cmd)}\n{out.strip()}\n"
    except Exception as e:
        return f"$ {' '.join(cmd)}\n<error> {e}\n"


def _git_snapshot(cwd: Path) -> str:
    return "\n".join(
        [
            _git_try(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd),
            _git_try(["git", "rev-parse", "HEAD"], cwd),
            _git_try(["git", "status", "--porcelain=v1"], cwd),
            _git_try(["git", "diff", "--stat"], cwd),
        ]
    )


def _extract_stats_ids(stats_md: str) -> list[str]:
    found = set(re.findall(r"\b(?:CX|GM|CL)-\d+\b", stats_md))
    return sorted(found)


def _parse_smoke_route(raw: str) -> list[str]:
    if not raw:
        return []
    items = [item.strip() for item in raw.split(",") if item.strip()]
    for item in items:
        if item not in AGENTS:
            raise ValueError(f"Invalid smoke-route agent: {item}")
    return items


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n\n<TRUNCATED: {len(s) - limit} chars>\n"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_milestone_id(design_doc_text: str) -> str:
    m = re.search(r"\*\*Milestone:\*\*\s*(M\d+)\b", design_doc_text)
    return m.group(1) if m else "M?"


def _run_verify(project_root: Path, out_json: Path, strict_git: bool) -> tuple[int, str, str]:
    cmd = [sys.executable, "-m", "tools.verify", "--json", str(out_json)]
    if strict_git:
        cmd.append("--strict-git")
    return _run_cmd(cmd, cwd=project_root, env=os.environ.copy())


def _extract_section(lines: list[str], header_prefix: str) -> list[str]:
    """Extract a markdown section by prefix.

    Example header_prefix values:
      - "## Normative Requirements"
      - "## Definition of Done"
      - "## Test Matrix"

    The function:
      - finds the first line whose stripped lowercase startswith(header_prefix.lower())
      - returns lines until the next '## ' header (exclusive)
    """

    hp = header_prefix.strip().lower()
    start: int | None = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(hp):
            start = i
            break
    if start is None:
        return []

    out: list[str] = [lines[start]]
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            break
        out.append(lines[j])
    return out


def _design_doc_excerpt(design_doc_text: str, max_chars: int = 60000) -> str:
    """Return a prompt-safe excerpt that always includes critical spec sections."""

    text = design_doc_text.strip()
    if not text:
        return ""

    lines = text.splitlines()

    # Preamble: everything up to (but excluding) Normative Requirements, if present.
    preamble: list[str] = []
    nr_idx: int | None = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("## normative requirements"):
            nr_idx = i
            break
    if nr_idx is None:
        preamble = lines[:120]
    else:
        preamble = lines[:nr_idx]

    sections: list[list[str]] = []
    sections.append(_extract_section(lines, "## Normative Requirements"))
    sections.append(_extract_section(lines, "## Definition of Done"))
    sections.append(_extract_section(lines, "## Test Matrix"))

    # Optional but often useful.
    sections.append(_extract_section(lines, "## References"))

    # Compose with clear separators.
    parts: list[str] = []
    parts.extend(preamble)

    seen = set()
    for sec in sections:
        if not sec:
            continue
        key = "\n".join(sec[:1])
        if key in seen:
            continue
        seen.add(key)
        if parts and parts[-1].strip() != "":
            parts.append("")
        parts.extend(sec)

    excerpt = "\n".join(parts).strip() + "\n"
    if len(excerpt) > max_chars:
        excerpt = _truncate(excerpt, max_chars)

    # Add an explicit note so agents know the excerpt may not contain everything.
    if len(text) > len(excerpt):
        excerpt += "\n[NOTE] Full DESIGN_DOCUMENT.md is longer; excerpt includes the normative contract sections.\n"

    return excerpt


def build_prompt(
    *,
    agent: str,
    system_prompt: str,
    design_doc_text: str,
    milestone_id: str,
    repo_info: str,
    verify_report_text: str,
    history: list[dict[str, Any]],
    next_prompt: str,
    call_counts: dict[str, int],
    disabled_by_quota: dict[str, bool],
    stats_ids: list[str],
) -> str:
    last_summaries = "\n".join([f"- ({h['agent']}) {h['summary']}" for h in history[-4:]])

    state_blob = json.dumps(
        {
            "agent": agent,
            "milestone_id": milestone_id,
            "call_counts": call_counts,
            "disabled_by_quota": disabled_by_quota,
            "known_stats_ids": stats_ids,
        },
        indent=2,
        sort_keys=True,
    )

    parts = [
        system_prompt.strip(),
        "\n\n---\n\n# Orchestrator State (read-only)\n",
        state_blob,
        "\n\n---\n\n# Milestone Spec (DESIGN_DOCUMENT.md)\n",
        _design_doc_excerpt(design_doc_text, max_chars=60000),
        "\n\n---\n\n# Verification Report (tools.verify)\n",
        _truncate(verify_report_text.strip(), 20000),
        "\n\n---\n\n# Repo Snapshot\n",
        repo_info.strip(),
        "\n\n---\n\n",
    ]

    if last_summaries.strip():
        parts += ["# Recent Turn Summaries\n", last_summaries.strip(), "\n\n---\n\n"]

    parts += ["# Your Work Item\n", next_prompt.strip() or "(Decide the next best step.)", "\n"]
    return "".join(parts)


def _try_parse_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty output")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start == -1:
        raise ValueError("no '{' found in output")

    depth = 0
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                return json.loads(candidate)

    raise ValueError("unbalanced braces in output")


def _validate_turn(obj: Any, expected_agent: str, stats_id_set: set[str], milestone_id: str) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "turn is not an object"

    required = [
        "agent",
        "milestone_id",
        "phase",
        "work_completed",
        "project_complete",
        "summary",
        "gates_passed",
        "requirement_progress",
        "next_agent",
        "next_prompt",
        "delegate_rationale",
        "stats_refs",
        "needs_write_access",
        "artifacts",
    ]

    for k in required:
        if k not in obj:
            return False, f"missing key: {k}"

    if obj.get("agent") != expected_agent:
        return False, f"agent mismatch: expected {expected_agent}, got {obj.get('agent')}"

    if str(obj.get("milestone_id")) != milestone_id:
        return False, f"milestone_id mismatch: expected {milestone_id}, got {obj.get('milestone_id')}"

    if obj["agent"] not in AGENTS or obj["next_agent"] not in AGENTS:
        return False, "invalid agent id in agent/next_agent"

    if obj.get("phase") not in ("plan", "implement", "verify", "finalize"):
        return False, "invalid phase"

    if not isinstance(obj["work_completed"], bool) or not isinstance(obj["project_complete"], bool):
        return False, "work_completed/project_complete must be boolean"

    if not isinstance(obj["summary"], str) or not isinstance(obj["next_prompt"], str) or not isinstance(
        obj["delegate_rationale"], str
    ):
        return False, "summary/next_prompt/delegate_rationale must be strings"

    if not isinstance(obj["needs_write_access"], bool):
        return False, "needs_write_access must be boolean"

    if not isinstance(obj["gates_passed"], list) or not all(isinstance(x, str) for x in obj["gates_passed"]):
        return False, "gates_passed must be array of strings"

    if not isinstance(obj["stats_refs"], list) or not all(isinstance(x, str) for x in obj["stats_refs"]):
        return False, "stats_refs must be array of strings"

    if not obj["stats_refs"]:
        return False, "stats_refs is empty"
    unknown = [x for x in obj["stats_refs"] if x not in stats_id_set]
    if unknown:
        return False, f"unknown stats_refs: {unknown}"

    rp = obj.get("requirement_progress")
    if not isinstance(rp, dict):
        return False, "requirement_progress must be object"
    for k in ("covered_req_ids", "tests_added_or_modified", "commands_run"):
        if k not in rp:
            return False, f"requirement_progress missing key: {k}"
        if not isinstance(rp[k], list) or not all(isinstance(x, str) for x in rp[k]):
            return False, f"requirement_progress.{k} must be array of strings"

    if not isinstance(obj["artifacts"], list):
        return False, "artifacts must be an array"
    for i, a in enumerate(obj["artifacts"]):
        if not isinstance(a, dict):
            return False, f"artifact[{i}] must be object"
        if set(a.keys()) != {"path", "description"}:
            return False, f"artifact[{i}] must have exactly keys: path, description"
        if not isinstance(a["path"], str) or not isinstance(a["description"], str):
            return False, f"artifact[{i}] path/description must be strings"

    allowed = set(required)
    extra = set(obj.keys()) - allowed
    if extra:
        return False, f"unexpected keys present: {sorted(extra)}"

    return True, "ok"


def _is_quota_error(agent: str, text: str, config: RunConfig) -> bool:
    pats = config.quota_error_patterns.get(agent, [])
    return any(re.search(p, text, flags=re.IGNORECASE) for p in pats)


def _pick_fallback(config: RunConfig, state: RunState) -> str | None:
    for a in config.fallback_order:
        if a not in AGENTS:
            continue
        if state.disabled_by_quota.get(a, False):
            continue
        if state.call_counts.get(a, 0) >= config.max_calls_per_agent:
            continue
        return a
    return None


def _override_next_agent(requested: str, config: RunConfig, state: RunState) -> tuple[str, str | None]:
    if requested not in AGENTS:
        fb = _pick_fallback(config, state)
        return fb or requested, f"requested invalid agent '{requested}'"

    if state.disabled_by_quota.get(requested, False):
        fb = _pick_fallback(config, state)
        return fb or requested, f"requested agent '{requested}' disabled by quota"

    if state.call_counts.get(requested, 0) >= config.max_calls_per_agent:
        fb = _pick_fallback(config, state)
        return fb or requested, f"requested agent '{requested}' exceeded call cap"

    return requested, None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _git_init_if_needed(project_root: Path) -> None:
    if not (project_root / ".git").exists():
        _run_cmd(["git", "init"], cwd=project_root, env=os.environ.copy())
        _run_cmd(["git", "config", "user.email", "agent@example.com"], cwd=project_root, env=os.environ.copy())
        _run_cmd(["git", "config", "user.name", "Agent Runner"], cwd=project_root, env=os.environ.copy())
        _run_cmd(["git", "add", "-A"], cwd=project_root, env=os.environ.copy())
        _run_cmd(["git", "commit", "-m", "chore: initial state"], cwd=project_root, env=os.environ.copy())


def _checkout_agent_branch(project_root: Path, run_id: str) -> None:
    branch = f"agent-run/{run_id}"
    rc, _, _ = _run_cmd(["git", "checkout", "-b", branch], cwd=project_root, env=os.environ.copy())
    if rc != 0:
        # If already exists, checkout.
        _run_cmd(["git", "checkout", branch], cwd=project_root, env=os.environ.copy())


def _completion_gates_ok(project_root: Path) -> tuple[bool, str]:
    from tools.completion_gates import CompletionGateInputs, evaluate_completion_gates

    rc, out, err = _run_cmd(
        [sys.executable, "-m", "tools.verify", "--strict-git"],
        cwd=project_root,
        env=os.environ.copy(),
    )
    verify_rc = rc

    _, porcelain, _ = _run_cmd(
        ["git", "status", "--porcelain=v1"],
        cwd=project_root,
        env=os.environ.copy(),
    )
    _, head, _ = _run_cmd(["git", "rev-parse", "HEAD"], cwd=project_root, env=os.environ.copy())

    verdict = evaluate_completion_gates(
        CompletionGateInputs(verify_rc=verify_rc, git_porcelain=porcelain, head_sha=head)
    )
    if not verdict.ok:
        detail = ""
        if verify_rc != 0:
            detail = f"\n{out}\n{err}".strip()
        return False, f"{verdict.reason}{detail}"
    return True, "ok"


def _run_agent_live(
    *,
    agent: str,
    prompt_path: Path,
    schema_path: Path,
    out_path: Path,
    config: RunConfig,
    state: RunState,
) -> tuple[int, str, str]:
    script_rel = config.agent_scripts[agent]
    script = (state.project_root / script_rel).resolve()

    env = os.environ.copy()
    force_write = agent in ("gemini", "claude", "codex")
    wants_write = state.grant_write_access or force_write
    env["WRITE_ACCESS"] = "1" if (wants_write and config.supports_write_access.get(agent, False)) else "0"

    cmd = [str(script), str(prompt_path), str(schema_path), str(out_path)]
    return _run_cmd(cmd, cwd=state.project_root, env=env)


def _run_agent_mock(
    *,
    agent: str,
    scenario: dict[str, Any],
    mock_indices: dict[str, int],
) -> tuple[int, str, str]:
    block = scenario["agents"].get(agent, [])
    idx = mock_indices.get(agent, 0)
    if idx >= len(block):
        return 1, "", f"mock scenario: no more responses for agent '{agent}' (idx={idx})"
    mock_indices[agent] = idx + 1

    item = block[idx]
    t = item.get("type")
    if t == "quota_error":
        return int(item.get("exit_code", 1)), item.get("stdout", ""), item.get("stderr", "TerminalQuotaError")
    if t == "error":
        return int(item.get("exit_code", 1)), item.get("stdout", ""), item.get("stderr", "mock error")

    if t != "ok":
        return 1, "", f"mock scenario: unknown item type '{t}'"

    resp = item.get("response")
    return 0, json.dumps(resp, indent=2, sort_keys=True), ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--config", default="bridge/config.json")
    ap.add_argument("--schema", default="bridge/turn.schema.json")
    ap.add_argument("--system-prompt", default="bridge/prompts/system.md")
    ap.add_argument("--design-doc", default="DESIGN_DOCUMENT.md")
    ap.add_argument("--start-agent", choices=AGENTS, default="gemini")
    ap.add_argument("--mode", choices=["live", "mock"], default="mock")
    ap.add_argument("--mock-scenario", default="bridge/mock_scenarios/milestone_demo.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-agent-branch", action="store_true")
    ap.add_argument(
        "--smoke-route",
        default="",
        help="Force a fixed agent sequence for smoke tests (comma-separated).",
    )

    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    config = load_config(project_root / args.config)

    schema_path = project_root / args.schema
    system_prompt_path = project_root / args.system_prompt
    design_doc_path = (project_root / args.design_doc).resolve()

    run_id = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    runs_dir = project_root / "runs" / run_id
    _ensure_dir(runs_dir)

    _git_init_if_needed(project_root)
    if not args.no_agent_branch:
        _checkout_agent_branch(project_root, run_id)

    state = RunState(
        run_id=run_id,
        project_root=project_root,
        runs_dir=runs_dir,
        schema_path=schema_path,
        system_prompt_path=system_prompt_path,
        design_doc_path=design_doc_path,
    )

    stats_md = (project_root / "STATS.md").read_text(encoding="utf-8")
    stats_ids = _extract_stats_ids(stats_md)
    stats_id_set = set(stats_ids)

    system_prompt = system_prompt_path.read_text(encoding="utf-8")

    # Mock scenario support.
    scenario: dict[str, Any] = {}
    mock_indices: dict[str, int] = {a: 0 for a in AGENTS}
    if args.mode == "mock":
        scenario = _load_json(project_root / args.mock_scenario)

    next_prompt = textwrap.dedent(
        """
        You are at the start of the milestone run.

        1) Read STATS.md.
        2) Read DESIGN_DOCUMENT.md and identify the milestone ID + all normative requirements.
        3) If the design document is not in the required format, rewrite it to comply.
        4) Propose the next smallest implementation step that can be proven with pytest.

        IMPORTANT: always keep the Test Matrix up to date and runnable.
        """
    ).strip()

    try:
        smoke_route = _parse_smoke_route(args.smoke_route)
    except ValueError as exc:
        print(f"[orchestrator] ERROR: {exc}")
        return 2

    smoke_index = 0
    agent = smoke_route[0] if smoke_route else args.start_agent

    print(f"[orchestrator] run_id={state.run_id} mode={args.mode} project_root={project_root}")
    print(
        f"[orchestrator] limits: max_calls_per_agent={config.max_calls_per_agent} "
        f"quota_retry_attempts={config.quota_retry_attempts} max_total_calls={config.max_total_calls}"
    )

    call_no = 0
    while state.total_calls < config.max_total_calls:
        if state.call_counts.get(agent, 0) >= config.max_calls_per_agent:
            fb = _pick_fallback(config, state)
            if not fb:
                print("[orchestrator] ERROR: all agents exhausted call cap; stopping.")
                return 2
            print(f"[orchestrator] OVERRIDE: agent '{agent}' exceeded call cap; switching to '{fb}'")
            agent = fb

        call_no += 1
        state.total_calls += 1
        state.call_counts[agent] += 1

        call_dir = runs_dir / f"call_{call_no:04d}"
        _ensure_dir(call_dir)

        design_doc_text = _read_text(design_doc_path) if design_doc_path.exists() else ""
        milestone_id = _parse_milestone_id(design_doc_text)

        # Run verify (non-strict by default) and embed the report into the prompt.
        verify_json = call_dir / "verify.json"
        if os.environ.get("FF_SKIP_VERIFY") == "1":
            verify_report_text = "{}"
        else:
            _run_verify(project_root, verify_json, strict_git=False)
            verify_report_text = verify_json.read_text(encoding="utf-8") if verify_json.exists() else "{}"

        repo_info = _git_snapshot(project_root)

        prompt_text = build_prompt(
            agent=agent,
            system_prompt=system_prompt,
            design_doc_text=design_doc_text,
            milestone_id=milestone_id,
            repo_info=repo_info,
            verify_report_text=verify_report_text,
            history=state.history,
            next_prompt=next_prompt,
            call_counts=state.call_counts,
            disabled_by_quota=state.disabled_by_quota,
            stats_ids=stats_ids,
        )

        prompt_path = call_dir / "prompt.txt"
        raw_path = call_dir / "raw.txt"
        out_path = call_dir / "out.json"
        _write_text(prompt_path, prompt_text)

        print("=" * 88)
        wants_write = state.grant_write_access or agent in AGENTS
        write_access = "1" if (wants_write and config.supports_write_access.get(agent, False)) else "0"
        print(
            f"CALL {call_no:04d} | agent={agent} | total_calls={state.total_calls} | "
            f"agent_calls={state.call_counts[agent]}/{config.max_calls_per_agent} | "
            f"write_access={write_access}"
        )

        if args.mode == "mock":
            rc, out, err = _run_agent_mock(agent=agent, scenario=scenario, mock_indices=mock_indices)
        else:
            rc, out, err = _run_agent_live(
                agent=agent,
                prompt_path=prompt_path,
                schema_path=schema_path,
                out_path=out_path,
                config=config,
                state=state,
            )

        _write_text(raw_path, (out or "") + ("\n" if out and err else "") + (err or ""))

        if args.verbose and err.strip():
            print("[orchestrator] stderr:\n" + err.strip())

        if rc != 0:
            combined = (out + "\n" + err).strip()
            quota = _is_quota_error(agent, combined, config)
            label = "QUOTA" if quota else "ERROR"
            print(f"[orchestrator] {label}: agent '{agent}' exited rc={rc}")

            if quota:
                state.quota_failures[agent] += 1
                remaining = config.quota_retry_attempts - state.quota_failures[agent]
                if remaining > 0 and state.call_counts[agent] < config.max_calls_per_agent:
                    print(
                        f"[orchestrator] quota retry {state.quota_failures[agent]}/"
                        f"{config.quota_retry_attempts} for agent '{agent}' (remaining={remaining})"
                    )
                    continue

                print(f"[orchestrator] DISABLE: agent '{agent}' marked disabled_by_quota=true")
                state.disabled_by_quota[agent] = True
                fb = _pick_fallback(config, state)
                if not fb:
                    print("[orchestrator] ERROR: no fallback agents available; stopping.")
                    return 3
                print(f"[orchestrator] FAILOVER: switching to '{fb}' with same work item")
                agent = fb
                continue

            fb = _pick_fallback(config, state)
            if not fb:
                print("[orchestrator] ERROR: no fallback agents available; stopping.")
                return 4
            print(f"[orchestrator] FAILOVER: switching to '{fb}' after error")
            agent = fb
            continue

        # Parse and validate JSON.
        parse_ok = False
        correction_attempts = 0
        last_err = ""
        turn_obj: Any = None

        while not parse_ok and correction_attempts <= config.max_json_correction_attempts:
            try:
                turn_obj = _try_parse_json(out)
                ok, msg = _validate_turn(
                    turn_obj, expected_agent=agent, stats_id_set=stats_id_set, milestone_id=milestone_id
                )
                if not ok:
                    raise ValueError(msg)
                parse_ok = True
            except Exception as e:
                last_err = str(e)
                correction_attempts += 1
                if correction_attempts > config.max_json_correction_attempts:
                    break

                print(f"[orchestrator] JSON INVALID: {last_err}")
                print(
                    f"[orchestrator] correction attempt {correction_attempts}/{config.max_json_correction_attempts} (same agent)"
                )

                next_prompt = textwrap.dedent(
                    f"""
                    Your previous output was invalid JSON for the required schema.

                    Error: {last_err}

                    Re-emit the JSON object ONLY, matching the schema exactly.
                    """
                ).strip()
                break

        if not parse_ok:
            print(f"[orchestrator] ERROR: could not parse/validate JSON: {last_err}")
            fb = _pick_fallback(config, state)
            if not fb:
                print("[orchestrator] ERROR: no fallback agents available; stopping.")
                return 5
            print(f"[orchestrator] FAILOVER: switching to '{fb}'")
            agent = fb
            continue

        turn_path = call_dir / "turn.json"
        _write_text(turn_path, json.dumps(turn_obj, indent=2, sort_keys=True))
        state.history.append(turn_obj)

        print(f"[orchestrator] summary: {turn_obj['summary']}")

        # Update write-access grant for next call.
        state.grant_write_access = bool(turn_obj.get("needs_write_access", False))

        # Completion gates: only stop if they actually pass.
        if bool(turn_obj["project_complete"]):
            if smoke_route and smoke_index + 1 < len(smoke_route):
                print("[orchestrator] PROJECT_COMPLETE ignored due to smoke_route")
                requested = str(turn_obj["next_agent"])
                effective, override_reason = _override_next_agent(requested, config, state)
                smoke_index += 1
                forced = smoke_route[smoke_index]
                print(f"[orchestrator] SMOKE_ROUTE next_agent={forced} (override {effective})")
                next_prompt = str(turn_obj["next_prompt"]).strip()
                agent = forced
                continue

            ok, msg = _completion_gates_ok(project_root)
            if ok:
                print("[orchestrator] PROJECT COMPLETE (gates passed)")
                state_path = runs_dir / "state.json"
                _write_text(
                    state_path,
                    json.dumps(dataclasses.asdict(state), indent=2, sort_keys=True, default=str),
                )
                return 0

            print("[orchestrator] PROJECT_COMPLETE rejected: completion gates failed")
            print(_truncate(msg, 2000))
            next_prompt = textwrap.dedent(
                f"""
                You attempted to mark the project complete, but completion gates failed.

                Gate failure details:
                {msg}

                Fix the repo so that:
                - python -m tools.verify --strict-git passes
                - git status --porcelain is empty
                - changes are committed

                Then try completion again.
                """
            ).strip()

            # Heuristic: if it looks like only commit/cleanup, ask Claude; otherwise Codex.
            agent = "claude" if "git status not clean" in msg or "no commit" in msg else "codex"
            continue

        # Decide next agent.
        requested = str(turn_obj["next_agent"])
        effective, override_reason = _override_next_agent(requested, config, state)
        if smoke_route and smoke_index + 1 < len(smoke_route):
            smoke_index += 1
            forced = smoke_route[smoke_index]
            print(f"[orchestrator] SMOKE_ROUTE next_agent={forced} (override {effective})")
            effective = forced
            override_reason = "smoke_route"

        if override_reason == "smoke_route":
            print(f"[orchestrator] next_agent={effective} (smoke_route)")
        elif override_reason:
            print(f"[orchestrator] OVERRIDE next_agent: {requested} -> {effective} ({override_reason})")
        else:
            print(f"[orchestrator] next_agent={effective} (as requested)")

        next_prompt = str(turn_obj["next_prompt"]).strip()
        agent = effective

    print("[orchestrator] ERROR: max_total_calls exceeded")
    return 6


if __name__ == "__main__":
    raise SystemExit(main())
