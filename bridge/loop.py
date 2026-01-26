#!/usr/bin/env python3
"""Two-agent orchestration loop (Codex + Claude).

This loop:
- Embeds repo context + verify report + recent summaries into a single prompt.
- Calls one of two agent wrappers (bridge/agents/*.sh) and streams their output.
- Validates the agent's response against the expected turn schema (bridge/turn.schema.json)
  and additional local constraints (agent id, milestone id, stats refs).

Notes:
- This runner is intentionally opinionated: unless project_complete=true, it enforces a
  Codex <-> Claude alternation when both are enabled.
- Verbose streaming output is preserved in live mode: agent stdout/stderr are forwarded
  to your terminal while also being captured to runs/<run_id>/call_XXXX/raw.txt.
"""

from __future__ import annotations

import argparse
import ast
import collections
import concurrent.futures
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
import traceback
from pathlib import Path
from typing import Any

# When run as `python bridge/loop.py`, Python sets sys.path[0] to `bridge/`.
# We want to import sibling packages (e.g. `tools`) from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Atomic I/O for robust file operations
from bridge.atomic_io import atomic_write_json, validate_json_file

# Turn normalization - extracted to submodule for tooling readability
from bridge.loop_pkg.turn_normalizer import (
    NormalizationResult,
    TurnNormalizer,
    normalize_agent_output,
)
from bridge.loop_pkg.turn_normalizer import (
    validate_turn_lenient as _validate_turn_lenient,
)

# Patch integration for commit-free worker operation
from bridge.patch_integration import PatchIntegrator, collect_patch_artifact, save_patch_artifact
from bridge.scheduler import BackfillGenerator, FillerTask, LaneConfig, TwoLaneScheduler
from bridge.smoke_route import resolve_smoke_route
from bridge.streaming import run_cmd_with_streaming
from bridge.verify_repair import (
    create_repair_callback,
    run_verify_repair_loop,
    write_repair_report,
)

# Design document parsing - modular adapter layer
from bridge.design_doc import (
    ContractMode,
    DesignDocSpec,
    parse_design_doc,
    parse_design_doc_text,
)

AGENTS: tuple[str, ...] = ("codex", "claude")


# -----------------------------
# Config + state
# -----------------------------


@dataclasses.dataclass(frozen=True)
class ParallelSettings:
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
class AgentCapabilities:
    """Capabilities for an agent - defines what tools/features are available."""

    supports_tools: bool = True
    supports_fs_read: bool = True
    supports_fs_write: bool = True
    supports_bash: bool = True
    supports_write_access: bool = True


@dataclasses.dataclass(frozen=True)
class RunConfig:
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
    agent_capabilities: dict[str, AgentCapabilities]
    parallel: ParallelSettings


@dataclasses.dataclass
class RunState:
    run_id: str
    project_root: Path
    runs_dir: Path
    schema_path: Path
    system_prompt_path: Path
    design_doc_path: Path
    smoke_route: tuple[str, ...] = tuple()
    readonly: bool = False

    total_calls: int = 0
    call_counts: dict[str, int] = dataclasses.field(default_factory=lambda: {a: 0 for a in AGENTS})
    quota_failures: dict[str, int] = dataclasses.field(default_factory=lambda: {a: 0 for a in AGENTS})
    disabled_by_quota: dict[str, bool] = dataclasses.field(default_factory=lambda: {a: False for a in AGENTS})
    history: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    # Dynamic write access policy (set by previous turn)
    grant_write_access: bool = False


# -----------------------------
# AgentPolicy: Centralized agent selection enforcement
# -----------------------------


class AgentPolicyViolation(Exception):
    """Raised when code attempts to use an agent that violates the policy."""

    pass


@dataclasses.dataclass
class AgentPolicy:
    """Centralized agent selection policy.

    When forced_agent is set (via --only-codex or --only-claude), ALL agent
    selections must go through this policy and will be overridden to use
    only the forced agent.
    """

    forced_agent: str | None = None  # Set by --only-* flags
    allowed_agents: tuple[str, ...] = AGENTS
    runs_dir: Path | None = None  # For writing violation artifacts

    def enforce(self, requested_agent: str, context: str = "") -> str:
        """Enforce the agent policy, returning the agent to use.

        Args:
            requested_agent: The agent that was requested
            context: Description of where this request originated (for error messages)

        Returns:
            The agent to actually use (forced_agent if set, otherwise requested)

        Raises:
            AgentPolicyViolation: If forced mode is active and code tries to use wrong agent
        """
        if self.forced_agent:
            if requested_agent != self.forced_agent and requested_agent in AGENTS:
                # Log the override
                print(f"[AgentPolicy] OVERRIDE: {requested_agent} -> {self.forced_agent} ({context})")
            return self.forced_agent

        # No forced agent - verify requested is allowed
        if requested_agent not in self.allowed_agents:
            if self.allowed_agents:
                return self.allowed_agents[0]
            return AGENTS[0]

        return requested_agent

    def enforce_strict(self, requested_agent: str, context: str = "") -> str:
        """Strict enforcement - raises exception if wrong agent is requested.

        Use this for code paths that should NEVER attempt to use the wrong agent
        (e.g., fallback logic that might try to switch agents).
        """
        if self.forced_agent and requested_agent != self.forced_agent:
            msg = (
                f"AGENT POLICY VIOLATION: Attempted to use '{requested_agent}' "
                f"when --only-{self.forced_agent} is active. Context: {context}"
            )
            self._write_violation_artifact(msg, requested_agent, context)
            raise AgentPolicyViolation(msg)
        return self.enforce(requested_agent, context)

    def _write_violation_artifact(self, msg: str, requested: str, context: str) -> None:
        """Write an artifact explaining the policy violation."""
        if not self.runs_dir:
            return
        artifact_path = self.runs_dir / "agent_policy_violation.txt"
        content = f"""AGENT POLICY VIOLATION
======================

Timestamp: {dt.datetime.utcnow().isoformat()}Z
Forced Agent: {self.forced_agent}
Requested Agent: {requested}
Context: {context}

Message:
{msg}

This file was created because code attempted to invoke an agent that
violates the --only-{self.forced_agent} flag. This indicates a bug in
the orchestrator's agent selection logic.
"""
        try:
            artifact_path.write_text(content, encoding="utf-8")
            print(f"[AgentPolicy] Violation artifact written to: {artifact_path}")
        except Exception as e:
            print(f"[AgentPolicy] Failed to write violation artifact: {e}")

    def is_forced_mode(self) -> bool:
        """Return True if a forced agent mode is active."""
        return self.forced_agent is not None

    def get_prompt_header(self) -> str:
        """Get a header to inject into prompts when in forced mode.

        This tells the agent it's the only one and must implement, not just review.
        """
        if not self.forced_agent:
            return ""

        return f"""## AGENT POLICY OVERRIDE

**IMPORTANT**: You are running in `--only-{self.forced_agent}` mode.

- You are the ONLY agent allowed in this session.
- You MUST implement all changes yourself. Do NOT suggest handing off to another agent.
- You MUST verify your own changes. Do NOT assume another agent will review.
- Set `next_agent` to `"{self.forced_agent}"` in your response (it will be enforced anyway).
- Focus on both implementation AND verification - you are responsible for the full cycle.

"""


# Global policy instance (set during main() based on CLI flags)
_agent_policy: AgentPolicy | None = None


def get_agent_policy() -> AgentPolicy:
    """Get the global agent policy. Returns a default policy if not set."""
    global _agent_policy
    if _agent_policy is None:
        _agent_policy = AgentPolicy()
    return _agent_policy


def set_agent_policy(policy: AgentPolicy) -> None:
    """Set the global agent policy."""
    global _agent_policy
    _agent_policy = policy


# -----------------------------
# Small helpers
# -----------------------------


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    # Keep both ends: helpful for logs that end with stack traces.
    head = max(0, int(limit * 0.7))
    tail = max(0, limit - head - 64)
    return text[:head] + "\n\n[...TRUNCATED... try opening the raw log file for full output ...]\n\n" + text[-tail:]


def _extract_stats_ids(stats_md_text: str) -> list[str]:
    """Extract stable stats identifiers from STATS.md.

    IDs are intentionally simple: CX-* and CL-* only (two-agent mode).
    """

    ids = sorted(set(re.findall(r"\b(?:CX|CL)-\d+\b", stats_md_text)))
    return ids


def _parse_milestone_id(design_doc_text: str) -> str:
    m = re.search(r"\*\*Milestone:\*\*\s*(M\d+)\b", design_doc_text)
    return m.group(1) if m else "M0"


def _parse_all_milestones(design_doc_text: str) -> list[str]:
    """Parse all milestone IDs from a design document that may contain multiple milestone docs.

    Looks for patterns like:
    - **Milestone:** M1
    - # M2 Design Document
    - # M3 Design Document
    """
    milestones = set()

    # Pattern 1: **Milestone:** M1 format
    for m in re.finditer(r"\*\*Milestone:\*\*\s*(M\d+)\b", design_doc_text):
        milestones.add(m.group(1))

    # Pattern 2: # M2 Design Document format
    for m in re.finditer(r"^#\s+(M\d+)\s+Design\s+Document", design_doc_text, re.MULTILINE):
        milestones.add(m.group(1))

    if not milestones:
        return ["M0"]

    # Sort by milestone number
    return sorted(milestones, key=lambda x: int(x[1:]))


def _parse_smoke_route_arg(value: str) -> list[str]:
    route = [tok.strip() for tok in value.split(",") if tok.strip()]
    if not route:
        raise argparse.ArgumentTypeError("smoke-route must include at least one agent name")
    unknown = [tok for tok in route if tok not in AGENTS]
    if unknown:
        allowed = ", ".join(AGENTS)
        raise argparse.ArgumentTypeError(
            f"smoke-route contains unknown agent(s): {', '.join(unknown)} (allowed: {allowed})"
        )
    return route


def _extract_milestone_from_task_id(task_id: str, fallback: str = "M0") -> str:
    """Extract milestone prefix from task ID (e.g., 'M2-SIM-SCHEMA' -> 'M2')."""
    m = re.match(r"^(M\d+)-", task_id)
    return m.group(1) if m else fallback


def _run_cmd(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    *,
    stream: bool = False,
) -> tuple[int, str, str]:
    """Run a subprocess.

    When stream=True, stdout/stderr are forwarded live while also being captured.
    """

    if not stream:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True)
        return proc.returncode, proc.stdout or "", proc.stderr or ""

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )

    out_chunks: list[str] = []
    err_chunks: list[str] = []

    def _pump(src, sink, chunks: list[str]) -> None:
        try:
            assert src is not None
            for line in iter(src.readline, ""):
                sink.write(line)
                sink.flush()
                chunks.append(line)
        finally:
            with contextlib.suppress(Exception):
                src.close()

    t_out = threading.Thread(target=_pump, args=(proc.stdout, sys.stdout, out_chunks), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, sys.stderr, err_chunks), daemon=True)
    t_out.start()
    t_err.start()

    rc = proc.wait()
    t_out.join(timeout=1)
    t_err.join(timeout=1)
    return rc, "".join(out_chunks), "".join(err_chunks)


# -----------------------------
# Resource monitoring (parallel runner)
# -----------------------------


def _total_ram_bytes() -> int | None:
    """Best-effort total physical RAM bytes (no external deps)."""
    # POSIX sysconf (Linux + many Unixes)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages and page_size:
            return int(pages) * int(page_size)
    except Exception:
        pass

    # Linux /proc fallback
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    # kB
                    kb = int(parts[1])
                    return kb * 1024
    except Exception:
        pass

    return None


def _parse_ps_time_to_seconds(s: str) -> float:
    """Parse ps TIME / CPUTIME strings into seconds.

    Accepts: [[dd-]hh:]mm:ss, mm:ss, or ss.
    """
    s = s.strip()
    if not s:
        return 0.0

    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        try:
            days = int(d)
        except ValueError:
            days = 0

    parts = s.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return 0.0

    h = 0
    m = 0
    sec = 0
    if len(nums) == 3:
        h, m, sec = nums
    elif len(nums) == 2:
        m, sec = nums
    elif len(nums) == 1:
        sec = nums[0]
    else:
        # Unexpected format
        return 0.0

    return float(days * 86400 + h * 3600 + m * 60 + sec)


def _ps_list_pids_in_pgid(pgid: int) -> list[int]:
    """Return pids in a process group via ps (portable-ish)."""
    candidates = [
        ["ps", "-o", "pid=", "-g", str(pgid)],
        ["ps", "-o", "pid=", "--pgid", str(pgid)],
    ]
    for cmd in candidates:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            pids: list[int] = []
            for tok in out.split():
                tok = tok.strip()
                if tok.isdigit():
                    pids.append(int(tok))
            if pids:
                return sorted(set(pids))
        except Exception:
            continue
    return []


def _ps_sample_pids(pids: list[int]) -> tuple[float, int]:
    """Return (total_cpu_time_seconds, total_rss_bytes) for pids via ps."""
    if not pids:
        return 0.0, 0

    pid_arg = ",".join(str(p) for p in pids)
    candidates = [
        ["ps", "-o", "pid=", "-o", "time=", "-o", "rss=", "-p", pid_arg],
    ]
    for cmd in candidates:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            total_cpu = 0.0
            total_rss_kb = 0
            for line in out.splitlines():
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                # pid = parts[0]
                cpu_s = _parse_ps_time_to_seconds(parts[1])
                try:
                    rss_kb = int(parts[2])
                except ValueError:
                    rss_kb = 0
                total_cpu += cpu_s
                total_rss_kb += rss_kb
            return total_cpu, total_rss_kb * 1024
        except Exception:
            continue

    return 0.0, 0


@dataclasses.dataclass
class MonitoredResult:
    returncode: int
    killed_for_resources: bool
    kill_reason: str | None
    max_cpu_pct_total: float
    max_mem_pct_total: float
    tail_stdout: str
    tail_stderr: str


def _terminate_process_group(proc: subprocess.Popen, *, grace_s: float) -> None:
    """Try hard to stop proc + its process group without leaving orphans."""
    if proc.poll() is not None:
        return

    # First: SIGINT
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except Exception:
        with contextlib.suppress(Exception):
            proc.send_signal(signal.SIGINT)

    t0 = time.monotonic()
    while time.monotonic() - t0 < max(0.5, grace_s * 0.5):
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    # Second: SIGTERM
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        with contextlib.suppress(Exception):
            proc.terminate()

    t1 = time.monotonic()
    while time.monotonic() - t1 < max(0.5, grace_s * 0.4):
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    # Last resort: SIGKILL
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        with contextlib.suppress(Exception):
            proc.kill()


def _run_cmd_monitored(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    *,
    prefix: str,
    raw_log_path: Path,
    stream_to_terminal: bool,
    terminal_max_bytes: int,
    terminal_max_line_length: int,
    cpu_threshold_pct_total: float,
    mem_threshold_pct_total: float,
    sample_interval_s: float,
    consecutive_samples: int,
    kill_grace_s: float,
    allow_resource_intensive: bool,
) -> MonitoredResult:
    """Run a subprocess with live streaming + best-effort resource monitoring.

    - Always writes full combined stdout+stderr to raw_log_path.
    - Optionally forwards output to the terminal with per-process truncation.
    - If allow_resource_intensive is False, terminates the process group when its
      CPU or memory exceeds the configured thresholds for consecutive_samples.

    Returns a MonitoredResult containing a small tail buffer for debugging.
    """

    raw_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Tail buffers (keep last ~64KB total)
    tail_limit = 64 * 1024
    tail_out = collections.deque()  # type: ignore[var-annotated]
    tail_err = collections.deque()  # type: ignore[var-annotated]
    tail_out_size = 0
    tail_err_size = 0

    def _append_tail(dq: collections.deque, current_size: int, chunk: str) -> int:
        nonlocal tail_limit
        dq.append(chunk)
        current_size += len(chunk)
        while current_size > tail_limit and dq:
            removed = dq.popleft()
            current_size -= len(removed)
        return current_size

    printed_bytes = 0
    printed_truncated = False

    total_ram = _total_ram_bytes()
    num_cores = os.cpu_count() or 1

    killed_for_resources = False
    kill_reason: str | None = None
    max_cpu_pct = 0.0
    max_mem_pct = 0.0

    with raw_log_path.open("w", encoding="utf-8", errors="replace") as raw_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            start_new_session=True,
        )

        # Resource monitor state
        prev_cpu_time: float | None = None
        prev_t = time.monotonic()
        consecutive_hits = 0

        def _emit(line: str, *, is_err: bool) -> None:
            nonlocal printed_bytes, printed_truncated, tail_out_size, tail_err_size
            # Raw log: always write full line
            raw_f.write(line)
            raw_f.flush()

            # Tail buffers
            if is_err:
                tail_err_size = _append_tail(tail_err, tail_err_size, line)
            else:
                tail_out_size = _append_tail(tail_out, tail_out_size, line)

            # Terminal streaming (possibly truncated)
            if not stream_to_terminal:
                return
            if terminal_max_bytes > 0 and printed_bytes >= terminal_max_bytes:
                if not printed_truncated:
                    msg = f"{prefix} [terminal output truncated; see {raw_log_path}]\n"
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    printed_truncated = True
                return

            # Line truncation
            out_line = line
            if terminal_max_line_length > 0 and len(out_line) > terminal_max_line_length:
                out_line = out_line[:terminal_max_line_length] + "...\n"

            # Prefix each physical line
            # Preserve newlines by splitting; cheaper than regex.
            for seg in out_line.splitlines(True):
                sys.stdout.write(f"{prefix} {seg}")
            sys.stdout.flush()
            printed_bytes += len(out_line)

        def _pump(src, *, is_err: bool) -> None:
            try:
                assert src is not None
                for line in iter(src.readline, ""):
                    _emit(line, is_err=is_err)
            finally:
                with contextlib.suppress(Exception):
                    src.close()

        t_out = threading.Thread(target=_pump, args=(proc.stdout,), kwargs={"is_err": False}, daemon=True)
        t_err = threading.Thread(target=_pump, args=(proc.stderr,), kwargs={"is_err": True}, daemon=True)
        t_out.start()
        t_err.start()

        # Monitoring loop
        while proc.poll() is None:
            time.sleep(max(0.25, sample_interval_s))

            if allow_resource_intensive:
                continue

            try:
                pgid = proc.pid
                pids = _ps_list_pids_in_pgid(pgid) or [pgid]
                cpu_time_s, rss_bytes = _ps_sample_pids(pids)
                now = time.monotonic()
                dt_s = max(1e-3, now - prev_t)

                if prev_cpu_time is None:
                    prev_cpu_time = cpu_time_s
                    prev_t = now
                    continue

                delta_cpu = max(0.0, cpu_time_s - prev_cpu_time)
                prev_cpu_time = cpu_time_s
                prev_t = now

                # CPU as pct of total machine
                cpu_pct_total = (delta_cpu / dt_s) * 100.0 / float(num_cores)
                max_cpu_pct = max(max_cpu_pct, cpu_pct_total)

                mem_pct_total = 0.0
                if total_ram and total_ram > 0:
                    mem_pct_total = (rss_bytes / float(total_ram)) * 100.0
                    max_mem_pct = max(max_mem_pct, mem_pct_total)

                hit = False
                if cpu_threshold_pct_total > 0 and cpu_pct_total > cpu_threshold_pct_total:
                    hit = True
                    kill_reason = f"cpu>{cpu_threshold_pct_total:.1f}% (saw {cpu_pct_total:.1f}%)"
                if mem_threshold_pct_total > 0 and total_ram and mem_pct_total > mem_threshold_pct_total:
                    hit = True
                    kill_reason = f"mem>{mem_threshold_pct_total:.1f}% (saw {mem_pct_total:.1f}%)"

                if hit:
                    consecutive_hits += 1
                else:
                    consecutive_hits = 0

                if consecutive_hits >= max(1, consecutive_samples):
                    killed_for_resources = True
                    _emit(f"{prefix} RESOURCE_INTENSIVE: {kill_reason} -> terminating process group\n", is_err=True)
                    _terminate_process_group(proc, grace_s=kill_grace_s)
                    break

            except Exception:
                # Monitoring is best-effort; don't crash the run.
                continue

        rc = proc.wait()
        t_out.join(timeout=1)
        t_err.join(timeout=1)

    return MonitoredResult(
        returncode=rc,
        killed_for_resources=killed_for_resources,
        kill_reason=kill_reason,
        max_cpu_pct_total=max_cpu_pct,
        max_mem_pct_total=max_mem_pct,
        tail_stdout="".join(tail_out),
        tail_stderr="".join(tail_err),
    )


def _git_snapshot(project_root: Path) -> str:
    """A compact repo snapshot embedded into the prompt."""

    env = os.environ.copy()

    def safe(cmd: list[str]) -> str:
        rc, out, err = _run_cmd(cmd, cwd=project_root, env=env, stream=False)
        if rc != 0:
            return f"$ {' '.join(cmd)}\n(rc={rc})\n{(out + err).strip()}"
        return out.strip()

    if not (project_root / ".git").exists():
        return "(no git repo detected)"

    head = safe(["git", "rev-parse", "HEAD"])
    branch = safe(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = safe(["git", "status", "--porcelain=v1"])
    diff_stat = safe(["git", "diff", "--stat"])
    log = safe(["git", "log", "-5", "--oneline", "--decorate"])

    parts = [
        f"branch: {branch}\nHEAD: {head}",
        "\n# git status --porcelain\n" + (status or "(clean)"),
        "\n# git diff --stat\n" + (diff_stat or "(no diff)"),
        "\n# git log -5 --oneline --decorate\n" + (log or "(no commits?)"),
    ]
    return "\n".join(parts).strip()


# -----------------------------
# Config loading
# -----------------------------


def load_config(config_path: Path) -> RunConfig:
    data = _load_json(config_path)
    limits = data.get("limits", {})

    max_calls_per_agent = int(limits.get("max_calls_per_agent", 15))
    quota_retry_attempts = int(limits.get("quota_retry_attempts", 3))
    max_total_calls = int(limits.get("max_total_calls", 150))
    max_json_correction_attempts = int(limits.get("max_json_correction_attempts", 3))

    fallback_order = list(data.get("fallback_order", ["codex", "claude"]))
    enable_agents = list(data.get("enable_agents", ["codex", "claude"]))

    agents_cfg = data.get("agents", {})

    agent_scripts: dict[str, str] = {}
    agent_models: dict[str, str] = {}
    quota_pats: dict[str, list[str]] = {}
    supports_write: dict[str, bool] = {}
    agent_capabilities: dict[str, AgentCapabilities] = {}

    for a in AGENTS:
        if a not in agents_cfg:
            raise ValueError(f"Missing agent in config: {a}")
        agent_scripts[a] = str(agents_cfg[a].get("script", ""))
        agent_models[a] = str(agents_cfg[a].get("model", "(default)"))
        quota_pats[a] = list(agents_cfg[a].get("quota_error_patterns", []))
        supports_write[a] = bool(agents_cfg[a].get("supports_write_access", False))
        # Parse agent capabilities
        agent_capabilities[a] = AgentCapabilities(
            supports_tools=bool(agents_cfg[a].get("supports_tools", True)),
            supports_fs_read=bool(agents_cfg[a].get("supports_fs_read", True)),
            supports_fs_write=bool(agents_cfg[a].get("supports_fs_write", True)),
            supports_bash=bool(agents_cfg[a].get("supports_bash", True)),
            supports_write_access=supports_write[a],
        )

    parallel_cfg = data.get("parallel", {}) or {}
    parallel = ParallelSettings(
        max_workers_default=int(parallel_cfg.get("max_workers_default", 8)),
        cpu_intensive_threshold_pct=float(parallel_cfg.get("cpu_intensive_threshold_pct", 40.0)),
        mem_intensive_threshold_pct=float(parallel_cfg.get("mem_intensive_threshold_pct", 40.0)),
        sample_interval_s=float(parallel_cfg.get("sample_interval_s", 1.0)),
        consecutive_samples=int(parallel_cfg.get("consecutive_samples", 3)),
        kill_grace_s=float(parallel_cfg.get("kill_grace_s", 8.0)),
        terminal_max_bytes_per_worker=int(parallel_cfg.get("terminal_max_bytes_per_worker", 40000)),
        terminal_max_line_length=int(parallel_cfg.get("terminal_max_line_length", 600)),
        disable_gpu_by_default=bool(parallel_cfg.get("disable_gpu_by_default", True)),
    )

    return RunConfig(
        max_calls_per_agent=max_calls_per_agent,
        quota_retry_attempts=quota_retry_attempts,
        max_total_calls=max_total_calls,
        max_json_correction_attempts=max_json_correction_attempts,
        fallback_order=fallback_order,
        enable_agents=enable_agents,
        smoke_route=tuple(),
        agent_scripts=agent_scripts,
        agent_models=agent_models,
        quota_error_patterns=quota_pats,
        supports_write_access=supports_write,
        agent_capabilities=agent_capabilities,
        parallel=parallel,
    )


# -----------------------------
# Prompt building
# -----------------------------


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
    readonly: bool,
) -> str:
    last_summaries = "\n".join([f"- ({h['agent']}) {h['summary']}" for h in history[-4:]])

    state_blob = json.dumps(
        {
            "agent": agent,
            "milestone_id": milestone_id,
            "call_counts": call_counts,
            "disabled_by_quota": disabled_by_quota,
            "known_stats_ids": stats_ids,
            "readonly": readonly,
        },
        indent=2,
        sort_keys=True,
    )

    # Get agent policy header if in forced mode
    policy_header = get_agent_policy().get_prompt_header()

    parts: list[str] = [
        system_prompt.strip(),
    ]

    # Inject policy header right after system prompt if in forced mode
    if policy_header:
        parts.append("\n\n")
        parts.append(policy_header.strip())

    parts.extend(
        [
            "\n\n---\n\n# Orchestrator State (read-only)\n",
            state_blob,
            "\n\n---\n\n# Milestone Spec (DESIGN_DOCUMENT.md)\n",
            _truncate(design_doc_text.strip(), 20000),
            "\n\n---\n\n# Verification Report (tools.verify)\n",
            _truncate(verify_report_text.strip(), 20000),
            "\n\n---\n\n# Repo Snapshot\n",
            _truncate(repo_info.strip(), 20000),
            "\n\n---\n\n",
        ]
    )

    if last_summaries.strip():
        parts += ["# Recent Turn Summaries\n", _truncate(last_summaries.strip(), 5000), "\n\n---\n\n"]

    parts += ["# Your Work Item\n", next_prompt.strip() or "(Decide the next best step.)", "\n"]
    return "".join(parts)


# -----------------------------
# Turn parsing/validation
# -----------------------------


def _try_parse_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty output")

    # Fast path.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Extract first balanced JSON object.
    start = stripped.find("{")
    if start == -1:
        raise ValueError("no '{' found in output")

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                return json.loads(candidate)

    raise ValueError("unbalanced braces in output")


def _validate_turn(
    obj: Any, *, expected_agent: str, expected_milestone_id: str | None = None, stats_id_set: set[str]
) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "turn is not an object"

    required_keys = [
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

    for k in required_keys:
        if k not in obj:
            return False, f"missing key: {k}"

    if obj.get("agent") != expected_agent:
        return False, f"agent mismatch: expected {expected_agent}, got {obj.get('agent')}"

    # Validate milestone_id if expected_milestone_id is provided
    if expected_milestone_id is not None and str(obj.get("milestone_id")) != expected_milestone_id:
        return False, f"milestone_id mismatch: expected {expected_milestone_id}, got {obj.get('milestone_id')}"

    if obj["agent"] not in AGENTS or obj["next_agent"] not in AGENTS:
        return False, "invalid agent id in agent/next_agent"

    if obj.get("phase") not in ("plan", "implement", "verify", "finalize"):
        return False, "invalid phase"

    if not isinstance(obj["work_completed"], bool) or not isinstance(obj["project_complete"], bool):
        return False, "work_completed/project_complete must be boolean"

    for k in ("summary", "next_prompt", "delegate_rationale"):
        if not isinstance(obj.get(k), str):
            return False, f"{k} must be a string"

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
        if not isinstance(a.get("path"), str) or not isinstance(a.get("description"), str):
            return False, f"artifact[{i}] path/description must be strings"

    extra = set(obj.keys()) - set(required_keys)
    if extra:
        return False, f"unexpected keys present: {sorted(extra)}"

    return True, "ok"


# -----------------------------
# Contract hardening for agent outputs
# -----------------------------

# Patterns that indicate non-compliant agent output requiring strict reprompt
# CRITICAL: These patterns detect when agents incorrectly claim they can't use tools
NONCOMPLIANT_OUTPUT_PATTERNS: list[tuple[str, str]] = [
    # Tool availability claims - these are CRITICAL violations in execution mode
    (r"tools?\s+(are\s+)?(disabled|unavailable|not\s+available)", "tools_disabled_claim"),
    (r"cannot\s+(use|access|execute)\s+tools?", "tools_access_claim"),
    (r"don'?t\s+have\s+(access|permission)\s+to\s+tools?", "tools_permission_claim"),
    (r"tool\s+(execution|use|usage)\s+(is\s+)?(blocked|disabled)", "tools_execution_claim"),
    (r"i\s+(am\s+)?(unable|not\s+able)\s+to\s+(use|access|execute)\s+tools?", "tools_unable_claim"),
    (r"no\s+(access|permission)\s+to\s+(read|write|edit|execute)", "no_access_claim"),
    (r"(cannot|can't|unable)\s+(read|write|edit|modify)\s+(files?|code)", "file_access_claim"),
    (r"(cannot|can't|unable)\s+(run|execute)\s+(commands?|bash|shell)", "command_access_claim"),
    # Formatting violations
    (r"```(json|python|bash|)", "markdown_code_block"),
    (r"^(Here'?s?|I'?ll|Let me|Sure,)", "prose_prefix"),
    (r"^\s*#(?!\s*\{)", "comment_line_as_output"),  # Exclude lines that might be starting JSON with #
]

# Patterns that are CRITICAL violations requiring immediate retry with tools-enabled prompt
CRITICAL_TOOL_VIOLATIONS = {
    "tools_disabled_claim",
    "tools_access_claim",
    "tools_permission_claim",
    "tools_execution_claim",
    "tools_unable_claim",
    "no_access_claim",
    "file_access_claim",
    "command_access_claim",
}

# Hot files: critical integration points that should not be concurrently edited
# by multiple workers. When multiple tasks list these files in touched_paths,
# a shared lock is automatically injected to prevent merge conflicts.
HOT_FILES: tuple[str, ...] = (
    "**/api.py",
    "**/board_writer.py",
    "**/cli_main.py",
    "**/pipeline.py",
    "**/resolve.py",
    "**/manifest.py",
    "**/coupon_runner.py",
)


def _inject_hot_file_locks(tasks: list) -> list:
    """Inject shared locks for hot files touched by multiple tasks.

    When multiple tasks declare the same hot file in their touched_paths,
    this function adds a shared lock (e.g., 'hot:api.py') to serialize
    access and prevent merge conflicts.

    Args:
        tasks: List of ParallelTask objects

    Returns:
        Modified list with locks injected (same list, modified in place)
    """
    import fnmatch

    # Map hot file base names to task IDs that touch them
    hot_file_tasks: dict[str, list[str]] = {}

    for task in tasks:
        for path in task.touched_paths:
            path_str = str(path)
            for hot_pattern in HOT_FILES:
                if fnmatch.fnmatch(path_str, hot_pattern):
                    # Extract base name for lock key
                    base_name = path_str.split("/")[-1] if "/" in path_str else path_str
                    if base_name not in hot_file_tasks:
                        hot_file_tasks[base_name] = []
                    if task.id not in hot_file_tasks[base_name]:
                        hot_file_tasks[base_name].append(task.id)
                    break  # Don't match multiple patterns for same path

    # Inject locks where multiple tasks touch the same hot file
    tasks_by_id = {t.id: t for t in tasks}
    injected_count = 0

    for base_name, task_ids in hot_file_tasks.items():
        if len(task_ids) > 1:
            lock_name = f"hot:{base_name}"
            for tid in task_ids:
                task = tasks_by_id.get(tid)
                if task and lock_name not in task.locks:
                    task.locks.append(lock_name)
                    injected_count += 1

    if injected_count > 0:
        print(f"[orchestrator] hot-file guardrail: injected {injected_count} lock(s) for concurrent hot file access")

    return tasks


def _inject_overlap_locks(tasks: list) -> list:
    """Inject file-based locks for any files touched by multiple tasks.

    This provides generalized collision prevention beyond just hot files.
    When multiple tasks declare the same file in their touched_paths,
    this function adds a file-specific lock to serialize access.

    This is "as narrow as possible, as strong as necessary" locking:
    - Only files actually touched by >1 task get locks
    - Tasks with different touched_paths can run in parallel
    - Tasks missing touched_paths get a conservative directory-based lock

    Args:
        tasks: List of ParallelTask objects

    Returns:
        Modified list with locks injected (same list, modified in place)
    """
    import hashlib

    # Map file paths to task IDs that touch them
    file_to_tasks: dict[str, list[str]] = {}

    for task in tasks:
        if task.status != "pending":
            continue

        if task.touched_paths:
            for path in task.touched_paths:
                path_str = str(path).replace("\\", "/")
                if path_str not in file_to_tasks:
                    file_to_tasks[path_str] = []
                if task.id not in file_to_tasks[path_str]:
                    file_to_tasks[path_str].append(task.id)
        else:
            # For tasks missing touched_paths, derive a conservative lock from task title/description
            # This ensures tasks without declared paths don't silently collide
            # Use a hash of title to create a "bucket" lock
            title_hash = hashlib.md5(task.title.encode()).hexdigest()[:8]
            # Derive directory from task ID (e.g., M1-IMPLEMENT-FOO -> dir:M1)
            dir_lock = f"dir:unknown:{title_hash}"
            if dir_lock not in task.locks:
                task.locks.append(dir_lock)

    # Inject locks where multiple tasks touch the same file
    tasks_by_id = {t.id: t for t in tasks}
    injected_count = 0

    for file_path, task_ids in file_to_tasks.items():
        if len(task_ids) > 1:
            # Create a hash-based lock for the file path
            path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
            base_name = file_path.split("/")[-1] if "/" in file_path else file_path
            lock_name = f"file:{base_name}:{path_hash}"

            for tid in task_ids:
                task = tasks_by_id.get(tid)
                if task and lock_name not in task.locks:
                    task.locks.append(lock_name)
                    injected_count += 1

    if injected_count > 0:
        print(f"[orchestrator] overlap-lock: injected {injected_count} lock(s) for shared file access")

    return tasks


def _detect_noncompliant_output(text: str) -> tuple[bool, list[str], bool]:
    """Detect patterns in agent output that indicate non-compliance.

    Returns:
        (is_noncompliant, list_of_violations, has_critical_tool_violation)

    has_critical_tool_violation is True if the agent incorrectly claims tools
    are disabled - this requires immediate retry with explicit tools-enabled prompt.
    """
    violations: list[str] = []
    text_lower = text.lower()
    has_critical_tool_violation = False

    for pattern, description in NONCOMPLIANT_OUTPUT_PATTERNS:
        if re.search(pattern, text_lower, flags=re.IGNORECASE | re.MULTILINE):
            violations.append(description)
            # Check if this is a critical tool availability violation
            if description in CRITICAL_TOOL_VIOLATIONS:
                has_critical_tool_violation = True

    return bool(violations), violations, has_critical_tool_violation


def _build_strict_correction_prompt(
    agent: str,
    milestone_id: str,
    validation_error: str,
    task_title: str,
    task_description: str,
    noncompliant_violations: list[str] | None = None,
    attempt_number: int = 1,
    has_critical_tool_violation: bool = False,
) -> str:
    """Build an increasingly strict correction prompt based on attempt number.

    Attempt 1: Standard correction with clear requirements
    Attempt 2+: Stricter template with explicit prohibitions

    If has_critical_tool_violation is True, adds explicit "TOOLS ARE ENABLED" messaging
    to counteract any mistaken belief that tools are disabled.
    """
    stats_ref = "CL-1" if agent == "claude" else "CX-1"

    # Base violations text
    violations_text = ""
    if noncompliant_violations:
        violations_text = f"""
VIOLATIONS DETECTED: {', '.join(noncompliant_violations)}
These patterns are STRICTLY FORBIDDEN in your response."""

    # CRITICAL: If agent claimed tools were disabled, explicitly correct this
    tools_enabled_message = ""
    if has_critical_tool_violation:
        tools_enabled_message = """

## CRITICAL CORRECTION: TOOLS ARE ENABLED

Your previous response incorrectly claimed that tools are disabled.
THIS IS FALSE. You have FULL ACCESS to the following tools:
- Read: Read files from the filesystem
- Edit: Edit files with precise replacements
- Write: Create or overwrite files
- Bash: Execute shell commands
- Grep: Search file contents
- Glob: Find files by pattern

You CAN and SHOULD use these tools to complete your task.
DO NOT claim tools are disabled. Use them to implement the required changes.
"""

    if attempt_number == 1:
        return textwrap.dedent(f"""
            STRICT JSON CORRECTION REQUIRED
            {tools_enabled_message}
            Your previous response was INVALID: {validation_error}
            {violations_text}

            You MUST output ONLY a valid JSON object with these EXACT fields:
            - agent: "{agent}" (EXACTLY this string)
            - milestone_id: "{milestone_id}" (EXACTLY this string)
            - phase: "implement"
            - work_completed: true or false
            - project_complete: false
            - summary: "description of work"
            - gates_passed: []
            - requirement_progress: {{"covered_req_ids": [], "tests_added_or_modified": [], "commands_run": []}}
            - next_agent: "{agent}"
            - next_prompt: ""
            - delegate_rationale: ""
            - stats_refs: ["{stats_ref}"]
            - needs_write_access: true
            - artifacts: []

            CRITICAL RULES:
            - Output ONLY the JSON object, nothing else
            - NO markdown (no ```)
            - NO explanatory text before or after
            - NO comments
            - agent must be EXACTLY "{agent}"
            - milestone_id must be EXACTLY "{milestone_id}"

            FORBIDDEN PHRASES (do NOT output these):
            - "tools disabled" or "tools are disabled"
            - "cannot use tools" or "cannot access tools"
            - "don't have access to tools"
            - Any prose before/after the JSON

            Your task: {task_title}
            {task_description}
        """).strip()
    else:
        # Stricter template for attempt 2+
        return textwrap.dedent(f'''
            FINAL CORRECTION ATTEMPT - STRICT JSON ONLY
            {tools_enabled_message}
            ERROR: {validation_error}
            {violations_text}

            OUTPUT THIS EXACT JSON (modify values as needed):

            {{"agent": "{agent}", "milestone_id": "{milestone_id}", "phase": "implement", "work_completed": true, "project_complete": false, "summary": "Completed task", "gates_passed": [], "requirement_progress": {{"covered_req_ids": [], "tests_added_or_modified": [], "commands_run": []}}, "next_agent": "{agent}", "next_prompt": "", "delegate_rationale": "Task completed", "stats_refs": ["{stats_ref}"], "needs_write_access": true, "artifacts": []}}

            ABSOLUTE REQUIREMENTS:
            1. agent = "{agent}" (exact match required)
            2. milestone_id = "{milestone_id}" (exact match required)
            3. Output ONLY the JSON - no markdown, no prose, no explanation
            4. Do NOT claim tools are disabled - TOOLS ARE ENABLED
            5. Do NOT use ``` code blocks

            Task: {task_title}
        ''').strip()


def _is_noncompliant_and_should_use_stricter_prompt(text: str) -> tuple[bool, list[str], bool]:
    """Check if output warrants a stricter reprompt.

    Returns (should_use_stricter, violations, has_critical_tool_violation).

    has_critical_tool_violation indicates the agent incorrectly claimed tools are
    disabled, requiring explicit correction in the retry prompt.
    """
    is_noncompliant, violations, has_critical_tool_violation = _detect_noncompliant_output(text)

    # Also check if it looks like the agent dumped prose instead of JSON
    stripped = text.strip()
    if not stripped.startswith("{"):
        violations.append("output does not start with JSON")
        is_noncompliant = True

    return is_noncompliant, violations, has_critical_tool_violation


# NOTE: TurnNormalizer has been moved to bridge/loop_pkg/turn_normalizer.py
# for better tooling readability. Import is at the top of this file.


# -----------------------------
# Agent selection + quota (TurnNormalizer moved to loop_pkg/turn_normalizer.py)
# -----------------------------


# -----------------------------
# Agent selection + quota
# -----------------------------


def _is_quota_error(agent: str, text: str, config: RunConfig) -> bool:
    pats = config.quota_error_patterns.get(agent, [])
    return any(re.search(p, text, flags=re.IGNORECASE) for p in pats)


def _pick_fallback(config: RunConfig, state: RunState, current_agent: str | None) -> str:
    policy = get_agent_policy()

    # In forced mode, always return the forced agent (no fallback to other agent)
    if policy.forced_agent:
        return policy.enforce_strict(policy.forced_agent, "fallback selection")

    enabled = [
        a
        for a in config.enable_agents
        if (a in AGENTS)
        and (not state.disabled_by_quota.get(a, False))
        and (state.call_counts.get(a, 0) < config.max_calls_per_agent)
    ]

    if not enabled:
        return current_agent or AGENTS[0]

    enabled_others = [a for a in enabled if a != current_agent] or enabled

    for a in config.fallback_order:
        if a in enabled_others:
            return a

    return enabled_others[0]


def _other_agent(agent: str) -> str | None:
    if agent not in AGENTS:
        return None
    return "claude" if agent == "codex" else "codex"


def _override_next_agent(requested: str, config: RunConfig, state: RunState) -> tuple[str, str | None]:
    policy = get_agent_policy()

    # In forced mode, always return the forced agent
    if policy.forced_agent:
        forced = policy.enforce(requested, "next_agent override")
        if forced != requested:
            return forced, f"agent policy (--only-{policy.forced_agent})"
        return forced, None

    if config.smoke_route:
        idx = state.total_calls % len(config.smoke_route)
        routed, reason = resolve_smoke_route(requested=requested, route=config.smoke_route, index=idx)
        return routed, reason

    enabled = [
        a
        for a in config.enable_agents
        if (a in AGENTS)
        and (not state.disabled_by_quota.get(a, False))
        and (state.call_counts.get(a, 0) < config.max_calls_per_agent)
    ]

    if not enabled:
        return requested, None

    current = state.history[-1]["agent"] if state.history else None
    if current:
        other = _other_agent(current)
        if other and other in enabled:
            if requested != other:
                return other, "two-agent alternation (codex <-> claude)"
            return other, None

    if requested in enabled:
        return requested, None

    return enabled[0], f"requested agent '{requested}' disabled or unknown"


# -----------------------------
# Git helpers + verify
# -----------------------------


def _git_init_if_needed(project_root: Path) -> None:
    if not (project_root / ".git").exists():
        _run_cmd(["git", "init"], cwd=project_root, env=os.environ.copy())
        _run_cmd(["git", "config", "user.email", "agent@example.com"], cwd=project_root, env=os.environ.copy())
        _run_cmd(["git", "config", "user.name", "Agent Runner"], cwd=project_root, env=os.environ.copy())
        _run_cmd(["git", "add", "-A"], cwd=project_root, env=os.environ.copy())
        _run_cmd(["git", "commit", "-m", "chore: initial state"], cwd=project_root, env=os.environ.copy())


# Protected file that must never be reverted by auto-stash or checkout
_PROTECTED_FILE = "DESIGN_DOCUMENT.md"


def _capture_protected_file(project_root: Path) -> tuple[bytes | None, str | None]:
    """Capture current bytes and SHA256 of DESIGN_DOCUMENT.md before git operations.

    Returns (content_bytes, sha256_hex) or (None, None) if file doesn't exist.
    """
    protected_path = project_root / _PROTECTED_FILE
    if not protected_path.exists():
        return None, None
    try:
        content = protected_path.read_bytes()
        sha256_hex = hashlib.sha256(content).hexdigest()
        return content, sha256_hex
    except OSError:
        return None, None


def _restore_protected_file_if_changed(
    project_root: Path,
    original_content: bytes | None,
    original_hash: str | None,
    operation_name: str,
) -> bool:
    """Restore DESIGN_DOCUMENT.md if it was changed by a git operation.

    Returns True if file was restored, False otherwise.
    """
    if original_content is None:
        # File didn't exist before, nothing to restore
        return False

    protected_path = project_root / _PROTECTED_FILE

    # Check current state
    if not protected_path.exists():
        # File was deleted by git operation - restore it
        try:
            protected_path.write_bytes(original_content)
            print(f"[orchestrator] Restored {_PROTECTED_FILE} after {operation_name} (file was deleted)")
            return True
        except OSError as e:
            print(f"[orchestrator] WARNING: Failed to restore {_PROTECTED_FILE}: {e}")
            return False

    try:
        current_content = protected_path.read_bytes()
        current_hash = hashlib.sha256(current_content).hexdigest()
    except OSError:
        current_hash = None

    if current_hash != original_hash:
        # Content changed - restore original
        try:
            protected_path.write_bytes(original_content)
            print(f"[orchestrator] Restored {_PROTECTED_FILE} after {operation_name} to preserve local edits")
            return True
        except OSError as e:
            print(f"[orchestrator] WARNING: Failed to restore {_PROTECTED_FILE}: {e}")
            return False

    return False


def _checkout_agent_branch(project_root: Path, run_id: str) -> None:
    branch = f"agent-run/{run_id}"

    # Capture protected file state before checkout
    original_content, original_hash = _capture_protected_file(project_root)

    rc, _, _ = _run_cmd(["git", "checkout", "-b", branch], cwd=project_root, env=os.environ.copy())
    if rc != 0:
        _run_cmd(["git", "checkout", branch], cwd=project_root, env=os.environ.copy())

    # Restore protected file if it was changed by checkout
    _restore_protected_file_if_changed(project_root, original_content, original_hash, "checkout")


def _run_verify(project_root: Path, out_json: Path, strict_git: bool) -> tuple[int, str, str]:
    cmd = [sys.executable, "-m", "tools.verify", "--json", str(out_json)]
    if strict_git:
        cmd.append("--strict-git")
    return _run_cmd(cmd, cwd=project_root, env=os.environ.copy())


def _completion_gates_ok(project_root: Path) -> tuple[bool, str]:
    """The repo is 'complete' when strict verify passes and git status is clean."""

    env = os.environ.copy()
    rc, out, err = _run_cmd([sys.executable, "-m", "tools.verify", "--strict-git", "--include-m0"], cwd=project_root, env=env)
    if rc != 0:
        return False, (out + "\n" + err).strip()

    rc2, porcelain, err2 = _run_cmd(["git", "status", "--porcelain=v1"], cwd=project_root, env=env)
    if rc2 != 0:
        return False, (porcelain + "\n" + err2).strip()
    if porcelain.strip():
        return False, "git status not clean"

    return True, "ok"


def _preflight_check_repo(
    project_root: Path,
    *,
    auto_stash: bool = False,
    force_dirty: bool = False,
    runs_dir: Path | None = None,
) -> tuple[bool, str, str | None]:
    """Preflight check for dirty repo before starting parallel runs.

    Returns:
        (ok, message, stash_ref) where:
        - ok: True if repo is clean or was auto-stashed
        - message: Human-readable status message
        - stash_ref: The stash reference if auto-stash was used, else None
    """
    env = os.environ.copy()

    # Check for uncommitted changes
    rc, porcelain, _ = _run_cmd(["git", "status", "--porcelain=v1"], cwd=project_root, env=env)
    if rc != 0:
        return False, "Failed to run git status", None

    is_dirty = bool(porcelain.strip())

    # Also check for unmerged files (merge conflict state)
    rc2, unmerged, _ = _run_cmd(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=project_root,
        env=env,
    )
    has_conflicts = rc2 == 0 and bool(unmerged.strip())

    if has_conflicts:
        conflict_files = unmerged.strip().split("\n")
        return False, f"PREFLIGHT FAILED: Unresolved merge conflicts in: {conflict_files}", None

    if not is_dirty:
        return True, "Repository is clean", None

    # Repo is dirty
    if force_dirty:
        return True, "WARNING: Proceeding with dirty repo (--force-dirty)", None

    if auto_stash:
        # Capture protected file state BEFORE stash (pathspec excludes are unreliable)
        original_content, original_hash = _capture_protected_file(project_root)

        # Stash all changes including untracked files
        # Note: pathspec exclude kept for belt-and-suspenders, but we don't rely on it
        stash_msg = f"orchestrator-auto-stash-{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        rc_stash, out_stash, err_stash = _run_cmd(
            ["git", "stash", "push", "-u", "-m", stash_msg, "--", ":/", ":(exclude):/DESIGN_DOCUMENT.md"],
            cwd=project_root,
            env=env,
        )
        if rc_stash != 0:
            return False, f"Failed to auto-stash: {err_stash}", None

        # CRITICAL: Restore protected file if stash reverted it (pathspec exclude unreliable)
        _restore_protected_file_if_changed(project_root, original_content, original_hash, "auto-stash")

        # Get the stash ref
        rc_ref, stash_list, _ = _run_cmd(
            ["git", "stash", "list", "--format=%gd %s"],
            cwd=project_root,
            env=env,
        )
        stash_ref = None
        if rc_ref == 0:
            for line in stash_list.strip().split("\n"):
                if stash_msg in line:
                    stash_ref = line.split()[0] if line else None
                    break

        # Write stash info to runs_dir if provided
        if runs_dir and stash_ref:
            _ensure_dir(runs_dir)
            stash_info_path = runs_dir / "stash_info.txt"
            stash_info_path.write_text(
                f"Auto-stash created: {stash_ref}\n"
                f"Message: {stash_msg}\n\n"
                f"To restore:\n  git stash pop {stash_ref}\n\n"
                f"Changes stashed:\n{porcelain}\n",
                encoding="utf-8",
            )

        return True, f"Auto-stashed uncommitted changes to {stash_ref}", stash_ref

    # Fail fast with clear error
    dirty_files = porcelain.strip().split("\n")[:10]  # First 10 files
    msg = (
        f"PREFLIGHT FAILED: Repository has uncommitted changes.\n"
        f"  Dirty files ({len(porcelain.strip().split(chr(10)))} total):\n"
        + "\n".join(f"    {f}" for f in dirty_files)
        + "\n\n"
        "  This will cause tools.verify --strict-git to fail during the run.\n"
        "  Options:\n"
        "    1. Commit or stash changes before running\n"
        "    2. Use --auto-stash to automatically stash changes\n"
        "    3. Use --force-dirty --verify-mode=skip-git to proceed anyway (not recommended)\n"
    )
    return False, msg, None


def _ast_extract_init_components(source: str) -> tuple[bool, str | None, list[str], list[str], list[tuple[str, str]]]:
    """Extract components from __init__.py source using AST for safety.

    Returns:
        (valid, docstring, imports, all_items, safe_assignments)

    safe_assignments is a list of (name, source_line) for simple string/int/version assignments.
    """
    imports: list[str] = []
    all_items: list[str] = []
    safe_assignments: list[tuple[str, str]] = []
    docstring: str | None = None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False, None, [], [], []

    # Get docstring
    docstring = ast.get_docstring(tree)

    source_lines = source.split("\n")

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            # Module docstring - already handled
            continue

        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # Reconstruct import statement from source lines
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            import_lines = source_lines[start:end]
            import_str = "\n".join(import_lines).strip()
            if import_str:
                imports.append(import_str)

        elif isinstance(node, ast.Assign):
            # Check for __all__ or safe constant assignments
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "__all__" and isinstance(node.value, ast.List):
                        # Extract __all__ items
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                all_items.append(elt.value)
                    elif target.id.startswith("_") or target.id.isupper() or target.id == "__version__":
                        # Safe assignments: private vars, constants (ALL_CAPS), __version__
                        if isinstance(node.value, ast.Constant):
                            value = node.value.value
                            if isinstance(value, str):
                                safe_assignments.append((target.id, f'{target.id} = "{value}"'))
                            elif isinstance(value, (int, float)):
                                safe_assignments.append((target.id, f'{target.id} = {value}'))

    return True, docstring, imports, all_items, safe_assignments


def _text_extract_init_components(source: str) -> tuple[str | None, set[str], set[str]]:
    """Fallback text-based extraction for malformed Python.

    Returns:
        (docstring, imports, all_items)
    """
    imports: set[str] = set()
    all_items: set[str] = set()
    docstring: str | None = None

    lines = source.split("\n")
    in_docstring = False
    docstring_lines: list[str] = []
    in_multiline_all = False
    all_buffer: list[str] = []
    in_multiline_import = False
    import_buffer: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Handle docstrings
        if not docstring:
            if '"""' in stripped or "'''" in stripped:
                quote = '"""' if '"""' in stripped else "'''"
                if not in_docstring:
                    in_docstring = True
                    docstring_lines.append(line)
                    # Check for single-line docstring
                    if stripped.count(quote) >= 2 and not stripped.endswith(quote + quote):
                        in_docstring = False
                        docstring = "\n".join(docstring_lines)
                else:
                    docstring_lines.append(line)
                    in_docstring = False
                    docstring = "\n".join(docstring_lines)
                continue
            elif in_docstring:
                docstring_lines.append(line)
                continue

        # Handle multiline imports (with parentheses)
        if in_multiline_import:
            # Check if this line starts a new import (abort the incomplete one)
            if stripped.startswith("from ") or stripped.startswith("import "):
                in_multiline_import = False
                import_buffer = []
                # Don't continue - fall through to process as new import
            elif ")" in stripped:
                import_buffer.append(stripped)
                in_multiline_import = False
                # Validate the import is complete
                full_import = " ".join(import_buffer)
                if full_import.count("(") == full_import.count(")"):
                    imports.add(full_import)
                import_buffer = []
                continue
            else:
                import_buffer.append(stripped)
                continue

        # Handle multiline __all__
        if in_multiline_all:
            all_buffer.append(stripped)
            if "]" in stripped:
                in_multiline_all = False
                full_all = " ".join(all_buffer)
                for item in re.findall(r'"([^"]+)"|\'([^\']+)\'', full_all):
                    all_items.add(item[0] or item[1])
                all_buffer = []
            continue

        # Handle imports
        if stripped.startswith("from ") and " import " in stripped:
            # Check for multiline import (unclosed paren)
            if "(" in stripped and ")" not in stripped:
                in_multiline_import = True
                import_buffer = [stripped]
            elif "(" in stripped and ")" in stripped:
                # Complete single-line import with parens
                imports.add(stripped)
            elif "(" not in stripped:
                # Simple import without parens
                imports.add(stripped)
            # Skip incomplete imports (has open paren but no close)
        elif stripped.startswith("import "):
            imports.add(stripped)

        # Handle __all__ (single line or start of multiline)
        if "__all__" in stripped and "=" in stripped:
            if "[" in stripped and "]" in stripped:
                # Single line __all__
                for item in re.findall(r'"([^"]+)"|\'([^\']+)\'', stripped):
                    all_items.add(item[0] or item[1])
            elif "[" in stripped:
                # Start of multiline __all__
                in_multiline_all = True
                all_buffer.append(stripped)

    return docstring, imports, all_items


def _parse_conflict_blocks(content: str) -> list[tuple[str, str]]:
    """Parse all conflict blocks from content.

    Returns list of (ours_content, theirs_content) tuples for each conflict block.
    """
    conflicts: list[tuple[str, str]] = []
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        if lines[i].startswith("<<<<<<< "):
            ours_lines: list[str] = []
            theirs_lines: list[str] = []
            i += 1

            # Collect "ours" side
            while i < len(lines) and not lines[i].startswith("======="):
                ours_lines.append(lines[i])
                i += 1

            i += 1  # Skip =======

            # Collect "theirs" side
            while i < len(lines) and not lines[i].startswith(">>>>>>> "):
                theirs_lines.append(lines[i])
                i += 1

            conflicts.append(("\n".join(ours_lines), "\n".join(theirs_lines)))
        i += 1

    return conflicts


def _remove_conflict_markers(content: str) -> str:
    """Remove conflict markers while keeping common content."""
    lines = content.split("\n")
    result: list[str] = []
    state = "normal"

    for line in lines:
        if line.startswith("<<<<<<< "):
            state = "ours"
        elif line.startswith("=======") and state == "ours":
            state = "theirs"
        elif line.startswith(">>>>>>> ") and state == "theirs":
            state = "normal"
        elif state == "normal":
            result.append(line)
        # In conflict state, we skip lines (they'll be re-added by merge)

    return "\n".join(result)


def _write_manual_resolution_artifact(file_path: Path, reason: str, ours: str, theirs: str) -> None:
    """Write a .manual_resolution file when auto-resolution fails."""
    artifact_path = file_path.parent / f"{file_path.name}.manual_resolution"
    content = f"""# MANUAL RESOLUTION REQUIRED
# File: {file_path}
# Reason: {reason}
# Timestamp: {dt.datetime.now(dt.timezone.utc).isoformat()}

# === OURS (HEAD) ===
{ours}

# === THEIRS (incoming) ===
{theirs}

# === ACTION REQUIRED ===
# 1. Edit {file_path} to resolve the conflict manually
# 2. Run: python3 -m py_compile {file_path}
# 3. Run: git add {file_path}
# 4. Delete this file: rm {artifact_path}
"""
    try:
        artifact_path.write_text(content, encoding="utf-8")
    except Exception:
        pass  # Best effort


def _auto_resolve_init_py_conflict(file_path: Path, project_root: Path) -> tuple[bool, str]:
    """Auto-resolve merge conflict in __init__.py files using AST-safe union merge.

    Strategy:
    1. Parse all conflict blocks
    2. Try AST-based extraction on each side (fall back to text-based for invalid syntax)
    3. Extract: docstring, imports, __all__ items, safe assignments
    4. Union-merge: deduplicate imports, union __all__, prefer longer docstring
    5. Validate result with py_compile
    6. If unresolvable, write manual resolution artifact

    Returns:
        (success, message)
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Cannot read conflict file: {e}"

    # Check for conflict markers
    has_markers = "<<<<<<< " in content and "=======" in content and ">>>>>>> " in content

    if not has_markers:
        return False, "No conflict markers found"

    # Parse all conflict blocks
    conflict_blocks = _parse_conflict_blocks(content)
    if not conflict_blocks:
        return False, "Failed to parse conflict blocks"

    # Get common content (before/after conflict blocks, with markers removed)
    common_content = _remove_conflict_markers(content)

    # Collect all components
    all_docstrings: list[str] = []
    all_imports: set[str] = set()
    all_all_items: set[str] = set()
    all_safe_assigns: dict[str, str] = {}

    # Process common content
    common_valid, common_doc, common_imports, common_all, common_assigns = _ast_extract_init_components(common_content)
    if common_valid:
        if common_doc:
            all_docstrings.append(common_doc)
        all_imports.update(common_imports)
        all_all_items.update(common_all)
        for name, line in common_assigns:
            all_safe_assigns[name] = line
    else:
        # Fallback to text extraction
        doc, imps, items = _text_extract_init_components(common_content)
        if doc:
            all_docstrings.append(doc)
        all_imports.update(imps)
        all_all_items.update(items)

    # Process each conflict block
    ours_full: list[str] = []
    theirs_full: list[str] = []

    for ours_content, theirs_content in conflict_blocks:
        ours_full.append(ours_content)
        theirs_full.append(theirs_content)

        # Process "ours" side
        ours_valid, ours_doc, ours_imports, ours_all, ours_assigns = _ast_extract_init_components(ours_content)
        if ours_valid:
            if ours_doc:
                all_docstrings.append(ours_doc)
            all_imports.update(ours_imports)
            all_all_items.update(ours_all)
            for name, line in ours_assigns:
                all_safe_assigns[name] = line
        else:
            # Fallback to text extraction for invalid syntax
            doc, imps, items = _text_extract_init_components(ours_content)
            if doc:
                all_docstrings.append(doc)
            all_imports.update(imps)
            all_all_items.update(items)

        # Process "theirs" side
        theirs_valid, theirs_doc, theirs_imports, theirs_all, theirs_assigns = _ast_extract_init_components(theirs_content)
        if theirs_valid:
            if theirs_doc:
                all_docstrings.append(theirs_doc)
            all_imports.update(theirs_imports)
            all_all_items.update(theirs_all)
            for name, line in theirs_assigns:
                all_safe_assigns[name] = line
        else:
            # Fallback to text extraction for invalid syntax
            doc, imps, items = _text_extract_init_components(theirs_content)
            if doc:
                all_docstrings.append(doc)
            all_imports.update(imps)
            all_all_items.update(items)

    # Choose docstring (prefer longer)
    docstring = '"""Package exports."""'
    if all_docstrings:
        all_docstrings.sort(key=len, reverse=True)
        docstring = all_docstrings[0]
        # Ensure docstring is properly formatted
        if not docstring.startswith(('"""', "'''")):
            docstring = f'"""{docstring}"""'

    # Sort imports for determinism (group from imports and regular imports)
    from_imports = sorted([i for i in all_imports if i.startswith("from ")])
    regular_imports = sorted([i for i in all_imports if i.startswith("import ")])
    sorted_imports = from_imports + regular_imports

    # Sort __all__ items for determinism
    sorted_all_items = sorted(all_all_items)

    # Build result
    result_lines: list[str] = []

    # Add docstring
    result_lines.append(docstring)
    result_lines.append("")

    # Add imports
    for imp in sorted_imports:
        result_lines.append(imp)

    if sorted_imports:
        result_lines.append("")

    # Add safe assignments (like __version__)
    for name in sorted(all_safe_assigns.keys()):
        result_lines.append(all_safe_assigns[name])

    if all_safe_assigns:
        result_lines.append("")

    # Add __all__
    if sorted_all_items:
        result_lines.append("__all__ = [")
        for item in sorted_all_items:
            result_lines.append(f'    "{item}",')
        result_lines.append("]")

    result_content = "\n".join(result_lines)
    if not result_content.endswith("\n"):
        result_content += "\n"

    # Validate syntax with compile()
    try:
        compile(result_content, str(file_path), "exec")
    except SyntaxError as e:
        # Write manual resolution artifact
        _write_manual_resolution_artifact(
            file_path,
            f"Generated code has syntax error: {e}",
            "\n".join(ours_full),
            "\n".join(theirs_full),
        )
        return False, f"Generated code has syntax error: {e}. Manual resolution artifact written."

    # Double-check with py_compile for extra safety
    import py_compile
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(result_content)
        tf_path = tf.name

    try:
        py_compile.compile(tf_path, doraise=True)
    except py_compile.PyCompileError as e:
        os.unlink(tf_path)
        _write_manual_resolution_artifact(
            file_path,
            f"py_compile failed: {e}",
            "\n".join(ours_full),
            "\n".join(theirs_full),
        )
        return False, f"py_compile validation failed: {e}. Manual resolution artifact written."
    finally:
        if os.path.exists(tf_path):
            os.unlink(tf_path)

    # Write resolved file
    try:
        file_path.write_text(result_content, encoding="utf-8")
    except Exception as e:
        return False, f"Cannot write resolved file: {e}"

    return True, "Auto-resolved __init__.py conflict by AST-safe union merge"


def _attempt_auto_merge_resolution(
    project_root: Path,
    task_id: str,
    runs_dir: Path,
    task_context: str = "",
    milestone_id: str = "M0",
) -> tuple[bool, str]:
    """Attempt to auto-resolve merge conflicts.

    This function:
    1. Tries built-in resolution for __init__.py files (deterministic merge)
    2. Uses agent-based resolution for other files (Claude-powered intelligent merge)
    3. Falls back to manual only if all automated approaches fail

    Returns:
        (success, message)
    """
    env = os.environ.copy()

    # Get list of conflicted files
    rc, conflict_out, _ = _run_cmd(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=project_root,
        env=env,
    )
    if rc != 0 or not conflict_out.strip():
        return False, "No conflict files detected or git command failed"

    conflict_files = [f.strip() for f in conflict_out.strip().split("\n") if f.strip()]
    if not conflict_files:
        return False, "No conflict files"

    resolved_files: list[str] = []
    unresolved_files: list[str] = []
    init_py_files: list[str] = []
    other_files: list[str] = []

    # Categorize files
    for cf in conflict_files:
        if cf.endswith("__init__.py"):
            init_py_files.append(cf)
        else:
            other_files.append(cf)

    # Phase 1: Try built-in resolution for __init__.py files (deterministic)
    for cf in init_py_files:
        file_path = project_root / cf
        success, msg = _auto_resolve_init_py_conflict(file_path, project_root)
        if success:
            # Stage the resolved file
            _run_cmd(["git", "add", cf], cwd=project_root, env=env)
            resolved_files.append(cf)
            print(f"[orchestrator] AUTO-RESOLVED (__init__.py): {cf} ({msg})")
        else:
            unresolved_files.append(cf)
            print(f"[orchestrator] CANNOT AUTO-RESOLVE (__init__.py): {cf} ({msg})")

    # Phase 2: Try agent-based resolution for other files
    if other_files:
        print(f"[orchestrator] Trying agent-based merge resolution for {len(other_files)} file(s): {other_files}")
        try:
            from bridge.merge_resolver import attempt_agent_merge_resolution

            merge_result = attempt_agent_merge_resolution(
                project_root=project_root,
                runs_dir=runs_dir,
                task_id=task_id,
                task_context=task_context,
                milestone_id=milestone_id,
                max_attempts=3,
            )

            if merge_result.success:
                resolved_files.extend(merge_result.resolved_files)
                print(f"[orchestrator] AGENT-RESOLVED: {merge_result.resolved_files} after {merge_result.attempt} attempt(s)")
            else:
                unresolved_files.extend(merge_result.unresolved_files)
                print(f"[orchestrator] AGENT RESOLUTION FAILED: {merge_result.error}")
                print(f"[orchestrator] Unresolved: {merge_result.unresolved_files}")

        except Exception as e:
            print(f"[orchestrator] Agent merge resolver exception: {e}")
            unresolved_files.extend(other_files)

    if unresolved_files:
        return False, f"Unresolved conflicts remain: {unresolved_files}"

    # All conflicts resolved - commit
    if resolved_files:
        commit_msg = f"auto-resolve: merge conflict in {', '.join(resolved_files)}"
        rc_commit, _, err_commit = _run_cmd(
            ["git", "commit", "-m", commit_msg],
            cwd=project_root,
            env=env,
        )
        if rc_commit != 0:
            return False, f"Failed to commit resolved conflicts: {err_commit}"

        # Log resolution
        resolution_log_path = runs_dir / "auto_merge_resolutions.txt"
        with resolution_log_path.open("a", encoding="utf-8") as f:
            f.write(f"Task: {task_id}\n")
            f.write(f"Resolved: {resolved_files}\n")
            f.write(f"Commit: {commit_msg}\n")
            f.write("-" * 40 + "\n")

        return True, f"Auto-resolved conflicts in: {resolved_files}"

    return False, "No files were resolved"


# -----------------------------
# Agent runners
# -----------------------------


def _run_agent_live(
    *,
    agent: str,
    prompt_path: Path,
    schema_path: Path,
    out_path: Path,
    config: RunConfig,
    state: RunState,
    stream_agent_output: str = "none",
    call_dir: Path | None = None,
) -> tuple[int, str, str]:
    script_rel = config.agent_scripts.get(agent, "")
    if not script_rel:
        return 1, "", f"No script configured for agent {agent}"

    script_path = state.project_root / script_rel
    if not script_path.exists():
        return 1, "", f"Agent script not found: {script_path}"

    _ensure_dir(out_path.parent)

    cmd = [
        str(script_path),
        str(prompt_path),
        str(schema_path),
        str(out_path),
    ]

    env = os.environ.copy()
    write_access = state.grant_write_access and not state.readonly
    # Support both names (some wrappers read WRITE_ACCESS, older versions used ORCH_WRITE_ACCESS).
    env["WRITE_ACCESS"] = "1" if write_access else "0"
    env["ORCH_WRITE_ACCESS"] = env["WRITE_ACCESS"]
    env["ORCH_READONLY"] = "1" if state.readonly else "0"
    # Signal to the agent wrapper that this is a turn schema (enables turn normalization)
    env["ORCH_SCHEMA_KIND"] = "turn"

    if call_dir:
        return run_cmd_with_streaming(
            cmd=cmd,
            cwd=state.project_root,
            env=env,
            agent=agent,
            stream_mode=stream_agent_output,
            call_dir=call_dir,
        )

    stream_enabled = stream_agent_output != "none"
    return _run_cmd(cmd, cwd=state.project_root, env=env, stream=stream_enabled)


def _run_agent_mock(*, agent: str, scenario: dict[str, Any], mock_indices: dict[str, int]) -> tuple[int, str, str]:
    block = scenario.get("agents", {}).get(agent, [])
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


# -----------------------------
# Parallel runner
# -----------------------------


@dataclasses.dataclass
class ParallelTask:
    id: str
    title: str
    description: str
    agent: str
    intensity: str = "low"  # low|medium|high
    locks: list[str] = dataclasses.field(default_factory=list)
    touched_paths: list[str] = dataclasses.field(default_factory=list)
    depends_on: list[str] = dataclasses.field(default_factory=list)
    solo: bool = False

    # runtime
    # Status values:
    #   pending - not yet started
    #   running - currently executing
    #   done - completed successfully
    #   failed - execution failed (root failure)
    #   manual - needs manual intervention (root failure)
    #   blocked - transitively blocked by a root failure
    #   pending_rerun - agent returned work_completed=false (root failure, needs retry)
    #   resource_killed - killed due to resource limits (root failure)
    status: str = "pending"
    worker_id: int | None = None
    branch: str | None = None
    base_sha: str | None = None
    worktree_path: Path | None = None
    task_dir: Path | None = None
    prompt_path: Path | None = None
    out_path: Path | None = None
    raw_log_path: Path | None = None
    manual_path: Path | None = None
    error: str | None = None
    max_cpu_pct_total: float = 0.0
    max_mem_pct_total: float = 0.0
    # Turn output tracking
    work_completed: bool | None = None
    commit_sha: str | None = None
    turn_summary: str | None = None
    # Patch integration (commit-free worker operation)
    patch_path: Path | None = None
    has_patch: bool = False
    # Retry tracking for plan-only responses
    retry_count: int = 0
    max_retries: int = 2


def _sanitize_branch_fragment(text: str) -> str:
    frag = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    frag = frag.strip("-.")
    return frag or "task"


def _collect_machine_info() -> dict[str, Any]:
    cores = os.cpu_count() or 1
    total_ram = _total_ram_bytes()
    return {
        "cpu_cores": int(cores),
        "ram_bytes": int(total_ram) if total_ram else None,
    }


def _analyze_plan_width(
    plan_obj: dict[str, Any],
    max_workers_limit: int,
) -> dict[str, Any]:
    """Analyze a plan for throughput mode compliance.

    Returns:
        dict with keys:
        - task_count: int
        - root_count: int (tasks with empty depends_on)
        - plan_cap: int (max_parallel_tasks from plan)
        - min_plan_cap: int (required minimum for throughput)
        - min_root: int (required minimum roots for throughput)
        - min_task_count: int (recommended, not required)
        - is_thin: bool (True only if HARD requirements not met)
        - issues: list[str] (human-readable issues - hard failures)
        - warnings: list[str] (soft warnings - don't trigger reprompt)

    Note: is_thin is True only for HARD failures (plan_cap, root_count).
    total_tasks < 2*workers is a soft warning that does NOT trigger reprompt.
    """
    tasks = plan_obj.get("tasks", [])
    task_count = len(tasks) if isinstance(tasks, list) else 0
    root_count = sum(1 for t in (tasks if isinstance(tasks, list) else [])
                     if isinstance(t, dict) and not t.get("depends_on"))
    plan_cap = int(plan_obj.get("max_parallel_tasks", 1) or 1)

    # Throughput mode thresholds
    min_plan_cap = max_workers_limit
    min_root = max_workers_limit
    min_task_count = max_workers_limit * 2  # Recommended (soft), not required

    # HARD failures (must reprompt)
    issues: list[str] = []
    if plan_cap < min_plan_cap:
        issues.append(f"max_parallel_tasks={plan_cap} < required {min_plan_cap}")
    if root_count < min_root:
        issues.append(f"root_tasks={root_count} < required {min_root}")

    # SOFT warnings (do NOT trigger reprompt on their own)
    warnings: list[str] = []
    if task_count < min_task_count:
        warnings.append(f"total_tasks={task_count} < recommended {min_task_count}")

    return {
        "task_count": task_count,
        "root_count": root_count,
        "plan_cap": plan_cap,
        "min_plan_cap": min_plan_cap,
        "min_root": min_root,
        "min_task_count": min_task_count,
        "is_thin": len(issues) > 0,  # Only hard failures trigger is_thin
        "issues": issues,
        "warnings": warnings,
    }


# ------------------------------------
# Plan Quality Scoring (Throughput Mode)
# ------------------------------------

def _compute_lock_pressure(plan_obj: dict[str, Any]) -> dict[str, Any]:
    """Compute lock pressure metrics for the plan.

    Returns dict with:
    - lock_counts: dict mapping lock key -> count of tasks using it
    - max_lock_count: highest count for any single lock
    - high_pressure_locks: list of locks with > 2 tasks (likely serialization)
    - pressure_score: 0.0 (no pressure) to 1.0 (severe serialization)
    """
    tasks = plan_obj.get("tasks", [])
    if not isinstance(tasks, list):
        return {
            "lock_counts": {},
            "max_lock_count": 0,
            "high_pressure_locks": [],
            "pressure_score": 0.0,
        }

    lock_counts: dict[str, int] = {}
    for t in tasks:
        if not isinstance(t, dict):
            continue
        locks = t.get("locks", [])
        if not isinstance(locks, list):
            continue
        for lock in locks:
            if isinstance(lock, str) and lock.strip():
                lock_counts[lock.strip()] = lock_counts.get(lock.strip(), 0) + 1

    max_lock_count = max(lock_counts.values()) if lock_counts else 0
    high_pressure_locks = [k for k, v in lock_counts.items() if v > 2]

    # Pressure score: 0 if max <= 2, scaled up to 1.0 if max >= 8
    pressure_score = 0.0
    if max_lock_count > 2:
        pressure_score = min(1.0, (max_lock_count - 2) / 6)

    return {
        "lock_counts": lock_counts,
        "max_lock_count": max_lock_count,
        "high_pressure_locks": high_pressure_locks,
        "pressure_score": pressure_score,
    }


def _compute_task_focus_score(task: dict[str, Any]) -> float:
    """Compute focus score for a single task (0.0 = vague, 1.0 = highly focused).

    Scoring:
    - +0.3 if description mentions specific files/modules (patterns: .py, .ts, src/, tests/)
    - +0.2 if description mentions specific functions/classes (patterns: def, class, function)
    - +0.2 if description mentions tests or test coverage
    - +0.2 if description is sufficiently detailed (>100 chars)
    - +0.1 if title is specific (not generic like "implement", "do", "setup")
    - -0.3 if description is vague (contains "subsystem", "everything", "all of")
    """
    desc = str(task.get("description", "")).lower()
    title = str(task.get("title", "")).lower()

    score = 0.0

    # +0.3 for file/module references
    file_patterns = [".py", ".ts", ".js", ".json", "src/", "tests/", "bridge/", "docs/"]
    if any(pat in desc for pat in file_patterns):
        score += 0.3

    # +0.2 for function/class references
    code_patterns = ["def ", "class ", "function ", "const ", "import ", "from "]
    if any(pat in desc for pat in code_patterns):
        score += 0.2

    # +0.2 for test mentions
    test_patterns = ["test", "pytest", "assert", "validate", "verify", "coverage"]
    if any(pat in desc for pat in test_patterns):
        score += 0.2

    # +0.2 for detailed description
    if len(desc) > 100:
        score += 0.2

    # +0.1 for specific title (not generic)
    vague_titles = ["implement", "do ", "setup", "handle", "work on", "complete"]
    if not any(vt in title for vt in vague_titles):
        score += 0.1

    # -0.3 for vague descriptions
    vague_patterns = ["subsystem", "everything", "all of ", "entire ", "whole "]
    if any(vp in desc for vp in vague_patterns):
        score -= 0.3

    return max(0.0, min(1.0, score))


def _compute_coverage_intent(plan_obj: dict[str, Any]) -> float:
    """Compute proportion of tasks that mention tests or validation outputs.

    Returns float in [0.0, 1.0] representing the proportion.
    Throughput mode target: >= 0.30 (30%)
    """
    tasks = plan_obj.get("tasks", [])
    if not isinstance(tasks, list) or len(tasks) == 0:
        return 0.0

    test_patterns = ["test", "pytest", "assert", "validate", "verify", "spec", "check"]
    test_count = 0

    for t in tasks:
        if not isinstance(t, dict):
            continue
        desc = str(t.get("description", "")).lower()
        title = str(t.get("title", "")).lower()
        text = desc + " " + title
        if any(pat in text for pat in test_patterns):
            test_count += 1

    return test_count / len(tasks)


def _analyze_risk_flags(plan_obj: dict[str, Any]) -> dict[str, Any]:
    """Analyze risk flags in the plan.

    Returns dict with:
    - solo_tasks: list of task IDs with solo=True
    - high_intensity_tasks: list of task IDs with intensity="high"
    - risk_count: total count of risky tasks
    - risk_ratio: proportion of risky tasks (target: < 0.2 in throughput mode)
    """
    tasks = plan_obj.get("tasks", [])
    if not isinstance(tasks, list):
        return {
            "solo_tasks": [],
            "high_intensity_tasks": [],
            "risk_count": 0,
            "risk_ratio": 0.0,
        }

    solo_tasks = []
    high_intensity_tasks = []

    for t in tasks:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id", "unknown"))
        if t.get("solo", False):
            solo_tasks.append(tid)
        intensity = str(t.get("estimated_intensity", t.get("intensity", "low"))).lower()
        if intensity == "high":
            high_intensity_tasks.append(tid)

    risk_count = len(set(solo_tasks) | set(high_intensity_tasks))
    risk_ratio = risk_count / len(tasks) if tasks else 0.0

    return {
        "solo_tasks": solo_tasks,
        "high_intensity_tasks": high_intensity_tasks,
        "risk_count": risk_count,
        "risk_ratio": risk_ratio,
    }


def _build_plan_quality_report(
    plan_obj: dict[str, Any],
    max_workers_limit: int,
    planner_profile: str = "balanced",
) -> dict[str, Any]:
    """Build comprehensive plan quality report for throughput mode.

    Returns dict with:
    - metrics: all computed metrics
    - issues: list of actionable issue strings
    - should_reprompt: bool - whether to trigger a reprompt
    - reprompt_message: str - targeted guidance for the reprompt
    - hard_failures: list of issues that are hard failures (must reprompt)
    - soft_warnings: list of issues that are warnings only
    """
    # Only do quality scoring in throughput mode
    if planner_profile != "throughput":
        return {
            "metrics": {},
            "issues": [],
            "should_reprompt": False,
            "reprompt_message": "",
            "hard_failures": [],
            "soft_warnings": [],
        }

    tasks = plan_obj.get("tasks", [])
    task_count = len(tasks) if isinstance(tasks, list) else 0
    root_count = sum(1 for t in (tasks if isinstance(tasks, list) else [])
                     if isinstance(t, dict) and not t.get("depends_on"))
    plan_cap = int(plan_obj.get("max_parallel_tasks", 1) or 1)

    # Compute quality metrics
    lock_pressure = _compute_lock_pressure(plan_obj)
    coverage_intent = _compute_coverage_intent(plan_obj)
    risk_flags = _analyze_risk_flags(plan_obj)

    # Compute average focus score
    focus_scores = []
    for t in (tasks if isinstance(tasks, list) else []):
        if isinstance(t, dict):
            focus_scores.append(_compute_task_focus_score(t))
    avg_focus_score = sum(focus_scores) / len(focus_scores) if focus_scores else 0.0

    metrics = {
        "plan_cap": plan_cap,
        "root_ready": root_count,
        "total_tasks": task_count,
        "max_lock_count": lock_pressure["max_lock_count"],
        "high_pressure_locks": lock_pressure["high_pressure_locks"],
        "lock_pressure_score": lock_pressure["pressure_score"],
        "avg_focus_score": round(avg_focus_score, 2),
        "coverage_intent": round(coverage_intent, 2),
        "risk_ratio": round(risk_flags["risk_ratio"], 2),
        "solo_tasks": risk_flags["solo_tasks"],
        "high_intensity_tasks": risk_flags["high_intensity_tasks"],
    }

    # Categorize issues into hard failures vs soft warnings
    hard_failures: list[str] = []
    soft_warnings: list[str] = []

    # HARD FAILURE: plan_cap < max_workers
    if plan_cap < max_workers_limit:
        hard_failures.append(
            f"max_parallel_tasks={plan_cap} < required {max_workers_limit}"
        )

    # HARD FAILURE: root_ready < max_workers
    if root_count < max_workers_limit:
        hard_failures.append(
            f"root_tasks={root_count} < required {max_workers_limit}"
        )

    # HARD FAILURE: high lock pressure (serializes workers)
    if lock_pressure["max_lock_count"] > 3:
        locks_str = ", ".join(lock_pressure["high_pressure_locks"][:3])
        hard_failures.append(
            f"lock_pressure=high (max {lock_pressure['max_lock_count']} tasks share a lock: {locks_str})"
        )

    # SOFT WARNING: total_tasks < 2*workers (only if root_ready is also borderline)
    min_task_count = max_workers_limit * 2
    if task_count < min_task_count:
        if root_count < max_workers_limit + 2:  # borderline roots
            soft_warnings.append(
                f"total_tasks={task_count} < recommended {min_task_count} (with borderline roots)"
            )
        else:
            soft_warnings.append(
                f"total_tasks={task_count} < recommended {min_task_count} (acceptable: roots={root_count})"
            )

    # SOFT WARNING: low coverage intent
    if coverage_intent < 0.30:
        soft_warnings.append(
            f"coverage_intent={coverage_intent:.0%} < recommended 30% (few tasks mention tests)"
        )

    # SOFT WARNING: low focus scores
    if avg_focus_score < 0.4:
        soft_warnings.append(
            f"avg_focus_score={avg_focus_score:.2f} < recommended 0.40 (tasks may be vague)"
        )

    # SOFT WARNING: high risk ratio
    if risk_flags["risk_ratio"] > 0.2:
        soft_warnings.append(
            f"risk_ratio={risk_flags['risk_ratio']:.0%} > recommended 20% (too many solo/high tasks)"
        )

    # Build reprompt message
    all_issues = hard_failures + soft_warnings
    should_reprompt = len(hard_failures) > 0

    reprompt_message = ""
    if should_reprompt:
        issues_text = "\n".join(f"  - {issue}" for issue in all_issues)
        reprompt_message = f"Issues detected:\n{issues_text}"

        # Add targeted guidance based on failure type
        if any("lock_pressure" in hf for hf in hard_failures):
            reprompt_message += "\n\nYour locks are serializing workers. Use distinct locks per file/module or separate files to enable true parallelism."

    return {
        "metrics": metrics,
        "issues": all_issues,
        "should_reprompt": should_reprompt,
        "reprompt_message": reprompt_message,
        "hard_failures": hard_failures,
        "soft_warnings": soft_warnings,
    }


def _build_throughput_correction_prompt(
    analysis: dict[str, Any],
    max_workers_limit: int,
    *,
    quality_report: dict[str, Any] | None = None,
    attempt_number: int = 1,
) -> str:
    """Build a strict correction prompt for thin plans in throughput mode.

    Args:
        analysis: Result from _analyze_plan_width
        max_workers_limit: Target worker count
        quality_report: Optional result from _build_plan_quality_report (for detailed guidance)
        attempt_number: Current retry attempt (1-based); adds targeted guidance on attempt 2+
    """
    # Combine issues from analysis and quality report
    all_issues = list(analysis.get("issues", []))
    all_warnings = list(analysis.get("warnings", []))

    if quality_report:
        # Add quality-specific issues
        for issue in quality_report.get("hard_failures", []):
            if issue not in all_issues:
                all_issues.append(issue)
        for warning in quality_report.get("soft_warnings", []):
            if warning not in all_warnings:
                all_warnings.append(warning)

    issues_text = "\n".join(f"  - {issue}" for issue in all_issues)
    warnings_text = "\n".join(f"  - {warning}" for warning in all_warnings) if all_warnings else ""

    # Build targeted guidance based on failure type and attempt number
    targeted_guidance = ""

    # Check for lock pressure issues
    has_lock_pressure = quality_report and any(
        "lock_pressure" in hf for hf in quality_report.get("hard_failures", [])
    )

    if has_lock_pressure or attempt_number >= 2:
        # On second attempt or if lock pressure detected, add explicit lock guidance
        targeted_guidance = """
        **LOCK SERIALIZATION WARNING:**
        Your locks are serializing workers! This kills parallelism.

        FIX THIS:
        - Use DISTINCT lock keys per file or module (e.g., "m4-types", "m4-io-reader", "m5-eval")
        - Do NOT put multiple tasks behind the same lock unless they truly conflict
        - If a shared integration file must be touched, create ONE "INTEGRATION" task that runs LAST
        - Example: 8 tasks using lock "init-m4" = serial execution = BAD
                   8 tasks using locks "m4-types", "m4-io", "m4-preprocess", etc. = parallel = GOOD
        """

    base_prompt = textwrap.dedent(f"""
        PLAN REJECTED: Your plan does not meet throughput mode requirements.

        **HARD FAILURES (must fix):**
        {issues_text}
        {f'''
        **WARNINGS (should address):**
        {warnings_text}
        ''' if warnings_text else ''}
        {targeted_guidance}

        **THROUGHPUT MODE REQUIREMENTS (MANDATORY):**
        1. Set max_parallel_tasks = {max_workers_limit}
        2. Create at least {max_workers_limit} root tasks (depends_on = [])
        3. Target {max_workers_limit * 2}+ total tasks (recommended, not strictly required)
        4. Use DISTINCT locks per file/module - avoid lock serialization
        5. Split large tasks into smaller, file-focused tasks (each touching ≤3 files)

        **QUALITY REQUIREMENTS:**
        - Each task description MUST include files/modules touched
        - Each task SHOULD have clear acceptance criteria (what makes it "done")
        - Prefer 10-20 meaningful tasks over 50 micro-edits
        - Avoid tasks that only rename/reformat without functional change

        **If you cannot produce {max_workers_limit}+ root tasks without junk tasks:**
        Produce fewer tasks but ensure root_ready >= {max_workers_limit} by splitting along meaningful seams.

        Please regenerate the plan with these requirements. Output ONLY the corrected JSON.
    """).strip()

    return base_prompt


def _build_task_plan_prompt(
    *,
    design_doc_text: str,
    milestone_id: str,
    max_workers_limit: int,
    cpu_threshold_pct_total: float,
    mem_threshold_pct_total: float,
    machine_info: dict[str, Any],
    planner_profile: str = "balanced",
) -> str:
    design_excerpt = _truncate(design_doc_text, 40000)  # Increased for multi-doc support
    cores = machine_info.get("cpu_cores")
    ram_b = machine_info.get("ram_bytes")
    ram_gb = None
    if isinstance(ram_b, int) and ram_b > 0:
        ram_gb = round(ram_b / (1024**3), 1)

    # Detect all milestones in the document
    all_milestones = _parse_all_milestones(design_doc_text)
    is_multi_milestone = len(all_milestones) > 1

    if is_multi_milestone:
        milestone_instruction = f"""
MULTI-MILESTONE MODE: The design document contains {len(all_milestones)} milestone documents: {", ".join(all_milestones)}.

You MUST:
1. Create tasks for ALL milestones, not just one.
2. Prefix each task ID with its milestone (e.g., M1-DSL-SCHEMA, M2-OPENEMS-SETUP, M3-ARTIFACT-STORE).
3. Include the milestone prefix in lock keys to avoid collisions (e.g., M1-dsl, M2-openems, M3-manifest).
4. Dependencies across milestones are allowed if later milestones explicitly require earlier ones.
5. Set milestone_id to "MULTI" in your output.

For each milestone in the document, identify and create the necessary tasks."""
    else:
        milestone_instruction = f"Use milestone_id exactly: {milestone_id}"

    # Check agent policy for forced mode
    policy = get_agent_policy()
    if policy.forced_agent:
        planner_role = policy.forced_agent.upper()
        agent_assignment_instruction = f"""
        Agent assignment (POLICY OVERRIDE ACTIVE):
        - **--only-{policy.forced_agent}** flag is active.
        - You MUST assign preferred_agent as "{policy.forced_agent}" for ALL tasks.
        - Do NOT assign any tasks to the other agent - they will be overridden anyway.
"""
    else:
        planner_role = "CODEX"
        agent_assignment_instruction = """
        Agent assignment (CRITICAL):
        - You MUST assign preferred_agent as ONLY "codex" or "claude". NEVER use "either".
        - At least 30-40% of tasks MUST be assigned to "claude".
        - Recommended assignment heuristics:
          * Claude is better for: schemas, documentation, code review, edge-case analysis, test writing, refactoring, API design
          * Codex is better for: heavy implementation, low-level code, CLI tools, integration work, file I/O, build systems
        - If unsure, prefer Claude for design/spec tasks and Codex for implementation tasks.
"""

    header = textwrap.dedent(
        f"""
        You are {planner_role} acting as a **task planner** for a parallel agent runner.

        Your job:
        - Read the design document below.
        - Split the work into a set of independent tasks that can be executed in parallel safely.
        - Output ONLY a single JSON object that matches the provided JSON schema.

        Hardware context:
        - CPU cores: {cores}
        - RAM (GB): {ram_gb if ram_gb is not None else "unknown"}

        Parallel safety policy:
        - Tasks that touch the same subsystem or files should share a lock key in `locks`.
        - If a task might require a heavy local command (full test suite, large build, large format/lint, GPU/ML workloads), set `solo: true` and `intensity: "high"`.
        - The runner will automatically stop any single task if it uses > {cpu_threshold_pct_total:.1f}% CPU or > {mem_threshold_pct_total:.1f}% RAM for multiple samples.

        Concurrency:
        - Choose `max_parallel_tasks` <= {max_workers_limit}.
        - Prefer fewer parallel tasks if you believe tasks are likely to conflict or require heavy commands.

        Shared integration files policy (CRITICAL for avoiding merge conflicts):
        - These files are HIGH CONFLICT RISK and should be edited by AT MOST ONE task:
          * src/formula_foundry/*/\\_\\_init\\_\\_.py (package exports)
          * pyproject.toml (dependencies, config)
        - If multiple tasks need to add exports to __init__.py:
          * DO NOT have each task edit __init__.py directly
          * Instead, have ONE dedicated task (e.g., "API-EXPORTS" or "INTEGRATE-EXPORTS") that:
            - Runs LAST (depends_on all implementation tasks)
            - Collects all new exports and updates __init__.py once
          * Implementation tasks should focus on creating modules, not updating package exports
        - Use lock keys to prevent conflicts: e.g., locks: ["init-m4"], locks: ["pyproject"]
{agent_assignment_instruction}
        {milestone_instruction}
        """
    ).strip()

    # Add THROUGHPUT MODE block if enabled
    throughput_block = ""
    if planner_profile == "throughput":
        throughput_block = textwrap.dedent(f"""
            ---

            # THROUGHPUT MODE (ENABLED) - HIGH-OUTPUT, HIGH-QUALITY

            You are in **throughput mode** - maximize parallel worker utilization while maintaining quality.
            This is NOT "microtask spam" mode. Each task must produce meaningful progress.

            **MANDATORY REQUIREMENTS:**

            1. **Set max_parallel_tasks = {max_workers_limit}** (the allowed limit).

            2. **Create at least {max_workers_limit} root tasks** (tasks with `depends_on: []`) that can start immediately.
               - Workers idle when there are no ready tasks. More roots = faster start.

            3. **Target {max_workers_limit * 2}+ total tasks** to keep the worker pipeline busy.
               - This is a soft target. If you cannot produce this many meaningful tasks, produce fewer.

            4. **Use DISTINCT locks per file/module:**
               - BAD: 8 tasks all using `locks: ["init-m4"]` (serial execution = no parallelism)
               - GOOD: Tasks using `locks: ["m4-types"]`, `locks: ["m4-io"]`, `locks: ["m4-preprocess"]` (true parallelism)
               - Do NOT share a lock across more than 2-3 tasks unless truly necessary.

            5. **Prefer 10-20 meaningful tasks over 50 micro-edits:**
               - Each task should produce a testable unit of work (feature, function, module, test file).
               - Avoid tasks that only rename/reformat without functional change.
               - Avoid tasks that are too vague ("implement subsystem") or too tiny ("add import line").

            **TASK QUALITY REQUIREMENTS:**

            Each task description MUST include:

            6. **Files/modules touched** (or lock key):
               - Example: "Files: src/formula_foundry/m4/types.py, tests/test_m4_types.py"
               - This helps validate lock correctness.

            7. **Acceptance criteria** (what makes the task "done"):
               - Example: "Done when: NetworkConfig dataclass exists with fields for host, port, timeout; unit tests pass"
               - Must be concrete and verifiable.

            8. **Why this task is valuable** (1 sentence):
               - Example: "Provides typed configuration for network connections, enabling type-safe API calls"

            **INTEGRATION TASK PATTERN:**

            If a shared integration file (e.g., `__init__.py`) must be touched by multiple features:
            - Create ONE "INTEGRATION" or "API-EXPORTS" task that runs LAST (`depends_on` all impl tasks)
            - Implementation tasks should create modules WITHOUT updating package exports
            - The integration task collects all exports and updates `__init__.py` once

            **ANTI-PATTERNS TO AVOID:**

            - Splitting into tasks that only rename/reformat unless paired with a functional change
            - Creating 50 tiny tasks just to hit a task count target
            - Putting 8+ tasks behind the same lock (kills parallelism)
            - Vague descriptions like "implement subsystem" or "do M4 work"

            **EXAMPLE of a good throughput plan for {max_workers_limit} workers:**
            - {max_workers_limit}+ root tasks (all can start immediately)
            - {max_workers_limit * 2}+ total tasks (with meaningful scope each)
            - max_parallel_tasks = {max_workers_limit}
            - Each task names specific files, has clear acceptance criteria
            - Uses DISTINCT lock keys: "m4-types", "m4-io-reader", "m4-io-writer", "m5-eval", etc.

            **If you cannot produce {max_workers_limit}+ root tasks without junk tasks:**
            Produce fewer tasks but ensure root_ready >= {max_workers_limit} by splitting along meaningful seams (modules, test files, features).
        """).strip()

    parts = [header]
    if throughput_block:
        parts.append(throughput_block)
    parts.append("---\n\n# Design Document\n" + design_excerpt)

    return "\n\n".join(parts)


def _build_parallel_task_prompt(
    *,
    system_prompt: str,
    task: ParallelTask,
    worker_id: int,
    milestone_id: str,
    repo_snapshot: str,
    design_doc_text: str,
    resource_policy: dict[str, Any],
) -> str:
    design_excerpt = _truncate(design_doc_text, 18000)

    state_blob = {
        "runner_mode": "parallel-worker",
        "worker_id": worker_id,
        "milestone_id": milestone_id,
        "task": {
            "id": task.id,
            "title": task.title,
            "intensity": task.intensity,
            "solo": task.solo,
            "locks": task.locks,
            "depends_on": task.depends_on,
        },
        "resource_policy": resource_policy,
    }

    instructions = textwrap.dedent(
        f"""
        # Parallel Worker Instructions

        You are worker {worker_id} executing task {task.id}: {task.title}

        Do:
        - Implement ONLY this task.
        - Keep changes focused.
        - Make the repo consistent and runnable.
        - DO NOT run git add or git commit - the orchestrator handles commits.

        Resource safety:
        - Avoid running heavy commands while other agents may be running.
        - If the next step requires a heavy command, DO NOT run it.
          Instead, explain in your summary and provide the exact shell command the user can run later.

        Output:
        - Output ONLY a single JSON object matching the schema.
        - In parallel-worker mode you may set next_agent to yourself and next_prompt to an empty string.
        """
    ).strip()

    # Get agent policy header if in forced mode
    policy_header = get_agent_policy().get_prompt_header()

    parts = [system_prompt.strip()]

    # Inject policy header right after system prompt if in forced mode
    if policy_header:
        parts.append(policy_header.strip())

    parts.extend(
        [
            "---\n\n# Orchestrator State\n" + json.dumps(state_blob, indent=2),
            "---\n\n# Task\n" + task.description.strip(),
            "---\n\n# Repo Snapshot\n" + repo_snapshot.strip(),
            "---\n\n# Design Doc (truncated)\n" + design_excerpt.strip(),
            "---\n\n" + instructions,
        ]
    )

    return "\n\n".join(parts)


def _select_only_tasks(all_tasks: list[ParallelTask], only_ids: list[str]) -> list[ParallelTask]:
    if not only_ids:
        return all_tasks

    ids = {s.strip() for s in only_ids if s.strip()}
    by_id = {t.id: t for t in all_tasks}

    # Include dependencies recursively.
    keep = set()

    def dfs(tid: str) -> None:
        if tid in keep:
            return
        keep.add(tid)
        t = by_id.get(tid)
        if not t:
            return
        for dep in t.depends_on:
            dfs(dep)

    for tid in list(ids):
        dfs(tid)

    return [t for t in all_tasks if t.id in keep]


def _is_backfill_task_id(task_id: str) -> bool:
    """Check if a task ID represents an optional backfill task (FILLER-* prefix)."""
    return task_id.startswith("FILLER-")


def _write_manual_task_file(
    *,
    manual_dir: Path,
    task: ParallelTask,
    reason: str,
    agent_cmd: list[str],
    schema_path: Path,
    prompt_path: Path,
    out_path: Path,
    raw_log_path: Path,
) -> Path:
    _ensure_dir(manual_dir)
    path = manual_dir / f"manual_{_sanitize_branch_fragment(task.id)}.md"

    cmd_str = " ".join(agent_cmd)
    content = f"""# Manual Run Required: {task.id} — {task.title}

This task was stopped or needs manual intervention.

Reason:
- {reason}

Worker:
- {task.worker_id}

Artifacts:
- Prompt: {prompt_path}
- Schema: {schema_path}
- Intended output JSON: {out_path}
- Raw log: {raw_log_path}

Suggested manual rerun (run when no other agents are running):

```bash
cd {task.worktree_path}
export WRITE_ACCESS=1
export ORCH_WRITE_ACCESS=1
{cmd_str}
```

If you prefer rerunning through the orchestrator (single-worker):

```bash
./run_parallel.sh --only-task {task.id} --max-workers 1 --allow-resource-intensive
```
"""

    path.write_text(content, encoding="utf-8")
    return path


def _mark_task_manual(
    *,
    task: ParallelTask,
    reason: str,
    manual_dir: Path,
    schema_path: Path,
) -> None:
    """Mark a task as manual and ensure manual_path is written.

    Sets task.status to 'manual', task.error to the reason, and if task.manual_path
    is None, writes a manual file using _write_manual_task_file.
    """
    task.status = "manual"
    task.error = reason
    if task.manual_path is None:
        task.manual_path = _write_manual_task_file(
            manual_dir=manual_dir,
            task=task,
            reason=reason,
            agent_cmd=[],
            schema_path=schema_path,
            prompt_path=task.prompt_path or Path(""),
            out_path=task.out_path or Path(""),
            raw_log_path=task.raw_log_path or Path(""),
        )


def _generate_run_summary(
    *,
    tasks: list[ParallelTask],
    runs_dir: Path,
    verify_exit_code: int,
) -> dict[str, Any]:
    """Generate a structured summary of the parallel run.

    Backfill tasks (FILLER-* prefix) are optional and do NOT cause run failure.
    They are tracked in 'optional_tasks' for visibility but excluded from
    root_failures, blocked_tasks, and success calculation.
    """
    root_failures = []
    blocked_tasks = []
    completed_tasks = []
    pending_rerun_tasks = []
    optional_tasks = []

    # Root failure statuses: tasks that failed on their own (not due to dependencies)
    root_failure_statuses = ("failed", "manual", "pending_rerun", "resource_killed")

    for t in tasks:
        task_info = {
            "id": t.id,
            "title": t.title,
            "status": t.status,
            "agent": t.agent,
            "work_completed": t.work_completed,
            "commit_sha": t.commit_sha,
            "error": t.error,
            "manual_path": str(t.manual_path) if t.manual_path else None,
            "raw_log_path": str(t.raw_log_path) if t.raw_log_path else None,
            "turn_summary": t.turn_summary,
        }

        is_backfill = _is_backfill_task_id(t.id)

        # Backfill tasks are optional - track separately for visibility
        if is_backfill:
            optional_tasks.append(task_info)
            # Still count done backfills as completed for stats
            if t.status == "done":
                completed_tasks.append(task_info)
            continue

        # Non-backfill task processing
        if t.status == "done":
            completed_tasks.append(task_info)
        elif t.status == "pending_rerun":
            pending_rerun_tasks.append(task_info)
            root_failures.append(task_info)
        elif t.status in root_failure_statuses:
            root_failures.append(task_info)
        elif t.status in ("blocked", "skipped"):
            # Both "blocked" (new) and "skipped" (legacy) are transitively blocked
            blocked_tasks.append(task_info)

    # Count non-backfill tasks only for failure stats
    non_backfill_tasks = [t for t in tasks if not _is_backfill_task_id(t.id)]
    failed_count = len([t for t in non_backfill_tasks if t.status in ("failed", "manual", "resource_killed")])
    pending_rerun_count = len([t for t in non_backfill_tasks if t.status == "pending_rerun"])
    blocked_count = len([t for t in non_backfill_tasks if t.status in ("blocked", "skipped")])

    summary = {
        "run_dir": str(runs_dir),
        "total_tasks": len(tasks),
        "completed": len(completed_tasks),
        "failed": failed_count,
        "pending_rerun": pending_rerun_count,
        "blocked": blocked_count,
        "verify_exit_code": verify_exit_code,
        "success": verify_exit_code == 0 and len(root_failures) == 0,
        "completed_tasks": completed_tasks,
        "root_failures": root_failures,
        "blocked_tasks": blocked_tasks,
        "optional_tasks": optional_tasks,
    }

    return summary


def _generate_continuation_prompt(
    *,
    summary: dict[str, Any],
    tasks: list[ParallelTask],
    design_doc_text: str,
    runs_dir: Path,
) -> str:
    """Generate a continuation prompt for the next run based on failures."""
    lines = [
        "# Continuation Run - Fix Failures and Complete Remaining Work",
        "",
        "The previous parallel run did not complete all tasks successfully.",
        "",
        "## Previous Run Summary",
        f"- Completed: {summary['completed']}/{summary['total_tasks']}",
        f"- Failed: {summary['failed']}",
        f"- Pending Rerun (planning-only): {summary['pending_rerun']}",
        f"- Blocked: {summary['blocked']}",
        f"- Verify exit code: {summary['verify_exit_code']}",
        "",
    ]

    if summary["root_failures"]:
        lines.append("## Root Failures (must be fixed)")
        lines.append("")
        for t in summary["root_failures"]:
            lines.append(f"### {t['id']}: {t['title']}")
            lines.append(f"- Status: {t['status']}")
            lines.append(f"- Agent: {t['agent']}")
            if t["error"]:
                lines.append(f"- Error: {t['error']}")
            if t["turn_summary"]:
                lines.append(f"- Last summary: {t['turn_summary'][:300]}")
            if t["manual_path"]:
                lines.append(f"- Manual file: {t['manual_path']}")
            if t["raw_log_path"]:
                # Include tail of log
                try:
                    log_path = Path(t["raw_log_path"])
                    if log_path.exists():
                        log_text = log_path.read_text(encoding="utf-8", errors="replace")
                        tail = log_text[-2000:] if len(log_text) > 2000 else log_text
                        lines.append(f"- Log tail:\n```\n{tail}\n```")
                except Exception:
                    pass
            lines.append("")

    if summary["blocked_tasks"]:
        lines.append("## Blocked Tasks (will unblock once failures are fixed)")
        lines.append("")
        for t in summary["blocked_tasks"]:
            lines.append(f"- {t['id']}: {t['error'] or 'blocked by prerequisites'}")
        lines.append("")

    if summary["completed_tasks"]:
        lines.append("## Completed Tasks (DO NOT re-implement)")
        lines.append("")
        for t in summary["completed_tasks"]:
            lines.append(f"- {t['id']}: {t['title']}")
        lines.append("")

    lines.extend(
        [
            "## Instructions for This Run",
            "",
            "1. **Focus ONLY on root failures** - do not re-implement completed tasks",
            "2. For tasks marked 'pending_rerun', the agent only produced a plan - now IMPLEMENT the actual code",
            "3. For failed tasks, analyze the error and fix the issue",
            "4. After fixing root failures, blocked tasks will automatically become runnable",
            "5. Produce a new task plan that includes ONLY:",
            "   - Tasks from root_failures that need to be fixed/implemented",
            "   - Any blocked tasks that will become runnable",
            "",
            "IMPORTANT: Do NOT include completed tasks in the new plan.",
            "",
        ]
    )

    return "\n".join(lines)


def _compute_transitive_blocked(
    tasks: list[ParallelTask],
    root_failure_ids: set,
) -> set:
    """Compute the transitive closure of all tasks blocked by root failures.

    A task is blocked if:
    - It depends directly on a root failure
    - It depends on another blocked task (transitive)

    Returns the set of task IDs that are transitively blocked.
    """
    {t.id: t for t in tasks}
    blocked_ids: set = set()

    # Build reverse dependency graph
    dependents: dict[str, list[str]] = {t.id: [] for t in tasks}
    for t in tasks:
        for dep in t.depends_on:
            if dep in dependents:
                dependents[dep].append(t.id)

    # BFS from root failures to find all transitively blocked tasks
    queue_ids = list(root_failure_ids)
    visited = set(root_failure_ids)

    while queue_ids:
        current_id = queue_ids.pop(0)
        # All tasks that depend on this one are blocked
        for dependent_id in dependents.get(current_id, []):
            if dependent_id not in visited:
                visited.add(dependent_id)
                blocked_ids.add(dependent_id)
                queue_ids.append(dependent_id)

    return blocked_ids


def _build_implement_now_prompt(
    *,
    task: ParallelTask,
    worker_id: int,
    milestone_id: str,
    repo_snapshot: str,
    previous_summary: str,
) -> str:
    """Build a focused "IMPLEMENT NOW" prompt for plan-only retry.

    This is intentionally smaller than the full design doc to avoid context bloat.
    """
    return textwrap.dedent(f"""
        # CRITICAL: IMPLEMENT NOW - DO NOT PLAN

        You are worker {worker_id} retrying task {task.id}: {task.title}

        YOUR PREVIOUS ATTEMPT ONLY PRODUCED A PLAN. THIS IS NOT ACCEPTABLE.

        ## Task Description
        {task.description}

        ## Your Previous Summary
        {previous_summary}

        ## Current Repo State
        {repo_snapshot}

        ## REQUIREMENTS
        1. You MUST write actual code and commit changes
        2. You MUST set work_completed=true in your response
        3. DO NOT just describe what you would do - ACTUALLY DO IT
        4. If you cannot complete the task, explain WHY in your error field

        ## Output Schema Reminder
        Your response MUST be a JSON object with:
        - agent: "{task.agent}"
        - milestone_id: "{milestone_id}"
        - phase: "implement"
        - work_completed: true (REQUIRED - you must complete work!)
        - project_complete: false
        - summary: "what you actually implemented"
        - gates_passed: []
        - requirement_progress: {{"covered_req_ids": [], "tests_added_or_modified": [], "commands_run": []}}
        - next_agent: "{task.agent}"
        - next_prompt: ""
        - delegate_rationale: "completed task"
        - stats_refs: ["CL-1"] or ["CX-1"]
        - needs_write_access: true
        - artifacts: [list of files created/modified]

        NOW IMPLEMENT THE TASK. NO MORE PLANNING.
    """).strip()


def _run_parallel_task(
    *,
    task: ParallelTask,
    worker_id: int,
    state: RunState,
    config: RunConfig,
    milestone_id: str,
    design_doc_text: str,
    system_prompt: str,
    stats_id_set: set,
    tasks_dir: Path,
    worktrees_dir: Path,
    manual_dir: Path,
    git_lock: threading.Lock,
    cpu_threshold_pct_total: float,
    mem_threshold_pct_total: float,
    terminal_max_bytes: int,
    terminal_max_line_length: int,
    allow_resource_intensive: bool,
) -> None:
    task.worker_id = worker_id
    task.task_dir = tasks_dir / _sanitize_branch_fragment(task.id)
    _ensure_dir(task.task_dir)

    # Create a fresh worktree/branch for this task.
    with git_lock:
        rc, base_sha_out, err = _run_cmd(["git", "rev-parse", "HEAD"], cwd=state.project_root, env=os.environ.copy())
        if rc != 0:
            task.status = "failed"
            task.error = (base_sha_out + "\n" + err).strip()
            return
        base_sha = base_sha_out.strip()
        task.base_sha = base_sha

        branch = f"task/{state.run_id}/{_sanitize_branch_fragment(task.id)}"
        task.branch = branch

        worktree_path = worktrees_dir / f"w{worker_id:02d}_{_sanitize_branch_fragment(task.id)}"
        task.worktree_path = worktree_path
        if worktree_path.exists():
            # Keep it safe: don't delete unknown paths. Use a unique suffix instead.
            worktree_path = worktrees_dir / f"w{worker_id:02d}_{_sanitize_branch_fragment(task.id)}_{int(time.time())}"
            task.worktree_path = worktree_path

        rc2, out2, err2 = _run_cmd(
            ["git", "worktree", "add", "-b", branch, str(worktree_path), base_sha],
            cwd=state.project_root,
            env=os.environ.copy(),
        )
        if rc2 != 0:
            task.status = "failed"
            task.error = (out2 + "\n" + err2).strip()
            return

    # Agent execution with retry loop for plan-only responses
    agent = task.agent if task.agent in AGENTS else "codex"
    script_rel = config.agent_scripts.get(agent, "")
    script_path = (task.worktree_path / script_rel) if script_rel else None
    if not script_path or not script_path.exists():
        task.status = "failed"
        task.error = f"Agent script not found: {script_path}"
        return

    schema_path = state.schema_path
    env = os.environ.copy()
    env["WRITE_ACCESS"] = "1"
    env["ORCH_WRITE_ACCESS"] = "1"
    env["FF_WORKER_ID"] = str(worker_id)
    # Signal to the agent wrapper that this is a turn schema (enables turn normalization)
    env["ORCH_SCHEMA_KIND"] = "turn"

    # Disable GPU by default in parallel mode unless explicitly allowed.
    if config.parallel.disable_gpu_by_default and env.get("FF_ALLOW_GPU") != "1":
        env.setdefault("CUDA_VISIBLE_DEVICES", "")

    prefix = f"[w{worker_id:02d} {agent} {task.id}]"

    # Retry loop: try up to max_retries+1 times (initial + retries)
    while True:
        attempt = task.retry_count + 1
        attempt_suffix = f"_attempt{attempt}" if task.retry_count > 0 else ""

        # Build prompt (use "IMPLEMENT NOW" prompt for retries)
        repo_snapshot = _git_snapshot(task.worktree_path)

        if task.retry_count == 0:
            # First attempt: use full design doc prompt
            resource_policy = {
                "cpu_threshold_pct_total": cpu_threshold_pct_total,
                "mem_threshold_pct_total": mem_threshold_pct_total,
                "resource_intensive_definition": "> 40% CPU or RAM",
                "allow_resource_intensive": bool(allow_resource_intensive),
            }
            prompt_text = _build_parallel_task_prompt(
                system_prompt=system_prompt,
                task=task,
                worker_id=worker_id,
                milestone_id=milestone_id,
                repo_snapshot=repo_snapshot,
                design_doc_text=design_doc_text,
                resource_policy=resource_policy,
            )
        else:
            # Retry: use focused "IMPLEMENT NOW" prompt
            print(f"[orchestrator] {task.id}: RETRY {task.retry_count}/{task.max_retries} with IMPLEMENT NOW prompt")
            prompt_text = _build_implement_now_prompt(
                task=task,
                worker_id=worker_id,
                milestone_id=milestone_id,
                repo_snapshot=repo_snapshot,
                previous_summary=task.turn_summary or "(no summary)",
            )

        prompt_path = task.task_dir / f"prompt{attempt_suffix}.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        task.prompt_path = prompt_path

        out_path = task.task_dir / f"turn{attempt_suffix}.json"
        raw_log_path = task.task_dir / f"raw{attempt_suffix}.log"
        task.out_path = out_path
        task.raw_log_path = raw_log_path

        cmd = [str(script_path), str(prompt_path), str(schema_path), str(out_path)]

        res = _run_cmd_monitored(
            cmd,
            cwd=task.worktree_path,
            env=env,
            prefix=prefix,
            raw_log_path=raw_log_path,
            stream_to_terminal=True,
            terminal_max_bytes=terminal_max_bytes,
            terminal_max_line_length=terminal_max_line_length,
            cpu_threshold_pct_total=cpu_threshold_pct_total,
            mem_threshold_pct_total=mem_threshold_pct_total,
            sample_interval_s=config.parallel.sample_interval_s,
            consecutive_samples=config.parallel.consecutive_samples,
            kill_grace_s=config.parallel.kill_grace_s,
            allow_resource_intensive=allow_resource_intensive,
        )

        task.max_cpu_pct_total = max(task.max_cpu_pct_total, res.max_cpu_pct_total)
        task.max_mem_pct_total = max(task.max_mem_pct_total, res.max_mem_pct_total)

        if res.killed_for_resources:
            task.status = "resource_killed"
            task.error = f"Stopped for resources: {res.kill_reason}"
            task.manual_path = _write_manual_task_file(
                manual_dir=manual_dir,
                task=task,
                reason=task.error,
                agent_cmd=cmd,
                schema_path=schema_path,
                prompt_path=prompt_path,
                out_path=out_path,
                raw_log_path=raw_log_path,
            )
            return

        if res.returncode != 0:
            task.status = "failed"
            task.error = f"Agent exit code {res.returncode}"
            task.manual_path = _write_manual_task_file(
                manual_dir=manual_dir,
                task=task,
                reason=task.error,
                agent_cmd=cmd,
                schema_path=schema_path,
                prompt_path=prompt_path,
                out_path=out_path,
                raw_log_path=raw_log_path,
            )
            return

        # Validate JSON output with auto-recovery for bad output
        # Track correction attempts for this specific agent call
        if not hasattr(task, '_json_correction_count'):
            task._json_correction_count = 0
        if not hasattr(task, '_agent_fallback_used'):
            task._agent_fallback_used = False

        validation_error: str | None = None
        turn_obj: dict | None = None
        normalization_warnings: list[str] = []

        if not out_path.exists():
            validation_error = "Agent did not produce output file"
        else:
            # Validate file is not 0-bytes (critical robustness check)
            is_valid, turn_data, file_error = validate_json_file(out_path)
            if not is_valid:
                if "empty (0 bytes)" in (file_error or ""):
                    print(f"[orchestrator] {task.id}: CRITICAL: 0-byte turn.json detected, triggering retry")
                    validation_error = "0-byte output file - agent output was not written correctly"
                else:
                    validation_error = f"Invalid turn.json: {file_error}"
            else:
                turn_text = out_path.read_text(encoding="utf-8")

                # Use TurnNormalizer for robust payload extraction and invariant override
                norm_result = normalize_agent_output(
                    turn_text,
                    expected_agent=agent,
                    expected_milestone_id=milestone_id,
                    stats_id_set=stats_id_set,
                    default_phase="implement",
                )

                if norm_result.success and norm_result.turn:
                    turn_obj = norm_result.turn
                    normalization_warnings = norm_result.warnings
                    # Log normalization warnings (auto-corrections)
                    for warning in normalization_warnings:
                        print(f"[orchestrator] {task.id}: NORMALIZED: {warning}")

                    # Use lenient validation (auto-corrects mismatches with warnings)
                    ok, msg, val_warnings = _validate_turn_lenient(
                        turn_obj,
                        expected_agent=agent,
                        expected_milestone_id=milestone_id,
                        stats_id_set=stats_id_set,
                    )
                    for warning in val_warnings:
                        print(f"[orchestrator] {task.id}: VALIDATION: {warning}")
                    if not ok:
                        validation_error = f"Invalid JSON output after normalization: {msg}"
                    else:
                        # Write normalized turn atomically to ensure file integrity
                        try:
                            atomic_write_json(out_path, turn_obj, indent=2)
                            print(f"[orchestrator] {task.id}: turn.json written atomically")
                        except Exception as e:
                            print(f"[orchestrator] {task.id}: WARNING: atomic write failed: {e}")
                else:
                    # Normalization failed - fall back to direct parsing
                    turn_obj = _try_parse_json(turn_text)
                    if turn_obj is None:
                        validation_error = f"Cannot extract JSON payload: {norm_result.error or 'unknown error'}"
                    else:
                        # Got JSON but didn't pass normalizer - try lenient validation directly
                        ok, msg, val_warnings = _validate_turn_lenient(
                            turn_obj,
                            expected_agent=agent,
                            expected_milestone_id=milestone_id,
                            stats_id_set=stats_id_set,
                        )
                        for warning in val_warnings:
                            print(f"[orchestrator] {task.id}: VALIDATION: {warning}")
                        if not ok:
                            validation_error = f"Invalid JSON output: {msg}"

        # Auto-recovery: reprompt on validation error
        if validation_error:
            max_json_corrections = config.max_json_correction_attempts
            task._json_correction_count += 1

            # Check for noncompliant output patterns (tools disabled, markdown, etc.)
            raw_text = ""
            noncompliant_violations: list[str] = []
            has_critical_tool_violation = False
            if out_path.exists():
                try:
                    raw_text = out_path.read_text(encoding="utf-8")
                    _, noncompliant_violations, has_critical_tool_violation = _is_noncompliant_and_should_use_stricter_prompt(raw_text)
                except Exception:
                    pass

            if noncompliant_violations:
                print(f"[orchestrator] {task.id}: Noncompliant output detected: {noncompliant_violations}")
                if has_critical_tool_violation:
                    print(f"[orchestrator] {task.id}: CRITICAL: Agent incorrectly claimed tools are disabled - will explicitly correct")

            if task._json_correction_count <= max_json_corrections:
                # Reprompt with strict correction prompt
                print(f"[orchestrator] {task.id}: JSON validation failed ({validation_error}), "
                      f"auto-reprompt {task._json_correction_count}/{max_json_corrections}")

                # Build strict correction prompt using contract hardening
                correction_prompt = _build_strict_correction_prompt(
                    agent=agent,
                    milestone_id=milestone_id,
                    validation_error=validation_error,
                    task_title=task.title,
                    task_description=task.description,
                    noncompliant_violations=noncompliant_violations if noncompliant_violations else None,
                    attempt_number=task._json_correction_count,
                    has_critical_tool_violation=has_critical_tool_violation,
                )

                # Save correction prompt and retry
                correction_path = task.task_dir / f"correction_{task._json_correction_count}.txt"
                correction_path.write_text(correction_prompt, encoding="utf-8")
                task.prompt_path = correction_path

                correction_out_path = task.task_dir / f"turn_correction_{task._json_correction_count}.json"
                correction_raw_log = task.task_dir / f"raw_correction_{task._json_correction_count}.log"

                cmd = [str(script_path), str(correction_path), str(schema_path), str(correction_out_path)]
                res = _run_cmd_monitored(
                    cmd,
                    cwd=task.worktree_path,
                    env=env,
                    prefix=f"{prefix} correction",
                    raw_log_path=correction_raw_log,
                    stream_to_terminal=True,
                    terminal_max_bytes=terminal_max_bytes,
                    terminal_max_line_length=terminal_max_line_length,
                    cpu_threshold_pct_total=cpu_threshold_pct_total,
                    mem_threshold_pct_total=mem_threshold_pct_total,
                    sample_interval_s=config.parallel.sample_interval_s,
                    consecutive_samples=config.parallel.consecutive_samples,
                    kill_grace_s=config.parallel.kill_grace_s,
                    allow_resource_intensive=allow_resource_intensive,
                )

                if res.returncode == 0 and correction_out_path.exists():
                    # Re-validate the correction output using TurnNormalizer
                    corr_text = correction_out_path.read_text(encoding="utf-8")

                    # Check for noncompliant patterns in correction output
                    is_still_noncompliant, new_violations, new_has_critical = _is_noncompliant_and_should_use_stricter_prompt(corr_text)
                    if is_still_noncompliant:
                        print(f"[orchestrator] {task.id}: Correction output still noncompliant: {new_violations}")
                        # Force another correction attempt with stricter prompt
                        noncompliant_violations = new_violations
                        has_critical_tool_violation = new_has_critical
                    else:
                        # Use TurnNormalizer for correction output
                        corr_norm = normalize_agent_output(
                            corr_text,
                            expected_agent=agent,
                            expected_milestone_id=milestone_id,
                            stats_id_set=stats_id_set,
                            default_phase="implement",
                        )
                        if corr_norm.success and corr_norm.turn:
                            corr_obj = corr_norm.turn
                            for w in corr_norm.warnings:
                                print(f"[orchestrator] {task.id}: CORRECTION NORMALIZED: {w}")
                            # Use lenient validation
                            ok2, msg2, val_w = _validate_turn_lenient(
                                corr_obj,
                                expected_agent=agent,
                                expected_milestone_id=milestone_id,
                                stats_id_set=stats_id_set,
                            )
                            for w in val_w:
                                print(f"[orchestrator] {task.id}: CORRECTION VALIDATION: {w}")
                            if ok2:
                                # Correction succeeded!
                                print(f"[orchestrator] {task.id}: JSON correction succeeded on attempt {task._json_correction_count}")
                                turn_obj = corr_obj
                                out_path = correction_out_path
                                validation_error = None

                # If still invalid, try fallback agent
                if validation_error and not task._agent_fallback_used:
                    other_agent = "claude" if agent == "codex" else "codex"
                    # Check if other agent is enabled
                    enabled = config.enable_agents or []
                    if other_agent in enabled:
                        print(f"[orchestrator] {task.id}: Falling back to {other_agent} after {agent} failed")
                        task._agent_fallback_used = True
                        task._json_correction_count = 0

                        # Update agent and script
                        agent = other_agent
                        script_rel = config.agent_scripts.get(agent, "")
                        script_path = (task.worktree_path / script_rel) if script_rel else None
                        if script_path and script_path.exists():
                            prefix = f"[w{worker_id:02d} {agent} {task.id}]"
                            # Retry with new agent (continue loop)
                            continue

        # If validation still fails after all attempts, mark as failed
        if validation_error:
            task.status = "failed"
            task.error = f"JSON validation failed after recovery attempts: {validation_error}"
            task.manual_path = _write_manual_task_file(
                manual_dir=manual_dir,
                task=task,
                reason=task.error,
                agent_cmd=cmd,
                schema_path=schema_path,
                prompt_path=prompt_path,
                out_path=out_path,
                raw_log_path=raw_log_path,
            )
            return

        # Track turn output metadata
        task.work_completed = bool(turn_obj.get("work_completed", False))
        task.turn_summary = str(turn_obj.get("summary", ""))[:500]

        # Collect changes as patch artifact (no git commit required from worker)
        # This eliminates the ".git/worktrees/*/index.lock permission denied" class of failures
        rc, porcelain, err = _run_cmd(["git", "status", "--porcelain=v1"], cwd=task.worktree_path, env=os.environ.copy())
        has_changes = rc == 0 and bool(porcelain.strip())
        if has_changes:
            # Collect patch artifact from worktree
            patch_artifact = collect_patch_artifact(
                worktree_path=task.worktree_path,
                task_id=task.id,
                base_sha=task.base_sha,
            )
            if patch_artifact.success:
                # Save patch artifacts atomically
                patch_path, manifest_path = save_patch_artifact(patch_artifact, task.task_dir)
                task.patch_path = patch_path
                task.has_patch = True
                print(f"[orchestrator] {task.id}: collected patch with {len(patch_artifact.changes)} changed files")
            else:
                print(f"[orchestrator] WARNING: {task.id}: patch collection failed: {patch_artifact.error}")
                # Fall back to legacy worktree-based commit attempt (may fail in sandbox)
                _run_cmd(["git", "add", "-A"], cwd=task.worktree_path, env=os.environ.copy())
                rc_commit, out_commit, err_commit = _run_cmd(
                    ["git", "commit", "-m", f"task({task.id}): auto commit"],
                    cwd=task.worktree_path,
                    env=os.environ.copy(),
                )
                if rc_commit == 0:
                    rc_sha, sha_out, _ = _run_cmd(["git", "rev-parse", "HEAD"], cwd=task.worktree_path, env=os.environ.copy())
                    if rc_sha == 0:
                        task.commit_sha = sha_out.strip()
                # If commit fails, that's OK - we'll rely on the patch artifact

        # CRITICAL: Task is only "done" if work_completed==true
        if task.work_completed:
            task.status = "done"
            task.error = None
            return

        # Check if there were actual changes (patch or commit) despite work_completed=false
        if getattr(task, 'has_patch', False) or (task.commit_sha and task.commit_sha != task.base_sha):
            # Agent made changes but claimed work_completed=false - trust the changes, mark done
            print(f"[orchestrator] NOTE: {task.id} has work_completed=false but made changes; marking done")
            task.status = "done"
            task.error = None
            return

        # Planning-only turn with no implementation - check if we can retry
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            print(f"[orchestrator] {task.id}: work_completed=false, scheduling retry {task.retry_count}/{task.max_retries}")
            # Continue to next iteration of the retry loop
            continue
        else:
            # All retries exhausted - mark for manual rerun
            task.status = "pending_rerun"
            task.error = f"Agent returned work_completed=false after {task.retry_count + 1} attempts; needs manual implementation"
            task.manual_path = _write_manual_task_file(
                manual_dir=manual_dir,
                task=task,
                reason=f"{task.error}. Summary: {task.turn_summary}",
                agent_cmd=cmd,
                schema_path=schema_path,
                prompt_path=prompt_path,
                out_path=out_path,
                raw_log_path=raw_log_path,
            )
            return


def _run_selftest_task(
    *,
    task: ParallelTask,
    worker_id: int,
    tasks_dir: Path,
) -> None:
    """Execute a selftest task (no real agent, just a trivial command)."""
    task.worker_id = worker_id
    task.task_dir = tasks_dir / _sanitize_branch_fragment(task.id)
    _ensure_dir(task.task_dir)

    # Write a synthetic prompt
    prompt_path = task.task_dir / "prompt.txt"
    prompt_text = f"[SELFTEST] Task: {task.id}\nTitle: {task.title}\nDescription: {task.description}\n"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    task.prompt_path = prompt_path

    # Raw log
    raw_log_path = task.task_dir / "raw.log"
    task.raw_log_path = raw_log_path

    prefix = f"[w{worker_id:02d} SELFTEST-{task.id}]"

    # Execute a trivial command
    cmd = [
        sys.executable,
        "-c",
        f"import time; print('selftest {task.id} started'); time.sleep(0.5); print('selftest {task.id} done')",
    ]

    try:
        with raw_log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout or []:
                log_f.write(line)
                log_f.flush()
                sys.stdout.write(f"{prefix} {line}")
                sys.stdout.flush()
            proc.wait()

        if proc.returncode != 0:
            task.status = "failed"
            task.error = f"Selftest command exited with code {proc.returncode}"
        else:
            task.status = "done"
    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        # Write exception.txt
        exc_path = task.task_dir / "exception.txt"
        exc_path.write_text(traceback.format_exc(), encoding="utf-8")


def run_parallel(
    *,
    args: argparse.Namespace,
    config: RunConfig,
    state: RunState,
    stats_ids: list[str],
    stats_id_set: set,
    system_prompt: str,
) -> int:
    machine_info = _collect_machine_info()
    selftest_mode = getattr(args, "selftest_parallel", False)

    # Preflight check: detect dirty repo before spending credits
    # Skip in selftest mode since we're not doing real work
    if not selftest_mode:
        auto_stash = getattr(args, "auto_stash", False)
        force_dirty = getattr(args, "force_dirty", False)
        verify_mode = getattr(args, "verify_mode", "strict")

        preflight_ok, preflight_msg, stash_ref = _preflight_check_repo(
            state.project_root,
            auto_stash=auto_stash,
            force_dirty=force_dirty,
            runs_dir=state.runs_dir,
        )

        if not preflight_ok:
            print(f"\n[orchestrator] {preflight_msg}")
            # Write error to runs_dir for debugging
            error_path = state.runs_dir / "preflight_error.txt"
            error_path.write_text(preflight_msg, encoding="utf-8")
            stderr_path = state.runs_dir / "stderr.log"
            stderr_path.write_text(f"PREFLIGHT FAILURE\n\n{preflight_msg}\n", encoding="utf-8")
            # Write minimal run.json
            run_json_path = state.runs_dir / "run.json"
            run_json_path.write_text(
                json.dumps(
                    {
                        "status": "preflight_failed",
                        "error": preflight_msg,
                        "run_id": state.run_id,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return 2

        if stash_ref:
            print(f"[orchestrator] {preflight_msg}")
            print(f"[orchestrator] IMPORTANT: Run 'git stash pop {stash_ref}' to restore your changes after the run")
        elif "WARNING" in preflight_msg:
            print(f"[orchestrator] {preflight_msg}")

        # Run bootstrap to ensure environment is consistent before tasks start
        print("[orchestrator] Running environment bootstrap...")
        from bridge.verify_repair.bootstrap import run_bootstrap
        bootstrap_log = state.runs_dir / "bootstrap_start.log"
        bootstrap_result = run_bootstrap(
            state.project_root,
            log_path=bootstrap_log,
            verbose=True,
        )
        if bootstrap_result.success:
            if not bootstrap_result.skipped:
                print(f"[orchestrator] Bootstrap completed in {bootstrap_result.elapsed_s:.1f}s")
        else:
            print(f"[orchestrator] WARNING: Bootstrap failed: {bootstrap_result.stderr[:200]}")
            print("[orchestrator] Continuing anyway - verify will catch any missing dependencies")

    # In selftest mode, we use synthetic tasks
    if selftest_mode:
        print("[orchestrator] SELFTEST MODE: using synthetic tasks (no real agents)")
        tasks: list[ParallelTask] = [
            ParallelTask(
                id="SELFTEST-A",
                title="Selftest task A (no deps)",
                description="First selftest task with no dependencies",
                agent="codex",
                intensity="low",
                locks=["lock-a"],
                depends_on=[],
                solo=False,
            ),
            ParallelTask(
                id="SELFTEST-B",
                title="Selftest task B (depends on A)",
                description="Second selftest task depending on A",
                agent="codex",
                intensity="low",
                locks=["lock-b"],
                depends_on=["SELFTEST-A"],
                solo=False,
            ),
            ParallelTask(
                id="SELFTEST-C",
                title="Selftest task C (depends on A)",
                description="Third selftest task depending on A",
                agent="codex",
                intensity="low",
                locks=["lock-c"],
                depends_on=["SELFTEST-A"],
                solo=False,
            ),
        ]
        milestone_id = "SELFTEST"
        design_doc_text = "# Selftest Design Document\n"
        max_workers = 2
        safe_cap = 2
    else:
        # Parse design document using modular adapter layer
        # This supports arbitrary markdown formats without rigid heading assumptions
        contract_mode: ContractMode = getattr(args, "design_doc_contract", "loose")
        milestone_override = getattr(args, "milestone_id", None)

        design_spec = parse_design_doc(
            state.design_doc_path,
            contract_mode=contract_mode,
            milestone_override=milestone_override,
        )

        # Log design doc parsing results
        print(f"[orchestrator] design_doc: path={state.design_doc_path}, hash={design_spec.doc_hash[:12]}")
        print(f"[orchestrator] design_doc: milestone={design_spec.milestone_id or '(not found)'}, "
              f"requirements={len(design_spec.requirements)}, contract_mode={contract_mode}")

        if design_spec.warnings:
            for warning in design_spec.warnings[:5]:
                print(f"[orchestrator] design_doc WARNING: {warning}")
            if len(design_spec.warnings) > 5:
                print(f"[orchestrator] design_doc: ... and {len(design_spec.warnings) - 5} more warnings")

        # Check for errors based on contract mode
        if design_spec.errors:
            for error in design_spec.errors:
                print(f"[orchestrator] design_doc ERROR: {error}")
            if contract_mode != "off":
                print(f"[orchestrator] ERROR: Design doc validation failed. Use --design-doc-contract=off to bypass.")
                return 2

        # Extract values for use in prompts
        design_doc_text = design_spec.raw_text
        if not design_doc_text.strip():
            print(f"[orchestrator] ERROR: design doc not found or empty: {state.design_doc_path}")
            return 2

        # Use milestone from spec (may be CLI override or extracted)
        milestone_id = design_spec.milestone_id or "M0"

        # Save design spec artifact for debugging
        design_spec_artifact = state.runs_dir / "design_doc_spec.json"
        try:
            import json
            design_spec_artifact.write_text(json.dumps(design_spec.to_dict(), indent=2), encoding="utf-8")
        except Exception:
            pass  # Non-fatal if we can't write the artifact

        # Thresholds & stream limits
        cpu_thr = args.cpu_threshold if args.cpu_threshold > 0 else config.parallel.cpu_intensive_threshold_pct
        mem_thr = args.mem_threshold if args.mem_threshold > 0 else config.parallel.mem_intensive_threshold_pct
        term_bytes = args.terminal_max_bytes if args.terminal_max_bytes > 0 else config.parallel.terminal_max_bytes_per_worker
        term_line = args.terminal_max_line_len if args.terminal_max_line_len > 0 else config.parallel.terminal_max_line_length

        allow_resource_intensive = bool(args.allow_resource_intensive)

        # Planner step (CODEX)
        plan_schema_path = (state.project_root / args.task_plan_schema).resolve()
        plan_prompt_path = state.runs_dir / "task_planner_prompt.txt"
        plan_out_path = state.runs_dir / "task_plan.json"

        # Safety cap hint: default to cores - 6 (leaving headroom for OS + orchestrator), capped at 12.
        # This gives 10 on a 16-core machine. Use --max-workers to override if needed.
        cores = machine_info.get("cpu_cores") or 1
        safe_cap = min(12, max(4, int(cores) - 6))

        # If CLI explicitly sets --max-workers, use that as the planner limit too
        cli_cap = int(args.max_workers) if args.max_workers and int(args.max_workers) > 0 else 0
        planner_max_workers_limit = cli_cap if cli_cap > 0 else safe_cap

        # Get planner profile (default: balanced)
        planner_profile = getattr(args, "planner_profile", "balanced") or "balanced"

        # Log throughput mode settings
        if planner_profile == "throughput":
            print(f"[orchestrator] planner_profile=throughput (target_roots={planner_max_workers_limit}, target_plan_cap={planner_max_workers_limit})")

        plan_prompt = _build_task_plan_prompt(
            design_doc_text=design_doc_text,
            milestone_id=milestone_id,
            max_workers_limit=planner_max_workers_limit,
            cpu_threshold_pct_total=cpu_thr,
            mem_threshold_pct_total=mem_thr,
            machine_info=machine_info,
            planner_profile=planner_profile,
        )
        plan_prompt_path.write_text(plan_prompt, encoding="utf-8")

        # Select planner agent through policy (default: codex, but --only-* overrides)
        policy = get_agent_policy()
        planner_agent = policy.enforce("codex", "planner agent selection")
        planner_script = state.project_root / config.agent_scripts.get(planner_agent, "")
        if not planner_script.exists():
            print(f"[orchestrator] ERROR: planner script not found: {planner_script}")
            return 2

        print(f"[orchestrator] parallel: planning tasks via {planner_agent} (schema={plan_schema_path})")
        env = os.environ.copy()
        env["WRITE_ACCESS"] = "0"
        env["ORCH_WRITE_ACCESS"] = "0"
        # Signal to the agent wrapper that this is a task_plan schema (not turn schema)
        # This prevents turn normalization from corrupting the planner output
        env["ORCH_SCHEMA_KIND"] = "task_plan"

        # Planner loop with auto-reprompt for throughput mode
        # Uses comprehensive plan quality scoring to determine reprompt
        max_planner_attempts = 3 if planner_profile == "throughput" else 1
        plan_obj: dict[str, Any] | None = None
        final_quality_report: dict[str, Any] | None = None

        for planner_attempt in range(max_planner_attempts):
            current_prompt_path = plan_prompt_path if planner_attempt == 0 else state.runs_dir / f"task_planner_prompt_retry{planner_attempt}.txt"
            current_out_path = plan_out_path if planner_attempt == 0 else state.runs_dir / f"task_plan_retry{planner_attempt}.json"

            rc, _, err = _run_cmd(
                [str(planner_script), str(current_prompt_path), str(plan_schema_path), str(current_out_path)],
                cwd=state.project_root,
                env=env,
                stream=True,
            )
            if rc != 0 or not current_out_path.exists():
                print("[orchestrator] ERROR: planning step failed")
                if err:
                    print(err)
                return 2

            plan_obj = _try_parse_json(current_out_path.read_text(encoding="utf-8"))
            if not isinstance(plan_obj, dict):
                print("[orchestrator] ERROR: could not parse task_plan.json")
                return 2

            # Analyze plan width (basic metrics)
            analysis = _analyze_plan_width(plan_obj, planner_max_workers_limit)

            # Build comprehensive quality report (throughput mode only)
            quality_report = _build_plan_quality_report(
                plan_obj, planner_max_workers_limit, planner_profile
            )
            final_quality_report = quality_report

            # Log plan quality metrics
            metrics = quality_report.get("metrics", {})
            if planner_profile == "throughput" and metrics:
                max_lock = metrics.get("max_lock_count", 0)
                tests_intent = metrics.get("coverage_intent", 0)
                should_reprompt = quality_report.get("should_reprompt", False)
                print(
                    f"[orchestrator] plan_quality: cap={metrics.get('plan_cap', 0)} "
                    f"roots={metrics.get('root_ready', 0)} tasks={metrics.get('total_tasks', 0)} "
                    f"lock_pressure=max(lock)={max_lock} tests_intent={tests_intent:.2f} "
                    f"reprompt={'yes' if should_reprompt else 'no'}"
                )
            else:
                # Balanced mode: simple width log
                print(f"[orchestrator] plan_width: tasks={analysis['task_count']}, roots={analysis['root_count']}, plan_cap={analysis['plan_cap']}")

            # In throughput mode, use quality report to determine reprompt
            if planner_profile == "throughput" and quality_report.get("should_reprompt", False):
                if planner_attempt < max_planner_attempts - 1:
                    print(f"[orchestrator] WARNING: Plan does not meet throughput quality requirements (attempt {planner_attempt + 1}/{max_planner_attempts})")
                    for issue in quality_report.get("hard_failures", []):
                        print(f"[orchestrator]   FAIL: {issue}")
                    for warning in quality_report.get("soft_warnings", []):
                        print(f"[orchestrator]   WARN: {warning}")
                    print("[orchestrator] Re-prompting planner with targeted correction...")

                    # Build correction prompt with quality report and attempt number
                    correction_prompt = _build_throughput_correction_prompt(
                        analysis,
                        planner_max_workers_limit,
                        quality_report=quality_report,
                        attempt_number=planner_attempt + 1,
                    )
                    # Append correction to original prompt
                    combined_prompt = plan_prompt + "\n\n" + correction_prompt
                    next_prompt_path = state.runs_dir / f"task_planner_prompt_retry{planner_attempt + 1}.txt"
                    next_prompt_path.write_text(combined_prompt, encoding="utf-8")
                    continue  # Retry with correction
                else:
                    # Exhausted retries - warn loudly but proceed (do not fail run)
                    print(f"[orchestrator] WARNING: Plan still has quality issues after {max_planner_attempts} attempts. Proceeding anyway.")
                    for issue in quality_report.get("hard_failures", []):
                        print(f"[orchestrator]   FAIL: {issue}")
                    for warning in quality_report.get("soft_warnings", []):
                        print(f"[orchestrator]   WARN: {warning}")
                    break
            else:
                # Plan is acceptable (or balanced mode)
                if planner_profile == "throughput":
                    # Log any soft warnings even for acceptable plans
                    for warning in quality_report.get("soft_warnings", []):
                        print(f"[orchestrator] note: {warning}")
                break

        # Write plan quality report to runs directory (observability)
        if final_quality_report and planner_profile == "throughput":
            quality_report_path = state.runs_dir / "plan_quality_report.json"
            try:
                quality_report_path.write_text(
                    json.dumps(final_quality_report, indent=2, default=str),
                    encoding="utf-8",
                )
                print("[orchestrator] wrote plan_quality_report.json")
            except Exception as e:
                print(f"[orchestrator] warning: could not write plan_quality_report.json: {e}")

        if plan_obj is None:
            print("[orchestrator] ERROR: no valid plan generated")
            return 2

        raw_tasks = plan_obj.get("tasks", [])
        plan_max_parallel = int(plan_obj.get("max_parallel_tasks", safe_cap) or safe_cap)

        # Helper to select agent when "either" is specified
        agent_round_robin_counter = [0]  # mutable for closure

        def _select_agent_for_task(preferred: str, task_title: str, task_desc: str) -> str:
            """Select agent using heuristics when 'either' or invalid agent is specified.

            NOTE: This returns a heuristic-based selection. The caller must still
            apply AgentPolicy.enforce() to the result for forced mode compliance.
            """
            if preferred in AGENTS:
                return preferred

            # Heuristic: check title/description for keywords
            text = (task_title + " " + task_desc).lower()
            claude_keywords = ["schema", "doc", "review", "test", "edge", "api", "refactor", "spec", "design", "validate"]
            codex_keywords = ["implement", "build", "cli", "integration", "low-level", "pipeline", "backend", "export"]

            claude_score = sum(1 for kw in claude_keywords if kw in text)
            codex_score = sum(1 for kw in codex_keywords if kw in text)

            if claude_score > codex_score:
                return "claude"
            elif codex_score > claude_score:
                return "codex"
            else:
                # Round-robin fallback for ties
                agent_round_robin_counter[0] += 1
                return "claude" if agent_round_robin_counter[0] % 2 == 0 else "codex"

        tasks = []
        for t in raw_tasks if isinstance(raw_tasks, list) else []:
            if not isinstance(t, dict):
                continue
            tid = str(t.get("id", "")).strip()
            if not tid:
                continue
            # The planner schema uses preferred_agent/estimated_intensity, but we also
            # accept legacy keys (agent/intensity) for compatibility.
            agent = str(t.get("preferred_agent", t.get("agent", "either"))).strip().lower()
            task_title = str(t.get("title", tid)).strip()
            task_desc = str(t.get("description", "")).strip()
            if agent == "either" or agent not in AGENTS:
                agent = _select_agent_for_task(agent, task_title, task_desc)
            enabled = getattr(config, "enable_agents", None) or getattr(config, "enabled_agents", None) or []
            if enabled and agent not in enabled:
                agent = enabled[0]
            # Apply agent policy enforcement (--only-* flags)
            agent = policy.enforce(agent, f"task {tid} agent selection")
            tasks.append(
                ParallelTask(
                    id=tid,
                    title=str(t.get("title", tid)).strip(),
                    description=str(t.get("description", "")).strip() or str(t.get("title", tid)).strip(),
                    agent=agent,
                    intensity=str(t.get("estimated_intensity", t.get("intensity", "low"))).strip().lower(),
                    locks=list(t.get("locks", []) or []),
                    touched_paths=list(t.get("touched_paths", []) or []),
                    depends_on=list(t.get("depends_on", []) or []),
                    solo=bool(t.get("solo", False)),
                )
            )

        tasks = _select_only_tasks(tasks, args.only_task)

        # Inject hot-file locks to prevent concurrent edits to critical files
        tasks = _inject_hot_file_locks(tasks)

        # Inject general overlap locks for any files touched by multiple tasks
        # This provides "as narrow as possible, as strong as necessary" locking
        tasks = _inject_overlap_locks(tasks)

        if not tasks:
            print("[orchestrator] ERROR: no tasks in plan")
            return 2

        # Final worker count (CLI override takes precedence and bypasses safe_cap)
        if args.max_workers and int(args.max_workers) > 0:
            max_workers = min(16, int(args.max_workers))
        else:
            max_workers = min(16, safe_cap, plan_max_parallel)
            max_workers = max(1, max_workers)

    # Common setup for both selftest and real mode
    if not selftest_mode:
        cpu_thr = args.cpu_threshold if args.cpu_threshold > 0 else config.parallel.cpu_intensive_threshold_pct
        mem_thr = args.mem_threshold if args.mem_threshold > 0 else config.parallel.mem_intensive_threshold_pct
        term_bytes = args.terminal_max_bytes if args.terminal_max_bytes > 0 else config.parallel.terminal_max_bytes_per_worker
        term_line = args.terminal_max_line_len if args.terminal_max_line_len > 0 else config.parallel.terminal_max_line_length
        allow_resource_intensive = bool(args.allow_resource_intensive)
    else:
        cpu_thr = 0.0
        mem_thr = 0.0
        term_bytes = 10000
        term_line = 200
        allow_resource_intensive = True

    # Verbose logging of all worker limits
    if not selftest_mode:
        cli_val = int(args.max_workers) if args.max_workers and int(args.max_workers) > 0 else 0
        print(f"[orchestrator] parallel: max_workers={max_workers} (safe_cap={safe_cap}, plan_cap={plan_max_parallel}, cli_cap={cli_val if cli_val > 0 else 'auto'})")
    else:
        print(f"[orchestrator] parallel: max_workers={max_workers} (selftest mode)")

    tasks_dir = state.runs_dir / "tasks"
    worktrees_dir = state.runs_dir / "worktrees"
    manual_dir = state.runs_dir / "manual"
    _ensure_dir(tasks_dir)
    _ensure_dir(worktrees_dir)
    _ensure_dir(manual_dir)

    by_id = {t.id: t for t in tasks}

    def deps_satisfied(t: ParallelTask) -> bool:
        """Check if all dependencies are satisfied.

        Only "done" counts as satisfied. "blocked", "failed", "manual",
        "pending_rerun", and "resource_killed" do NOT satisfy dependencies.
        """
        for dep in t.depends_on:
            dt = by_id.get(dep)
            if dt is None:
                # Unknown dependency - treat as satisfied (might be external)
                continue
            if dt.status != "done":
                return False
        return True

    def is_root_failure(t: ParallelTask) -> bool:
        """Check if a task is a root failure (not transitively blocked)."""
        return t.status in ("failed", "manual", "pending_rerun", "resource_killed")

    # Scheduler state
    held_locks: set = set()
    running: dict[str, concurrent.futures.Future] = {}  # task_id -> future
    git_lock = threading.Lock()

    # Patch integrator for commit-free integration
    patch_integrator = PatchIntegrator(state.project_root, state.runs_dir)
    patch_integrator.set_lock(git_lock)

    def locks_available(t: ParallelTask) -> bool:
        # Check named locks
        if set(map(str, t.locks)) & held_locks:
            return False
        # Check touched_paths overlap - auto-generate path-based locks
        task_paths = set(str(p) for p in t.touched_paths)
        return not task_paths & held_locks

    # Create two-lane scheduler for high-utilization execution
    lane_config = LaneConfig.from_max_workers(max_workers)
    two_lane_scheduler = TwoLaneScheduler(
        lane_config=lane_config,
        tasks=tasks,
        deps_satisfied_fn=deps_satisfied,
        locks_available_fn=locks_available,
    )
    print(f"[orchestrator] two-lane scheduler: coding_lane={lane_config.coding_lane_size}, executor_lane={lane_config.executor_lane_size}")

    # Create backfill generator to keep workers busy when primary tasks are blocked
    backfill_generator = BackfillGenerator(
        project_root=str(state.project_root),
        min_queue_depth=max_workers * 2,  # Keep 2x workers worth of tasks ready
    )
    backfill_tasks_generated = 0
    max_backfill_tasks = max_workers * 3  # Cap total backfill to avoid runaway

    def can_start(t: ParallelTask) -> bool:
        """Check if task can start using two-lane scheduler."""
        return two_lane_scheduler.can_start(t)

    def get_ready_tasks() -> list[ParallelTask]:
        """Get ready tasks sorted by priority."""
        return two_lane_scheduler.get_ready_tasks()

    # Allowed directories for backfill tasks (to prevent merge conflicts)
    # NOTE: bridge/ removed from default to protect orchestrator core
    BACKFILL_ALLOWED_DIRS = ["tests/", "docs/", ".github/"]

    # Track recently rejected backfill paths for cooldown (prevents spam loops)
    # Maps path -> cycle_count when rejected
    backfill_rejection_cooldown: dict[str, int] = {}
    BACKFILL_COOLDOWN_CYCLES = 5  # Number of cycles before retry after rejection

    def derive_backfill_lock(filler: FillerTask) -> list[str]:
        """Derive fine-grained locks for a backfill task based on its target.

        Instead of a global "backfill" lock, we use task-type-specific locks.
        This allows multiple backfill tasks to run concurrently when they
        target different areas (e.g., lint and docs can run together).

        Lock strategy:
        - backfill:type:<task_type> - prevents two lint tasks colliding
        - Different task types can run in parallel
        """
        # Use task type as lock key - allows parallel backfill of different types
        task_type = filler.task_type
        return [f"backfill:type:{task_type}"]

    def convert_filler_to_parallel_task(filler: FillerTask) -> ParallelTask:
        """Convert a FillerTask to a ParallelTask for execution.

        Backfill tasks are scope-constrained to safe directories to prevent
        merge conflicts with primary tasks.

        Uses file-derived locks instead of global "backfill" lock to allow
        concurrent backfill when tasks target different areas.
        """
        # Add scope constraint to description
        scope_note = (
            f"\n\nSCOPE CONSTRAINT: This is a FILLER task. You MUST only modify files in: "
            f"{', '.join(BACKFILL_ALLOWED_DIRS)}. "
            f"Do NOT touch files outside these directories (especially src/, api.py, bridge/loop.py, DESIGN_DOCUMENT.md, etc.)"
        )

        # Derive locks based on task type - allows parallel execution of different types
        task_locks = derive_backfill_lock(filler)

        return ParallelTask(
            id=filler.id,
            title=filler.title,
            description=filler.description + scope_note,
            agent="claude",  # Use Claude for safe filler tasks
            intensity="light",  # Filler tasks are always light
            locks=task_locks,  # Type-based locks instead of global "backfill"
            depends_on=[],  # No dependencies
            touched_paths=[],  # Will be filled during execution
            solo=False,
            max_retries=1,  # One retry max for filler tasks
        )

    def maybe_generate_backfill() -> int:
        """Generate backfill tasks if needed. Returns count of tasks added."""
        nonlocal backfill_tasks_generated, tasks, by_id

        # Don't generate in selftest mode
        if selftest_mode:
            return 0

        # Check if we've hit the backfill cap
        if backfill_tasks_generated >= max_backfill_tasks:
            return 0

        # Check if we should generate based on queue depth
        pending_count = sum(1 for t in tasks if t.status == "pending")
        if not backfill_generator.should_generate(pending_count, max_workers):
            return 0

        # Check if there are available workers but no ready tasks
        ready_count = len(get_ready_tasks())
        available_workers = len(available_worker_ids)
        if ready_count >= available_workers:
            return 0

        # Generate filler tasks
        count_needed = min(
            available_workers - ready_count,
            max_backfill_tasks - backfill_tasks_generated,
            5,  # Generate at most 5 at a time
        )

        if count_needed <= 0:
            return 0

        filler_tasks = backfill_generator.generate_filler_tasks(count_needed)
        added = 0
        for filler in filler_tasks:
            parallel_task = convert_filler_to_parallel_task(filler)
            tasks.append(parallel_task)
            by_id[parallel_task.id] = parallel_task
            backfill_tasks_generated += 1
            added += 1
            print(f"[orchestrator] BACKFILL: Generated {parallel_task.id} - {parallel_task.title}")

        # Update scheduler with new tasks
        if added > 0:
            two_lane_scheduler.update_tasks(tasks)

        return added

    def record_backfill_rejection(task_id: str, reason: str, rejected_paths: list[str]) -> None:
        """Record a backfill rejection for cooldown tracking.

        When a backfill patch is scope-rejected, this records the rejection
        to prevent generating similar tasks for a cooldown period.

        Args:
            task_id: The FILLER-* task ID that was rejected
            reason: The rejection reason (e.g., "SCOPE_REJECTED")
            rejected_paths: List of paths that caused the rejection
        """
        nonlocal backfill_rejection_cooldown

        current_cycle = backfill_tasks_generated  # Use as cycle counter
        for path in rejected_paths:
            backfill_rejection_cooldown[path] = current_cycle
            print(f"[orchestrator] BACKFILL-COOLDOWN: {path} on cooldown for {BACKFILL_COOLDOWN_CYCLES} cycles (rejected in {task_id})")

    def is_backfill_task_type_on_cooldown(task_type: str) -> bool:
        """Check if a backfill task type is on cooldown due to recent rejections.

        Returns True if this task type should not be generated yet.
        """
        current_cycle = backfill_tasks_generated
        # Check if any rejected paths are still on cooldown for this task type
        for path, rejection_cycle in list(backfill_rejection_cooldown.items()):
            if current_cycle - rejection_cycle < BACKFILL_COOLDOWN_CYCLES:
                # This path is still on cooldown
                # For now, we just check if any rejections are recent
                # More sophisticated: match task_type to path patterns
                if task_type in ("lint", "type_hints") and "bridge/" in path:
                    return True
                if task_type in ("test",) and "tests/" in path:
                    return True
            else:
                # Cooldown expired, remove from tracking
                del backfill_rejection_cooldown[path]
        return False

    def cleanup_expired_cooldowns() -> None:
        """Clean up expired cooldown entries."""
        nonlocal backfill_rejection_cooldown
        current_cycle = backfill_tasks_generated
        expired = [
            path for path, cycle in backfill_rejection_cooldown.items()
            if current_cycle - cycle >= BACKFILL_COOLDOWN_CYCLES
        ]
        for path in expired:
            del backfill_rejection_cooldown[path]

    # Integration helper: prefers patch-based integration, falls back to git merge
    def merge_task(t: ParallelTask) -> bool:
        if selftest_mode or not t.branch:
            return True

        # Compute task milestone from ID prefix
        task_milestone = _extract_milestone_from_task_id(t.id, fallback=milestone_id)

        # Try patch-based integration first (preferred - no sandbox issues)
        if getattr(t, 'has_patch', False) and t.task_dir:
            success, msg, commit_sha = patch_integrator.integrate_task(
                task_id=t.id,
                task_dir=t.task_dir,
                task_branch=t.branch,
                agent_name=t.agent,
                task_context=f"Task: {t.title}\nDescription: {t.description}",
                milestone_id=task_milestone,
            )
            if success:
                if commit_sha:
                    t.commit_sha = commit_sha
                    # Record successful result for backfill cooldown tracking
                    if t.id.startswith("FILLER-"):
                        backfill_generator.record_successful_result(t.id)
                elif "No changes to commit" in msg:
                    # No-op result - track for backfill cooldown
                    if t.id.startswith("FILLER-"):
                        backfill_generator.record_noop_result(t.id)
                        print(f"[orchestrator] FILLER NO-OP: {t.id} produced no changes")
                print(f"[orchestrator] PATCH INTEGRATED: {t.id} - {msg}")
                return True

            # Handle SCOPE_REJECTED - do NOT fall back to git merge
            if "SCOPE_REJECTED" in msg:
                print(f"[orchestrator] SCOPE_REJECTED for {t.id}: {msg}")
                if t.id.startswith("FILLER-"):
                    backfill_generator.record_rejection(t.id)
                t.status = "failed"
                t.error = msg
                return False

            # Only fall back to git merge if:
            # 1. Not needs_manual_resolution
            # 2. Worker actually created a commit (has commit_sha or branch differs from base)
            if "needs_manual_resolution" in msg:
                print(f"[orchestrator] Patch integration needs manual resolution for {t.id}: {msg}")
                # Backfill tasks are optional - don't mark manual, just log and return success
                if _is_backfill_task_id(t.id):
                    t.error = msg  # Record for visibility
                    print(f"[orchestrator] FILLER OPTIONAL SKIP: {t.id} needs manual resolution but is optional")
                    return True
                # Non-backfill tasks: mark manual with proper manual_path
                _mark_task_manual(
                    task=t,
                    reason=msg,
                    manual_dir=manual_dir,
                    schema_path=state.schema_path,
                )
                return False

            # Check if there's actually a commit to merge
            # If worker only produced a patch artifact without committing, git merge would be a no-op
            if not getattr(t, 'commit_sha', None):
                # No commit SHA means no actual commit to merge - treat as integration failure
                print(f"[orchestrator] Patch integration failed for {t.id} (no commit to merge): {msg}")
                t.status = "failed"
                t.error = f"Patch integration failed (no commit): {msg}"
                return False

            print(f"[orchestrator] Patch integration issue for {t.id}: {msg}, trying git merge...")

        # Fall back to git merge (for legacy commits or when patch integration fails)
        with git_lock:
            rc, out, err = _run_cmd(
                ["git", "merge", "--no-ff", "--no-edit", t.branch],
                cwd=state.project_root,
                env=os.environ.copy(),
                stream=True,
            )
            if rc == 0:
                return True

            # Merge conflict detected - attempt auto-resolution
            print(f"[orchestrator] MERGE CONFLICT detected for {t.id}, attempting auto-resolution...")

            # Try auto-resolution (handles __init__.py files and uses agent for others)
            auto_ok, auto_msg = _attempt_auto_merge_resolution(
                project_root=state.project_root,
                task_id=t.id,
                runs_dir=state.runs_dir,
                task_context=f"Task: {t.title}\nDescription: {t.description}",
                milestone_id=task_milestone,
            )

            if auto_ok:
                print(f"[orchestrator] AUTO-RESOLVED: {auto_msg}")
                return True

            # Auto-resolution failed - abort and mark for manual
            print(f"[orchestrator] AUTO-RESOLUTION FAILED: {auto_msg}")
            _run_cmd(["git", "merge", "--abort"], cwd=state.project_root, env=os.environ.copy())
            t.status = "manual"
            t.error = f"Merge conflict; auto-resolve failed: {auto_msg}"
            t.manual_path = _write_manual_task_file(
                manual_dir=manual_dir,
                task=t,
                reason=t.error,
                agent_cmd=[],
                schema_path=state.schema_path,
                prompt_path=t.prompt_path or Path(""),
                out_path=t.out_path or Path(""),
                raw_log_path=t.raw_log_path or Path(""),
            )
            return False

    # Task execution wrapper with exception handling
    def execute_task(t: ParallelTask, worker_id: int) -> str:
        """Execute a task and return its ID when done. Handles exceptions."""
        # Compute per-task milestone_id from task ID prefix (e.g., "M2-SIM-SCHEMA" -> "M2")
        # This ensures each task gets the correct milestone context in MULTI mode
        task_milestone_id = _extract_milestone_from_task_id(t.id, fallback=milestone_id)
        try:
            if selftest_mode:
                _run_selftest_task(task=t, worker_id=worker_id, tasks_dir=tasks_dir)
            else:
                _run_parallel_task(
                    task=t,
                    worker_id=worker_id,
                    state=state,
                    config=config,
                    milestone_id=task_milestone_id,
                    design_doc_text=design_doc_text,
                    system_prompt=system_prompt,
                    stats_id_set=stats_id_set,
                    tasks_dir=tasks_dir,
                    worktrees_dir=worktrees_dir,
                    manual_dir=manual_dir,
                    git_lock=git_lock,
                    cpu_threshold_pct_total=cpu_thr,
                    mem_threshold_pct_total=mem_thr,
                    terminal_max_bytes=term_bytes,
                    terminal_max_line_length=term_line,
                    allow_resource_intensive=allow_resource_intensive,
                )
        except Exception as e:
            t.status = "failed"
            t.error = f"Exception: {e}"
            # Write exception.txt
            if t.task_dir:
                _ensure_dir(t.task_dir)
                exc_path = t.task_dir / "exception.txt"
                exc_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[orchestrator] ERROR: task {t.id} raised exception: {e}")
        return t.id

    # Main scheduler loop using concurrent.futures with FIRST_COMPLETED
    print(f"[orchestrator] parallel: {len(tasks)} task(s) queued")
    last_heartbeat = time.monotonic()
    heartbeat_interval = 15.0  # seconds

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        worker_ids = list(range(max_workers))
        available_worker_ids = list(worker_ids)

        while True:
            # Calculate stats
            done_count = sum(1 for t in tasks if t.status == "done")
            failed_count = sum(1 for t in tasks if t.status in ("failed", "manual", "pending_rerun", "resource_killed"))
            blocked_count = sum(1 for t in tasks if t.status == "blocked")
            running_count = len(running)
            pending_count = sum(1 for t in tasks if t.status == "pending")
            ready_tasks = get_ready_tasks()

            # Heartbeat logging with lane stats
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_interval:
                lane_stats = two_lane_scheduler.get_lane_stats()
                # Sample metrics for utilization tracking
                two_lane_scheduler.sample_metrics(queue_depth=pending_count)
                print(
                    f"[orchestrator] parallel: progress done={done_count} running={running_count} queued={pending_count} ready={len(ready_tasks)} "
                    f"lanes[coding={lane_stats['coding_active']}/{lane_stats['coding_capacity']}, exec={lane_stats['executor_active']}/{lane_stats['executor_capacity']}]"
                )
                last_heartbeat = now

            # Check completion - terminal states end the run
            terminal_states = ("done", "failed", "manual", "blocked", "pending_rerun", "resource_killed")
            all_finished = all(t.status in terminal_states for t in tasks)
            if all_finished:
                break

            # Detect stuck state: no tasks running, no tasks ready, but tasks remain pending
            # SELF-HEALING: Instead of immediately marking as stuck, try recovery actions
            if not running and not ready_tasks and pending_count > 0:
                # Identify root failures (tasks that failed on their own, not due to dependencies)
                root_failure_ids = {t.id for t in tasks if is_root_failure(t)}
                root_failures = [t for t in tasks if t.id in root_failure_ids]

                # Compute transitive closure of all blocked tasks
                blocked_ids = _compute_transitive_blocked(tasks, root_failure_ids)
                pending_but_not_blocked = [t for t in tasks if t.status == "pending" and t.id not in blocked_ids]

                print(f"\n[orchestrator] STUCK DETECTED: {pending_count} task(s) cannot proceed")
                print("[orchestrator] Attempting SELF-HEALING recovery...")

                # Track if we recovered
                recovered = False
                recovery_reason = ""

                # SELF-HEALING STRATEGY 1: Check for lock contention without root failures
                # Tasks may be waiting on locks held by no one - re-check lock availability
                if pending_but_not_blocked and not root_failures:
                    print("[orchestrator] RECOVERY: Checking for stale lock contention...")
                    for t in pending_but_not_blocked:
                        # Re-check if locks are available now
                        locks_free = all(str(lk) not in held_locks for lk in t.locks)
                        paths_free = all(str(p) not in held_locks for p in t.touched_paths)
                        deps_done = all(by_id.get(d) and by_id[d].status == "done" for d in t.depends_on)
                        if locks_free and paths_free and deps_done:
                            print(f"[orchestrator] RECOVERY: Task {t.id} is now ready (stale lock cleared)")
                            recovered = True
                            recovery_reason = "stale_lock_cleared"
                            break  # Break out of for loop, will continue scheduler loop

                # SELF-HEALING STRATEGY 2: If there are root failures, schedule them for rerun
                # This converts permanent failures into retriable tasks
                if not recovered and root_failures:
                    print(f"[orchestrator] RECOVERY: {len(root_failures)} root failure(s) detected")
                    rerun_count = 0
                    for t in root_failures:
                        # Only retry if the task hasn't been retried too many times
                        current_retries = getattr(t, '_self_heal_retries', 0)
                        max_self_heal_retries = 2
                        if current_retries < max_self_heal_retries:
                            t._self_heal_retries = current_retries + 1
                            # Reset the task for rerun
                            print(f"[orchestrator] RECOVERY: Scheduling {t.id} for self-heal rerun (attempt {t._self_heal_retries})")
                            t.status = "pending"
                            t.error = None
                            t.retry_count = 0  # Reset retry counter
                            rerun_count += 1
                        else:
                            print(f"[orchestrator] RECOVERY: {t.id} exhausted self-heal retries ({max_self_heal_retries})")

                    if rerun_count > 0:
                        recovered = True
                        recovery_reason = f"scheduled_{rerun_count}_reruns"
                        print(f"[orchestrator] RECOVERY: {rerun_count} task(s) scheduled for self-heal rerun")

                # If still not recovered, print detailed diagnostics and mark as truly stuck
                if not recovered:
                    print(f"\n[orchestrator] SELF-HEALING FAILED: Marking run as STUCK")

                    if root_failures:
                        print(f"\n  ROOT FAILURES ({len(root_failures)}):")
                        for t in root_failures:
                            reason = t.error or t.status
                            manual_hint = f" -> {t.manual_path}" if t.manual_path else ""
                            retries = getattr(t, '_self_heal_retries', 0)
                            print(f"    - {t.id}: {reason} (retries={retries}){manual_hint}")

                    if blocked_ids:
                        blocked_pending = [t for t in tasks if t.status == "pending" and t.id in blocked_ids]
                        if blocked_pending:
                            print(f"\n  BLOCKED TASKS ({len(blocked_pending)}):")
                            for t in blocked_pending:
                                # Find which root failures this task is transitively blocked by
                                blocking_roots = []
                                visited = set()
                                queue = list(t.depends_on)
                                while queue:
                                    dep_id = queue.pop(0)
                                    if dep_id in visited:
                                        continue
                                    visited.add(dep_id)
                                    if dep_id in root_failure_ids:
                                        blocking_roots.append(dep_id)
                                    elif dep_id in blocked_ids:
                                        # This dep is also blocked, trace further
                                        dep_task = by_id.get(dep_id)
                                        if dep_task:
                                            queue.extend(dep_task.depends_on)
                                if blocking_roots:
                                    print(f"    - {t.id}: blocked by root failures: {blocking_roots}")
                                else:
                                    print(f"    - {t.id}: transitively blocked")

                    # Check for true cycles (tasks pending but not blocked by any root failure)
                    if pending_but_not_blocked:
                        print(f"\n  DEPENDENCY CYCLES ({len(pending_but_not_blocked)}):")
                        for t in pending_but_not_blocked:
                            unmet = [d for d in t.depends_on if by_id.get(d) and by_id[d].status != "done"]
                            print(f"    - {t.id}: cycle/deadlock waiting on: {unmet}")

                    # Mark stuck tasks with proper status
                    for t in tasks:
                        if t.status == "pending":
                            if t.id in blocked_ids:
                                # Transitively blocked by a root failure
                                t.status = "blocked"
                                # Find the immediate failing dependencies
                                blocking_roots = [d for d in t.depends_on if d in root_failure_ids]
                                blocked_deps = [d for d in t.depends_on if d in blocked_ids]
                                if blocking_roots:
                                    t.error = f"Blocked by root failures: {blocking_roots}"
                                elif blocked_deps:
                                    t.error = f"Blocked by transitively blocked tasks: {blocked_deps}"
                                else:
                                    t.error = "Transitively blocked"
                            else:
                                # True dependency cycle
                                unmet = [d for d in t.depends_on if by_id.get(d) and by_id[d].status != "done"]
                                t.status = "failed"
                                t.error = f"Dependency cycle: waiting on {unmet}"
                    break
                else:
                    # Recovery succeeded - continue the scheduler loop
                    print(f"[orchestrator] SELF-HEALING: Recovered via {recovery_reason}, continuing...")
                    # Small delay to avoid tight loop
                    time.sleep(0.5)
                    continue

            # BACKFILL: Generate safe filler tasks to keep workers busy
            # This prevents worker idle time when primary tasks are blocked
            if available_worker_ids and not ready_tasks:
                backfill_added = maybe_generate_backfill()
                if backfill_added > 0:
                    # Re-check ready tasks after backfill
                    ready_tasks = get_ready_tasks()

            # Start ready tasks up to capacity using two-lane scheduler
            started_any = False
            while available_worker_ids and ready_tasks and len(running) < max_workers:
                # Re-check ready tasks since state may have changed
                ready_tasks = get_ready_tasks()
                if not ready_tasks:
                    break

                t = ready_tasks[0]
                worker_id = available_worker_ids.pop(0)

                t.status = "running"
                t.worker_id = worker_id

                # Assign to lane using two-lane scheduler
                lane = two_lane_scheduler.assign_to_lane(t.id)

                # Acquire named locks
                for lk in t.locks:
                    held_locks.add(str(lk))
                # Acquire path-based locks from touched_paths
                for path in t.touched_paths:
                    held_locks.add(str(path))

                # Log agent selection with lane info
                lane_stats = two_lane_scheduler.get_lane_stats()
                print(f"[orchestrator] parallel: starting {t.id} on worker {worker_id} (agent={t.agent}, lane={lane}, coding={lane_stats['coding_active']}/{lane_stats['coding_capacity']}, exec={lane_stats['executor_active']}/{lane_stats['executor_capacity']})")
                future = executor.submit(execute_task, t, worker_id)
                running[t.id] = future
                started_any = True

            # Wait for any task to complete (with timeout for heartbeat)
            if running:
                try:
                    done_futures, _ = concurrent.futures.wait(
                        running.values(), timeout=min(5.0, heartbeat_interval), return_when=concurrent.futures.FIRST_COMPLETED
                    )
                except Exception as e:
                    print(f"[orchestrator] WARNING: wait() raised: {e}")
                    time.sleep(1.0)
                    continue

                # Process completed tasks
                for future in done_futures:
                    # Find the task for this future
                    completed_tid = None
                    for tid, f in list(running.items()):
                        if f is future:
                            completed_tid = tid
                            break

                    if completed_tid:
                        del running[completed_tid]
                        t = by_id.get(completed_tid)
                        if t:
                            # Release from two-lane scheduler
                            two_lane_scheduler.release_from_lane(completed_tid)
                            # Release named locks
                            for lk in t.locks:
                                held_locks.discard(str(lk))
                            # Release path-based locks
                            for path in t.touched_paths:
                                held_locks.discard(str(path))
                            if t.worker_id is not None:
                                available_worker_ids.append(t.worker_id)

                            # Get result (might raise if task raised)
                            try:
                                future.result()
                            except Exception as e:
                                if t.status not in ("failed", "manual"):
                                    t.status = "failed"
                                    t.error = f"Future exception: {e}"
                                print(f"[orchestrator] ERROR: task {t.id} future raised: {e}")

                            # Merge if successful
                            if t.status == "done":
                                print(f"[orchestrator] parallel: {t.id} completed successfully")
                                merge_task(t)
                            else:
                                print(f"[orchestrator] parallel: {t.id} finished with status={t.status}")
            elif not started_any:
                # Nothing running and nothing started, wait a bit before retrying
                time.sleep(0.5)

    # Final verification with auto-repair loop (skip in selftest mode; respect verify_mode)
    # STALL PREVENTION: The repair callback ensures verify failures trigger automatic
    # repair actions instead of waiting for manual intervention.
    if not selftest_mode:
        verify_mode = getattr(args, "verify_mode", "strict")
        verify_json = state.runs_dir / "final_verify.json"
        repair_report_path = state.runs_dir / "verify_repair_report.json"
        max_repair_attempts = getattr(args, "max_repair_attempts", 5)

        if verify_mode == "off":
            print("[orchestrator] parallel: skipping final verification (--verify-mode=off)")
            rc_v = 0
        else:
            strict_git = (verify_mode != "skip-git")
            mode_label = "strict" if strict_git else "skip-git"
            print(f"[orchestrator] parallel: running verify auto-repair loop ({mode_label}, max_attempts={max_repair_attempts})")

            # Create repair callback for automatic execution of repair tasks
            # This prevents the orchestrator from stalling on verify failures
            repair_callback = create_repair_callback(
                project_root=state.project_root,
                runs_dir=state.runs_dir,
                verbose=True,
                scheduler_callback=None,  # No external scheduler, use internal executor
            )

            repair_result = run_verify_repair_loop(
                project_root=state.project_root,
                verify_json_path=verify_json,
                max_attempts=max_repair_attempts,
                strict_git=strict_git,
                verbose=True,
                runs_dir=state.runs_dir,
                bootstrap_on_start=True,
                agent_task_callback=repair_callback,  # CRITICAL: enables auto-repair
            )

            rc_v = repair_result.final_exit_code
            write_repair_report(repair_result, repair_report_path)

            if repair_result.success:
                print(f"[orchestrator] parallel: verify PASSED after {repair_result.total_attempts} attempt(s)")
            else:
                print(f"[orchestrator] parallel: verify FAILED after {repair_result.total_attempts} attempt(s)")
                print(f"[orchestrator] parallel: remaining failures: {repair_result.remaining_failures}")
                print(f"[orchestrator] parallel: repair report: {repair_report_path}")
                # Write final verify attempt artifact
                final_verify_artifact = state.runs_dir / f"final_verify_attempt_{repair_result.total_attempts}.json"
                if verify_json.exists():
                    shutil.copy(verify_json, final_verify_artifact)
                    print(f"[orchestrator] parallel: final verify artifact: {final_verify_artifact}")
                # Check for out-of-scope repairs that need manual intervention
                out_of_scope_path = state.runs_dir / "out_of_scope_repairs.json"
                if out_of_scope_path.exists():
                    print(f"[orchestrator] parallel: out-of-scope repairs (need manual): {out_of_scope_path}")
    else:
        rc_v = 0

    # Summary
    print("\n[orchestrator] parallel summary")
    done_count = 0
    failed_count = 0
    pending_rerun_count = 0
    blocked_count = 0
    for t in tasks:
        is_backfill = _is_backfill_task_id(t.id)
        extra = ""
        if t.status in ("failed", "manual", "pending_rerun", "resource_killed") and t.manual_path:
            extra = f" (see {t.manual_path})"
        elif t.status in ("failed", "manual", "pending_rerun", "resource_killed", "blocked") and t.error:
            extra = f" ({t.error})"
        # Mark backfill tasks as optional in printed output
        optional_suffix = " (optional)" if is_backfill else ""
        print(f"- {t.id}: {t.status}{extra}{optional_suffix}")

        # Only count non-backfill tasks for needs_continuation calculation
        if is_backfill:
            continue
        if t.status == "done":
            done_count += 1
        elif t.status in ("failed", "manual", "resource_killed"):
            failed_count += 1
        elif t.status == "pending_rerun":
            pending_rerun_count += 1
        elif t.status in ("blocked", "skipped"):
            blocked_count += 1

    if selftest_mode:
        if done_count == len(tasks):
            print("\n[orchestrator] SELFTEST PASSED: all tasks completed successfully")
            return 0
        else:
            print(f"\n[orchestrator] SELFTEST FAILED: {failed_count} task(s) failed")
            return 1

    # Safety net: ensure all manual tasks have manual_path written
    for t in tasks:
        if t.status == "manual" and t.manual_path is None:
            _mark_task_manual(
                task=t,
                reason=t.error or "Task requires manual intervention",
                manual_dir=manual_dir,
                schema_path=state.schema_path,
            )

    # Generate structured summary.json
    summary = _generate_run_summary(
        tasks=tasks,
        runs_dir=state.runs_dir,
        verify_exit_code=rc_v,
    )
    summary_path = state.runs_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[orchestrator] Summary written to: {summary_path}")

    # Generate continuation_prompt.txt if there are failures
    needs_continuation = (failed_count + pending_rerun_count + blocked_count) > 0 or rc_v != 0
    if needs_continuation:
        continuation_prompt = _generate_continuation_prompt(
            summary=summary,
            tasks=tasks,
            design_doc_text=design_doc_text,
            runs_dir=state.runs_dir,
        )
        continuation_path = state.runs_dir / "continuation_prompt.txt"
        continuation_path.write_text(continuation_prompt, encoding="utf-8")
        print(f"[orchestrator] Continuation prompt written to: {continuation_path}")

        print("\n[orchestrator] RUN INCOMPLETE:")
        print(f"  - Completed: {done_count}")
        print(f"  - Failed: {failed_count}")
        print(f"  - Pending Rerun: {pending_rerun_count}")
        print(f"  - Blocked: {blocked_count}")
        print(f"  - Verify: {'PASS' if rc_v == 0 else 'FAIL'}")
        print("\n[orchestrator] To continue, run:")
        print(f"  ./run_parallel.sh --continuation {continuation_path}")
    else:
        print(f"\n[orchestrator] RUN COMPLETE: all {done_count} tasks succeeded, verify passed")

    return 0 if rc_v == 0 and not needs_continuation else 1


# -----------------------------
# Auto-continue loop
# -----------------------------


def _generate_repair_context_for_failures(
    root_failures: list[dict[str, Any]],
    runs_dir: Path,
) -> str:
    """Generate repair context for root failures to feed into the next planning cycle.

    This enables self-healing by giving the planner specific guidance on what failed
    and how to approach fixing it.
    """
    if not root_failures:
        return ""

    lines = [
        "\n\n# REPAIR CONTEXT - CRITICAL",
        "",
        "The previous run had the following root failures that need targeted repair:",
        "",
    ]

    for failure in root_failures:
        task_id = failure.get("id", "unknown")
        title = failure.get("title", "unknown task")
        error = failure.get("error", "unknown error")
        agent = failure.get("agent", "unknown")

        lines.append(f"## REPAIR NEEDED: {task_id}")
        lines.append(f"- Original task: {title}")
        lines.append(f"- Agent: {agent}")
        lines.append(f"- Error: {error}")
        lines.append("")
        lines.append("**REPAIR GUIDANCE:**")

        # Provide specific guidance based on error type
        error_lower = error.lower() if error else ""
        if "tools" in error_lower and ("disabled" in error_lower or "cannot" in error_lower):
            lines.append("- The agent incorrectly believed tools were disabled.")
            lines.append("- Tools ARE ENABLED. Use Read, Edit, Write, Bash to implement changes.")
            lines.append("- Do NOT claim tools are unavailable.")
        elif "merge conflict" in error_lower or "conflict" in error_lower:
            lines.append("- There was a merge conflict during integration.")
            lines.append("- Ensure changes don't conflict with other parallel tasks.")
            lines.append("- Consider more granular changes that are less likely to conflict.")
        elif "work_completed=false" in error_lower or "plan" in error_lower:
            lines.append("- The agent returned work_completed=false without making changes.")
            lines.append("- IMPLEMENT the changes directly. Do NOT just plan.")
            lines.append("- Set work_completed=true after implementing.")
        elif "json" in error_lower or "validation" in error_lower:
            lines.append("- JSON output was invalid.")
            lines.append("- Ensure output is a valid JSON object matching the schema.")
            lines.append("- No markdown fences. No prose before/after JSON.")
        else:
            lines.append("- Analyze the error and implement a fix.")
            lines.append("- If the task is too complex, break it into smaller sub-tasks.")

        lines.append("")

    lines.append("## INSTRUCTIONS FOR THIS RUN")
    lines.append("")
    lines.append("1. Generate a repair-focused task plan that addresses the above failures.")
    lines.append("2. Each repair task should be atomic and independently verifiable.")
    lines.append("3. Include targeted test coverage for repaired functionality.")
    lines.append("4. Prioritize unblocking dependent tasks.")
    lines.append("")

    return "\n".join(lines)


def _run_parallel_with_auto_continue(
    *,
    args: argparse.Namespace,
    config: RunConfig,
    project_root: Path,
    schema_path: Path,
    system_prompt_path: Path,
    stats_ids: list[str],
    stats_id_set: set,
    system_prompt: str,
) -> int:
    """Run the parallel runner with auto-continue until success or max runs.

    This function implements SELF-HEALING behavior:
    - Does NOT immediately give up on planning failures (rc==2)
    - Generates repair context for failed tasks to guide subsequent runs
    - Has a high tolerance for stalled runs (max_stalled=10) before escalation
    - Continues burning credits to maximize chance of success
    """
    max_runs = max(1, args.max_continuation_runs)
    run_count = 0
    last_summary: dict[str, Any] | None = None
    prev_root_failure_ids: set = set()
    stalled_count = 0
    # SELF-HEALING: Increased from 3 to 10 - we prefer burning credits to giving up early
    max_stalled = 10
    planning_failure_count = 0
    max_planning_failures = 3  # Retry planning up to 3 times before escalating

    print(f"[orchestrator] AUTO-CONTINUE MODE (SELF-HEALING): max_runs={max_runs}, max_stalled={max_stalled}")

    while run_count < max_runs:
        run_count += 1
        print(f"\n{'=' * 80}")
        print(f"[orchestrator] AUTO-CONTINUE: Starting run {run_count}/{max_runs}")
        print(f"{'=' * 80}\n")

        # Create new run state for this iteration
        run_id = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        runs_dir = project_root / "runs" / run_id
        _ensure_dir(runs_dir)

        # Update agent policy with runs_dir for violation artifacts
        get_agent_policy().runs_dir = runs_dir

        # Handle continuation from previous run
        design_doc_path = (project_root / args.design_doc).resolve()
        if run_count > 1 and last_summary:
            # Use continuation prompt from previous run
            prev_runs_dir = Path(last_summary.get("run_dir", ""))
            continuation_path = prev_runs_dir / "continuation_prompt.txt"
            if continuation_path.exists():
                # Append continuation context to design doc
                continuation_text = continuation_path.read_text(encoding="utf-8")
                design_doc_text = design_doc_path.read_text(encoding="utf-8") if design_doc_path.exists() else ""
                augmented_doc = design_doc_text + "\n\n---\n\n" + continuation_text
                augmented_path = runs_dir / "augmented_design_doc.md"
                augmented_path.write_text(augmented_doc, encoding="utf-8")
                # Temporarily override design doc path
                original_design_doc = args.design_doc
                args.design_doc = str(augmented_path.relative_to(project_root))

        _git_init_if_needed(project_root)
        if not args.no_agent_branch:
            _checkout_agent_branch(project_root, run_id)

        state = RunState(
            run_id=run_id,
            project_root=project_root,
            runs_dir=runs_dir,
            schema_path=schema_path,
            system_prompt_path=system_prompt_path,
            design_doc_path=(project_root / args.design_doc).resolve(),
            smoke_route=config.smoke_route,
            readonly=getattr(args, "readonly", False),
        )

        # Run parallel
        rc = run_parallel(
            args=args,
            config=config,
            state=state,
            stats_ids=stats_ids,
            stats_id_set=stats_id_set,
            system_prompt=system_prompt,
        )

        # Restore design doc path if we modified it
        if run_count > 1 and "original_design_doc" in dir():
            args.design_doc = original_design_doc

        # Check result
        summary_path = runs_dir / "summary.json"
        if summary_path.exists():
            last_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            last_summary = None

        if rc == 0:
            print(f"\n[orchestrator] AUTO-CONTINUE: SUCCESS after {run_count} run(s)")
            return 0

        # SELF-HEALING: Planning failures (rc == 2) get retried with repair context
        # We do NOT immediately give up - we burn credits to maximize success chance
        if rc == 2:
            planning_failure_count += 1
            print(f"\n[orchestrator] AUTO-CONTINUE: Planning failure (rc=2), attempt {planning_failure_count}/{max_planning_failures}")

            if planning_failure_count >= max_planning_failures:
                print(f"\n[orchestrator] AUTO-CONTINUE: ESCALATING - Planning failed {planning_failure_count} times")
                print("[orchestrator] This may be a structural issue with the design document.")
                # Print debug file locations for the user
                plan_debug = runs_dir / "task_plan.extract_debug.txt"
                raw_stream = runs_dir / "task_plan.json.wrapper_schema_claude_raw_stream.txt"
                if plan_debug.exists():
                    print(f"[orchestrator] Debug file: {plan_debug}")
                if raw_stream.exists():
                    print(f"[orchestrator] Raw stream: {raw_stream}")
                print("[orchestrator] Check the files above to diagnose the planning failure.")
                _write_escalation_file(runs_dir, {"root_failures": [{"id": "PLANNING", "title": "Planning Step", "error": "Planning step failed repeatedly", "agent": "unknown"}]})
                return 2

            # SELF-HEALING: Generate repair context for planning issues
            print("[orchestrator] AUTO-CONTINUE: Retrying planning with simplified context...")
            # Continue to next iteration - the repair context will help
            continue

        # Check for progress - compare root failures
        if last_summary:
            current_root_failure_ids = {t["id"] for t in last_summary.get("root_failures", [])}

            if current_root_failure_ids == prev_root_failure_ids:
                stalled_count += 1
                print(f"[orchestrator] AUTO-CONTINUE: No progress (same failures). Stalled count: {stalled_count}/{max_stalled}")

                # SELF-HEALING: Generate repair context for stalled failures
                root_failures = last_summary.get("root_failures", [])
                if root_failures:
                    repair_context = _generate_repair_context_for_failures(root_failures, runs_dir)
                    repair_path = runs_dir / "repair_context.md"
                    repair_path.write_text(repair_context, encoding="utf-8")
                    print(f"[orchestrator] AUTO-CONTINUE: Generated repair context at {repair_path}")

                if stalled_count >= max_stalled:
                    print(f"\n[orchestrator] AUTO-CONTINUE: ESCALATING - no progress for {stalled_count} consecutive runs")
                    print("[orchestrator] Root failures that could not be resolved automatically:")
                    for t in last_summary.get("root_failures", []):
                        print(f"  - {t['id']}: {t.get('error', 'unknown')}")
                    _write_escalation_file(runs_dir, last_summary)
                    return 1
            else:
                stalled_count = 0
                planning_failure_count = 0  # Reset planning failures on progress
                print("[orchestrator] AUTO-CONTINUE: Progress detected (different failures)")

            prev_root_failure_ids = current_root_failure_ids
        else:
            stalled_count += 1

        print(f"[orchestrator] AUTO-CONTINUE: Run {run_count} incomplete, continuing with self-healing...")

    print(f"\n[orchestrator] AUTO-CONTINUE: GIVING UP - max runs ({max_runs}) reached")
    if last_summary:
        _write_escalation_file(project_root / "runs" / "escalation", last_summary)
    return 1


def _write_escalation_file(runs_dir: Path, summary: dict[str, Any]) -> None:
    """Write an escalation file explaining what's stuck and how to proceed."""
    _ensure_dir(runs_dir)
    escalation_path = runs_dir / "ESCALATION_REQUIRED.md"

    lines = [
        "# Manual Escalation Required",
        "",
        "The auto-continue loop could not resolve the following failures:",
        "",
    ]

    for t in summary.get("root_failures", []):
        lines.append(f"## {t['id']}: {t.get('title', 'Unknown')}")
        lines.append(f"- Status: {t['status']}")
        lines.append(f"- Agent: {t['agent']}")
        if t.get("error"):
            lines.append(f"- Error: {t['error']}")
        if t.get("manual_path"):
            lines.append(f"- Manual file: {t['manual_path']}")
        lines.append("")
        lines.append("### To run this task manually:")
        lines.append("```bash")
        lines.append(f"./run_parallel.sh --only-task {t['id']} --max-workers 1 --allow-resource-intensive")
        lines.append("```")
        lines.append("")

    lines.extend(
        [
            "## Next Steps",
            "",
            "1. Review the manual files in runs/*/manual/",
            "2. Run failing tasks individually with --only-task",
            "3. Once fixed, re-run the full parallel run",
            "",
        ]
    )

    escalation_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[orchestrator] Escalation file written to: {escalation_path}")


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--config", default="bridge/config.json")
    ap.add_argument("--schema", default="bridge/turn.schema.json")
    ap.add_argument("--system-prompt", default="bridge/prompts/system.md")
    ap.add_argument("--design-doc", default="DESIGN_DOCUMENT.md")
    ap.add_argument(
        "--design-doc-contract",
        choices=["strict", "loose", "off"],
        default="loose",
        help="Design doc contract validation mode: strict (fail if missing fields), "
        "loose (warn but continue), off (no validation)",
    )
    ap.add_argument(
        "--milestone-id",
        type=str,
        default=None,
        help="Override milestone ID from design doc (required if strict mode and parser can't infer)",
    )
    ap.add_argument("--start-agent", choices=AGENTS, default="codex")
    ap.add_argument(
        "--smoke-route",
        type=_parse_smoke_route_arg,
        default=None,
        help="Comma-separated agent route for smoke tests (e.g. 'codex,claude').",
    )
    ap.add_argument("--mode", choices=["live", "mock"], default="mock")
    ap.add_argument("--runner", choices=["sequential", "parallel"], default="sequential")
    # Parallel runner knobs (safe defaults; can override on CLI)
    ap.add_argument("--task-plan-schema", default="bridge/task_plan.schema.json")
    ap.add_argument("--max-workers", type=int, default=0, help="0=auto")
    ap.add_argument("--only-task", action="append", default=[], help="Run only specific task id(s) from the generated plan")
    ap.add_argument(
        "--allow-resource-intensive", action="store_true", help="Do not auto-kill tasks that exceed resource thresholds"
    )
    ap.add_argument(
        "--planner-profile",
        choices=["balanced", "throughput"],
        default="balanced",
        help="Planner profile: 'balanced' (default) for conservative planning, "
        "'throughput' for many small parallel tasks to maximize worker utilization",
    )
    ap.add_argument("--cpu-threshold", type=float, default=0.0, help="Override CPU%% threshold for auto-stop (0=use config)")
    ap.add_argument("--mem-threshold", type=float, default=0.0, help="Override RAM%% threshold for auto-stop (0=use config)")
    ap.add_argument("--terminal-max-bytes", type=int, default=0, help="Override per-worker terminal output cap (0=use config)")
    ap.add_argument(
        "--terminal-max-line-len", type=int, default=0, help="Override per-worker terminal line length cap (0=use config)"
    )
    ap.add_argument("--mock-scenario", default="bridge/mock_scenarios/milestone_demo.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--stream-agent-output",
        choices=["stdout", "stderr", "both", "none"],
        default="none",
        help="Stream agent output to console (stdout, stderr, both, none)",
    )
    ap.add_argument("--no-agent-branch", action="store_true")
    ap.add_argument(
        "--readonly",
        action="store_true",
        help="Force read-only mode: ignore needs_write_access and keep WRITE_ACCESS=0.",
    )
    ap.add_argument(
        "--selftest-parallel",
        action="store_true",
        help="Run selftest mode: synthetic tasks with trivial commands to verify scheduler",
    )
    # Auto-continue support
    # Default: enabled if ORCH_AUTO_CONTINUE=1 is set, otherwise disabled
    auto_continue_default = os.environ.get("ORCH_AUTO_CONTINUE", "").strip() in ("1", "true", "yes")
    ap.add_argument(
        "--auto-continue",
        action="store_true",
        default=auto_continue_default,
        help="Automatically continue with new runs until all tasks complete or max runs reached. "
        "Also enabled by ORCH_AUTO_CONTINUE=1 environment variable.",
    )
    ap.add_argument("--no-auto-continue", action="store_true", help="Disable auto-continue even if ORCH_AUTO_CONTINUE is set")
    ap.add_argument(
        "--max-continuation-runs", type=int, default=5, help="Maximum number of continuation runs before giving up (default: 5)"
    )
    ap.add_argument(
        "--continuation",
        type=str,
        default="",
        help="Path to continuation_prompt.txt from a previous run (use instead of design doc)",
    )

    # Agent-only flags for hard enforcement
    ap.add_argument(
        "--only-codex", action="store_true", help="ONLY use Codex agent for ALL operations (planner, workers, fallbacks)"
    )
    ap.add_argument(
        "--only-claude", action="store_true", help="ONLY use Claude agent for ALL operations (planner, workers, fallbacks)"
    )

    # Robustness flags for unattended runs
    ap.add_argument(
        "--auto-stash",
        action="store_true",
        help="Auto-stash uncommitted changes before starting (prints stash ref, does NOT auto-pop)",
    )
    ap.add_argument(
        "--verify-mode",
        choices=["strict", "skip-git", "off"],
        default="strict",
        help="Verification mode: strict (fail on dirty repo), skip-git (skip git checks during run), off (no verification)",
    )
    ap.add_argument(
        "--force-dirty",
        action="store_true",
        help="Allow starting with dirty repo (DANGEROUS: may cause verify stalls, use with --verify-mode=skip-git)",
    )
    ap.add_argument(
        "--max-repair-attempts",
        type=int,
        default=3,
        help="Maximum verify auto-repair attempts before giving up (default: 3)",
    )

    args = ap.parse_args()

    # Validate mutually exclusive agent flags
    if args.only_codex and args.only_claude:
        print("[orchestrator] ERROR: --only-codex and --only-claude are mutually exclusive")
        sys.exit(1)

    readonly_env = os.environ.get("ORCH_READONLY", "").strip().lower() in ("1", "true", "yes")
    args.readonly = args.readonly or readonly_env
    if args.readonly and not args.no_agent_branch:
        print("[orchestrator] READONLY: enabling --no-agent-branch")
        args.no_agent_branch = True

    # Handle --no-auto-continue override
    if args.no_auto_continue:
        args.auto_continue = False

    project_root = Path(args.project_root).resolve()
    config = load_config(project_root / args.config)
    config = dataclasses.replace(config, smoke_route=tuple(args.smoke_route or ()))

    if args.readonly and (args.runner == "parallel" or args.selftest_parallel):
        print("[orchestrator] ERROR: --readonly is only supported for sequential runner; use worktree isolation for parallel.")
        return 2

    # Initialize agent policy based on --only-* flags
    forced_agent: str | None = None
    if args.only_codex:
        forced_agent = "codex"
    elif args.only_claude:
        forced_agent = "claude"

    if forced_agent:
        # Override start-agent to match forced agent
        args.start_agent = forced_agent
        print(f"[orchestrator] AGENT POLICY: --only-{forced_agent} active. ALL operations will use {forced_agent} only.")

    # Note: runs_dir is set later, so we'll update the policy then
    policy = AgentPolicy(
        forced_agent=forced_agent,
        allowed_agents=tuple(config.enable_agents) if config.enable_agents else AGENTS,
    )
    set_agent_policy(policy)

    schema_path = project_root / args.schema
    system_prompt_path = project_root / args.system_prompt
    design_doc_path = (project_root / args.design_doc).resolve()

    run_id = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    runs_dir = project_root / "runs" / run_id
    _ensure_dir(runs_dir)

    # Update agent policy with runs_dir for violation artifacts
    get_agent_policy().runs_dir = runs_dir

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
        smoke_route=config.smoke_route,
        readonly=args.readonly,
    )

    stats_md_path = project_root / "STATS.md"
    stats_md = _read_text(stats_md_path) if stats_md_path.exists() else ""
    stats_ids = _extract_stats_ids(stats_md)
    stats_id_set = set(stats_ids)

    system_prompt = _read_text(system_prompt_path) if system_prompt_path.exists() else ""

    if args.runner == "parallel" or args.selftest_parallel:
        if not args.selftest_parallel and args.mode != "live":
            print("[orchestrator] ERROR: parallel runner supports --mode live only.")
            return 2

        # Auto-continue loop
        if args.auto_continue:
            return _run_parallel_with_auto_continue(
                args=args,
                config=config,
                project_root=project_root,
                schema_path=schema_path,
                system_prompt_path=system_prompt_path,
                stats_ids=stats_ids,
                stats_id_set=stats_id_set,
                system_prompt=system_prompt,
            )
        else:
            return run_parallel(
                args=args, config=config, state=state, stats_ids=stats_ids, stats_id_set=stats_id_set, system_prompt=system_prompt
            )

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

    agent = args.start_agent
    if config.smoke_route:
        if agent != config.smoke_route[0]:
            print(
                f"[orchestrator] SMOKE ROUTE: overriding --start-agent {agent} -> {config.smoke_route[0]}"
            )
        agent = config.smoke_route[0]

    print(f"[orchestrator] run_id={state.run_id} mode={args.mode} project_root={project_root}")
    print(
        f"[orchestrator] limits: max_calls_per_agent={config.max_calls_per_agent} "
        f"quota_retry_attempts={config.quota_retry_attempts} max_total_calls={config.max_total_calls}"
    )

    call_no = 0

    # Tracks consecutive JSON parse/validation failures for the current agent.
    correction_agent: str | None = None
    correction_attempts = 0

    while state.total_calls < config.max_total_calls:
        if state.call_counts.get(agent, 0) >= config.max_calls_per_agent:
            fb = _pick_fallback(config, state, current_agent=agent)
            if not fb:
                print("[orchestrator] ERROR: all agents exhausted call cap; stopping.")
                return 2
            print(f"[orchestrator] OVERRIDE: agent '{agent}' exceeded call cap; switching to '{fb}'")
            agent = fb

        call_no += 1
        state.total_calls += 1
        state.call_counts[agent] += 1

        call_dir = runs_dir / "calls" / f"call_{call_no:04d}"
        _ensure_dir(call_dir)

        design_doc_text = _read_text(design_doc_path) if design_doc_path.exists() else ""
        milestone_id = _parse_milestone_id(design_doc_text)

        verify_json = call_dir / "verify.json"
        if os.environ.get("FF_SKIP_VERIFY") == "1":
            verify_report_text = "{}"
        else:
            verify_rc, verify_out, verify_err = _run_verify(project_root, verify_json, strict_git=False)
            if verify_rc != 0 and not verify_json.exists():
                # Ensure something is present for embedding.
                _write_text(verify_json, json.dumps({"error": (verify_out + "\n" + verify_err).strip()}))
            verify_report_text = _read_text(verify_json) if verify_json.exists() else "{}"

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
            readonly=state.readonly,
        )

        prompt_path = call_dir / "prompt.txt"
        raw_path = call_dir / "raw.txt"
        out_path = call_dir / "out.json"
        _write_text(prompt_path, prompt_text)

        print("=" * 88)
        model = config.agent_models.get(agent, "(default)")
        effective_write_access = state.grant_write_access and not state.readonly
        print(
            f"CALL {call_no:04d} | agent={agent} | total_calls={state.total_calls} | "
            f"agent_calls={state.call_counts[agent]}/{config.max_calls_per_agent} | "
            f"write_access={'1' if effective_write_access else '0'} | "
            f"readonly={'1' if state.readonly else '0'}"
        )
        print(f"[orchestrator] TURN (agent={agent} model={model})")

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
                stream_agent_output=args.stream_agent_output,
                call_dir=call_dir,
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
                fb = _pick_fallback(config, state, current_agent=agent)
                print(f"[orchestrator] FAILOVER: switching to '{fb}' with same work item")
                agent = fb
                continue

            fb = _pick_fallback(config, state, current_agent=agent)
            print(f"[orchestrator] FAILOVER: switching to '{fb}' after error")
            agent = fb
            continue

        # Parse and validate JSON using TurnNormalizer for robustness.
        try:
            out_for_parse = out
            if out_path.exists() and out_path.stat().st_size > 0:
                out_for_parse = _read_text(out_path)

            # Use TurnNormalizer for robust payload extraction
            norm_result = normalize_agent_output(
                out_for_parse,
                expected_agent=agent,
                expected_milestone_id=milestone_id,
                stats_id_set=stats_id_set,
                default_phase="implement",
            )

            if norm_result.success and norm_result.turn:
                turn_obj = norm_result.turn
                # Log normalization warnings (auto-corrections)
                for warning in norm_result.warnings:
                    print(f"[orchestrator] NORMALIZED: {warning}")

                # Use lenient validation (auto-corrects mismatches with warnings)
                ok, msg, val_warnings = _validate_turn_lenient(
                    turn_obj,
                    expected_agent=agent,
                    expected_milestone_id=milestone_id,
                    stats_id_set=stats_id_set,
                )
                for warning in val_warnings:
                    print(f"[orchestrator] VALIDATION: {warning}")
                if not ok:
                    raise ValueError(msg)
            else:
                # Normalization failed - try legacy direct parsing
                turn_obj = _try_parse_json(out_for_parse)
                ok, msg, val_warnings = _validate_turn_lenient(
                    turn_obj,
                    expected_agent=agent,
                    expected_milestone_id=milestone_id,
                    stats_id_set=stats_id_set,
                )
                for warning in val_warnings:
                    print(f"[orchestrator] VALIDATION: {warning}")
                if not ok:
                    raise ValueError(msg)
        except Exception as e:
            print(f"[orchestrator] JSON INVALID: {e}")

            if correction_agent != agent:
                correction_agent = agent
                correction_attempts = 0
            correction_attempts += 1

            if correction_attempts <= config.max_json_correction_attempts:
                print(
                    f"[orchestrator] correction attempt {correction_attempts}/{config.max_json_correction_attempts} (same agent)"
                )
                prev_prompt = next_prompt
                next_prompt = textwrap.dedent(
                    f"""
                    Your previous response could not be parsed/validated as a single JSON object matching bridge/turn.schema.json.

                    Error: {e}

                    Return ONLY one JSON object (no prose, no markdown, no code fences) that satisfies the schema.
                    Re-answer the ORIGINAL PROMPT below:

                    --- ORIGINAL PROMPT START ---
                    {prev_prompt}
                    --- ORIGINAL PROMPT END ---
                    """
                ).strip()
                continue

            correction_agent = None
            correction_attempts = 0

            fb = _pick_fallback(config, state, current_agent=agent)
            print(f"[orchestrator] FAILOVER: switching to '{fb}'")
            agent = fb
            continue

        # Successful parse/validate.
        correction_agent = None
        correction_attempts = 0

        turn_path = call_dir / "turn.json"
        _write_text(turn_path, json.dumps(turn_obj, indent=2, sort_keys=True))
        state.history.append(turn_obj)

        print(f"[orchestrator] summary: {turn_obj['summary']}")

        # Update write-access grant for next call (readonly mode blocks escalation).
        requested_write_access = bool(turn_obj.get("needs_write_access", False))
        if state.readonly and requested_write_access:
            print("[orchestrator] READONLY: ignoring needs_write_access request")
        state.grant_write_access = requested_write_access and not state.readonly

        # Completion gates: only stop if they actually pass.
        if bool(turn_obj["project_complete"]):
            ok, msg = _completion_gates_ok(project_root)
            if ok:
                print("[orchestrator] PROJECT COMPLETE (gates passed)")
                state_path = runs_dir / "state.json"
                _write_text(state_path, json.dumps(dataclasses.asdict(state), indent=2, sort_keys=True, default=str))
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
            heuristic_agent = "claude" if "git status not clean" in msg or "committed" in msg else "codex"
            agent = get_agent_policy().enforce(heuristic_agent, "project completion failure heuristic")
            continue

        # Decide next agent.
        requested = str(turn_obj["next_agent"])
        effective, override_reason = _override_next_agent(requested, config, state)
        if override_reason:
            print(f"[orchestrator] OVERRIDE next_agent: {requested} -> {effective} ({override_reason})")
        else:
            print(f"[orchestrator] next_agent={effective} (as requested)")

        next_prompt = str(turn_obj["next_prompt"]).strip()
        agent = effective

    print("[orchestrator] ERROR: max_total_calls exceeded")
    return 6


if __name__ == "__main__":
    raise SystemExit(main())
