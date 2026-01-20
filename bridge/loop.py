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
import dataclasses
import datetime as dt
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import traceback
import collections
import queue
import signal
import time
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# When run as `python bridge/loop.py`, Python sets sys.path[0] to `bridge/`.
# We want to import sibling packages (e.g. `tools`) from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AGENTS: Tuple[str, ...] = ("codex", "claude")


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
class RunConfig:
    max_calls_per_agent: int
    quota_retry_attempts: int
    max_total_calls: int
    max_json_correction_attempts: int
    fallback_order: List[str]
    enable_agents: List[str]

    agent_scripts: Dict[str, str]
    quota_error_patterns: Dict[str, List[str]]
    supports_write_access: Dict[str, bool]
    parallel: ParallelSettings



@dataclasses.dataclass
class RunState:
    run_id: str
    project_root: Path
    runs_dir: Path
    schema_path: Path
    system_prompt_path: Path
    design_doc_path: Path

    total_calls: int = 0
    call_counts: Dict[str, int] = dataclasses.field(default_factory=lambda: {a: 0 for a in AGENTS})
    quota_failures: Dict[str, int] = dataclasses.field(default_factory=lambda: {a: 0 for a in AGENTS})
    disabled_by_quota: Dict[str, bool] = dataclasses.field(default_factory=lambda: {a: False for a in AGENTS})
    history: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

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
    forced_agent: Optional[str] = None  # Set by --only-* flags
    allowed_agents: Tuple[str, ...] = AGENTS
    runs_dir: Optional[Path] = None  # For writing violation artifacts

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
_agent_policy: Optional[AgentPolicy] = None


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
    return (
        text[:head]
        + "\n\n[...TRUNCATED... try opening the raw log file for full output ...]\n\n"
        + text[-tail:]
    )


def _extract_stats_ids(stats_md_text: str) -> List[str]:
    """Extract stable stats identifiers from STATS.md.

    IDs are intentionally simple: CX-* and CL-* only (two-agent mode).
    """

    ids = sorted(set(re.findall(r"\b(?:CX|CL)-\d+\b", stats_md_text)))
    return ids


def _parse_milestone_id(design_doc_text: str) -> str:
    m = re.search(r"\*\*Milestone:\*\*\s*(M\d+)\b", design_doc_text)
    return m.group(1) if m else "M0"


def _parse_all_milestones(design_doc_text: str) -> List[str]:
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


def _extract_milestone_from_task_id(task_id: str, fallback: str = "M0") -> str:
    """Extract milestone prefix from task ID (e.g., 'M2-SIM-SCHEMA' -> 'M2')."""
    m = re.match(r"^(M\d+)-", task_id)
    return m.group(1) if m else fallback


def _run_cmd(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    *,
    stream: bool = False,
) -> Tuple[int, str, str]:
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

    out_chunks: List[str] = []
    err_chunks: List[str] = []

    def _pump(src, sink, chunks: List[str]) -> None:
        try:
            assert src is not None
            for line in iter(src.readline, ""):
                sink.write(line)
                sink.flush()
                chunks.append(line)
        finally:
            try:
                src.close()
            except Exception:
                pass

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

def _total_ram_bytes() -> Optional[int]:
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


def _ps_list_pids_in_pgid(pgid: int) -> List[int]:
    """Return pids in a process group via ps (portable-ish)."""
    candidates = [
        ["ps", "-o", "pid=", "-g", str(pgid)],
        ["ps", "-o", "pid=", "--pgid", str(pgid)],
    ]
    for cmd in candidates:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            pids: List[int] = []
            for tok in out.split():
                tok = tok.strip()
                if tok.isdigit():
                    pids.append(int(tok))
            if pids:
                return sorted(set(pids))
        except Exception:
            continue
    return []


def _ps_sample_pids(pids: List[int]) -> Tuple[float, int]:
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
    kill_reason: Optional[str]
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
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass

    t0 = time.monotonic()
    while time.monotonic() - t0 < max(0.5, grace_s * 0.5):
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    # Second: SIGTERM
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    t1 = time.monotonic()
    while time.monotonic() - t1 < max(0.5, grace_s * 0.4):
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    # Last resort: SIGKILL
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _run_cmd_monitored(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
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
    kill_reason: Optional[str] = None
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
        prev_cpu_time: Optional[float] = None
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
                out_line = out_line[:terminal_max_line_length] + '...\n'

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
                try:
                    src.close()
                except Exception:
                    pass

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

    def safe(cmd: List[str]) -> str:
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

    agent_scripts: Dict[str, str] = {}
    quota_pats: Dict[str, List[str]] = {}
    supports_write: Dict[str, bool] = {}

    for a in AGENTS:
        if a not in agents_cfg:
            raise ValueError(f"Missing agent in config: {a}")
        agent_scripts[a] = str(agents_cfg[a].get("script", ""))
        quota_pats[a] = list(agents_cfg[a].get("quota_error_patterns", []))
        supports_write[a] = bool(agents_cfg[a].get("supports_write_access", False))

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
        agent_scripts=agent_scripts,
        quota_error_patterns=quota_pats,
        supports_write_access=supports_write,
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
    history: List[Dict[str, Any]],
    next_prompt: str,
    call_counts: Dict[str, int],
    disabled_by_quota: Dict[str, bool],
    stats_ids: List[str],
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

    # Get agent policy header if in forced mode
    policy_header = get_agent_policy().get_prompt_header()

    parts: List[str] = [
        system_prompt.strip(),
    ]

    # Inject policy header right after system prompt if in forced mode
    if policy_header:
        parts.append("\n\n")
        parts.append(policy_header.strip())

    parts.extend([
        "\n\n---\n\n# Orchestrator State (read-only)\n",
        state_blob,
        "\n\n---\n\n# Milestone Spec (DESIGN_DOCUMENT.md)\n",
        _truncate(design_doc_text.strip(), 20000),
        "\n\n---\n\n# Verification Report (tools.verify)\n",
        _truncate(verify_report_text.strip(), 20000),
        "\n\n---\n\n# Repo Snapshot\n",
        _truncate(repo_info.strip(), 20000),
        "\n\n---\n\n",
    ])

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


def _validate_turn(obj: Any, *, expected_agent: str, expected_milestone_id: Optional[str] = None, stats_id_set: set[str]) -> Tuple[bool, str]:
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
# Agent selection + quota
# -----------------------------


def _is_quota_error(agent: str, text: str, config: RunConfig) -> bool:
    pats = config.quota_error_patterns.get(agent, [])
    return any(re.search(p, text, flags=re.IGNORECASE) for p in pats)


def _pick_fallback(config: RunConfig, state: RunState, current_agent: Optional[str]) -> str:
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


def _other_agent(agent: str) -> Optional[str]:
    if agent not in AGENTS:
        return None
    return "claude" if agent == "codex" else "codex"


def _override_next_agent(requested: str, config: RunConfig, state: RunState) -> Tuple[str, Optional[str]]:
    policy = get_agent_policy()

    # In forced mode, always return the forced agent
    if policy.forced_agent:
        forced = policy.enforce(requested, "next_agent override")
        if forced != requested:
            return forced, f"agent policy (--only-{policy.forced_agent})"
        return forced, None

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


def _checkout_agent_branch(project_root: Path, run_id: str) -> None:
    branch = f"agent-run/{run_id}"
    rc, _, _ = _run_cmd(["git", "checkout", "-b", branch], cwd=project_root, env=os.environ.copy())
    if rc != 0:
        _run_cmd(["git", "checkout", branch], cwd=project_root, env=os.environ.copy())


def _run_verify(project_root: Path, out_json: Path, strict_git: bool) -> Tuple[int, str, str]:
    cmd = [sys.executable, "-m", "tools.verify", "--json", str(out_json)]
    if strict_git:
        cmd.append("--strict-git")
    return _run_cmd(cmd, cwd=project_root, env=os.environ.copy())


def _completion_gates_ok(project_root: Path) -> Tuple[bool, str]:
    """The repo is 'complete' when strict verify passes and git status is clean."""

    env = os.environ.copy()
    rc, out, err = _run_cmd([sys.executable, "-m", "tools.verify", "--strict-git"], cwd=project_root, env=env)
    if rc != 0:
        return False, (out + "\n" + err).strip()

    rc2, porcelain, err2 = _run_cmd(["git", "status", "--porcelain=v1"], cwd=project_root, env=env)
    if rc2 != 0:
        return False, (porcelain + "\n" + err2).strip()
    if porcelain.strip():
        return False, "git status not clean"

    return True, "ok"


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
) -> Tuple[int, str, str]:
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
    # Support both names (some wrappers read WRITE_ACCESS, older versions used ORCH_WRITE_ACCESS).
    env["WRITE_ACCESS"] = "1" if state.grant_write_access else "0"
    env["ORCH_WRITE_ACCESS"] = env["WRITE_ACCESS"]
    # Signal to the agent wrapper that this is a turn schema (enables turn normalization)
    env["ORCH_SCHEMA_KIND"] = "turn"

    return _run_cmd(cmd, cwd=state.project_root, env=env, stream=True)


def _run_agent_mock(*, agent: str, scenario: Dict[str, Any], mock_indices: Dict[str, int]) -> Tuple[int, str, str]:
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
    locks: List[str] = dataclasses.field(default_factory=list)
    depends_on: List[str] = dataclasses.field(default_factory=list)
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
    worker_id: Optional[int] = None
    branch: Optional[str] = None
    base_sha: Optional[str] = None
    worktree_path: Optional[Path] = None
    task_dir: Optional[Path] = None
    prompt_path: Optional[Path] = None
    out_path: Optional[Path] = None
    raw_log_path: Optional[Path] = None
    manual_path: Optional[Path] = None
    error: Optional[str] = None
    max_cpu_pct_total: float = 0.0
    max_mem_pct_total: float = 0.0
    # Turn output tracking
    work_completed: Optional[bool] = None
    commit_sha: Optional[str] = None
    turn_summary: Optional[str] = None
    # Retry tracking for plan-only responses
    retry_count: int = 0
    max_retries: int = 2


def _sanitize_branch_fragment(text: str) -> str:
    frag = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    frag = frag.strip("-.")
    return frag or "task"


def _collect_machine_info() -> Dict[str, Any]:
    cores = os.cpu_count() or 1
    total_ram = _total_ram_bytes()
    return {
        "cpu_cores": int(cores),
        "ram_bytes": int(total_ram) if total_ram else None,
    }


def _build_task_plan_prompt(
    *,
    design_doc_text: str,
    milestone_id: str,
    max_workers_limit: int,
    cpu_threshold_pct_total: float,
    mem_threshold_pct_total: float,
    machine_info: Dict[str, Any],
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
MULTI-MILESTONE MODE: The design document contains {len(all_milestones)} milestone documents: {', '.join(all_milestones)}.

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
        - RAM (GB): {ram_gb if ram_gb is not None else 'unknown'}

        Parallel safety policy:
        - Tasks that touch the same subsystem or files should share a lock key in `locks`.
        - If a task might require a heavy local command (full test suite, large build, large format/lint, GPU/ML workloads), set `solo: true` and `intensity: "high"`.
        - The runner will automatically stop any single task if it uses > {cpu_threshold_pct_total:.1f}% CPU or > {mem_threshold_pct_total:.1f}% RAM for multiple samples.

        Concurrency:
        - Choose `max_parallel_tasks` <= {max_workers_limit}.
        - Prefer fewer parallel tasks if you believe tasks are likely to conflict or require heavy commands.
{agent_assignment_instruction}
        {milestone_instruction}
        """
    ).strip()

    return "\n\n".join(
        [
            header,
            "---\n\n# Design Document\n" + design_excerpt,
        ]
    )


def _build_parallel_task_prompt(
    *,
    system_prompt: str,
    task: ParallelTask,
    worker_id: int,
    milestone_id: str,
    repo_snapshot: str,
    design_doc_text: str,
    resource_policy: Dict[str, Any],
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
        - Commit your changes when done.

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

    parts.extend([
        "---\n\n# Orchestrator State\n" + json.dumps(state_blob, indent=2),
        "---\n\n# Task\n" + task.description.strip(),
        "---\n\n# Repo Snapshot\n" + repo_snapshot.strip(),
        "---\n\n# Design Doc (truncated)\n" + design_excerpt.strip(),
        "---\n\n" + instructions,
    ])

    return "\n\n".join(parts)


def _select_only_tasks(all_tasks: List[ParallelTask], only_ids: List[str]) -> List[ParallelTask]:
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


def _write_manual_task_file(
    *,
    manual_dir: Path,
    task: ParallelTask,
    reason: str,
    agent_cmd: List[str],
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


def _generate_run_summary(
    *,
    tasks: List[ParallelTask],
    runs_dir: Path,
    verify_exit_code: int,
) -> Dict[str, Any]:
    """Generate a structured summary of the parallel run."""
    root_failures = []
    blocked_tasks = []
    completed_tasks = []
    pending_rerun_tasks = []

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

    summary = {
        "run_dir": str(runs_dir),
        "total_tasks": len(tasks),
        "completed": len(completed_tasks),
        "failed": len([t for t in tasks if t.status in ("failed", "manual", "resource_killed")]),
        "pending_rerun": len(pending_rerun_tasks),
        "blocked": len(blocked_tasks),
        "verify_exit_code": verify_exit_code,
        "success": verify_exit_code == 0 and len(root_failures) == 0,
        "completed_tasks": completed_tasks,
        "root_failures": root_failures,
        "blocked_tasks": blocked_tasks,
    }

    return summary


def _generate_continuation_prompt(
    *,
    summary: Dict[str, Any],
    tasks: List[ParallelTask],
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
            if t['error']:
                lines.append(f"- Error: {t['error']}")
            if t['turn_summary']:
                lines.append(f"- Last summary: {t['turn_summary'][:300]}")
            if t['manual_path']:
                lines.append(f"- Manual file: {t['manual_path']}")
            if t['raw_log_path']:
                # Include tail of log
                try:
                    log_path = Path(t['raw_log_path'])
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

    lines.extend([
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
    ])

    return "\n".join(lines)


def _compute_transitive_blocked(
    tasks: List[ParallelTask],
    root_failure_ids: set,
) -> set:
    """Compute the transitive closure of all tasks blocked by root failures.

    A task is blocked if:
    - It depends directly on a root failure
    - It depends on another blocked task (transitive)

    Returns the set of task IDs that are transitively blocked.
    """
    by_id = {t.id: t for t in tasks}
    blocked_ids: set = set()

    # Build reverse dependency graph
    dependents: Dict[str, List[str]] = {t.id: [] for t in tasks}
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

        # Validate JSON output (STRICT - unparseable or missing output is FAILED)
        if not out_path.exists():
            task.status = "failed"
            task.error = "Agent did not produce output file"
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

        turn_text = out_path.read_text(encoding="utf-8")
        turn_obj = _try_parse_json(turn_text)
        if turn_obj is None:
            task.status = "failed"
            task.error = f"Agent output is not valid JSON: {turn_text[:500]}"
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

        ok, msg = _validate_turn(
            turn_obj,
            expected_agent=agent,
            expected_milestone_id=milestone_id,
            stats_id_set=stats_id_set,
        )
        if not ok:
            task.status = "failed"
            task.error = f"Invalid JSON output: {msg}"
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

        # Ensure a commit exists (auto-commit any uncommitted changes)
        rc, porcelain, err = _run_cmd(["git", "status", "--porcelain=v1"], cwd=task.worktree_path, env=os.environ.copy())
        has_changes = rc == 0 and bool(porcelain.strip())
        if has_changes:
            _run_cmd(["git", "add", "-A"], cwd=task.worktree_path, env=os.environ.copy())
            _run_cmd(
                ["git", "commit", "-m", f"task({task.id}): auto commit"],
                cwd=task.worktree_path,
                env=os.environ.copy(),
            )
            # Get commit SHA
            rc_sha, sha_out, _ = _run_cmd(["git", "rev-parse", "HEAD"], cwd=task.worktree_path, env=os.environ.copy())
            if rc_sha == 0:
                task.commit_sha = sha_out.strip()

        # CRITICAL: Task is only "done" if work_completed==true
        if task.work_completed:
            task.status = "done"
            task.error = None
            return

        # Check if there were actual code changes committed despite work_completed=false
        if task.commit_sha and task.commit_sha != task.base_sha:
            # Agent made changes but claimed work_completed=false - trust the commit, mark done
            print(f"[orchestrator] NOTE: {task.id} has work_completed=false but made commits; marking done")
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
    cmd = [sys.executable, "-c", f"import time; print('selftest {task.id} started'); time.sleep(0.5); print('selftest {task.id} done')"]

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
    stats_ids: List[str],
    stats_id_set: set,
    system_prompt: str,
) -> int:
    machine_info = _collect_machine_info()
    selftest_mode = getattr(args, "selftest_parallel", False)

    # In selftest mode, we use synthetic tasks
    if selftest_mode:
        print("[orchestrator] SELFTEST MODE: using synthetic tasks (no real agents)")
        tasks: List[ParallelTask] = [
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
        design_doc_text = _read_text(state.design_doc_path) if state.design_doc_path.exists() else ""
        if not design_doc_text.strip():
            print(f"[orchestrator] ERROR: design doc not found: {state.design_doc_path}")
            return 2

        milestone_id = _parse_milestone_id(design_doc_text)

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

        # Safety cap hint: default to half cores (e.g. 8 on a 16-core machine), up to 16.
        cores = machine_info.get("cpu_cores") or 1
        safe_cap = min(16, max(1, int(cores) // 2))

        plan_prompt = _build_task_plan_prompt(
            design_doc_text=design_doc_text,
            milestone_id=milestone_id,
            max_workers_limit=safe_cap,
            cpu_threshold_pct_total=cpu_thr,
            mem_threshold_pct_total=mem_thr,
            machine_info=machine_info,
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

        rc, _, err = _run_cmd(
            [str(planner_script), str(plan_prompt_path), str(plan_schema_path), str(plan_out_path)],
            cwd=state.project_root,
            env=env,
            stream=True,
        )
        if rc != 0 or not plan_out_path.exists():
            print("[orchestrator] ERROR: planning step failed")
            if err:
                print(err)
            return 2

        plan_obj = _try_parse_json(plan_out_path.read_text(encoding="utf-8"))
        if not isinstance(plan_obj, dict):
            print("[orchestrator] ERROR: could not parse task_plan.json")
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
                    depends_on=list(t.get("depends_on", []) or []),
                    solo=bool(t.get("solo", False)),
                )
            )

        tasks = _select_only_tasks(tasks, args.only_task)

        if not tasks:
            print("[orchestrator] ERROR: no tasks in plan")
            return 2

        # Final worker count
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

    print(f"[orchestrator] parallel: max_workers={max_workers} (safe_cap={safe_cap})")

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
    running: Dict[str, concurrent.futures.Future] = {}  # task_id -> future
    git_lock = threading.Lock()

    def locks_available(t: ParallelTask) -> bool:
        return not (set(map(str, t.locks)) & held_locks)

    def can_start(t: ParallelTask) -> bool:
        if t.status != "pending":
            return False
        if not deps_satisfied(t):
            return False
        if t.solo and running:
            return False
        if not locks_available(t):
            return False
        return True

    def get_ready_tasks() -> List[ParallelTask]:
        return [t for t in tasks if can_start(t)]

    # Merge helper
    def merge_task(t: ParallelTask) -> bool:
        if selftest_mode or not t.branch:
            return True
        with git_lock:
            rc, out, err = _run_cmd(
                ["git", "merge", "--no-ff", "--no-edit", t.branch],
                cwd=state.project_root,
                env=os.environ.copy(),
                stream=True,
            )
            if rc == 0:
                return True
            # Abort merge if conflicted
            _run_cmd(["git", "merge", "--abort"], cwd=state.project_root, env=os.environ.copy())
            t.status = "manual"
            t.error = "Merge conflict; manual rebase/resolve required"
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

            # Heartbeat logging
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_interval:
                print(f"[orchestrator] parallel: progress done={done_count} running={running_count} queued={pending_count} ready={len(ready_tasks)}")
                last_heartbeat = now

            # Check completion - terminal states end the run
            terminal_states = ("done", "failed", "manual", "blocked", "pending_rerun", "resource_killed")
            all_finished = all(t.status in terminal_states for t in tasks)
            if all_finished:
                break

            # Detect stuck state: no tasks running, no tasks ready, but tasks remain pending
            if not running and not ready_tasks and pending_count > 0:
                # Identify root failures (tasks that failed on their own, not due to dependencies)
                root_failure_ids = {t.id for t in tasks if is_root_failure(t)}
                root_failures = [t for t in tasks if t.id in root_failure_ids]

                # Compute transitive closure of all blocked tasks
                blocked_ids = _compute_transitive_blocked(tasks, root_failure_ids)
                pending_but_not_blocked = [t for t in tasks if t.status == "pending" and t.id not in blocked_ids]

                print(f"\n[orchestrator] STUCK: {pending_count} task(s) cannot proceed")

                if root_failures:
                    print(f"\n  ROOT FAILURES ({len(root_failures)}):")
                    for t in root_failures:
                        reason = t.error or t.status
                        manual_hint = f" -> {t.manual_path}" if t.manual_path else ""
                        print(f"    - {t.id}: {reason}{manual_hint}")

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

            # Start ready tasks up to capacity
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
                for lk in t.locks:
                    held_locks.add(str(lk))

                # Log agent selection explicitly
                agent_script = config.agent_scripts.get(t.agent, "(unknown)")
                print(f"[orchestrator] parallel: starting {t.id} on worker {worker_id} (agent={t.agent}, script={agent_script})")
                future = executor.submit(execute_task, t, worker_id)
                running[t.id] = future
                started_any = True

            # Wait for any task to complete (with timeout for heartbeat)
            if running:
                try:
                    done_futures, _ = concurrent.futures.wait(
                        running.values(),
                        timeout=min(5.0, heartbeat_interval),
                        return_when=concurrent.futures.FIRST_COMPLETED
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
                            # Release locks and worker
                            for lk in t.locks:
                                held_locks.discard(str(lk))
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

    # Final verification (skip in selftest mode)
    if not selftest_mode:
        verify_json = state.runs_dir / "final_verify.json"
        print("[orchestrator] parallel: running tools.verify (strict)")
        rc_v, _, _ = _run_verify(state.project_root, verify_json, strict_git=True)
    else:
        rc_v = 0

    # Summary
    print("\n[orchestrator] parallel summary")
    done_count = 0
    failed_count = 0
    pending_rerun_count = 0
    blocked_count = 0
    for t in tasks:
        extra = ""
        if t.status in ("failed", "manual", "pending_rerun", "resource_killed") and t.manual_path:
            extra = f" (see {t.manual_path})"
        elif t.status in ("failed", "manual", "pending_rerun", "resource_killed", "blocked") and t.error:
            extra = f" ({t.error})"
        print(f"- {t.id}: {t.status}{extra}")
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

        print(f"\n[orchestrator] RUN INCOMPLETE:")
        print(f"  - Completed: {done_count}")
        print(f"  - Failed: {failed_count}")
        print(f"  - Pending Rerun: {pending_rerun_count}")
        print(f"  - Blocked: {blocked_count}")
        print(f"  - Verify: {'PASS' if rc_v == 0 else 'FAIL'}")
        print(f"\n[orchestrator] To continue, run:")
        print(f"  ./run_parallel.sh --continuation {continuation_path}")
    else:
        print(f"\n[orchestrator] RUN COMPLETE: all {done_count} tasks succeeded, verify passed")

    return 0 if rc_v == 0 and not needs_continuation else 1

# -----------------------------
# Auto-continue loop
# -----------------------------


def _run_parallel_with_auto_continue(
    *,
    args: argparse.Namespace,
    config: RunConfig,
    project_root: Path,
    schema_path: Path,
    system_prompt_path: Path,
    stats_ids: List[str],
    stats_id_set: set,
    system_prompt: str,
) -> int:
    """Run the parallel runner with auto-continue until success or max runs."""
    max_runs = max(1, args.max_continuation_runs)
    run_count = 0
    last_summary: Optional[Dict[str, Any]] = None
    prev_root_failure_ids: set = set()
    stalled_count = 0
    max_stalled = 3  # Stop if same failures for N consecutive runs

    print(f"[orchestrator] AUTO-CONTINUE MODE: max_runs={max_runs}")

    while run_count < max_runs:
        run_count += 1
        print(f"\n{'='*80}")
        print(f"[orchestrator] AUTO-CONTINUE: Starting run {run_count}/{max_runs}")
        print(f"{'='*80}\n")

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
        if run_count > 1 and 'original_design_doc' in dir():
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

        # Check for progress - compare root failures
        if last_summary:
            current_root_failure_ids = {t["id"] for t in last_summary.get("root_failures", [])}

            if current_root_failure_ids == prev_root_failure_ids:
                stalled_count += 1
                print(f"[orchestrator] AUTO-CONTINUE: No progress (same failures). Stalled count: {stalled_count}/{max_stalled}")
                if stalled_count >= max_stalled:
                    print(f"\n[orchestrator] AUTO-CONTINUE: GIVING UP - no progress for {stalled_count} consecutive runs")
                    print(f"[orchestrator] Root failures that cannot be resolved automatically:")
                    for t in last_summary.get("root_failures", []):
                        print(f"  - {t['id']}: {t.get('error', 'unknown')}")
                    _write_escalation_file(runs_dir, last_summary)
                    return 1
            else:
                stalled_count = 0
                print(f"[orchestrator] AUTO-CONTINUE: Progress detected (different failures)")

            prev_root_failure_ids = current_root_failure_ids
        else:
            stalled_count += 1

        print(f"[orchestrator] AUTO-CONTINUE: Run {run_count} incomplete, continuing...")

    print(f"\n[orchestrator] AUTO-CONTINUE: GIVING UP - max runs ({max_runs}) reached")
    if last_summary:
        _write_escalation_file(project_root / "runs" / "escalation", last_summary)
    return 1


def _write_escalation_file(runs_dir: Path, summary: Dict[str, Any]) -> None:
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
        if t.get('error'):
            lines.append(f"- Error: {t['error']}")
        if t.get('manual_path'):
            lines.append(f"- Manual file: {t['manual_path']}")
        lines.append("")
        lines.append("### To run this task manually:")
        lines.append("```bash")
        lines.append(f"./run_parallel.sh --only-task {t['id']} --max-workers 1 --allow-resource-intensive")
        lines.append("```")
        lines.append("")

    lines.extend([
        "## Next Steps",
        "",
        "1. Review the manual files in runs/*/manual/",
        "2. Run failing tasks individually with --only-task",
        "3. Once fixed, re-run the full parallel run",
        "",
    ])

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
    ap.add_argument("--start-agent", choices=AGENTS, default="codex")
    ap.add_argument("--mode", choices=["live", "mock"], default="mock")
    ap.add_argument("--runner", choices=["sequential", "parallel"], default="sequential")
    # Parallel runner knobs (safe defaults; can override on CLI)
    ap.add_argument("--task-plan-schema", default="bridge/task_plan.schema.json")
    ap.add_argument("--max-workers", type=int, default=0, help="0=auto")
    ap.add_argument("--only-task", action="append", default=[], help="Run only specific task id(s) from the generated plan")
    ap.add_argument("--allow-resource-intensive", action="store_true", help="Do not auto-kill tasks that exceed resource thresholds")
    ap.add_argument("--cpu-threshold", type=float, default=0.0, help="Override CPU%% threshold for auto-stop (0=use config)")
    ap.add_argument("--mem-threshold", type=float, default=0.0, help="Override RAM%% threshold for auto-stop (0=use config)")
    ap.add_argument("--terminal-max-bytes", type=int, default=0, help="Override per-worker terminal output cap (0=use config)")
    ap.add_argument("--terminal-max-line-len", type=int, default=0, help="Override per-worker terminal line length cap (0=use config)")
    ap.add_argument("--mock-scenario", default="bridge/mock_scenarios/milestone_demo.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-agent-branch", action="store_true")
    ap.add_argument("--selftest-parallel", action="store_true",
                    help="Run selftest mode: synthetic tasks with trivial commands to verify scheduler")
    # Auto-continue support
    # Default: enabled if ORCH_AUTO_CONTINUE=1 is set, otherwise disabled
    auto_continue_default = os.environ.get("ORCH_AUTO_CONTINUE", "").strip() in ("1", "true", "yes")
    ap.add_argument("--auto-continue", action="store_true", default=auto_continue_default,
                    help="Automatically continue with new runs until all tasks complete or max runs reached. "
                         "Also enabled by ORCH_AUTO_CONTINUE=1 environment variable.")
    ap.add_argument("--no-auto-continue", action="store_true",
                    help="Disable auto-continue even if ORCH_AUTO_CONTINUE is set")
    ap.add_argument("--max-continuation-runs", type=int, default=5,
                    help="Maximum number of continuation runs before giving up (default: 5)")
    ap.add_argument("--continuation", type=str, default="",
                    help="Path to continuation_prompt.txt from a previous run (use instead of design doc)")

    # Agent-only flags for hard enforcement
    ap.add_argument("--only-codex", action="store_true",
                    help="ONLY use Codex agent for ALL operations (planner, workers, fallbacks)")
    ap.add_argument("--only-claude", action="store_true",
                    help="ONLY use Claude agent for ALL operations (planner, workers, fallbacks)")

    args = ap.parse_args()

    # Validate mutually exclusive agent flags
    if args.only_codex and args.only_claude:
        print("[orchestrator] ERROR: --only-codex and --only-claude are mutually exclusive")
        sys.exit(1)

    # Handle --no-auto-continue override
    if args.no_auto_continue:
        args.auto_continue = False

    project_root = Path(args.project_root).resolve()
    config = load_config(project_root / args.config)

    # Initialize agent policy based on --only-* flags
    forced_agent: Optional[str] = None
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
            return run_parallel(args=args, config=config, state=state, stats_ids=stats_ids, stats_id_set=stats_id_set, system_prompt=system_prompt)

    # Mock scenario support.
    scenario: Dict[str, Any] = {}
    mock_indices: Dict[str, int] = {a: 0 for a in AGENTS}
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

    print(f"[orchestrator] run_id={state.run_id} mode={args.mode} project_root={project_root}")
    print(
        f"[orchestrator] limits: max_calls_per_agent={config.max_calls_per_agent} "
        f"quota_retry_attempts={config.quota_retry_attempts} max_total_calls={config.max_total_calls}"
    )

    call_no = 0

    # Tracks consecutive JSON parse/validation failures for the current agent.
    correction_agent: Optional[str] = None
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

        call_dir = runs_dir / f"call_{call_no:04d}"
        _ensure_dir(call_dir)

        design_doc_text = _read_text(design_doc_path) if design_doc_path.exists() else ""
        milestone_id = _parse_milestone_id(design_doc_text)

        verify_json = call_dir / "verify.json"
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
        )

        prompt_path = call_dir / "prompt.txt"
        raw_path = call_dir / "raw.txt"
        out_path = call_dir / "out.json"
        _write_text(prompt_path, prompt_text)

        print("=" * 88)
        print(
            f"CALL {call_no:04d} | agent={agent} | total_calls={state.total_calls} | "
            f"agent_calls={state.call_counts[agent]}/{config.max_calls_per_agent} | "
            f"write_access={'1' if state.grant_write_access else '0'}"
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
                fb = _pick_fallback(config, state, current_agent=agent)
                print(f"[orchestrator] FAILOVER: switching to '{fb}' with same work item")
                agent = fb
                continue

            fb = _pick_fallback(config, state, current_agent=agent)
            print(f"[orchestrator] FAILOVER: switching to '{fb}' after error")
            agent = fb
            continue

        # Parse and validate JSON.
        try:
            out_for_parse = out
            if out_path.exists() and out_path.stat().st_size > 0:
                out_for_parse = _read_text(out_path)

            turn_obj = _try_parse_json(out_for_parse)
            ok, msg = _validate_turn(
                turn_obj,
                expected_agent=agent,
                expected_milestone_id=milestone_id,
                stats_id_set=stats_id_set,
            )
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

        # Update write-access grant for next call.
        state.grant_write_access = bool(turn_obj.get("needs_write_access", False))

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
