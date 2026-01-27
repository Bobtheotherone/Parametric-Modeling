# Orchestrator Hardening Reference

## Max-Workers Selection Logic

Worker count is determined by a single function `compute_effective_max_workers()`:

1. **CLI override** (`--max-workers N` where N>0): Uses N directly.
2. **Auto mode** (`--max-workers 0` or unset): Uses `parallel.max_workers_default` from `bridge/config.json` (default: 10).
3. **Hard cap**: `min(parallel.max_workers_hard_cap, 32)` — configurable in `bridge/config.json`, aligned with `task_plan.schema.json` maximum.
4. **Plan cap**: If the planner requests fewer parallel tasks, the worker count is clamped down.
5. **Minimum**: Always at least 1 worker.

The old heuristic (`cores - 6, cap 12`) has been removed. Config default is used instead.

### Configuration

In `bridge/config.json`:
```json
{
  "parallel": {
    "max_workers_default": 10,
    "max_workers_hard_cap": 32,
    "force_alternation": false
  }
}
```

### Logging

The orchestrator logs the effective worker count and why:
```
[orchestrator] planner target workers: 10 (cli_override=10)
[orchestrator] parallel: max_workers=10 (cli_override=10)
```

---

## Directive File Behavior (CLAUDE.md / AGENTS.md)

### How it works

Agent behavioral directives are delivered via **working-directory files**, not prompt injection.

- `CLAUDE.md` — Primary directive for Claude agents. Placed in each worktree before Claude is invoked.
- `AGENTS.md` — General directive. Also copied to worktrees for consistency.

### Materialization rules

Before each agent invocation, `materialize_directive_file()` runs:

| Agent   | Source preference                        | Target in worktree |
|---------|------------------------------------------|--------------------|
| Claude  | `CLAUDE.md` (preferred) or `AGENTS.md`   | `CLAUDE.md`        |
| Any     | `AGENTS.md` (if exists)                  | `AGENTS.md`        |

- Copies are **atomic** (temp file + rename) and **idempotent** (skip if content matches).
- Verbose mode logs what was copied and where.

### What changed

Previously, AGENTS.md content was appended to the system prompt in engineering mode.
This has been removed. The directive file is now a working-directory file that Claude
reads directly, which provides stronger behavioral influence.

---

## Sequential Alternation

Two-agent alternation (codex <-> claude) is now **configurable**:

- `parallel.force_alternation: true` — Default for balanced/throughput profiles. Forces agent switch each turn.
- `parallel.force_alternation: false` — Default for engineering profile (set in `bridge/config.json`). Agents keep their task without forced handoff.

This prevents pointless context-switching in engineering mode where each task should be owned end-to-end.
