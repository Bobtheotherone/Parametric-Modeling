# Loop Testing Documentation

This document describes how to run orchestrator loop tests in worktree/readonly mode with **zero tracked repository writes**.

## Overview

The orchestrator operates on a strict separation of concerns:

- **EDIT mode**: Agents may modify code/tests/docs (commits are small and atomic)
- **TEST mode**: No repository modifications allowed; only ignored artifacts are created

Loop testing must always run in TEST mode to prevent interference with ongoing development work.

## Worktree Isolation (Recommended Approach)

Git worktrees provide complete isolation for loop testing. Each worker operates in a separate worktree created from a specific commit, ensuring the main repository state is never modified.

### How It Works

1. The orchestrator creates a dedicated worktree for each parallel worker:
   ```bash
   git worktree add -b <branch> <worktree_path> <base_sha>
   ```

2. All agent work happens inside the worktree directory (e.g., `runs/<timestamp>/worktrees/w01_<task_id>/`)

3. Changes are collected as **patch artifacts** rather than direct commits:
   - Workers never write to `.git/worktrees/*/index.lock`
   - No sandbox permission issues
   - Centralized conflict resolution by the orchestrator

4. After testing, worktrees are cleaned up:
   ```bash
   git worktree remove -f <worktree_path>
   ```

### Manual Worktree Testing

To manually test the loop without modifying the main repo:

```bash
# 1. Ensure working tree is clean
git status --porcelain  # must be empty

# 2. Create a disposable worktree from current HEAD
git worktree add -f /tmp/ff_loop_test HEAD

# 3. Run loop tests ONLY in that worktree
cd /tmp/ff_loop_test
python -m bridge.loop --parallel --max-workers=2 ...

# 4. Remove the worktree afterwards
cd -
git worktree remove -f /tmp/ff_loop_test
```

## Write Access Control

The orchestrator uses environment variables to control write access:

| Variable | Description |
|----------|-------------|
| `WRITE_ACCESS` | Set to `"1"` to enable file edits + commands; `"0"` for read-only |
| `ORCH_WRITE_ACCESS` | Legacy alias for `WRITE_ACCESS` (both are set for compatibility) |

### Readonly Mode for Agents

When `WRITE_ACCESS=0`:
- Agents receive prompts but cannot modify tracked files
- Agent wrappers (e.g., `bridge/agents/claude.sh`, `bridge/agents/codex.sh`) check this variable
- Smoke-test file writes are gated behind `WRITE_ACCESS=1`

Example from agent wrapper:
```bash
WRITE_ACCESS="${WRITE_ACCESS:-0}"

if [[ "$WRITE_ACCESS" == "1" ]]; then
  # Enable full-auto mode, smoke writes, etc.
fi
```

### Planning Phase (Readonly by Design)

During the planning phase, the orchestrator explicitly sets `WRITE_ACCESS=0`:
```python
env["WRITE_ACCESS"] = "0"
env["ORCH_WRITE_ACCESS"] = "0"
env["ORCH_SCHEMA_KIND"] = "task_plan"
```

This ensures task planning never modifies the repository.

## Patch-Based Change Collection

Rather than having workers commit directly (which can cause lock contention), the orchestrator uses a patch-based workflow:

1. **Collect changes**: After agent execution, `git diff --binary HEAD` captures all modifications
2. **Store as artifact**: Changes saved to `<task_dir>/patch.diff`
3. **Centralized integration**: The orchestrator applies patches to the main branch sequentially

This eliminates the "`.git/worktrees/*/index.lock` permission denied" class of failures.

## Environment Variables Reference

| Variable | Purpose |
|----------|---------|
| `WRITE_ACCESS` | Enable/disable file modifications (`1`/`0`) |
| `ORCH_WRITE_ACCESS` | Legacy alias for compatibility |
| `ORCH_SCHEMA_KIND` | Schema type: `turn`, `task_plan`, or `json` |
| `FF_WORKER_ID` | Worker ID in parallel mode |
| `FF_AGENT_SMOKE_DIR` | Directory for smoke-test artifacts |
| `FF_SMOKE` | Enable smoke-test mode (`1`/`0`) |

## Runs Directory Structure

All loop artifacts are written to gitignored paths:

```
runs/
└── <timestamp>/
    ├── worktrees/           # Git worktrees for each worker
    │   ├── w01_<task_id>/
    │   └── w02_<task_id>/
    ├── tasks/               # Task artifacts (prompts, outputs, patches)
    │   └── <task_id>/
    │       ├── prompt.txt
    │       ├── output.txt
    │       └── patch.diff
    ├── manual/              # Tasks requiring manual intervention
    └── final_verify.json    # Verification results
```

## Pre-Test Checklist

Before running loop tests:

1. **Ensure clean working tree**:
   ```bash
   git status --porcelain  # must be empty
   ```

2. **Verify the runs/ directory is gitignored**:
   ```bash
   grep -q "^runs/" .gitignore && echo "OK" || echo "MISSING"
   ```

3. **Use worktree isolation** for multi-agent tests

4. **Do not modify code during loop tests**; if fixes are required, exit TEST mode, commit changes, then retest

## Verification

After loop testing:

```bash
# Confirm no tracked files were modified
git status --porcelain  # must still be empty

# Check for uncommitted worktree artifacts
ls runs/  # should only contain timestamped directories
```

## Troubleshooting

### "index.lock permission denied"
This occurs when multiple processes try to write to the same git index. Solutions:
- Use the patch-based workflow (default in current orchestrator)
- Ensure workers operate in separate worktrees

### Changes appearing in main repo
Verify:
- `WRITE_ACCESS=0` is set during testing
- Worktree isolation is properly configured
- Tests are not running in the main working directory

### Stale worktrees
Clean up with:
```bash
git worktree list
git worktree remove -f <path>
git worktree prune
```
