# Orchestrator Locking and Backfill Policy

This document describes the locking and backfill strategies used by the orchestrator to prevent task collisions while maintaining high throughput.

## Key Principles

1. **As narrow as possible, as strong as necessary**: Locks should be fine-grained to allow maximum concurrency while preventing actual collisions.

2. **User-owned files are read-only**: `DESIGN_DOCUMENT.md` is manually controlled by humans and must never be modified by agents.

3. **Orchestrator core files are protected**: Critical files like `bridge/loop.py`, `bridge/patch_integration.py`, etc. are excluded from backfill to prevent destabilization.

## Path Normalization

Git diff paths use `a/` and `b/` prefixes (e.g., `a/bridge/loop.py`). The `normalize_diff_path()` function safely removes these prefixes using explicit prefix matching, NOT `lstrip()`.

**Critical Bug Prevention**: Using `lstrip("a/")` or `lstrip("b/")` would incorrectly strip characters from paths like `bridge/` (producing `ridge/`). The fix uses explicit `startswith()` checks.

## Locking Strategies

### Hot-File Locks
When multiple tasks touch known critical files (api.py, board_writer.py, etc.), a shared `hot:<filename>` lock is automatically injected.

### Overlap Locks
When multiple pending tasks declare the same file in `touched_paths`, a file-specific lock is injected (`file:<basename>:<hash>`). This prevents collisions for any shared file, not just hot files.

### Backfill Type Locks
Backfill tasks use type-based locks (`backfill:type:<task_type>`) instead of a global `backfill` lock. This allows:
- Lint tasks and doc tasks to run concurrently
- Test tasks and type hint tasks to run concurrently
- Only tasks of the same type serialize

## Backfill Scope

### Default Allowlist
- `tests/**`
- `docs/**`
- `.github/**`

Note: `bridge/**` is excluded by default for safety.

### Extended Allowlist (with `--backfill-allow-bridge`)
- `tests/**`
- `docs/**`
- `bridge/**` (excluding core files)
- `.github/**`

### Always Excluded (Orchestrator Core)
These files are NEVER allowed for backfill, even with `--backfill-allow-bridge`:
- `bridge/loop.py`
- `bridge/patch_integration.py`
- `bridge/scheduler.py`
- `bridge/design_doc.py`
- `bridge/merge_resolver.py`
- `bridge/loop_pkg/**`
- `DESIGN_DOCUMENT.md`

## Rejection Cooldown

When a backfill patch is scope-rejected, the rejected path is tracked with a cooldown period (5 cycles). This prevents spam loops where the same failing task type is repeatedly generated.

## Testing

See `tests/test_orchestrator_selfheal.py` for comprehensive tests covering:
- Path normalization regression tests
- User-owned file protection
- Dynamic backfill locks
- Overlap locking
- Concurrent backfill scheduling
- Orchestrator core exclusion
