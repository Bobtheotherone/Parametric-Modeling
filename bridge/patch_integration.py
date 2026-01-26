#!/usr/bin/env python3
"""Patch-based integration for orchestrator workers.

This module implements patch-artifact integration where workers never need to
commit changes directly. Instead, the orchestrator collects all changes as
patches and applies them centrally.

Key benefits:
- Workers never write to .git/worktrees/*/index.lock
- No sandbox permission issues
- Centralized conflict resolution
- Atomic integration with deterministic commits

Additionally, this module provides ScopeGuard for anti-drift enforcement:
- Validates patches against allowlist/denylist patterns
- Rejects patches that touch out-of-scope files
- Writes rejected_patch artifacts for audit
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# =============================================================================
# Default scope configuration for orchestrator repair tasks
# =============================================================================

DEFAULT_DENYLIST: tuple[str, ...] = (
    "src/**",
    "tools/**",
    "docs/**",
    "DESIGN_DOCUMENT.md",
    "*.md",  # Most markdown files
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    ".github/**",
    ".gitignore",
)

DEFAULT_ALLOWLIST: tuple[str, ...] = (
    "bridge/**",
    "tests/test_orchestrator*.py",
    "tests/test_verify_repair*.py",
)

# Files that are always allowed regardless of denylist
ALWAYS_ALLOWED: tuple[str, ...] = ("bridge/**",)

# =============================================================================
# User-owned files - NEVER modify these
# These files are manually controlled by humans and must not be touched by agents
# =============================================================================

USER_OWNED_FILES: tuple[str, ...] = ("DESIGN_DOCUMENT.md",)

# =============================================================================
# Backfill task scope configuration
# Backfill tasks (FILLER-*) are limited to safe directories to prevent conflicts
# =============================================================================

BACKFILL_ALLOWLIST: tuple[str, ...] = (
    "tests/**",
    "docs/**",
    # NOTE: bridge/** removed from default backfill allowlist for safety
    # Use --backfill-allow-bridge to opt-in
    ".github/**",
)

BACKFILL_DENYLIST: tuple[str, ...] = (
    "src/**",
    "coupongen/**",
    "tools/**",
    "*.toml",
    "*.cfg",
    # Hot integration files that shouldn't be touched by backfill
    "**/api.py",
    "**/board_writer.py",
    "**/cli_main.py",
    "**/pipeline.py",
)

# =============================================================================
# Orchestrator core files - excluded from backfill unless explicitly allowed
# These files are critical for orchestrator stability
# =============================================================================

ORCHESTRATOR_CORE_FILES: tuple[str, ...] = (
    "bridge/loop.py",
    "bridge/design_doc.py",
    "bridge/patch_integration.py",
    "bridge/merge_resolver.py",
    "bridge/scheduler.py",
    "bridge/loop_pkg/**",
    "DESIGN_DOCUMENT.md",
)

# Extended backfill allowlist (when --backfill-allow-bridge is used)
BACKFILL_ALLOWLIST_WITH_BRIDGE: tuple[str, ...] = (
    "tests/**",
    "docs/**",
    "bridge/**",
    ".github/**",
)

# Extended backfill denylist (blocks orchestrator core even when bridge is allowed)
BACKFILL_DENYLIST_CORE: tuple[str, ...] = BACKFILL_DENYLIST + ORCHESTRATOR_CORE_FILES


def is_backfill_task(task_id: str) -> bool:
    """Check if a task ID indicates a backfill (filler) task."""
    return task_id.startswith("FILLER-")


def normalize_diff_path(path: str) -> str:
    """Normalize a path from git diff output to a clean relative path.

    Git diff paths often include "a/" or "b/" prefixes (e.g., "a/bridge/loop.py").
    This function safely removes those prefixes without accidentally stripping
    characters from the actual path.

    IMPORTANT: Uses removeprefix(), NOT lstrip(), to avoid bugs like:
    - lstrip("a/") on "a/bridge/foo.py" would produce "ridge/foo.py" (WRONG!)
    - removeprefix("a/") on "a/bridge/foo.py" produces "bridge/foo.py" (CORRECT!)

    Args:
        path: A file path, potentially with git diff prefixes

    Returns:
        Clean relative path without leading prefixes
    """
    # Normalize path separators first
    path = path.replace("\\", "/")

    # Remove git diff prefixes using safe removeprefix (Python 3.9+)
    # Order matters: check "a/" and "b/" prefixes that git adds to diff output
    if path.startswith("a/"):
        path = path[2:]  # Equivalent to removeprefix("a/")
    elif path.startswith("b/"):
        path = path[2:]  # Equivalent to removeprefix("b/")

    # Also handle leading "./" (current directory prefix)
    if path.startswith("./"):
        path = path[2:]

    # Handle edge cases with multiple leading slashes
    path = path.lstrip("/")

    return path


def is_user_owned_file(path: str) -> bool:
    """Check if a path is a user-owned file that agents must never modify.

    Args:
        path: File path to check

    Returns:
        True if this is a user-owned file
    """
    normalized = normalize_diff_path(path)
    return normalized in USER_OWNED_FILES or any(fnmatch.fnmatch(normalized, pattern) for pattern in USER_OWNED_FILES)


def create_backfill_scope_guard(
    runs_dir: Path | None = None,
    allow_bridge: bool = False,
) -> ScopeGuard:
    """Create a scope guard for backfill tasks with restricted allowlist.

    Args:
        runs_dir: Directory for writing rejected patch artifacts
        allow_bridge: If True, allows bridge/** files (excluding core files).
                     Default is False for maximum safety.

    Returns:
        ScopeGuard configured for backfill task constraints
    """
    if allow_bridge:
        # Allow bridge/** but still block orchestrator core files
        # Use ignore_always_allowed to enforce denylist over ALWAYS_ALLOWED
        return ScopeGuard(
            allowlist=BACKFILL_ALLOWLIST_WITH_BRIDGE,
            denylist=BACKFILL_DENYLIST_CORE,
            runs_dir=runs_dir,
            ignore_always_allowed=True,  # Enforce denylist even for bridge/**
        )
    else:
        # Default: most restrictive - only tests/docs
        return ScopeGuard(
            allowlist=BACKFILL_ALLOWLIST,
            denylist=BACKFILL_DENYLIST,
            runs_dir=runs_dir,
            ignore_always_allowed=True,  # Backfill guards need strict constraints
        )


@dataclass
class ScopeViolation:
    """Represents a single scope violation in a patch."""

    path: str
    reason: str  # "denylist_match" or "not_in_allowlist"
    matched_pattern: str


@dataclass
class ScopeCheckResult:
    """Result of scope validation for a patch."""

    allowed: bool
    violations: list[ScopeViolation] = field(default_factory=list)
    checked_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "allowed": self.allowed,
            "violations": [
                {
                    "path": v.path,
                    "reason": v.reason,
                    "matched_pattern": v.matched_pattern,
                }
                for v in self.violations
            ],
            "checked_paths": self.checked_paths,
        }


class ScopeGuard:
    """Validates patches against allowlist/denylist for anti-drift enforcement.

    The guard uses a two-phase check:
    1. If path matches any denylist pattern AND is not in ALWAYS_ALLOWED: rejected
    2. If allowlist is provided: path must match at least one allowlist pattern

    Usage:
        guard = ScopeGuard(
            allowlist=["bridge/**", "tests/test_orchestrator*.py"],
            denylist=["src/**", "tools/**"],
        )
        result = guard.check_paths(["bridge/loop.py", "src/foo.py"])
        if not result.allowed:
            # Reject the patch
    """

    def __init__(
        self,
        allowlist: tuple[str, ...] | list[str] | None = None,
        denylist: tuple[str, ...] | list[str] | None = None,
        runs_dir: Path | None = None,
        ignore_always_allowed: bool = False,
    ) -> None:
        """Initialize scope guard.

        Args:
            allowlist: Glob patterns for allowed paths. If None, uses DEFAULT_ALLOWLIST.
            denylist: Glob patterns for denied paths. If None, uses DEFAULT_DENYLIST.
            runs_dir: Directory for writing rejected patch artifacts.
            ignore_always_allowed: If True, don't bypass checks for ALWAYS_ALLOWED paths.
                                   Used for backfill guards that need stricter constraints.
        """
        self.allowlist: tuple[str, ...] = tuple(allowlist) if allowlist else DEFAULT_ALLOWLIST
        self.denylist: tuple[str, ...] = tuple(denylist) if denylist else DEFAULT_DENYLIST
        self.runs_dir = runs_dir
        self.ignore_always_allowed = ignore_always_allowed

    def _matches_any(self, path: str, patterns: tuple[str, ...]) -> str | None:
        """Check if path matches any of the patterns.

        Returns the matched pattern or None.
        """
        # Normalize path separators
        path = path.replace("\\", "/")

        for pattern in patterns:
            # Check direct match
            if fnmatch.fnmatch(path, pattern):
                return pattern
            # Check if path starts with pattern prefix (for ** patterns)
            if pattern.endswith("/**"):
                prefix = pattern[:-3]
                if path.startswith(prefix + "/") or path == prefix:
                    return pattern
            # Check basename match (for *.ext patterns)
            if pattern.startswith("*."):
                if fnmatch.fnmatch(os.path.basename(path), pattern):
                    return pattern

        return None

    def check_paths(self, paths: list[str]) -> ScopeCheckResult:
        """Check if all paths are within allowed scope.

        Args:
            paths: List of file paths to check (relative to project root)

        Returns:
            ScopeCheckResult with allowed=True if all paths are valid
        """
        violations: list[ScopeViolation] = []

        for path in paths:
            # Normalize path using safe prefix removal (NOT lstrip!)
            # This prevents bugs like "bridge/foo.py" becoming "ridge/foo.py"
            path = normalize_diff_path(path)

            # CRITICAL: Check if this is a user-owned file FIRST
            # User-owned files (like DESIGN_DOCUMENT.md) must NEVER be modified
            if is_user_owned_file(path):
                violations.append(
                    ScopeViolation(
                        path=path,
                        reason="user_owned_file",
                        matched_pattern="USER_OWNED_FILES",
                    )
                )
                continue

            # Check denylist BEFORE ALWAYS_ALLOWED when ignore_always_allowed is set
            # This is used for backfill guards that need stricter constraints
            denied_pattern = self._matches_any(path, self.denylist)
            if denied_pattern:
                violations.append(
                    ScopeViolation(
                        path=path,
                        reason="denylist_match",
                        matched_pattern=denied_pattern,
                    )
                )
                continue

            # Check if in always-allowed list (unless ignore_always_allowed is set)
            if not self.ignore_always_allowed and self._matches_any(path, ALWAYS_ALLOWED):
                continue

            # Check allowlist (if provided)
            if self.allowlist:
                allowed_pattern = self._matches_any(path, self.allowlist)
                if not allowed_pattern:
                    violations.append(
                        ScopeViolation(
                            path=path,
                            reason="not_in_allowlist",
                            matched_pattern="",
                        )
                    )

        return ScopeCheckResult(
            allowed=len(violations) == 0,
            violations=violations,
            checked_paths=paths,
        )

    def check_patch_artifact(self, artifact: PatchArtifact) -> ScopeCheckResult:
        """Check if a patch artifact is within allowed scope.

        Args:
            artifact: The patch artifact to validate

        Returns:
            ScopeCheckResult with validation details
        """
        paths = [change.path for change in artifact.changes]
        paths.extend(artifact.untracked_patches.keys())
        return self.check_paths(paths)

    def write_rejected_artifact(
        self,
        task_id: str,
        agent_name: str,
        result: ScopeCheckResult,
        runs_dir: Path | None = None,
    ) -> Path | None:
        """Write a rejected patch artifact for audit.

        Args:
            task_id: The task ID that generated the patch
            agent_name: Name of the agent that generated the patch
            result: The scope check result with violations
            runs_dir: Directory to write to (uses self.runs_dir if not provided)

        Returns:
            Path to the written artifact, or None if no runs_dir
        """
        output_dir = runs_dir or self.runs_dir
        if not output_dir:
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / f"rejected_patch_{task_id}.json"

        artifact_data = {
            "task_id": task_id,
            "agent_name": agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scope_check": result.to_dict(),
            "rejection_reason": "Patch touches files outside allowed scope",
            "remediation_prompt": (
                "You attempted to modify out-of-scope files. "
                f"Only files matching these patterns are allowed: {list(self.allowlist)}. "
                f"These patterns are denied: {list(self.denylist)}. "
                "Please revise your changes to only modify allowed files."
            ),
        }

        artifact_path.write_text(json.dumps(artifact_data, indent=2), encoding="utf-8")
        return artifact_path

    def get_remediation_prompt(self, result: ScopeCheckResult) -> str:
        """Generate a prompt to send back to the agent for re-attempt.

        Args:
            result: The scope check result with violations

        Returns:
            A remediation prompt string
        """
        violation_paths = [v.path for v in result.violations]
        return (
            "ERROR: Your changes were rejected because they modify out-of-scope files.\n"
            f"Rejected paths: {violation_paths}\n\n"
            f"You may ONLY modify files matching: {list(self.allowlist)}\n"
            f"You may NOT modify files matching: {list(self.denylist)}\n\n"
            "Please revise your changes to stay within the allowed scope. "
            "Focus only on bridge/** files and orchestrator tests."
        )


@dataclass
class FileChange:
    """Represents a single file change."""

    path: str
    operation: str  # "modify", "add", "delete", "binary"
    sha256: str | None = None
    old_mode: str | None = None
    new_mode: str | None = None


@dataclass
class PatchArtifact:
    """Complete patch artifact from a worker."""

    task_id: str
    base_sha: str
    changes: list[FileChange] = field(default_factory=list)
    patch_content: str = ""
    untracked_patches: dict[str, str] = field(default_factory=dict)  # path -> patch content
    success: bool = True
    error: str | None = None


def _run_cmd(
    cmd: list[str],
    cwd: Path | str,
    env: dict[str, str] | None = None,
    check: bool = False,
) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env or os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def _compute_sha256(file_path: Path) -> str | None:
    """Compute SHA256 hash of a file."""
    try:
        if not file_path.exists() or file_path.is_dir():
            return None
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def collect_patch_artifact(
    worktree_path: Path,
    task_id: str,
    base_sha: str,
) -> PatchArtifact:
    """Collect all changes from a worker's worktree as a patch artifact.

    This function:
    1. Gets tracked modifications/deletions via `git diff --binary`
    2. Detects untracked new files and generates patches for them
    3. Creates a manifest of all changed files with their SHA256 hashes

    Args:
        worktree_path: Path to the worker's worktree
        task_id: The task ID for this work
        base_sha: The base commit SHA the worktree was created from

    Returns:
        PatchArtifact containing all changes
    """
    artifact = PatchArtifact(task_id=task_id, base_sha=base_sha)

    # Step 1: Get tracked changes (modified, deleted files)
    rc, diff_out, diff_err = _run_cmd(
        ["git", "diff", "--binary", "HEAD"],
        cwd=worktree_path,
    )
    if rc != 0:
        # Try diff against base_sha if HEAD doesn't work
        rc, diff_out, diff_err = _run_cmd(
            ["git", "diff", "--binary", base_sha],
            cwd=worktree_path,
        )

    if rc == 0 and diff_out.strip():
        artifact.patch_content = diff_out

    # Step 2: Get list of modified/deleted files for manifest
    rc, status_out, _ = _run_cmd(
        ["git", "status", "--porcelain=v1"],
        cwd=worktree_path,
    )

    if rc != 0:
        artifact.success = False
        artifact.error = f"Failed to get git status: {status_out}"
        return artifact

    # Parse status output
    # Git porcelain v1 format: "XY PATH" or "XY\tPATH" (tab for rename detection)
    # X = index status, Y = worktree status, then space/tab, then path
    for line in status_out.strip().split("\n"):
        if not line or len(line) < 3:
            continue

        status_code = line[:2]

        # Handle both space and tab delimiters (git uses tab for rename detection)
        # The delimiter is at index 2; path starts at index 3
        rest = line[2:]
        if rest.startswith(" "):
            file_path = rest[1:].strip()
        elif rest.startswith("\t"):
            file_path = rest[1:].strip()
        else:
            # Fallback: try to find first space/tab after status code
            for i, c in enumerate(rest):
                if c in " \t":
                    file_path = rest[i+1:].strip()
                    break
            else:
                # No delimiter found - use entire rest as path (shouldn't happen)
                file_path = rest.strip()

        # Handle renamed files (R status shows "old -> new" with space or tab separator)
        if " -> " in file_path:
            _, file_path = file_path.split(" -> ", 1)
        elif "\t" in file_path and status_code.startswith("R"):
            # Git can use tab to separate old and new names in rename detection
            parts = file_path.split("\t")
            if len(parts) == 2:
                file_path = parts[1]

        # VALIDATION: Detect path truncation - a valid path should not start with
        # common directory suffixes that indicate truncation
        truncation_indicators = ["ests/", "ocs/", "rc/", "idge/", "ools/"]
        for indicator in truncation_indicators:
            if file_path.startswith(indicator):
                # Log warning but don't fail - attempt to recover
                expected_prefix = {"ests/": "t", "ocs/": "d", "rc/": "s", "idge/": "br", "ools/": "t"}
                prefix = expected_prefix.get(indicator, "")
                print(f"[patch] WARNING: Possible path truncation detected: '{file_path}' - attempting recovery")
                file_path = prefix + file_path
                break

        # Strip any quotes from filenames
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]

        full_path = worktree_path / file_path

        # Determine operation type
        if status_code.startswith("D") or status_code.endswith("D"):
            operation = "delete"
            sha256 = None
        elif status_code == "??" or status_code.startswith("A"):
            operation = "add"
            sha256 = _compute_sha256(full_path)

            # For untracked/new files, generate a diff patch
            if status_code == "??":
                # Use git diff --no-index to create a patch for untracked files
                rc2, patch_out, _ = _run_cmd(
                    ["git", "diff", "--binary", "--no-index", "/dev/null", file_path],
                    cwd=worktree_path,
                )
                # git diff --no-index returns 1 when files differ (which is expected)
                if patch_out.strip():
                    artifact.untracked_patches[file_path] = patch_out
        else:
            operation = "modify"
            sha256 = _compute_sha256(full_path)

        # Detect binary files
        if full_path.exists() and not full_path.is_dir():
            try:
                with open(full_path, "rb") as f:
                    chunk = f.read(8192)
                    if b"\x00" in chunk:
                        operation = "binary"
            except Exception:
                pass

        artifact.changes.append(
            FileChange(
                path=file_path,
                operation=operation,
                sha256=sha256,
            )
        )

    return artifact


def save_patch_artifact(
    artifact: PatchArtifact,
    task_dir: Path,
) -> tuple[Path, Path]:
    """Save patch artifact to disk atomically.

    Args:
        artifact: The patch artifact to save
        task_dir: Directory to save artifacts in

    Returns:
        Tuple of (patch_file_path, manifest_file_path)
    """
    task_dir.mkdir(parents=True, exist_ok=True)

    # Save main patch file atomically
    patch_path = task_dir / "changes.patch"
    _atomic_write(patch_path, artifact.patch_content)

    # Save untracked file patches if any
    if artifact.untracked_patches:
        untracked_dir = task_dir / "untracked_patches"
        untracked_dir.mkdir(exist_ok=True)
        for file_path, patch_content in artifact.untracked_patches.items():
            safe_name = file_path.replace("/", "__").replace("\\", "__")
            patch_file = untracked_dir / f"{safe_name}.patch"
            _atomic_write(patch_file, patch_content)

    # Save manifest atomically
    manifest_path = task_dir / "changed_files.json"
    manifest = {
        "task_id": artifact.task_id,
        "base_sha": artifact.base_sha,
        "success": artifact.success,
        "error": artifact.error,
        "files": [
            {
                "path": c.path,
                "operation": c.operation,
                "sha256": c.sha256,
            }
            for c in artifact.changes
        ],
        "untracked_files": list(artifact.untracked_patches.keys()),
    }
    _atomic_write_json(manifest_path, manifest)

    return patch_path, manifest_path


def _atomic_write(path: Path, content: str) -> None:
    """Write content to a file atomically."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.rename(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON data to a file atomically."""
    content = json.dumps(data, indent=2)
    _atomic_write(path, content)


def _run_auto_ruff(
    project_root: Path,
    python_files: list[str],
) -> tuple[bool, str]:
    """Run ruff format and ruff check --fix on staged Python files.

    This automatically fixes lint/format issues during patch integration,
    reducing the need for separate lint agent runs.

    Args:
        project_root: Root of the git repository
        python_files: List of Python file paths (relative to project_root)

    Returns:
        Tuple of (success, message)
    """
    if not python_files:
        return True, "No Python files to format"

    # Check if auto-ruff is enabled (default: ON)
    auto_ruff_enabled = os.environ.get("ORCH_AUTO_RUFF", "1").lower() not in ("0", "false", "no", "off")
    if not auto_ruff_enabled:
        return True, "Auto-ruff disabled via ORCH_AUTO_RUFF=0"

    # Filter to only existing files
    existing_files = [f for f in python_files if (project_root / f).exists()]
    if not existing_files:
        return True, "No existing Python files to format"

    # Try running ruff format
    try:
        rc, out, err = _run_cmd(
            ["ruff", "format"] + existing_files,
            cwd=project_root,
        )
        if rc != 0:
            # Non-fatal: log warning and continue
            print(f"[patch] Warning: ruff format returned {rc}: {err}")
    except Exception as e:
        print(f"[patch] Warning: ruff format failed ({e}), continuing without format")

    # Try running ruff check --fix
    try:
        rc, out, err = _run_cmd(
            ["ruff", "check", "--fix"] + existing_files,
            cwd=project_root,
        )
        if rc != 0 and "error" in err.lower():
            # Non-fatal: log warning and continue
            print(f"[patch] Warning: ruff check --fix returned {rc}: {err}")
    except Exception as e:
        print(f"[patch] Warning: ruff check --fix failed ({e}), continuing without lint fixes")

    # Re-stage the files that were potentially modified
    for file_path in existing_files:
        _run_cmd(["git", "add", file_path], cwd=project_root)

    return True, f"Auto-ruff ran on {len(existing_files)} Python files"


def apply_patch_artifact(
    project_root: Path,
    task_dir: Path,
    task_id: str,
    task_branch: str,
    scope_guard: ScopeGuard | None = None,
    runs_dir: Path | None = None,
    agent_name: str = "unknown",
) -> tuple[bool, str, str | None]:
    """Apply a patch artifact to the project and create a commit.

    This function:
    1. Validates patch against scope guard (if provided)
    2. Applies the main patch using `git apply --binary`
    3. Applies untracked file patches
    4. Stages all changes
    5. Creates a deterministic commit

    Args:
        project_root: Root of the git repository
        task_dir: Directory containing the patch artifacts
        task_id: The task ID for commit message
        task_branch: The branch name for this task
        scope_guard: Optional ScopeGuard for drift prevention
        runs_dir: Directory for writing rejected patch artifacts
        agent_name: Name of the agent that created the patch

    Returns:
        Tuple of (success, message, commit_sha)
    """
    patch_path = task_dir / "changes.patch"
    manifest_path = task_dir / "changed_files.json"

    # Read manifest
    if not manifest_path.exists():
        return False, "Manifest file not found", None

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        return False, f"Failed to read manifest: {e}", None

    # Scope guard validation - CRITICAL for drift prevention
    if scope_guard is not None:
        files_to_check = [f["path"] for f in manifest.get("files", [])]
        files_to_check.extend(manifest.get("untracked_files", []))

        scope_result = scope_guard.check_paths(files_to_check)
        if not scope_result.allowed:
            # Write rejected patch artifact
            artifact_dir = runs_dir or task_dir
            scope_guard.write_rejected_artifact(
                task_id=task_id,
                agent_name=agent_name,
                result=scope_result,
                runs_dir=artifact_dir,
            )
            # Return rejection with remediation info
            violation_paths = [v.path for v in scope_result.violations]
            return (
                False,
                f"SCOPE_REJECTED: Patch touches out-of-scope files: {violation_paths}",
                None,
            )

    # Apply main patch if it exists and has content
    if patch_path.exists():
        patch_content = patch_path.read_text(encoding="utf-8")
        if patch_content.strip():
            rc, out, err = _run_cmd(
                ["git", "apply", "--binary", "--3way", str(patch_path)],
                cwd=project_root,
            )
            if rc != 0:
                # Try without 3way
                rc, out, err = _run_cmd(
                    ["git", "apply", "--binary", str(patch_path)],
                    cwd=project_root,
                )
                if rc != 0:
                    return False, f"Failed to apply patch: {err}", None

    # Apply untracked file patches
    untracked_dir = task_dir / "untracked_patches"
    if untracked_dir.exists():
        for patch_file in untracked_dir.glob("*.patch"):
            rc, out, err = _run_cmd(
                ["git", "apply", "--binary", str(patch_file)],
                cwd=project_root,
            )
            if rc != 0:
                # Non-fatal: log and continue
                print(f"[patch] Warning: Failed to apply untracked patch {patch_file.name}: {err}")

    # Stage all changes
    files_to_stage = [f["path"] for f in manifest.get("files", [])]
    files_to_stage.extend(manifest.get("untracked_files", []))

    if files_to_stage:
        # Stage specific files to avoid including unrelated changes
        for file_path in files_to_stage:
            full_path = project_root / file_path
            if full_path.exists():
                _run_cmd(["git", "add", file_path], cwd=project_root)
            elif any(f["path"] == file_path and f["operation"] == "delete" for f in manifest.get("files", [])):
                # Handle deleted files
                _run_cmd(["git", "add", file_path], cwd=project_root)

    # Auto-run ruff format and lint fix on staged Python files
    # This reduces the need for separate lint agent runs
    python_files = [f for f in files_to_stage if f.endswith(".py")]
    if python_files:
        ruff_ok, ruff_msg = _run_auto_ruff(project_root, python_files)
        if ruff_ok:
            print(f"[patch] {ruff_msg}")

    # Check if there are staged changes
    rc, status, _ = _run_cmd(
        ["git", "diff", "--cached", "--quiet"],
        cwd=project_root,
    )

    if rc == 0:
        # No staged changes
        return True, "No changes to commit", None

    # Create commit with deterministic message
    commit_msg = f"task({task_id}): integrate worker patch"
    rc, out, err = _run_cmd(
        ["git", "commit", "-m", commit_msg],
        cwd=project_root,
    )

    if rc != 0:
        return False, f"Failed to commit: {err}", None

    # Get commit SHA
    rc, sha_out, _ = _run_cmd(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
    )

    commit_sha = sha_out.strip() if rc == 0 else None

    return True, "Patch applied and committed successfully", commit_sha


def attempt_conflict_resolution(
    project_root: Path,
    task_id: str,
    patch_path: Path,
) -> tuple[bool, str]:
    """Attempt to resolve conflicts when patch application fails.

    This function:
    1. Rebases the current branch
    2. Retries patch application with 3-way merge
    3. If still failing, marks as needing manual resolution

    Args:
        project_root: Root of the git repository
        task_id: The task ID for logging
        patch_path: Path to the patch file

    Returns:
        Tuple of (success, message)
    """
    # First, try to update to latest main
    rc, out, err = _run_cmd(
        ["git", "fetch", "origin"],
        cwd=project_root,
    )

    # Try rebasing
    rc, out, err = _run_cmd(
        ["git", "rebase", "origin/main"],
        cwd=project_root,
    )

    if rc != 0:
        # Abort rebase and return
        _run_cmd(["git", "rebase", "--abort"], cwd=project_root)
        return False, f"Rebase failed: {err}"

    # Retry patch application with 3-way merge
    rc, out, err = _run_cmd(
        ["git", "apply", "--binary", "--3way", str(patch_path)],
        cwd=project_root,
    )

    if rc == 0:
        return True, "Conflict resolved after rebase"

    return False, f"Conflict resolution failed: {err}"


class PatchIntegrator:
    """High-level patch integration manager.

    This class manages the integration of patches from multiple workers,
    handling conflicts and maintaining integration order.

    With scope guard enabled, patches that touch out-of-scope files are
    rejected with a SCOPE_REJECTED status, and artifacts are written for audit.

    When merge conflicts occur, the integrator will:
    1. Try basic rebase-based conflict resolution
    2. Try agent-based intelligent conflict resolution (up to N attempts)
    3. Only fall back to manual resolution if all automated approaches fail
    """

    def __init__(
        self,
        project_root: Path,
        runs_dir: Path,
        scope_guard: ScopeGuard | None = None,
        max_merge_resolution_attempts: int = 3,
    ):
        self.project_root = project_root
        self.runs_dir = runs_dir
        self.scope_guard = scope_guard
        self.max_merge_resolution_attempts = max_merge_resolution_attempts
        self._integration_lock = None  # Set externally if threading is used

    def set_lock(self, lock) -> None:
        """Set the threading lock for integration operations."""
        self._integration_lock = lock

    def set_scope_guard(self, scope_guard: ScopeGuard) -> None:
        """Set or update the scope guard."""
        self.scope_guard = scope_guard

    def integrate_task(
        self,
        task_id: str,
        task_dir: Path,
        task_branch: str,
        agent_name: str = "unknown",
        task_context: str = "",
        milestone_id: str = "M0",
    ) -> tuple[bool, str, str | None]:
        """Integrate a task's changes into the main branch.

        Args:
            task_id: The task ID
            task_dir: Directory containing the patch artifacts
            task_branch: The branch name (for fallback if needed)
            agent_name: Name of the agent that created the patch
            task_context: Context about the task (for merge resolver)
            milestone_id: Current milestone ID

        Returns:
            Tuple of (success, message, commit_sha)
            If scope_guard rejects: (False, "SCOPE_REJECTED: ...", None)
        """
        if self._integration_lock:
            with self._integration_lock:
                return self._do_integrate(task_id, task_dir, task_branch, agent_name, task_context, milestone_id)
        else:
            return self._do_integrate(task_id, task_dir, task_branch, agent_name, task_context, milestone_id)

    def _do_integrate(
        self,
        task_id: str,
        task_dir: Path,
        task_branch: str,
        agent_name: str = "unknown",
        task_context: str = "",
        milestone_id: str = "M0",
    ) -> tuple[bool, str, str | None]:
        """Internal integration implementation."""
        # Determine which scope guard to use
        # Backfill tasks (FILLER-*) get a restricted scope to prevent conflicts
        effective_scope_guard = self.scope_guard
        if is_backfill_task(task_id):
            backfill_guard = create_backfill_scope_guard(runs_dir=self.runs_dir)
            # Use backfill guard if no custom scope guard is set, or combine with it
            if effective_scope_guard is None:
                effective_scope_guard = backfill_guard
            else:
                # For backfill tasks, the backfill restrictions take precedence
                effective_scope_guard = backfill_guard

        # First, try patch-based integration (with scope guard if enabled)
        success, message, commit_sha = apply_patch_artifact(
            self.project_root,
            task_dir,
            task_id,
            task_branch,
            scope_guard=effective_scope_guard,
            runs_dir=self.runs_dir,
            agent_name=agent_name,
        )

        if success:
            return True, message, commit_sha

        # If scope rejected, do not attempt conflict resolution - it's a policy failure
        if "SCOPE_REJECTED" in message:
            return False, message, None

        # If patch failed, try conflict resolution
        patch_path = task_dir / "changes.patch"
        if patch_path.exists():
            # Step 1: Try basic rebase-based conflict resolution
            resolved, resolve_msg = attempt_conflict_resolution(
                self.project_root,
                task_id,
                patch_path,
            )
            if resolved:
                # Retry integration (scope guard already passed on first attempt)
                return apply_patch_artifact(
                    self.project_root,
                    task_dir,
                    task_id,
                    task_branch,
                    scope_guard=self.scope_guard,
                    runs_dir=self.runs_dir,
                    agent_name=agent_name,
                )

            # Step 2: Try agent-based intelligent conflict resolution
            print(f"[patch_integration] {task_id}: Basic resolution failed, trying agent-based merge resolver")
            try:
                from bridge.merge_resolver import attempt_agent_merge_resolution

                merge_result = attempt_agent_merge_resolution(
                    project_root=self.project_root,
                    runs_dir=self.runs_dir,
                    task_id=task_id,
                    task_context=task_context,
                    milestone_id=milestone_id,
                    max_attempts=self.max_merge_resolution_attempts,
                )

                if merge_result.success:
                    print(
                        f"[patch_integration] {task_id}: Agent merge resolver succeeded after {merge_result.attempt} attempt(s)"
                    )
                    # Retry integration after agent resolution
                    return apply_patch_artifact(
                        self.project_root,
                        task_dir,
                        task_id,
                        task_branch,
                        scope_guard=self.scope_guard,
                        runs_dir=self.runs_dir,
                        agent_name=agent_name,
                    )
                else:
                    print(f"[patch_integration] {task_id}: Agent merge resolver failed: {merge_result.error}")
                    print(f"[patch_integration] {task_id}: Unresolved files: {merge_result.unresolved_files}")

            except Exception as e:
                print(f"[patch_integration] {task_id}: Agent merge resolver raised exception: {e}")

        # All automated resolution attempts failed - mark as needing manual resolution
        return False, f"needs_manual_resolution: {message}", None
