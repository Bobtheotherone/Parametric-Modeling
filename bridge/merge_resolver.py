#!/usr/bin/env python3
"""Agent-driven merge conflict resolver.

This module provides intelligent merge conflict resolution using Claude
to understand and resolve conflicts while preserving intent from both sides.

Key features:
- Detects conflict files via git status (UU paths)
- Creates a merge-resolution context for Claude
- Runs a Claude repair pass that resolves conflict markers
- Retries after resolution with bounded attempts
- Falls back to manual only after N attempts fail
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import textwrap
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConflictFile:
    """Represents a file with merge conflicts."""

    path: str
    content: str
    conflict_count: int
    ours_content: str  # Content from our branch
    theirs_content: str  # Content from their branch


@dataclass
class MergeResolutionResult:
    """Result of a merge resolution attempt."""

    success: bool
    resolved_files: list[str] = field(default_factory=list)
    unresolved_files: list[str] = field(default_factory=list)
    error: str | None = None
    attempt: int = 0
    agent_output: str = ""


def _run_cmd(
    cmd: list[str],
    cwd: Path | str,
    env: dict[str, str] | None = None,
    timeout: int = 60,
) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env or os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def detect_conflict_files(project_root: Path) -> list[str]:
    """Detect files with merge conflicts via git status.

    Returns list of file paths with UU (unmerged) status.
    """
    rc, stdout, _ = _run_cmd(["git", "status", "--porcelain=v1"], cwd=project_root)
    if rc != 0:
        return []

    conflict_files = []
    for line in stdout.strip().split("\n"):
        if not line or len(line) < 3:
            continue
        status = line[:2]
        file_path = line[3:].strip()
        # UU = unmerged, both modified (conflict)
        # AA = unmerged, both added
        # DD = unmerged, both deleted
        if status in ("UU", "AA", "UD", "DU"):
            conflict_files.append(file_path)

    return conflict_files


def parse_conflict_file(project_root: Path, file_path: str) -> ConflictFile | None:
    """Parse a file with merge conflicts and extract conflict regions.

    Returns ConflictFile with extracted ours/theirs content.
    """
    full_path = project_root / file_path
    if not full_path.exists():
        return None

    try:
        content = full_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    # Count conflict markers
    conflict_count = content.count("<<<<<<< ")
    if conflict_count == 0:
        return None

    # Extract ours and theirs sections (simplified extraction)
    ours_parts = []
    theirs_parts = []

    conflict_pattern = re.compile(r"<<<<<<< ([^\n]+)\n(.*?)=======\n(.*?)>>>>>>> ([^\n]+)", re.DOTALL)

    for match in conflict_pattern.finditer(content):
        ours_parts.append(match.group(2))
        theirs_parts.append(match.group(3))

    return ConflictFile(
        path=file_path,
        content=content,
        conflict_count=conflict_count,
        ours_content="\n".join(ours_parts),
        theirs_content="\n".join(theirs_parts),
    )


def build_merge_resolution_prompt(
    conflicts: list[ConflictFile],
    task_context: str = "",
    milestone_id: str = "M0",
) -> str:
    """Build a prompt for Claude to resolve merge conflicts."""

    conflict_sections = []
    for cf in conflicts:
        conflict_sections.append(f"""
### File: {cf.path} ({cf.conflict_count} conflict(s))

```
{cf.content}
```

The conflict markers are:
- `<<<<<<< HEAD` or `<<<<<<< ours`: Start of our version
- `=======`: Separator
- `>>>>>>> theirs` or similar: End of their version

Your task: Resolve this conflict by combining BOTH versions intelligently:
- Preserve functionality from BOTH sides
- Don't lose any code changes from either side
- Make the result syntactically valid and functional
""")

    return textwrap.dedent(f"""
        # Merge Conflict Resolution Task

        You are resolving merge conflicts in a codebase. Your goal is to produce
        a clean, working version of each conflicted file that preserves the intent
        and changes from BOTH sides of the merge.

        ## Context
        {task_context}

        ## Conflicts to Resolve
        {"".join(conflict_sections)}

        ## Instructions

        1. For each conflicted file, produce the RESOLVED content.
        2. Remove all conflict markers (<<<<<<, =======, >>>>>>>).
        3. Combine changes intelligently - don't just pick one side.
        4. Ensure the result is syntactically valid.
        5. If you cannot resolve a conflict safely, explain why.

        ## Output Format

        Output a JSON object with this structure:
        {{
            "resolutions": [
                {{
                    "path": "path/to/file.py",
                    "resolved_content": "full resolved file content here",
                    "notes": "brief explanation of how you resolved conflicts"
                }}
            ],
            "unresolvable": [
                {{
                    "path": "path/to/other.py",
                    "reason": "why this conflict cannot be safely auto-resolved"
                }}
            ]
        }}

        Output ONLY the JSON object, no markdown fences or extra text.
    """).strip()


AgentRunnerCallback = Callable[
    [list[ConflictFile], str, str, int],  # conflicts, task_context, milestone_id, attempt
    dict[str, Any],  # resolution JSON output
]


class MergeResolver:
    """Agent-driven merge conflict resolver.

    This resolver uses Claude to intelligently resolve merge conflicts,
    preserving intent from both sides of the merge.

    For testing, an agent_runner callback can be injected to avoid calling real agents.
    """

    def __init__(
        self,
        project_root: Path,
        runs_dir: Path,
        max_attempts: int = 3,
        agent_script: str = "bridge/agents/claude.sh",
        agent_runner: AgentRunnerCallback | None = None,
    ):
        """Initialize merge resolver.

        Args:
            project_root: Path to the git repository
            runs_dir: Directory for resolution artifacts
            max_attempts: Maximum resolution attempts
            agent_script: Path to the agent script (relative to project_root)
            agent_runner: Optional callback for testing. If provided, this is called
                         instead of invoking the real agent script. The callback
                         receives (conflicts, task_context, milestone_id, attempt)
                         and should return a resolution JSON dict.
        """
        self.project_root = project_root
        self.runs_dir = runs_dir
        self.max_attempts = max_attempts
        self.agent_script = agent_script
        self.agent_runner = agent_runner  # For dependency injection in tests
        self._lock = threading.Lock()

    def resolve_conflicts(
        self,
        task_id: str = "",
        task_context: str = "",
        milestone_id: str = "M0",
    ) -> MergeResolutionResult:
        """Attempt to resolve all merge conflicts in the repository.

        Args:
            task_id: ID of the task that caused the conflicts
            task_context: Additional context about what the merge was trying to do
            milestone_id: Current milestone ID

        Returns:
            MergeResolutionResult with resolution status
        """
        with self._lock:
            return self._resolve_conflicts_impl(task_id, task_context, milestone_id)

    def _resolve_conflicts_impl(
        self,
        task_id: str,
        task_context: str,
        milestone_id: str,
    ) -> MergeResolutionResult:
        """Internal implementation of conflict resolution."""

        # Detect conflicted files
        conflict_paths = detect_conflict_files(self.project_root)
        if not conflict_paths:
            return MergeResolutionResult(
                success=True,
                resolved_files=[],
                unresolved_files=[],
                error=None,
                attempt=0,
            )

        print(f"[merge_resolver] Detected {len(conflict_paths)} conflicted file(s): {conflict_paths}")

        # Parse conflict files
        conflicts = []
        for path in conflict_paths:
            cf = parse_conflict_file(self.project_root, path)
            if cf:
                conflicts.append(cf)

        if not conflicts:
            # Files have UU status but no conflict markers - try simple git add
            for path in conflict_paths:
                _run_cmd(["git", "add", path], cwd=self.project_root)
            return MergeResolutionResult(
                success=True,
                resolved_files=conflict_paths,
                unresolved_files=[],
                attempt=0,
            )

        # Try agent-based resolution with bounded attempts
        for attempt in range(1, self.max_attempts + 1):
            print(f"[merge_resolver] Attempt {attempt}/{self.max_attempts}")

            result = self._run_agent_resolution(conflicts, task_context, milestone_id, attempt)

            if result.success:
                return result

            # Check if there are still conflicts after this attempt
            remaining = detect_conflict_files(self.project_root)
            if not remaining:
                result.success = True
                return result

            print(f"[merge_resolver] {len(remaining)} conflict(s) remain after attempt {attempt}")

        # All attempts failed
        return MergeResolutionResult(
            success=False,
            resolved_files=[],
            unresolved_files=conflict_paths,
            error=f"Failed to resolve conflicts after {self.max_attempts} attempts",
            attempt=self.max_attempts,
        )

    def _run_agent_resolution(
        self,
        conflicts: list[ConflictFile],
        task_context: str,
        milestone_id: str,
        attempt: int,
    ) -> MergeResolutionResult:
        """Run a single agent resolution attempt."""

        # Create resolution workspace
        resolution_dir = self.runs_dir / f"merge_resolution_{attempt}"
        resolution_dir.mkdir(parents=True, exist_ok=True)

        # Build prompt
        prompt = build_merge_resolution_prompt(
            conflicts=conflicts,
            task_context=task_context,
            milestone_id=milestone_id,
        )

        prompt_path = resolution_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        # Prepare schema (simple JSON schema for resolution output)
        schema = {
            "type": "object",
            "properties": {
                "resolutions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "resolved_content": {"type": "string"},
                            "notes": {"type": "string"},
                        },
                        "required": ["path", "resolved_content"],
                    },
                },
                "unresolvable": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["path", "reason"],
                    },
                },
            },
            "required": ["resolutions"],
        }

        schema_path = resolution_dir / "resolution.schema.json"
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

        out_path = resolution_dir / "resolution.json"

        # Get agent output - either from callback (for testing) or real agent
        output: dict[str, Any] | None = None

        if self.agent_runner is not None:
            # Use injected callback (for testing)
            print("[merge_resolver] Using injected agent_runner callback...")
            try:
                output = self.agent_runner(conflicts, task_context, milestone_id, attempt)
                # Save output for debugging
                out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
            except Exception as e:
                return MergeResolutionResult(
                    success=False,
                    unresolved_files=[cf.path for cf in conflicts],
                    error=f"Agent callback failed: {e}",
                    attempt=attempt,
                )
        else:
            # Run real agent
            agent_script_path = self.project_root / self.agent_script
            if not agent_script_path.exists():
                return MergeResolutionResult(
                    success=False,
                    unresolved_files=[cf.path for cf in conflicts],
                    error=f"Agent script not found: {agent_script_path}",
                    attempt=attempt,
                )

            env = os.environ.copy()
            # Use generic JSON mode (NOT task_plan which expects milestone_id/tasks/etc.)
            # The "json" mode outputs pure JSON matching arbitrary schemas without task_plan keys
            env["ORCH_SCHEMA_KIND"] = "json"

            cmd = [str(agent_script_path), str(prompt_path), str(schema_path), str(out_path)]

            print("[merge_resolver] Running agent for conflict resolution...")
            rc, stdout, stderr = _run_cmd(cmd, cwd=self.project_root, env=env, timeout=300)

            if rc != 0:
                return MergeResolutionResult(
                    success=False,
                    unresolved_files=[cf.path for cf in conflicts],
                    error=f"Agent failed with exit code {rc}: {stderr[:500]}",
                    attempt=attempt,
                    agent_output=stdout[:2000],
                )

            # Parse agent output
            if not out_path.exists():
                return MergeResolutionResult(
                    success=False,
                    unresolved_files=[cf.path for cf in conflicts],
                    error="Agent did not produce output file",
                    attempt=attempt,
                )

            try:
                output = json.loads(out_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                return MergeResolutionResult(
                    success=False,
                    unresolved_files=[cf.path for cf in conflicts],
                    error=f"Failed to parse agent output: {e}",
                    attempt=attempt,
                )

        if output is None:
            return MergeResolutionResult(
                success=False,
                unresolved_files=[cf.path for cf in conflicts],
                error="No output from agent",
                attempt=attempt,
            )

        # Apply resolutions
        resolved = []
        unresolved = [cf.path for cf in conflicts]

        resolutions = output.get("resolutions", [])
        for resolution in resolutions:
            path = resolution.get("path", "")
            content = resolution.get("resolved_content", "")

            if not path or not content:
                continue

            full_path = self.project_root / path
            if not full_path.exists():
                continue

            try:
                # Write resolved content
                full_path.write_text(content, encoding="utf-8")

                # Stage the file
                rc, _, _ = _run_cmd(["git", "add", path], cwd=self.project_root)
                if rc == 0:
                    resolved.append(path)
                    if path in unresolved:
                        unresolved.remove(path)
                    print(f"[merge_resolver] Resolved: {path}")
            except Exception as e:
                print(f"[merge_resolver] Failed to apply resolution for {path}: {e}")

        # Handle explicitly unresolvable files
        for unresolv in output.get("unresolvable", []):
            path = unresolv.get("path", "")
            reason = unresolv.get("reason", "unknown")
            print(f"[merge_resolver] Agent marked as unresolvable: {path} - {reason}")

        # Check if all conflicts are resolved
        remaining = detect_conflict_files(self.project_root)

        return MergeResolutionResult(
            success=len(remaining) == 0,
            resolved_files=resolved,
            unresolved_files=remaining,
            error=None if len(remaining) == 0 else f"{len(remaining)} conflict(s) remain",
            attempt=attempt,
        )


def attempt_agent_merge_resolution(
    project_root: Path,
    runs_dir: Path,
    task_id: str = "",
    task_context: str = "",
    milestone_id: str = "M0",
    max_attempts: int = 3,
) -> MergeResolutionResult:
    """Convenience function to attempt merge conflict resolution.

    This is the main entry point for the merge resolver.

    Args:
        project_root: Path to the git repository
        runs_dir: Directory for resolution artifacts
        task_id: ID of the task that caused the conflicts
        task_context: Additional context about the merge
        milestone_id: Current milestone ID
        max_attempts: Maximum resolution attempts before falling back to manual

    Returns:
        MergeResolutionResult with resolution status
    """
    resolver = MergeResolver(
        project_root=project_root,
        runs_dir=runs_dir,
        max_attempts=max_attempts,
    )

    return resolver.resolve_conflicts(
        task_id=task_id,
        task_context=task_context,
        milestone_id=milestone_id,
    )
