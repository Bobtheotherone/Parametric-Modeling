"""Directive-file materialization for agent invocations.

Ensures CLAUDE.md / AGENTS.md are present in the agent's working directory
before every invocation, regardless of code path (loop, merge_resolver, etc.).
"""

from __future__ import annotations

from pathlib import Path

from bridge.atomic_io import atomic_copy_file


def materialize_directive_file(
    *,
    project_root: Path,
    target_dir: Path,
    agent_name: str,
    verbose: bool = False,
) -> None:
    """Ensure the correct directive file exists in *target_dir* before agent invocation.

    For Claude invocations:
      - Prefer project_root/CLAUDE.md; fall back to project_root/AGENTS.md.
      - Copy to target_dir/CLAUDE.md.
    For all agents:
      - If project_root/AGENTS.md exists, copy to target_dir/AGENTS.md.

    Copies are atomic and idempotent (skip if content already matches).
    """
    def _copy_if_needed(src: Path, dst: Path) -> bool:
        """Copy *src* to *dst* atomically if content differs. Returns True if written."""
        if not src.exists():
            return False
        try:
            if dst.exists() and dst.read_bytes() == src.read_bytes():
                return False
        except OSError:
            pass
        atomic_copy_file(src, dst)
        return True

    agents_src = project_root / "AGENTS.md"
    claude_src = project_root / "CLAUDE.md"

    # For Claude agents, materialize CLAUDE.md
    if agent_name == "claude":
        source = claude_src if claude_src.exists() else (agents_src if agents_src.exists() else None)
        if source:
            wrote = _copy_if_needed(source, target_dir / "CLAUDE.md")
            if verbose and wrote:
                print(f"[orchestrator] directive: copied {source.name} -> {target_dir / 'CLAUDE.md'}")

    # For all agents, materialize AGENTS.md if present in project root
    if agents_src.exists():
        wrote = _copy_if_needed(agents_src, target_dir / "AGENTS.md")
        if verbose and wrote:
            print(f"[orchestrator] directive: copied AGENTS.md -> {target_dir / 'AGENTS.md'}")
