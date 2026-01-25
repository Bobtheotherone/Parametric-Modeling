"""Deterministic repair actions (Layer 1).

This module provides safe, deterministic auto-fix operations that do not
require LLM intervention:
- ruff --fix for lint errors
- ruff format for formatting
- Bootstrap reinstall for missing dependencies

These are "Layer 1" repairs - they can be run automatically without
spawning agent tasks.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RepairAction:
    """A repair action that was taken."""

    name: str
    command: list[str] | None
    success: bool
    output: str
    files_modified: int = 0


def repair_ruff(project_root: Path, *, verbose: bool = True) -> RepairAction:
    """Run ruff with auto-fix.

    This fixes many lint issues including:
    - Import sorting (I001)
    - Unused imports (F401)
    - Various auto-fixable rules
    """
    if verbose:
        print("[repair] Running ruff check --fix")

    try:
        # First run ruff check --fix
        proc = subprocess.run(
            ["ruff", "check", ".", "--fix", "--unsafe-fixes"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120,
        )

        output_parts = [f"ruff check --fix: rc={proc.returncode}"]
        if proc.stdout:
            output_parts.append(proc.stdout[:2000])

        # Count fixed files from output
        files_fixed = proc.stdout.count("Fixed") if proc.stdout else 0

        # Then run ruff format
        if verbose:
            print("[repair] Running ruff format")

        proc2 = subprocess.run(
            ["ruff", "format", "."],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120,
        )

        output_parts.append(f"ruff format: rc={proc2.returncode}")
        if proc2.stdout:
            output_parts.append(proc2.stdout[:500])
            # Count formatted files
            formatted_count = proc2.stdout.count("reformatted")
            files_fixed += formatted_count

        return RepairAction(
            name="ruff_autofix",
            command=["ruff", "check", ".", "--fix"],
            success=True,
            output="\n".join(output_parts),
            files_modified=files_fixed,
        )

    except FileNotFoundError:
        return RepairAction(
            name="ruff_autofix",
            command=["ruff", "check", ".", "--fix"],
            success=False,
            output="ruff not found in PATH",
        )
    except subprocess.TimeoutExpired:
        return RepairAction(
            name="ruff_autofix",
            command=["ruff", "check", ".", "--fix"],
            success=False,
            output="ruff timed out after 120s",
        )
    except Exception as e:
        return RepairAction(
            name="ruff_autofix",
            command=["ruff", "check", ".", "--fix"],
            success=False,
            output=str(e),
        )


def repair_isort(project_root: Path, *, verbose: bool = True) -> RepairAction:
    """Run isort to fix import ordering (fallback if ruff unavailable)."""
    if verbose:
        print("[repair] Running isort")

    try:
        proc = subprocess.run(
            ["isort", "."],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120,
        )

        return RepairAction(
            name="isort",
            command=["isort", "."],
            success=(proc.returncode == 0),
            output=proc.stdout[:2000] if proc.stdout else "",
        )

    except FileNotFoundError:
        return RepairAction(
            name="isort",
            command=["isort", "."],
            success=False,
            output="isort not found",
        )
    except Exception as e:
        return RepairAction(
            name="isort",
            command=["isort", "."],
            success=False,
            output=str(e),
        )


def get_applicable_repairs(
    categories: set[str],
    summary_data: dict[str, Any] | None = None,
) -> list[str]:
    """Determine which Layer 1 repairs are applicable for the given categories.

    Returns list of repair names that should be attempted.
    """
    repairs: list[str] = []

    from bridge.verify_repair.classify import FailureCategory

    # Convert string categories to enum values for comparison
    cat_set = {c.value if hasattr(c, "value") else c for c in categories}

    if FailureCategory.LINT_RUFF.value in cat_set:
        repairs.append("ruff_autofix")

    if FailureCategory.LINT_FORMAT.value in cat_set:
        repairs.append("ruff_format")

    if FailureCategory.MISSING_DEPENDENCY.value in cat_set:
        repairs.append("bootstrap")

    return repairs


def apply_repair(
    repair_name: str,
    project_root: Path,
    *,
    verbose: bool = True,
) -> RepairAction:
    """Apply a specific repair by name."""
    if repair_name == "ruff_autofix":
        return repair_ruff(project_root, verbose=verbose)
    elif repair_name == "ruff_format":
        # ruff_autofix already includes format, but this allows explicit format-only
        return repair_ruff(project_root, verbose=verbose)
    elif repair_name == "isort":
        return repair_isort(project_root, verbose=verbose)
    elif repair_name == "bootstrap":
        # Bootstrap is handled separately by the loop
        return RepairAction(
            name="bootstrap",
            command=None,
            success=True,
            output="Bootstrap delegated to loop",
        )
    else:
        return RepairAction(
            name=repair_name,
            command=None,
            success=False,
            output=f"Unknown repair: {repair_name}",
        )
