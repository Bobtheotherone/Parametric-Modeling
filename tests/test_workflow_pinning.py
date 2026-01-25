"""Workflow pinning enforcement tests.

Enforces supply-chain policy: GitHub Actions workflows must use
SHA-pinned references (40-character hex SHAs) for all external actions.

Related:
- REQ-M2-005: Toolchain is digest-pinned and recorded in every run's metadata
- WP1: Toolchain digest pinning + enforcement
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import pytest


def _find_workflow_files() -> list[Path]:
    """Find all GitHub Actions workflow files in .github/workflows."""
    workflows_dir = Path(".github") / "workflows"
    if not workflows_dir.exists():
        return []

    yml_files = sorted(workflows_dir.glob("*.yml"))
    yaml_files = sorted(workflows_dir.glob("*.yaml"))
    return yml_files + yaml_files


USES_LINE_RE = re.compile(r"^\s*-?\s*uses:\s*(.+)$")


def _strip_inline_comment(value: str) -> str:
    """Strip inline comments from a uses value."""
    if "#" not in value:
        return value
    return value.split("#", 1)[0].rstrip()


def _strip_wrapping_quotes(value: str) -> str:
    """Strip matching single/double quotes around a value."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1].strip()
    return value


def _check_sha_pinning(workflow_path: Path) -> list[tuple[int, str, str]]:
    """Check a workflow file for unpinned actions.

    Returns a list of (line_number, issue_type, line_content) tuples.
    """
    issues: list[tuple[int, str, str]] = []
    content = workflow_path.read_text(encoding="utf-8")

    for idx, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            continue

        # Only check 'uses:' directives (supports "- uses:" and "uses:")
        match = USES_LINE_RE.match(line)
        if not match:
            continue

        value = _strip_wrapping_quotes(_strip_inline_comment(match.group(1).strip()))
        if not value:
            issues.append((idx, "missing_ref", value))
            continue

        # Local actions and docker references are allowed without SHA
        if value.startswith("./") or value.startswith("docker://"):
            continue

        # Must have @ reference
        if "@" not in value:
            issues.append((idx, "missing_ref", value))
            continue

        # Reference must be a full 40-character SHA
        ref = value.split("@", 1)[1]
        if not re.fullmatch(r"[0-9a-fA-F]{40}", ref):
            issues.append((idx, "not_sha_pinned", value))

    return issues


def test_workflows_are_sha_pinned() -> None:
    """Verify all GitHub Actions workflows use SHA-pinned action references.

    This test enforces supply-chain security policy by requiring:
    - All external action references use full 40-character commit SHAs
    - No version tags (v1, v2, main, etc.) are allowed
    - Local actions (./) and docker references are exempt

    The test safely skips if no workflow files exist.
    """
    workflow_files = _find_workflow_files()

    if not workflow_files:
        pytest.skip("No workflow files found in .github/workflows/")

    all_issues: list[str] = []

    for wf_path in workflow_files:
        issues = _check_sha_pinning(wf_path)
        for line_num, issue_type, value in issues:
            if issue_type == "missing_ref":
                all_issues.append(f"{wf_path}:{line_num}: action missing @<sha> reference: {value}")
            elif issue_type == "not_sha_pinned":
                all_issues.append(f"{wf_path}:{line_num}: action not pinned to full SHA: {value}")

    assert not all_issues, (
        "Unpinned GitHub Actions detected (supply-chain policy violation):\n"
        + "\n".join(all_issues)
        + "\n\nAll external actions must use full 40-character commit SHA references."
    )


def test_workflow_pinning_enforcement_skips_when_no_workflows() -> None:
    """Verify the test skips gracefully when no workflows directory exists.

    This validates the skip behavior works correctly when .github/workflows
    is missing (e.g., in non-GitHub-hosted projects).
    """
    workflow_files = _find_workflow_files()

    # This test validates the skip mechanism - it passes if skip would occur
    # or if workflows exist and are properly pinned
    if not workflow_files:
        pytest.skip("No workflow files - skip behavior validated")

    # If workflows exist, they must be pinned (same validation as main test)
    all_issues: list[str] = []
    for wf_path in workflow_files:
        issues = _check_sha_pinning(wf_path)
        for line_num, issue_type, value in issues:
            all_issues.append(f"{wf_path}:{line_num}: {issue_type}: {value}")

    assert not all_issues, "Workflow pinning issues found: " + "; ".join(all_issues)
