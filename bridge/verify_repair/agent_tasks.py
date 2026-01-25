"""Agent-driven repair task generation (Layer 2).

When deterministic repairs (Layer 1) cannot fix the failures, this module
generates targeted repair tasks that can be executed by agents.

The generated tasks are:
- Scoped to only the failing gates
- Include reproduction commands
- Include explicit "do not touch" rules
- Require agents to provide minimal plans and concrete diffs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bridge.verify_repair.classify import (
    FailureCategory,
    extract_import_errors,
    extract_mypy_errors,
)
from bridge.verify_repair.data import VerifySummary


@dataclass
class RepairTask:
    """A repair task to be executed by an agent."""

    id: str
    title: str
    description: str
    agent: str = "claude"
    intensity: str = "medium"
    failing_gates: list[str] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    reproduction_commands: list[str] = field(default_factory=list)
    do_not_touch: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for task scheduling."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "agent": self.agent,
            "intensity": self.intensity,
        }


def _build_import_error_task(
    summary: VerifySummary,
    import_errors: list[dict[str, Any]],
) -> RepairTask | None:
    """Build a repair task for import errors."""
    if not import_errors:
        return None

    # Group by type
    cannot_import = [e for e in import_errors if e["type"] == "import_error"]
    module_not_found = [e for e in import_errors if e["type"] == "module_not_found"]
    internal_missing = [e for e in module_not_found if e.get("is_internal")]

    if not cannot_import and not internal_missing:
        # Only external missing modules - should be handled by bootstrap
        return None

    # Build description
    desc_parts = [
        "## Verify Repair Task: Fix Import Errors",
        "",
        "The following import errors are blocking pytest/spec_lint:",
        "",
    ]

    if cannot_import:
        desc_parts.append("### ImportError (cannot import name)")
        for err in cannot_import[:5]:  # Limit to 5
            desc_parts.append(f"- Cannot import `{err['name']}` (detected in {err['source_gate']})")
        desc_parts.append("")

    if internal_missing:
        desc_parts.append("### ModuleNotFoundError (internal modules)")
        for err in internal_missing[:5]:
            desc_parts.append(f"- Missing module `{err['module']}` (detected in {err['source_gate']})")
        desc_parts.append("")

    # Add reproduction commands
    desc_parts.extend([
        "## Reproduction",
        "",
        "Run the following to reproduce:",
        "```bash",
        "python -m pytest -q --collect-only 2>&1 | head -100",
        "```",
        "",
        "## Requirements",
        "",
        "1. Fix ONLY the import errors listed above",
        "2. Do NOT add new features or refactor unrelated code",
        "3. Do NOT modify tests - fix the source modules",
        "4. Verify your fix with: `python -m pytest -q --collect-only`",
        "",
    ])

    # Get target files from error context
    target_files: list[str] = []
    for err in import_errors:
        if "module" in err:
            # Convert module path to file path guess
            module_path = err["module"].replace(".", "/")
            target_files.append(f"src/{module_path}.py")
            target_files.append(f"src/{module_path}/__init__.py")

    return RepairTask(
        id="REPAIR-IMPORT-ERRORS",
        title="Fix import errors blocking test collection",
        description="\n".join(desc_parts),
        agent="claude",
        intensity="medium",
        failing_gates=["pytest", "spec_lint"],
        target_files=list(set(target_files))[:10],
        reproduction_commands=[
            "python -m pytest -q --collect-only",
            "python -m tools.spec_lint DESIGN_DOCUMENT.md --collect",
        ],
        do_not_touch=["tests/**", "bridge/**", "tools/**"],
        constraints=[
            "Fix only the import errors listed",
            "Do not add new features",
            "Do not modify test files",
        ],
    )


def _build_mypy_error_task(
    summary: VerifySummary,
    mypy_errors: list[dict[str, str]],
) -> RepairTask | None:
    """Build a repair task for mypy errors."""
    if not mypy_errors:
        return None

    desc_parts = [
        "## Verify Repair Task: Fix Type Errors",
        "",
        "The following mypy type errors need to be fixed:",
        "",
    ]

    target_files: set[str] = set()
    for err in mypy_errors[:10]:  # Limit to 10
        desc_parts.append(f"- `{err['file']}:{err['line']}`: {err['message']}")
        target_files.add(err["file"])

    desc_parts.extend([
        "",
        "## Reproduction",
        "",
        "```bash",
        "mypy .",
        "```",
        "",
        "## Requirements",
        "",
        "1. Fix ONLY the type errors listed above",
        "2. Use proper type annotations, not `# type: ignore`",
        "3. If a type error reveals a real bug, fix the bug",
        "4. Do NOT change unrelated code",
        "",
    ])

    return RepairTask(
        id="REPAIR-MYPY-ERRORS",
        title="Fix mypy type errors",
        description="\n".join(desc_parts),
        agent="claude",
        intensity="low",
        failing_gates=["mypy"],
        target_files=list(target_files)[:10],
        reproduction_commands=["mypy ."],
        do_not_touch=["tests/**", "bridge/**", "tools/**"],
        constraints=[
            "Fix only the type errors listed",
            "Do not use # type: ignore",
            "Do not change unrelated code",
        ],
    )


def _build_test_failure_task(
    summary: VerifySummary,
) -> RepairTask | None:
    """Build a repair task for test failures (not collection errors)."""
    gate = summary.results_by_gate.get("pytest")
    if not gate:
        return None

    combined = gate.stdout + "\n" + gate.stderr

    # Extract failed test names
    failed_tests: list[str] = []
    for line in combined.split("\n"):
        if line.startswith("FAILED "):
            test_name = line.split(" ")[1].split("::")[0] if "::" in line else ""
            if test_name and test_name not in failed_tests:
                failed_tests.append(test_name)

    if not failed_tests:
        return None

    desc_parts = [
        "## Verify Repair Task: Fix Failing Tests",
        "",
        "The following tests are failing:",
        "",
    ]

    for test in failed_tests[:10]:
        desc_parts.append(f"- `{test}`")

    desc_parts.extend([
        "",
        "## Reproduction",
        "",
        "```bash",
        "python -m pytest -q",
        "```",
        "",
        "## Requirements",
        "",
        "1. Analyze WHY the tests are failing",
        "2. Fix the SOURCE CODE if there's a bug",
        "3. Fix the TEST if the test is incorrect",
        "4. Do NOT delete or skip tests",
        "",
    ])

    return RepairTask(
        id="REPAIR-TEST-FAILURES",
        title="Fix failing tests",
        description="\n".join(desc_parts),
        agent="claude",
        intensity="medium",
        failing_gates=["pytest"],
        target_files=failed_tests[:10],
        reproduction_commands=["python -m pytest -q"],
        constraints=[
            "Do not delete or skip tests",
            "Fix bugs in source code when appropriate",
        ],
    )


def generate_repair_tasks(
    summary: VerifySummary,
    classification: dict[str, list[FailureCategory]],
) -> list[RepairTask]:
    """Generate repair tasks for the given failures.

    Returns a list of RepairTask objects that can be executed by agents.
    Tasks are prioritized by likelihood of success.
    """
    tasks: list[RepairTask] = []

    # Get all categories
    all_cats: set[FailureCategory] = set()
    for cats in classification.values():
        all_cats.update(cats)

    # 1. Import error repairs (highest priority - blocks everything)
    if FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR in all_cats:
        import_errors = extract_import_errors(summary)
        task = _build_import_error_task(summary, import_errors)
        if task:
            tasks.append(task)

    # 2. Mypy error repairs
    if FailureCategory.TYPECHECK_MYPY in all_cats:
        mypy_errors = extract_mypy_errors(summary)
        task = _build_mypy_error_task(summary, mypy_errors)
        if task:
            tasks.append(task)

    # 3. Test failure repairs (lower priority - may be fixed by above)
    if FailureCategory.PYTEST_TEST_FAILURE in all_cats:
        task = _build_test_failure_task(summary)
        if task:
            tasks.append(task)

    return tasks


def build_repair_task_prompt(task: RepairTask) -> str:
    """Build a complete prompt for executing a repair task.

    This prompt is designed to be given to an agent with clear boundaries.
    """
    prompt_parts = [
        "# AUTOMATED REPAIR TASK",
        "",
        f"Task ID: {task.id}",
        f"Title: {task.title}",
        "",
        "---",
        "",
        task.description,
        "",
        "---",
        "",
        "## Constraints",
        "",
    ]

    for constraint in task.constraints:
        prompt_parts.append(f"- {constraint}")

    if task.do_not_touch:
        prompt_parts.extend([
            "",
            "## DO NOT MODIFY",
            "",
        ])
        for pattern in task.do_not_touch:
            prompt_parts.append(f"- `{pattern}`")

    prompt_parts.extend([
        "",
        "## Expected Output",
        "",
        "1. A brief analysis of the root cause",
        "2. The specific files you will modify",
        "3. The changes (as diffs or complete new content)",
        "4. Verification that the fix works",
        "",
    ])

    return "\n".join(prompt_parts)
