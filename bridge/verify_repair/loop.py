"""Verify auto-repair loop implementation.

This is the main repair loop that orchestrates:
1. Running verify
2. Classifying failures
3. Applying Layer 1 deterministic repairs
4. Generating Layer 2 agent tasks when needed
5. Tracking progress and detecting stalls
6. Writing reports and artifacts
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bridge.verify_repair.bootstrap import run_bootstrap
from bridge.verify_repair.classify import (
    FailureCategory,
    classify_failures,
    compute_failure_signature,
    get_all_categories,
)
from bridge.verify_repair.data import (
    RepairAttemptRecord,
    RepairLoopReport,
    VerifySummary,
)
from bridge.verify_repair.repairs import apply_repair, get_applicable_repairs


@dataclass
class RepairLoopResult:
    """Result of the complete repair loop (backwards compatible)."""

    success: bool
    total_attempts: int
    final_exit_code: int
    reports: list[Any] = field(default_factory=list)  # Legacy format
    remaining_failures: list[str] = field(default_factory=list)
    repair_report: RepairLoopReport | None = None


def _run_verify(
    project_root: Path,
    out_json: Path,
    strict_git: bool,
) -> tuple[int, str, str]:
    """Run tools.verify and capture output."""
    cmd = [sys.executable, "-m", "tools.verify", "--json", str(out_json)]
    if strict_git:
        cmd.append("--strict-git")

    env = os.environ.copy()
    # Ensure src is in PYTHONPATH
    pythonpath = env.get("PYTHONPATH", "")
    if "src" not in pythonpath:
        env["PYTHONPATH"] = f"src:{pythonpath}"

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "verify timeout after 300s"
    except Exception as e:
        return -1, "", str(e)


def _parse_verify_json(json_path: Path) -> VerifySummary | None:
    """Parse verify JSON output into VerifySummary."""
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return VerifySummary.from_json(data)
    except (json.JSONDecodeError, OSError):
        return None


def run_verify_repair_loop(
    project_root: Path,
    verify_json_path: Path,
    *,
    max_attempts: int = 5,
    strict_git: bool = True,
    verbose: bool = True,
    runs_dir: Path | None = None,
    bootstrap_on_start: bool = True,
    agent_task_callback: Callable[[list[Any]], bool] | None = None,
) -> RepairLoopResult:
    """Run verify with automatic repair loop.

    This is the main entry point for the verify auto-repair system.

    Args:
        project_root: Root directory of the project
        verify_json_path: Path to write verify JSON output
        max_attempts: Maximum repair attempts before giving up (default 5)
        strict_git: Whether to use --strict-git flag
        verbose: Whether to print progress
        runs_dir: Directory for run artifacts (optional)
        bootstrap_on_start: Run bootstrap before first verify
        agent_task_callback: Optional callback to execute agent tasks
            Signature: callback(tasks) -> bool (success)
            If not provided, agent tasks are generated but not executed

    Returns:
        RepairLoopResult containing success status and repair reports
    """
    start_time = datetime.now(timezone.utc)
    attempts: list[RepairAttemptRecord] = []
    artifacts_written: list[str] = []
    signature_history: list[str] = []
    stable_signature_count = 0

    def _log(msg: str) -> None:
        if verbose:
            print(f"[verify_repair] {msg}")

    # Ensure runs_dir exists
    if runs_dir is None:
        runs_dir = verify_json_path.parent
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Bootstrap on start if requested
    if bootstrap_on_start:
        _log("Running initial bootstrap")
        bootstrap_log = runs_dir / "bootstrap.log"
        bootstrap_result = run_bootstrap(
            project_root,
            log_path=bootstrap_log,
            verbose=verbose,
        )
        if not bootstrap_result.skipped:
            artifacts_written.append(str(bootstrap_log))
        if not bootstrap_result.success:
            _log(f"Bootstrap failed: {bootstrap_result.stderr[:200]}")
            # Continue anyway - verify will tell us what's wrong

    for attempt_idx in range(1, max_attempts + 1):
        attempt_start = datetime.now(timezone.utc)
        _log(f"=== Attempt {attempt_idx}/{max_attempts} ===")

        # Run verify
        rc, stdout, stderr = _run_verify(project_root, verify_json_path, strict_git)

        if rc == 0:
            _log(f"SUCCESS after {attempt_idx} attempt(s)")
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            return RepairLoopResult(
                success=True,
                total_attempts=attempt_idx,
                final_exit_code=0,
                remaining_failures=[],
                repair_report=RepairLoopReport(
                    success=True,
                    total_attempts=attempt_idx,
                    final_failed_gates=[],
                    elapsed_s=elapsed,
                    stable_failure_signature_count=stable_signature_count,
                    artifacts_written=artifacts_written,
                    attempts=attempts,
                ),
            )

        # Parse verify result
        summary = _parse_verify_json(verify_json_path)
        if summary is None:
            _log("Failed to parse verify JSON")
            summary = VerifySummary(
                ok=False,
                failed_gates=["unknown"],
                first_failed_gate="unknown",
                results_by_gate={},
            )

        _log(f"Failed gates: {summary.failed_gates}")

        # Compute failure signature for stall detection
        signature = compute_failure_signature(summary)
        if signature in signature_history:
            stable_signature_count += 1
            _log(f"Repeated failure signature detected (count: {stable_signature_count})")
            if stable_signature_count >= 2:
                _log("Stopping: same failures repeated without progress")
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                return RepairLoopResult(
                    success=False,
                    total_attempts=attempt_idx,
                    final_exit_code=rc,
                    remaining_failures=summary.failed_gates,
                    repair_report=RepairLoopReport(
                        success=False,
                        total_attempts=attempt_idx,
                        final_failed_gates=summary.failed_gates,
                        elapsed_s=elapsed,
                        stable_failure_signature_count=stable_signature_count,
                        artifacts_written=artifacts_written,
                        attempts=attempts,
                        early_stop_reason="Repeated identical failure signature",
                    ),
                )
        else:
            stable_signature_count = 0
        signature_history.append(signature)

        # Classify failures
        classification = classify_failures(summary)
        all_categories = get_all_categories(classification)
        category_names = [c.value for c in all_categories]
        _log(f"Failure categories: {category_names}")

        # Attempt Layer 1 repairs (deterministic)
        actions_taken: list[str] = []
        repairs_applied = False

        # Check for bootstrap-fixable issues
        if FailureCategory.MISSING_DEPENDENCY in all_categories:
            _log("Running bootstrap for missing dependency")
            bootstrap_log = runs_dir / f"bootstrap_repair_{attempt_idx}.log"
            bootstrap_result = run_bootstrap(
                project_root,
                log_path=bootstrap_log,
                force=True,
                verbose=verbose,
            )
            artifacts_written.append(str(bootstrap_log))
            if bootstrap_result.success:
                actions_taken.append("bootstrap_reinstall")
                repairs_applied = True

        # Apply applicable Layer 1 repairs
        applicable = get_applicable_repairs(all_categories)
        for repair_name in applicable:
            if repair_name == "bootstrap":
                continue  # Already handled above
            _log(f"Applying repair: {repair_name}")
            action = apply_repair(repair_name, project_root, verbose=verbose)
            actions_taken.append(f"{repair_name}:{action.success}")
            if action.success and action.files_modified > 0:
                repairs_applied = True
                _log(f"  Modified {action.files_modified} files")

        # Check if we need Layer 2 (agent-driven repairs)
        needs_agent_repair = False
        agent_categories = {
            FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR,
            FailureCategory.PYTEST_TEST_FAILURE,
            FailureCategory.TYPECHECK_MYPY,
            FailureCategory.SPEC_LINT_FAILURE,
            FailureCategory.MISSING_MODULE_INTERNAL,
        }
        if all_categories & agent_categories:
            needs_agent_repair = True

        # Generate and potentially execute agent tasks
        if needs_agent_repair and agent_task_callback is not None:
            _log("Generating agent repair tasks")
            from bridge.verify_repair.agent_tasks import generate_repair_tasks

            repair_tasks = generate_repair_tasks(summary, classification)
            if repair_tasks:
                _log(f"Generated {len(repair_tasks)} repair task(s)")
                # Write task specs to artifacts
                tasks_file = runs_dir / f"repair_tasks_{attempt_idx}.json"
                tasks_data = [t.to_dict() for t in repair_tasks]
                tasks_file.write_text(json.dumps(tasks_data, indent=2), encoding="utf-8")
                artifacts_written.append(str(tasks_file))

                # Execute via callback
                success = agent_task_callback(repair_tasks)
                if success:
                    actions_taken.append("agent_repair_tasks")
                    repairs_applied = True
                else:
                    actions_taken.append("agent_repair_tasks:failed")

        # Record attempt
        attempt_elapsed = (datetime.now(timezone.utc) - attempt_start).total_seconds()
        attempts.append(RepairAttemptRecord(
            attempt_index=attempt_idx,
            detected_categories=category_names,
            actions_taken=actions_taken,
            verify_before=summary,
            verify_after=None,  # Will be set by next iteration
            diff_applied=repairs_applied,
            elapsed_s=attempt_elapsed,
        ))

        if not repairs_applied:
            _log("No repairs could be applied")
            # Check if we have agent tasks but no callback
            if needs_agent_repair and agent_task_callback is None:
                _log("Agent repairs needed but no callback provided")
                # Write pending tasks for orchestrator to pick up
                from bridge.verify_repair.agent_tasks import generate_repair_tasks

                repair_tasks = generate_repair_tasks(summary, classification)
                if repair_tasks:
                    pending_tasks_file = runs_dir / "pending_repair_tasks.json"
                    tasks_data = [t.to_dict() for t in repair_tasks]
                    # Include full description for manual review
                    for i, task in enumerate(repair_tasks):
                        tasks_data[i]["full_description"] = task.description
                    pending_tasks_file.write_text(json.dumps(tasks_data, indent=2), encoding="utf-8")
                    artifacts_written.append(str(pending_tasks_file))
                    _log(f"Wrote pending repair tasks to: {pending_tasks_file}")
            break

    # Final verify to get current state
    rc, _, _ = _run_verify(project_root, verify_json_path, strict_git)
    final_summary = _parse_verify_json(verify_json_path)
    final_failed = final_summary.failed_gates if final_summary else ["unknown"]

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

    _log(f"{'SUCCESS' if rc == 0 else 'FAILED'} after {len(attempts)} attempt(s)")
    if rc != 0:
        _log(f"Remaining failures: {final_failed}")

    return RepairLoopResult(
        success=(rc == 0),
        total_attempts=len(attempts),
        final_exit_code=rc,
        remaining_failures=final_failed if rc != 0 else [],
        repair_report=RepairLoopReport(
            success=(rc == 0),
            total_attempts=len(attempts),
            final_failed_gates=final_failed if rc != 0 else [],
            elapsed_s=elapsed,
            stable_failure_signature_count=stable_signature_count,
            artifacts_written=artifacts_written,
            attempts=attempts,
        ),
    )


def write_repair_report(
    result: RepairLoopResult,
    output_path: Path,
) -> None:
    """Write repair loop report to JSON file."""
    if result.repair_report:
        report_data = result.repair_report.to_dict()
    else:
        # Legacy format fallback
        report_data = {
            "success": result.success,
            "total_attempts": result.total_attempts,
            "final_exit_code": result.final_exit_code,
            "remaining_failures": result.remaining_failures,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")


# Legacy compatibility - maintain original API
RepairReport = RepairAttemptRecord  # Alias for backwards compatibility
