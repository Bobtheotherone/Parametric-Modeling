"""Verify auto-repair: Automatic repair loop for verify failures.

This module is a backwards-compatible shim. The implementation has been
moved to bridge/verify_repair/ package for better organization.

See bridge/verify_repair/__init__.py for the full API.
"""

from __future__ import annotations

# Re-export all public API for backwards compatibility
from bridge.verify_repair import (
    FailureCategory,
    RepairAttemptRecord,
    RepairLoopReport,
    RepairLoopResult,
    VerifyGateResult,
    VerifySummary,
    classify_failures,
    compute_failure_signature,
    run_verify_repair_loop,
    write_repair_report,
)

# Legacy aliases
RepairReport = RepairAttemptRecord

__all__ = [
    "RepairReport",
    "RepairLoopResult",
    "run_verify_repair_loop",
    "write_repair_report",
    # New exports
    "FailureCategory",
    "RepairAttemptRecord",
    "RepairLoopReport",
    "VerifyGateResult",
    "VerifySummary",
    "classify_failures",
    "compute_failure_signature",
]


# CLI entrypoint for testing
def main() -> int:
    """CLI entrypoint for testing the repair loop."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run verify auto-repair loop")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--max-attempts", type=int, default=5, help="Maximum repair attempts")
    parser.add_argument("--strict-git", action="store_true", help="Use strict git mode")
    parser.add_argument("--output", default="verify_repair_report.json", help="Output report path")
    parser.add_argument("--no-bootstrap", action="store_true", help="Skip initial bootstrap")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    verify_json = project_root / "verify_repair_temp.json"

    result = run_verify_repair_loop(
        project_root=project_root,
        verify_json_path=verify_json,
        max_attempts=args.max_attempts,
        strict_git=args.strict_git,
        verbose=True,
        bootstrap_on_start=not args.no_bootstrap,
    )

    output_path = project_root / args.output
    write_repair_report(result, output_path)
    print(f"\n[verify_repair] Report written to: {output_path}")

    return 0 if result.success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
