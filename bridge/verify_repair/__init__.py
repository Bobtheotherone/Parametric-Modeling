"""Verify auto-repair module for the orchestrator.

This module provides automatic repair functionality for verify gate failures
including failure classification, deterministic auto-fixes, and agent-driven
repair task generation.

The repair loop runs until verify passes, max_attempts is reached, or
repeated identical failures indicate no progress is being made.

STALL PREVENTION: The executor module provides the agent_task_callback that
enables the repair loop to automatically execute repairs instead of waiting
for manual intervention.
"""

from bridge.verify_repair.classify import (
    FailureCategory,
    classify_failures,
    compute_failure_signature,
)
from bridge.verify_repair.data import (
    RepairAttemptRecord,
    RepairLoopReport,
    VerifyGateResult,
    VerifySummary,
)
from bridge.verify_repair.executor import (
    RepairExecutionResult,
    RepairExecutor,
    create_repair_callback,
)
from bridge.verify_repair.loop import (
    RepairLoopResult,
    run_verify_repair_loop,
    write_repair_report,
)

__all__ = [
    # Data structures
    "VerifyGateResult",
    "VerifySummary",
    "RepairAttemptRecord",
    "RepairLoopReport",
    # Classification
    "FailureCategory",
    "classify_failures",
    "compute_failure_signature",
    # Loop
    "RepairLoopResult",
    "run_verify_repair_loop",
    "write_repair_report",
    # Executor (STALL PREVENTION)
    "RepairExecutionResult",
    "RepairExecutor",
    "create_repair_callback",
]
