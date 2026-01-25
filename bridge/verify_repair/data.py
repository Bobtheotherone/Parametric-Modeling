"""Data structures for verify repair operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerifyGateResult:
    """Result from a single verify gate."""

    name: str
    returncode: int | None
    passed: bool
    stdout: str
    stderr: str
    cmd: list[str] | None = None
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifyGateResult:
        """Create from verify JSON gate result."""
        return cls(
            name=data.get("name", ""),
            returncode=data.get("returncode"),
            passed=data.get("passed", False),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            cmd=data.get("cmd"),
            note=data.get("note", ""),
        )


@dataclass
class VerifySummary:
    """Summary of a verify run."""

    ok: bool
    failed_gates: list[str]
    first_failed_gate: str
    results_by_gate: dict[str, VerifyGateResult]

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> VerifySummary:
        """Create from verify JSON output."""
        results_by_gate: dict[str, VerifyGateResult] = {}
        for r in data.get("results", []):
            gate = VerifyGateResult.from_dict(r)
            results_by_gate[gate.name] = gate
        return cls(
            ok=data.get("ok", False),
            failed_gates=data.get("failed_gates", []),
            first_failed_gate=data.get("first_failed_gate", ""),
            results_by_gate=results_by_gate,
        )


@dataclass
class RepairAttemptRecord:
    """Record of a single repair attempt."""

    attempt_index: int
    detected_categories: list[str]
    actions_taken: list[str]
    verify_before: VerifySummary | None
    verify_after: VerifySummary | None
    diff_applied: bool
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "attempt_index": self.attempt_index,
            "detected_categories": self.detected_categories,
            "actions_taken": self.actions_taken,
            "diff_applied": self.diff_applied,
            "elapsed_s": self.elapsed_s,
            "verify_before_ok": self.verify_before.ok if self.verify_before else None,
            "verify_before_failed": self.verify_before.failed_gates if self.verify_before else [],
            "verify_after_ok": self.verify_after.ok if self.verify_after else None,
            "verify_after_failed": self.verify_after.failed_gates if self.verify_after else [],
        }


@dataclass
class RepairLoopReport:
    """Final report of the repair loop."""

    success: bool
    total_attempts: int
    final_failed_gates: list[str]
    elapsed_s: float
    stable_failure_signature_count: int
    artifacts_written: list[str]
    attempts: list[RepairAttemptRecord] = field(default_factory=list)
    early_stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "success": self.success,
            "total_attempts": self.total_attempts,
            "final_failed_gates": self.final_failed_gates,
            "elapsed_s": self.elapsed_s,
            "stable_failure_signature_count": self.stable_failure_signature_count,
            "artifacts_written": self.artifacts_written,
            "early_stop_reason": self.early_stop_reason,
            "attempts": [a.to_dict() for a in self.attempts],
        }
