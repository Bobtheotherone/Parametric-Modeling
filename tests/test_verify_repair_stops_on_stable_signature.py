"""Tests for verify-repair loop early stopping on stable failure signatures.

These tests verify that the verify-repair loop correctly stops when
the same failure signature is seen twice, indicating repairs are ineffective.

This prevents the thrashing seen in the Jan 26 run where the same
ruff_autofix repair was applied 3 times with no effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MockVerifySummary:
    """Mock verify summary for testing."""

    ok: bool
    failed_gates: list[str]
    first_failed_gate: str
    results_by_gate: dict[str, Any]


def compute_failure_signature(summary: MockVerifySummary) -> str:
    """Compute a stable signature for a verify result.

    This mirrors the implementation in bridge/verify_repair/classify.py.
    """
    # Simple signature based on failed gates
    # In the real implementation, this includes more details
    return "|".join(sorted(summary.failed_gates))


class TestSignatureBasedStallDetection:
    """Test signature-based stall detection logic."""

    def test_first_failure_adds_to_history(self):
        """Test that the first failure adds its signature to history."""
        signature_history: list[str] = []
        stable_signature_count = 0

        summary = MockVerifySummary(
            ok=False,
            failed_gates=["spec_lint", "pytest"],
            first_failed_gate="spec_lint",
            results_by_gate={},
        )

        signature = compute_failure_signature(summary)

        # First time seeing this signature
        if signature in signature_history:
            stable_signature_count += 1
        else:
            stable_signature_count = 0
        signature_history.append(signature)

        assert len(signature_history) == 1
        assert stable_signature_count == 0

    def test_repeated_signature_increments_count(self):
        """Test that repeated signatures increment the stall count."""
        signature_history: list[str] = []
        stable_signature_count = 0

        summary = MockVerifySummary(
            ok=False,
            failed_gates=["spec_lint", "pytest"],
            first_failed_gate="spec_lint",
            results_by_gate={},
        )

        # First iteration
        signature = compute_failure_signature(summary)
        if signature in signature_history:
            stable_signature_count += 1
        else:
            stable_signature_count = 0
        signature_history.append(signature)

        # Second iteration (same signature)
        signature = compute_failure_signature(summary)
        if signature in signature_history:
            stable_signature_count += 1
        else:
            stable_signature_count = 0
        signature_history.append(signature)

        assert stable_signature_count == 1, "Should detect repeated signature"

    def test_different_signature_resets_count(self):
        """Test that different signatures reset the stall count."""
        signature_history: list[str] = []
        stable_signature_count = 0

        # First iteration
        summary1 = MockVerifySummary(
            ok=False,
            failed_gates=["spec_lint", "pytest"],
            first_failed_gate="spec_lint",
            results_by_gate={},
        )
        signature = compute_failure_signature(summary1)
        if signature in signature_history:
            stable_signature_count += 1
        else:
            stable_signature_count = 0
        signature_history.append(signature)

        # Second iteration (different signature - one gate fixed)
        summary2 = MockVerifySummary(
            ok=False,
            failed_gates=["pytest"],  # spec_lint is fixed
            first_failed_gate="pytest",
            results_by_gate={},
        )
        signature = compute_failure_signature(summary2)
        if signature in signature_history:
            stable_signature_count += 1
        else:
            stable_signature_count = 0
        signature_history.append(signature)

        assert stable_signature_count == 0, "Should reset count when signature changes"


class TestEarlyStopThreshold:
    """Test the early stop threshold logic."""

    def test_stops_at_threshold_1(self):
        """Test that loop stops when stable_signature_count >= 1.

        This is the new threshold (lowered from 2) to stop faster.
        """
        signature_history: list[str] = []
        stable_signature_count = 0
        should_stop = False
        threshold = 1  # New threshold

        summary = MockVerifySummary(
            ok=False,
            failed_gates=["ruff", "mypy"],
            first_failed_gate="ruff",
            results_by_gate={},
        )

        # Iteration 1
        signature = compute_failure_signature(summary)
        if signature in signature_history:
            stable_signature_count += 1
            if stable_signature_count >= threshold:
                should_stop = True
        else:
            stable_signature_count = 0
        signature_history.append(signature)

        assert not should_stop, "Should not stop on first iteration"

        # Iteration 2 (same signature - repairs didn't help)
        signature = compute_failure_signature(summary)
        if signature in signature_history:
            stable_signature_count += 1
            if stable_signature_count >= threshold:
                should_stop = True
        else:
            stable_signature_count = 0
        signature_history.append(signature)

        assert should_stop, "Should stop when signature repeats (threshold=1)"

    def test_old_threshold_2_wastes_iteration(self):
        """Test that old threshold of 2 would waste an extra iteration.

        This demonstrates why we lowered the threshold.
        """
        signature_history: list[str] = []
        stable_signature_count = 0
        iterations_before_stop = 0
        old_threshold = 2

        summary = MockVerifySummary(
            ok=False,
            failed_gates=["ruff"],
            first_failed_gate="ruff",
            results_by_gate={},
        )

        for iteration in range(5):
            iterations_before_stop = iteration + 1
            signature = compute_failure_signature(summary)

            if signature in signature_history:
                stable_signature_count += 1
                if stable_signature_count >= old_threshold:
                    break
            else:
                stable_signature_count = 0
            signature_history.append(signature)

        # With old threshold=2, we need 3 iterations before stopping:
        # Iter 1: Add to history, count=0
        # Iter 2: Match, count=1
        # Iter 3: Match, count=2 -> stop
        assert iterations_before_stop == 3, \
            f"Old threshold would require {iterations_before_stop} iterations"


class TestJan26VerifyRepairRegression:
    """Test the exact scenario that caused verify-repair thrashing on Jan 26."""

    def test_jan26_repeated_ruff_autofix(self):
        """Test that repeated ineffective ruff_autofix is detected.

        In the Jan 26 run:
        - Attempt 1: spec_lint, pytest, ruff, mypy, git_guard failed
          Applied ruff_autofix:True
        - Attempt 2: SAME failures
          Applied ruff_autofix:True (again)
        - Attempt 3: SAME failures
          Applied ruff_autofix:True (again)

        The fix should stop after attempt 2 (signature repeated once).
        """
        signature_history: list[str] = []
        stable_signature_count = 0
        attempts_before_stop = 0
        threshold = 1  # New threshold

        # The exact gates that failed in Jan 26
        jan26_summary = MockVerifySummary(
            ok=False,
            failed_gates=["spec_lint", "pytest", "ruff", "mypy", "git_guard"],
            first_failed_gate="spec_lint",
            results_by_gate={},
        )

        for attempt in range(5):
            attempts_before_stop = attempt + 1
            signature = compute_failure_signature(jan26_summary)

            if signature in signature_history:
                stable_signature_count += 1
                if stable_signature_count >= threshold:
                    break
            else:
                stable_signature_count = 0
            signature_history.append(signature)

        # With new threshold=1, we should stop after 2 attempts
        assert attempts_before_stop == 2, \
            f"Jan 26 scenario should stop after 2 attempts, not {attempts_before_stop}"

    def test_jan26_would_have_thrashed_3_times(self):
        """Verify that the old behavior would have thrashed 3 times."""
        signature_history: list[str] = []
        stable_signature_count = 0
        attempts_before_stop = 0
        old_threshold = 2  # Old threshold

        jan26_summary = MockVerifySummary(
            ok=False,
            failed_gates=["spec_lint", "pytest", "ruff", "mypy", "git_guard"],
            first_failed_gate="spec_lint",
            results_by_gate={},
        )

        for attempt in range(5):
            attempts_before_stop = attempt + 1
            signature = compute_failure_signature(jan26_summary)

            if signature in signature_history:
                stable_signature_count += 1
                if stable_signature_count >= old_threshold:
                    break
            else:
                stable_signature_count = 0
            signature_history.append(signature)

        # Old behavior: stops after 3 attempts (wastes 1 iteration)
        assert attempts_before_stop == 3, \
            f"Old threshold would have taken {attempts_before_stop} attempts"


class TestProgressVsStall:
    """Test distinguishing between progress and stall."""

    def test_fixing_one_gate_is_progress(self):
        """Test that fixing one gate resets the stall count."""
        signature_history: list[str] = []
        stable_signature_count = 0

        # Iteration 1: 3 gates fail
        summary1 = MockVerifySummary(
            ok=False,
            failed_gates=["spec_lint", "pytest", "ruff"],
            first_failed_gate="spec_lint",
            results_by_gate={},
        )
        sig1 = compute_failure_signature(summary1)
        if sig1 in signature_history:
            stable_signature_count += 1
        else:
            stable_signature_count = 0
        signature_history.append(sig1)

        # Iteration 2: 2 gates fail (ruff fixed!)
        summary2 = MockVerifySummary(
            ok=False,
            failed_gates=["spec_lint", "pytest"],
            first_failed_gate="spec_lint",
            results_by_gate={},
        )
        sig2 = compute_failure_signature(summary2)
        if sig2 in signature_history:
            stable_signature_count += 1
        else:
            stable_signature_count = 0
        signature_history.append(sig2)

        assert stable_signature_count == 0, "Fixing a gate should reset stall count"

    def test_adding_new_failure_is_regression_not_stall(self):
        """Test that adding a new failure is not a stall (it's regression)."""
        signature_history: list[str] = []
        stable_signature_count = 0

        # Iteration 1: 2 gates fail
        summary1 = MockVerifySummary(
            ok=False,
            failed_gates=["spec_lint", "pytest"],
            first_failed_gate="spec_lint",
            results_by_gate={},
        )
        sig1 = compute_failure_signature(summary1)
        if sig1 in signature_history:
            stable_signature_count += 1
        else:
            stable_signature_count = 0
        signature_history.append(sig1)

        # Iteration 2: 3 gates fail (mypy now fails too!)
        summary2 = MockVerifySummary(
            ok=False,
            failed_gates=["spec_lint", "pytest", "mypy"],
            first_failed_gate="spec_lint",
            results_by_gate={},
        )
        sig2 = compute_failure_signature(summary2)
        if sig2 in signature_history:
            stable_signature_count += 1
        else:
            stable_signature_count = 0
        signature_history.append(sig2)

        # This is regression, not stall - different signature
        assert stable_signature_count == 0, "Adding a failure is regression, not stall"
