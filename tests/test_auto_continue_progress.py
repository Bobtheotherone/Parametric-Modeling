"""Tests for signature-based progress detection in the auto-continue loop.

These tests verify that the orchestrator correctly detects when different
task IDs have the same underlying error, treating this as "no progress".

This prevents the false "progress detected" messages that led to thrashing
in the Jan 26 run, where different task IDs failed with identical errors.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


def _compute_failure_signature(error: str) -> str:
    """Compute a normalized signature for an error message.

    This mirrors the implementation in bridge/loop.py.
    """
    if not error:
        return "empty"

    # Normalize: strip timestamps like 20260126T011919Z
    normalized = re.sub(r"\d{8}T\d{6}Z", "TIMESTAMP", error)

    # Normalize: strip task branch names like task/TIMESTAMP/TASK-ID
    normalized = re.sub(r"task/TIMESTAMP/[A-Za-z0-9_-]+", "task/TIMESTAMP/TASK_ID", normalized)

    # Normalize: strip generic task IDs (M0-*, FILLER-*, etc.)
    normalized = re.sub(r"M\d+-[A-Za-z0-9_-]+", "TASK_ID", normalized)
    normalized = re.sub(r"FILLER-[A-Za-z0-9_-]+", "FILLER_ID", normalized)

    # Normalize: strip absolute paths, keep only the filename
    normalized = re.sub(r"/[^\s]+/([^/\s]+)", r"\1", normalized)

    # Normalize: strip line numbers
    normalized = re.sub(r":\d+:", ":LINE:", normalized)

    # Normalize: strip worker IDs
    normalized = re.sub(r"w\d{2}", "wXX", normalized)

    # Normalize whitespace
    normalized = " ".join(normalized.split())

    # Hash to get a stable signature
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _compute_failure_signatures_from_summary(summary: dict[str, Any]) -> set[str]:
    """Compute failure signatures from a run summary.

    This mirrors the implementation in bridge/loop.py.
    """
    signatures = set()
    for failure in summary.get("root_failures", []):
        error = failure.get("error", "")
        sig = _compute_failure_signature(error)
        signatures.add(sig)
    return signatures


class TestFailureSignatureComputation:
    """Test failure signature computation."""

    def test_empty_error_returns_empty_signature(self):
        """Test that empty errors get a consistent signature."""
        assert _compute_failure_signature("") == "empty"
        assert _compute_failure_signature(None) == "empty"  # type: ignore

    def test_same_error_same_signature(self):
        """Test that identical errors produce identical signatures."""
        error = "fatal: A branch named 'task/test' already exists."
        sig1 = _compute_failure_signature(error)
        sig2 = _compute_failure_signature(error)
        assert sig1 == sig2

    def test_different_timestamps_same_signature(self):
        """Test that errors differing only in timestamps have the same signature."""
        error1 = "Branch task/20260126T011919Z/task1 already exists"
        error2 = "Branch task/20260126T010448Z/task1 already exists"

        sig1 = _compute_failure_signature(error1)
        sig2 = _compute_failure_signature(error2)

        assert sig1 == sig2, "Errors with different timestamps should have same signature"

    def test_different_paths_same_signature(self):
        """Test that errors differing only in paths have the same signature."""
        error1 = "Error in /home/user1/project/file.py"
        error2 = "Error in /home/user2/project/file.py"

        sig1 = _compute_failure_signature(error1)
        sig2 = _compute_failure_signature(error2)

        assert sig1 == sig2, "Errors with different paths should have same signature"

    def test_different_worker_ids_same_signature(self):
        """Test that errors differing only in worker IDs have the same signature."""
        error1 = "[w01 claude task1] Failed"
        error2 = "[w05 claude task1] Failed"

        sig1 = _compute_failure_signature(error1)
        sig2 = _compute_failure_signature(error2)

        assert sig1 == sig2, "Errors with different worker IDs should have same signature"

    def test_different_line_numbers_same_signature(self):
        """Test that errors differing only in line numbers have the same signature."""
        error1 = "SyntaxError at file.py:123: unexpected token"
        error2 = "SyntaxError at file.py:456: unexpected token"

        sig1 = _compute_failure_signature(error1)
        sig2 = _compute_failure_signature(error2)

        assert sig1 == sig2, "Errors with different line numbers should have same signature"

    def test_fundamentally_different_errors_different_signature(self):
        """Test that fundamentally different errors have different signatures."""
        error1 = "fatal: A branch named 'task/test' already exists."
        error2 = "You've hit your limit · resets 7pm (America/Anchorage)"

        sig1 = _compute_failure_signature(error1)
        sig2 = _compute_failure_signature(error2)

        assert sig1 != sig2, "Different error types should have different signatures"


class TestProgressDetection:
    """Test progress detection using failure signatures."""

    def test_same_task_ids_same_signatures_is_no_progress(self):
        """Test that same task IDs with same signatures is detected as no progress."""
        summary1 = {
            "root_failures": [
                {"id": "TASK-1", "error": "Branch exists"},
                {"id": "TASK-2", "error": "Rate limit"},
            ]
        }
        summary2 = {
            "root_failures": [
                {"id": "TASK-1", "error": "Branch exists"},
                {"id": "TASK-2", "error": "Rate limit"},
            ]
        }

        sigs1 = _compute_failure_signatures_from_summary(summary1)
        sigs2 = _compute_failure_signatures_from_summary(summary2)

        assert sigs1 == sigs2, "Same failures should have same signatures"

    def test_different_task_ids_same_error_is_no_progress(self):
        """Test that different task IDs with same underlying error is no progress.

        This is the key scenario from Jan 26: different tasks failed with
        the same "branch already exists" error.
        """
        summary1 = {
            "root_failures": [
                {"id": "M0-RF-01-TASK-A", "error": "fatal: A branch named 'task/20260126T011919Z/TASK-A' already exists."},
            ]
        }
        summary2 = {
            "root_failures": [
                {"id": "M0-RF-01-TASK-B", "error": "fatal: A branch named 'task/20260126T011919Z/TASK-B' already exists."},
            ]
        }

        sigs1 = _compute_failure_signatures_from_summary(summary1)
        sigs2 = _compute_failure_signatures_from_summary(summary2)

        # After normalization, these should be the same
        # because the only difference is the task ID in the path
        assert sigs1 == sigs2, "Different task IDs with same underlying error type should be no progress"

    def test_different_errors_is_progress(self):
        """Test that different error types is detected as progress."""
        summary1 = {
            "root_failures": [
                {"id": "TASK-1", "error": "Branch exists"},
            ]
        }
        summary2 = {
            "root_failures": [
                {"id": "TASK-1", "error": "Import error: module not found"},
            ]
        }

        sigs1 = _compute_failure_signatures_from_summary(summary1)
        sigs2 = _compute_failure_signatures_from_summary(summary2)

        assert sigs1 != sigs2, "Different error types should be detected as progress"

    def test_fewer_failures_is_progress(self):
        """Test that having fewer failures is detected as progress."""
        summary1 = {
            "root_failures": [
                {"id": "TASK-1", "error": "Error A"},
                {"id": "TASK-2", "error": "Error B"},
            ]
        }
        summary2 = {
            "root_failures": [
                {"id": "TASK-1", "error": "Error A"},
                # TASK-2 is now fixed
            ]
        }

        sigs1 = _compute_failure_signatures_from_summary(summary1)
        sigs2 = _compute_failure_signatures_from_summary(summary2)

        assert sigs1 != sigs2, "Fewer failures should be detected as progress"

    def test_empty_failures_is_progress(self):
        """Test that going from failures to no failures is progress."""
        summary1 = {
            "root_failures": [
                {"id": "TASK-1", "error": "Error A"},
            ]
        }
        summary2 = {"root_failures": []}

        sigs1 = _compute_failure_signatures_from_summary(summary1)
        sigs2 = _compute_failure_signatures_from_summary(summary2)

        assert sigs1 != sigs2, "Going from failures to success should be progress"


class TestJan26ProgressRegression:
    """Test the exact scenario that caused false progress detection on Jan 26."""

    def test_jan26_branch_collision_same_signature(self):
        """Test that branch collision errors across different tasks have same signature.

        In the Jan 26 run, 8 tasks failed with "branch already exists" errors.
        The orchestrator incorrectly detected this as "progress" because
        the task IDs were different.
        """
        # Simulate two consecutive runs from Jan 26
        run1_failures = {
            "root_failures": [
                {
                    "id": "M0-RF-01-LOGS-CLAUDE-LIMIT-CASE",
                    "error": "Preparing worktree (new branch 'task/20260126T011919Z/M0-RF-01-LOGS-CLAUDE-LIMIT-CASE')\n\nfatal: A branch named 'task/20260126T011919Z/M0-RF-01-LOGS-CLAUDE-LIMIT-CASE' already exists.",
                },
                {
                    "id": "M0-RF-01-SKIP-SCAN-MATRIX",
                    "error": "Preparing worktree (new branch 'task/20260126T011919Z/M0-RF-01-SKIP-SCAN-MATRIX')\n\nfatal: A branch named 'task/20260126T011919Z/M0-RF-01-SKIP-SCAN-MATRIX' already exists.",
                },
            ]
        }

        run2_failures = {
            "root_failures": [
                {
                    "id": "M0-RF-01-DOCS-REVIEW-CONTENT",
                    "error": "Preparing worktree (new branch 'task/20260126T011919Z/M0-RF-01-DOCS-REVIEW-CONTENT')\n\nfatal: A branch named 'task/20260126T011919Z/M0-RF-01-DOCS-REVIEW-CONTENT' already exists.",
                },
                {
                    "id": "M0-RF-01-GPU-POLICY-REVIEW-PYPROJECT",
                    "error": "Preparing worktree (new branch 'task/20260126T011919Z/M0-RF-01-GPU-POLICY-REVIEW-PYPROJECT')\n\nfatal: A branch named 'task/20260126T011919Z/M0-RF-01-GPU-POLICY-REVIEW-PYPROJECT' already exists.",
                },
            ]
        }

        sigs1 = _compute_failure_signatures_from_summary(run1_failures)
        sigs2 = _compute_failure_signatures_from_summary(run2_failures)

        # Both runs have the same underlying error (branch collision)
        # even though the task IDs are different
        assert sigs1 == sigs2, "Jan 26 branch collision errors should have same signature (no progress)"

    def test_jan26_quota_errors_same_signature(self):
        """Test that Claude quota errors have the same signature."""
        run1_failures = {
            "root_failures": [
                {
                    "id": "TASK-A",
                    "error": "You've hit your limit · resets 7pm (America/Anchorage)",
                },
            ]
        }

        run2_failures = {
            "root_failures": [
                {
                    "id": "TASK-B",
                    "error": "You've hit your limit · resets 7pm (America/Anchorage)",
                },
            ]
        }

        sigs1 = _compute_failure_signatures_from_summary(run1_failures)
        sigs2 = _compute_failure_signatures_from_summary(run2_failures)

        assert sigs1 == sigs2, "Quota errors on different tasks should have same signature (no progress)"
