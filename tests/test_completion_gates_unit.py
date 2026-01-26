# SPDX-License-Identifier: MIT
"""Unit tests for tools/completion_gates.py.

Tests the completion gate helpers used by the orchestrator to determine
when tasks can be considered complete. Key functionality tested:
- CompletionGateInputs dataclass
- CompletionGateResult dataclass
- evaluate_completion_gates function
- verify_args_for_completion function
- _milestone_requires_m0 helper function
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.completion_gates import (
    CompletionGateInputs,
    CompletionGateResult,
    _milestone_requires_m0,
    evaluate_completion_gates,
    verify_args_for_completion,
)

# -----------------------------------------------------------------------------
# CompletionGateInputs dataclass tests
# -----------------------------------------------------------------------------


class TestCompletionGateInputs:
    """Tests for CompletionGateInputs dataclass."""

    def test_creates_inputs_with_all_fields(self) -> None:
        """Creates inputs with all required fields."""
        inputs = CompletionGateInputs(
            verify_rc=0,
            git_porcelain="",
            head_sha="abc123",
        )
        assert inputs.verify_rc == 0
        assert inputs.git_porcelain == ""
        assert inputs.head_sha == "abc123"

    def test_inputs_are_frozen(self) -> None:
        """Inputs dataclass is frozen (immutable)."""
        inputs = CompletionGateInputs(
            verify_rc=0,
            git_porcelain="",
            head_sha="abc123",
        )
        with pytest.raises(AttributeError):
            inputs.verify_rc = 1  # type: ignore[misc]

    def test_inputs_equality(self) -> None:
        """Two inputs with same values are equal."""
        inputs1 = CompletionGateInputs(verify_rc=0, git_porcelain="", head_sha="abc")
        inputs2 = CompletionGateInputs(verify_rc=0, git_porcelain="", head_sha="abc")
        assert inputs1 == inputs2

    def test_inputs_inequality(self) -> None:
        """Two inputs with different values are not equal."""
        inputs1 = CompletionGateInputs(verify_rc=0, git_porcelain="", head_sha="abc")
        inputs2 = CompletionGateInputs(verify_rc=1, git_porcelain="", head_sha="abc")
        assert inputs1 != inputs2


# -----------------------------------------------------------------------------
# CompletionGateResult dataclass tests
# -----------------------------------------------------------------------------


class TestCompletionGateResult:
    """Tests for CompletionGateResult dataclass."""

    def test_creates_success_result(self) -> None:
        """Creates successful result."""
        result = CompletionGateResult(ok=True, reason="ok")
        assert result.ok is True
        assert result.reason == "ok"

    def test_creates_failure_result(self) -> None:
        """Creates failure result with reason."""
        result = CompletionGateResult(ok=False, reason="tools.verify failed")
        assert result.ok is False
        assert result.reason == "tools.verify failed"

    def test_result_is_frozen(self) -> None:
        """Result dataclass is frozen (immutable)."""
        result = CompletionGateResult(ok=True, reason="ok")
        with pytest.raises(AttributeError):
            result.ok = False  # type: ignore[misc]

    def test_result_equality(self) -> None:
        """Two results with same values are equal."""
        result1 = CompletionGateResult(ok=True, reason="ok")
        result2 = CompletionGateResult(ok=True, reason="ok")
        assert result1 == result2


# -----------------------------------------------------------------------------
# evaluate_completion_gates tests
# -----------------------------------------------------------------------------


class TestEvaluateCompletionGates:
    """Tests for evaluate_completion_gates function."""

    def test_all_gates_pass(self) -> None:
        """All gates pass with clean inputs."""
        inputs = CompletionGateInputs(
            verify_rc=0,
            git_porcelain="",
            head_sha="abc123def456",
        )
        result = evaluate_completion_gates(inputs)
        assert result.ok is True
        assert result.reason == "ok"

    def test_verify_failure(self) -> None:
        """Fails when verify returns non-zero."""
        inputs = CompletionGateInputs(
            verify_rc=1,
            git_porcelain="",
            head_sha="abc123",
        )
        result = evaluate_completion_gates(inputs)
        assert result.ok is False
        assert "verify" in result.reason.lower()

    def test_verify_failure_with_exit_code_2(self) -> None:
        """Fails when verify returns exit code 2."""
        inputs = CompletionGateInputs(
            verify_rc=2,
            git_porcelain="",
            head_sha="abc123",
        )
        result = evaluate_completion_gates(inputs)
        assert result.ok is False
        assert "verify" in result.reason.lower()

    def test_dirty_working_tree(self) -> None:
        """Fails when git status is not clean."""
        inputs = CompletionGateInputs(
            verify_rc=0,
            git_porcelain=" M foo.py\n",
            head_sha="abc123",
        )
        result = evaluate_completion_gates(inputs)
        assert result.ok is False
        assert "git" in result.reason.lower() or "clean" in result.reason.lower()

    def test_dirty_working_tree_staged_changes(self) -> None:
        """Fails when git has staged changes."""
        inputs = CompletionGateInputs(
            verify_rc=0,
            git_porcelain="M  bar.py\n",
            head_sha="abc123",
        )
        result = evaluate_completion_gates(inputs)
        assert result.ok is False
        assert "git" in result.reason.lower() or "clean" in result.reason.lower()

    def test_dirty_working_tree_untracked_files(self) -> None:
        """Fails when git has untracked files in porcelain output."""
        inputs = CompletionGateInputs(
            verify_rc=0,
            git_porcelain="?? newfile.py\n",
            head_sha="abc123",
        )
        result = evaluate_completion_gates(inputs)
        assert result.ok is False

    def test_no_commit_empty_sha(self) -> None:
        """Fails when head_sha is empty."""
        inputs = CompletionGateInputs(
            verify_rc=0,
            git_porcelain="",
            head_sha="",
        )
        result = evaluate_completion_gates(inputs)
        assert result.ok is False
        assert "commit" in result.reason.lower()

    def test_no_commit_whitespace_sha(self) -> None:
        """Fails when head_sha is only whitespace."""
        inputs = CompletionGateInputs(
            verify_rc=0,
            git_porcelain="",
            head_sha="   \n\t  ",
        )
        result = evaluate_completion_gates(inputs)
        assert result.ok is False
        assert "commit" in result.reason.lower()

    def test_multiple_failures_first_wins(self) -> None:
        """When multiple gates fail, first failure reason is returned."""
        inputs = CompletionGateInputs(
            verify_rc=1,
            git_porcelain=" M foo.py\n",
            head_sha="",
        )
        result = evaluate_completion_gates(inputs)
        assert result.ok is False
        # First check is verify_rc
        assert "verify" in result.reason.lower()

    def test_whitespace_only_porcelain_is_clean(self) -> None:
        """Whitespace-only porcelain output is considered clean."""
        inputs = CompletionGateInputs(
            verify_rc=0,
            git_porcelain="   \n  ",
            head_sha="abc123",
        )
        result = evaluate_completion_gates(inputs)
        # strip() is applied, so whitespace-only should be clean
        assert result.ok is True

    def test_long_sha_accepted(self) -> None:
        """Full 40-character SHA is accepted."""
        inputs = CompletionGateInputs(
            verify_rc=0,
            git_porcelain="",
            head_sha="abc123def456789012345678901234567890abcd",
        )
        result = evaluate_completion_gates(inputs)
        assert result.ok is True


# -----------------------------------------------------------------------------
# verify_args_for_completion tests
# -----------------------------------------------------------------------------


class TestVerifyArgsForCompletion:
    """Tests for verify_args_for_completion function."""

    def test_returns_list(self) -> None:
        """Returns a list of command arguments."""
        args = verify_args_for_completion("M0")
        assert isinstance(args, list)
        assert len(args) > 0

    def test_includes_python_executable(self) -> None:
        """Includes Python executable as first argument."""
        args = verify_args_for_completion("M0")
        assert args[0] == sys.executable

    def test_includes_tools_verify_module(self) -> None:
        """Includes tools.verify module."""
        args = verify_args_for_completion("M0")
        assert "-m" in args
        assert "tools.verify" in args

    def test_includes_strict_git_flag(self) -> None:
        """Includes --strict-git flag."""
        args = verify_args_for_completion("M0")
        assert "--strict-git" in args

    def test_m0_excludes_include_m0(self) -> None:
        """M0 milestone does not include --include-m0 flag."""
        args = verify_args_for_completion("M0")
        assert "--include-m0" not in args

    def test_m1_includes_include_m0(self) -> None:
        """M1 milestone includes --include-m0 flag."""
        args = verify_args_for_completion("M1")
        assert "--include-m0" in args

    def test_m2_includes_include_m0(self) -> None:
        """M2 milestone includes --include-m0 flag."""
        args = verify_args_for_completion("M2")
        assert "--include-m0" in args

    def test_m10_includes_include_m0(self) -> None:
        """M10 milestone includes --include-m0 flag."""
        args = verify_args_for_completion("M10")
        assert "--include-m0" in args

    def test_non_milestone_format(self) -> None:
        """Non-standard milestone format does not include --include-m0."""
        args = verify_args_for_completion("CUSTOM")
        assert "--include-m0" not in args


# -----------------------------------------------------------------------------
# _milestone_requires_m0 tests
# -----------------------------------------------------------------------------


class TestMilestoneRequiresM0:
    """Tests for _milestone_requires_m0 helper function."""

    def test_m0_does_not_require(self) -> None:
        """M0 does not require M0 check."""
        assert _milestone_requires_m0("M0") is False

    def test_m1_requires(self) -> None:
        """M1 requires M0 check."""
        assert _milestone_requires_m0("M1") is True

    def test_m2_requires(self) -> None:
        """M2 requires M0 check."""
        assert _milestone_requires_m0("M2") is True

    def test_m99_requires(self) -> None:
        """M99 requires M0 check."""
        assert _milestone_requires_m0("M99") is True

    def test_non_m_prefix_does_not_require(self) -> None:
        """Non-M prefixed IDs do not require M0 check."""
        assert _milestone_requires_m0("TASK-1") is False
        assert _milestone_requires_m0("1") is False
        assert _milestone_requires_m0("") is False

    def test_invalid_m_format_does_not_require(self) -> None:
        """Invalid M format does not require M0 check."""
        assert _milestone_requires_m0("Mabc") is False
        assert _milestone_requires_m0("M") is False
        assert _milestone_requires_m0("M-1") is False

    def test_lowercase_m_does_not_match(self) -> None:
        """Lowercase 'm' prefix does not match."""
        assert _milestone_requires_m0("m1") is False
        assert _milestone_requires_m0("m2") is False


# -----------------------------------------------------------------------------
# Integration-style tests
# -----------------------------------------------------------------------------


class TestCompletionGatesIntegration:
    """Integration-style tests combining multiple components."""

    def test_typical_success_scenario(self) -> None:
        """Typical successful completion scenario."""
        # Simulate successful verify, clean git, valid commit
        inputs = CompletionGateInputs(
            verify_rc=0,
            git_porcelain="",
            head_sha="abc123def456789012345678901234567890abcd",
        )
        result = evaluate_completion_gates(inputs)
        assert result.ok is True

        # Args should be generated correctly for M1
        args = verify_args_for_completion("M1")
        assert "--include-m0" in args

    def test_typical_failure_scenario_verify(self) -> None:
        """Typical failure when verify fails."""
        inputs = CompletionGateInputs(
            verify_rc=1,
            git_porcelain="",
            head_sha="abc123",
        )
        result = evaluate_completion_gates(inputs)
        assert result.ok is False
        # Should indicate verify failed
        assert "verify" in result.reason.lower()

    def test_typical_failure_scenario_uncommitted(self) -> None:
        """Typical failure with uncommitted changes."""
        inputs = CompletionGateInputs(
            verify_rc=0,
            git_porcelain="MM src/main.py\n A  tests/test_new.py\n",
            head_sha="abc123",
        )
        result = evaluate_completion_gates(inputs)
        assert result.ok is False
        # Should indicate git is not clean
        assert "git" in result.reason.lower() or "clean" in result.reason.lower()
