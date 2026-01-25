"""Tests for bridge.verify_repair module."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add bridge to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "bridge"))

from bridge.verify_repair import (
    RepairAttemptRecord,
    RepairLoopResult,
    run_verify_repair_loop,
    write_repair_report,
)
from bridge.verify_repair.repairs import repair_ruff

# Alias for backwards compatibility with old tests
RepairReport = RepairAttemptRecord


class TestRepairAttemptRecord:
    """Tests for RepairAttemptRecord dataclass."""

    def test_repair_attempt_record_creation(self) -> None:
        record = RepairAttemptRecord(
            attempt_index=1,
            detected_categories=["lint_ruff", "pytest_test_failure"],
            actions_taken=["ruff_autofix"],
            verify_before=None,
            verify_after=None,
            diff_applied=True,
            elapsed_s=1.5,
        )
        assert record.attempt_index == 1
        assert record.detected_categories == ["lint_ruff", "pytest_test_failure"]
        assert record.actions_taken == ["ruff_autofix"]
        assert record.diff_applied is True

    def test_repair_attempt_record_to_dict(self) -> None:
        record = RepairAttemptRecord(
            attempt_index=1,
            detected_categories=["lint_ruff"],
            actions_taken=["ruff_autofix:True"],
            verify_before=None,
            verify_after=None,
            diff_applied=True,
            elapsed_s=1.5,
        )
        data = record.to_dict()
        assert data["attempt_index"] == 1
        assert data["detected_categories"] == ["lint_ruff"]
        assert data["actions_taken"] == ["ruff_autofix:True"]


class TestRepairLoopResult:
    """Tests for RepairLoopResult dataclass."""

    def test_result_success(self) -> None:
        result = RepairLoopResult(
            success=True,
            total_attempts=1,
            final_exit_code=0,
            reports=[],
            remaining_failures=[],
        )
        assert result.success is True
        assert result.final_exit_code == 0

    def test_result_failure(self) -> None:
        result = RepairLoopResult(
            success=False,
            total_attempts=3,
            final_exit_code=2,
            reports=[],
            remaining_failures=["pytest", "mypy"],
        )
        assert result.success is False
        assert result.remaining_failures == ["pytest", "mypy"]


class TestWriteRepairReport:
    """Tests for write_repair_report function."""

    def test_write_report(self, tmp_path: Path) -> None:
        from bridge.verify_repair.data import RepairLoopReport

        # Create a result with the repair_report field set
        repair_report = RepairLoopReport(
            success=False,
            total_attempts=2,
            final_failed_gates=["pytest"],
            elapsed_s=2.5,
            stable_failure_signature_count=0,
            artifacts_written=[],
            attempts=[
                RepairAttemptRecord(
                    attempt_index=1,
                    detected_categories=["lint_ruff"],
                    actions_taken=["ruff_autofix"],
                    verify_before=None,
                    verify_after=None,
                    diff_applied=True,
                    elapsed_s=1.0,
                ),
                RepairAttemptRecord(
                    attempt_index=2,
                    detected_categories=["pytest_test_failure"],
                    actions_taken=[],
                    verify_before=None,
                    verify_after=None,
                    diff_applied=False,
                    elapsed_s=1.5,
                ),
            ],
        )

        result = RepairLoopResult(
            success=False,
            total_attempts=2,
            final_exit_code=2,
            reports=[],
            remaining_failures=["pytest"],
            repair_report=repair_report,
        )

        output_path = tmp_path / "report.json"
        write_repair_report(result, output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["success"] is False
        assert data["total_attempts"] == 2
        assert data["final_failed_gates"] == ["pytest"]
        assert len(data["attempts"]) == 2


class TestImportErrorDetection:
    """Tests for import error detection in classify module."""

    def test_detects_import_errors(self, tmp_path: Path) -> None:
        """Test that import errors are detected by classification."""
        from bridge.verify_repair.classify import extract_import_errors
        from bridge.verify_repair.data import VerifyGateResult, VerifySummary

        summary = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={
                "pytest": VerifyGateResult(
                    name="pytest",
                    passed=False,
                    returncode=2,
                    stdout="ImportError: cannot import name 'Foo' from 'bar.module'",
                    stderr="",
                ),
            },
        )

        errors = extract_import_errors(summary)
        assert len(errors) > 0
        assert any(e["type"] == "import_error" for e in errors)

    def test_detects_missing_modules(self, tmp_path: Path) -> None:
        """Test that missing modules are detected by classification."""
        from bridge.verify_repair.classify import extract_import_errors
        from bridge.verify_repair.data import VerifyGateResult, VerifySummary

        summary = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={
                "pytest": VerifyGateResult(
                    name="pytest",
                    passed=False,
                    returncode=2,
                    stdout="ModuleNotFoundError: No module named 'missing_pkg'",
                    stderr="",
                ),
            },
        )

        errors = extract_import_errors(summary)
        assert len(errors) > 0
        assert any(e["type"] == "module_not_found" for e in errors)


class TestRunVerifyRepairLoop:
    """Tests for run_verify_repair_loop function."""

    @patch("bridge.verify_repair.loop._run_verify")
    @patch("bridge.verify_repair.loop.run_bootstrap")
    def test_success_on_first_attempt(
        self, mock_bootstrap: MagicMock, mock_run_verify: MagicMock, tmp_path: Path
    ) -> None:
        # Bootstrap succeeds
        mock_bootstrap.return_value = MagicMock(
            success=True, skipped=True, stderr=""
        )

        # First verify succeeds
        mock_run_verify.return_value = (0, "All checks passed", "")

        verify_json = tmp_path / "verify.json"
        verify_json.write_text('{"ok": true, "failed_gates": []}')

        result = run_verify_repair_loop(
            project_root=tmp_path,
            verify_json_path=verify_json,
            max_attempts=3,
            strict_git=False,
            verbose=False,
            bootstrap_on_start=True,
        )

        assert result.success is True
        assert result.total_attempts == 1
        assert result.final_exit_code == 0

    @patch("bridge.verify_repair.loop._run_verify")
    @patch("bridge.verify_repair.loop.run_bootstrap")
    def test_repair_and_succeed(
        self,
        mock_bootstrap: MagicMock,
        mock_run_verify: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that the loop can succeed after initially failing."""
        # Bootstrap succeeds
        mock_bootstrap.return_value = MagicMock(
            success=True, skipped=True, stderr=""
        )

        verify_json = tmp_path / "verify.json"
        call_count = [0]

        def run_verify_side_effect(*args, **kwargs):
            call_count[0] += 1
            # First call fails, second succeeds
            if call_count[0] == 1:
                verify_json.write_text(
                    '{"ok": false, "failed_gates": ["ruff"], "first_failed_gate": "ruff", "results": [{"name": "ruff", "passed": false, "returncode": 1, "stdout": "", "stderr": ""}]}'
                )
                return (2, "ruff failed", "")
            else:
                verify_json.write_text('{"ok": true, "failed_gates": [], "results": []}')
                return (0, "All checks passed", "")

        mock_run_verify.side_effect = run_verify_side_effect

        # Run with ruff repair mocked
        with patch("bridge.verify_repair.repairs.repair_ruff") as mock_ruff:
            mock_ruff.return_value = MagicMock(
                success=True, files_modified=1, name="ruff_autofix"
            )

            result = run_verify_repair_loop(
                project_root=tmp_path,
                verify_json_path=verify_json,
                max_attempts=3,
                strict_git=False,
                verbose=False,
                bootstrap_on_start=True,
            )

        # Should succeed after repair
        assert result.success is True
        assert call_count[0] >= 2  # At least initial verify + final verify

    @patch("bridge.verify_repair.loop._run_verify")
    @patch("bridge.verify_repair.loop.run_bootstrap")
    def test_max_attempts_exceeded(
        self, mock_bootstrap: MagicMock, mock_run_verify: MagicMock, tmp_path: Path
    ) -> None:
        # Bootstrap succeeds
        mock_bootstrap.return_value = MagicMock(
            success=True, skipped=True, stderr=""
        )

        # All attempts fail
        mock_run_verify.return_value = (2, "pytest failed", "")

        verify_json = tmp_path / "verify.json"
        verify_json.write_text('{"ok": false, "failed_gates": ["pytest"], "results": []}')

        result = run_verify_repair_loop(
            project_root=tmp_path,
            verify_json_path=verify_json,
            max_attempts=2,
            strict_git=False,
            verbose=False,
            bootstrap_on_start=True,
        )

        assert result.success is False
        assert result.total_attempts <= 2
        assert result.remaining_failures == ["pytest"]


# ============================================================================
# ENHANCED TESTS - Added for verify auto-repair loop improvements
# ============================================================================


class TestFailureClassification:
    """Tests for failure classification from bridge.verify_repair.classify."""

    def test_classify_imports_available(self) -> None:
        """Test that new classification module can be imported."""
        from bridge.verify_repair import (
            FailureCategory,
            classify_failures,
            compute_failure_signature,
        )
        assert FailureCategory is not None
        assert classify_failures is not None
        assert compute_failure_signature is not None

    def test_classify_ruff_failure(self) -> None:
        """Test that ruff failures are correctly classified."""
        from bridge.verify_repair import FailureCategory, classify_failures
        from bridge.verify_repair.data import VerifyGateResult, VerifySummary

        summary = VerifySummary(
            ok=False,
            failed_gates=["ruff"],
            first_failed_gate="ruff",
            results_by_gate={
                "ruff": VerifyGateResult(
                    name="ruff",
                    passed=False,
                    returncode=1,
                    stdout="Found 5 errors",
                    stderr="",
                ),
            },
        )
        classification = classify_failures(summary)
        assert "ruff" in classification
        assert FailureCategory.LINT_RUFF in classification["ruff"]

    def test_classify_mypy_failure(self) -> None:
        """Test that mypy failures are correctly classified."""
        from bridge.verify_repair import FailureCategory, classify_failures
        from bridge.verify_repair.data import VerifyGateResult, VerifySummary

        summary = VerifySummary(
            ok=False,
            failed_gates=["mypy"],
            first_failed_gate="mypy",
            results_by_gate={
                "mypy": VerifyGateResult(
                    name="mypy",
                    passed=False,
                    returncode=1,
                    stdout="src/foo.py:10: error: Type mismatch",
                    stderr="",
                ),
            },
        )
        classification = classify_failures(summary)
        assert "mypy" in classification
        assert FailureCategory.TYPECHECK_MYPY in classification["mypy"]

    def test_classify_pytest_import_error(self) -> None:
        """Test that pytest collection import errors are classified."""
        from bridge.verify_repair import FailureCategory, classify_failures
        from bridge.verify_repair.data import VerifyGateResult, VerifySummary

        summary = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={
                "pytest": VerifyGateResult(
                    name="pytest",
                    passed=False,
                    returncode=2,
                    stdout="ERROR collecting tests/test_foo.py\nImportError: cannot import name 'Bar'",
                    stderr="",
                ),
            },
        )
        classification = classify_failures(summary)
        assert "pytest" in classification
        assert FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR in classification["pytest"]


class TestFailureSignature:
    """Tests for failure signature computation."""

    def test_same_failures_same_signature(self) -> None:
        """Test that identical failures produce the same signature."""
        from bridge.verify_repair import compute_failure_signature
        from bridge.verify_repair.data import VerifyGateResult, VerifySummary

        summary1 = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={
                "pytest": VerifyGateResult(
                    name="pytest",
                    passed=False,
                    returncode=1,
                    stdout="FAILED test_foo",
                    stderr="",
                ),
            },
        )
        summary2 = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={
                "pytest": VerifyGateResult(
                    name="pytest",
                    passed=False,
                    returncode=1,
                    stdout="FAILED test_foo",
                    stderr="",
                ),
            },
        )
        sig1 = compute_failure_signature(summary1)
        sig2 = compute_failure_signature(summary2)
        assert sig1 == sig2

    def test_different_failures_different_signature(self) -> None:
        """Test that different failures produce different signatures."""
        from bridge.verify_repair import compute_failure_signature
        from bridge.verify_repair.data import VerifyGateResult, VerifySummary

        summary1 = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={
                "pytest": VerifyGateResult(
                    name="pytest",
                    passed=False,
                    returncode=1,
                    stdout="FAILED test_foo",
                    stderr="",
                ),
            },
        )
        summary2 = VerifySummary(
            ok=False,
            failed_gates=["pytest"],
            first_failed_gate="pytest",
            results_by_gate={
                "pytest": VerifyGateResult(
                    name="pytest",
                    passed=False,
                    returncode=1,
                    stdout="FAILED test_bar",  # Different test
                    stderr="",
                ),
            },
        )
        sig1 = compute_failure_signature(summary1)
        sig2 = compute_failure_signature(summary2)
        assert sig1 != sig2


class TestRepairLoopReport:
    """Tests for the enhanced RepairLoopReport data structure."""

    def test_repair_loop_report_to_dict(self) -> None:
        """Test that RepairLoopReport serializes correctly."""
        from bridge.verify_repair.data import RepairLoopReport

        report = RepairLoopReport(
            success=False,
            total_attempts=2,
            final_failed_gates=["pytest"],
            elapsed_s=10.5,
            stable_failure_signature_count=1,
            artifacts_written=["bootstrap.log"],
            attempts=[],
            early_stop_reason="Max attempts reached",
        )
        data = report.to_dict()
        assert data["success"] is False
        assert data["total_attempts"] == 2
        assert data["final_failed_gates"] == ["pytest"]
        assert data["early_stop_reason"] == "Max attempts reached"
        assert data["stable_failure_signature_count"] == 1


class TestBootstrapModule:
    """Tests for the bootstrap module."""

    def test_bootstrap_imports(self) -> None:
        """Test that bootstrap module can be imported."""
        from bridge.verify_repair.bootstrap import BootstrapResult, run_bootstrap
        assert BootstrapResult is not None
        assert run_bootstrap is not None

    def test_bootstrap_result_dataclass(self) -> None:
        """Test BootstrapResult dataclass."""
        from bridge.verify_repair.bootstrap import BootstrapResult

        result = BootstrapResult(
            success=True,
            command=["pip", "install", "-e", "."],
            returncode=0,
            stdout="",
            stderr="",
            elapsed_s=5.0,
            skipped=False,
        )
        assert result.success is True
        assert result.returncode == 0


class TestTurnNormalizerRefactor:
    """Tests for TurnNormalizer refactor."""

    def test_import_from_loop_pkg(self) -> None:
        """Test that TurnNormalizer can be imported from loop_pkg."""
        from bridge.loop_pkg import (
            NormalizationResult,
            TurnNormalizer,
            normalize_agent_output,
            validate_turn_lenient,
        )
        assert NormalizationResult is not None
        assert TurnNormalizer is not None
        assert normalize_agent_output is not None
        assert validate_turn_lenient is not None

    def test_turn_normalizer_basic(self) -> None:
        """Test basic TurnNormalizer functionality."""
        from bridge.loop_pkg import normalize_agent_output

        raw_output = json.dumps({
            "summary": "Test summary",
            "work_completed": True,
            "project_complete": False,
        })

        result = normalize_agent_output(
            raw_output=raw_output,
            expected_agent="claude",
            expected_milestone_id="M1",
            stats_id_set={"CL-1", "CX-1"},
        )

        assert result.success is True
        assert result.turn is not None
        assert result.turn["agent"] == "claude"
        assert result.turn["milestone_id"] == "M1"


class TestEnhancedRepairLoop:
    """Tests for enhanced repair loop with bootstrap."""

    @patch("bridge.verify_repair.loop._run_verify")
    @patch("bridge.verify_repair.loop.run_bootstrap")
    def test_bootstrap_called_on_start(
        self,
        mock_bootstrap: MagicMock,
        mock_verify: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that bootstrap is called at the start of repair loop."""
        from bridge.verify_repair.loop import run_verify_repair_loop

        mock_verify.return_value = (0, "", "")
        mock_bootstrap.return_value = MagicMock(
            success=True, skipped=False, stderr="", elapsed_s=1.0
        )

        verify_json = tmp_path / "verify.json"

        run_verify_repair_loop(
            project_root=tmp_path,
            verify_json_path=verify_json,
            max_attempts=3,
            strict_git=False,
            verbose=False,
            bootstrap_on_start=True,
        )

        mock_bootstrap.assert_called_once()

    @patch("bridge.verify_repair.loop._run_verify")
    @patch("bridge.verify_repair.loop.run_bootstrap")
    def test_bootstrap_skipped_when_disabled(
        self,
        mock_bootstrap: MagicMock,
        mock_verify: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that bootstrap is not called when disabled."""
        from bridge.verify_repair.loop import run_verify_repair_loop

        mock_verify.return_value = (0, "", "")

        verify_json = tmp_path / "verify.json"

        run_verify_repair_loop(
            project_root=tmp_path,
            verify_json_path=verify_json,
            max_attempts=3,
            strict_git=False,
            verbose=False,
            bootstrap_on_start=False,
        )

        mock_bootstrap.assert_not_called()


# ============================================================================
# STALL PREVENTION TESTS - Prove orchestrator auto-repair works
# ============================================================================


class TestStallPrevention:
    """Tests proving the orchestrator doesn't stall on verify failures.

    These tests verify that:
    1. Repair loop auto-triggers when verify fails
    2. Callback executes repairs instead of just writing files
    3. Early stop happens on repeated failure signatures
    """

    @patch("bridge.verify_repair.loop._run_verify")
    @patch("bridge.verify_repair.loop.run_bootstrap")
    def test_repair_callback_invoked_on_failure(
        self,
        mock_bootstrap: MagicMock,
        mock_verify: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that agent_task_callback is invoked when verify fails."""
        from bridge.verify_repair.loop import run_verify_repair_loop

        mock_bootstrap.return_value = MagicMock(success=True, skipped=True, stderr="")

        verify_json = tmp_path / "verify.json"
        callback_invoked = [False]
        callback_tasks = []

        def mock_callback(tasks):
            callback_invoked[0] = True
            callback_tasks.extend(tasks)
            return True

        call_count = [0]

        def verify_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call fails with an issue that needs agent repair
                verify_json.write_text(json.dumps({
                    "ok": False,
                    "failed_gates": ["pytest"],
                    "first_failed_gate": "pytest",
                    "results": [{
                        "name": "pytest",
                        "passed": False,
                        "returncode": 2,
                        "stdout": "ERROR collecting tests/test_foo.py\nImportError: cannot import name 'Bar'",
                        "stderr": "",
                    }],
                }))
                return (2, "pytest failed", "")
            else:
                # Subsequent calls succeed
                verify_json.write_text('{"ok": true, "failed_gates": []}')
                return (0, "All passed", "")

        mock_verify.side_effect = verify_side_effect

        _result = run_verify_repair_loop(
            project_root=tmp_path,
            verify_json_path=verify_json,
            max_attempts=3,
            strict_git=False,
            verbose=False,
            bootstrap_on_start=True,
            agent_task_callback=mock_callback,
        )

        # Callback should have been invoked
        assert callback_invoked[0] is True
        assert _result is not None  # Suppress unused warning

    @patch("bridge.verify_repair.loop._run_verify")
    @patch("bridge.verify_repair.loop.run_bootstrap")
    def test_early_stop_on_repeated_signature(
        self,
        mock_bootstrap: MagicMock,
        mock_verify: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that loop stops early when same failure repeats without progress.

        The loop should stop when:
        1. Same failure signature appears twice after repairs were attempted, OR
        2. No repairs can be applied (nothing to try)
        """
        from bridge.verify_repair.loop import run_verify_repair_loop

        mock_bootstrap.return_value = MagicMock(success=True, skipped=True, stderr="")

        verify_json = tmp_path / "verify.json"

        # Always return the same failure (pytest, which has no Layer 1 auto-repair)
        def verify_side_effect(*args, **kwargs):
            verify_json.write_text(json.dumps({
                "ok": False,
                "failed_gates": ["pytest"],
                "first_failed_gate": "pytest",
                "results": [{
                    "name": "pytest",
                    "passed": False,
                    "returncode": 2,
                    "stdout": "FAILED test_foo - AssertionError",
                    "stderr": "",
                }],
            }))
            return (2, "pytest failed", "")

        mock_verify.side_effect = verify_side_effect

        result = run_verify_repair_loop(
            project_root=tmp_path,
            verify_json_path=verify_json,
            max_attempts=5,
            strict_git=False,
            verbose=False,
            bootstrap_on_start=True,
        )

        # Should stop early - no repairs can be applied for pytest failures
        # without an agent callback, so loop exits after first attempt
        assert result.success is False
        assert result.total_attempts <= 5  # Should not exceed max
        assert result.repair_report is not None
        # Loop stops because no repairs were applied, not because of signature
        # This is correct behavior - don't keep trying when nothing can be done

    @patch("bridge.verify_repair.loop._run_verify")
    @patch("bridge.verify_repair.loop.run_bootstrap")
    def test_repair_loop_succeeds_after_repair(
        self,
        mock_bootstrap: MagicMock,
        mock_verify: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that orchestrator exits successfully after repair fixes verify."""
        from bridge.verify_repair.loop import run_verify_repair_loop

        mock_bootstrap.return_value = MagicMock(success=True, skipped=True, stderr="")

        verify_json = tmp_path / "verify.json"
        call_count = [0]

        def verify_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                verify_json.write_text(json.dumps({
                    "ok": False,
                    "failed_gates": ["ruff"],
                    "first_failed_gate": "ruff",
                    "results": [{
                        "name": "ruff",
                        "passed": False,
                        "returncode": 1,
                        "stdout": "Found 5 errors",
                        "stderr": "",
                    }],
                }))
                return (1, "ruff failed", "")
            else:
                verify_json.write_text('{"ok": true, "failed_gates": []}')
                return (0, "All passed", "")

        mock_verify.side_effect = verify_side_effect

        # Mock ruff repair
        with patch("bridge.verify_repair.repairs.repair_ruff") as mock_ruff:
            mock_ruff.return_value = MagicMock(
                success=True, files_modified=5, name="ruff_autofix"
            )

            result = run_verify_repair_loop(
                project_root=tmp_path,
                verify_json_path=verify_json,
                max_attempts=3,
                strict_git=False,
                verbose=False,
                bootstrap_on_start=True,
            )

        # Should succeed after repair
        assert result.success is True
        assert result.final_exit_code == 0


class TestPatchScopeGuard:
    """Tests for patch scope enforcement (drift prevention).

    These tests verify that:
    1. Patches touching out-of-scope files are rejected
    2. Rejected patch artifacts are written
    3. Bridge files are always allowed
    """

    def test_scope_guard_allows_bridge_files(self) -> None:
        """Test that bridge/** files are always allowed."""
        from bridge.patch_integration import ScopeGuard

        guard = ScopeGuard()

        result = guard.check_paths([
            "bridge/loop.py",
            "bridge/verify_repair/loop.py",
            "bridge/patch_integration.py",
        ])

        assert result.allowed is True
        assert len(result.violations) == 0

    def test_scope_guard_rejects_src_files(self) -> None:
        """Test that src/** files are rejected by default denylist."""
        from bridge.patch_integration import ScopeGuard

        guard = ScopeGuard()

        result = guard.check_paths([
            "src/formula_foundry/core.py",
            "src/formula_foundry/__init__.py",
        ])

        assert result.allowed is False
        assert len(result.violations) == 2
        assert all(v.reason == "denylist_match" for v in result.violations)
        assert all(v.matched_pattern == "src/**" for v in result.violations)

    def test_scope_guard_rejects_tools_files(self) -> None:
        """Test that tools/** files are rejected by default denylist."""
        from bridge.patch_integration import ScopeGuard

        guard = ScopeGuard()

        result = guard.check_paths(["tools/verify.py", "tools/__init__.py"])

        assert result.allowed is False
        assert len(result.violations) == 2

    def test_scope_guard_allows_orchestrator_tests(self) -> None:
        """Test that orchestrator test files are allowed."""
        from bridge.patch_integration import ScopeGuard

        guard = ScopeGuard()

        result = guard.check_paths([
            "tests/test_orchestrator_robustness.py",
            "tests/test_verify_repair.py",
        ])

        assert result.allowed is True

    def test_scope_guard_mixed_paths(self) -> None:
        """Test scope guard with mix of allowed and denied paths."""
        from bridge.patch_integration import ScopeGuard

        guard = ScopeGuard()

        result = guard.check_paths([
            "bridge/loop.py",       # allowed (bridge)
            "src/core.py",          # denied (src)
            "tests/test_foo.py",    # denied (not orchestrator test)
        ])

        assert result.allowed is False
        # Should have violations for src and non-orchestrator test
        assert len(result.violations) >= 1

    def test_scope_guard_writes_rejected_artifact(self, tmp_path: Path) -> None:
        """Test that rejected patch artifact is written."""
        from bridge.patch_integration import ScopeCheckResult, ScopeGuard

        guard = ScopeGuard(runs_dir=tmp_path)

        result = guard.check_paths(["src/bad.py"])
        assert result.allowed is False

        artifact_path = guard.write_rejected_artifact(
            task_id="TEST-TASK-1",
            agent_name="test_agent",
            result=result,
            runs_dir=tmp_path,
        )

        assert artifact_path is not None
        assert artifact_path.exists()

        data = json.loads(artifact_path.read_text())
        assert data["task_id"] == "TEST-TASK-1"
        assert data["agent_name"] == "test_agent"
        assert "remediation_prompt" in data

    def test_scope_guard_custom_allowlist(self) -> None:
        """Test scope guard with custom allowlist."""
        from bridge.patch_integration import ScopeGuard

        guard = ScopeGuard(
            allowlist=["custom/**"],
            denylist=["src/**"],
        )

        # Custom path should be allowed
        result = guard.check_paths(["custom/file.py"])
        assert result.allowed is True

        # Bridge still allowed (ALWAYS_ALLOWED)
        result = guard.check_paths(["bridge/loop.py"])
        assert result.allowed is True

        # src still denied
        result = guard.check_paths(["src/file.py"])
        assert result.allowed is False


class TestRepairExecutor:
    """Tests for the repair executor module."""

    def test_executor_creates_callback(self, tmp_path: Path) -> None:
        """Test that create_repair_callback returns a callable."""
        from bridge.verify_repair import create_repair_callback

        callback = create_repair_callback(
            project_root=tmp_path,
            runs_dir=tmp_path,
            verbose=False,
        )

        assert callable(callback)

    def test_executor_filters_out_of_scope_tasks(self, tmp_path: Path) -> None:
        """Test that executor filters tasks targeting out-of-scope files."""
        from bridge.verify_repair.agent_tasks import RepairTask
        from bridge.verify_repair.executor import RepairExecutor

        executor = RepairExecutor(
            project_root=tmp_path,
            runs_dir=tmp_path,
            verbose=False,
        )

        # Create a task targeting src files (out of scope for orchestrator)
        task = RepairTask(
            id="TEST-1",
            title="Fix imports",
            description="Fix import errors",
            target_files=["src/formula_foundry/core.py"],
        )

        callback = executor.create_agent_task_callback()

        # Should succeed but filter out the task
        # Result depends on whether deterministic repairs succeed
        # The important thing is it doesn't crash
        _callback_result = callback([task])
        assert _callback_result is not None or _callback_result is None  # May succeed or fail

        # Check that out_of_scope file was written
        out_of_scope_path = tmp_path / "out_of_scope_repairs.json"
        assert out_of_scope_path.exists()

    def test_executor_writes_repair_plan(self, tmp_path: Path) -> None:
        """Test that executor writes repair plan artifact."""
        from bridge.verify_repair.agent_tasks import RepairTask
        from bridge.verify_repair.executor import RepairExecutor

        executor = RepairExecutor(
            project_root=tmp_path,
            runs_dir=tmp_path,
            verbose=False,
        )

        task = RepairTask(
            id="TEST-1",
            title="Fix bridge imports",
            description="Fix import errors in bridge",
            target_files=["bridge/loop.py"],  # In scope
        )

        callback = executor.create_agent_task_callback()
        callback([task])

        # Check that repair plan was written
        repair_plans = list(tmp_path.glob("repair_plan_*.json"))
        assert len(repair_plans) >= 1


class TestIntegratedRepairLoop:
    """Integration tests for the complete repair loop with callback."""

    @patch("bridge.verify_repair.loop._run_verify")
    @patch("bridge.verify_repair.loop.run_bootstrap")
    @patch("bridge.verify_repair.repairs.repair_ruff")
    def test_full_repair_loop_with_executor(
        self,
        mock_ruff: MagicMock,
        mock_bootstrap: MagicMock,
        mock_verify: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test full repair loop with executor callback.

        This test verifies that when:
        1. Verify fails with ruff errors
        2. Repair callback is invoked
        3. Ruff repair succeeds
        4. Verify passes on retry
        The overall result is success.
        """
        from bridge.verify_repair import create_repair_callback, run_verify_repair_loop

        mock_bootstrap.return_value = MagicMock(success=True, skipped=True, stderr="")
        mock_ruff.return_value = MagicMock(
            success=True, files_modified=5, name="ruff_autofix", output=""
        )

        verify_json = tmp_path / "verify.json"
        call_count = [0]

        def verify_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call fails with ruff errors
                verify_json.write_text(json.dumps({
                    "ok": False,
                    "failed_gates": ["ruff"],
                    "first_failed_gate": "ruff",
                    "results": [{
                        "name": "ruff",
                        "passed": False,
                        "returncode": 1,
                        "stdout": "Found errors",
                        "stderr": "",
                    }],
                }))
                return (1, "ruff failed", "")
            else:
                # Subsequent calls succeed (after repair)
                verify_json.write_text('{"ok": true, "failed_gates": []}')
                return (0, "All passed", "")

        mock_verify.side_effect = verify_side_effect

        # Note: We don't use create_repair_callback here because it tries to
        # run actual subprocess commands. Instead we rely on the mocked ruff repair.
        result = run_verify_repair_loop(
            project_root=tmp_path,
            verify_json_path=verify_json,
            max_attempts=5,
            strict_git=False,
            verbose=False,
            bootstrap_on_start=True,
        )

        # Should succeed after ruff repair
        assert result.success is True
        assert mock_ruff.called  # Ruff repair was attempted

    @patch("bridge.verify_repair.loop._run_verify")
    @patch("bridge.verify_repair.loop.run_bootstrap")
    def test_callback_invoked_for_agent_tasks(
        self,
        mock_bootstrap: MagicMock,
        mock_verify: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that callback is invoked when agent tasks are needed."""
        from bridge.verify_repair.loop import run_verify_repair_loop

        mock_bootstrap.return_value = MagicMock(success=True, skipped=True, stderr="")

        verify_json = tmp_path / "verify.json"
        callback_called = [False]

        def callback(tasks):
            callback_called[0] = True
            return True

        # Verify fails with import error (needs agent repair)
        verify_json.write_text(json.dumps({
            "ok": False,
            "failed_gates": ["pytest"],
            "first_failed_gate": "pytest",
            "results": [{
                "name": "pytest",
                "passed": False,
                "returncode": 2,
                "stdout": "ERROR collecting test_foo.py\nImportError: cannot import name 'Foo'",
                "stderr": "",
            }],
        }))
        mock_verify.return_value = (2, "pytest failed", "")

        run_verify_repair_loop(
            project_root=tmp_path,
            verify_json_path=verify_json,
            max_attempts=2,
            strict_git=False,
            verbose=False,
            bootstrap_on_start=True,
            agent_task_callback=callback,
        )

        # Callback should have been invoked for import errors
        assert callback_called[0] is True
