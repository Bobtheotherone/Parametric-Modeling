"""Tests for calibration regression with metric-based comparison.

This module tests REQ-M2-022: Calibration regression is metric-based
(not bitwise) and merge-blocking for threshold violations.

The tests verify:
1. Regression uses metric-based comparison, not bitwise data comparison
2. Thresholds can be configured and are enforced
3. Threshold violations block merges when configured
4. Small numerical differences don't cause false failures
"""

from __future__ import annotations

import pytest
from formula_foundry.calibration.library import REQUIRED_CALIBRATION_IDS
from formula_foundry.calibration.regression import (
    DEFAULT_THRESHOLDS,
    STRICT_THRESHOLDS,
    MetricComparison,
    RegressionComparison,
    RegressionResult,
    RegressionStatus,
    RegressionThresholds,
    check_regression_gate,
    compare_calibration_result,
    compare_calibration_results,
)
from formula_foundry.calibration.runner import (
    CalibrationResult,
    CalibrationRun,
    run_calibration_suite,
)


class TestRegressionThresholds:
    """Test RegressionThresholds configuration."""

    def test_default_thresholds_are_valid(self) -> None:
        """Default thresholds should be reasonable values."""
        thresholds = DEFAULT_THRESHOLDS
        assert 0 < thresholds.metric_delta_max <= 1
        assert thresholds.s11_mean_delta_db > 0
        assert thresholds.s21_mean_delta_db > 0
        assert 0 <= thresholds.pass_rate_delta_max <= 1
        assert thresholds.block_on_threshold_violation is True

    def test_strict_thresholds_are_stricter(self) -> None:
        """Strict thresholds should be tighter than defaults."""
        assert STRICT_THRESHOLDS.metric_delta_max < DEFAULT_THRESHOLDS.metric_delta_max
        assert STRICT_THRESHOLDS.s11_mean_delta_db < DEFAULT_THRESHOLDS.s11_mean_delta_db
        assert STRICT_THRESHOLDS.s21_mean_delta_db < DEFAULT_THRESHOLDS.s21_mean_delta_db

    def test_custom_thresholds_validation(self) -> None:
        """Custom thresholds must have valid values."""
        # Valid custom thresholds
        custom = RegressionThresholds(
            metric_delta_max=0.2,
            s11_mean_delta_db=2.0,
            s21_mean_delta_db=2.0,
            pass_rate_delta_max=0.1,
        )
        assert custom.metric_delta_max == 0.2

        # Invalid metric_delta_max
        with pytest.raises(ValueError, match="metric_delta_max"):
            RegressionThresholds(metric_delta_max=0)

        with pytest.raises(ValueError, match="metric_delta_max"):
            RegressionThresholds(metric_delta_max=1.5)

        # Invalid s11_mean_delta_db
        with pytest.raises(ValueError, match="s11_mean_delta_db"):
            RegressionThresholds(s11_mean_delta_db=-1.0)


class TestMetricComparison:
    """Test individual metric comparisons."""

    def test_metric_within_threshold_passes(self) -> None:
        """Metrics within threshold should pass."""
        cmp = MetricComparison(
            metric_name="test_metric",
            baseline_value=0.9,
            current_value=0.92,
            delta=0.02,
            threshold=0.1,
            status=RegressionStatus.PASS,
            message="test",
        )
        assert cmp.status == RegressionStatus.PASS
        assert cmp.delta == pytest.approx(0.02)

    def test_metric_exceeding_threshold_fails(self) -> None:
        """Metrics exceeding threshold should fail."""
        cmp = MetricComparison(
            metric_name="metric",  # Use actual metric name
            baseline_value=0.9,
            current_value=0.5,
            delta=-0.4,
            threshold=0.1,
            status=RegressionStatus.FAIL,
            message="test",
        )
        assert cmp.status == RegressionStatus.FAIL
        assert cmp.is_degradation is True  # metric decreased is degradation

    def test_is_degradation_for_metric(self) -> None:
        """is_degradation should detect when metric worsens."""
        # Metric decreased (bad for pass_rate, metric, mean_metric)
        cmp = MetricComparison(
            metric_name="metric",
            baseline_value=0.9,
            current_value=0.8,
            delta=-0.1,
            threshold=0.1,
            status=RegressionStatus.WARN,
            message="test",
        )
        assert cmp.is_degradation is True

        # Metric increased (good)
        cmp_better = MetricComparison(
            metric_name="metric",
            baseline_value=0.8,
            current_value=0.9,
            delta=0.1,
            threshold=0.1,
            status=RegressionStatus.PASS,
            message="test",
        )
        assert cmp_better.is_degradation is False

    def test_to_dict_serialization(self) -> None:
        """MetricComparison should serialize to dict."""
        cmp = MetricComparison(
            metric_name="s11_mean_db",
            baseline_value=-25.0,
            current_value=-24.5,
            delta=0.5,
            threshold=1.0,
            status=RegressionStatus.PASS,
            message="S11 within threshold",
        )
        d = cmp.to_dict()
        assert d["metric_name"] == "s11_mean_db"
        assert d["status"] == "pass"
        assert "baseline_value" in d
        assert "current_value" in d


class TestCompareCalibrationResult:
    """Test comparison of individual calibration results."""

    @pytest.fixture
    def baseline_result(self) -> CalibrationResult:
        """Create a baseline calibration result."""
        return CalibrationResult(
            case_id="CAL-2",
            status="pass",
            metric=0.95,
            detail="Baseline result",
            s11_mag_db_mean=-25.0,
            s21_mag_db_mean=-0.5,
            frequency_range_hz=(1e9, 10e9),
        )

    def test_identical_results_pass(self, baseline_result: CalibrationResult) -> None:
        """Identical results should pass with no delta."""
        current = CalibrationResult(
            case_id=baseline_result.case_id,
            status=baseline_result.status,
            metric=baseline_result.metric,
            detail="Current result",
            s11_mag_db_mean=baseline_result.s11_mag_db_mean,
            s21_mag_db_mean=baseline_result.s21_mag_db_mean,
            frequency_range_hz=baseline_result.frequency_range_hz,
        )

        result = compare_calibration_result(baseline_result, current, DEFAULT_THRESHOLDS)

        assert result.status == RegressionStatus.PASS
        assert result.case_id == "CAL-2"
        for cmp in result.comparisons:
            assert cmp.delta == pytest.approx(0.0)

    def test_small_numerical_difference_passes(self, baseline_result: CalibrationResult) -> None:
        """Small numerical differences should not cause failure."""
        current = CalibrationResult(
            case_id=baseline_result.case_id,
            status="pass",
            metric=0.94,  # 0.01 difference
            detail="Current result",
            s11_mag_db_mean=-24.8,  # 0.2 dB difference
            s21_mag_db_mean=-0.55,  # 0.05 dB difference
            frequency_range_hz=(1e9, 10e9),
        )

        result = compare_calibration_result(baseline_result, current, DEFAULT_THRESHOLDS)

        assert result.status == RegressionStatus.PASS
        assert "within threshold" in result.message.lower() or result.message == "All metrics within thresholds"

    def test_large_metric_regression_fails(self, baseline_result: CalibrationResult) -> None:
        """Large metric regression should fail."""
        current = CalibrationResult(
            case_id=baseline_result.case_id,
            status="pass",
            metric=0.5,  # Significant regression
            detail="Current result",
            s11_mag_db_mean=-25.0,
            s21_mag_db_mean=-0.5,
            frequency_range_hz=(1e9, 10e9),
        )

        result = compare_calibration_result(baseline_result, current, DEFAULT_THRESHOLDS)

        assert result.status == RegressionStatus.FAIL
        assert "metric" in result.message.lower()

    def test_s11_regression_detected(self, baseline_result: CalibrationResult) -> None:
        """S11 regression should be detected."""
        current = CalibrationResult(
            case_id=baseline_result.case_id,
            status="pass",
            metric=0.95,
            detail="Current result",
            s11_mag_db_mean=-20.0,  # 5 dB worse
            s21_mag_db_mean=-0.5,
            frequency_range_hz=(1e9, 10e9),
        )

        result = compare_calibration_result(baseline_result, current, DEFAULT_THRESHOLDS)

        # Should fail or warn due to S11 change
        assert result.status in (RegressionStatus.FAIL, RegressionStatus.WARN)

    def test_case_id_mismatch_raises(self, baseline_result: CalibrationResult) -> None:
        """Comparing different case IDs should raise error."""
        current = CalibrationResult(
            case_id="CAL-0",  # Different case
            status="pass",
            metric=0.95,
            detail="Current result",
            s11_mag_db_mean=-25.0,
            s21_mag_db_mean=None,
            frequency_range_hz=(1e9, 10e9),
        )

        with pytest.raises(ValueError, match="different cases"):
            compare_calibration_result(baseline_result, current, DEFAULT_THRESHOLDS)


class TestCompareCalibrationResults:
    """Test comparison of complete calibration runs."""

    def test_identical_runs_pass(self) -> None:
        """Identical runs should pass regression check."""
        baseline = run_calibration_suite()
        current = run_calibration_suite()

        comparison = compare_calibration_results(baseline, current)

        assert comparison.status == RegressionStatus.PASS
        assert comparison.should_block_merge is False
        assert len(comparison.violations) == 0

    def test_comparison_is_metric_based_not_bitwise(self) -> None:
        """Verify comparison uses metrics, not bitwise data comparison.

        REQ-M2-022: Regression compares metrics rather than bitwise data.
        """
        baseline = run_calibration_suite()
        current = run_calibration_suite()

        comparison = compare_calibration_results(baseline, current)

        # Check that comparison uses metric values
        for case_result in comparison.case_results:
            # Should have metric comparisons, not raw data comparisons
            metric_names = [c.metric_name for c in case_result.comparisons]
            assert "metric" in metric_names
            assert "s11_mean_db" in metric_names

            # Each comparison should be numerical, not bitwise
            for cmp in case_result.comparisons:
                assert isinstance(cmp.baseline_value, float)
                assert isinstance(cmp.current_value, float)
                assert isinstance(cmp.delta, float)

    def test_threshold_violation_blocks_merge(self) -> None:
        """Threshold violations should block merge when configured.

        REQ-M2-022: Blocks on threshold violations.
        """
        baseline = run_calibration_suite()

        # Create a degraded run by modifying results
        current_results = []
        for r in baseline.results:
            degraded = CalibrationResult(
                case_id=r.case_id,
                status="fail",
                metric=0.1,  # Very bad metric
                detail="Degraded result",
                s11_mag_db_mean=r.s11_mag_db_mean + 10.0,  # 10 dB worse
                s21_mag_db_mean=r.s21_mag_db_mean + 5.0 if r.s21_mag_db_mean else None,
                frequency_range_hz=r.frequency_range_hz,
            )
            current_results.append(degraded)

        current = CalibrationRun(
            run_id="degraded-run",
            results=current_results,
            summary={"total": len(current_results), "passed": 0, "failed": len(current_results), "mean_metric": 0.1},
        )

        comparison = compare_calibration_results(baseline, current)

        assert comparison.status == RegressionStatus.FAIL
        assert comparison.should_block_merge is True
        assert len(comparison.violations) > 0

    def test_non_blocking_threshold_config(self) -> None:
        """Non-blocking thresholds should not block merge."""
        baseline = run_calibration_suite()

        # Create a slightly degraded run
        current_results = []
        for r in baseline.results:
            degraded = CalibrationResult(
                case_id=r.case_id,
                status=r.status,
                metric=r.metric - 0.3,  # Enough to fail threshold
                detail="Slightly degraded",
                s11_mag_db_mean=r.s11_mag_db_mean + 3.0,
                s21_mag_db_mean=r.s21_mag_db_mean if r.s21_mag_db_mean is None else r.s21_mag_db_mean + 2.0,
                frequency_range_hz=r.frequency_range_hz,
            )
            current_results.append(degraded)

        current = CalibrationRun(
            run_id="degraded-run",
            results=current_results,
            summary={"total": len(current_results), "passed": len(current_results), "failed": 0, "mean_metric": 0.7},
        )

        # Use thresholds that don't block
        non_blocking = RegressionThresholds(
            block_on_threshold_violation=False,
        )

        comparison = compare_calibration_results(baseline, current, non_blocking)

        # Even if status is FAIL, should not block
        assert comparison.should_block_merge is False

    def test_pass_rate_degradation_detected(self) -> None:
        """Pass rate degradation should be detected."""
        baseline = run_calibration_suite()

        # Create run where half fail
        current_results = []
        for i, r in enumerate(baseline.results):
            if i % 2 == 0:
                # Fail half the cases
                degraded = CalibrationResult(
                    case_id=r.case_id,
                    status="fail",
                    metric=0.3,
                    detail="Failed",
                    s11_mag_db_mean=r.s11_mag_db_mean,
                    s21_mag_db_mean=r.s21_mag_db_mean,
                    frequency_range_hz=r.frequency_range_hz,
                )
                current_results.append(degraded)
            else:
                current_results.append(r)

        n_failed = len([r for r in current_results if r.status == "fail"])
        current = CalibrationRun(
            run_id="half-failed-run",
            results=current_results,
            summary={
                "total": len(current_results),
                "passed": len(current_results) - n_failed,
                "failed": n_failed,
                "mean_metric": 0.6,
            },
        )

        comparison = compare_calibration_results(baseline, current)

        # Should detect pass rate degradation
        assert comparison.summary["pass_rate_delta"] < 0
        # Should have violation for pass rate
        pass_rate_violations = [v for v in comparison.violations if "pass rate" in v.lower()]
        assert len(pass_rate_violations) > 0


class TestCheckRegressionGate:
    """Test the regression gate check function."""

    def test_gate_passes_for_identical_runs(self) -> None:
        """Gate should pass for identical runs."""
        baseline = run_calibration_suite()
        current = run_calibration_suite()

        assert check_regression_gate(baseline, current) is True

    def test_gate_fails_for_degraded_runs(self) -> None:
        """Gate should fail for significantly degraded runs."""
        baseline = run_calibration_suite()

        # Create severely degraded run
        current_results = [
            CalibrationResult(
                case_id=r.case_id,
                status="fail",
                metric=0.1,
                detail="Severely degraded",
                s11_mag_db_mean=r.s11_mag_db_mean + 20.0,
                s21_mag_db_mean=r.s21_mag_db_mean if r.s21_mag_db_mean is None else r.s21_mag_db_mean + 10.0,
                frequency_range_hz=r.frequency_range_hz,
            )
            for r in baseline.results
        ]

        current = CalibrationRun(
            run_id="severely-degraded",
            results=current_results,
            summary={"total": len(current_results), "passed": 0, "failed": len(current_results), "mean_metric": 0.1},
        )

        assert check_regression_gate(baseline, current) is False


class TestRegressionComparison:
    """Test RegressionComparison data structure."""

    def test_to_dict_serialization(self) -> None:
        """RegressionComparison should serialize to dict."""
        baseline = run_calibration_suite()
        current = run_calibration_suite()

        comparison = compare_calibration_results(baseline, current)
        d = comparison.to_dict()

        assert "baseline_run_id" in d
        assert "current_run_id" in d
        assert "status" in d
        assert "case_results" in d
        assert "should_block_merge" in d
        assert "violations" in d
        assert isinstance(d["case_results"], list)

    def test_violations_list_immutable(self) -> None:
        """Violations should be accessed as a copy."""
        baseline = run_calibration_suite()
        current = run_calibration_suite()

        comparison = compare_calibration_results(baseline, current)
        violations1 = comparison.violations
        violations2 = comparison.violations

        # Should be separate list instances
        assert violations1 is not violations2


def test_calibration_regression_is_metric_based_and_merge_blocking() -> None:
    """Integration test for REQ-M2-022.

    Verifies that:
    1. Calibration regression uses metric-based comparison
    2. It blocks merges on threshold violations
    """
    # Run calibration suite twice (deterministic, should be identical)
    baseline = run_calibration_suite()
    current = run_calibration_suite()

    # Compare using metric-based regression
    comparison = compare_calibration_results(baseline, current)

    # Verify it's metric-based (has numeric comparisons)
    for case_result in comparison.case_results:
        assert len(case_result.comparisons) > 0
        for cmp in case_result.comparisons:
            # All comparisons should be numeric
            assert isinstance(cmp.baseline_value, float)
            assert isinstance(cmp.current_value, float)
            assert isinstance(cmp.threshold, float)

    # Identical runs should pass
    assert comparison.status == RegressionStatus.PASS
    assert comparison.should_block_merge is False

    # Now test merge blocking with a degraded run
    degraded_results = [
        CalibrationResult(
            case_id=r.case_id,
            status="fail",
            metric=0.0,  # Complete failure
            detail="Completely broken",
            s11_mag_db_mean=0.0,  # Way off
            s21_mag_db_mean=None,
            frequency_range_hz=r.frequency_range_hz,
        )
        for r in baseline.results
    ]

    degraded = CalibrationRun(
        run_id="broken-run",
        results=degraded_results,
        summary={"total": len(degraded_results), "passed": 0, "failed": len(degraded_results), "mean_metric": 0.0},
    )

    degraded_comparison = compare_calibration_results(baseline, degraded)

    # Should block merge due to threshold violations
    assert degraded_comparison.status == RegressionStatus.FAIL
    assert degraded_comparison.should_block_merge is True
    assert len(degraded_comparison.violations) > 0
