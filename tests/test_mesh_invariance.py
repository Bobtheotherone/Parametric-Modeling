"""Tests for mesh invariance gate.

REQ-M2-023: test_mesh_invariance_gate_runs_baseline_vs_refined_and_enforces_thresholds

These tests validate:
- Invariance gate compares baseline vs refined mesh S-parameters
- Delta-S thresholds are enforced (magnitude, phase, dB)
- Reports correctly identify violations
- Statistical criteria work when fail_on_any=False
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from formula_foundry.mesh.invariance import (
    InvarianceReport,
    InvarianceThresholds,
    InvarianceViolation,
    SParameterData,
    check_invariance,
    run_invariance_gate,
)


def _make_sparam_data(
    freqs: list[float],
    s11_values: list[complex],
    s21_values: list[complex],
    mesh_hash: str = "test_hash",
) -> SParameterData:
    """Helper to create 2-port S-parameter data."""
    return SParameterData(
        frequencies_hz=freqs,
        s_params={
            "S11": s11_values,
            "S21": s21_values,
            "S12": s21_values,  # Assume reciprocal
            "S22": s11_values,  # Assume symmetric
        },
        mesh_hash=mesh_hash,
    )


class TestInvarianceThresholds:
    """Tests for InvarianceThresholds validation."""

    def test_default_thresholds_are_reasonable(self) -> None:
        thresholds = InvarianceThresholds()
        assert thresholds.max_delta_s_mag == 0.02
        assert thresholds.max_delta_s_phase_deg == 3.0
        assert thresholds.max_delta_s_db == 0.3
        assert thresholds.fail_on_any is False
        assert thresholds.percentile_threshold == 95.0

    def test_negative_magnitude_raises(self) -> None:
        with pytest.raises(ValueError, match="max_delta_s_mag"):
            InvarianceThresholds(max_delta_s_mag=-0.1)

    def test_negative_phase_raises(self) -> None:
        with pytest.raises(ValueError, match="max_delta_s_phase_deg"):
            InvarianceThresholds(max_delta_s_phase_deg=-1.0)

    def test_negative_db_raises(self) -> None:
        with pytest.raises(ValueError, match="max_delta_s_db"):
            InvarianceThresholds(max_delta_s_db=-0.5)

    def test_invalid_percentile_raises(self) -> None:
        with pytest.raises(ValueError, match="percentile_threshold"):
            InvarianceThresholds(percentile_threshold=101.0)


class TestCheckInvariance:
    """Tests for the check_invariance function."""

    def test_identical_data_passes(self) -> None:
        """Identical baseline and refined should pass."""
        freqs = [1e9, 2e9, 3e9]
        s11 = [complex(-0.1, 0.05), complex(-0.12, 0.06), complex(-0.15, 0.08)]
        s21 = [complex(0.9, 0.1), complex(0.85, 0.15), complex(0.8, 0.2)]

        baseline = _make_sparam_data(freqs, s11, s21, "hash_baseline")
        refined = _make_sparam_data(freqs, s11, s21, "hash_refined")

        report = check_invariance(baseline, refined)

        assert report.passed is True
        assert len(report.violations) == 0
        assert report.baseline_mesh_hash == "hash_baseline"
        assert report.refined_mesh_hash == "hash_refined"

    def test_small_differences_pass(self) -> None:
        """Small differences within threshold should pass."""
        freqs = [1e9, 2e9, 3e9]
        s11_baseline = [complex(-0.1, 0.05), complex(-0.12, 0.06), complex(-0.15, 0.08)]
        s21_baseline = [complex(0.9, 0.1), complex(0.85, 0.15), complex(0.8, 0.2)]

        # Add small perturbations (well within default thresholds)
        s11_refined = [s + complex(0.001, 0.001) for s in s11_baseline]
        s21_refined = [s + complex(0.005, 0.002) for s in s21_baseline]

        baseline = _make_sparam_data(freqs, s11_baseline, s21_baseline, "baseline")
        refined = _make_sparam_data(freqs, s11_refined, s21_refined, "refined")

        report = check_invariance(baseline, refined)
        assert report.passed is True

    def test_large_magnitude_difference_fails_with_fail_on_any(self) -> None:
        """Large magnitude difference should fail when fail_on_any=True."""
        freqs = [1e9, 2e9, 3e9]
        s11_baseline = [complex(-0.1, 0.05), complex(-0.12, 0.06), complex(-0.15, 0.08)]
        s21_baseline = [complex(0.9, 0.1), complex(0.85, 0.15), complex(0.8, 0.2)]

        # Large perturbation at second frequency
        s11_refined = s11_baseline.copy()
        s11_refined[1] = complex(-0.3, 0.1)  # Big change

        baseline = _make_sparam_data(freqs, s11_baseline, s21_baseline, "baseline")
        refined = _make_sparam_data(freqs, s11_refined, s21_baseline, "refined")

        thresholds = InvarianceThresholds(fail_on_any=True, max_delta_s_mag=0.02)
        report = check_invariance(baseline, refined, thresholds)

        assert report.passed is False
        assert len(report.violations) > 0
        # Should have magnitude violation
        mag_violations = [v for v in report.violations if v.metric == "magnitude"]
        assert len(mag_violations) > 0

    def test_large_phase_difference_fails(self) -> None:
        """Large phase difference should create violation."""
        freqs = [1e9, 2e9]
        # Same magnitude but different phase
        s11_baseline = [complex(0.1, 0.0), complex(0.1, 0.0)]  # 0 degrees
        s11_refined = [complex(0.0, 0.1), complex(0.0, 0.1)]  # 90 degrees

        baseline = _make_sparam_data(freqs, s11_baseline, s11_baseline, "baseline")
        refined = _make_sparam_data(freqs, s11_refined, s11_refined, "refined")

        thresholds = InvarianceThresholds(fail_on_any=True, max_delta_s_phase_deg=3.0)
        report = check_invariance(baseline, refined, thresholds)

        assert report.passed is False
        phase_violations = [v for v in report.violations if v.metric == "phase"]
        assert len(phase_violations) > 0

    def test_percentile_mode_tolerates_outliers(self) -> None:
        """Percentile mode should pass if 95th percentile is within threshold."""
        # Create 100 frequency points, 5 with violations
        n_points = 100
        freqs = [1e9 + i * 1e7 for i in range(n_points)]

        s11_baseline = [complex(-0.1, 0.05)] * n_points
        s21_baseline = [complex(0.9, 0.1)] * n_points

        s11_refined = s11_baseline.copy()
        s21_refined = s21_baseline.copy()

        # Add 5 outliers (5% of data)
        for i in range(5):
            s11_refined[i] = complex(-0.5, 0.3)  # Large deviation

        baseline = _make_sparam_data(freqs, s11_baseline, s21_baseline, "baseline")
        refined = _make_sparam_data(freqs, s11_refined, s21_refined, "refined")

        # fail_on_any=False uses percentile
        thresholds = InvarianceThresholds(
            fail_on_any=False,
            percentile_threshold=95.0,
            max_delta_s_mag=0.05,
        )
        report = check_invariance(baseline, refined, thresholds)

        # Should pass because 95th percentile excludes the 5% outliers
        assert report.passed is True
        # Violations list should be empty in percentile mode
        assert len(report.violations) == 0

    def test_frequency_mismatch_raises(self) -> None:
        """Mismatched frequency counts should raise ValueError."""
        baseline = _make_sparam_data([1e9, 2e9], [complex(0.1, 0)], [complex(0.9, 0)], "b")
        refined = _make_sparam_data([1e9, 2e9, 3e9], [complex(0.1, 0)], [complex(0.9, 0)], "r")

        with pytest.raises(ValueError, match="Frequency count mismatch"):
            check_invariance(baseline, refined)

    def test_sparam_key_mismatch_raises(self) -> None:
        """Mismatched S-parameter keys should raise ValueError."""
        baseline = SParameterData(
            frequencies_hz=[1e9],
            s_params={"S11": [complex(0.1, 0)]},
            mesh_hash="b",
        )
        refined = SParameterData(
            frequencies_hz=[1e9],
            s_params={"S22": [complex(0.1, 0)]},
            mesh_hash="r",
        )

        with pytest.raises(ValueError, match="S-parameter mismatch"):
            check_invariance(baseline, refined)

    def test_statistics_computed_correctly(self) -> None:
        """Verify statistics are computed in report."""
        freqs = [1e9, 2e9, 3e9]
        s11_baseline = [complex(-0.1, 0.05), complex(-0.12, 0.06), complex(-0.15, 0.08)]
        s21_baseline = [complex(0.9, 0.1), complex(0.85, 0.15), complex(0.8, 0.2)]

        # Small perturbation
        s11_refined = [s + complex(0.01, 0.005) for s in s11_baseline]
        s21_refined = [s + complex(0.01, 0.005) for s in s21_baseline]

        baseline = _make_sparam_data(freqs, s11_baseline, s21_baseline, "baseline")
        refined = _make_sparam_data(freqs, s11_refined, s21_refined, "refined")

        report = check_invariance(baseline, refined)

        assert "magnitude" in report.statistics
        assert "phase" in report.statistics
        assert "dB" in report.statistics

        mag_stats = report.statistics["magnitude"]
        assert "min" in mag_stats
        assert "max" in mag_stats
        assert "mean" in mag_stats
        assert "std" in mag_stats
        assert "percentile_value" in mag_stats


class TestInvarianceReport:
    """Tests for InvarianceReport serialization."""

    def test_to_dict_includes_all_fields(self) -> None:
        """Report should serialize to dict with all fields."""
        thresholds = InvarianceThresholds()
        report = InvarianceReport(
            passed=True,
            baseline_mesh_hash="abc123",
            refined_mesh_hash="def456",
            thresholds=thresholds,
            violations=[],
            statistics={"magnitude": {"mean": 0.001}},
            frequency_range_hz=(1e9, 10e9),
        )

        d = report.to_dict()
        assert d["passed"] is True
        assert d["baseline_mesh_hash"] == "abc123"
        assert d["refined_mesh_hash"] == "def456"
        assert "thresholds" in d
        assert d["thresholds"]["max_delta_s_mag"] == 0.02
        assert d["violations"] == []
        assert d["frequency_range_hz"] == [1e9, 10e9]

    def test_write_json_creates_valid_file(self, tmp_path: Path) -> None:
        """Report should write valid JSON file."""
        thresholds = InvarianceThresholds()
        violation = InvarianceViolation(
            frequency_hz=5e9,
            s_parameter="S11",
            metric="magnitude",
            baseline_value=0.1,
            refined_value=0.15,
            delta=0.05,
            threshold=0.02,
        )
        report = InvarianceReport(
            passed=False,
            baseline_mesh_hash="base",
            refined_mesh_hash="refined",
            thresholds=thresholds,
            violations=[violation],
            statistics={},
            frequency_range_hz=(1e9, 10e9),
        )

        report_path = tmp_path / "invariance_report.json"
        report.write_json(report_path)

        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["passed"] is False
        assert len(data["violations"]) == 1
        assert data["violations"][0]["s_parameter"] == "S11"


class TestRunInvarianceGate:
    """Tests for run_invariance_gate with Touchstone files."""

    def test_run_from_touchstone_files(self, tmp_path: Path) -> None:
        """Test running invariance gate from Touchstone files."""
        # Create baseline .s2p file
        baseline_content = """! Baseline S-parameters
# Hz S RI R 50
1000000000 -0.1 0.05 0.9 0.1 0.9 0.1 -0.1 0.05
2000000000 -0.12 0.06 0.85 0.15 0.85 0.15 -0.12 0.06
3000000000 -0.15 0.08 0.8 0.2 0.8 0.2 -0.15 0.08
"""
        baseline_path = tmp_path / "baseline.s2p"
        baseline_path.write_text(baseline_content)

        # Create refined .s2p file (slightly different)
        refined_content = """! Refined S-parameters
# Hz S RI R 50
1000000000 -0.101 0.051 0.901 0.101 0.901 0.101 -0.101 0.051
2000000000 -0.121 0.061 0.851 0.151 0.851 0.151 -0.121 0.061
3000000000 -0.151 0.081 0.801 0.201 0.801 0.201 -0.151 0.081
"""
        refined_path = tmp_path / "refined.s2p"
        refined_path.write_text(refined_content)

        report_path = tmp_path / "report.json"

        report = run_invariance_gate(
            baseline_sparams_path=baseline_path,
            refined_sparams_path=refined_path,
            baseline_mesh_hash="baseline_hash",
            refined_mesh_hash="refined_hash",
            report_path=report_path,
        )

        assert report.passed is True
        assert report_path.exists()


def test_mesh_invariance_gate_runs_baseline_vs_refined_and_enforces_thresholds() -> None:
    """REQ-M2-023: Mesh invariance gate runs baseline vs refined and enforces thresholds.

    This is the primary test for the design document requirement.
    """
    # Setup: Create baseline mesh S-parameters (coarser mesh)
    frequencies = [f * 1e9 for f in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

    # Typical return loss and insertion loss for a transmission line
    baseline_s11 = [complex(-0.1 - i * 0.01, 0.05 + i * 0.01) for i in range(10)]
    baseline_s21 = [complex(0.95 - i * 0.02, 0.1 + i * 0.02) for i in range(10)]

    # Refined mesh should give similar results within threshold
    refined_s11 = [s + complex(0.005, 0.003) for s in baseline_s11]
    refined_s21 = [s + complex(0.008, 0.004) for s in baseline_s21]

    baseline = _make_sparam_data(frequencies, baseline_s11, baseline_s21, "mesh_hash_coarse")
    refined = _make_sparam_data(frequencies, refined_s11, refined_s21, "mesh_hash_fine")

    # Test with default thresholds
    thresholds = InvarianceThresholds(
        max_delta_s_mag=0.02,
        max_delta_s_phase_deg=3.0,
        max_delta_s_db=0.3,
        fail_on_any=False,
        percentile_threshold=95.0,
    )

    report = check_invariance(baseline, refined, thresholds)

    # Assertions per REQ-M2-023
    assert report.baseline_mesh_hash == "mesh_hash_coarse"
    assert report.refined_mesh_hash == "mesh_hash_fine"
    assert report.passed is True
    assert report.frequency_range_hz == (1e9, 10e9)

    # Statistics should be computed
    assert report.statistics["magnitude"]["max"] < thresholds.max_delta_s_mag
    assert report.statistics["phase"]["percentile_value"] < thresholds.max_delta_s_phase_deg

    # Now test that violations are caught
    bad_refined_s11 = baseline_s11.copy()
    bad_refined_s11[5] = complex(-0.5, 0.3)  # Large deviation at 6 GHz

    bad_refined = _make_sparam_data(frequencies, bad_refined_s11, refined_s21, "mesh_hash_bad")

    strict_thresholds = InvarianceThresholds(
        max_delta_s_mag=0.02,
        max_delta_s_phase_deg=3.0,
        max_delta_s_db=0.3,
        fail_on_any=True,  # Strict mode
    )

    strict_report = check_invariance(baseline, bad_refined, strict_thresholds)

    assert strict_report.passed is False
    assert len(strict_report.violations) > 0

    # Verify violation details
    mag_violation = next((v for v in strict_report.violations if v.metric == "magnitude"), None)
    assert mag_violation is not None
    assert mag_violation.frequency_hz == 6e9
    assert mag_violation.s_parameter in ("S11", "S22")  # Symmetric
    assert mag_violation.delta > thresholds.max_delta_s_mag
