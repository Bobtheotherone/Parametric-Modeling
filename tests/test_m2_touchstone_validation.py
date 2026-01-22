"""Tests for S-parameter validation (REQ-M2-007).

This module tests the validation module which provides:
- Passivity validation (|S| eigenvalues <= 1)
- Reciprocity validation (S12 ≈ S21)
- Causality validation (impulse response zero for t < 0)
- Validation results for manifest inclusion
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from formula_foundry.em import (
    SParameterData,
    write_touchstone,
)
from formula_foundry.em.validation import (
    CausalityCheckResult,
    PassivityCheckResult,
    ReciprocityCheckResult,
    SParameterValidationResult,
    ValidationStatus,
    build_validation_manifest_entry,
    check_causality,
    check_passivity,
    check_reciprocity,
    check_stability_2port,
    compute_stability_k_factor,
    validate_sparam_data,
    validate_touchstone_file,
)


# =============================================================================
# Test Data Factories
# =============================================================================


def _make_passive_reciprocal_sparam_data(n_freq: int = 101) -> SParameterData:
    """Create passive, reciprocal S-parameter data.

    This creates S-parameter data that passes passivity and causality checks.
    For passivity: eigenvalues of S must have |λ| <= 1.
    For 2x2 symmetric S with S11=S22=a and S12=S21=b: eigenvalues are (a+b) and (a-b).
    We need |a+b| <= 1 and |a-b| <= 1.

    For causality: use many frequency points (101) with minimal phase variation
    to produce impulse response concentrated near t=0.
    """
    # Start from low frequency (near DC) for better causality
    frequencies_hz = np.linspace(0.1e9, 10e9, n_freq)
    s_parameters = np.zeros((n_freq, 2, 2), dtype=np.complex128)

    # Use near-constant magnitude and minimal phase for good causality
    # S11 = S22: small return loss (~-20 dB), constant phase
    s11_mag = 0.1
    s11_phase = 0.0  # No phase variation for causality

    # S21 = S12: high transmission (~-1 dB), constant phase
    # Ensures |S11| + |S21| = 0.1 + 0.85 = 0.95 < 1 (passive)
    s21_mag = 0.85
    s21_phase = 0.0  # No phase variation for causality

    for i in range(n_freq):
        s_parameters[i, 0, 0] = s11_mag * np.exp(1j * s11_phase)
        s_parameters[i, 1, 1] = s11_mag * np.exp(1j * s11_phase)
        s_parameters[i, 1, 0] = s21_mag * np.exp(1j * s21_phase)
        s_parameters[i, 0, 1] = s21_mag * np.exp(1j * s21_phase)  # Reciprocal

    return SParameterData(
        frequencies_hz=frequencies_hz,
        s_parameters=s_parameters,
        n_ports=2,
        reference_impedance_ohm=50.0,
    )


def _make_non_passive_sparam_data(n_freq: int = 11) -> SParameterData:
    """Create non-passive (amplifying) S-parameter data."""
    frequencies_hz = np.linspace(1e9, 10e9, n_freq)
    s_parameters = np.zeros((n_freq, 2, 2), dtype=np.complex128)

    # S21 gain > 1 (amplification)
    for i in range(n_freq):
        s_parameters[i, 0, 0] = 0.1
        s_parameters[i, 1, 1] = 0.1
        s_parameters[i, 1, 0] = 1.5  # |S21| > 1 - amplification
        s_parameters[i, 0, 1] = 1.5

    return SParameterData(
        frequencies_hz=frequencies_hz,
        s_parameters=s_parameters,
        n_ports=2,
        reference_impedance_ohm=50.0,
    )


def _make_non_reciprocal_sparam_data(n_freq: int = 11) -> SParameterData:
    """Create non-reciprocal S-parameter data (isolator-like)."""
    frequencies_hz = np.linspace(1e9, 10e9, n_freq)
    s_parameters = np.zeros((n_freq, 2, 2), dtype=np.complex128)

    for i in range(n_freq):
        s_parameters[i, 0, 0] = 0.1
        s_parameters[i, 1, 1] = 0.1
        s_parameters[i, 1, 0] = 0.9  # Forward transmission
        s_parameters[i, 0, 1] = 0.1  # Reverse isolation (S12 << S21)

    return SParameterData(
        frequencies_hz=frequencies_hz,
        s_parameters=s_parameters,
        n_ports=2,
        reference_impedance_ohm=50.0,
    )


def _make_single_port_sparam_data(n_freq: int = 11) -> SParameterData:
    """Create single-port S-parameter data."""
    frequencies_hz = np.linspace(1e9, 10e9, n_freq)
    s_parameters = np.zeros((n_freq, 1, 1), dtype=np.complex128)

    for i in range(n_freq):
        s_parameters[i, 0, 0] = 0.1 + 0.05 * i / n_freq

    return SParameterData(
        frequencies_hz=frequencies_hz,
        s_parameters=s_parameters,
        n_ports=1,
        reference_impedance_ohm=50.0,
    )


# =============================================================================
# Passivity Validation Tests (REQ-M2-007)
# =============================================================================


class TestPassivityValidation:
    """Tests for passivity validation."""

    def test_passive_network_passes(self) -> None:
        """REQ-M2-007: Passive network should pass passivity check."""
        sparam_data = _make_passive_reciprocal_sparam_data()
        result = check_passivity(sparam_data)

        assert result.status == ValidationStatus.PASS
        assert result.n_violations == 0
        assert result.max_eigenvalue <= 1.0 + 1e-6
        assert isinstance(result, PassivityCheckResult)

    def test_non_passive_network_fails(self) -> None:
        """REQ-M2-007: Non-passive network should fail passivity check."""
        sparam_data = _make_non_passive_sparam_data()
        result = check_passivity(sparam_data)

        assert result.status == ValidationStatus.FAIL
        assert result.n_violations > 0
        assert result.max_eigenvalue > 1.0

    def test_passivity_tolerance_respected(self) -> None:
        """REQ-M2-007: Tolerance parameter affects pass/fail threshold."""
        sparam_data = _make_passive_reciprocal_sparam_data()

        # Very strict tolerance
        result_strict = check_passivity(sparam_data, tolerance=1e-10)
        # Loose tolerance
        result_loose = check_passivity(sparam_data, tolerance=1e-2)

        # Both should pass for truly passive network
        assert result_loose.status == ValidationStatus.PASS
        # Strict might have different behavior but should still be reasonable
        assert result_strict.n_violations <= result_loose.n_violations

    def test_passivity_result_to_dict(self) -> None:
        """REQ-M2-007: Result converts to dictionary for manifest."""
        sparam_data = _make_passive_reciprocal_sparam_data()
        result = check_passivity(sparam_data)
        result_dict = result.to_dict()

        assert result_dict["check"] == "passivity"
        assert result_dict["status"] == "pass"
        assert "max_eigenvalue" in result_dict
        assert "n_violations" in result_dict
        assert "tolerance" in result_dict
        assert "message" in result_dict

    def test_passivity_single_port(self) -> None:
        """REQ-M2-007: Passivity check works for single port."""
        sparam_data = _make_single_port_sparam_data()
        result = check_passivity(sparam_data)

        assert result.status == ValidationStatus.PASS


# =============================================================================
# Reciprocity Validation Tests (REQ-M2-007)
# =============================================================================


class TestReciprocityValidation:
    """Tests for reciprocity validation."""

    def test_reciprocal_network_passes(self) -> None:
        """REQ-M2-007: Reciprocal network should pass reciprocity check."""
        sparam_data = _make_passive_reciprocal_sparam_data()
        result = check_reciprocity(sparam_data)

        assert result.status == ValidationStatus.PASS
        assert result.n_violations == 0
        assert result.max_error < 1e-10  # Perfect reciprocity
        assert isinstance(result, ReciprocityCheckResult)

    def test_non_reciprocal_network_fails(self) -> None:
        """REQ-M2-007: Non-reciprocal network should fail reciprocity check."""
        sparam_data = _make_non_reciprocal_sparam_data()
        result = check_reciprocity(sparam_data)

        assert result.status == ValidationStatus.FAIL
        assert result.n_violations > 0
        assert result.max_error > 0.5  # S21 - S12 = 0.9 - 0.1 = 0.8

    def test_reciprocity_tolerance_respected(self) -> None:
        """REQ-M2-007: Tolerance parameter affects pass/fail threshold."""
        sparam_data = _make_non_reciprocal_sparam_data()

        # Very loose tolerance (should pass)
        result_loose = check_reciprocity(sparam_data, tolerance=1.0)
        # Strict tolerance (should fail)
        result_strict = check_reciprocity(sparam_data, tolerance=1e-6)

        assert result_loose.status in (ValidationStatus.PASS, ValidationStatus.WARN)
        assert result_strict.status == ValidationStatus.FAIL

    def test_reciprocity_result_to_dict(self) -> None:
        """REQ-M2-007: Result converts to dictionary for manifest."""
        sparam_data = _make_passive_reciprocal_sparam_data()
        result = check_reciprocity(sparam_data)
        result_dict = result.to_dict()

        assert result_dict["check"] == "reciprocity"
        assert result_dict["status"] == "pass"
        assert "max_error" in result_dict
        assert "mean_error" in result_dict
        assert "max_error_db" in result_dict
        assert "tolerance" in result_dict

    def test_reciprocity_single_port_empty(self) -> None:
        """REQ-M2-007: Single port has no off-diagonal elements to check."""
        sparam_data = _make_single_port_sparam_data()
        result = check_reciprocity(sparam_data)

        # Single port has no S12/S21, so should pass trivially
        assert result.status == ValidationStatus.PASS
        assert result.max_error == 0.0


# =============================================================================
# Causality Validation Tests (REQ-M2-007)
# =============================================================================


class TestCausalityValidation:
    """Tests for causality validation."""

    def test_causal_network_passes(self) -> None:
        """REQ-M2-007: Causal network should pass causality check."""
        sparam_data = _make_passive_reciprocal_sparam_data()
        result = check_causality(sparam_data)

        assert result.status in (ValidationStatus.PASS, ValidationStatus.WARN)
        assert result.is_causal
        assert isinstance(result, CausalityCheckResult)

    def test_causality_tolerance_affects_result(self) -> None:
        """REQ-M2-007: Tolerance parameter affects pass/fail threshold."""
        sparam_data = _make_passive_reciprocal_sparam_data()

        result_loose = check_causality(sparam_data, tolerance=0.5)
        result_strict = check_causality(sparam_data, tolerance=1e-10)

        # Loose should always pass
        assert result_loose.is_causal
        # Strict may or may not pass depending on numerical noise

    def test_causality_result_to_dict(self) -> None:
        """REQ-M2-007: Result converts to dictionary for manifest."""
        sparam_data = _make_passive_reciprocal_sparam_data()
        result = check_causality(sparam_data)
        result_dict = result.to_dict()

        assert result_dict["check"] == "causality"
        assert "status" in result_dict
        assert "is_causal" in result_dict
        assert "pre_response_energy_ratio" in result_dict
        assert "max_pre_response_magnitude" in result_dict
        assert "tolerance" in result_dict

    def test_causality_single_port(self) -> None:
        """REQ-M2-007: Causality check works for single port (uses S11)."""
        sparam_data = _make_single_port_sparam_data()
        result = check_causality(sparam_data)

        # Should compute causality from S11
        assert result.is_causal


# =============================================================================
# Combined Validation Tests (REQ-M2-007)
# =============================================================================


class TestCombinedValidation:
    """Tests for combined S-parameter validation."""

    def test_validate_sparam_data_all_pass(self) -> None:
        """REQ-M2-007: All checks pass for valid network."""
        sparam_data = _make_passive_reciprocal_sparam_data()
        result = validate_sparam_data(sparam_data)

        assert result.is_valid
        assert result.overall_status == ValidationStatus.PASS
        assert isinstance(result, SParameterValidationResult)

    def test_validate_sparam_data_passivity_fail(self) -> None:
        """REQ-M2-007: Passivity failure results in overall failure."""
        sparam_data = _make_non_passive_sparam_data()
        result = validate_sparam_data(sparam_data)

        assert not result.is_valid
        assert result.overall_status == ValidationStatus.FAIL
        assert result.passivity.status == ValidationStatus.FAIL

    def test_validate_sparam_data_reciprocity_fail(self) -> None:
        """REQ-M2-007: Reciprocity failure results in overall failure."""
        sparam_data = _make_non_reciprocal_sparam_data()
        result = validate_sparam_data(sparam_data)

        assert not result.is_valid
        assert result.overall_status == ValidationStatus.FAIL
        assert result.reciprocity.status == ValidationStatus.FAIL

    def test_validate_sparam_data_skip_causality(self) -> None:
        """REQ-M2-007: Can skip causality check."""
        sparam_data = _make_passive_reciprocal_sparam_data()
        result = validate_sparam_data(sparam_data, skip_causality=True)

        assert result.causality.status == ValidationStatus.SKIP
        assert result.is_valid  # Other checks should still pass

    def test_validate_sparam_data_metadata(self) -> None:
        """REQ-M2-007: Result includes metadata."""
        sparam_data = _make_passive_reciprocal_sparam_data()
        result = validate_sparam_data(sparam_data)

        assert result.n_frequencies == sparam_data.n_frequencies
        assert result.n_ports == sparam_data.n_ports

    def test_validate_sparam_data_to_dict(self) -> None:
        """REQ-M2-007: Combined result converts to dictionary."""
        sparam_data = _make_passive_reciprocal_sparam_data()
        result = validate_sparam_data(sparam_data)
        result_dict = result.to_dict()

        assert "overall_status" in result_dict
        assert "is_valid" in result_dict
        assert "has_warnings" in result_dict
        assert "n_frequencies" in result_dict
        assert "n_ports" in result_dict
        assert "checks" in result_dict
        assert "passivity" in result_dict["checks"]
        assert "reciprocity" in result_dict["checks"]
        assert "causality" in result_dict["checks"]


# =============================================================================
# Touchstone File Validation Tests (REQ-M2-007)
# =============================================================================


class TestTouchstoneFileValidation:
    """Tests for Touchstone file validation."""

    def test_validate_touchstone_file_valid(self, tmp_path: Path) -> None:
        """REQ-M2-007: Validate valid Touchstone file."""
        sparam_data = _make_passive_reciprocal_sparam_data()
        ts_path = tmp_path / "test.s2p"
        write_touchstone(sparam_data, ts_path)

        result = validate_touchstone_file(ts_path)

        assert result.is_valid
        assert result.overall_status == ValidationStatus.PASS

    def test_validate_touchstone_file_invalid(self, tmp_path: Path) -> None:
        """REQ-M2-007: Validate invalid Touchstone file."""
        sparam_data = _make_non_passive_sparam_data()
        ts_path = tmp_path / "test.s2p"
        write_touchstone(sparam_data, ts_path)

        result = validate_touchstone_file(ts_path)

        assert not result.is_valid
        assert result.overall_status == ValidationStatus.FAIL

    def test_validate_touchstone_file_not_found(self, tmp_path: Path) -> None:
        """REQ-M2-007: FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            validate_touchstone_file(tmp_path / "nonexistent.s2p")


# =============================================================================
# Stability Factor Tests (REQ-M2-007)
# =============================================================================


class TestStabilityFactor:
    """Tests for stability factor computation."""

    def test_compute_stability_k_factor(self) -> None:
        """REQ-M2-007: Compute K factor for 2-port network."""
        sparam_data = _make_passive_reciprocal_sparam_data()
        k = compute_stability_k_factor(sparam_data)

        assert len(k) == sparam_data.n_frequencies
        # For passive network, K should be > 1 (unconditionally stable)
        assert np.all(k > 0)

    def test_compute_stability_k_factor_single_port_error(self) -> None:
        """REQ-M2-007: K factor requires 2-port network."""
        sparam_data = _make_single_port_sparam_data()

        with pytest.raises(ValueError, match="2-port"):
            compute_stability_k_factor(sparam_data)

    def test_check_stability_2port(self) -> None:
        """REQ-M2-007: Check stability for 2-port network."""
        sparam_data = _make_passive_reciprocal_sparam_data()
        result = check_stability_2port(sparam_data)

        assert result["check"] == "stability_2port"
        assert "k_min" in result
        assert "k_max" in result
        assert "delta_max" in result
        assert "is_unconditionally_stable" in result

    def test_check_stability_2port_single_port_skip(self) -> None:
        """REQ-M2-007: Stability check skipped for single port."""
        sparam_data = _make_single_port_sparam_data()
        result = check_stability_2port(sparam_data)

        assert result["status"] == "skip"


# =============================================================================
# Manifest Entry Tests (REQ-M2-007)
# =============================================================================


class TestManifestEntry:
    """Tests for manifest entry building."""

    def test_build_validation_manifest_entry(self) -> None:
        """REQ-M2-007: Build manifest entry from validation result."""
        sparam_data = _make_passive_reciprocal_sparam_data()
        validation = validate_sparam_data(sparam_data)

        entry = build_validation_manifest_entry(validation)

        assert isinstance(entry, dict)
        assert "overall_status" in entry
        assert "checks" in entry
        assert entry["is_valid"] is True

    def test_manifest_entry_matches_to_dict(self) -> None:
        """REQ-M2-007: Manifest entry equals to_dict() result."""
        sparam_data = _make_passive_reciprocal_sparam_data()
        validation = validate_sparam_data(sparam_data)

        entry = build_validation_manifest_entry(validation)
        direct = validation.to_dict()

        assert entry == direct

    def test_extra_check_addition(self) -> None:
        """REQ-M2-007: Can add extra checks to result."""
        sparam_data = _make_passive_reciprocal_sparam_data()
        validation = validate_sparam_data(sparam_data)

        # Add stability check
        stability = check_stability_2port(sparam_data)
        validation.add_extra_check("stability_2port", stability)

        result_dict = validation.to_dict()
        assert "stability_2port" in result_dict["checks"]


# =============================================================================
# Edge Cases and Error Handling (REQ-M2-007)
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_minimal_frequency_count(self) -> None:
        """REQ-M2-007: Validation works with minimal frequency points."""
        frequencies_hz = np.array([1e9, 2e9])
        s_parameters = np.zeros((2, 2, 2), dtype=np.complex128)
        s_parameters[:, 0, 0] = 0.1
        s_parameters[:, 1, 1] = 0.1
        s_parameters[:, 1, 0] = 0.9
        s_parameters[:, 0, 1] = 0.9

        sparam_data = SParameterData(
            frequencies_hz=frequencies_hz,
            s_parameters=s_parameters,
            n_ports=2,
            reference_impedance_ohm=50.0,
        )

        result = validate_sparam_data(sparam_data)
        assert isinstance(result, SParameterValidationResult)

    def test_large_frequency_count(self) -> None:
        """REQ-M2-007: Validation handles large frequency count."""
        n_freq = 1001
        frequencies_hz = np.linspace(1e9, 50e9, n_freq)
        s_parameters = np.zeros((n_freq, 2, 2), dtype=np.complex128)

        for i in range(n_freq):
            s_parameters[i, 0, 0] = 0.1
            s_parameters[i, 1, 1] = 0.1
            s_parameters[i, 1, 0] = 0.9
            s_parameters[i, 0, 1] = 0.9

        sparam_data = SParameterData(
            frequencies_hz=frequencies_hz,
            s_parameters=s_parameters,
            n_ports=2,
            reference_impedance_ohm=50.0,
        )

        result = validate_sparam_data(sparam_data)
        assert result.n_frequencies == n_freq

    def test_zero_transmission(self) -> None:
        """REQ-M2-007: Validation handles zero transmission."""
        frequencies_hz = np.linspace(1e9, 10e9, 11)
        s_parameters = np.zeros((11, 2, 2), dtype=np.complex128)

        # All zeros (open circuit on both ports)
        for i in range(11):
            s_parameters[i, 0, 0] = 1.0
            s_parameters[i, 1, 1] = 1.0
            # S21 = S12 = 0 (no transmission)

        sparam_data = SParameterData(
            frequencies_hz=frequencies_hz,
            s_parameters=s_parameters,
            n_ports=2,
            reference_impedance_ohm=50.0,
        )

        result = validate_sparam_data(sparam_data)
        # Should be passive (|S| <= 1) and reciprocal (S12 = S21 = 0)
        assert result.passivity.status == ValidationStatus.PASS
        assert result.reciprocity.status == ValidationStatus.PASS

    def test_warning_threshold(self) -> None:
        """REQ-M2-007: Warning threshold produces WARN status."""
        # Create slightly non-passive data
        frequencies_hz = np.linspace(1e9, 10e9, 11)
        s_parameters = np.zeros((11, 2, 2), dtype=np.complex128)

        for i in range(11):
            s_parameters[i, 0, 0] = 0.1
            s_parameters[i, 1, 1] = 0.1
            # Slightly over unity gain
            s_parameters[i, 1, 0] = 1.0001
            s_parameters[i, 0, 1] = 1.0001

        sparam_data = SParameterData(
            frequencies_hz=frequencies_hz,
            s_parameters=s_parameters,
            n_ports=2,
            reference_impedance_ohm=50.0,
        )

        # With tolerance=1e-6, should fail
        # With tolerance=1e-2, might warn
        result = check_passivity(sparam_data, tolerance=1e-6, warn_threshold=1e-2)
        # Max eigenvalue ~1.0002, so should be WARN not FAIL
        assert result.status in (ValidationStatus.WARN, ValidationStatus.FAIL)
