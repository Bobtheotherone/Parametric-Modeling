"""Additional unit tests for S-parameter validation edge cases.

This module supplements test_m2_touchstone_validation.py with coverage for:
- ValidationStatus enum behavior
- Dataclass serialization edge cases
- Stability factor K edge cases
- Numerical stability of passivity/reciprocity checks
- Warning vs failure threshold behavior

Tests ensure robustness of the em.validation module.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from formula_foundry.em.touchstone import SParameterData
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
)

# =============================================================================
# ValidationStatus Tests
# =============================================================================


class TestValidationStatus:
    """Tests for ValidationStatus enum."""

    def test_status_values(self) -> None:
        """ValidationStatus has expected string values."""
        assert ValidationStatus.PASS.value == "pass"
        assert ValidationStatus.FAIL.value == "fail"
        assert ValidationStatus.WARN.value == "warn"
        assert ValidationStatus.SKIP.value == "skip"

    def test_status_string_comparison(self) -> None:
        """ValidationStatus inherits from str."""
        assert ValidationStatus.PASS == "pass"
        assert ValidationStatus.FAIL == "fail"

    def test_status_from_string(self) -> None:
        """ValidationStatus can be created from string."""
        assert ValidationStatus("pass") == ValidationStatus.PASS
        assert ValidationStatus("fail") == ValidationStatus.FAIL


# =============================================================================
# Result Dataclass Serialization Tests
# =============================================================================


class TestResultSerialization:
    """Tests for result dataclass to_dict serialization."""

    def test_passivity_result_to_dict_keys(self) -> None:
        """PassivityCheckResult.to_dict has required keys."""
        result = PassivityCheckResult(
            status=ValidationStatus.PASS,
            max_eigenvalue=0.95,
            n_violations=0,
            violation_frequencies_hz=(),
            tolerance=1e-6,
            message="Test message",
        )
        d = result.to_dict()

        assert d["check"] == "passivity"
        assert d["status"] == "pass"
        assert d["max_eigenvalue"] == 0.95
        assert d["n_violations"] == 0
        assert d["violation_frequencies_hz"] == []
        assert d["tolerance"] == 1e-6
        assert d["message"] == "Test message"

    def test_reciprocity_result_to_dict_keys(self) -> None:
        """ReciprocityCheckResult.to_dict has required keys."""
        result = ReciprocityCheckResult(
            status=ValidationStatus.PASS,
            max_error=1e-10,
            mean_error=5e-11,
            max_error_db=-200.0,
            n_violations=0,
            tolerance=1e-6,
            message="Reciprocal network",
        )
        d = result.to_dict()

        assert d["check"] == "reciprocity"
        assert d["status"] == "pass"
        assert d["max_error"] == 1e-10
        assert d["mean_error"] == 5e-11
        assert "max_error_db" in d
        assert d["n_violations"] == 0

    def test_causality_result_to_dict_keys(self) -> None:
        """CausalityCheckResult.to_dict has required keys."""
        result = CausalityCheckResult(
            status=ValidationStatus.PASS,
            is_causal=True,
            pre_response_energy_ratio=1e-6,
            max_pre_response_magnitude=1e-5,
            tolerance=1e-3,
            message="Causal network",
        )
        d = result.to_dict()

        assert d["check"] == "causality"
        assert d["status"] == "pass"
        assert d["is_causal"] is True
        assert d["pre_response_energy_ratio"] == 1e-6
        assert d["max_pre_response_magnitude"] == 1e-5


class TestSParameterValidationResultProperties:
    """Tests for SParameterValidationResult properties."""

    def _make_pass_result(self) -> SParameterValidationResult:
        """Create a result where all checks pass."""
        return SParameterValidationResult(
            passivity=PassivityCheckResult(
                status=ValidationStatus.PASS,
                max_eigenvalue=0.95,
                n_violations=0,
                violation_frequencies_hz=(),
                tolerance=1e-6,
                message="Pass",
            ),
            reciprocity=ReciprocityCheckResult(
                status=ValidationStatus.PASS,
                max_error=1e-10,
                mean_error=5e-11,
                max_error_db=-200.0,
                n_violations=0,
                tolerance=1e-6,
                message="Pass",
            ),
            causality=CausalityCheckResult(
                status=ValidationStatus.PASS,
                is_causal=True,
                pre_response_energy_ratio=1e-6,
                max_pre_response_magnitude=1e-5,
                tolerance=1e-3,
                message="Pass",
            ),
            overall_status=ValidationStatus.PASS,
            n_frequencies=101,
            n_ports=2,
        )

    def test_is_valid_property_pass(self) -> None:
        """is_valid is True when overall status is PASS."""
        result = self._make_pass_result()
        assert result.is_valid is True

    def test_is_valid_property_fail(self) -> None:
        """is_valid is False when overall status is not PASS."""
        result = self._make_pass_result()
        result.overall_status = ValidationStatus.FAIL
        assert result.is_valid is False

    def test_has_warnings_property_no_warnings(self) -> None:
        """has_warnings is False when no warnings."""
        result = self._make_pass_result()
        assert result.has_warnings is False

    def test_has_warnings_property_with_warning(self) -> None:
        """has_warnings is True when any check has warning."""
        result = self._make_pass_result()
        result.passivity = PassivityCheckResult(
            status=ValidationStatus.WARN,
            max_eigenvalue=1.0005,
            n_violations=1,
            violation_frequencies_hz=(1e9,),
            tolerance=1e-6,
            message="Marginal",
        )
        assert result.has_warnings is True

    def test_add_extra_check(self) -> None:
        """Extra checks can be added to result."""
        result = self._make_pass_result()
        result.add_extra_check("custom_check", {"status": "pass", "value": 42})

        d = result.to_dict()
        assert "custom_check" in d["checks"]
        assert d["checks"]["custom_check"]["value"] == 42


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability of validation functions."""

    def _make_sparam_data(
        self,
        s11: complex = 0.1,
        s21: complex = 0.9,
        n_freq: int = 11,
    ) -> SParameterData:
        """Create S-parameter data with specified values."""
        frequencies_hz = np.linspace(1e9, 10e9, n_freq)
        s_parameters = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        s_parameters[:, 0, 0] = s11
        s_parameters[:, 1, 1] = s11
        s_parameters[:, 1, 0] = s21
        s_parameters[:, 0, 1] = s21  # Reciprocal
        return SParameterData(
            frequencies_hz=frequencies_hz,
            s_parameters=s_parameters,
            n_ports=2,
        )

    def test_passivity_near_unity_boundary(self) -> None:
        """Passivity check handles values near unity boundary."""
        # S21 magnitude exactly 1.0 should pass with small tolerance
        data = self._make_sparam_data(s11=0.0, s21=1.0)
        result = check_passivity(data, tolerance=1e-6)
        assert result.status == ValidationStatus.PASS

    def test_passivity_slightly_above_unity(self) -> None:
        """Passivity check catches slightly active network."""
        # S21 magnitude slightly above 1.0
        data = self._make_sparam_data(s11=0.0, s21=1.001)
        result = check_passivity(data, tolerance=1e-6)
        assert result.status in (ValidationStatus.WARN, ValidationStatus.FAIL)

    def test_reciprocity_perfect_symmetry(self) -> None:
        """Reciprocity check handles perfectly symmetric network."""
        data = self._make_sparam_data()  # S12 = S21
        result = check_reciprocity(data)
        assert result.max_error < 1e-15
        assert result.status == ValidationStatus.PASS

    def test_reciprocity_tiny_asymmetry(self) -> None:
        """Reciprocity check detects tiny asymmetry."""
        frequencies_hz = np.linspace(1e9, 10e9, 11)
        s_parameters = np.zeros((11, 2, 2), dtype=np.complex128)
        s_parameters[:, 0, 0] = 0.1
        s_parameters[:, 1, 1] = 0.1
        s_parameters[:, 1, 0] = 0.9
        s_parameters[:, 0, 1] = 0.9 + 1e-8  # Tiny asymmetry
        data = SParameterData(
            frequencies_hz=frequencies_hz,
            s_parameters=s_parameters,
            n_ports=2,
        )

        result = check_reciprocity(data, tolerance=1e-6)
        assert result.max_error > 0
        assert result.status == ValidationStatus.PASS  # Within tolerance

    def test_passivity_zero_s_matrix(self) -> None:
        """Passivity check handles all-zero S-matrix."""
        data = self._make_sparam_data(s11=0.0, s21=0.0)
        result = check_passivity(data)
        assert result.status == ValidationStatus.PASS
        assert result.max_eigenvalue == 0.0

    def test_reciprocity_zero_s_matrix(self) -> None:
        """Reciprocity check handles all-zero S-matrix."""
        data = self._make_sparam_data(s11=0.0, s21=0.0)
        result = check_reciprocity(data)
        assert result.status == ValidationStatus.PASS
        assert result.max_error == 0.0


# =============================================================================
# Threshold Behavior Tests
# =============================================================================


class TestThresholdBehavior:
    """Tests for threshold behavior between pass/warn/fail."""

    def _make_non_passive_data(self, gain: float, s11: float = 0.0) -> SParameterData:
        """Create non-passive data with specified gain and optional s11."""
        frequencies_hz = np.linspace(1e9, 10e9, 11)
        s_parameters = np.zeros((11, 2, 2), dtype=np.complex128)
        s_parameters[:, 0, 0] = s11
        s_parameters[:, 1, 1] = s11
        s_parameters[:, 1, 0] = gain
        s_parameters[:, 0, 1] = gain
        return SParameterData(
            frequencies_hz=frequencies_hz,
            s_parameters=s_parameters,
            n_ports=2,
        )

    def test_passivity_warn_threshold(self) -> None:
        """Passivity returns WARN between tolerance and warn_threshold."""
        # S21=1.0002 with s11=0 gives eigenvalue ~ 1.0004 (which is |S21|^2)
        # This is between 1+tolerance (1.000001) and 1+warn_threshold (1.001)
        data = self._make_non_passive_data(1.0002, s11=0.0)

        # With tolerance=1e-6 and warn_threshold=1e-3 (=0.001)
        # eigenvalue ~1.0004, which exceeds 1+tolerance but within 1+warn_threshold
        result = check_passivity(data, tolerance=1e-6, warn_threshold=1e-3)
        assert result.status == ValidationStatus.WARN

    def test_passivity_fail_threshold(self) -> None:
        """Passivity returns FAIL above warn_threshold."""
        # S21=1.02 with s11=0 gives eigenvalue ~ 1.04 (which is |S21|^2)
        # This exceeds 1+warn_threshold (1.001)
        data = self._make_non_passive_data(1.02, s11=0.0)

        result = check_passivity(data, tolerance=1e-6, warn_threshold=1e-3)
        assert result.status == ValidationStatus.FAIL

    def test_reciprocity_warn_threshold(self) -> None:
        """Reciprocity returns WARN between tolerance and warn_threshold."""
        frequencies_hz = np.linspace(1e9, 10e9, 11)
        s_parameters = np.zeros((11, 2, 2), dtype=np.complex128)
        s_parameters[:, 0, 0] = 0.1
        s_parameters[:, 1, 1] = 0.1
        s_parameters[:, 1, 0] = 0.9
        s_parameters[:, 0, 1] = 0.9 + 1e-4  # Small asymmetry
        data = SParameterData(
            frequencies_hz=frequencies_hz,
            s_parameters=s_parameters,
            n_ports=2,
        )

        result = check_reciprocity(data, tolerance=1e-6, warn_threshold=1e-3)
        # max_error ~1e-4, exceeds tolerance, within warn_threshold
        assert result.status == ValidationStatus.WARN


# =============================================================================
# Stability Factor Tests
# =============================================================================


class TestStabilityFactor:
    """Tests for stability factor K computation."""

    def _make_stable_data(self) -> SParameterData:
        """Create unconditionally stable S-parameter data."""
        frequencies_hz = np.linspace(1e9, 10e9, 11)
        s_parameters = np.zeros((11, 2, 2), dtype=np.complex128)
        # Lossy network with good return loss
        s_parameters[:, 0, 0] = 0.1
        s_parameters[:, 1, 1] = 0.1
        s_parameters[:, 1, 0] = 0.8
        s_parameters[:, 0, 1] = 0.8
        return SParameterData(
            frequencies_hz=frequencies_hz,
            s_parameters=s_parameters,
            n_ports=2,
        )

    def test_k_factor_requires_2port(self) -> None:
        """K factor computation requires 2-port network."""
        freqs = np.array([1e9])
        s_params = np.zeros((1, 1, 1), dtype=np.complex128)
        data = SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=1,
        )

        with pytest.raises(ValueError, match="2-port"):
            compute_stability_k_factor(data)

    def test_k_factor_array_length(self) -> None:
        """K factor array has one value per frequency."""
        data = self._make_stable_data()
        k = compute_stability_k_factor(data)

        assert len(k) == data.n_frequencies

    def test_check_stability_2port_skip_1port(self) -> None:
        """Stability check skips for 1-port network."""
        freqs = np.array([1e9])
        s_params = np.zeros((1, 1, 1), dtype=np.complex128)
        data = SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=1,
        )

        result = check_stability_2port(data)
        assert result["status"] == "skip"

    def test_check_stability_2port_structure(self) -> None:
        """Stability check result has expected structure."""
        data = self._make_stable_data()
        result = check_stability_2port(data)

        assert result["check"] == "stability_2port"
        assert "is_unconditionally_stable" in result
        assert "k_min" in result
        assert "k_max" in result
        assert "k_mean" in result
        assert "delta_max" in result
        assert "message" in result

    def test_k_factor_zero_transmission(self) -> None:
        """K factor handles zero S21/S12 (infinite K)."""
        frequencies_hz = np.linspace(1e9, 10e9, 11)
        s_parameters = np.zeros((11, 2, 2), dtype=np.complex128)
        s_parameters[:, 0, 0] = 0.5  # Some reflection
        s_parameters[:, 1, 1] = 0.5
        # S21 = S12 = 0 -> denominator = 0 -> K = inf
        data = SParameterData(
            frequencies_hz=frequencies_hz,
            s_parameters=s_parameters,
            n_ports=2,
        )

        k = compute_stability_k_factor(data)
        assert np.all(np.isinf(k))


# =============================================================================
# Combined Validation Tests
# =============================================================================


class TestCombinedValidation:
    """Tests for combined validation function."""

    def _make_good_data(self) -> SParameterData:
        """Create data that passes all validations."""
        n_freq = 101  # More points for better causality
        frequencies_hz = np.linspace(0.1e9, 10e9, n_freq)
        s_parameters = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        s_parameters[:, 0, 0] = 0.1
        s_parameters[:, 1, 1] = 0.1
        s_parameters[:, 1, 0] = 0.85
        s_parameters[:, 0, 1] = 0.85
        return SParameterData(
            frequencies_hz=frequencies_hz,
            s_parameters=s_parameters,
            n_ports=2,
        )

    def test_validate_sparam_data_overall_pass(self) -> None:
        """Overall status is PASS when all checks pass."""
        data = self._make_good_data()
        result = validate_sparam_data(data)

        assert result.overall_status == ValidationStatus.PASS
        assert result.is_valid

    def test_validate_sparam_data_passivity_fails_overall(self) -> None:
        """Overall status is FAIL when passivity fails."""
        frequencies_hz = np.linspace(1e9, 10e9, 11)
        s_parameters = np.zeros((11, 2, 2), dtype=np.complex128)
        s_parameters[:, 1, 0] = 1.5  # Active network
        s_parameters[:, 0, 1] = 1.5
        data = SParameterData(
            frequencies_hz=frequencies_hz,
            s_parameters=s_parameters,
            n_ports=2,
        )

        result = validate_sparam_data(data)
        assert result.overall_status == ValidationStatus.FAIL
        assert result.passivity.status == ValidationStatus.FAIL

    def test_validate_sparam_data_skip_causality(self) -> None:
        """skip_causality option skips causality check."""
        data = self._make_good_data()
        result = validate_sparam_data(data, skip_causality=True)

        assert result.causality.status == ValidationStatus.SKIP
        assert "skipped" in result.causality.message.lower()

    def test_validate_sparam_data_custom_tolerances(self) -> None:
        """Custom tolerances are applied."""
        data = self._make_good_data()
        result = validate_sparam_data(
            data,
            passivity_tolerance=1e-10,
            reciprocity_tolerance=1e-10,
            causality_tolerance=1e-6,
        )

        # All should still pass with good data
        assert result.is_valid


class TestManifestEntry:
    """Tests for manifest entry building."""

    def _make_validation_result(self) -> SParameterValidationResult:
        """Create a validation result."""
        return SParameterValidationResult(
            passivity=PassivityCheckResult(
                status=ValidationStatus.PASS,
                max_eigenvalue=0.95,
                n_violations=0,
                violation_frequencies_hz=(),
                tolerance=1e-6,
                message="Pass",
            ),
            reciprocity=ReciprocityCheckResult(
                status=ValidationStatus.PASS,
                max_error=1e-10,
                mean_error=5e-11,
                max_error_db=-200.0,
                n_violations=0,
                tolerance=1e-6,
                message="Pass",
            ),
            causality=CausalityCheckResult(
                status=ValidationStatus.PASS,
                is_causal=True,
                pre_response_energy_ratio=1e-6,
                max_pre_response_magnitude=1e-5,
                tolerance=1e-3,
                message="Pass",
            ),
            overall_status=ValidationStatus.PASS,
            n_frequencies=101,
            n_ports=2,
        )

    def test_build_validation_manifest_entry_structure(self) -> None:
        """Manifest entry has expected structure."""
        result = self._make_validation_result()
        entry = build_validation_manifest_entry(result)

        assert "overall_status" in entry
        assert "is_valid" in entry
        assert "has_warnings" in entry
        assert "n_frequencies" in entry
        assert "n_ports" in entry
        assert "checks" in entry
        assert "passivity" in entry["checks"]
        assert "reciprocity" in entry["checks"]
        assert "causality" in entry["checks"]

    def test_build_validation_manifest_entry_equals_to_dict(self) -> None:
        """Manifest entry equals to_dict output."""
        result = self._make_validation_result()
        entry = build_validation_manifest_entry(result)
        direct = result.to_dict()

        assert entry == direct

    def test_manifest_entry_with_extra_checks(self) -> None:
        """Manifest entry includes extra checks."""
        result = self._make_validation_result()
        result.add_extra_check("stability", {"status": "pass", "k_min": 1.5})

        entry = build_validation_manifest_entry(result)
        assert "stability" in entry["checks"]
        assert entry["checks"]["stability"]["k_min"] == 1.5
