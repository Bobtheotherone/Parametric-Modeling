"""Tests for verification suite metrics (REQ-M2-020).

These tests validate that the verification suite enforces passivity, reciprocity,
and causality checks in strict mode with configurable thresholds.

REQ-M2-020: test_verification_enforces_passivity_reciprocity_and_causality_in_strict_mode
"""

from __future__ import annotations

import numpy as np
import pytest
from formula_foundry.verification.metrics import (
    CausalityCheckConfig,
    CausalityFailure,
    CausalityMetrics,
    PassivityCheckConfig,
    PassivityFailure,
    PassivityMetrics,
    ReciprocityCheckConfig,
    ReciprocityFailure,
    ReciprocityMetrics,
    VerificationConfig,
    VerificationMetrics,
    VerificationReport,
    VerificationStatus,
    check_causality,
    check_passivity,
    check_reciprocity,
    create_non_passive_sparam_2port,
    create_non_reciprocal_sparam_2port,
    create_passive_sparam_2port,
    run_verification_suite,
)
from numpy.typing import NDArray

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def frequencies_hz() -> NDArray[np.float64]:
    """Standard frequency array for testing."""
    return np.linspace(1e9, 10e9, 101)


@pytest.fixture
def passive_2port(frequencies_hz: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Create passive 2-port S-parameters for testing."""
    return create_passive_sparam_2port(frequencies_hz, insertion_loss_db=-0.1)


@pytest.fixture
def non_passive_2port(frequencies_hz: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Create non-passive 2-port S-parameters (with gain)."""
    return create_non_passive_sparam_2port(frequencies_hz, gain_db=3.0)


@pytest.fixture
def non_reciprocal_2port(frequencies_hz: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Create non-reciprocal 2-port S-parameters."""
    return create_non_reciprocal_sparam_2port(frequencies_hz, asymmetry=0.5)


# =============================================================================
# Passivity Check Tests
# =============================================================================


class TestPassivityCheck:
    """Tests for passivity validation."""

    def test_passive_network_passes(
        self,
        passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Passive network should pass passivity check."""
        result = check_passivity(passive_2port, frequencies_hz, strict=True)
        assert result.status == VerificationStatus.PASS
        assert result.max_singular_value <= 1.0 + 1e-6
        assert result.n_violations == 0
        assert "passive" in result.message.lower()

    def test_non_passive_network_fails_strict(
        self,
        non_passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Non-passive network should raise PassivityFailure in strict mode."""
        with pytest.raises(PassivityFailure) as exc_info:
            check_passivity(non_passive_2port, frequencies_hz, strict=True)
        assert exc_info.value.max_singular_value > 1.0
        assert exc_info.value.n_violations > 0

    def test_non_passive_network_non_strict(
        self,
        non_passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Non-passive network in non-strict mode returns FAIL status."""
        result = check_passivity(non_passive_2port, frequencies_hz, strict=False)
        assert result.status == VerificationStatus.FAIL
        assert result.max_singular_value > 1.0

    def test_passivity_threshold_configurable(
        self,
        non_passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Passivity threshold should be configurable."""
        # With a high threshold, even non-passive network passes
        config = PassivityCheckConfig(threshold=2.0, warn_margin=1.0)
        result = check_passivity(non_passive_2port, frequencies_hz, config=config, strict=True)
        assert result.status == VerificationStatus.PASS

    def test_passivity_disabled(
        self,
        non_passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Disabled passivity check returns SKIP status."""
        config = PassivityCheckConfig(enabled=False)
        result = check_passivity(non_passive_2port, frequencies_hz, config=config, strict=True)
        assert result.status == VerificationStatus.SKIP
        assert "skipped" in result.message.lower()

    def test_passivity_reports_violation_frequencies(
        self,
        non_passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Passivity check should report frequencies where violations occur."""
        result = check_passivity(non_passive_2port, frequencies_hz, strict=False)
        assert len(result.violation_frequencies_hz) > 0
        # All violation frequencies should be from our frequency array
        for f in result.violation_frequencies_hz:
            assert f in frequencies_hz

    def test_passivity_metrics_to_dict(
        self,
        passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """PassivityMetrics should convert to dict for serialization."""
        result = check_passivity(passive_2port, frequencies_hz, strict=True)
        d = result.to_dict()
        assert d["check"] == "passivity"
        assert d["status"] == "pass"
        assert "max_singular_value" in d
        assert "threshold" in d


# =============================================================================
# Reciprocity Check Tests
# =============================================================================


class TestReciprocityCheck:
    """Tests for reciprocity validation."""

    def test_reciprocal_network_passes(
        self,
        passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Reciprocal network should pass reciprocity check."""
        result = check_reciprocity(passive_2port, frequencies_hz, strict=True)
        assert result.status == VerificationStatus.PASS
        assert result.max_error < 1e-6
        assert "reciprocal" in result.message.lower()

    def test_non_reciprocal_network_fails_strict(
        self,
        non_reciprocal_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Non-reciprocal network should raise ReciprocityFailure in strict mode."""
        with pytest.raises(ReciprocityFailure) as exc_info:
            check_reciprocity(non_reciprocal_2port, frequencies_hz, strict=True)
        assert exc_info.value.max_error > 1e-6
        assert exc_info.value.worst_pair == (1, 2)

    def test_non_reciprocal_network_non_strict(
        self,
        non_reciprocal_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Non-reciprocal network in non-strict mode returns FAIL status."""
        result = check_reciprocity(non_reciprocal_2port, frequencies_hz, strict=False)
        assert result.status == VerificationStatus.FAIL
        assert result.max_error > 0

    def test_reciprocity_threshold_configurable(
        self,
        non_reciprocal_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Reciprocity threshold should be configurable."""
        # With a high threshold, even non-reciprocal network passes
        config = ReciprocityCheckConfig(threshold=1.0, warn_margin=1.0)
        result = check_reciprocity(non_reciprocal_2port, frequencies_hz, config=config, strict=True)
        assert result.status == VerificationStatus.PASS

    def test_reciprocity_disabled(
        self,
        non_reciprocal_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Disabled reciprocity check returns SKIP status."""
        config = ReciprocityCheckConfig(enabled=False)
        result = check_reciprocity(non_reciprocal_2port, frequencies_hz, config=config, strict=True)
        assert result.status == VerificationStatus.SKIP

    def test_reciprocity_reports_worst_pair(
        self,
        non_reciprocal_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Reciprocity check should report the worst port pair."""
        result = check_reciprocity(non_reciprocal_2port, frequencies_hz, strict=False)
        assert result.worst_pair == (1, 2)  # 1-based port indices

    def test_reciprocity_metrics_to_dict(
        self,
        passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """ReciprocityMetrics should convert to dict for serialization."""
        result = check_reciprocity(passive_2port, frequencies_hz, strict=True)
        d = result.to_dict()
        assert d["check"] == "reciprocity"
        assert "max_error" in d
        assert "worst_pair" in d


# =============================================================================
# Causality Check Tests
# =============================================================================


class TestCausalityCheck:
    """Tests for causality validation."""

    def test_causal_network_passes(
        self,
        passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Causal network should pass causality check."""
        result = check_causality(passive_2port, frequencies_hz, strict=True)
        assert result.status in (VerificationStatus.PASS, VerificationStatus.WARN)
        assert result.is_causal is True

    def test_non_causal_network_fails_strict(
        self,
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Non-causal network should raise CausalityFailure in strict mode."""
        # Create a non-causal response by adding pre-response energy
        n_freq = len(frequencies_hz)
        s_params = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        # Random phase that creates non-causal response
        s_params[:, 1, 0] = 0.9 * np.exp(1j * np.random.uniform(-np.pi, np.pi, n_freq))
        s_params[:, 0, 1] = s_params[:, 1, 0]

        # Use very tight threshold to force failure
        config = CausalityCheckConfig(threshold=1e-10, warn_margin=1e-9)
        with pytest.raises(CausalityFailure):
            check_causality(s_params, frequencies_hz, config=config, strict=True)

    def test_non_causal_network_non_strict(
        self,
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Non-causal network in non-strict mode returns FAIL status."""
        n_freq = len(frequencies_hz)
        s_params = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        s_params[:, 1, 0] = 0.9 * np.exp(1j * np.random.uniform(-np.pi, np.pi, n_freq))
        s_params[:, 0, 1] = s_params[:, 1, 0]

        config = CausalityCheckConfig(threshold=1e-10, warn_margin=1e-9)
        result = check_causality(s_params, frequencies_hz, config=config, strict=False)
        assert result.status == VerificationStatus.FAIL

    def test_causality_threshold_configurable(
        self,
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Causality threshold should be configurable."""
        n_freq = len(frequencies_hz)
        s_params = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        s_params[:, 1, 0] = 0.9 * np.exp(1j * np.random.uniform(-np.pi, np.pi, n_freq))

        # With a very high threshold, even problematic data passes
        config = CausalityCheckConfig(threshold=1.0, warn_margin=1.0)
        result = check_causality(s_params, frequencies_hz, config=config, strict=True)
        assert result.status == VerificationStatus.PASS

    def test_causality_disabled(
        self,
        passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Disabled causality check returns SKIP status."""
        config = CausalityCheckConfig(enabled=False)
        result = check_causality(passive_2port, frequencies_hz, config=config, strict=True)
        assert result.status == VerificationStatus.SKIP

    def test_causality_metrics_to_dict(
        self,
        passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """CausalityMetrics should convert to dict for serialization."""
        result = check_causality(passive_2port, frequencies_hz, strict=True)
        d = result.to_dict()
        assert d["check"] == "causality"
        assert "pre_response_energy_ratio" in d
        assert "is_causal" in d


# =============================================================================
# Combined Verification Suite Tests (REQ-M2-020)
# =============================================================================


class TestVerificationSuite:
    """Tests for the complete verification suite (REQ-M2-020)."""

    def test_verification_enforces_passivity_reciprocity_and_causality_in_strict_mode(
        self,
        passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """REQ-M2-020: Verification enforces all checks in strict mode."""
        config = VerificationConfig(strict=True)
        report = run_verification_suite(passive_2port, frequencies_hz, config=config)

        # All checks should pass for a valid passive, reciprocal, causal network
        assert report.metrics.all_passed is True
        assert report.strict_mode_active is True
        assert report.metrics.passivity.status in (VerificationStatus.PASS, VerificationStatus.WARN)
        assert report.metrics.reciprocity.status in (VerificationStatus.PASS, VerificationStatus.WARN)
        assert report.metrics.causality.status in (
            VerificationStatus.PASS,
            VerificationStatus.WARN,
            VerificationStatus.SKIP,
        )

    def test_strict_mode_raises_on_passivity_failure(
        self,
        non_passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Strict mode should raise PassivityFailure on passivity violation."""
        config = VerificationConfig(strict=True)
        with pytest.raises(PassivityFailure):
            run_verification_suite(non_passive_2port, frequencies_hz, config=config)

    def test_strict_mode_raises_on_reciprocity_failure(
        self,
        non_reciprocal_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Strict mode should raise ReciprocityFailure on reciprocity violation."""
        config = VerificationConfig(strict=True)
        with pytest.raises(ReciprocityFailure):
            run_verification_suite(non_reciprocal_2port, frequencies_hz, config=config)

    def test_non_strict_mode_reports_failures(
        self,
        non_passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Non-strict mode should report failures without raising."""
        config = VerificationConfig(strict=False)
        report = run_verification_suite(non_passive_2port, frequencies_hz, config=config)

        assert report.metrics.has_failures is True
        assert report.metrics.passivity.status == VerificationStatus.FAIL

    def test_verification_report_structure(
        self,
        passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """VerificationReport should have correct structure."""
        config = VerificationConfig(strict=True)
        report = run_verification_suite(passive_2port, frequencies_hz, config=config)

        assert isinstance(report, VerificationReport)
        assert isinstance(report.metrics, VerificationMetrics)
        assert isinstance(report.config, VerificationConfig)
        assert isinstance(report.metrics.passivity, PassivityMetrics)
        assert isinstance(report.metrics.reciprocity, ReciprocityMetrics)
        assert isinstance(report.metrics.causality, CausalityMetrics)

    def test_verification_report_to_dict(
        self,
        passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """VerificationReport should convert to dict for serialization."""
        config = VerificationConfig(strict=True)
        report = run_verification_suite(passive_2port, frequencies_hz, config=config)

        d = report.to_dict()
        assert "metrics" in d
        assert "config" in d
        assert "strict_mode_active" in d
        assert d["metrics"]["checks"]["passivity"]["check"] == "passivity"
        assert d["metrics"]["checks"]["reciprocity"]["check"] == "reciprocity"
        assert d["metrics"]["checks"]["causality"]["check"] == "causality"

    def test_selective_check_disabling(
        self,
        non_passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Individual checks can be disabled."""
        config = VerificationConfig(
            strict=True,
            passivity=PassivityCheckConfig(enabled=False),
            reciprocity=ReciprocityCheckConfig(enabled=True),
            causality=CausalityCheckConfig(enabled=False),
        )
        # Should not raise because passivity check is disabled
        report = run_verification_suite(non_passive_2port, frequencies_hz, config=config)

        assert report.metrics.passivity.status == VerificationStatus.SKIP
        assert report.metrics.causality.status == VerificationStatus.SKIP
        assert report.metrics.reciprocity.status in (VerificationStatus.PASS, VerificationStatus.WARN)

    def test_overall_status_reflects_worst_check(
        self,
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Overall status should be worst of individual checks."""
        # Create S-params that pass passivity/causality but fail reciprocity
        s_params = create_non_reciprocal_sparam_2port(frequencies_hz, asymmetry=0.01)
        # Make it passive (low insertion loss)
        s_params[:, 0, 0] = 0.1
        s_params[:, 1, 1] = 0.1

        config = VerificationConfig(
            strict=False,
            reciprocity=ReciprocityCheckConfig(threshold=1e-9),  # Very tight
        )
        report = run_verification_suite(s_params, frequencies_hz, config=config)

        # Reciprocity should fail, so overall should be FAIL
        assert report.metrics.reciprocity.status == VerificationStatus.FAIL
        assert report.metrics.overall_status == VerificationStatus.FAIL

    def test_n_ports_recorded(
        self,
        passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Number of ports should be recorded in metrics."""
        report = run_verification_suite(passive_2port, frequencies_hz)
        assert report.metrics.n_ports == 2

    def test_n_frequencies_recorded(
        self,
        passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Number of frequencies should be recorded in metrics."""
        report = run_verification_suite(passive_2port, frequencies_hz)
        assert report.metrics.n_frequencies == len(frequencies_hz)


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for exception classes."""

    def test_passivity_failure_message(self) -> None:
        """PassivityFailure should have informative message."""
        error = PassivityFailure(
            max_singular_value=1.5,
            threshold=1.0,
            n_violations=10,
            violation_frequencies_hz=(1e9, 2e9, 3e9),
        )
        assert error.max_singular_value == 1.5
        assert error.threshold == 1.0
        assert "1.5" in str(error)
        assert "exceeds threshold" in str(error).lower()

    def test_reciprocity_failure_message(self) -> None:
        """ReciprocityFailure should have informative message."""
        error = ReciprocityFailure(
            max_error=0.1,
            threshold=1e-6,
            n_violations=100,
            worst_pair=(1, 2),
        )
        assert error.max_error == 0.1
        assert error.worst_pair == (1, 2)
        assert "S12" in str(error) or "S21" in str(error)

    def test_causality_failure_message(self) -> None:
        """CausalityFailure should have informative message."""
        error = CausalityFailure(
            pre_response_energy_ratio=0.5,
            threshold=1e-3,
        )
        assert error.pre_response_energy_ratio == 0.5
        assert "non-causal" in str(error).lower()


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_passive_sparam_2port(
        self,
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """create_passive_sparam_2port should create valid passive network."""
        s_params = create_passive_sparam_2port(frequencies_hz)
        assert s_params.shape == (len(frequencies_hz), 2, 2)
        # Check passivity
        for f_idx in range(len(frequencies_hz)):
            svs = np.linalg.svd(s_params[f_idx], compute_uv=False)
            assert np.max(svs) <= 1.0 + 1e-6

    def test_create_non_passive_sparam_2port(
        self,
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """create_non_passive_sparam_2port should create network with gain."""
        s_params = create_non_passive_sparam_2port(frequencies_hz, gain_db=6.0)
        # |S21| should be > 1 (gain)
        s21_mag = np.abs(s_params[:, 1, 0])
        assert np.all(s21_mag > 1.0)

    def test_create_non_reciprocal_sparam_2port(
        self,
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """create_non_reciprocal_sparam_2port should create S12 != S21."""
        s_params = create_non_reciprocal_sparam_2port(frequencies_hz, asymmetry=0.3)
        # |S21 - S12| should be significant
        asymmetry = np.abs(s_params[:, 1, 0] - s_params[:, 0, 1])
        assert np.all(asymmetry > 0.2)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_frequency_point(self) -> None:
        """Verification should work with single frequency point."""
        frequencies_hz = np.array([1e9])
        s_params = create_passive_sparam_2port(frequencies_hz)
        report = run_verification_suite(s_params, frequencies_hz)
        assert report.metrics.n_frequencies == 1

    def test_1port_network(self) -> None:
        """Verification should work with 1-port network."""
        frequencies_hz = np.linspace(1e9, 10e9, 101)
        s_params = np.zeros((len(frequencies_hz), 1, 1), dtype=np.complex128)
        s_params[:, 0, 0] = 0.1  # Low reflection

        report = run_verification_suite(s_params, frequencies_hz)
        assert report.metrics.n_ports == 1
        # Reciprocity check should pass trivially for 1-port
        assert report.metrics.reciprocity.status in (VerificationStatus.PASS, VerificationStatus.SKIP)

    def test_4port_network(self) -> None:
        """Verification should work with 4-port network."""
        frequencies_hz = np.linspace(1e9, 10e9, 101)
        n_freq = len(frequencies_hz)
        s_params = np.zeros((n_freq, 4, 4), dtype=np.complex128)
        # Make it passive and reciprocal
        for i in range(4):
            s_params[:, i, i] = 0.1
            for j in range(i + 1, 4):
                s_params[:, i, j] = 0.2
                s_params[:, j, i] = 0.2

        report = run_verification_suite(s_params, frequencies_hz)
        assert report.metrics.n_ports == 4
        assert report.metrics.passivity.status == VerificationStatus.PASS
        assert report.metrics.reciprocity.status == VerificationStatus.PASS

    def test_marginal_passivity_warning(
        self,
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """Marginal passivity should produce warning, not failure."""
        n_freq = len(frequencies_hz)
        s_params = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        # Slightly above threshold but within warning margin
        s_params[:, 1, 0] = 1.0001
        s_params[:, 0, 1] = 1.0001

        config = VerificationConfig(
            strict=False,
            passivity=PassivityCheckConfig(threshold=1.0, warn_margin=0.001),
        )
        report = run_verification_suite(s_params, frequencies_hz, config=config)
        assert report.metrics.passivity.status == VerificationStatus.WARN

    def test_all_checks_disabled(
        self,
        non_passive_2port: NDArray[np.complex128],
        frequencies_hz: NDArray[np.float64],
    ) -> None:
        """All checks disabled should produce SKIP overall status."""
        config = VerificationConfig(
            strict=True,
            passivity=PassivityCheckConfig(enabled=False),
            reciprocity=ReciprocityCheckConfig(enabled=False),
            causality=CausalityCheckConfig(enabled=False),
        )
        report = run_verification_suite(non_passive_2port, frequencies_hz, config=config)

        assert report.metrics.passivity.status == VerificationStatus.SKIP
        assert report.metrics.reciprocity.status == VerificationStatus.SKIP
        assert report.metrics.causality.status == VerificationStatus.SKIP
        assert report.metrics.overall_status == VerificationStatus.SKIP
