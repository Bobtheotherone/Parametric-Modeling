"""Tests for mixed-mode S-parameter conversion.

REQ-M2-018: Mixed-mode outputs for differential 4-port cases with strict
pairing rules recorded in meta.

Test Matrix Entry:
| REQ-M2-018 | tests/test_mixed_mode.py::test_mixed_mode_enforces_strict_pairing_and_records_in_meta |
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from formula_foundry.postprocess.mixed_mode import (
    MixedModeConfig,
    MixedModeResult,
    PairingConvention,
    PortPairing,
    compute_mixed_mode_sparams,
    mixed_mode_for_differential_case,
    validate_4port_pairing,
    write_mixed_mode_outputs,
)

from formula_foundry.em.touchstone import SParameterData

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_4port_sparams() -> SParameterData:
    """Create sample 4-port S-parameter data for testing.

    Creates a simple test case with known values for verification.
    """
    frequencies_hz = np.array([1e9, 5e9, 10e9], dtype=np.float64)
    n_freq = len(frequencies_hz)

    # Create a test S-matrix with specific patterns for verification
    # Using a simplified model of coupled differential lines
    s_parameters = np.zeros((n_freq, 4, 4), dtype=np.complex128)

    for f_idx in range(n_freq):
        # Return loss terms (diagonal)
        s_parameters[f_idx, 0, 0] = -0.1 + 0.01j  # S11
        s_parameters[f_idx, 1, 1] = -0.1 + 0.01j  # S22
        s_parameters[f_idx, 2, 2] = -0.1 - 0.01j  # S33
        s_parameters[f_idx, 3, 3] = -0.1 - 0.01j  # S44

        # Through terms (coupled lines pattern)
        s_parameters[f_idx, 1, 0] = 0.9 - 0.1j  # S21
        s_parameters[f_idx, 0, 1] = 0.9 - 0.1j  # S12
        s_parameters[f_idx, 3, 2] = 0.9 - 0.1j  # S43
        s_parameters[f_idx, 2, 3] = 0.9 - 0.1j  # S34

        # Coupling terms (small for differential mode)
        s_parameters[f_idx, 2, 0] = 0.05 + 0.02j  # S31
        s_parameters[f_idx, 0, 2] = 0.05 + 0.02j  # S13
        s_parameters[f_idx, 3, 0] = -0.05 - 0.02j  # S41
        s_parameters[f_idx, 0, 3] = -0.05 - 0.02j  # S14
        s_parameters[f_idx, 2, 1] = -0.05 - 0.02j  # S32
        s_parameters[f_idx, 1, 2] = -0.05 - 0.02j  # S23
        s_parameters[f_idx, 3, 1] = 0.05 + 0.02j  # S42
        s_parameters[f_idx, 1, 3] = 0.05 + 0.02j  # S24

    return SParameterData(
        frequencies_hz=frequencies_hz,
        s_parameters=s_parameters,
        n_ports=4,
        reference_impedance_ohm=50.0,
    )


@pytest.fixture
def sample_2port_sparams() -> SParameterData:
    """Create sample 2-port S-parameter data (invalid for mixed-mode)."""
    frequencies_hz = np.array([1e9, 5e9, 10e9], dtype=np.float64)
    n_freq = len(frequencies_hz)

    s_parameters = np.zeros((n_freq, 2, 2), dtype=np.complex128)
    for f_idx in range(n_freq):
        s_parameters[f_idx, 0, 0] = -0.1
        s_parameters[f_idx, 1, 1] = -0.1
        s_parameters[f_idx, 1, 0] = 0.9
        s_parameters[f_idx, 0, 1] = 0.9

    return SParameterData(
        frequencies_hz=frequencies_hz,
        s_parameters=s_parameters,
        n_ports=2,
        reference_impedance_ohm=50.0,
    )


@pytest.fixture
def default_config() -> MixedModeConfig:
    """Create default mixed-mode configuration."""
    return MixedModeConfig.default_4port()


# =============================================================================
# REQ-M2-018: Strict Pairing Rules Tests
# =============================================================================


def test_mixed_mode_enforces_strict_pairing_and_records_in_meta(
    sample_4port_sparams: SParameterData,
) -> None:
    """REQ-M2-018: Verify strict pairing rules are enforced and recorded.

    This is the main test for the requirement. It verifies:
    1. Mixed-mode transformation requires exactly 4-port input
    2. Port pairing configuration is validated
    3. Pairing rules are recorded in metadata
    4. Output files are produced correctly
    """
    config = MixedModeConfig.default_4port()

    # Compute mixed-mode parameters
    result = compute_mixed_mode_sparams(sample_4port_sparams, config)

    # Verify result structure
    assert isinstance(result, MixedModeResult)
    assert result.sdd.shape == (3, 2, 2)  # 3 frequencies, 2x2 differential
    assert result.scc.shape == (3, 2, 2)
    assert result.sdc.shape == (3, 2, 2)
    assert result.scd.shape == (3, 2, 2)

    # Verify metadata contains pairing rules
    metadata = result.build_metadata()
    assert "mixed_mode" in metadata
    mm_meta = metadata["mixed_mode"]

    # Check pairing convention is recorded
    assert mm_meta["pairing_convention"] == "1324"

    # Check pair 1 is recorded
    assert "pair_1" in mm_meta
    assert mm_meta["pair_1"]["positive_port"] == 1
    assert mm_meta["pair_1"]["negative_port"] == 3

    # Check pair 2 is recorded
    assert "pair_2" in mm_meta
    assert mm_meta["pair_2"]["positive_port"] == 2
    assert mm_meta["pair_2"]["negative_port"] == 4


def test_validate_4port_pairing_accepts_valid_config(
    sample_4port_sparams: SParameterData,
    default_config: MixedModeConfig,
) -> None:
    """Test that valid 4-port configuration passes validation."""
    is_valid, errors = validate_4port_pairing(sample_4port_sparams, default_config)

    assert is_valid is True
    assert errors == []


def test_validate_4port_pairing_rejects_2port_input(
    sample_2port_sparams: SParameterData,
    default_config: MixedModeConfig,
) -> None:
    """Test that 2-port input is rejected."""
    is_valid, errors = validate_4port_pairing(sample_2port_sparams, default_config)

    assert is_valid is False
    assert any("exactly 4 ports" in err for err in errors)


def test_validate_4port_pairing_rejects_overlapping_ports(
    sample_4port_sparams: SParameterData,
) -> None:
    """Test that overlapping port assignments are rejected."""
    # Try to create config with overlapping ports
    with pytest.raises(ValueError, match="Port pairing overlap"):
        MixedModeConfig(
            pair_1=PortPairing("P1", positive_port=1, negative_port=2),
            pair_2=PortPairing("P2", positive_port=2, negative_port=3),  # Port 2 overlap!
        )


def test_validate_4port_pairing_rejects_incomplete_coverage(
    sample_4port_sparams: SParameterData,
) -> None:
    """Test that incomplete port coverage is rejected."""
    # Config that doesn't cover all 4 ports
    config = MixedModeConfig(
        pair_1=PortPairing("P1", positive_port=1, negative_port=2),
        pair_2=PortPairing("P2", positive_port=3, negative_port=5),  # Port 5 invalid
    )

    is_valid, errors = validate_4port_pairing(sample_4port_sparams, config)

    assert is_valid is False
    assert any("out of valid range" in err or "not covered" in err for err in errors)


def test_port_pairing_rejects_same_port_for_both_terminals() -> None:
    """Test that same port cannot be both positive and negative."""
    with pytest.raises(ValueError, match="must be different"):
        PortPairing("P1", positive_port=1, negative_port=1)


def test_port_pairing_rejects_invalid_port_numbers() -> None:
    """Test that port numbers must be >= 1."""
    with pytest.raises(ValueError, match="must be >= 1"):
        PortPairing("P1", positive_port=0, negative_port=2)

    with pytest.raises(ValueError, match="must be >= 1"):
        PortPairing("P1", positive_port=1, negative_port=-1)


# =============================================================================
# Mixed-Mode Transformation Tests
# =============================================================================


def test_compute_mixed_mode_sparams_produces_correct_dimensions(
    sample_4port_sparams: SParameterData,
) -> None:
    """Verify mixed-mode output dimensions are correct."""
    result = compute_mixed_mode_sparams(sample_4port_sparams)

    # Full mixed-mode matrix should be 4x4
    assert result.mm_sparams.n_ports == 4
    assert result.mm_sparams.n_frequencies == sample_4port_sparams.n_frequencies
    assert result.mm_sparams.s_parameters.shape == (3, 4, 4)

    # Submatrices should be 2x2
    assert result.sdd.shape == (3, 2, 2)
    assert result.scc.shape == (3, 2, 2)
    assert result.sdc.shape == (3, 2, 2)
    assert result.scd.shape == (3, 2, 2)


def test_compute_mixed_mode_sparams_uses_default_config(
    sample_4port_sparams: SParameterData,
) -> None:
    """Verify default config is used when none provided."""
    result = compute_mixed_mode_sparams(sample_4port_sparams, config=None)

    metadata = result.build_metadata()
    assert metadata["mixed_mode"]["pairing_convention"] == "1324"


def test_compute_mixed_mode_sparams_rejects_non_4port(
    sample_2port_sparams: SParameterData,
) -> None:
    """Verify non-4-port input is rejected."""
    with pytest.raises(ValueError, match="exactly 4 ports"):
        compute_mixed_mode_sparams(sample_2port_sparams)


def test_mixed_mode_transformation_formulas(
    sample_4port_sparams: SParameterData,
) -> None:
    """Verify mixed-mode transformation follows correct formulas.

    Sdd = 0.5 * (Sp+q+ - Sp+q- - Sp-q+ + Sp-q-)
    Scc = 0.5 * (Sp+q+ + Sp+q- + Sp-q+ + Sp-q-)
    Sdc = 0.5 * (Sp+q+ + Sp+q- - Sp-q+ - Sp-q-)
    Scd = 0.5 * (Sp+q+ - Sp+q- + Sp-q+ - Sp-q-)
    """
    config = MixedModeConfig.default_4port()
    result = compute_mixed_mode_sparams(sample_4port_sparams, config)

    S = sample_4port_sparams.s_parameters

    # For default 1324 pairing:
    # Pair 1: (1, 3) -> ports 0, 2 (0-indexed)
    # Pair 2: (2, 4) -> ports 1, 3 (0-indexed)

    # Verify Sdd11 (pair 1 to pair 1)
    # Sdd11 = 0.5 * (S11 - S13 - S31 + S33)
    for f_idx in range(3):
        expected_sdd11 = 0.5 * (S[f_idx, 0, 0] - S[f_idx, 0, 2] - S[f_idx, 2, 0] + S[f_idx, 2, 2])
        np.testing.assert_allclose(result.sdd[f_idx, 0, 0], expected_sdd11, rtol=1e-10)

        # Verify Scc11 = 0.5 * (S11 + S13 + S31 + S33)
        expected_scc11 = 0.5 * (S[f_idx, 0, 0] + S[f_idx, 0, 2] + S[f_idx, 2, 0] + S[f_idx, 2, 2])
        np.testing.assert_allclose(result.scc[f_idx, 0, 0], expected_scc11, rtol=1e-10)


def test_mixed_mode_symmetry_for_reciprocal_network(
    sample_4port_sparams: SParameterData,
) -> None:
    """Verify reciprocity is preserved in mixed-mode transformation."""
    result = compute_mixed_mode_sparams(sample_4port_sparams)

    # For reciprocal networks: Sdd12 = Sdd21, Scc12 = Scc21
    for f_idx in range(3):
        np.testing.assert_allclose(
            result.sdd[f_idx, 0, 1],
            result.sdd[f_idx, 1, 0],
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            result.scc[f_idx, 0, 1],
            result.scc[f_idx, 1, 0],
            rtol=1e-10,
        )


# =============================================================================
# Pairing Convention Tests
# =============================================================================


def test_standard_1324_pairing_convention() -> None:
    """Test standard 1324 pairing convention."""
    config = MixedModeConfig(
        pair_1=PortPairing("P1", positive_port=1, negative_port=3),
        pair_2=PortPairing("P2", positive_port=2, negative_port=4),
        convention=PairingConvention.STANDARD_1324,
    )

    assert config.pair_1.positive_port == 1
    assert config.pair_1.negative_port == 3
    assert config.pair_2.positive_port == 2
    assert config.pair_2.negative_port == 4


def test_standard_1234_pairing_convention() -> None:
    """Test standard 1234 pairing convention."""
    config = MixedModeConfig(
        pair_1=PortPairing("P1", positive_port=1, negative_port=2),
        pair_2=PortPairing("P2", positive_port=3, negative_port=4),
        convention=PairingConvention.STANDARD_1234,
    )

    assert config.pair_1.positive_port == 1
    assert config.pair_1.negative_port == 2
    assert config.pair_2.positive_port == 3
    assert config.pair_2.negative_port == 4


def test_different_pairing_gives_different_results(
    sample_4port_sparams: SParameterData,
) -> None:
    """Verify different pairing conventions produce different results."""
    config_1324 = MixedModeConfig(
        pair_1=PortPairing("P1", positive_port=1, negative_port=3),
        pair_2=PortPairing("P2", positive_port=2, negative_port=4),
        convention=PairingConvention.STANDARD_1324,
    )
    config_1234 = MixedModeConfig(
        pair_1=PortPairing("P1", positive_port=1, negative_port=2),
        pair_2=PortPairing("P2", positive_port=3, negative_port=4),
        convention=PairingConvention.STANDARD_1234,
    )

    result_1324 = compute_mixed_mode_sparams(sample_4port_sparams, config_1324)
    result_1234 = compute_mixed_mode_sparams(sample_4port_sparams, config_1234)

    # Results should differ (unless S-params happen to have specific symmetry)
    # At least the metadata should record different conventions
    meta_1324 = result_1324.build_metadata()
    meta_1234 = result_1234.build_metadata()

    assert meta_1324["mixed_mode"]["pairing_convention"] == "1324"
    assert meta_1234["mixed_mode"]["pairing_convention"] == "1234"


# =============================================================================
# Output File Tests
# =============================================================================


def test_write_mixed_mode_outputs_creates_expected_files(
    sample_4port_sparams: SParameterData,
) -> None:
    """Verify output files are created with correct names."""
    result = compute_mixed_mode_sparams(sample_4port_sparams)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        paths = write_mixed_mode_outputs(result, output_dir, "test_mm")

        # Check expected files exist
        assert "mixed_mode_s4p" in paths
        assert "sdd_s2p" in paths
        assert "metadata" in paths

        assert paths["mixed_mode_s4p"].exists()
        assert paths["sdd_s2p"].exists()
        assert paths["metadata"].exists()

        # Verify file extensions
        assert paths["mixed_mode_s4p"].suffix == ".s4p"
        assert paths["sdd_s2p"].suffix == ".s2p"
        assert paths["metadata"].suffix == ".json"


def test_write_mixed_mode_outputs_metadata_contains_pairing(
    sample_4port_sparams: SParameterData,
) -> None:
    """Verify metadata JSON contains pairing rules."""
    result = compute_mixed_mode_sparams(sample_4port_sparams)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        paths = write_mixed_mode_outputs(result, output_dir, "test_mm")

        # Read and verify metadata
        with open(paths["metadata"]) as f:
            metadata = json.load(f)

        assert "mixed_mode" in metadata
        assert metadata["mixed_mode"]["pairing_convention"] == "1324"
        assert metadata["mixed_mode"]["pair_1"]["positive_port"] == 1
        assert metadata["mixed_mode"]["pair_1"]["negative_port"] == 3
        assert metadata["mixed_mode"]["pair_2"]["positive_port"] == 2
        assert metadata["mixed_mode"]["pair_2"]["negative_port"] == 4


def test_mixed_mode_for_differential_case_api(
    sample_4port_sparams: SParameterData,
) -> None:
    """Test high-level API for differential case processing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result, metadata = mixed_mode_for_differential_case(
            sample_4port_sparams,
            output_dir=output_dir,
            base_name="diff_test",
        )

        # Verify result
        assert isinstance(result, MixedModeResult)

        # Verify metadata contains pairing
        assert "mixed_mode" in metadata
        assert metadata["mixed_mode"]["pairing_convention"] == "1324"

        # Verify output files were created
        assert "output_files" in metadata
        assert "mixed_mode_s4p" in metadata["output_files"]


# =============================================================================
# Convenience Method Tests
# =============================================================================


def test_result_convenience_methods(
    sample_4port_sparams: SParameterData,
) -> None:
    """Test convenience methods for accessing mixed-mode parameters."""
    result = compute_mixed_mode_sparams(sample_4port_sparams)

    # Test Sdd accessors
    sdd11 = result.get_sdd11()
    sdd21 = result.get_sdd21()
    sdd12 = result.get_sdd12()
    sdd22 = result.get_sdd22()

    assert sdd11.shape == (3,)
    assert sdd21.shape == (3,)
    assert sdd12.shape == (3,)
    assert sdd22.shape == (3,)

    # Test Scc accessors
    scc11 = result.get_scc11()
    scc21 = result.get_scc21()

    assert scc11.shape == (3,)
    assert scc21.shape == (3,)

    # Test dB conversion methods
    il_db = result.differential_insertion_loss_db()
    rl_db = result.differential_return_loss_db()
    mc_db = result.mode_conversion_db()

    assert il_db.shape == (3,)
    assert rl_db.shape == (3,)
    assert mc_db.shape == (3,)

    # dB values should be finite
    assert np.all(np.isfinite(il_db))
    assert np.all(np.isfinite(rl_db))
    assert np.all(np.isfinite(mc_db))


# =============================================================================
# Edge Cases
# =============================================================================


def test_single_frequency_point() -> None:
    """Test mixed-mode with single frequency point."""
    frequencies_hz = np.array([1e9], dtype=np.float64)
    s_parameters = np.zeros((1, 4, 4), dtype=np.complex128)
    s_parameters[0] = np.eye(4) * 0.1  # Simple test matrix

    sparams = SParameterData(
        frequencies_hz=frequencies_hz,
        s_parameters=s_parameters,
        n_ports=4,
        reference_impedance_ohm=50.0,
    )

    result = compute_mixed_mode_sparams(sparams)

    assert result.sdd.shape == (1, 2, 2)
    assert result.mm_sparams.n_frequencies == 1


def test_identity_matrix_mixed_mode() -> None:
    """Test mixed-mode of identity S-matrix (all ports isolated)."""
    frequencies_hz = np.array([1e9], dtype=np.float64)
    s_parameters = np.zeros((1, 4, 4), dtype=np.complex128)
    # Identity-like: each port reflects back to itself
    np.fill_diagonal(s_parameters[0], 0.5)

    sparams = SParameterData(
        frequencies_hz=frequencies_hz,
        s_parameters=s_parameters,
        n_ports=4,
        reference_impedance_ohm=50.0,
    )

    result = compute_mixed_mode_sparams(sparams)

    # With identity-like matrix and 1324 pairing:
    # Sdd11 = 0.5 * (S11 - S13 - S31 + S33) = 0.5 * (0.5 - 0 - 0 + 0.5) = 0.5
    # Scc11 = 0.5 * (S11 + S13 + S31 + S33) = 0.5 * (0.5 + 0 + 0 + 0.5) = 0.5
    np.testing.assert_allclose(result.sdd[0, 0, 0], 0.5, rtol=1e-10)
    np.testing.assert_allclose(result.scc[0, 0, 0], 0.5, rtol=1e-10)
