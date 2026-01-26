"""Additional unit tests for Touchstone parsing edge cases.

This module supplements test_touchstone_export.py with coverage for:
- SParameterData validation and edge cases
- TouchstoneOptions configuration validation
- Complex format conversions (DB, MA, RI)
- String/file round-trip consistency
- Multi-port parsing edge cases

Tests ensure robustness of the em.touchstone module.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from formula_foundry.em.touchstone import (
    FrequencyUnit,
    NetworkType,
    SParameterData,
    SParameterFormat,
    TouchstoneOptions,
    create_empty_sparam_data,
    create_thru_sparam_data,
    merge_sparam_data,
    read_touchstone,
    read_touchstone_from_string,
    write_touchstone,
    write_touchstone_to_string,
)

# =============================================================================
# SParameterData Validation Tests
# =============================================================================


class TestSParameterDataValidation:
    """Tests for SParameterData dataclass validation."""

    def test_frequencies_must_be_1d(self) -> None:
        """Frequencies array must be 1-dimensional."""
        freqs = np.array([[1e9, 2e9], [3e9, 4e9]])  # 2D array
        s_params = np.zeros((2, 2, 2, 2), dtype=np.complex128)

        with pytest.raises(ValueError, match="frequencies_hz must be 1D"):
            SParameterData(
                frequencies_hz=freqs,  # Pass 2D array directly
                s_parameters=s_params[0],  # Valid 3D shape
                n_ports=2,
            )

    def test_s_parameters_must_be_3d(self) -> None:
        """S-parameters array must be 3-dimensional."""
        freqs = np.array([1e9, 2e9])
        s_params = np.zeros((2, 2), dtype=np.complex128)  # 2D array

        with pytest.raises(ValueError, match="s_parameters must be 3D"):
            SParameterData(
                frequencies_hz=freqs,
                s_parameters=s_params,
                n_ports=2,
            )

    def test_shape_mismatch_rejected(self) -> None:
        """S-parameters shape must match (n_freq, n_ports, n_ports)."""
        freqs = np.array([1e9, 2e9])  # 2 frequencies
        s_params = np.zeros((3, 2, 2), dtype=np.complex128)  # 3 frequencies

        with pytest.raises(ValueError, match="does not match expected"):
            SParameterData(
                frequencies_hz=freqs,
                s_parameters=s_params,
                n_ports=2,
            )

    def test_negative_zref_rejected(self) -> None:
        """Negative reference impedance is rejected."""
        freqs = np.array([1e9, 2e9])
        s_params = np.zeros((2, 2, 2), dtype=np.complex128)

        with pytest.raises(ValueError, match="reference_impedance_ohm must be positive"):
            SParameterData(
                frequencies_hz=freqs,
                s_parameters=s_params,
                n_ports=2,
                reference_impedance_ohm=-50.0,
            )

    def test_zero_zref_rejected(self) -> None:
        """Zero reference impedance is rejected."""
        freqs = np.array([1e9, 2e9])
        s_params = np.zeros((2, 2, 2), dtype=np.complex128)

        with pytest.raises(ValueError, match="reference_impedance_ohm must be positive"):
            SParameterData(
                frequencies_hz=freqs,
                s_parameters=s_params,
                n_ports=2,
                reference_impedance_ohm=0.0,
            )

    def test_unsorted_frequencies_rejected(self) -> None:
        """Frequencies must be strictly increasing."""
        freqs = np.array([2e9, 1e9, 3e9])  # Not sorted
        s_params = np.zeros((3, 2, 2), dtype=np.complex128)

        with pytest.raises(ValueError, match="strictly increasing"):
            SParameterData(
                frequencies_hz=freqs,
                s_parameters=s_params,
                n_ports=2,
            )

    def test_duplicate_frequencies_rejected(self) -> None:
        """Duplicate frequencies are rejected."""
        freqs = np.array([1e9, 2e9, 2e9, 3e9])  # Duplicate
        s_params = np.zeros((4, 2, 2), dtype=np.complex128)

        with pytest.raises(ValueError, match="strictly increasing"):
            SParameterData(
                frequencies_hz=freqs,
                s_parameters=s_params,
                n_ports=2,
            )


class TestSParameterDataProperties:
    """Tests for SParameterData properties and methods."""

    def _make_test_data(self) -> SParameterData:
        """Create test data."""
        freqs = np.array([1e9, 2e9, 5e9, 10e9])
        s_params = np.zeros((4, 2, 2), dtype=np.complex128)
        s_params[:, 0, 0] = [0.1 + 0.05j, 0.12 + 0.06j, 0.15 + 0.08j, 0.2 + 0.1j]
        s_params[:, 1, 0] = [0.9 - 0.1j, 0.85 - 0.12j, 0.8 - 0.15j, 0.7 - 0.2j]
        s_params[:, 0, 1] = s_params[:, 1, 0]  # Reciprocal
        s_params[:, 1, 1] = s_params[:, 0, 0]  # Symmetric

        return SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=2,
            reference_impedance_ohm=50.0,
            comment="Test data",
        )

    def test_n_frequencies_property(self) -> None:
        """n_frequencies returns correct count."""
        data = self._make_test_data()
        assert data.n_frequencies == 4

    def test_f_min_hz_property(self) -> None:
        """f_min_hz returns minimum frequency."""
        data = self._make_test_data()
        assert data.f_min_hz == 1e9

    def test_f_max_hz_property(self) -> None:
        """f_max_hz returns maximum frequency."""
        data = self._make_test_data()
        assert data.f_max_hz == 10e9

    def test_get_s_with_1based_indices(self) -> None:
        """get_s uses 1-based port indices."""
        data = self._make_test_data()

        s11 = data.get_s(1, 1)
        s21 = data.get_s(2, 1)

        np.testing.assert_array_equal(s11, data.s_parameters[:, 0, 0])
        np.testing.assert_array_equal(s21, data.s_parameters[:, 1, 0])

    def test_get_s_invalid_port_raises(self) -> None:
        """get_s raises for invalid port indices."""
        data = self._make_test_data()

        with pytest.raises(ValueError, match="Port indices must be"):
            data.get_s(0, 1)  # 0-based index

        with pytest.raises(ValueError, match="Port indices must be"):
            data.get_s(3, 1)  # Out of range

    def test_s11_s21_s12_s22_methods(self) -> None:
        """S-parameter accessor methods work correctly."""
        data = self._make_test_data()

        np.testing.assert_array_equal(data.s11(), data.get_s(1, 1))
        np.testing.assert_array_equal(data.s21(), data.get_s(2, 1))
        np.testing.assert_array_equal(data.s12(), data.get_s(1, 2))
        np.testing.assert_array_equal(data.s22(), data.get_s(2, 2))

    def test_s21_requires_2port(self) -> None:
        """S21 method requires at least 2 ports."""
        freqs = np.array([1e9])
        s_params = np.zeros((1, 1, 1), dtype=np.complex128)
        data = SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=1,
        )

        with pytest.raises(ValueError, match="requires at least 2 ports"):
            data.s21()

    def test_magnitude_db(self) -> None:
        """magnitude_db computes correct dB values."""
        data = self._make_test_data()
        mag_db = data.magnitude_db(1, 1)

        expected = 20.0 * np.log10(np.abs(data.s11()))
        np.testing.assert_array_almost_equal(mag_db, expected)

    def test_phase_deg(self) -> None:
        """phase_deg computes correct degree values."""
        data = self._make_test_data()
        phase = data.phase_deg(1, 1)

        expected = np.degrees(np.angle(data.s11()))
        np.testing.assert_array_almost_equal(phase, expected)

    def test_interpolate_to_new_frequencies(self) -> None:
        """interpolate creates new data at target frequencies."""
        data = self._make_test_data()
        new_freqs = np.array([1.5e9, 3e9, 7e9])

        interpolated = data.interpolate(new_freqs)

        assert interpolated.n_frequencies == 3
        assert interpolated.n_ports == 2
        np.testing.assert_array_equal(interpolated.frequencies_hz, new_freqs)


# =============================================================================
# TouchstoneOptions Tests
# =============================================================================


class TestTouchstoneOptions:
    """Tests for TouchstoneOptions configuration."""

    def test_default_options(self) -> None:
        """Default options have correct values."""
        opts = TouchstoneOptions()

        assert opts.frequency_unit == FrequencyUnit.GHZ
        assert opts.parameter_format == SParameterFormat.RI
        assert opts.network_type == NetworkType.S
        assert opts.reference_impedance_ohm == 50.0
        assert opts.version == "1.0"

    def test_invalid_zref_rejected(self) -> None:
        """Non-positive reference impedance is rejected."""
        with pytest.raises(ValueError, match="reference_impedance_ohm must be positive"):
            TouchstoneOptions(reference_impedance_ohm=0.0)

        with pytest.raises(ValueError, match="reference_impedance_ohm must be positive"):
            TouchstoneOptions(reference_impedance_ohm=-50.0)

    def test_invalid_version_rejected(self) -> None:
        """Invalid version string is rejected."""
        with pytest.raises(ValueError, match="version must be"):
            TouchstoneOptions(version="3.0")


class TestFrequencyUnitMultiplier:
    """Tests for FrequencyUnit multiplier property."""

    def test_hz_multiplier(self) -> None:
        """Hz multiplier is 1.0."""
        assert FrequencyUnit.HZ.multiplier == 1.0

    def test_khz_multiplier(self) -> None:
        """kHz multiplier is 1e3."""
        assert FrequencyUnit.KHZ.multiplier == 1e3

    def test_mhz_multiplier(self) -> None:
        """MHz multiplier is 1e6."""
        assert FrequencyUnit.MHZ.multiplier == 1e6

    def test_ghz_multiplier(self) -> None:
        """GHz multiplier is 1e9."""
        assert FrequencyUnit.GHZ.multiplier == 1e9


# =============================================================================
# Format Conversion Tests
# =============================================================================


class TestFormatRoundTrip:
    """Tests for format conversion round-trips."""

    def _make_test_data(self) -> SParameterData:
        """Create test data with known values."""
        freqs = np.array([1e9, 5e9, 10e9])
        s_params = np.zeros((3, 2, 2), dtype=np.complex128)
        s_params[:, 0, 0] = [0.1 + 0.2j, 0.15 + 0.25j, 0.2 + 0.3j]
        s_params[:, 1, 0] = [0.8 - 0.1j, 0.75 - 0.15j, 0.7 - 0.2j]
        s_params[:, 0, 1] = s_params[:, 1, 0]
        s_params[:, 1, 1] = s_params[:, 0, 0]

        return SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=2,
        )

    def test_ri_format_round_trip(self, tmp_path: Path) -> None:
        """RI format round-trip preserves data."""
        data = self._make_test_data()
        path = tmp_path / "test.s2p"

        opts = TouchstoneOptions(
            frequency_unit=FrequencyUnit.HZ,
            parameter_format=SParameterFormat.RI,
        )
        write_touchstone(data, path, options=opts)
        recovered = read_touchstone(path)

        np.testing.assert_allclose(recovered.frequencies_hz, data.frequencies_hz, rtol=1e-6)
        np.testing.assert_allclose(recovered.s_parameters, data.s_parameters, rtol=1e-6)

    def test_ma_format_round_trip(self, tmp_path: Path) -> None:
        """MA format round-trip preserves data."""
        data = self._make_test_data()
        path = tmp_path / "test.s2p"

        opts = TouchstoneOptions(
            frequency_unit=FrequencyUnit.GHZ,
            parameter_format=SParameterFormat.MA,
        )
        write_touchstone(data, path, options=opts)
        recovered = read_touchstone(path)

        np.testing.assert_allclose(recovered.frequencies_hz, data.frequencies_hz, rtol=1e-6)
        np.testing.assert_allclose(recovered.s_parameters, data.s_parameters, rtol=1e-5)

    def test_db_format_round_trip(self, tmp_path: Path) -> None:
        """DB format round-trip preserves data."""
        data = self._make_test_data()
        path = tmp_path / "test.s2p"

        opts = TouchstoneOptions(
            frequency_unit=FrequencyUnit.MHZ,
            parameter_format=SParameterFormat.DB,
        )
        write_touchstone(data, path, options=opts)
        recovered = read_touchstone(path)

        np.testing.assert_allclose(recovered.frequencies_hz, data.frequencies_hz, rtol=1e-6)
        np.testing.assert_allclose(recovered.s_parameters, data.s_parameters, rtol=1e-4)

    def test_string_round_trip(self) -> None:
        """String write/read round-trip preserves data."""
        data = self._make_test_data()

        content = write_touchstone_to_string(data)
        recovered = read_touchstone_from_string(content, n_ports=2)

        np.testing.assert_allclose(recovered.s_parameters, data.s_parameters, rtol=1e-6)


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_empty_sparam_data(self) -> None:
        """create_empty_sparam_data creates zero-initialized data."""
        freqs = np.array([1e9, 5e9, 10e9])
        data = create_empty_sparam_data(freqs, n_ports=3, reference_impedance_ohm=75.0)

        assert data.n_frequencies == 3
        assert data.n_ports == 3
        assert data.reference_impedance_ohm == 75.0
        assert np.all(data.s_parameters == 0)

    def test_create_thru_sparam_data_lossless(self) -> None:
        """create_thru_sparam_data creates ideal thru with zero loss."""
        freqs = np.array([1e9, 5e9, 10e9])
        data = create_thru_sparam_data(freqs, insertion_loss_db=0.0)

        assert data.n_ports == 2
        # S21 = S12 = 1.0 for lossless thru
        np.testing.assert_allclose(np.abs(data.s21()), 1.0, rtol=1e-10)
        np.testing.assert_allclose(np.abs(data.s12()), 1.0, rtol=1e-10)
        # S11 = S22 = 0 for perfect match
        np.testing.assert_allclose(np.abs(data.s11()), 0.0, rtol=1e-10)

    def test_create_thru_sparam_data_with_loss(self) -> None:
        """create_thru_sparam_data creates thru with specified loss."""
        freqs = np.array([1e9, 5e9, 10e9])
        data = create_thru_sparam_data(freqs, insertion_loss_db=-3.0)

        expected_mag = 10 ** (-3.0 / 20.0)
        np.testing.assert_allclose(np.abs(data.s21()), expected_mag, rtol=1e-10)

    def test_merge_sparam_data_empty_list(self) -> None:
        """merge_sparam_data returns empty list for empty input."""
        result = merge_sparam_data([])
        assert result == []

    def test_merge_sparam_data_single_item(self) -> None:
        """merge_sparam_data with single item returns interpolated copy."""
        freqs = np.array([1e9, 5e9, 10e9])
        data = create_thru_sparam_data(freqs)

        result = merge_sparam_data([data])

        assert len(result) == 1
        np.testing.assert_array_equal(result[0].frequencies_hz, data.frequencies_hz)

    def test_merge_sparam_data_to_common_frequencies(self) -> None:
        """merge_sparam_data interpolates to common frequency grid."""
        freqs1 = np.array([1e9, 5e9, 10e9])
        freqs2 = np.array([2e9, 6e9, 8e9, 12e9])
        common_freqs = np.array([1e9, 2e9, 3e9, 4e9, 5e9])

        data1 = create_thru_sparam_data(freqs1)
        data2 = create_thru_sparam_data(freqs2)

        result = merge_sparam_data([data1, data2], common_frequencies_hz=common_freqs)

        assert len(result) == 2
        for d in result:
            np.testing.assert_array_equal(d.frequencies_hz, common_freqs)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_frequency_point(self, tmp_path: Path) -> None:
        """Single frequency point is handled correctly."""
        freqs = np.array([1e9])
        s_params = np.zeros((1, 2, 2), dtype=np.complex128)
        s_params[0, 0, 0] = 0.1
        s_params[0, 1, 0] = 0.9
        s_params[0, 0, 1] = 0.9
        s_params[0, 1, 1] = 0.1

        data = SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=2,
        )

        path = tmp_path / "single.s2p"
        write_touchstone(data, path)
        recovered = read_touchstone(path)

        assert recovered.n_frequencies == 1
        np.testing.assert_allclose(recovered.s_parameters, data.s_parameters, rtol=1e-6)

    def test_single_port_network(self, tmp_path: Path) -> None:
        """Single-port network is handled correctly."""
        freqs = np.array([1e9, 5e9, 10e9])
        s_params = np.zeros((3, 1, 1), dtype=np.complex128)
        s_params[:, 0, 0] = [0.1 + 0.05j, 0.12 + 0.06j, 0.15 + 0.08j]

        data = SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=1,
        )

        path = tmp_path / "oneport.s1p"
        write_touchstone(data, path)
        recovered = read_touchstone(path)

        assert recovered.n_ports == 1
        np.testing.assert_allclose(recovered.s_parameters, data.s_parameters, rtol=1e-6)

    def test_four_port_network(self, tmp_path: Path) -> None:
        """Four-port network is handled correctly."""
        freqs = np.array([1e9, 5e9])
        s_params = np.zeros((2, 4, 4), dtype=np.complex128)
        # Set some values
        for i in range(4):
            for j in range(4):
                s_params[:, i, j] = 0.1 * (i + 1) + 0.05j * (j + 1)

        data = SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=4,
        )

        path = tmp_path / "fourport.s4p"
        write_touchstone(data, path)
        recovered = read_touchstone(path)

        assert recovered.n_ports == 4
        np.testing.assert_allclose(recovered.s_parameters, data.s_parameters, rtol=1e-6)

    def test_file_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_touchstone(tmp_path / "nonexistent.s2p")

    def test_invalid_extension(self, tmp_path: Path) -> None:
        """Invalid extension raises ValueError."""
        path = tmp_path / "test.txt"
        path.write_text("dummy content")

        with pytest.raises(ValueError, match="Invalid Touchstone extension"):
            read_touchstone(path)
