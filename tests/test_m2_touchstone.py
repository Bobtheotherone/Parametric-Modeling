"""Tests for formula_foundry.em.touchstone module.

Tests cover:
- SParameterData validation and operations
- Touchstone file reading (various formats)
- Touchstone file writing (round-trip verification)
- FrequencyUnit, SParameterFormat, TouchstoneOptions
- Utility functions (create_empty_sparam_data, create_thru_sparam_data)
- Interpolation and merge operations
- scikit-rf integration (optional, skipped if not installed)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from formula_foundry.em.touchstone import (
    FrequencyUnit,
    NetworkType,
    SParameterData,
    SParameterFormat,
    TouchstoneOptions,
    _complex_to_format,
    _convert_to_complex,
    _parse_option_line,
    create_empty_sparam_data,
    create_thru_sparam_data,
    merge_sparam_data,
    read_touchstone,
    read_touchstone_from_string,
    write_touchstone,
    write_touchstone_to_string,
)


class TestFrequencyUnit:
    """Tests for FrequencyUnit enum."""

    def test_multiplier_hz(self) -> None:
        """Hz multiplier is 1.0."""
        assert FrequencyUnit.HZ.multiplier == 1.0

    def test_multiplier_khz(self) -> None:
        """kHz multiplier is 1e3."""
        assert FrequencyUnit.KHZ.multiplier == 1e3

    def test_multiplier_mhz(self) -> None:
        """MHz multiplier is 1e6."""
        assert FrequencyUnit.MHZ.multiplier == 1e6

    def test_multiplier_ghz(self) -> None:
        """GHz multiplier is 1e9."""
        assert FrequencyUnit.GHZ.multiplier == 1e9


class TestSParameterFormat:
    """Tests for SParameterFormat enum."""

    def test_all_formats_exist(self) -> None:
        """All expected formats are defined."""
        assert SParameterFormat.DB.value == "DB"
        assert SParameterFormat.MA.value == "MA"
        assert SParameterFormat.RI.value == "RI"


class TestNetworkType:
    """Tests for NetworkType enum."""

    def test_all_types_exist(self) -> None:
        """All expected network types are defined."""
        assert NetworkType.S.value == "S"
        assert NetworkType.Y.value == "Y"
        assert NetworkType.Z.value == "Z"
        assert NetworkType.H.value == "H"
        assert NetworkType.G.value == "G"


class TestTouchstoneOptions:
    """Tests for TouchstoneOptions dataclass."""

    def test_default_options(self) -> None:
        """Default options are correct."""
        opts = TouchstoneOptions()
        assert opts.frequency_unit == FrequencyUnit.GHZ
        assert opts.parameter_format == SParameterFormat.RI
        assert opts.network_type == NetworkType.S
        assert opts.reference_impedance_ohm == 50.0
        assert opts.version == "1.0"

    def test_custom_options(self) -> None:
        """Custom options are accepted."""
        opts = TouchstoneOptions(
            frequency_unit=FrequencyUnit.MHZ,
            parameter_format=SParameterFormat.DB,
            network_type=NetworkType.Y,
            reference_impedance_ohm=75.0,
            version="2.0",
        )
        assert opts.frequency_unit == FrequencyUnit.MHZ
        assert opts.parameter_format == SParameterFormat.DB
        assert opts.reference_impedance_ohm == 75.0

    def test_invalid_impedance_rejected(self) -> None:
        """Non-positive impedance is rejected."""
        with pytest.raises(ValueError, match="reference_impedance_ohm must be positive"):
            TouchstoneOptions(reference_impedance_ohm=0.0)
        with pytest.raises(ValueError, match="reference_impedance_ohm must be positive"):
            TouchstoneOptions(reference_impedance_ohm=-50.0)

    def test_invalid_version_rejected(self) -> None:
        """Invalid version is rejected."""
        with pytest.raises(ValueError, match="version must be"):
            TouchstoneOptions(version="3.0")


class TestSParameterData:
    """Tests for SParameterData dataclass."""

    def test_valid_2port_data(self) -> None:
        """Valid 2-port data is accepted."""
        freqs = np.array([1e9, 2e9, 3e9])
        s_params = np.zeros((3, 2, 2), dtype=np.complex128)
        s_params[:, 0, 0] = 0.1  # S11
        s_params[:, 1, 0] = 0.9  # S21
        s_params[:, 0, 1] = 0.9  # S12
        s_params[:, 1, 1] = 0.1  # S22

        data = SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=2,
        )

        assert data.n_frequencies == 3
        assert data.n_ports == 2
        assert data.f_min_hz == 1e9
        assert data.f_max_hz == 3e9

    def test_invalid_frequency_dimension(self) -> None:
        """Non-1D frequency array is rejected."""
        freqs = np.array([[1e9, 2e9], [3e9, 4e9]])
        s_params = np.zeros((2, 2, 2), dtype=np.complex128)

        with pytest.raises(ValueError, match="frequencies_hz must be 1D"):
            SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

    def test_invalid_sparam_dimension(self) -> None:
        """Non-3D S-parameter array is rejected."""
        freqs = np.array([1e9, 2e9])
        s_params = np.zeros((2, 4), dtype=np.complex128)  # 2D instead of 3D

        with pytest.raises(ValueError, match="s_parameters must be 3D"):
            SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

    def test_shape_mismatch_rejected(self) -> None:
        """Mismatched shapes are rejected."""
        freqs = np.array([1e9, 2e9, 3e9])  # 3 frequencies
        s_params = np.zeros((4, 2, 2), dtype=np.complex128)  # 4 frequencies

        with pytest.raises(ValueError, match="does not match expected"):
            SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

    def test_unsorted_frequencies_rejected(self) -> None:
        """Non-increasing frequencies are rejected."""
        freqs = np.array([1e9, 3e9, 2e9])  # Not sorted
        s_params = np.zeros((3, 2, 2), dtype=np.complex128)

        with pytest.raises(ValueError, match="strictly increasing"):
            SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

    def test_get_s_method(self) -> None:
        """get_s returns correct S-parameter slice."""
        freqs = np.array([1e9, 2e9])
        s_params = np.zeros((2, 2, 2), dtype=np.complex128)
        s_params[:, 0, 0] = [0.1, 0.2]  # S11
        s_params[:, 1, 0] = [0.8, 0.7]  # S21

        data = SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

        # Port indices are 1-based
        np.testing.assert_array_equal(data.get_s(1, 1), [0.1, 0.2])
        np.testing.assert_array_equal(data.get_s(2, 1), [0.8, 0.7])

    def test_get_s_invalid_port(self) -> None:
        """Invalid port index raises ValueError."""
        freqs = np.array([1e9])
        s_params = np.zeros((1, 2, 2), dtype=np.complex128)
        data = SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

        with pytest.raises(ValueError, match="Port indices must be 1 to 2"):
            data.get_s(0, 1)
        with pytest.raises(ValueError, match="Port indices must be 1 to 2"):
            data.get_s(3, 1)

    def test_s11_s21_s12_s22_shortcuts(self) -> None:
        """Shortcut methods return correct parameters."""
        freqs = np.array([1e9])
        s_params = np.array([[[0.1 + 0.1j, 0.2 + 0.2j], [0.8 + 0.1j, 0.15 + 0.05j]]])
        data = SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

        np.testing.assert_array_equal(data.s11(), [0.1 + 0.1j])
        np.testing.assert_array_equal(data.s21(), [0.8 + 0.1j])
        np.testing.assert_array_equal(data.s12(), [0.2 + 0.2j])
        np.testing.assert_array_equal(data.s22(), [0.15 + 0.05j])

    def test_s21_requires_2_ports(self) -> None:
        """S21 raises error for 1-port data."""
        freqs = np.array([1e9])
        s_params = np.zeros((1, 1, 1), dtype=np.complex128)
        data = SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=1)

        with pytest.raises(ValueError, match="requires at least 2 ports"):
            data.s21()

    def test_magnitude_db(self) -> None:
        """magnitude_db returns correct dB values."""
        freqs = np.array([1e9])
        s_params = np.array([[[0.1, 0], [1.0, 0]]])  # |S11|=0.1, |S21|=1.0
        data = SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

        s11_db = data.magnitude_db(1, 1)
        s21_db = data.magnitude_db(2, 1)

        np.testing.assert_allclose(s11_db, [-20.0], rtol=1e-6)
        np.testing.assert_allclose(s21_db, [0.0], rtol=1e-6)

    def test_phase_deg(self) -> None:
        """phase_deg returns correct phase values."""
        freqs = np.array([1e9])
        # S11 = 1∠45°, S21 = 1∠-90°
        s_params = np.array([[[np.exp(1j * np.pi / 4), 0], [np.exp(-1j * np.pi / 2), 0]]])
        data = SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

        s11_phase = data.phase_deg(1, 1)
        s21_phase = data.phase_deg(2, 1)

        np.testing.assert_allclose(s11_phase, [45.0], rtol=1e-6)
        np.testing.assert_allclose(s21_phase, [-90.0], rtol=1e-6)

    def test_interpolate(self) -> None:
        """interpolate correctly interpolates to new frequencies."""
        freqs = np.array([1e9, 3e9])
        s_params = np.zeros((2, 2, 2), dtype=np.complex128)
        s_params[0, 0, 0] = 0.1 + 0.1j  # S11 at 1 GHz
        s_params[1, 0, 0] = 0.3 + 0.3j  # S11 at 3 GHz
        data = SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

        # Interpolate to 2 GHz (midpoint)
        new_freqs = np.array([2e9])
        interp_data = data.interpolate(new_freqs)

        # Linear interpolation: S11 at 2 GHz should be 0.2 + 0.2j
        np.testing.assert_allclose(interp_data.s11(), [0.2 + 0.2j], rtol=1e-10)


class TestConversionFunctions:
    """Tests for format conversion functions."""

    def test_ri_to_complex(self) -> None:
        """RI format converts correctly."""
        result = _convert_to_complex(3.0, 4.0, SParameterFormat.RI)
        assert result == 3.0 + 4.0j

    def test_ma_to_complex(self) -> None:
        """MA format converts correctly."""
        # 1∠90°
        result = _convert_to_complex(1.0, 90.0, SParameterFormat.MA)
        np.testing.assert_allclose(result, 1j, atol=1e-10)

    def test_db_to_complex(self) -> None:
        """DB format converts correctly."""
        # 0 dB, 0° = 1+0j
        result = _convert_to_complex(0.0, 0.0, SParameterFormat.DB)
        np.testing.assert_allclose(result, 1.0 + 0j, atol=1e-10)

        # -20 dB = 0.1 magnitude
        result = _convert_to_complex(-20.0, 0.0, SParameterFormat.DB)
        np.testing.assert_allclose(result, 0.1 + 0j, atol=1e-10)

    def test_complex_to_ri(self) -> None:
        """Complex to RI format is correct."""
        val1, val2 = _complex_to_format(3.0 + 4.0j, SParameterFormat.RI)
        assert val1 == 3.0
        assert val2 == 4.0

    def test_complex_to_ma(self) -> None:
        """Complex to MA format is correct."""
        val1, val2 = _complex_to_format(1j, SParameterFormat.MA)
        np.testing.assert_allclose(val1, 1.0, atol=1e-10)
        np.testing.assert_allclose(val2, 90.0, atol=1e-10)

    def test_complex_to_db(self) -> None:
        """Complex to DB format is correct."""
        val1, val2 = _complex_to_format(0.1 + 0j, SParameterFormat.DB)
        np.testing.assert_allclose(val1, -20.0, atol=1e-6)
        np.testing.assert_allclose(val2, 0.0, atol=1e-10)


class TestParseOptionLine:
    """Tests for option line parsing."""

    def test_default_options(self) -> None:
        """Empty option line uses defaults."""
        opts = _parse_option_line("")
        assert opts.frequency_unit == FrequencyUnit.GHZ
        assert opts.parameter_format == SParameterFormat.MA
        assert opts.network_type == NetworkType.S
        assert opts.reference_impedance_ohm == 50.0

    def test_full_option_line(self) -> None:
        """Full option line is parsed correctly."""
        opts = _parse_option_line("MHz S DB R 75")
        assert opts.frequency_unit == FrequencyUnit.MHZ
        assert opts.parameter_format == SParameterFormat.DB
        assert opts.network_type == NetworkType.S
        assert opts.reference_impedance_ohm == 75.0

    def test_case_insensitive(self) -> None:
        """Option parsing is case insensitive."""
        opts = _parse_option_line("ghz s ri r 50")
        assert opts.frequency_unit == FrequencyUnit.GHZ
        assert opts.parameter_format == SParameterFormat.RI


class TestReadTouchstone:
    """Tests for reading Touchstone files."""

    def test_read_simple_s2p(self) -> None:
        """Read a simple S2P file."""
        content = """! Simple 2-port test
# GHz S RI R 50
1.0 0.1 0.0 0.9 0.0 0.9 0.0 0.1 0.0
2.0 0.2 0.1 0.8 0.0 0.8 0.0 0.2 0.1
"""
        data = read_touchstone_from_string(content, n_ports=2)

        assert data.n_ports == 2
        assert data.n_frequencies == 2
        np.testing.assert_allclose(data.frequencies_hz, [1e9, 2e9])
        np.testing.assert_allclose(data.s11()[0], 0.1 + 0j)
        np.testing.assert_allclose(data.s21()[0], 0.9 + 0j)

    def test_read_ma_format(self) -> None:
        """Read file in MA format."""
        content = """# GHz S MA R 50
1.0 1.0 90.0 0.5 0.0 0.5 0.0 1.0 -90.0
"""
        data = read_touchstone_from_string(content, n_ports=2)

        # S11 = 1∠90° = j
        np.testing.assert_allclose(data.s11()[0], 1j, atol=1e-10)
        # S22 = 1∠-90° = -j
        np.testing.assert_allclose(data.s22()[0], -1j, atol=1e-10)

    def test_read_db_format(self) -> None:
        """Read file in dB format."""
        content = """# GHz S DB R 50
1.0 -20.0 0.0 0.0 0.0 0.0 0.0 -20.0 0.0
"""
        data = read_touchstone_from_string(content, n_ports=2)

        # -20 dB = 0.1 magnitude
        np.testing.assert_allclose(abs(data.s11()[0]), 0.1, rtol=1e-6)
        np.testing.assert_allclose(abs(data.s21()[0]), 1.0, rtol=1e-6)

    def test_read_with_comments(self) -> None:
        """Comments are captured."""
        content = """! This is a test file
! More comments here
# GHz S RI R 50
1.0 0.1 0.0 0.9 0.0 0.9 0.0 0.1 0.0
"""
        data = read_touchstone_from_string(content, n_ports=2)
        assert "test file" in data.comment
        assert "More comments" in data.comment

    def test_read_inline_comments(self) -> None:
        """Inline comments are handled."""
        content = """# GHz S RI R 50
1.0 0.1 0.0 0.9 0.0 0.9 0.0 0.1 0.0 ! This is inline
"""
        data = read_touchstone_from_string(content, n_ports=2)
        assert data.n_frequencies == 1

    def test_read_1port(self) -> None:
        """Read 1-port S-parameter file."""
        content = """# GHz S RI R 50
1.0 0.5 0.5
2.0 0.4 0.4
"""
        data = read_touchstone_from_string(content, n_ports=1)

        assert data.n_ports == 1
        assert data.n_frequencies == 2
        np.testing.assert_allclose(data.s11()[0], 0.5 + 0.5j)

    def test_read_khz_frequency(self) -> None:
        """Read file with kHz frequency unit."""
        content = """# kHz S RI R 50
1000.0 0.1 0.0 0.9 0.0 0.9 0.0 0.1 0.0
"""
        data = read_touchstone_from_string(content, n_ports=2)
        np.testing.assert_allclose(data.frequencies_hz, [1e6])  # 1000 kHz = 1 MHz

    def test_read_file_not_found(self) -> None:
        """FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_touchstone("/nonexistent/path/file.s2p")

    def test_read_invalid_extension(self) -> None:
        """Invalid extension raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid Touchstone extension"):
                read_touchstone(temp_path)
        finally:
            Path(temp_path).unlink()


class TestWriteTouchstone:
    """Tests for writing Touchstone files."""

    def test_write_simple_s2p(self) -> None:
        """Write a simple S2P file."""
        freqs = np.array([1e9, 2e9])
        s_params = np.zeros((2, 2, 2), dtype=np.complex128)
        s_params[:, 0, 0] = [0.1, 0.2]
        s_params[:, 1, 0] = [0.9, 0.8]
        s_params[:, 0, 1] = [0.9, 0.8]
        s_params[:, 1, 1] = [0.1, 0.2]

        data = SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)
        content = write_touchstone_to_string(data)

        assert "# GHz S RI R 50.0" in content
        assert "2-port" in content

    def test_write_with_comment(self) -> None:
        """Comments are written."""
        freqs = np.array([1e9])
        s_params = np.zeros((1, 2, 2), dtype=np.complex128)
        data = SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=2,
            comment="Test comment line",
        )

        content = write_touchstone_to_string(data)
        assert "! Test comment line" in content

    def test_write_db_format(self) -> None:
        """Write in dB format."""
        freqs = np.array([1e9])
        s_params = np.array([[[0.1, 0], [1.0, 0]]], dtype=np.complex128)
        data = SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

        opts = TouchstoneOptions(parameter_format=SParameterFormat.DB)
        content = write_touchstone_to_string(data, options=opts)

        assert "DB" in content

    def test_round_trip_ri(self) -> None:
        """Round-trip test in RI format."""
        freqs = np.array([1e9, 2e9, 3e9])
        s_params = np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)
        s_params *= 0.1  # Keep magnitudes small

        original = SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

        # Write and read back
        content = write_touchstone_to_string(original)
        recovered = read_touchstone_from_string(content, n_ports=2)

        np.testing.assert_allclose(recovered.frequencies_hz, original.frequencies_hz, rtol=1e-6)
        np.testing.assert_allclose(recovered.s_parameters, original.s_parameters, rtol=1e-6)

    def test_round_trip_ma(self) -> None:
        """Round-trip test in MA format."""
        freqs = np.array([1e9, 5e9])
        s_params = np.array(
            [
                [[0.1 + 0.1j, 0.9], [0.9, 0.1 + 0.1j]],
                [[0.2 + 0.2j, 0.8], [0.8, 0.2 + 0.2j]],
            ]
        )

        original = SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

        opts = TouchstoneOptions(parameter_format=SParameterFormat.MA)
        content = write_touchstone_to_string(original, options=opts)
        recovered = read_touchstone_from_string(content, n_ports=2)

        np.testing.assert_allclose(recovered.s_parameters, original.s_parameters, rtol=1e-5)

    def test_round_trip_file(self) -> None:
        """Round-trip test writing to actual file."""
        freqs = np.array([1e9, 2e9])
        s_params = np.zeros((2, 2, 2), dtype=np.complex128)
        s_params[:, 0, 0] = [0.1 + 0.05j, 0.15 + 0.1j]
        s_params[:, 1, 0] = [0.9 - 0.1j, 0.85 - 0.15j]
        s_params[:, 0, 1] = [0.9 - 0.1j, 0.85 - 0.15j]
        s_params[:, 1, 1] = [0.1 + 0.05j, 0.15 + 0.1j]

        original = SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.s2p"
            write_touchstone(original, path)
            recovered = read_touchstone(path)

        np.testing.assert_allclose(recovered.s_parameters, original.s_parameters, rtol=1e-6)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_empty_sparam_data(self) -> None:
        """create_empty_sparam_data creates zeroed S-parameters."""
        freqs = np.array([1e9, 2e9, 3e9])
        data = create_empty_sparam_data(freqs, n_ports=2)

        assert data.n_ports == 2
        assert data.n_frequencies == 3
        np.testing.assert_array_equal(data.s_parameters, 0)

    def test_create_thru_sparam_data(self) -> None:
        """create_thru_sparam_data creates ideal thru."""
        freqs = np.array([1e9, 2e9])
        data = create_thru_sparam_data(freqs)

        # S11 = S22 = 0 (perfect match)
        np.testing.assert_allclose(data.s11(), [0, 0])
        np.testing.assert_allclose(data.s22(), [0, 0])

        # S21 = S12 = 1 (perfect transmission)
        np.testing.assert_allclose(data.s21(), [1, 1])
        np.testing.assert_allclose(data.s12(), [1, 1])

    def test_create_thru_with_loss(self) -> None:
        """create_thru_sparam_data with insertion loss."""
        freqs = np.array([1e9])
        data = create_thru_sparam_data(freqs, insertion_loss_db=-3.0)

        # -3 dB insertion loss
        expected_mag = 10 ** (-3.0 / 20.0)
        np.testing.assert_allclose(abs(data.s21()), [expected_mag], rtol=1e-6)

    def test_merge_sparam_data(self) -> None:
        """merge_sparam_data interpolates to common grid."""
        freqs1 = np.array([1e9, 3e9])
        freqs2 = np.array([1e9, 2e9, 3e9])

        s1 = np.zeros((2, 2, 2), dtype=np.complex128)
        s1[0, 0, 0] = 0.1
        s1[1, 0, 0] = 0.3

        s2 = np.zeros((3, 2, 2), dtype=np.complex128)
        s2[0, 0, 0] = 0.1
        s2[1, 0, 0] = 0.2
        s2[2, 0, 0] = 0.3

        data1 = SParameterData(frequencies_hz=freqs1, s_parameters=s1, n_ports=2)
        data2 = SParameterData(frequencies_hz=freqs2, s_parameters=s2, n_ports=2)

        # Merge to data2's grid
        merged = merge_sparam_data([data1, data2], common_frequencies_hz=freqs2)

        assert len(merged) == 2
        assert merged[0].n_frequencies == 3
        assert merged[1].n_frequencies == 3

        # data1 interpolated: S11 at 2 GHz should be 0.2 (linear interp)
        np.testing.assert_allclose(merged[0].s11()[1], 0.2, rtol=1e-10)

    def test_merge_empty_list(self) -> None:
        """merge_sparam_data with empty list returns empty."""
        result = merge_sparam_data([])
        assert result == []


class TestScikitRFIntegration:
    """Tests for scikit-rf integration (optional)."""

    @pytest.fixture
    def has_skrf(self) -> bool:
        """Check if scikit-rf is available."""
        try:
            import skrf  # noqa: F401

            return True
        except ImportError:
            return False

    def test_to_skrf_network_import_error(self, has_skrf: bool) -> None:
        """to_skrf_network raises ImportError if skrf not available."""
        if has_skrf:
            pytest.skip("scikit-rf is installed")

        from formula_foundry.em.touchstone import to_skrf_network

        freqs = np.array([1e9])
        s_params = np.zeros((1, 2, 2), dtype=np.complex128)
        data = SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

        with pytest.raises(ImportError, match="scikit-rf is required"):
            to_skrf_network(data)

    def test_from_skrf_network_import_error(self, has_skrf: bool) -> None:
        """from_skrf_network raises ImportError if skrf not available."""
        if has_skrf:
            pytest.skip("scikit-rf is installed")

        from formula_foundry.em.touchstone import from_skrf_network

        with pytest.raises(ImportError):
            from_skrf_network(None)  # type: ignore[arg-type]

    def test_validate_with_skrf_import_error(self, has_skrf: bool) -> None:
        """validate_with_skrf raises ImportError if skrf not available."""
        if has_skrf:
            pytest.skip("scikit-rf is installed")

        from formula_foundry.em.touchstone import validate_with_skrf

        freqs = np.array([1e9])
        s_params = np.zeros((1, 2, 2), dtype=np.complex128)
        data = SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

        with pytest.raises(ImportError, match="scikit-rf is required"):
            validate_with_skrf(data)

    @pytest.mark.skipif(
        not pytest.importorskip("skrf", reason="scikit-rf not installed"),
        reason="scikit-rf required",
    )
    def test_to_from_skrf_roundtrip(self) -> None:
        """Round-trip conversion through scikit-rf."""
        from formula_foundry.em.touchstone import from_skrf_network, to_skrf_network

        freqs = np.array([1e9, 2e9, 3e9])
        s_params = np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)
        s_params *= 0.1

        original = SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

        network = to_skrf_network(original)
        recovered = from_skrf_network(network)

        np.testing.assert_allclose(recovered.frequencies_hz, original.frequencies_hz, rtol=1e-6)
        np.testing.assert_allclose(recovered.s_parameters, original.s_parameters, rtol=1e-6)

    @pytest.mark.skipif(
        not pytest.importorskip("skrf", reason="scikit-rf not installed"),
        reason="scikit-rf required",
    )
    def test_validate_with_skrf_passes(self) -> None:
        """validate_with_skrf returns True for valid data."""
        from formula_foundry.em.touchstone import validate_with_skrf

        freqs = np.array([1e9, 2e9])
        s_params = np.zeros((2, 2, 2), dtype=np.complex128)
        s_params[:, 0, 0] = 0.1
        s_params[:, 1, 0] = 0.9
        s_params[:, 0, 1] = 0.9
        s_params[:, 1, 1] = 0.1

        data = SParameterData(frequencies_hz=freqs, s_parameters=s_params, n_ports=2)

        assert validate_with_skrf(data) is True


class TestImmutability:
    """Tests for dataclass immutability."""

    def test_touchstone_options_frozen(self) -> None:
        """TouchstoneOptions is immutable."""
        opts = TouchstoneOptions()
        with pytest.raises(AttributeError):
            opts.reference_impedance_ohm = 75.0  # type: ignore[misc]
