"""Tests for canonical Touchstone export (REQ-M2-016).

Tests verify:
- Exports use RI format
- Frequency is in Hz
- Zref is recorded in metadata
- Output format matches specification
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from formula_foundry.postprocess.touchstone import (
    TouchstoneExportMetadata,
    export_touchstone_canonical,
    format_touchstone_canonical,
)

from formula_foundry.em.touchstone import (
    SParameterData,
    read_touchstone,
    read_touchstone_from_string,
)


class TestTouchstoneExportMetadata:
    """Tests for TouchstoneExportMetadata dataclass."""

    def test_default_metadata(self) -> None:
        """REQ-M2-016: Default metadata uses correct format."""
        meta = TouchstoneExportMetadata()

        assert meta.frequency_unit == "Hz"
        assert meta.parameter_format == "RI"
        assert meta.reference_impedance_ohm == 50.0
        assert meta.port_zref_ohms is None

    def test_custom_zref(self) -> None:
        """REQ-M2-016: Custom reference impedance is accepted."""
        meta = TouchstoneExportMetadata(reference_impedance_ohm=75.0)

        assert meta.reference_impedance_ohm == 75.0

    def test_per_port_zref(self) -> None:
        """REQ-M2-016: Per-port Zref is recorded."""
        meta = TouchstoneExportMetadata(
            reference_impedance_ohm=50.0,
            port_zref_ohms=(50.0, 75.0),
        )

        assert meta.port_zref_ohms == (50.0, 75.0)

    def test_invalid_zref_rejected(self) -> None:
        """REQ-M2-016: Non-positive Zref is rejected."""
        with pytest.raises(ValueError, match="reference_impedance_ohm must be positive"):
            TouchstoneExportMetadata(reference_impedance_ohm=0.0)

        with pytest.raises(ValueError, match="reference_impedance_ohm must be positive"):
            TouchstoneExportMetadata(reference_impedance_ohm=-50.0)

    def test_invalid_port_zref_rejected(self) -> None:
        """REQ-M2-016: Non-positive port Zref is rejected."""
        with pytest.raises(ValueError, match="All port impedances must be positive"):
            TouchstoneExportMetadata(port_zref_ohms=(50.0, -75.0))

    def test_format_is_immutable(self) -> None:
        """REQ-M2-016: Format fields cannot be changed."""
        meta = TouchstoneExportMetadata()

        # frequency_unit and parameter_format are init=False, so they're fixed
        assert meta.frequency_unit == "Hz"
        assert meta.parameter_format == "RI"

        # Frozen dataclass prevents modification
        with pytest.raises(AttributeError):
            meta.reference_impedance_ohm = 75.0  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """REQ-M2-016: Metadata converts to dict for manifest."""
        meta = TouchstoneExportMetadata(
            reference_impedance_ohm=75.0,
            port_zref_ohms=(75.0, 50.0),
            comment="Test export",
            provenance={"git_commit": "abc123"},
        )

        result = meta.to_dict()

        assert result["frequency_unit"] == "Hz"
        assert result["parameter_format"] == "RI"
        assert result["reference_impedance_ohm"] == 75.0
        assert result["port_zref_ohms"] == [75.0, 50.0]
        assert result["comment"] == "Test export"
        assert result["provenance"]["git_commit"] == "abc123"


class TestFormatTouchstoneCanonical:
    """Tests for format_touchstone_canonical function."""

    def _make_test_sparam_data(self) -> SParameterData:
        """Create test S-parameter data."""
        freqs = np.array([1e9, 2e9, 3e9])
        s_params = np.zeros((3, 2, 2), dtype=np.complex128)
        s_params[:, 0, 0] = 0.1 + 0.05j  # S11
        s_params[:, 1, 0] = 0.9 - 0.1j  # S21
        s_params[:, 0, 1] = 0.9 - 0.1j  # S12
        s_params[:, 1, 1] = 0.1 + 0.05j  # S22

        return SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=2,
        )

    def test_touchstone_exports_hz_ri_and_records_zref(self) -> None:
        """REQ-M2-016: Touchstone exports Hz, RI, and records Zref.

        This is the main test for REQ-M2-016 from the test matrix.
        """
        data = self._make_test_sparam_data()
        content = format_touchstone_canonical(data)

        # Verify Hz frequency unit in option line
        assert "# Hz S RI R" in content

        # Verify RI format in option line
        assert "RI" in content

        # Verify reference impedance is recorded
        assert "Reference impedance:" in content
        assert "50" in content

        # Verify frequency values are in Hz (not GHz or MHz)
        # 1e9, 2e9, 3e9 should appear as Hz values
        lines = content.split("\n")
        data_lines = [l for l in lines if l and not l.startswith("!") and not l.startswith("#")]
        assert len(data_lines) == 3  # 3 frequency points

        # First frequency should be 1e9 Hz
        first_freq = float(data_lines[0].split()[0])
        assert first_freq == pytest.approx(1e9, rel=1e-6)

    def test_output_uses_ri_format(self) -> None:
        """REQ-M2-016: Output uses real/imaginary format."""
        freqs = np.array([1e9])
        s_params = np.zeros((1, 2, 2), dtype=np.complex128)
        s_params[0, 0, 0] = 0.3 + 0.4j  # S11 = 0.3 + 0.4j

        data = SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=2,
        )

        content = format_touchstone_canonical(data)

        # Parse the data line
        lines = content.split("\n")
        data_line = [l for l in lines if l and not l.startswith("!") and not l.startswith("#")][0]
        tokens = data_line.split()

        # S11 real and imag should be after frequency
        s11_real = float(tokens[1])
        s11_imag = float(tokens[2])

        assert s11_real == pytest.approx(0.3, rel=1e-6)
        assert s11_imag == pytest.approx(0.4, rel=1e-6)

    def test_frequency_in_hz(self) -> None:
        """REQ-M2-016: Frequency values are in Hz."""
        freqs = np.array([1.5e9, 2.5e9])  # 1.5 GHz, 2.5 GHz
        s_params = np.zeros((2, 2, 2), dtype=np.complex128)
        s_params[:, 0, 0] = 0.1

        data = SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=2,
        )

        content = format_touchstone_canonical(data)

        lines = content.split("\n")
        data_lines = [l for l in lines if l and not l.startswith("!") and not l.startswith("#")]

        # Verify frequencies are written in Hz
        freq1 = float(data_lines[0].split()[0])
        freq2 = float(data_lines[1].split()[0])

        assert freq1 == pytest.approx(1.5e9, rel=1e-6)
        assert freq2 == pytest.approx(2.5e9, rel=1e-6)

    def test_zref_recorded_in_metadata(self) -> None:
        """REQ-M2-016: Reference impedance is recorded in metadata."""
        data = self._make_test_sparam_data()
        meta = TouchstoneExportMetadata(
            reference_impedance_ohm=75.0,
        )

        content = format_touchstone_canonical(data, metadata=meta)

        assert "Reference impedance: 75" in content
        assert "# Hz S RI R 75.0" in content

    def test_per_port_zref_recorded(self) -> None:
        """REQ-M2-016: Per-port Zref is recorded when specified."""
        data = self._make_test_sparam_data()
        meta = TouchstoneExportMetadata(
            reference_impedance_ohm=50.0,
            port_zref_ohms=(50.0, 75.0),
        )

        content = format_touchstone_canonical(data, metadata=meta)

        assert "Per-port Zref" in content
        assert "50.0" in content
        assert "75.0" in content

    def test_provenance_recorded(self) -> None:
        """REQ-M2-016: Provenance information is recorded."""
        data = self._make_test_sparam_data()
        meta = TouchstoneExportMetadata(
            provenance={
                "git_commit": "abc123",
                "toolchain_digest": "sha256:xyz",
            }
        )

        content = format_touchstone_canonical(data, metadata=meta)

        assert "Provenance:" in content
        assert "git_commit: abc123" in content
        assert "toolchain_digest: sha256:xyz" in content


class TestExportTouchstoneCanonical:
    """Tests for export_touchstone_canonical function."""

    def _make_test_sparam_data(self) -> SParameterData:
        """Create test S-parameter data."""
        freqs = np.array([1e9, 2e9])
        s_params = np.zeros((2, 2, 2), dtype=np.complex128)
        s_params[:, 0, 0] = [0.1 + 0.05j, 0.15 + 0.1j]
        s_params[:, 1, 0] = [0.9 - 0.1j, 0.85 - 0.15j]
        s_params[:, 0, 1] = [0.9 - 0.1j, 0.85 - 0.15j]
        s_params[:, 1, 1] = [0.1 + 0.05j, 0.15 + 0.1j]

        return SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=2,
        )

    def test_export_creates_file(self) -> None:
        """REQ-M2-016: Export creates valid Touchstone file."""
        data = self._make_test_sparam_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.s2p"
            meta = export_touchstone_canonical(data, path)

            assert path.exists()
            assert isinstance(meta, TouchstoneExportMetadata)

    def test_export_returns_metadata(self) -> None:
        """REQ-M2-016: Export returns metadata for manifest."""
        data = self._make_test_sparam_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.s2p"
            meta = export_touchstone_canonical(
                data,
                path,
                metadata=TouchstoneExportMetadata(
                    reference_impedance_ohm=75.0,
                    comment="Test export",
                ),
            )

            assert meta.reference_impedance_ohm == 75.0
            assert meta.comment == "Test export"
            assert meta.frequency_unit == "Hz"
            assert meta.parameter_format == "RI"

    def test_round_trip_preserves_data(self) -> None:
        """REQ-M2-016: Round-trip export/import preserves data."""
        data = self._make_test_sparam_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.s2p"
            export_touchstone_canonical(data, path)

            # Read back using standard reader
            recovered = read_touchstone(path)

            np.testing.assert_allclose(
                recovered.frequencies_hz,
                data.frequencies_hz,
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                recovered.s_parameters,
                data.s_parameters,
                rtol=1e-6,
            )

    def test_canonical_format_readable_by_standard_reader(self) -> None:
        """REQ-M2-016: Canonical format is standard-compliant."""
        data = self._make_test_sparam_data()
        content = format_touchstone_canonical(data)

        # Standard reader should parse it correctly
        recovered = read_touchstone_from_string(content, n_ports=2)

        assert recovered.n_ports == 2
        assert recovered.n_frequencies == 2
        np.testing.assert_allclose(
            recovered.s_parameters,
            data.s_parameters,
            rtol=1e-6,
        )


class TestSinglePortExport:
    """Tests for single-port Touchstone export."""

    def test_single_port_export(self) -> None:
        """REQ-M2-016: Single-port export works correctly."""
        freqs = np.array([1e9, 2e9])
        s_params = np.zeros((2, 1, 1), dtype=np.complex128)
        s_params[:, 0, 0] = [0.1 + 0.2j, 0.15 + 0.25j]

        data = SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=1,
        )

        content = format_touchstone_canonical(data)

        assert "1-port" in content
        assert "# Hz S RI R 50.0" in content

        # Parse data lines
        lines = content.split("\n")
        data_lines = [l for l in lines if l and not l.startswith("!") and not l.startswith("#")]
        assert len(data_lines) == 2


class TestMultiPortExport:
    """Tests for multi-port (>2) Touchstone export."""

    def test_four_port_export(self) -> None:
        """REQ-M2-016: 4-port export works correctly."""
        freqs = np.array([1e9])
        s_params = np.zeros((1, 4, 4), dtype=np.complex128)
        # Set some S-parameters
        for i in range(4):
            for j in range(4):
                s_params[0, i, j] = 0.1 * (i + 1) + 0.05j * (j + 1)

        data = SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=4,
        )

        content = format_touchstone_canonical(data)

        assert "4-port" in content
        assert "# Hz S RI R 50.0" in content
