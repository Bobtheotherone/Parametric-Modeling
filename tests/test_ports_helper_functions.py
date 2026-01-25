"""Unit tests for helper functions in openems/ports.py.

This module tests the internal helper functions used by the port building system:
- _extract_connector_position: Extract connector positions from parameters
- _fallback_port_positions: Calculate port positions from transmission line lengths
- calculate_cpwg_impedance: Calculate CPWG characteristic impedance

REQ-M2-005: Port placement logic with impedance validation.
"""

from __future__ import annotations

import pytest

from formula_foundry.openems.ports import (
    _extract_connector_position,
    _fallback_port_positions,
    calculate_cpwg_impedance,
)


class TestExtractConnectorPosition:
    """Tests for _extract_connector_position helper function."""

    def test_extract_left_connector(self) -> None:
        """Extract left connector position from parameters."""
        params = {
            "connectors.left.position_nm[0]": 5_000_000,
            "connectors.left.position_nm[1]": 1_000_000,
            "connectors.right.position_nm[0]": 75_000_000,
            "connectors.right.position_nm[1]": 1_000_000,
        }

        result = _extract_connector_position(params, "left")

        assert result is not None
        assert result == (5_000_000, 1_000_000)

    def test_extract_right_connector(self) -> None:
        """Extract right connector position from parameters."""
        params = {
            "connectors.left.position_nm[0]": 5_000_000,
            "connectors.left.position_nm[1]": 1_000_000,
            "connectors.right.position_nm[0]": 75_000_000,
            "connectors.right.position_nm[1]": 2_000_000,
        }

        result = _extract_connector_position(params, "right")

        assert result is not None
        assert result == (75_000_000, 2_000_000)

    def test_missing_x_returns_none(self) -> None:
        """Return None when X coordinate is missing."""
        params = {
            "connectors.left.position_nm[1]": 1_000_000,
        }

        result = _extract_connector_position(params, "left")

        assert result is None

    def test_missing_y_returns_none(self) -> None:
        """Return None when Y coordinate is missing."""
        params = {
            "connectors.left.position_nm[0]": 5_000_000,
        }

        result = _extract_connector_position(params, "left")

        assert result is None

    def test_empty_params_returns_none(self) -> None:
        """Return None when params is empty."""
        params: dict[str, int] = {}

        result = _extract_connector_position(params, "left")

        assert result is None

    def test_wrong_side_returns_none(self) -> None:
        """Return None when side doesn't match available data."""
        params = {
            "connectors.right.position_nm[0]": 75_000_000,
            "connectors.right.position_nm[1]": 1_000_000,
        }

        result = _extract_connector_position(params, "left")

        assert result is None

    def test_zero_position_valid(self) -> None:
        """Zero coordinates should be valid positions."""
        params = {
            "connectors.left.position_nm[0]": 0,
            "connectors.left.position_nm[1]": 0,
        }

        result = _extract_connector_position(params, "left")

        assert result is not None
        assert result == (0, 0)

    def test_negative_position_valid(self) -> None:
        """Negative coordinates should be valid positions."""
        params = {
            "connectors.left.position_nm[0]": -5_000_000,
            "connectors.left.position_nm[1]": -1_000_000,
        }

        result = _extract_connector_position(params, "left")

        assert result is not None
        assert result == (-5_000_000, -1_000_000)


class TestFallbackPortPositions:
    """Tests for _fallback_port_positions helper function."""

    def test_symmetric_transmission_line(self) -> None:
        """Calculate positions for symmetric transmission line."""
        params = {
            "transmission_line.length_left_nm": 25_000_000,
            "transmission_line.length_right_nm": 25_000_000,
        }

        left, right = _fallback_port_positions(params)

        assert left == (-25_000_000, 0)
        assert right == (25_000_000, 0)

    def test_asymmetric_transmission_line(self) -> None:
        """Calculate positions for asymmetric transmission line."""
        params = {
            "transmission_line.length_left_nm": 30_000_000,
            "transmission_line.length_right_nm": 20_000_000,
        }

        left, right = _fallback_port_positions(params)

        assert left == (-30_000_000, 0)
        assert right == (20_000_000, 0)

    def test_zero_left_length(self) -> None:
        """Handle zero left length."""
        params = {
            "transmission_line.length_left_nm": 0,
            "transmission_line.length_right_nm": 25_000_000,
        }

        left, right = _fallback_port_positions(params)

        assert left == (0, 0)
        assert right == (25_000_000, 0)

    def test_zero_right_length(self) -> None:
        """Handle zero right length."""
        params = {
            "transmission_line.length_left_nm": 25_000_000,
            "transmission_line.length_right_nm": 0,
        }

        left, right = _fallback_port_positions(params)

        assert left == (-25_000_000, 0)
        assert right == (0, 0)

    def test_missing_left_length_raises(self) -> None:
        """Raise KeyError when left length is missing."""
        params = {
            "transmission_line.length_right_nm": 25_000_000,
        }

        with pytest.raises(KeyError, match="Transmission line lengths required"):
            _fallback_port_positions(params)

    def test_missing_right_length_raises(self) -> None:
        """Raise KeyError when right length is missing."""
        params = {
            "transmission_line.length_left_nm": 25_000_000,
        }

        with pytest.raises(KeyError, match="Transmission line lengths required"):
            _fallback_port_positions(params)

    def test_empty_params_raises(self) -> None:
        """Raise KeyError when params is empty."""
        params: dict[str, int] = {}

        with pytest.raises(KeyError, match="Transmission line lengths required"):
            _fallback_port_positions(params)

    def test_y_coordinate_always_zero(self) -> None:
        """Y coordinate should always be 0 for fallback positions."""
        params = {
            "transmission_line.length_left_nm": 10_000_000,
            "transmission_line.length_right_nm": 15_000_000,
        }

        left, right = _fallback_port_positions(params)

        assert left[1] == 0
        assert right[1] == 0


class TestCalculateCpwgImpedanceExtended:
    """Extended tests for calculate_cpwg_impedance function.

    These tests complement the existing tests in test_m2_port_config.py
    with additional edge cases and physical validation.
    """

    def test_zero_width_returns_fallback(self) -> None:
        """Zero width should return 50 Ohm fallback."""
        z0 = calculate_cpwg_impedance(w_nm=0, gap_nm=180_000, er=4.2)
        assert z0 == 50.0

    def test_zero_gap_returns_fallback(self) -> None:
        """Zero gap should return 50 Ohm fallback."""
        z0 = calculate_cpwg_impedance(w_nm=300_000, gap_nm=0, er=4.2)
        assert z0 == 50.0

    def test_negative_width_returns_fallback(self) -> None:
        """Negative width should return 50 Ohm fallback."""
        z0 = calculate_cpwg_impedance(w_nm=-300_000, gap_nm=180_000, er=4.2)
        assert z0 == 50.0

    def test_negative_gap_returns_fallback(self) -> None:
        """Negative gap should return 50 Ohm fallback."""
        z0 = calculate_cpwg_impedance(w_nm=300_000, gap_nm=-180_000, er=4.2)
        assert z0 == 50.0

    def test_typical_cpwg_impedance_range(self) -> None:
        """Typical CPWG should have impedance in reasonable range."""
        z0 = calculate_cpwg_impedance(w_nm=300_000, gap_nm=180_000, er=4.2)
        # Should be in 30-80 Ohm range for typical CPWG
        assert 30.0 < z0 < 80.0

    def test_narrow_trace_higher_impedance(self) -> None:
        """Narrower trace should have higher impedance."""
        z_narrow = calculate_cpwg_impedance(w_nm=200_000, gap_nm=180_000, er=4.2)
        z_wide = calculate_cpwg_impedance(w_nm=400_000, gap_nm=180_000, er=4.2)
        assert z_narrow > z_wide

    def test_larger_gap_higher_impedance(self) -> None:
        """Larger gap should have higher impedance."""
        z_small_gap = calculate_cpwg_impedance(w_nm=300_000, gap_nm=100_000, er=4.2)
        z_large_gap = calculate_cpwg_impedance(w_nm=300_000, gap_nm=250_000, er=4.2)
        assert z_large_gap > z_small_gap

    def test_lower_er_higher_impedance(self) -> None:
        """Lower dielectric constant should give higher impedance."""
        z_high_er = calculate_cpwg_impedance(w_nm=300_000, gap_nm=180_000, er=4.5)
        z_low_er = calculate_cpwg_impedance(w_nm=300_000, gap_nm=180_000, er=3.5)
        assert z_low_er > z_high_er

    def test_very_wide_trace_low_impedance(self) -> None:
        """Very wide trace should have low impedance."""
        z0 = calculate_cpwg_impedance(w_nm=1_000_000, gap_nm=100_000, er=4.2)
        # Very wide trace should be below 40 Ohms
        assert z0 < 40.0

    def test_very_narrow_trace_high_impedance(self) -> None:
        """Very narrow trace should have high impedance."""
        z0 = calculate_cpwg_impedance(w_nm=100_000, gap_nm=300_000, er=4.2)
        # Very narrow trace with large gap should be above 60 Ohms
        assert z0 > 60.0

    def test_air_substrate_gives_higher_impedance(self) -> None:
        """Air substrate (er=1) should give highest impedance."""
        z_air = calculate_cpwg_impedance(w_nm=300_000, gap_nm=180_000, er=1.0)
        z_fr4 = calculate_cpwg_impedance(w_nm=300_000, gap_nm=180_000, er=4.2)
        assert z_air > z_fr4

    def test_optional_h_nm_parameter(self) -> None:
        """Test that h_nm parameter is accepted (for future use)."""
        # Currently h_nm doesn't affect the simplified calculation,
        # but it should be accepted without error
        z0 = calculate_cpwg_impedance(
            w_nm=300_000,
            gap_nm=180_000,
            er=4.2,
            h_nm=200_000,
        )
        assert 30.0 < z0 < 80.0

    def test_impedance_monotonicity_with_width(self) -> None:
        """Impedance should monotonically decrease with increasing width."""
        widths = [100_000, 200_000, 300_000, 400_000, 500_000]
        impedances = [calculate_cpwg_impedance(w_nm=w, gap_nm=180_000, er=4.2) for w in widths]

        # Each impedance should be less than the previous
        for i in range(1, len(impedances)):
            assert impedances[i] < impedances[i - 1], (
                f"Impedance should decrease with width: "
                f"Z({widths[i - 1]}) = {impedances[i - 1]}, "
                f"Z({widths[i]}) = {impedances[i]}"
            )

    def test_impedance_monotonicity_with_gap(self) -> None:
        """Impedance should monotonically increase with increasing gap."""
        gaps = [50_000, 100_000, 150_000, 200_000, 250_000]
        impedances = [calculate_cpwg_impedance(w_nm=300_000, gap_nm=g, er=4.2) for g in gaps]

        # Each impedance should be greater than the previous
        for i in range(1, len(impedances)):
            assert impedances[i] > impedances[i - 1], (
                f"Impedance should increase with gap: Z({gaps[i - 1]}) = {impedances[i - 1]}, Z({gaps[i]}) = {impedances[i]}"
            )

    def test_50_ohm_design_approximate(self) -> None:
        """Test typical 50 Ohm CPWG dimensions."""
        # Typical 50 Ohm CPWG on FR4 with ~300um trace
        # The simplified formula may not exactly hit 50, but should be close
        z0 = calculate_cpwg_impedance(w_nm=300_000, gap_nm=200_000, er=4.2)
        # Should be within 30% of 50 Ohm for this simplified model
        assert 35.0 < z0 < 65.0


class TestPortHelperIntegration:
    """Integration tests for port helper functions working together."""

    def test_connector_extraction_preferred_over_fallback(self) -> None:
        """Connector positions should be used when available."""
        params = {
            "connectors.left.position_nm[0]": 5_000_000,
            "connectors.left.position_nm[1]": 0,
            "connectors.right.position_nm[0]": 75_000_000,
            "connectors.right.position_nm[1]": 0,
            "transmission_line.length_left_nm": 25_000_000,
            "transmission_line.length_right_nm": 25_000_000,
        }

        # Connector extraction should work
        left_conn = _extract_connector_position(params, "left")
        right_conn = _extract_connector_position(params, "right")

        assert left_conn == (5_000_000, 0)
        assert right_conn == (75_000_000, 0)

        # Fallback would give different values
        left_fb, right_fb = _fallback_port_positions(params)
        assert left_fb == (-25_000_000, 0)  # Different from connector extraction

    def test_fallback_used_when_connector_missing(self) -> None:
        """Fallback should be used when connectors are not specified."""
        params = {
            "transmission_line.length_left_nm": 25_000_000,
            "transmission_line.length_right_nm": 25_000_000,
        }

        # Connector extraction returns None
        assert _extract_connector_position(params, "left") is None
        assert _extract_connector_position(params, "right") is None

        # Fallback works
        left, right = _fallback_port_positions(params)
        assert left == (-25_000_000, 0)
        assert right == (25_000_000, 0)
