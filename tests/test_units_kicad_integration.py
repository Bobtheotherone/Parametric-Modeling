# SPDX-License-Identifier: MIT
"""Additional unit tests for units module focusing on KiCad bounds integration.

This module provides extended coverage for the units module, specifically:
- KiCad 32-bit bounds integration with parsing functions
- Conversion precision and determinism guarantees
- Edge cases around supported unit combinations

Satisfies:
    - REQ-M1-002: Geometry must be represented as integer nanometers
    - REQ-M1-003: KiCad export compatibility (32-bit signed integer range)
"""

from __future__ import annotations

from decimal import Decimal

import pytest


class TestLengthNMConversionPrecision:
    """Tests for LengthNM conversion precision guarantees."""

    def test_sub_nanometer_precision_rejected(self) -> None:
        """Sub-nanometer values must be rejected to preserve integer semantics."""
        from formula_foundry.coupongen.units import parse_length_nm

        # 0.5 nm is not representable as integer
        with pytest.raises(ValueError, match="integer"):
            parse_length_nm("0.5nm")

        # 0.1 nm is not representable
        with pytest.raises(ValueError, match="integer"):
            parse_length_nm("0.1nm")

    def test_um_to_nm_exact_conversion(self) -> None:
        """Micrometer conversions must be exact to the nanometer."""
        from formula_foundry.coupongen.units import parse_length_nm

        # 1 um = 1000 nm exactly
        assert parse_length_nm("1um") == 1_000
        assert parse_length_nm("0.001um") == 1  # 1 nm

        # Fractional um that yields integer nm
        assert parse_length_nm("0.5um") == 500  # 500 nm
        assert parse_length_nm("1.5um") == 1_500  # 1500 nm

    def test_mm_to_nm_exact_conversion(self) -> None:
        """Millimeter conversions must be exact to the nanometer."""
        from formula_foundry.coupongen.units import parse_length_nm

        # 1 mm = 1,000,000 nm exactly
        assert parse_length_nm("1mm") == 1_000_000
        assert parse_length_nm("0.000001mm") == 1  # 1 nm

        # Common PCB dimensions
        assert parse_length_nm("0.15mm") == 150_000  # 150 um trace
        assert parse_length_nm("0.254mm") == 254_000  # 10 mil

    def test_mil_to_nm_exact_conversion(self) -> None:
        """Mil conversions must be exact (1 mil = 25,400 nm)."""
        from formula_foundry.coupongen.units import parse_length_nm

        # 1 mil = 25.4 um = 25,400 nm
        assert parse_length_nm("1mil") == 25_400
        assert parse_length_nm("10mil") == 254_000
        assert parse_length_nm("100mil") == 2_540_000

        # Fractional mil that yields integer nm
        assert parse_length_nm("0.1mil") == 2_540  # 2540 nm

    def test_mil_fractional_nm_rejected(self) -> None:
        """Mil values yielding fractional nm must be rejected."""
        from formula_foundry.coupongen.units import parse_length_nm

        # 0.01 mil = 254 nm (ok)
        assert parse_length_nm("0.01mil") == 254

        # 0.001 mil = 25.4 nm (fractional, should fail)
        with pytest.raises(ValueError, match="integer"):
            parse_length_nm("0.001mil")


class TestKicadBoundsIntegration:
    """Tests for KiCad 32-bit bounds integration with length parsing."""

    def test_typical_pcb_dimensions_within_kicad_bounds(self) -> None:
        """Common PCB dimensions should be within KiCad bounds."""
        from formula_foundry.coupongen.units import (
            is_within_kicad_bounds,
            parse_length_nm,
        )

        # 100mm x 100mm board
        board_dim = parse_length_nm("100mm")
        assert is_within_kicad_bounds(board_dim)

        # 300mm (12 inch) board
        large_board = parse_length_nm("300mm")
        assert is_within_kicad_bounds(large_board)

        # 1 meter board (still within bounds)
        meter_board = parse_length_nm("1000mm")
        assert is_within_kicad_bounds(meter_board)

    def test_near_kicad_bounds(self) -> None:
        """Values near KiCad's 32-bit bounds should be handled correctly."""
        from formula_foundry.coupongen.units import (
            KICAD_MAX_NM,
            KICAD_MIN_NM,
            check_kicad_bounds,
            is_within_kicad_bounds,
        )

        # Values just inside bounds
        just_inside_max = KICAD_MAX_NM - 1
        just_inside_min = KICAD_MIN_NM + 1

        assert is_within_kicad_bounds(just_inside_max)
        assert is_within_kicad_bounds(just_inside_min)
        check_kicad_bounds(just_inside_max)  # Should not raise
        check_kicad_bounds(just_inside_min)  # Should not raise

    def test_clamp_preserves_sign(self) -> None:
        """Clamping should preserve the sign of the original value."""
        from formula_foundry.coupongen.units import (
            KICAD_MAX_NM,
            KICAD_MIN_NM,
            clamp_to_kicad_bounds,
        )

        # Large positive -> clamped to max (positive)
        large_positive = 10 * KICAD_MAX_NM
        clamped_pos = clamp_to_kicad_bounds(large_positive)
        assert clamped_pos == KICAD_MAX_NM
        assert clamped_pos > 0

        # Large negative -> clamped to min (negative)
        large_negative = 10 * KICAD_MIN_NM
        clamped_neg = clamp_to_kicad_bounds(large_negative)
        assert clamped_neg == KICAD_MIN_NM
        assert clamped_neg < 0

    def test_kicad_bounds_approximately_2_meters(self) -> None:
        """Verify KiCad bounds represent approximately 2.1 meters."""
        from formula_foundry.coupongen.units import KICAD_MAX_NM, KICAD_MIN_NM

        # Convert to mm for human verification
        max_mm = KICAD_MAX_NM / 1_000_000
        min_mm = KICAD_MIN_NM / 1_000_000

        # Should be approximately +/- 2147 mm (2.147 meters)
        assert 2000 < max_mm < 2200
        assert -2200 < min_mm < -2000


class TestAngleParsing:
    """Tests for AngleMdeg parsing edge cases."""

    def test_full_rotation(self) -> None:
        """360 degrees should parse correctly."""
        from formula_foundry.coupongen.units import parse_angle_mdeg

        assert parse_angle_mdeg("360deg") == 360_000
        assert parse_angle_mdeg("-360deg") == -360_000

    def test_common_rotation_angles(self) -> None:
        """Common rotation angles used in PCB design."""
        from formula_foundry.coupongen.units import parse_angle_mdeg

        # 90-degree rotations
        assert parse_angle_mdeg("90deg") == 90_000
        assert parse_angle_mdeg("-90deg") == -90_000

        # 45-degree rotations
        assert parse_angle_mdeg("45deg") == 45_000
        assert parse_angle_mdeg("-45deg") == -45_000

        # 180-degree rotation
        assert parse_angle_mdeg("180deg") == 180_000

    def test_fractional_degree_precision(self) -> None:
        """Fractional degrees that yield integer millidegrees."""
        from formula_foundry.coupongen.units import parse_angle_mdeg

        # 0.5 deg = 500 mdeg
        assert parse_angle_mdeg("0.5deg") == 500

        # 0.001 deg = 1 mdeg
        assert parse_angle_mdeg("0.001deg") == 1

        # 0.1 deg = 100 mdeg
        assert parse_angle_mdeg("0.1deg") == 100

    def test_sub_millidegree_rejected(self) -> None:
        """Sub-millidegree values must be rejected."""
        from formula_foundry.coupongen.units import parse_angle_mdeg

        # 0.0001 deg = 0.1 mdeg (fractional, should fail)
        with pytest.raises(ValueError, match="integer"):
            parse_angle_mdeg("0.0001deg")


class TestFrequencyParsing:
    """Tests for FrequencyHz parsing edge cases."""

    def test_common_rf_frequencies(self) -> None:
        """Common RF frequencies used in PCB/EM simulations."""
        from formula_foundry.coupongen.units import parse_frequency_hz

        # WiFi 2.4 GHz
        assert parse_frequency_hz("2.4GHz") == 2_400_000_000

        # WiFi 5 GHz
        assert parse_frequency_hz("5GHz") == 5_000_000_000

        # 5G mmWave (28 GHz)
        assert parse_frequency_hz("28GHz") == 28_000_000_000

    def test_sub_ghz_frequencies(self) -> None:
        """Sub-GHz frequencies commonly used in IoT."""
        from formula_foundry.coupongen.units import parse_frequency_hz

        # 433 MHz
        assert parse_frequency_hz("433MHz") == 433_000_000

        # 868 MHz
        assert parse_frequency_hz("868MHz") == 868_000_000

        # 915 MHz
        assert parse_frequency_hz("915MHz") == 915_000_000

    def test_khz_range_frequencies(self) -> None:
        """kHz range frequencies."""
        from formula_foundry.coupongen.units import parse_frequency_hz

        assert parse_frequency_hz("100kHz") == 100_000
        assert parse_frequency_hz("1.5kHz") == 1_500
        assert parse_frequency_hz("0.5kHz") == 500

    def test_fractional_mhz(self) -> None:
        """Fractional MHz values that yield integer Hz."""
        from formula_foundry.coupongen.units import parse_frequency_hz

        # 2.437 GHz (WiFi channel 6 center)
        assert parse_frequency_hz("2.437GHz") == 2_437_000_000

        # 0.001 MHz = 1 kHz = 1000 Hz
        assert parse_frequency_hz("0.001MHz") == 1_000


class TestUnitConversionDeterminism:
    """Tests for deterministic behavior of unit conversions."""

    def test_length_conversion_deterministic(self) -> None:
        """Length conversion must be deterministic across calls."""
        from formula_foundry.coupongen.units import parse_length_nm

        values = ["0.254mm", "10mil", "254um", "254000nm", "254000"]

        # Parse each value multiple times
        for value in values:
            results = [parse_length_nm(value) for _ in range(10)]
            assert len(set(results)) == 1, f"Non-deterministic for {value}"

    def test_equivalent_representations(self) -> None:
        """Equivalent representations must produce identical results."""
        from formula_foundry.coupongen.units import parse_length_nm

        # All of these represent the same physical dimension
        representations = [
            ("1mm", "1000um", 1_000_000),
            ("10mil", "0.254mm", 254_000),
            ("1000nm", "1um", 1_000),
        ]

        for val1, val2, expected in representations:
            assert parse_length_nm(val1) == expected
            assert parse_length_nm(val2) == expected


class TestPydanticIntegration:
    """Tests for Pydantic integration of annotated types."""

    def test_lengthnm_json_schema(self) -> None:
        """LengthNM should have proper JSON schema for validation."""
        from pydantic import BaseModel

        from formula_foundry.coupongen.units import LengthNM

        class TestModel(BaseModel):
            width: LengthNM

        schema = TestModel.model_json_schema()
        # Schema should allow multiple input formats
        width_schema = schema["properties"]["width"]
        assert "anyOf" in width_schema or "type" in width_schema

    def test_anglemdeg_json_schema(self) -> None:
        """AngleMdeg should have proper JSON schema for validation."""
        from pydantic import BaseModel

        from formula_foundry.coupongen.units import AngleMdeg

        class TestModel(BaseModel):
            rotation: AngleMdeg

        schema = TestModel.model_json_schema()
        rotation_schema = schema["properties"]["rotation"]
        assert "anyOf" in rotation_schema or "type" in rotation_schema

    def test_frequencyhz_json_schema(self) -> None:
        """FrequencyHz should have proper JSON schema for validation."""
        from pydantic import BaseModel

        from formula_foundry.coupongen.units import FrequencyHz

        class TestModel(BaseModel):
            freq: FrequencyHz

        schema = TestModel.model_json_schema()
        freq_schema = schema["properties"]["freq"]
        assert "anyOf" in freq_schema or "type" in freq_schema

    def test_model_serialization_preserves_int(self) -> None:
        """Serialized model should preserve integer representation."""
        from pydantic import BaseModel

        from formula_foundry.coupongen.units import FrequencyHz, LengthNM

        class TestModel(BaseModel):
            width: LengthNM
            freq: FrequencyHz

        model = TestModel(width="1mm", freq="1GHz")  # type: ignore[arg-type]

        # model_dump should return integers
        data = model.model_dump()
        assert data["width"] == 1_000_000
        assert data["freq"] == 1_000_000_000
        assert isinstance(data["width"], int)
        assert isinstance(data["freq"], int)
