"""Extended unit tests for openems/units.py TimePS parsing.

This module provides comprehensive tests for the TimePS type and parse_time_ps function
in formula_foundry.openems.units, mirroring the thoroughness of the FrequencyHz tests
in test_m1_units.py.

REQ-M2-006: TimePS parsing for simulation time specifications.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from formula_foundry.openems.units import (
    FrequencyHz,
    TimePS,
    parse_frequency_hz,
    parse_time_ps,
)


class TestParseTimePS:
    """Comprehensive tests for TimePS parsing."""

    def test_string_with_ps_unit(self) -> None:
        """Parse picoseconds with ps unit."""
        assert parse_time_ps("100ps") == 100
        assert parse_time_ps("1ps") == 1
        assert parse_time_ps("1000000ps") == 1_000_000

    def test_string_with_ns_unit(self) -> None:
        """Parse nanoseconds with ns unit."""
        assert parse_time_ps("1ns") == 1_000
        assert parse_time_ps("50ns") == 50_000
        assert parse_time_ps("0.5ns") == 500

    def test_string_with_us_unit(self) -> None:
        """Parse microseconds with us unit."""
        assert parse_time_ps("1us") == 1_000_000
        assert parse_time_ps("10us") == 10_000_000
        assert parse_time_ps("0.001us") == 1_000

    def test_string_with_ms_unit(self) -> None:
        """Parse milliseconds with ms unit."""
        assert parse_time_ps("1ms") == 1_000_000_000
        assert parse_time_ps("0.001ms") == 1_000_000
        assert parse_time_ps("0.1ms") == 100_000_000

    def test_string_with_s_unit(self) -> None:
        """Parse seconds with s unit."""
        assert parse_time_ps("1s") == 1_000_000_000_000
        assert parse_time_ps("0.001s") == 1_000_000_000

    def test_integer_string_without_unit(self) -> None:
        """Plain integer strings are treated as picoseconds."""
        assert parse_time_ps("1000000") == 1_000_000
        assert parse_time_ps("0") == 0
        assert parse_time_ps("-500") == -500

    def test_raw_integer(self) -> None:
        """Raw integers pass through as ps."""
        assert parse_time_ps(1_000_000) == 1_000_000
        assert parse_time_ps(0) == 0
        assert parse_time_ps(-100) == -100

    def test_raw_float_integer_valued(self) -> None:
        """Float values that are exact integers are accepted."""
        assert parse_time_ps(1000.0) == 1_000
        assert parse_time_ps(1e9) == 1_000_000_000

    def test_returns_int_type(self) -> None:
        """Parser always returns Python int."""
        assert isinstance(parse_time_ps("1ns"), int)
        assert isinstance(parse_time_ps(1000), int)
        assert isinstance(parse_time_ps(1000.0), int)

    def test_whitespace_tolerance(self) -> None:
        """Whitespace around value and unit is tolerated."""
        assert parse_time_ps("  1ns  ") == 1_000
        assert parse_time_ps(" 100 ps ") == 100

    def test_negative_values(self) -> None:
        """Negative values are supported."""
        assert parse_time_ps("-1ns") == -1_000
        assert parse_time_ps("-100ps") == -100

    def test_positive_sign(self) -> None:
        """Explicit positive sign is accepted."""
        assert parse_time_ps("+1ns") == 1_000


class TestTimePSFailures:
    """TimePS parsers must fail deterministically on invalid inputs."""

    def test_unknown_unit_raises_valueerror(self) -> None:
        """Unknown unit suffix should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown TimePS unit"):
            parse_time_ps("100fs")  # femtoseconds not supported

    def test_unknown_unit_minutes(self) -> None:
        """Minutes not supported as unit."""
        with pytest.raises(ValueError, match="Unknown TimePS unit"):
            parse_time_ps("1min")

    def test_unitless_decimal_fails(self) -> None:
        """Decimal without unit is ambiguous - must fail."""
        with pytest.raises(ValueError):
            parse_time_ps("1.5")

    def test_non_integer_ps_fails(self) -> None:
        """Values that don't resolve to integer ps must fail."""
        with pytest.raises(ValueError, match="integer"):
            parse_time_ps("0.5ps")  # 0.5 ps is not an integer

    def test_non_integer_ns_conversion_fails(self) -> None:
        """ns conversion that yields non-integer ps must fail."""
        with pytest.raises(ValueError, match="integer"):
            parse_time_ps("0.0001ns")  # 0.1 ps, not integer

    def test_boolean_rejected(self) -> None:
        """Boolean values are rejected (bool is subclass of int)."""
        with pytest.raises(ValueError, match="boolean"):
            parse_time_ps(True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="boolean"):
            parse_time_ps(False)  # type: ignore[arg-type]

    def test_empty_string_fails(self) -> None:
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="numeric value"):
            parse_time_ps("")

    def test_whitespace_only_fails(self) -> None:
        """Whitespace-only string should raise ValueError."""
        with pytest.raises(ValueError, match="numeric value"):
            parse_time_ps("   ")

    def test_float_non_integer_fails(self) -> None:
        """Float that is not exactly integer must fail."""
        with pytest.raises(ValueError, match="integer"):
            parse_time_ps(1000.5)

    def test_invalid_format_unit_before_number(self) -> None:
        """Unit before number should fail."""
        with pytest.raises(ValueError):
            parse_time_ps("ns1")

    def test_invalid_format_multiple_numbers(self) -> None:
        """Multiple numbers should fail."""
        with pytest.raises(ValueError):
            parse_time_ps("1 2 ns")


class TestTimePSI64Bounds:
    """TimePS values must stay within signed 64-bit integer range."""

    def test_max_i64_accepted(self) -> None:
        """Maximum 64-bit signed integer should be accepted."""
        max_i64 = 2**63 - 1
        assert parse_time_ps(max_i64) == max_i64

    def test_min_i64_accepted(self) -> None:
        """Minimum 64-bit signed integer should be accepted."""
        min_i64 = -(2**63)
        assert parse_time_ps(min_i64) == min_i64

    def test_overflow_fails(self) -> None:
        """Value exceeding max 64-bit should fail."""
        with pytest.raises(ValueError, match="64-bit"):
            parse_time_ps(2**63)  # Just past max

    def test_underflow_fails(self) -> None:
        """Value below min 64-bit should fail."""
        with pytest.raises(ValueError, match="64-bit"):
            parse_time_ps(-(2**63) - 1)  # Just past min


class TestTimePSAnnotatedType:
    """Test the Pydantic-integrated TimePS type."""

    def test_timeps_can_be_used_in_pydantic_model(self) -> None:
        """TimePS should work in Pydantic models."""

        class TestModel(BaseModel):
            duration_ps: TimePS

        m = TestModel(duration_ps="1ns")  # type: ignore[arg-type]
        assert m.duration_ps == 1_000

        m2 = TestModel(duration_ps=5_000)
        assert m2.duration_ps == 5_000

    def test_timeps_validation_error_on_bad_unit(self) -> None:
        """Invalid unit should cause validation error."""

        class TestModel(BaseModel):
            duration_ps: TimePS

        with pytest.raises(ValidationError):
            TestModel(duration_ps="100fs")  # type: ignore[arg-type]


class TestFrequencyHzEdgeCases:
    """Additional edge case tests for FrequencyHz to complement test_m1_units.py."""

    def test_frequencyhz_case_variations(self) -> None:
        """Test various case combinations for FrequencyHz units."""
        assert parse_frequency_hz("1ghz") == 1_000_000_000
        assert parse_frequency_hz("1GHZ") == 1_000_000_000
        assert parse_frequency_hz("1Ghz") == 1_000_000_000
        assert parse_frequency_hz("1gHz") == 1_000_000_000

    def test_frequencyhz_decimal_units(self) -> None:
        """Test decimal frequency values with units."""
        assert parse_frequency_hz("2.5GHz") == 2_500_000_000
        assert parse_frequency_hz("1.5MHz") == 1_500_000
        assert parse_frequency_hz("0.1kHz") == 100

    def test_frequencyhz_large_values(self) -> None:
        """Test large frequency values (100 GHz range)."""
        assert parse_frequency_hz("100GHz") == 100_000_000_000
        assert parse_frequency_hz("250GHz") == 250_000_000_000


class TestFrequencyHzAnnotatedTypeExtended:
    """Extended tests for FrequencyHz Pydantic integration."""

    def test_frequencyhz_list_in_model(self) -> None:
        """FrequencyHz should work in list fields."""

        class TestModel(BaseModel):
            frequencies: list[FrequencyHz]

        m = TestModel(frequencies=["1GHz", "2GHz", 3_000_000_000])  # type: ignore[arg-type]
        assert m.frequencies == [1_000_000_000, 2_000_000_000, 3_000_000_000]

    def test_frequencyhz_optional_in_model(self) -> None:
        """FrequencyHz should work in optional fields."""

        class TestModel(BaseModel):
            freq: FrequencyHz | None = None

        m1 = TestModel()
        assert m1.freq is None

        m2 = TestModel(freq="5GHz")  # type: ignore[arg-type]
        assert m2.freq == 5_000_000_000


class TestTimePSPydanticIntegrationExtended:
    """Extended tests for TimePS Pydantic integration."""

    def test_timeps_list_in_model(self) -> None:
        """TimePS should work in list fields."""

        class TestModel(BaseModel):
            times: list[TimePS]

        m = TestModel(times=["1ns", "2ns", 3_000])  # type: ignore[arg-type]
        assert m.times == [1_000, 2_000, 3_000]

    def test_timeps_optional_in_model(self) -> None:
        """TimePS should work in optional fields."""

        class TestModel(BaseModel):
            duration: TimePS | None = None

        m1 = TestModel()
        assert m1.duration is None

        m2 = TestModel(duration="10ns")  # type: ignore[arg-type]
        assert m2.duration == 10_000


class TestCrossTypeConsistency:
    """Test consistency between FrequencyHz and TimePS types."""

    def test_both_types_reject_boolean(self) -> None:
        """Both types should reject boolean values consistently."""
        with pytest.raises(ValueError, match="boolean"):
            parse_frequency_hz(True)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="boolean"):
            parse_time_ps(True)  # type: ignore[arg-type]

    def test_both_types_handle_zero(self) -> None:
        """Both types should handle zero correctly."""
        assert parse_frequency_hz(0) == 0
        assert parse_time_ps(0) == 0
        assert parse_frequency_hz("0") == 0
        assert parse_time_ps("0") == 0

    def test_both_types_handle_negative(self) -> None:
        """Both types should handle negative values consistently."""
        assert parse_frequency_hz(-1_000_000) == -1_000_000
        assert parse_time_ps(-1_000_000) == -1_000_000
