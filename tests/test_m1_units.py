from __future__ import annotations

import pytest

from formula_foundry.coupongen.units import (
    AngleMdeg,
    FrequencyHz,
    LengthNM,
    parse_angle_mdeg,
    parse_frequency_hz,
    parse_length_nm,
)


class TestParseLengthNm:
    """Tests for LengthNM parsing - normalize inputs to integer nm and fail on ambiguous units."""

    def test_string_with_mm_unit(self) -> None:
        assert parse_length_nm("0.25mm") == 250_000
        assert parse_length_nm("1mm") == 1_000_000
        assert parse_length_nm("0.001mm") == 1_000

    def test_string_with_mil_unit(self) -> None:
        assert parse_length_nm("10mil") == 254_000
        assert parse_length_nm("1mil") == 25_400

    def test_string_with_um_unit(self) -> None:
        assert parse_length_nm("250um") == 250_000
        assert parse_length_nm("1um") == 1_000

    def test_string_with_nm_unit(self) -> None:
        assert parse_length_nm("1000nm") == 1_000
        assert parse_length_nm("1nm") == 1

    def test_integer_string_without_unit(self) -> None:
        """Plain integer strings are treated as nanometers."""
        assert parse_length_nm("1000") == 1_000
        assert parse_length_nm("0") == 0
        assert parse_length_nm("-500") == -500

    def test_raw_integer(self) -> None:
        """Raw integers pass through as nm."""
        assert parse_length_nm(1_000_000) == 1_000_000
        assert parse_length_nm(0) == 0
        assert parse_length_nm(-100) == -100

    def test_raw_float_integer_valued(self) -> None:
        """Float values that are exact integers are accepted."""
        assert parse_length_nm(1000.0) == 1_000
        assert parse_length_nm(1e6) == 1_000_000

    def test_returns_int_type(self) -> None:
        """Parser always returns Python int."""
        assert isinstance(parse_length_nm("1mm"), int)
        assert isinstance(parse_length_nm(1000), int)
        assert isinstance(parse_length_nm(1000.0), int)

    def test_whitespace_tolerance(self) -> None:
        """Whitespace around value and unit is tolerated."""
        assert parse_length_nm("  1mm  ") == 1_000_000
        assert parse_length_nm(" 10 mil ") == 254_000

    def test_negative_values(self) -> None:
        """Negative values are supported."""
        assert parse_length_nm("-1mm") == -1_000_000
        assert parse_length_nm("-10mil") == -254_000

    def test_positive_sign(self) -> None:
        """Explicit positive sign is accepted."""
        assert parse_length_nm("+1mm") == 1_000_000

    def test_case_insensitive_units(self) -> None:
        """Units are case-insensitive."""
        assert parse_length_nm("1MM") == 1_000_000
        assert parse_length_nm("1Mm") == 1_000_000
        assert parse_length_nm("10MIL") == 254_000
        assert parse_length_nm("1UM") == 1_000
        assert parse_length_nm("1NM") == 1


class TestAmbiguousUnitFailures:
    """Parsers must fail deterministically on ambiguous or unknown units."""

    def test_unknown_unit_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="Unknown LengthNM unit"):
            parse_length_nm("12.34foo")

    def test_unitless_decimal_fails(self) -> None:
        """Decimal without unit is ambiguous - must fail."""
        with pytest.raises(ValueError):
            parse_length_nm("1.5")

    def test_non_integer_nm_fails(self) -> None:
        """Values that don't resolve to integer nm must fail."""
        with pytest.raises(ValueError, match="integer"):
            parse_length_nm("0.5nm")  # 0.5 nm is not an integer

    def test_non_integer_um_conversion_fails(self) -> None:
        """um conversion that yields non-integer nm must fail."""
        with pytest.raises(ValueError, match="integer"):
            parse_length_nm("0.0001um")  # 0.1 nm, not integer

    def test_boolean_rejected(self) -> None:
        """Boolean values are rejected (bool is subclass of int)."""
        with pytest.raises(ValueError, match="boolean"):
            parse_length_nm(True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="boolean"):
            parse_length_nm(False)  # type: ignore[arg-type]

    def test_empty_string_fails(self) -> None:
        with pytest.raises(ValueError, match="numeric value"):
            parse_length_nm("")

    def test_whitespace_only_fails(self) -> None:
        with pytest.raises(ValueError, match="numeric value"):
            parse_length_nm("   ")

    def test_invalid_format_fails(self) -> None:
        """Malformed strings should fail."""
        with pytest.raises(ValueError):
            parse_length_nm("mm1")  # unit before number
        with pytest.raises(ValueError):
            parse_length_nm("1 2 mm")  # multiple numbers

    def test_float_non_integer_fails(self) -> None:
        """Float that is not exactly integer must fail."""
        with pytest.raises(ValueError, match="integer"):
            parse_length_nm(1000.5)


class TestI64Bounds:
    """Values must stay within signed 64-bit integer range."""

    def test_max_i64_accepted(self) -> None:
        max_i64 = 2**63 - 1
        assert parse_length_nm(max_i64) == max_i64

    def test_min_i64_accepted(self) -> None:
        min_i64 = -(2**63)
        assert parse_length_nm(min_i64) == min_i64

    def test_overflow_fails(self) -> None:
        with pytest.raises(ValueError, match="64-bit"):
            parse_length_nm(2**63)  # Just past max

    def test_underflow_fails(self) -> None:
        with pytest.raises(ValueError, match="64-bit"):
            parse_length_nm(-(2**63) - 1)  # Just past min


class TestLengthNMAnnotatedType:
    """Test the Pydantic-integrated LengthNM type."""

    def test_lengthnm_can_be_used_in_pydantic_model(self) -> None:
        from pydantic import BaseModel

        class TestModel(BaseModel):
            width_nm: LengthNM

        m = TestModel(width_nm="1mm")  # type: ignore[arg-type]
        assert m.width_nm == 1_000_000

        m2 = TestModel(width_nm=500_000)
        assert m2.width_nm == 500_000

    def test_lengthnm_validation_error_on_bad_unit(self) -> None:
        from pydantic import BaseModel, ValidationError

        class TestModel(BaseModel):
            width_nm: LengthNM

        with pytest.raises(ValidationError):
            TestModel(width_nm="10feet")  # type: ignore[arg-type]


class TestParseAngleMdeg:
    """Tests for AngleMdeg parsing - normalize inputs to integer millidegrees."""

    def test_string_with_deg_unit(self) -> None:
        assert parse_angle_mdeg("45deg") == 45_000
        assert parse_angle_mdeg("90deg") == 90_000
        assert parse_angle_mdeg("-180deg") == -180_000
        assert parse_angle_mdeg("0.5deg") == 500

    def test_string_with_mdeg_unit(self) -> None:
        assert parse_angle_mdeg("1000mdeg") == 1_000
        assert parse_angle_mdeg("45000mdeg") == 45_000
        assert parse_angle_mdeg("1mdeg") == 1

    def test_integer_string_without_unit(self) -> None:
        """Plain integer strings are treated as millidegrees."""
        assert parse_angle_mdeg("1000") == 1_000
        assert parse_angle_mdeg("0") == 0
        assert parse_angle_mdeg("-500") == -500

    def test_raw_integer(self) -> None:
        """Raw integers pass through as millidegrees."""
        assert parse_angle_mdeg(45_000) == 45_000
        assert parse_angle_mdeg(0) == 0
        assert parse_angle_mdeg(-90_000) == -90_000

    def test_raw_float_integer_valued(self) -> None:
        """Float values that are exact integers are accepted."""
        assert parse_angle_mdeg(45000.0) == 45_000
        assert parse_angle_mdeg(1e3) == 1_000

    def test_returns_int_type(self) -> None:
        """Parser always returns Python int."""
        assert isinstance(parse_angle_mdeg("45deg"), int)
        assert isinstance(parse_angle_mdeg(1000), int)
        assert isinstance(parse_angle_mdeg(1000.0), int)

    def test_whitespace_tolerance(self) -> None:
        """Whitespace around value and unit is tolerated."""
        assert parse_angle_mdeg("  45deg  ") == 45_000
        assert parse_angle_mdeg(" 1000 mdeg ") == 1_000

    def test_negative_values(self) -> None:
        """Negative values are supported."""
        assert parse_angle_mdeg("-45deg") == -45_000
        assert parse_angle_mdeg("-1000mdeg") == -1_000

    def test_positive_sign(self) -> None:
        """Explicit positive sign is accepted."""
        assert parse_angle_mdeg("+45deg") == 45_000

    def test_case_insensitive_units(self) -> None:
        """Units are case-insensitive."""
        assert parse_angle_mdeg("45DEG") == 45_000
        assert parse_angle_mdeg("45Deg") == 45_000
        assert parse_angle_mdeg("1000MDEG") == 1_000


class TestAngleMdegFailures:
    """AngleMdeg parsers must fail deterministically on ambiguous or unknown units."""

    def test_unknown_unit_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="Unknown AngleMdeg unit"):
            parse_angle_mdeg("45rad")

    def test_unitless_decimal_fails(self) -> None:
        """Decimal without unit is ambiguous - must fail."""
        with pytest.raises(ValueError):
            parse_angle_mdeg("1.5")

    def test_non_integer_mdeg_fails(self) -> None:
        """Values that don't resolve to integer millidegrees must fail."""
        with pytest.raises(ValueError, match="integer"):
            parse_angle_mdeg("0.5mdeg")

    def test_non_integer_deg_conversion_fails(self) -> None:
        """deg conversion that yields non-integer mdeg must fail."""
        with pytest.raises(ValueError, match="integer"):
            parse_angle_mdeg("0.0001deg")  # 0.1 mdeg, not integer

    def test_boolean_rejected(self) -> None:
        """Boolean values are rejected (bool is subclass of int)."""
        with pytest.raises(ValueError, match="boolean"):
            parse_angle_mdeg(True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="boolean"):
            parse_angle_mdeg(False)  # type: ignore[arg-type]

    def test_empty_string_fails(self) -> None:
        with pytest.raises(ValueError, match="numeric value"):
            parse_angle_mdeg("")

    def test_whitespace_only_fails(self) -> None:
        with pytest.raises(ValueError, match="numeric value"):
            parse_angle_mdeg("   ")

    def test_float_non_integer_fails(self) -> None:
        """Float that is not exactly integer must fail."""
        with pytest.raises(ValueError, match="integer"):
            parse_angle_mdeg(1000.5)


class TestAngleMdegI64Bounds:
    """Angle values must stay within signed 64-bit integer range."""

    def test_max_i64_accepted(self) -> None:
        max_i64 = 2**63 - 1
        assert parse_angle_mdeg(max_i64) == max_i64

    def test_min_i64_accepted(self) -> None:
        min_i64 = -(2**63)
        assert parse_angle_mdeg(min_i64) == min_i64

    def test_overflow_fails(self) -> None:
        with pytest.raises(ValueError, match="64-bit"):
            parse_angle_mdeg(2**63)

    def test_underflow_fails(self) -> None:
        with pytest.raises(ValueError, match="64-bit"):
            parse_angle_mdeg(-(2**63) - 1)


class TestAngleMdegAnnotatedType:
    """Test the Pydantic-integrated AngleMdeg type."""

    def test_anglemdeg_can_be_used_in_pydantic_model(self) -> None:
        from pydantic import BaseModel

        class TestModel(BaseModel):
            rotation_mdeg: AngleMdeg

        m = TestModel(rotation_mdeg="45deg")  # type: ignore[arg-type]
        assert m.rotation_mdeg == 45_000

        m2 = TestModel(rotation_mdeg=90_000)
        assert m2.rotation_mdeg == 90_000

    def test_anglemdeg_validation_error_on_bad_unit(self) -> None:
        from pydantic import BaseModel, ValidationError

        class TestModel(BaseModel):
            rotation_mdeg: AngleMdeg

        with pytest.raises(ValidationError):
            TestModel(rotation_mdeg="45rad")  # type: ignore[arg-type]


class TestParseFrequencyHz:
    """Tests for FrequencyHz parsing - normalize inputs to integer Hz."""

    def test_string_with_hz_unit(self) -> None:
        assert parse_frequency_hz("1000Hz") == 1_000
        assert parse_frequency_hz("1Hz") == 1
        assert parse_frequency_hz("1000000Hz") == 1_000_000

    def test_string_with_khz_unit(self) -> None:
        assert parse_frequency_hz("1kHz") == 1_000
        assert parse_frequency_hz("50kHz") == 50_000
        assert parse_frequency_hz("0.5kHz") == 500

    def test_string_with_mhz_unit(self) -> None:
        assert parse_frequency_hz("100MHz") == 100_000_000
        assert parse_frequency_hz("1MHz") == 1_000_000
        assert parse_frequency_hz("2.4MHz") == 2_400_000

    def test_string_with_ghz_unit(self) -> None:
        assert parse_frequency_hz("1GHz") == 1_000_000_000
        assert parse_frequency_hz("2.4GHz") == 2_400_000_000
        assert parse_frequency_hz("10GHz") == 10_000_000_000

    def test_integer_string_without_unit(self) -> None:
        """Plain integer strings are treated as Hz."""
        assert parse_frequency_hz("1000000000") == 1_000_000_000
        assert parse_frequency_hz("0") == 0
        assert parse_frequency_hz("-500") == -500

    def test_raw_integer(self) -> None:
        """Raw integers pass through as Hz."""
        assert parse_frequency_hz(1_000_000_000) == 1_000_000_000
        assert parse_frequency_hz(0) == 0
        assert parse_frequency_hz(-100) == -100

    def test_raw_float_integer_valued(self) -> None:
        """Float values that are exact integers are accepted."""
        assert parse_frequency_hz(1000.0) == 1_000
        assert parse_frequency_hz(1e9) == 1_000_000_000

    def test_returns_int_type(self) -> None:
        """Parser always returns Python int."""
        assert isinstance(parse_frequency_hz("1GHz"), int)
        assert isinstance(parse_frequency_hz(1000), int)
        assert isinstance(parse_frequency_hz(1000.0), int)

    def test_whitespace_tolerance(self) -> None:
        """Whitespace around value and unit is tolerated."""
        assert parse_frequency_hz("  1GHz  ") == 1_000_000_000
        assert parse_frequency_hz(" 100 MHz ") == 100_000_000

    def test_negative_values(self) -> None:
        """Negative values are supported (unusual but valid)."""
        assert parse_frequency_hz("-1MHz") == -1_000_000
        assert parse_frequency_hz("-1000Hz") == -1_000

    def test_positive_sign(self) -> None:
        """Explicit positive sign is accepted."""
        assert parse_frequency_hz("+1GHz") == 1_000_000_000

    def test_case_insensitive_units(self) -> None:
        """Units are case-insensitive."""
        assert parse_frequency_hz("1ghz") == 1_000_000_000
        assert parse_frequency_hz("1GHZ") == 1_000_000_000
        assert parse_frequency_hz("100mhz") == 100_000_000
        assert parse_frequency_hz("50KHZ") == 50_000
        assert parse_frequency_hz("1000HZ") == 1_000


class TestFrequencyHzFailures:
    """FrequencyHz parsers must fail deterministically on ambiguous or unknown units."""

    def test_unknown_unit_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="Unknown FrequencyHz unit"):
            parse_frequency_hz("100THz")

    def test_unitless_decimal_fails(self) -> None:
        """Decimal without unit is ambiguous - must fail."""
        with pytest.raises(ValueError):
            parse_frequency_hz("1.5")

    def test_non_integer_hz_fails(self) -> None:
        """Values that don't resolve to integer Hz must fail."""
        with pytest.raises(ValueError, match="integer"):
            parse_frequency_hz("0.5Hz")

    def test_non_integer_khz_conversion_fails(self) -> None:
        """kHz conversion that yields non-integer Hz must fail."""
        with pytest.raises(ValueError, match="integer"):
            parse_frequency_hz("0.0001kHz")  # 0.1 Hz, not integer

    def test_boolean_rejected(self) -> None:
        """Boolean values are rejected (bool is subclass of int)."""
        with pytest.raises(ValueError, match="boolean"):
            parse_frequency_hz(True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="boolean"):
            parse_frequency_hz(False)  # type: ignore[arg-type]

    def test_empty_string_fails(self) -> None:
        with pytest.raises(ValueError, match="numeric value"):
            parse_frequency_hz("")

    def test_whitespace_only_fails(self) -> None:
        with pytest.raises(ValueError, match="numeric value"):
            parse_frequency_hz("   ")

    def test_float_non_integer_fails(self) -> None:
        """Float that is not exactly integer must fail."""
        with pytest.raises(ValueError, match="integer"):
            parse_frequency_hz(1000.5)


class TestFrequencyHzI64Bounds:
    """Frequency values must stay within signed 64-bit integer range."""

    def test_max_i64_accepted(self) -> None:
        max_i64 = 2**63 - 1
        assert parse_frequency_hz(max_i64) == max_i64

    def test_min_i64_accepted(self) -> None:
        min_i64 = -(2**63)
        assert parse_frequency_hz(min_i64) == min_i64

    def test_overflow_fails(self) -> None:
        with pytest.raises(ValueError, match="64-bit"):
            parse_frequency_hz(2**63)

    def test_underflow_fails(self) -> None:
        with pytest.raises(ValueError, match="64-bit"):
            parse_frequency_hz(-(2**63) - 1)


class TestFrequencyHzAnnotatedType:
    """Test the Pydantic-integrated FrequencyHz type."""

    def test_frequencyhz_can_be_used_in_pydantic_model(self) -> None:
        from pydantic import BaseModel

        class TestModel(BaseModel):
            freq_hz: FrequencyHz

        m = TestModel(freq_hz="1GHz")  # type: ignore[arg-type]
        assert m.freq_hz == 1_000_000_000

        m2 = TestModel(freq_hz=100_000_000)
        assert m2.freq_hz == 100_000_000

    def test_frequencyhz_validation_error_on_bad_unit(self) -> None:
        from pydantic import BaseModel, ValidationError

        class TestModel(BaseModel):
            freq_hz: FrequencyHz

        with pytest.raises(ValidationError):
            TestModel(freq_hz="100THz")  # type: ignore[arg-type]


# =============================================================================
# REQ-M1-002: Integer nanometer representation requirement test
# =============================================================================


def test_lengthnm_parsing_integer_nm() -> None:
    """REQ-M1-002: Verify geometry is represented as integer nanometers with deterministic parsing.

    This test covers the requirement that all geometry must be internally represented
    as integer nanometers and that parsing of mm/mil/um inputs is deterministic.
    """
    # Integer nanometer representation
    assert parse_length_nm("1mm") == 1_000_000
    assert isinstance(parse_length_nm("1mm"), int)

    # Deterministic parsing of various unit inputs
    assert parse_length_nm("250um") == 250_000
    assert parse_length_nm("10mil") == 254_000
    assert parse_length_nm("1000nm") == 1_000
    assert parse_length_nm(500_000) == 500_000

    # Verify integer output type for all input types
    assert isinstance(parse_length_nm("0.5mm"), int)
    assert isinstance(parse_length_nm("100um"), int)
    assert isinstance(parse_length_nm("1mil"), int)
    assert isinstance(parse_length_nm(123456), int)
