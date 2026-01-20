from __future__ import annotations

import pytest

from formula_foundry.coupongen.units import LengthNM, parse_length_nm


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
        with pytest.raises(ValueError, match="integer number of nanometers"):
            parse_length_nm("0.5nm")  # 0.5 nm is not an integer

    def test_non_integer_um_conversion_fails(self) -> None:
        """um conversion that yields non-integer nm must fail."""
        with pytest.raises(ValueError, match="integer number of nanometers"):
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
        with pytest.raises(ValueError, match="integer number of nanometers"):
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
