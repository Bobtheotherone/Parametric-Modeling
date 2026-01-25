"""Unit tests for coupongen families module.

Tests the family validation and constraint helpers used for
enforcing family-specific rules on CouponSpecs (REQ-M1-002).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Direct import to avoid broken import chain in formula_foundry.__init__
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
_families_spec = importlib.util.spec_from_file_location(
    "families", _SRC_DIR / "formula_foundry" / "coupongen" / "families.py"
)
_families = importlib.util.module_from_spec(_families_spec)  # type: ignore[arg-type]
_families_spec.loader.exec_module(_families)  # type: ignore[union-attr]

FAMILY_F0 = _families.FAMILY_F0
FAMILY_F1 = _families.FAMILY_F1
F0_REQUIRED_FIELDS = _families.F0_REQUIRED_FIELDS
F1_ONLY_FIELDS = _families.F1_ONLY_FIELDS
SUPPORTED_FAMILIES = _families.SUPPORTED_FAMILIES
FamilyValidationError = _families.FamilyValidationError
get_family_forbidden_fields = _families.get_family_forbidden_fields
get_family_required_fields = _families.get_family_required_fields
validate_family = _families.validate_family


class TestFamilyConstants:
    """Tests for family constant definitions."""

    def test_family_f0_value(self) -> None:
        """F0 constant has correct value."""
        assert FAMILY_F0 == "F0_CAL_THRU_LINE"

    def test_family_f1_value(self) -> None:
        """F1 constant has correct value."""
        assert FAMILY_F1 == "F1_SINGLE_ENDED_VIA"

    def test_supported_families_contains_f0_f1(self) -> None:
        """SUPPORTED_FAMILIES includes F0 and F1."""
        assert FAMILY_F0 in SUPPORTED_FAMILIES
        assert FAMILY_F1 in SUPPORTED_FAMILIES

    def test_supported_families_length(self) -> None:
        """SUPPORTED_FAMILIES has exactly 2 entries."""
        assert len(SUPPORTED_FAMILIES) == 2

    def test_f1_only_fields_contains_discontinuity(self) -> None:
        """F1_ONLY_FIELDS includes discontinuity."""
        assert "discontinuity" in F1_ONLY_FIELDS

    def test_f0_required_fields_contains_length_right(self) -> None:
        """F0_REQUIRED_FIELDS includes transmission_line.length_right_nm."""
        assert "transmission_line.length_right_nm" in F0_REQUIRED_FIELDS


class TestFamilyValidationError:
    """Tests for FamilyValidationError exception class."""

    def test_error_with_field(self) -> None:
        """Error message includes family and field."""
        error = FamilyValidationError("F0_CAL_THRU_LINE", "discontinuity", "not allowed")
        assert error.family == "F0_CAL_THRU_LINE"
        assert error.field == "discontinuity"
        assert error.reason == "not allowed"
        assert "F0_CAL_THRU_LINE.discontinuity" in str(error)
        assert "not allowed" in str(error)

    def test_error_without_field(self) -> None:
        """Error message works when field is None."""
        error = FamilyValidationError("F99_UNKNOWN", None, "unsupported family")
        assert error.family == "F99_UNKNOWN"
        assert error.field is None
        assert error.reason == "unsupported family"
        assert "F99_UNKNOWN" in str(error)
        assert "unsupported family" in str(error)

    def test_error_is_valueerror_subclass(self) -> None:
        """FamilyValidationError is a ValueError subclass."""
        error = FamilyValidationError("F0", "field", "reason")
        assert isinstance(error, ValueError)

    def test_error_can_be_raised_and_caught(self) -> None:
        """Error can be raised and caught."""
        with pytest.raises(FamilyValidationError) as exc_info:
            raise FamilyValidationError("F1", "type", "invalid type")
        assert exc_info.value.family == "F1"


class TestGetFamilyForbiddenFields:
    """Tests for get_family_forbidden_fields function."""

    def test_f0_forbids_f1_only_fields(self) -> None:
        """F0 family forbids F1-only fields."""
        forbidden = get_family_forbidden_fields(FAMILY_F0)
        assert "discontinuity" in forbidden

    def test_f1_forbids_nothing(self) -> None:
        """F1 family has no forbidden fields."""
        forbidden = get_family_forbidden_fields(FAMILY_F1)
        assert len(forbidden) == 0
        assert isinstance(forbidden, frozenset)

    def test_unknown_family_forbids_nothing(self) -> None:
        """Unknown family returns empty frozenset (default behavior)."""
        forbidden = get_family_forbidden_fields("F99_UNKNOWN")
        assert len(forbidden) == 0
        assert isinstance(forbidden, frozenset)

    def test_returns_frozenset_type(self) -> None:
        """Function always returns frozenset."""
        for family in [FAMILY_F0, FAMILY_F1, "UNKNOWN"]:
            result = get_family_forbidden_fields(family)
            assert isinstance(result, frozenset)


class TestGetFamilyRequiredFields:
    """Tests for get_family_required_fields function."""

    def test_f0_requires_length_right_nm(self) -> None:
        """F0 family requires transmission_line.length_right_nm."""
        required = get_family_required_fields(FAMILY_F0)
        assert "transmission_line.length_right_nm" in required

    def test_f1_has_no_extra_required_fields(self) -> None:
        """F1 family has no extra required fields from this function."""
        required = get_family_required_fields(FAMILY_F1)
        assert len(required) == 0
        assert isinstance(required, frozenset)

    def test_unknown_family_has_no_required_fields(self) -> None:
        """Unknown family returns empty frozenset."""
        required = get_family_required_fields("F99_UNKNOWN")
        assert len(required) == 0
        assert isinstance(required, frozenset)

    def test_returns_frozenset_type(self) -> None:
        """Function always returns frozenset."""
        for family in [FAMILY_F0, FAMILY_F1, "UNKNOWN"]:
            result = get_family_required_fields(family)
            assert isinstance(result, frozenset)


class TestValidateFamilyF0:
    """Tests for validate_family with F0 family."""

    def _make_f0_spec_mock(
        self, discontinuity: Any = None
    ) -> MagicMock:
        """Create a mock F0 spec."""
        spec = MagicMock()
        spec.coupon_family = FAMILY_F0
        spec.discontinuity = discontinuity
        return spec

    def test_valid_f0_without_discontinuity(self) -> None:
        """Valid F0 spec without discontinuity passes."""
        spec = self._make_f0_spec_mock(discontinuity=None)
        # Should not raise
        validate_family(spec)

    def test_f0_with_discontinuity_raises(self) -> None:
        """F0 spec with discontinuity raises FamilyValidationError."""
        discontinuity_mock = MagicMock()
        spec = self._make_f0_spec_mock(discontinuity=discontinuity_mock)

        with pytest.raises(FamilyValidationError) as exc_info:
            validate_family(spec)

        assert exc_info.value.family == FAMILY_F0
        assert exc_info.value.field == "discontinuity"
        assert "F1-only" in exc_info.value.reason


class TestValidateFamilyF1:
    """Tests for validate_family with F1 family."""

    def _make_f1_spec_mock(
        self, discontinuity: Any = None, disc_type: str = "VIA_TRANSITION"
    ) -> MagicMock:
        """Create a mock F1 spec."""
        spec = MagicMock()
        spec.coupon_family = FAMILY_F1
        if discontinuity is False:
            spec.discontinuity = None
        else:
            disc = MagicMock()
            disc.type = disc_type
            spec.discontinuity = disc
        return spec

    def test_valid_f1_with_via_transition(self) -> None:
        """Valid F1 spec with VIA_TRANSITION discontinuity passes."""
        spec = self._make_f1_spec_mock(disc_type="VIA_TRANSITION")
        # Should not raise
        validate_family(spec)

    def test_f1_without_discontinuity_raises(self) -> None:
        """F1 spec without discontinuity raises FamilyValidationError."""
        spec = self._make_f1_spec_mock(discontinuity=False)

        with pytest.raises(FamilyValidationError) as exc_info:
            validate_family(spec)

        assert exc_info.value.family == FAMILY_F1
        assert exc_info.value.field == "discontinuity"
        assert "requires" in exc_info.value.reason

    def test_f1_with_wrong_discontinuity_type_raises(self) -> None:
        """F1 spec with wrong discontinuity type raises FamilyValidationError."""
        spec = self._make_f1_spec_mock(disc_type="WRONG_TYPE")

        with pytest.raises(FamilyValidationError) as exc_info:
            validate_family(spec)

        assert exc_info.value.family == FAMILY_F1
        assert exc_info.value.field == "discontinuity.type"
        assert "VIA_TRANSITION" in exc_info.value.reason
        assert "WRONG_TYPE" in exc_info.value.reason


class TestValidateFamilyUnsupported:
    """Tests for validate_family with unsupported families."""

    def test_unsupported_family_raises(self) -> None:
        """Unsupported family raises FamilyValidationError."""
        spec = MagicMock()
        spec.coupon_family = "F99_UNSUPPORTED_FAMILY"

        with pytest.raises(FamilyValidationError) as exc_info:
            validate_family(spec)

        assert exc_info.value.family == "F99_UNSUPPORTED_FAMILY"
        assert exc_info.value.field is None
        assert "unsupported" in exc_info.value.reason.lower()

    def test_empty_string_family_raises(self) -> None:
        """Empty string family raises FamilyValidationError."""
        spec = MagicMock()
        spec.coupon_family = ""

        with pytest.raises(FamilyValidationError) as exc_info:
            validate_family(spec)

        assert exc_info.value.family == ""

    def test_case_sensitive_family_check(self) -> None:
        """Family check is case-sensitive (lowercase f0 is unsupported)."""
        spec = MagicMock()
        spec.coupon_family = "f0_cal_thru_line"  # lowercase

        with pytest.raises(FamilyValidationError):
            validate_family(spec)


class TestFamilyIntegration:
    """Integration-style tests combining multiple family functions."""

    def test_f0_forbidden_matches_f1_only(self) -> None:
        """F0's forbidden fields match F1_ONLY_FIELDS constant."""
        forbidden = get_family_forbidden_fields(FAMILY_F0)
        assert forbidden == F1_ONLY_FIELDS

    def test_f0_required_matches_constant(self) -> None:
        """F0's required fields match F0_REQUIRED_FIELDS constant."""
        required = get_family_required_fields(FAMILY_F0)
        assert required == F0_REQUIRED_FIELDS

    def test_validation_error_from_validation_function(self) -> None:
        """Validation function produces proper FamilyValidationError."""
        spec = MagicMock()
        spec.coupon_family = FAMILY_F0
        spec.discontinuity = MagicMock()  # F1-only feature on F0

        with pytest.raises(FamilyValidationError) as exc_info:
            validate_family(spec)

        # Can access all attributes
        error = exc_info.value
        assert isinstance(error.family, str)
        assert error.field is not None
        assert isinstance(error.reason, str)
