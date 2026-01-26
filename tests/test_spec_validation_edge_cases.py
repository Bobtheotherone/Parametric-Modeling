# SPDX-License-Identifier: MIT
"""Unit tests for spec validation edge cases.

Tests edge cases in spec validation across multiple modules:
- CouponSpec strict validation error handling
- Family validation boundaries
- Constraint violation edge cases
- Layer validation edge cases
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from formula_foundry.coupongen.constraints.core import ConstraintResult as CoreConstraintResult
from formula_foundry.coupongen.constraints.core import ConstraintViolation
from formula_foundry.coupongen.constraints.tiers import (
    ConstraintResult as TiersConstraintResult,
)
from formula_foundry.coupongen.constraints.tiers import (
    ConstraintTier,
    ConstraintViolationError,
)
from formula_foundry.coupongen.families import FamilyValidationError
from formula_foundry.coupongen.layer_validation import LayerSetValidationError, LayerValidationResult
from formula_foundry.coupongen.spec import StrictValidationError

# =============================================================================
# Helper Factories
# =============================================================================


def _make_core_constraint_result(
    constraint_id: str = "TEST_001",
    passed: bool = False,
    description: str = "Test constraint",
    tier: str = "T0",
    value: float = 50.0,
    limit: float = 100.0,
    margin: float = -50.0,
) -> CoreConstraintResult:
    """Create a ConstraintResult from core module for testing.

    Note: ConstraintResult from constraints.core is a frozen dataclass with fields:
    constraint_id, description, tier, value, limit, margin, passed
    """
    return CoreConstraintResult(
        constraint_id=constraint_id,
        description=description,
        tier=tier,  # type: ignore[arg-type]
        value=value,
        limit=limit,
        margin=margin,
        passed=passed,
    )


def _make_tiers_constraint_result(
    constraint_id: str = "TEST_001",
    passed: bool = False,
    description: str = "Test constraint",
    tier: ConstraintTier = "T0",
    value: float = 50.0,
    limit: float = 100.0,
    margin: float = -50.0,
    reason: str = "",
) -> TiersConstraintResult:
    """Create a ConstraintResult from tiers module for testing.

    Note: ConstraintResult from constraints.tiers has an additional 'reason' field.
    ConstraintTier is Literal["T0", "T1", "T2", "T3", "T4"].
    """
    return TiersConstraintResult(
        constraint_id=constraint_id,
        description=description,
        tier=tier,
        value=value,
        limit=limit,
        margin=margin,
        passed=passed,
        reason=reason,
    )


def _make_layer_validation_result(
    family: str = "F0_CAL_THRU_LINE",
    copper_layer_count: int = 4,
    passed: bool = False,
    missing_layers: tuple[str, ...] | None = None,
) -> LayerValidationResult:
    """Create a LayerValidationResult for testing.

    Note: LayerValidationResult is a dataclass with fields:
    passed, missing_layers, extra_layers, expected_layers, actual_layers, copper_layer_count, family
    """
    return LayerValidationResult(
        passed=passed,
        missing_layers=missing_layers or ("In1.Cu",),
        extra_layers=(),
        expected_layers=("F.Cu", "B.Cu", "In1.Cu"),
        actual_layers=("F.Cu", "B.Cu"),
        copper_layer_count=copper_layer_count,
        family=family,
    )


# =============================================================================
# Exception Class Tests
# =============================================================================


class TestStrictValidationError:
    """Tests for StrictValidationError exception."""

    def test_is_exception(self) -> None:
        """StrictValidationError should inherit from Exception."""
        error = StrictValidationError(["Validation failed"])
        assert isinstance(error, Exception)
        assert "Validation failed" in str(error)

    def test_can_be_raised_and_caught(self) -> None:
        """StrictValidationError can be raised and caught."""
        with pytest.raises(StrictValidationError, match="invalid spec"):
            raise StrictValidationError(["invalid spec"])

    def test_preserves_error_details(self) -> None:
        """Error message preserves full details."""
        details = "Field 'width' is required but was None"
        error = StrictValidationError([details])
        assert details in str(error)

    def test_stores_error_list(self) -> None:
        """StrictValidationError stores the errors list."""
        errors = ["Error 1", "Error 2", "Error 3"]
        error = StrictValidationError(errors)
        assert error.errors == errors
        assert len(error.errors) == 3

    def test_message_includes_count(self) -> None:
        """Error message includes the count of errors."""
        error = StrictValidationError(["error1", "error2"])
        assert "2" in str(error)


class TestFamilyValidationError:
    """Tests for FamilyValidationError exception."""

    def test_is_value_error(self) -> None:
        """FamilyValidationError should inherit from ValueError."""
        error = FamilyValidationError("F0_CAL_THRU_LINE", None, "Invalid family configuration")
        assert isinstance(error, ValueError)
        assert "F0_CAL_THRU_LINE" in str(error)

    def test_preserves_family_name(self) -> None:
        """Error message should preserve the invalid family name."""
        error = FamilyValidationError("INVALID_FAMILY", None, "Unknown family type")
        assert "INVALID_FAMILY" in str(error)

    def test_includes_field_when_specified(self) -> None:
        """Error message includes field when specified."""
        error = FamilyValidationError("F1_SINGLE_ENDED_VIA", "discontinuity", "Required field missing")
        msg = str(error)
        assert "F1_SINGLE_ENDED_VIA" in msg
        assert "discontinuity" in msg
        assert "Required" in msg

    def test_stores_attributes(self) -> None:
        """FamilyValidationError stores family, field, and reason."""
        error = FamilyValidationError("F0_CAL_THRU_LINE", "transmission_line", "Invalid config")
        assert error.family == "F0_CAL_THRU_LINE"
        assert error.field == "transmission_line"
        assert error.reason == "Invalid config"


class TestLayerSetValidationError:
    """Tests for LayerSetValidationError exception."""

    def test_is_exception(self) -> None:
        """LayerSetValidationError should inherit from Exception."""
        result = _make_layer_validation_result()
        error = LayerSetValidationError(result)
        assert isinstance(error, Exception)
        assert "Layer set validation failed" in str(error)

    def test_can_contain_layer_details(self) -> None:
        """Error can contain layer details."""
        result = _make_layer_validation_result(missing_layers=("F.Cu", "In2.Cu"))
        error = LayerSetValidationError(result)
        msg = str(error)
        assert "F.Cu" in msg or "missing" in msg

    def test_stores_result(self) -> None:
        """LayerSetValidationError stores the validation result."""
        result = _make_layer_validation_result()
        error = LayerSetValidationError(result)
        assert error.result is result
        assert error.result.family == "F0_CAL_THRU_LINE"


class TestConstraintViolation:
    """Tests for ConstraintViolation exception (from core module)."""

    def test_is_value_error(self) -> None:
        """ConstraintViolation should inherit from ValueError."""
        violations = [_make_core_constraint_result()]
        error = ConstraintViolation(violations)
        assert isinstance(error, ValueError)

    def test_preserves_constraint_details(self) -> None:
        """Error preserves constraint IDs."""
        violations = [
            _make_core_constraint_result(constraint_id="MIN_TRACE_WIDTH"),
            _make_core_constraint_result(constraint_id="MIN_SPACING"),
        ]
        error = ConstraintViolation(violations)
        msg = str(error)
        assert "MIN_TRACE_WIDTH" in msg
        assert "MIN_SPACING" in msg

    def test_stores_violations_tuple(self) -> None:
        """ConstraintViolation stores violations as tuple."""
        violations = [_make_core_constraint_result(), _make_core_constraint_result()]
        error = ConstraintViolation(violations)
        assert isinstance(error.violations, tuple)
        assert len(error.violations) == 2


class TestConstraintViolationError:
    """Tests for ConstraintViolationError exception (from tiers module)."""

    def test_is_value_error(self) -> None:
        """ConstraintViolationError should inherit from ValueError."""
        violations = [_make_tiers_constraint_result()]
        error = ConstraintViolationError(violations)
        assert isinstance(error, ValueError)

    def test_different_from_core_violation(self) -> None:
        """ConstraintViolationError is distinct from ConstraintViolation."""
        core_violations = [_make_core_constraint_result()]
        tiers_violations = [_make_tiers_constraint_result()]
        error1 = ConstraintViolation(core_violations)
        error2 = ConstraintViolationError(tiers_violations)
        # Both are ValueErrors but different types
        assert type(error1) != type(error2)
        assert isinstance(error1, ValueError)
        assert isinstance(error2, ValueError)

    def test_stores_tier(self) -> None:
        """ConstraintViolationError can store tier information."""
        violations = [_make_tiers_constraint_result()]
        error = ConstraintViolationError(violations, tier="T0")
        assert error.tier == "T0"


# =============================================================================
# Validation Edge Case Tests
# =============================================================================


class TestValidationBoundaryConditions:
    """Tests for boundary conditions in validation."""

    def test_empty_string_in_errors_list(self) -> None:
        """Empty string in errors list should work."""
        error = StrictValidationError([""])
        assert "0 error(s)" not in str(error)  # Should show 1 error

    def test_unicode_in_error_message(self) -> None:
        """Unicode characters in error message should work."""
        error = FamilyValidationError("F0_CAL_THRU_LINE", None, "Famille_Française_🔧")
        assert "Famille_Française_🔧" in str(error)

    def test_multiline_reason(self) -> None:
        """Multiline reason messages should be preserved."""
        message = "Validation failed:\n  - Field A is invalid\n  - Field B is missing"
        error = FamilyValidationError("F1_SINGLE_ENDED_VIA", "config", message)
        assert "Field A" in str(error) or "Validation" in str(error)

    def test_error_with_none_like_string(self) -> None:
        """Error message with 'None' string should work."""
        error = StrictValidationError(["Field 'value' is None"])
        assert "None" in str(error)

    def test_multiple_errors_in_strict_validation(self) -> None:
        """StrictValidationError handles multiple errors."""
        errors = ["Error 1", "Error 2", "Error 3"]
        error = StrictValidationError(errors)
        assert "3 error(s)" in str(error)


class TestErrorChaining:
    """Tests for exception chaining patterns."""

    def test_chained_exception_preserves_cause(self) -> None:
        """Chained exceptions preserve the original cause."""
        original = ValueError("Original error")
        try:
            try:
                raise original
            except ValueError as e:
                raise StrictValidationError(["Validation failed"]) from e
        except StrictValidationError as e:
            assert e.__cause__ is original

    def test_constraint_errors_can_be_chained(self) -> None:
        """Constraint errors can be chained from lower-level errors."""
        violations = [_make_core_constraint_result()]
        try:
            try:
                raise KeyError("missing_field")
            except KeyError as e:
                raise ConstraintViolation(violations) from e
        except ConstraintViolation as e:
            assert isinstance(e.__cause__, KeyError)


class TestErrorCategories:
    """Tests for error categorization patterns."""

    def test_family_errors_are_value_errors(self) -> None:
        """Family validation errors can be caught as ValueError."""
        try:
            raise FamilyValidationError("F0_CAL_THRU_LINE", None, "Invalid family type")
        except ValueError as e:
            assert "Invalid family" in str(e) or "F0" in str(e)

    def test_constraint_errors_are_value_errors(self) -> None:
        """All constraint errors can be caught as ValueError."""
        core_violations = [_make_core_constraint_result()]
        tiers_violations = [_make_tiers_constraint_result()]
        errors = [
            ConstraintViolation(core_violations),
            ConstraintViolationError(tiers_violations),
        ]
        for error in errors:
            try:
                raise error
            except ValueError:
                pass  # Should be caught

    def test_layer_errors_not_value_errors(self) -> None:
        """Layer validation errors are not ValueErrors."""
        result = _make_layer_validation_result()
        error = LayerSetValidationError(result)
        assert not isinstance(error, ValueError)
        assert isinstance(error, Exception)


# =============================================================================
# Error Message Formatting Tests
# =============================================================================


class TestErrorMessageFormatting:
    """Tests for error message formatting consistency."""

    def test_strict_validation_message_format(self) -> None:
        """StrictValidationError messages should be clear."""
        error = StrictValidationError(["Required field 'trace_width_nm' is missing"])
        msg = str(error)
        assert "Required" in msg or "required" in msg
        assert "trace_width_nm" in msg

    def test_family_validation_with_field(self) -> None:
        """Family validation error includes field context."""
        error = FamilyValidationError("F1_SINGLE_ENDED_VIA", "discontinuity", "Required for F1 family")
        msg = str(error)
        assert "F1_SINGLE_ENDED_VIA" in msg
        assert "discontinuity" in msg

    def test_layer_validation_identifies_problem_layer(self) -> None:
        """Layer validation error identifies the problem layers."""
        result = _make_layer_validation_result(missing_layers=("In3.Cu",))
        error = LayerSetValidationError(result)
        msg = str(error)
        assert "In3.Cu" in msg or "missing" in msg

    def test_constraint_violation_shows_constraint_ids(self) -> None:
        """Constraint violation should show constraint IDs."""
        violations = [
            _make_core_constraint_result(constraint_id="TRACE_WIDTH"),
            _make_core_constraint_result(constraint_id="SPACING"),
        ]
        error = ConstraintViolation(violations)
        msg = str(error)
        assert "TRACE_WIDTH" in msg
        assert "SPACING" in msg


# =============================================================================
# Exception Hierarchy Tests
# =============================================================================


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_all_errors_are_exceptions(self) -> None:
        """All custom errors inherit from Exception."""
        core_violations = [_make_core_constraint_result()]
        tiers_violations = [_make_tiers_constraint_result()]
        result = _make_layer_validation_result()
        errors = [
            StrictValidationError(["test"]),
            FamilyValidationError("F0", None, "test"),
            LayerSetValidationError(result),
            ConstraintViolation(core_violations),
            ConstraintViolationError(tiers_violations),
        ]
        for error in errors:
            assert isinstance(error, Exception)

    def test_value_error_subclasses(self) -> None:
        """Identify which errors are ValueError subclasses."""
        core_violations = [_make_core_constraint_result()]
        tiers_violations = [_make_tiers_constraint_result()]
        # Create errors with proper constructor arguments
        family_error = FamilyValidationError("F0", None, "test")
        constraint_violation = ConstraintViolation(core_violations)
        constraint_violation_error = ConstraintViolationError(tiers_violations)

        assert isinstance(family_error, ValueError), "FamilyValidationError should be ValueError"
        assert isinstance(constraint_violation, ValueError), "ConstraintViolation should be ValueError"
        assert isinstance(constraint_violation_error, ValueError), "ConstraintViolationError should be ValueError"

    def test_non_value_error_types(self) -> None:
        """Identify which errors are NOT ValueError subclasses."""
        strict_error = StrictValidationError(["test"])
        result = _make_layer_validation_result()
        layer_error = LayerSetValidationError(result)

        assert not isinstance(strict_error, ValueError), "StrictValidationError should not be ValueError"
        assert not isinstance(layer_error, ValueError), "LayerSetValidationError should not be ValueError"


# =============================================================================
# Repr and Str Tests
# =============================================================================


class TestErrorReprAndStr:
    """Tests for error __repr__ and __str__ methods."""

    def test_str_returns_message_content(self) -> None:
        """__str__ returns the error message containing the content."""
        message = "This is the error message"
        error = StrictValidationError([message])
        assert message in str(error)

    def test_repr_is_valid(self) -> None:
        """__repr__ returns a valid representation."""
        error = FamilyValidationError("F0_CAL_THRU_LINE", None, "test message")
        repr_str = repr(error)
        assert "FamilyValidationError" in repr_str

    def test_errors_can_be_logged(self) -> None:
        """Errors can be converted to strings for logging."""
        violations = [_make_core_constraint_result()]
        result = _make_layer_validation_result()
        errors = [
            StrictValidationError(["strict error"]),
            FamilyValidationError("F0", None, "family error"),
            LayerSetValidationError(result),
            ConstraintViolation(violations),
        ]
        for error in errors:
            # Should not raise when converting to string
            log_message = f"Error occurred: {error}"
            assert len(log_message) > 0


# =============================================================================
# Integration with Exception Handling Patterns
# =============================================================================


class TestExceptionHandlingPatterns:
    """Tests for common exception handling patterns."""

    def test_multiple_exception_catch(self) -> None:
        """Multiple exception types can be caught together."""
        core_violations = [_make_core_constraint_result()]
        tiers_violations = [_make_tiers_constraint_result()]
        errors_to_test = [
            FamilyValidationError("F0", None, "family"),
            ConstraintViolation(core_violations),
            ConstraintViolationError(tiers_violations),
        ]

        for error in errors_to_test:
            try:
                raise error
            except (FamilyValidationError, ConstraintViolation, ConstraintViolationError):
                pass  # All should be caught

    def test_base_class_catch_pattern(self) -> None:
        """ValueError catches all value error subclasses."""
        core_violations = [_make_core_constraint_result()]
        tiers_violations = [_make_tiers_constraint_result()]
        errors = [
            FamilyValidationError("F0", None, "family"),
            ConstraintViolation(core_violations),
            ConstraintViolationError(tiers_violations),
        ]

        for error in errors:
            try:
                raise error
            except ValueError:
                pass  # Should be caught by base class

    def test_exception_context_in_reraise(self) -> None:
        """Exception context is preserved when re-raising."""
        try:
            try:
                raise KeyError("original")
            except KeyError:
                raise StrictValidationError(["wrapped"]) from None
        except StrictValidationError as e:
            # __cause__ is None due to 'from None', but __context__ may be set
            assert e.__cause__ is None


# =============================================================================
# Error Recovery Pattern Tests
# =============================================================================


class TestErrorRecoveryPatterns:
    """Tests for error recovery patterns using these exceptions."""

    def test_validation_errors_allow_retry(self) -> None:
        """Validation errors can be caught and allow retry logic."""
        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            try:
                attempts += 1
                if attempts < max_attempts:
                    raise StrictValidationError([f"Attempt {attempts} failed"])
                # Success on final attempt
                break
            except StrictValidationError:
                if attempts >= max_attempts:
                    raise
                continue

        assert attempts == max_attempts

    def test_constraint_violation_can_trigger_repair(self) -> None:
        """Constraint violations can trigger repair logic."""
        repaired = False
        violations = [_make_core_constraint_result(constraint_id="SPACING")]

        try:
            raise ConstraintViolation(violations)
        except ConstraintViolation:
            # In real code, this would trigger repair
            repaired = True

        assert repaired
