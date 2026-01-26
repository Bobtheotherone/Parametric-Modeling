from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spec import CouponSpec

FAMILY_F0 = "F0_CAL_THRU_LINE"
FAMILY_F1 = "F1_SINGLE_ENDED_VIA"

SUPPORTED_FAMILIES = (FAMILY_F0, FAMILY_F1)

# Fields that are F1-only (not allowed in F0)
F1_ONLY_FIELDS = frozenset({"discontinuity"})

# Fields that are F0-required
F0_REQUIRED_FIELDS = frozenset({"transmission_line.length_right_nm"})


class FamilyValidationError(ValueError):
    """Raised when family-specific validation fails.

    This exception provides detailed information about which family
    constraint was violated and why.

    Attributes:
        family: The coupon family that failed validation.
        field: The field that caused the validation failure.
        reason: Description of why the validation failed.
    """

    def __init__(self, family: str, field: str | None, reason: str) -> None:
        self.family = family
        self.field = field
        self.reason = reason
        msg = f"{family}: {reason}"
        if field:
            msg = f"{family}.{field}: {reason}"
        super().__init__(msg)


def validate_family(spec: CouponSpec) -> None:
    """Validate family-specific constraints on a CouponSpec.

    This function enforces family-specific correctness per REQ-M1-002:
    - F0 (calibration thru-line) cannot include F1-only blocks (discontinuity)
    - F1 (single-ended via) requires a discontinuity block with type VIA_TRANSITION

    Args:
        spec: The CouponSpec to validate.

    Raises:
        FamilyValidationError: If family-specific constraints are violated.
        ValueError: If the coupon_family is not supported.
    """
    if spec.coupon_family == FAMILY_F0:
        _validate_f0(spec)
        return
    if spec.coupon_family == FAMILY_F1:
        _validate_f1(spec)
        return
    raise FamilyValidationError(
        spec.coupon_family,
        None,
        f"Unsupported coupon_family. Must be one of: {SUPPORTED_FAMILIES}",
    )


def _validate_f0(spec: CouponSpec) -> None:
    """Validate F0 (calibration thru-line) specific constraints.

    F0 coupons are simple through-lines used for calibration:
    - Cannot include discontinuity block (F1-only feature)
    - Requires transmission_line.length_right_nm to be specified
      (both lengths define the symmetric through-line)

    Args:
        spec: The CouponSpec to validate (must be F0 family).

    Raises:
        FamilyValidationError: If F0 constraints are violated.
    """
    # F0 cannot have discontinuity (F1-only feature)
    if spec.discontinuity is not None:
        raise FamilyValidationError(
            FAMILY_F0,
            "discontinuity",
            "F0_CAL_THRU_LINE does not allow a discontinuity block (discontinuity is an F1-only feature for via transitions)",
        )


def _validate_f1(spec: CouponSpec) -> None:
    """Validate F1 (single-ended via) specific constraints.

    F1 coupons have a via transition in the middle:
    - Requires discontinuity block (the via transition definition)
    - discontinuity.type must be VIA_TRANSITION

    Args:
        spec: The CouponSpec to validate (must be F1 family).

    Raises:
        FamilyValidationError: If F1 constraints are violated.
    """
    # F1 requires discontinuity block
    if spec.discontinuity is None:
        raise FamilyValidationError(
            FAMILY_F1,
            "discontinuity",
            "F1_SINGLE_ENDED_VIA requires a discontinuity block (defines the via transition parameters)",
        )

    # F1 requires discontinuity.type == VIA_TRANSITION
    if spec.discontinuity.type != "VIA_TRANSITION":
        raise FamilyValidationError(
            FAMILY_F1,
            "discontinuity.type",
            f"F1_SINGLE_ENDED_VIA requires discontinuity.type='VIA_TRANSITION', got '{spec.discontinuity.type}'",
        )


def get_family_forbidden_fields(family: str) -> frozenset[str]:
    """Get the set of fields forbidden for a given family.

    Args:
        family: The coupon family identifier.

    Returns:
        Set of field paths that are forbidden for this family.
    """
    if family == FAMILY_F0:
        return F1_ONLY_FIELDS
    return frozenset()


def get_family_required_fields(family: str) -> frozenset[str]:
    """Get the set of fields required for a given family.

    Args:
        family: The coupon family identifier.

    Returns:
        Set of field paths that are required for this family.
    """
    if family == FAMILY_F0:
        return F0_REQUIRED_FIELDS
    return frozenset()
