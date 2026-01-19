from __future__ import annotations

from .spec import CouponSpec

FAMILY_F0 = "F0_CAL_THRU_LINE"
FAMILY_F1 = "F1_SINGLE_ENDED_VIA"

SUPPORTED_FAMILIES = (FAMILY_F0, FAMILY_F1)


def validate_family(spec: CouponSpec) -> None:
    if spec.coupon_family == FAMILY_F0:
        if spec.discontinuity is not None:
            raise ValueError("F0_CAL_THRU_LINE does not allow a discontinuity block.")
        return
    if spec.coupon_family == FAMILY_F1:
        if spec.discontinuity is None:
            raise ValueError("F1_SINGLE_ENDED_VIA requires a discontinuity block.")
        if spec.discontinuity.type != "VIA_TRANSITION":
            raise ValueError("F1_SINGLE_ENDED_VIA requires discontinuity.type=VIA_TRANSITION.")
        return
    raise ValueError(f"Unsupported coupon_family: {spec.coupon_family!r}")
