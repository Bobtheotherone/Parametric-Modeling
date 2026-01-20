"""Coupon family builders module.

This module provides family-specific coupon builders that implement
the feature composition pattern described in the design document.

Each builder composes features (board outline, ports, launch regions,
transmission line, discontinuity) into a complete coupon design.

Supported families:
- F0: Calibration Thru Line (REQ-M1-006)
- F1: Single-Ended Via Transition (REQ-M1-007)
"""

from __future__ import annotations

from .f0_builder import F0CouponBuilder, build_f0_coupon
from .f1_builder import F1CouponBuilder, build_f1_coupon

__all__ = [
    "F0CouponBuilder",
    "F1CouponBuilder",
    "build_f0_coupon",
    "build_f1_coupon",
]
