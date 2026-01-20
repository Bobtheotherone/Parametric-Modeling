"""Coupon family builders module.

This module provides family-specific coupon builders that implement
the feature composition pattern described in the design document.

Each builder composes features (board outline, ports, launch regions,
transmission line, discontinuity) into a complete coupon design.
"""

from __future__ import annotations

from .f0_builder import F0CouponBuilder, build_f0_coupon

__all__ = [
    "F0CouponBuilder",
    "build_f0_coupon",
]
