"""Derived feature and dimensionless group computation for equation discovery.

This module provides functions to compute physics-relevant dimensionless groups
and derived features for RF/microwave transmission line analysis. These groups
are essential for symbolic regression and equation discovery in M1.

Satisfies:
    - REQ-M1-015: Derived groups include CPWG/via/fence/launch-relevant
                  dimensionless groups and emit deterministically in manifest.json.
    - REQ-M1-014: Feature coverage for equation discovery.

The dimensionless groups are designed to capture the physical relationships
in coplanar waveguide with ground (CPWG) structures, including:
    - CPWG geometry ratios (gap/width, substrate height/width)
    - Via geometry ratios (drill/diameter, spacing/diameter)
    - Fence geometry ratios (pitch/wavelength, offset/gap)
    - Launch geometry ratios (connector/trace matching)

All group keys are sorted alphabetically for deterministic JSON emission.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from formula_foundry.coupongen.spec import CouponSpec


def compute_derived_features(spec: CouponSpec, length_right_nm: int | None = None) -> dict[str, int]:
    """Compute derived features from a CouponSpec.

    Derived features are integer-valued quantities computed from the spec
    parameters. They capture secondary geometric relationships that may be
    useful for equation discovery.

    Args:
        spec: The coupon specification.
        length_right_nm: Optional derived right length for F1 coupons.

    Returns:
        Dictionary of derived features with integer values.
        Keys are sorted alphabetically.
    """
    features: dict[str, int] = {}

    # Board derived features
    width_nm = int(spec.board.outline.width_nm)
    length_nm = int(spec.board.outline.length_nm)
    features["board_area_nm2"] = width_nm * length_nm
    features["board_perimeter_nm"] = 2 * (width_nm + length_nm)

    # Transmission line derived features
    w_nm = int(spec.transmission_line.w_nm)
    gap_nm = int(spec.transmission_line.gap_nm)
    length_left_nm = int(spec.transmission_line.length_left_nm)

    # Total ground opening = w + 2*gap (distance between ground planes)
    features["cpwg_ground_opening_nm"] = w_nm + 2 * gap_nm

    # Edge-to-edge distance of coplanar ground (footprint width)
    features["cpwg_footprint_width_nm"] = w_nm + 2 * gap_nm

    # Trace total length
    if length_right_nm is not None:
        trace_total = length_left_nm + length_right_nm
    elif spec.transmission_line.length_right_nm is not None:
        trace_total = length_left_nm + int(spec.transmission_line.length_right_nm)
    else:
        trace_total = length_left_nm
    features["trace_total_length_nm"] = trace_total

    # For F1 coupons, store the derived length_right_nm
    if length_right_nm is not None:
        features["length_right_nm"] = length_right_nm

    # Ground via fence derived features
    if spec.transmission_line.ground_via_fence is not None:
        fence = spec.transmission_line.ground_via_fence
        if fence.enabled:
            pitch_nm = int(fence.pitch_nm)
            offset_nm = int(fence.offset_from_gap_nm)
            via_drill = int(fence.via.drill_nm)
            via_diam = int(fence.via.diameter_nm)

            # Total offset from trace center to via center
            features["fence_via_offset_from_center_nm"] = w_nm // 2 + gap_nm + offset_nm

            # Annular ring of fence vias
            features["fence_via_annular_ring_nm"] = via_diam - via_drill

            # Number of vias along trace (approximation)
            if pitch_nm > 0:
                features["fence_via_count_per_side"] = max(1, trace_total // pitch_nm)

    # Discontinuity (signal via) derived features
    if spec.discontinuity is not None:
        signal_via = spec.discontinuity.signal_via
        pad_nm = int(signal_via.pad_diameter_nm)
        drill_nm = int(signal_via.drill_nm)
        via_diam = int(signal_via.diameter_nm)

        # Annular ring dimensions
        features["signal_via_annular_ring_nm"] = pad_nm - drill_nm
        features["signal_via_barrel_annular_nm"] = via_diam - drill_nm

        # Return vias derived features
        if spec.discontinuity.return_vias is not None:
            return_vias = spec.discontinuity.return_vias
            features["return_via_count"] = return_vias.count
            features["return_via_radius_nm"] = int(return_vias.radius_nm)

            ret_via = return_vias.via
            features["return_via_annular_ring_nm"] = int(ret_via.diameter_nm) - int(ret_via.drill_nm)

    # Connector derived features (launch geometry)
    left_x = int(spec.connectors.left.position_nm[0])
    right_x = int(spec.connectors.right.position_nm[0])
    features["connector_span_nm"] = abs(right_x - left_x)

    # Stackup derived features
    if "core" in spec.stackup.thicknesses_nm:
        core_nm = int(spec.stackup.thicknesses_nm["core"])
        features["stackup_core_thickness_nm"] = core_nm
    if "prepreg" in spec.stackup.thicknesses_nm:
        prepreg_nm = int(spec.stackup.thicknesses_nm["prepreg"])
        features["stackup_prepreg_thickness_nm"] = prepreg_nm

    # Sort keys for deterministic output
    return dict(sorted(features.items()))


def compute_dimensionless_groups(spec: CouponSpec) -> dict[str, float]:
    """Compute physics-relevant dimensionless groups for equation discovery.

    Dimensionless groups capture physical relationships that are scale-invariant
    and useful for symbolic regression. This function computes groups relevant
    to CPWG, via, fence, and launch structures.

    The groups are organized into categories:
    1. CPWG geometry groups (gap/width, substrate ratios)
    2. Via geometry groups (drill/diameter, pad ratios)
    3. Fence geometry groups (pitch/gap, offset ratios)
    4. Launch geometry groups (connector matching)
    5. Stackup geometry groups (layer thickness ratios)

    Args:
        spec: The coupon specification.

    Returns:
        Dictionary of dimensionless groups with float values.
        Keys are sorted alphabetically for deterministic JSON emission.
    """
    from formula_foundry.resolve.derived_groups import (
        compute_cpwg_groups,
        compute_fence_groups,
        compute_launch_groups,
        compute_stackup_groups,
        compute_via_groups,
    )

    groups: dict[str, float] = {}

    # CPWG geometry groups
    groups.update(compute_cpwg_groups(spec))

    # Via geometry groups
    groups.update(compute_via_groups(spec))

    # Fence geometry groups
    groups.update(compute_fence_groups(spec))

    # Launch geometry groups
    groups.update(compute_launch_groups(spec))

    # Stackup geometry groups
    groups.update(compute_stackup_groups(spec))

    # Sort keys for deterministic output
    return dict(sorted(groups.items()))


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Compute a ratio safely, returning 0.0 if denominator is zero.

    Args:
        numerator: The numerator value.
        denominator: The denominator value.

    Returns:
        The ratio, or 0.0 if the denominator is zero.
    """
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
