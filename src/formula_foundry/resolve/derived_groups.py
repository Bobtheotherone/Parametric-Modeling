"""Derived dimensionless groups for CPWG/via/fence/launch structures.

This module provides specialized functions to compute physics-relevant
dimensionless groups for each structural element in RF/microwave transmission
line coupons. These groups are essential inputs for symbolic regression and
equation discovery.

Satisfies:
    - REQ-M1-015: Derived groups include CPWG/via/fence/launch-relevant
                  dimensionless groups and emit deterministically in manifest.json.

Group naming conventions:
    - `cpwg_*`: Coplanar waveguide with ground geometry groups
    - `via_*`: Signal and return via geometry groups
    - `fence_*`: Ground via fence geometry groups
    - `launch_*`: Connector launch geometry groups
    - `stackup_*`: Layer stackup geometry groups

All groups are dimensionless ratios designed to capture scale-invariant
physical relationships for impedance, coupling, and field distribution.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from formula_foundry.coupongen.spec import CouponSpec


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Compute a ratio safely, returning 0.0 if denominator is zero."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def compute_cpwg_groups(spec: CouponSpec) -> dict[str, float]:
    """Compute CPWG (Coplanar Waveguide with Ground) geometry groups.

    These groups capture the key geometric relationships that determine
    CPWG characteristic impedance and wave propagation:

    - `cpwg_gap_over_w`: Gap to trace width ratio (s/w). Primary impedance driver.
    - `cpwg_w_over_gap`: Trace width to gap ratio (w/s). Inverse of above.
    - `cpwg_ground_opening_over_h`: Ground opening (w+2s) to substrate height ratio.
    - `cpwg_w_over_h`: Trace width to substrate height ratio.
    - `cpwg_gap_over_h`: Gap to substrate height ratio.
    - `cpwg_aspect_ratio`: Total ground opening to board length ratio.

    For CPWG, the effective dielectric constant and impedance depend on:
        Z0 ~ 1/sqrt(eps_eff) * f(s/w, h/(w+2s))

    Args:
        spec: The coupon specification.

    Returns:
        Dictionary of CPWG dimensionless groups.
    """
    groups: dict[str, float] = {}

    w_nm = int(spec.transmission_line.w_nm)
    gap_nm = int(spec.transmission_line.gap_nm)

    # Primary CPWG ratios
    groups["cpwg_gap_over_w"] = _safe_ratio(gap_nm, w_nm)
    groups["cpwg_w_over_gap"] = _safe_ratio(w_nm, gap_nm)

    # Ground opening (distance between ground planes) = w + 2*gap
    ground_opening_nm = w_nm + 2 * gap_nm
    groups["cpwg_ground_opening_nm"] = float(ground_opening_nm)  # Not dimensionless but useful

    # Board geometry ratios
    board_length = int(spec.board.outline.length_nm)
    board_width = int(spec.board.outline.width_nm)
    groups["board_aspect_ratio"] = _safe_ratio(board_length, board_width)

    # Ratios to substrate/core height (if available)
    if "core" in spec.stackup.thicknesses_nm:
        h_nm = int(spec.stackup.thicknesses_nm["core"])
        groups["cpwg_w_over_h"] = _safe_ratio(w_nm, h_nm)
        groups["cpwg_gap_over_h"] = _safe_ratio(gap_nm, h_nm)
        groups["cpwg_ground_opening_over_h"] = _safe_ratio(ground_opening_nm, h_nm)
        # Effective width parameter a = w + 2*gap, aspect ratio a/h
        groups["cpwg_a_over_h"] = _safe_ratio(ground_opening_nm, h_nm)

    # For grounded CPWG (conductor-backed), the substrate height is critical
    # Add the 2h/a ratio which appears in conformal mapping solutions
    if "core" in spec.stackup.thicknesses_nm:
        h_nm = int(spec.stackup.thicknesses_nm["core"])
        groups["cpwg_2h_over_a"] = _safe_ratio(2 * h_nm, ground_opening_nm)

    return groups


def compute_via_groups(spec: CouponSpec) -> dict[str, float]:
    """Compute via geometry groups for signal and return vias.

    These groups capture via characteristics relevant to:
    - Via inductance and capacitance (drill/pad ratios)
    - Return current path (return via radius/signal via size)
    - Impedance discontinuity (via pad to trace width)

    Key groups:
    - `via_drill_over_pad`: Drill to pad diameter ratio (affects capacitance).
    - `via_pad_over_drill`: Pad to drill ratio (inverse, affects annular ring).
    - `via_pad_over_trace_w`: Via pad to trace width ratio (impedance match).
    - `via_return_radius_over_pad`: Return via radius to signal pad ratio.
    - `via_return_count_normalized`: Return via count over expected minimum.

    Args:
        spec: The coupon specification.

    Returns:
        Dictionary of via dimensionless groups.
    """
    groups: dict[str, float] = {}

    if spec.discontinuity is None:
        return groups

    signal_via = spec.discontinuity.signal_via
    drill_nm = int(signal_via.drill_nm)
    diameter_nm = int(signal_via.diameter_nm)
    pad_nm = int(signal_via.pad_diameter_nm)
    w_nm = int(spec.transmission_line.w_nm)

    # Signal via ratios
    groups["via_drill_over_pad"] = _safe_ratio(drill_nm, pad_nm)
    groups["via_pad_over_drill"] = _safe_ratio(pad_nm, drill_nm)
    groups["via_diameter_over_drill"] = _safe_ratio(diameter_nm, drill_nm)

    # Trace matching ratios
    groups["via_pad_over_trace_w"] = _safe_ratio(pad_nm, w_nm)
    groups["via_drill_over_trace_w"] = _safe_ratio(drill_nm, w_nm)

    # Annular ring ratio (pad - drill) / pad
    annular_ring = pad_nm - drill_nm
    groups["via_annular_ring_ratio"] = _safe_ratio(annular_ring, pad_nm)

    # Return vias
    if spec.discontinuity.return_vias is not None:
        return_vias = spec.discontinuity.return_vias
        radius_nm = int(return_vias.radius_nm)
        count = return_vias.count

        ret_via = return_vias.via
        ret_drill = int(ret_via.drill_nm)
        ret_diam = int(ret_via.diameter_nm)

        # Return via to signal via ratios
        groups["via_return_radius_over_pad"] = _safe_ratio(radius_nm, pad_nm)
        groups["via_return_radius_over_drill"] = _safe_ratio(radius_nm, drill_nm)

        # Return via spacing along circumference
        if count > 0:
            circumference = 2 * math.pi * radius_nm
            via_spacing = circumference / count
            groups["via_return_spacing_over_diameter"] = _safe_ratio(via_spacing, ret_diam)

            # Normalized count: actual count / (circumference / via diameter)
            # Values > 1 means vias are closely spaced
            min_count = _safe_ratio(circumference, ret_diam) if ret_diam > 0 else 0
            groups["via_return_count_fill_ratio"] = _safe_ratio(count, min_count) if min_count > 0 else 0.0

        # Return via geometry
        groups["via_return_drill_over_diam"] = _safe_ratio(ret_drill, ret_diam)

    return groups


def compute_fence_groups(spec: CouponSpec) -> dict[str, float]:
    """Compute ground via fence geometry groups.

    The via fence provides a ground boundary for the CPWG to control
    the electromagnetic field and suppress parallel plate modes. Key groups:

    - `fence_pitch_over_gap`: Pitch to CPWG gap ratio (mode control).
    - `fence_offset_over_gap`: Via offset from gap edge to gap ratio.
    - `fence_offset_over_w`: Via offset to trace width ratio.
    - `fence_pitch_over_h`: Pitch to substrate height (wavelength proxy).
    - `fence_via_drill_over_pitch`: Via drill to pitch ratio (ground continuity).

    For effective mode suppression, pitch should be < lambda/4 at max frequency.

    Args:
        spec: The coupon specification.

    Returns:
        Dictionary of fence dimensionless groups.
    """
    groups: dict[str, float] = {}

    fence = spec.transmission_line.ground_via_fence
    if fence is None or not fence.enabled:
        return groups

    pitch_nm = int(fence.pitch_nm)
    offset_nm = int(fence.offset_from_gap_nm)
    via_drill = int(fence.via.drill_nm)
    via_diam = int(fence.via.diameter_nm)

    w_nm = int(spec.transmission_line.w_nm)
    gap_nm = int(spec.transmission_line.gap_nm)

    # Pitch ratios
    groups["fence_pitch_over_gap"] = _safe_ratio(pitch_nm, gap_nm)
    groups["fence_pitch_over_w"] = _safe_ratio(pitch_nm, w_nm)

    # Offset ratios
    groups["fence_offset_over_gap"] = _safe_ratio(offset_nm, gap_nm)
    groups["fence_offset_over_w"] = _safe_ratio(offset_nm, w_nm)

    # Total offset from trace center = w/2 + gap + offset
    total_offset = w_nm // 2 + gap_nm + offset_nm
    groups["fence_total_offset_over_w"] = _safe_ratio(total_offset, w_nm)

    # Via geometry ratios
    groups["fence_via_drill_over_pitch"] = _safe_ratio(via_drill, pitch_nm)
    groups["fence_via_diam_over_pitch"] = _safe_ratio(via_diam, pitch_nm)
    groups["fence_via_drill_over_diam"] = _safe_ratio(via_drill, via_diam)

    # Pitch to substrate height ratio (relevant for mode suppression)
    if "core" in spec.stackup.thicknesses_nm:
        h_nm = int(spec.stackup.thicknesses_nm["core"])
        groups["fence_pitch_over_h"] = _safe_ratio(pitch_nm, h_nm)
        groups["fence_offset_over_h"] = _safe_ratio(offset_nm, h_nm)

    return groups


def compute_launch_groups(spec: CouponSpec) -> dict[str, float]:
    """Compute launch/connector geometry groups.

    The launch region is where the connector transitions to the transmission
    line. These groups capture the geometric matching:

    - `launch_connector_span_over_trace_length`: Overall layout efficiency.
    - `launch_left_offset_over_w`: Left connector offset normalized.
    - `launch_symmetry`: Symmetry of connector placement.

    Args:
        spec: The coupon specification.

    Returns:
        Dictionary of launch dimensionless groups.
    """
    groups: dict[str, float] = {}

    left_pos = spec.connectors.left.position_nm
    right_pos = spec.connectors.right.position_nm

    left_x = int(left_pos[0])
    left_y = int(left_pos[1])
    right_x = int(right_pos[0])
    right_y = int(right_pos[1])

    connector_span_x = abs(right_x - left_x)
    connector_span_y = abs(right_y - left_y)

    board_length = int(spec.board.outline.length_nm)
    board_width = int(spec.board.outline.width_nm)
    w_nm = int(spec.transmission_line.w_nm)

    # Connector span ratios
    groups["launch_span_over_board_length"] = _safe_ratio(connector_span_x, board_length)
    groups["launch_span_over_trace_w"] = _safe_ratio(connector_span_x, w_nm)

    # Y-axis alignment (should be 0 for centered design)
    board_center_y = board_width // 2
    left_offset_y = abs(left_y - board_center_y)
    right_offset_y = abs(right_y - board_center_y)

    groups["launch_left_y_offset_ratio"] = _safe_ratio(left_offset_y, board_width)
    groups["launch_right_y_offset_ratio"] = _safe_ratio(right_offset_y, board_width)

    # X-axis placement relative to board edges
    groups["launch_left_x_ratio"] = _safe_ratio(left_x, board_length)
    groups["launch_right_x_ratio"] = _safe_ratio(right_x, board_length)

    # Symmetry measure (difference in edge distances)
    left_edge_dist = left_x
    right_edge_dist = board_length - right_x
    edge_diff = abs(left_edge_dist - right_edge_dist)
    groups["launch_edge_symmetry"] = _safe_ratio(edge_diff, board_length)

    # Trace length to connector span ratio
    length_left = int(spec.transmission_line.length_left_nm)
    if spec.transmission_line.length_right_nm is not None:
        length_right = int(spec.transmission_line.length_right_nm)
        total_trace = length_left + length_right
    else:
        total_trace = length_left

    groups["launch_trace_over_span"] = _safe_ratio(total_trace, connector_span_x)

    return groups


def compute_stackup_groups(spec: CouponSpec) -> dict[str, float]:
    """Compute stackup/layer geometry groups.

    These groups capture the dielectric layer relationships:

    - `stackup_copper_over_core`: Copper thickness to core ratio.
    - `stackup_prepreg_over_core`: Prepreg to core thickness ratio.
    - `stackup_total_over_core`: Total thickness to core ratio.

    Material groups:
    - `stackup_er`: Relative permittivity (not dimensionless but critical).
    - `stackup_loss_tangent`: Dielectric loss tangent.

    Args:
        spec: The coupon specification.

    Returns:
        Dictionary of stackup dimensionless groups.
    """
    groups: dict[str, float] = {}

    # Material properties (not strictly dimensionless but essential)
    groups["stackup_er"] = float(spec.stackup.materials.er)
    groups["stackup_loss_tangent"] = float(spec.stackup.materials.loss_tangent)

    # Derive sqrt(er) as it appears in impedance equations
    if spec.stackup.materials.er > 0:
        groups["stackup_sqrt_er"] = math.sqrt(spec.stackup.materials.er)
    else:
        groups["stackup_sqrt_er"] = 0.0

    thicknesses = spec.stackup.thicknesses_nm

    # Core thickness is typically the reference
    core_nm = int(thicknesses.get("core", 0))

    # Collect all thickness values for ratio calculations
    if core_nm > 0:
        for layer_name, thickness in thicknesses.items():
            if layer_name != "core":
                key = f"stackup_{layer_name}_over_core"
                groups[key] = _safe_ratio(int(thickness), core_nm)

    # If we have both copper and core, compute additional ratios
    if "copper" in thicknesses and core_nm > 0:
        copper_nm = int(thicknesses["copper"])
        groups["stackup_copper_over_core"] = _safe_ratio(copper_nm, core_nm)

        # Copper to trace width ratio
        w_nm = int(spec.transmission_line.w_nm)
        groups["stackup_copper_over_w"] = _safe_ratio(copper_nm, w_nm)

    # Total stackup thickness estimate
    total_thickness = sum(int(t) for t in thicknesses.values())
    if core_nm > 0:
        groups["stackup_total_over_core"] = _safe_ratio(total_thickness, core_nm)

    return groups
