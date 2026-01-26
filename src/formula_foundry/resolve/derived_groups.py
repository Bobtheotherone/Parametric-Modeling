"""Derived dimensionless groups for CPWG/via/fence/launch structures.

This module provides specialized functions to compute physics-relevant
dimensionless groups for each structural element in RF/microwave transmission
line coupons. These groups are essential inputs for symbolic regression and
equation discovery.

Satisfies:
    - REQ-M1-014: Derived features and dimensionless groups MUST be expanded to
                  include CPWG/via/fence/launch-relevant groups and MUST be emitted
                  deterministically in manifest.json.
    - REQ-M1-015: Key dimensional groups drive design hash sensitivity.

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


def _get_substrate_height(spec: CouponSpec) -> int | None:
    """Get the substrate height from stackup.

    Handles both legacy "core" naming and new "L1_to_L2" style naming.
    For CPWG on F.Cu (layer 1), the substrate height is typically the
    L1_to_L2 distance (distance to first reference ground plane).

    Args:
        spec: The coupon specification.

    Returns:
        Substrate height in nm, or None if not determinable.
    """
    thicknesses = spec.stackup.thicknesses_nm
    # Try legacy "core" key first
    if "core" in thicknesses:
        return int(thicknesses["core"])
    # For F.Cu (top layer), use L1_to_L2 as the substrate height
    # This is the distance from top copper to first internal ground plane
    if "L1_to_L2" in thicknesses:
        return int(thicknesses["L1_to_L2"])
    # Fallback: try to find any reasonable substrate thickness
    # (typically the thickest dielectric layer is the core)
    if thicknesses:
        return max(int(t) for t in thicknesses.values())
    return None


def compute_cpwg_groups(spec: CouponSpec) -> dict[str, float]:
    """Compute CPWG (Coplanar Waveguide with Ground) geometry groups.

    These groups capture the key geometric relationships that determine
    CPWG characteristic impedance and wave propagation:

    - `cpwg_gap_over_w`: Gap to trace width ratio (s/w). Primary impedance driver.
    - `cpwg_w_over_gap`: Trace width to gap ratio (w/s). Inverse of above.
    - `cpwg_ground_opening_over_h`: Ground opening (w+2s) to substrate height ratio.
    - `cpwg_w_over_h`: Trace width to substrate height ratio.
    - `cpwg_gap_over_h`: Gap to substrate height ratio.
    - `cpwg_k_ratio`: The k parameter (w/(w+2s)) used in elliptic integral formulas.
    - `cpwg_k_prime_ratio`: sqrt(1 - k^2), complementary k parameter.

    For CPWG, the effective dielectric constant and impedance depend on:
        Z0 ~ 1/sqrt(eps_eff) * f(s/w, h/(w+2s))

    The k-ratio is critical for conformal mapping solutions:
        k = w / (w + 2*s)
        Z0 = (30*pi/sqrt(eps_eff)) * K(k')/K(k)

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

    # The k-ratio is fundamental for CPWG impedance calculations
    # k = w / (w + 2*s), appears in complete elliptic integral of first kind
    k_ratio = _safe_ratio(w_nm, ground_opening_nm)
    groups["cpwg_k_ratio"] = k_ratio

    # Complementary k' = sqrt(1 - k^2), also used in elliptic integrals
    if k_ratio < 1.0:
        k_prime = math.sqrt(1.0 - k_ratio * k_ratio)
        groups["cpwg_k_prime_ratio"] = k_prime
    else:
        groups["cpwg_k_prime_ratio"] = 0.0

    # Board geometry ratios
    board_length = int(spec.board.outline.length_nm)
    board_width = int(spec.board.outline.width_nm)
    groups["board_aspect_ratio"] = _safe_ratio(board_length, board_width)

    # Ratios to substrate height using the helper function
    h_nm = _get_substrate_height(spec)
    if h_nm is not None and h_nm > 0:
        groups["cpwg_w_over_h"] = _safe_ratio(w_nm, h_nm)
        groups["cpwg_gap_over_h"] = _safe_ratio(gap_nm, h_nm)
        groups["cpwg_ground_opening_over_h"] = _safe_ratio(ground_opening_nm, h_nm)
        # Effective width parameter a = w + 2*gap, aspect ratio a/h
        groups["cpwg_a_over_h"] = _safe_ratio(ground_opening_nm, h_nm)
        # 2h/a ratio which appears in conformal mapping solutions
        groups["cpwg_2h_over_a"] = _safe_ratio(2 * h_nm, ground_opening_nm)
        # h/w ratio - substrate height to trace width
        groups["cpwg_h_over_w"] = _safe_ratio(h_nm, w_nm)

    return groups


def compute_via_groups(spec: CouponSpec) -> dict[str, float]:
    """Compute via geometry groups for signal and return vias.

    These groups capture via characteristics relevant to:
    - Via inductance and capacitance (drill/pad ratios)
    - Return current path (return via radius/signal via size)
    - Impedance discontinuity (via pad to trace width)
    - Via barrel length to diameter ratio (for inductance)

    Key groups:
    - `via_drill_over_pad`: Drill to pad diameter ratio (affects capacitance).
    - `via_pad_over_drill`: Pad to drill ratio (inverse, affects annular ring).
    - `via_pad_over_trace_w`: Via pad to trace width ratio (impedance match).
    - `via_return_radius_over_pad`: Return via radius to signal pad ratio.
    - `via_return_count_normalized`: Return via count over expected minimum.
    - `via_barrel_aspect`: Via barrel length to diameter ratio (inductance).

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
    gap_nm = int(spec.transmission_line.gap_nm)

    # Signal via ratios
    groups["via_drill_over_pad"] = _safe_ratio(drill_nm, pad_nm)
    groups["via_pad_over_drill"] = _safe_ratio(pad_nm, drill_nm)
    groups["via_diameter_over_drill"] = _safe_ratio(diameter_nm, drill_nm)

    # Trace matching ratios
    groups["via_pad_over_trace_w"] = _safe_ratio(pad_nm, w_nm)
    groups["via_drill_over_trace_w"] = _safe_ratio(drill_nm, w_nm)

    # Via pad to CPWG ground opening ratio (important for impedance matching)
    ground_opening_nm = w_nm + 2 * gap_nm
    groups["via_pad_over_cpwg_opening"] = _safe_ratio(pad_nm, ground_opening_nm)

    # Annular ring ratio (pad - drill) / pad
    annular_ring = pad_nm - drill_nm
    groups["via_annular_ring_ratio"] = _safe_ratio(annular_ring, pad_nm)

    # Via barrel aspect ratio (length/diameter) - important for via inductance
    # The barrel length is the total stackup thickness for through vias
    h_nm = _get_substrate_height(spec)
    if h_nm is not None and h_nm > 0:
        # For a 4-layer board, via goes through all layers
        # Approximate barrel length as substrate height
        groups["via_barrel_aspect"] = _safe_ratio(h_nm, diameter_nm)
        groups["via_drill_over_h"] = _safe_ratio(drill_nm, h_nm)

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
    h_nm = _get_substrate_height(spec)
    if h_nm is not None and h_nm > 0:
        groups["fence_pitch_over_h"] = _safe_ratio(pitch_nm, h_nm)
        groups["fence_offset_over_h"] = _safe_ratio(offset_nm, h_nm)
        # Fence via barrel aspect ratio
        groups["fence_via_barrel_aspect"] = _safe_ratio(h_nm, via_diam)

    return groups


def compute_launch_groups(spec: CouponSpec) -> dict[str, float]:
    """Compute launch/connector geometry groups.

    The launch region is where the connector transitions to the transmission
    line. These groups capture the geometric matching:

    - `launch_connector_span_over_trace_length`: Overall layout efficiency.
    - `launch_left_offset_over_w`: Left connector offset normalized.
    - `launch_symmetry`: Symmetry of connector placement.
    - `launch_length_left_over_span`: Left trace length normalized to span.
    - `launch_via_offset_ratio`: Discontinuity position along trace (F1 only).

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

    board_length = int(spec.board.outline.length_nm)
    board_width = int(spec.board.outline.width_nm)
    w_nm = int(spec.transmission_line.w_nm)
    gap_nm = int(spec.transmission_line.gap_nm)

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

    # Individual trace length ratios
    groups["launch_length_left_over_span"] = _safe_ratio(length_left, connector_span_x)

    # For F1 (via transition), the via/discontinuity position is critical
    # The via offset ratio is the position along the trace (0.5 = center)
    if spec.discontinuity is not None and total_trace > 0:
        # Via is at length_left from left connector
        via_offset_ratio = _safe_ratio(length_left, total_trace)
        groups["launch_via_offset_ratio"] = via_offset_ratio

        # Via pad clearance to CPWG ground opening
        pad_nm = int(spec.discontinuity.signal_via.pad_diameter_nm)
        ground_opening_nm = w_nm + 2 * gap_nm
        groups["launch_via_pad_over_cpwg"] = _safe_ratio(pad_nm, ground_opening_nm)

    return groups


def compute_stackup_groups(spec: CouponSpec) -> dict[str, float]:
    """Compute stackup/layer geometry groups.

    These groups capture the dielectric layer relationships:

    - `stackup_h_over_w`: Substrate height to trace width ratio (critical for Z0).
    - `stackup_total_over_h`: Total stackup to substrate height ratio.
    - `stackup_layer_ratio_*`: Ratios between different layer thicknesses.

    Material groups:
    - `stackup_er`: Relative permittivity (not dimensionless but critical).
    - `stackup_loss_tangent`: Dielectric loss tangent.
    - `stackup_sqrt_er`: Square root of er (appears in Z0 equations).

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
    w_nm = int(spec.transmission_line.w_nm)
    gap_nm = int(spec.transmission_line.gap_nm)

    # Get substrate height using the helper
    h_nm = _get_substrate_height(spec)
    if h_nm is not None and h_nm > 0:
        # Key ratio: substrate height to trace width
        groups["stackup_h_over_w"] = _safe_ratio(h_nm, w_nm)
        # Substrate height to gap ratio
        groups["stackup_h_over_gap"] = _safe_ratio(h_nm, gap_nm)
        # Ground opening to height ratio
        ground_opening_nm = w_nm + 2 * gap_nm
        groups["stackup_ground_opening_over_h"] = _safe_ratio(ground_opening_nm, h_nm)

    # Total stackup thickness
    total_thickness = sum(int(t) for t in thicknesses.values())
    groups["stackup_total_thickness_nm"] = float(total_thickness)

    if h_nm is not None and h_nm > 0:
        groups["stackup_total_over_h"] = _safe_ratio(total_thickness, h_nm)

    # Layer ratios for multi-layer stackups (L1_to_L2, L2_to_L3, etc.)
    # Compute ratios between adjacent layer thicknesses
    layer_keys = sorted([k for k in thicknesses if k.startswith("L")])
    if len(layer_keys) >= 2:
        # Reference layer is typically the thickest (core)
        ref_thickness = max(int(thicknesses[k]) for k in layer_keys)
        for layer_key in layer_keys:
            thickness = int(thicknesses[layer_key])
            key = f"stackup_{layer_key.lower()}_ratio"
            groups[key] = _safe_ratio(thickness, ref_thickness)

    # Handle legacy "core", "copper", "prepreg" naming
    if "core" in thicknesses:
        core_nm = int(thicknesses["core"])
        if "copper" in thicknesses:
            copper_nm = int(thicknesses["copper"])
            groups["stackup_copper_over_core"] = _safe_ratio(copper_nm, core_nm)
            groups["stackup_copper_over_w"] = _safe_ratio(copper_nm, w_nm)
        if "prepreg" in thicknesses:
            prepreg_nm = int(thicknesses["prepreg"])
            groups["stackup_prepreg_over_core"] = _safe_ratio(prepreg_nm, core_nm)

    # Copper layers count ratio (useful for via inductance)
    groups["stackup_copper_layers"] = float(spec.stackup.copper_layers)

    return groups
