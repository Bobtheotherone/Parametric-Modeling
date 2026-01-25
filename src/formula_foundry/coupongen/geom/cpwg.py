"""CPWG (Coplanar Waveguide with Ground) transmission line geometry generators.

This module provides functions to generate CPWG transmission line geometry,
including the signal trace, ground coplanar areas, and optional via fencing.

All coordinates use integer nanometers (LengthNM) to ensure determinism and
avoid cross-platform rounding drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .primitives import PositionNM, TrackSegment, Via, half_width_nm
from .via_patterns import scale_component_nm, segment_length_nm, symmetric_offsets_nm

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class CPWGSpec:
    """Specification for a CPWG transmission line segment.

    Attributes:
        w_nm: Signal trace width in nanometers.
        gap_nm: Gap between signal trace and coplanar ground in nanometers.
        length_nm: Length of the transmission line segment in nanometers.
        layer: Copper layer for the transmission line (e.g., "F.Cu").
        net_id: Net ID for the signal trace.
        ground_width_nm: Optional ground rail width in nanometers (defaults to w_nm).
        ground_net_id: Net ID for ground copper.
    """

    w_nm: int
    gap_nm: int
    length_nm: int
    layer: str = "F.Cu"
    net_id: int = 1
    ground_width_nm: int | None = None
    ground_net_id: int = 2


@dataclass(frozen=True, slots=True)
class GroundViaFenceSpec:
    """Specification for ground via fencing along a CPWG line.

    Attributes:
        pitch_nm: Via-to-via pitch along the transmission line in nanometers.
        offset_from_gap_nm: Offset from the gap edge to the via center in nanometers.
        drill_nm: Via drill diameter in nanometers.
        diameter_nm: Via pad diameter in nanometers.
        layers: Tuple of layer names the vias connect.
        net_id: Net ID for the ground vias (typically 0 or ground net).
    """

    pitch_nm: int
    offset_from_gap_nm: int
    drill_nm: int
    diameter_nm: int
    layers: tuple[str, str] = ("F.Cu", "B.Cu")
    net_id: int = 0


@dataclass(frozen=True, slots=True)
class CPWGResult:
    """Result of CPWG geometry generation.

    Attributes:
        signal_track: The signal trace track segment.
        ground_tracks: Tuple of ground rail track segments (positive, negative offset).
        fence_vias_positive_y: Vias on the +y side of the signal trace.
        fence_vias_negative_y: Vias on the -y side of the signal trace.
    """

    signal_track: TrackSegment
    ground_tracks: tuple[TrackSegment, TrackSegment]
    fence_vias_positive_y: tuple[Via, ...]
    fence_vias_negative_y: tuple[Via, ...]


def generate_cpwg_segment(
    start: PositionNM,
    end: PositionNM,
    spec: CPWGSpec,
) -> TrackSegment:
    """Generate a CPWG signal trace segment.

    Creates a straight track segment from start to end with the specified width.
    The track represents the center conductor of the CPWG structure.

    Args:
        start: Start position of the segment in nm.
        end: End position of the segment in nm.
        spec: CPWG specification with width, gap, and layer info.

    Returns:
        TrackSegment representing the signal trace.
    """
    return TrackSegment(
        start=start,
        end=end,
        width_nm=spec.w_nm,
        layer=spec.layer,
        net_id=spec.net_id,
    )


def _resolve_ground_width_nm(spec: CPWGSpec) -> int:
    if spec.ground_width_nm is None:
        return spec.w_nm
    if spec.ground_width_nm <= 0:
        raise ValueError("ground_width_nm must be positive")
    return spec.ground_width_nm


def generate_cpwg_ground_tracks(
    start: PositionNM,
    end: PositionNM,
    spec: CPWGSpec,
) -> tuple[TrackSegment, TrackSegment]:
    """Generate ground rail tracks parallel to the signal trace.

    The rails are offset from the centerline such that the gap between the
    signal edge and ground edge is at least spec.gap_nm.
    """
    if spec.gap_nm < 0:
        raise ValueError("gap_nm must be non-negative")

    dx = end.x - start.x
    dy = end.y - start.y
    segment_length = int(math.sqrt(dx * dx + dy * dy))

    if segment_length == 0:
        # Deterministic fallback: treat zero-length as horizontal for offset.
        perp_dx, perp_dy = 0.0, 1.0
    else:
        unit_dx = dx / segment_length
        unit_dy = dy / segment_length
        perp_dx = -unit_dy
        perp_dy = unit_dx

    ground_width = _resolve_ground_width_nm(spec)
    offset_from_center = (
        half_width_nm(spec.w_nm)
        + spec.gap_nm
        + half_width_nm(ground_width)
    )

    offset_dx = int(offset_from_center * perp_dx)
    offset_dy = int(offset_from_center * perp_dy)

    positive = TrackSegment(
        start=PositionNM(start.x + offset_dx, start.y + offset_dy),
        end=PositionNM(end.x + offset_dx, end.y + offset_dy),
        width_nm=ground_width,
        layer=spec.layer,
        net_id=spec.ground_net_id,
    )
    negative = TrackSegment(
        start=PositionNM(start.x - offset_dx, start.y - offset_dy),
        end=PositionNM(end.x - offset_dx, end.y - offset_dy),
        width_nm=ground_width,
        layer=spec.layer,
        net_id=spec.ground_net_id,
    )

    return (positive, negative)


def generate_cpwg_horizontal(
    origin: PositionNM,
    spec: CPWGSpec,
    direction: int = 1,
) -> TrackSegment:
    """Generate a horizontal CPWG signal trace from an origin point.

    Args:
        origin: Starting position of the segment in nm.
        spec: CPWG specification.
        direction: 1 for +x direction, -1 for -x direction.

    Returns:
        TrackSegment representing the signal trace.
    """
    if direction not in (1, -1):
        raise ValueError("direction must be 1 or -1")

    end = PositionNM(origin.x + direction * spec.length_nm, origin.y)
    return generate_cpwg_segment(origin, end, spec)


def generate_ground_via_fence(
    start: PositionNM,
    end: PositionNM,
    cpwg_spec: CPWGSpec,
    fence_spec: GroundViaFenceSpec,
) -> tuple[tuple[Via, ...], tuple[Via, ...]]:
    """Generate ground via fencing along a CPWG transmission line.

    Creates two rows of ground vias parallel to the signal trace, one on each
    side. The vias are placed at a specified offset from the gap edge.

    The offset is measured from the outer edge of the gap (signal trace edge + gap)
    to the center of the via.

    Args:
        start: Start position of the CPWG segment in nm.
        end: End position of the CPWG segment in nm.
        cpwg_spec: CPWG specification for width and gap.
        fence_spec: Ground via fence specification.

    Returns:
        Tuple of (positive_y_vias, negative_y_vias).
    """
    # Calculate segment vector and length
    dx = end.x - start.x
    dy = end.y - start.y
    segment_length = segment_length_nm(dx, dy)

    if segment_length == 0:
        return ((), ())

    # Validate fence parameters and compute deterministic offsets.
    if fence_spec.pitch_nm <= 0:
        raise ValueError("pitch_nm must be positive")

    if fence_spec.offset_from_gap_nm < 0:
        raise ValueError("offset_from_gap_nm must be non-negative")

    if fence_spec.pitch_nm < fence_spec.diameter_nm:
        raise ValueError("pitch_nm must be at least the via diameter to avoid overlap")

    via_radius_nm = fence_spec.diameter_nm // 2
    if fence_spec.offset_from_gap_nm < via_radius_nm:
        raise ValueError("offset_from_gap_nm must be at least the via radius")

    # Use via radius as minimum end clearance to keep pads within segment bounds.
    offsets = symmetric_offsets_nm(
        segment_length,
        fence_spec.pitch_nm,
        end_clearance_nm=via_radius_nm,
    )
    if not offsets:
        return ((), ())

    # Perpendicular vector derived from segment direction; orient toward +y for consistency.
    perp_dx = -dy
    perp_dy = dx
    if perp_dy < 0 or (perp_dy == 0 and perp_dx < 0):
        perp_dx = -perp_dx
        perp_dy = -perp_dy

    # Distance from centerline to via center:
    # signal_trace_half_width + gap + offset_from_gap
    via_offset_from_center = cpwg_spec.w_nm // 2 + cpwg_spec.gap_nm + fence_spec.offset_from_gap_nm
    perp_offset_x = scale_component_nm(perp_dx, via_offset_from_center, segment_length)
    perp_offset_y = scale_component_nm(perp_dy, via_offset_from_center, segment_length)

    positive_y_vias: list[Via] = []
    negative_y_vias: list[Via] = []

    for t in offsets:
        # Base position on the centerline
        base_x = start.x + scale_component_nm(dx, t, segment_length)
        base_y = start.y + scale_component_nm(dy, t, segment_length)

        # Positive Y side via
        pos_y_via = Via(
            position=PositionNM(
                base_x + perp_offset_x,
                base_y + perp_offset_y,
            ),
            diameter_nm=fence_spec.diameter_nm,
            drill_nm=fence_spec.drill_nm,
            layers=fence_spec.layers,
            net_id=fence_spec.net_id,
        )
        positive_y_vias.append(pos_y_via)

        # Negative Y side via
        neg_y_via = Via(
            position=PositionNM(
                base_x - perp_offset_x,
                base_y - perp_offset_y,
            ),
            diameter_nm=fence_spec.diameter_nm,
            drill_nm=fence_spec.drill_nm,
            layers=fence_spec.layers,
            net_id=fence_spec.net_id,
        )
        negative_y_vias.append(neg_y_via)

    return (tuple(positive_y_vias), tuple(negative_y_vias))


def generate_cpwg_with_fence(
    origin: PositionNM,
    cpwg_spec: CPWGSpec,
    fence_spec: GroundViaFenceSpec | None = None,
    direction: int = 1,
) -> CPWGResult:
    """Generate a complete CPWG structure with optional via fencing.

    Creates the signal trace and, if a fence spec is provided, two rows of
    ground vias parallel to the signal trace.

    Args:
        origin: Starting position of the CPWG segment in nm.
        cpwg_spec: CPWG specification.
        fence_spec: Optional ground via fence specification.
        direction: 1 for +x direction, -1 for -x direction.

    Returns:
        CPWGResult containing the signal track, ground rails, and via fence tuples.
    """
    signal_track = generate_cpwg_horizontal(origin, cpwg_spec, direction)
    ground_tracks = generate_cpwg_ground_tracks(signal_track.start, signal_track.end, cpwg_spec)

    if fence_spec is None:
        return CPWGResult(
            signal_track=signal_track,
            ground_tracks=ground_tracks,
            fence_vias_positive_y=(),
            fence_vias_negative_y=(),
        )

    positive_y_vias, negative_y_vias = generate_ground_via_fence(
        signal_track.start,
        signal_track.end,
        cpwg_spec,
        fence_spec,
    )

    return CPWGResult(
        signal_track=signal_track,
        ground_tracks=ground_tracks,
        fence_vias_positive_y=positive_y_vias,
        fence_vias_negative_y=negative_y_vias,
    )


def generate_symmetric_cpwg_pair(
    center: PositionNM,
    cpwg_spec: CPWGSpec,
    left_length_nm: int,
    right_length_nm: int,
    fence_spec: GroundViaFenceSpec | None = None,
) -> tuple[CPWGResult, CPWGResult]:
    """Generate symmetric CPWG segments extending left and right from center.

    Creates two CPWG segments: one extending in the -x direction (left) and
    one extending in the +x direction (right) from a central point. This is
    useful for creating symmetric coupon layouts.

    Args:
        center: Center position where both segments meet in nm.
        cpwg_spec: Base CPWG specification (length_nm will be overridden).
        left_length_nm: Length of the left segment in nm.
        right_length_nm: Length of the right segment in nm.
        fence_spec: Optional ground via fence specification.

    Returns:
        Tuple of (left_result, right_result) CPWGResults.
    """
    left_spec = CPWGSpec(
        w_nm=cpwg_spec.w_nm,
        gap_nm=cpwg_spec.gap_nm,
        length_nm=left_length_nm,
        layer=cpwg_spec.layer,
        net_id=cpwg_spec.net_id,
        ground_width_nm=cpwg_spec.ground_width_nm,
        ground_net_id=cpwg_spec.ground_net_id,
    )
    right_spec = CPWGSpec(
        w_nm=cpwg_spec.w_nm,
        gap_nm=cpwg_spec.gap_nm,
        length_nm=right_length_nm,
        layer=cpwg_spec.layer,
        net_id=cpwg_spec.net_id,
        ground_width_nm=cpwg_spec.ground_width_nm,
        ground_net_id=cpwg_spec.ground_net_id,
    )

    left_result = generate_cpwg_with_fence(center, left_spec, fence_spec, direction=-1)
    right_result = generate_cpwg_with_fence(center, right_spec, fence_spec, direction=1)

    return (left_result, right_result)
