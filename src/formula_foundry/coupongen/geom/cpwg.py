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

from .primitives import PositionNM, TrackSegment, Via

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
    """

    w_nm: int
    gap_nm: int
    length_nm: int
    layer: str = "F.Cu"
    net_id: int = 1


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
        fence_vias_positive_y: Vias on the +y side of the signal trace.
        fence_vias_negative_y: Vias on the -y side of the signal trace.
    """

    signal_track: TrackSegment
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
    segment_length = int(math.sqrt(dx * dx + dy * dy))

    if segment_length == 0:
        return ((), ())

    # Number of vias along the segment
    # Place first via at pitch/2 from start, then every pitch_nm
    if fence_spec.pitch_nm <= 0:
        raise ValueError("pitch_nm must be positive")

    # Calculate via positions along the segment
    # Start at half pitch from start, continue until we would exceed the end
    num_vias = max(1, (segment_length + fence_spec.pitch_nm // 2) // fence_spec.pitch_nm)

    # Normalize direction vector
    unit_dx = dx / segment_length if segment_length > 0 else 0
    unit_dy = dy / segment_length if segment_length > 0 else 0

    # Perpendicular vector (90 degrees counterclockwise for +y offset)
    perp_dx = -unit_dy
    perp_dy = unit_dx

    # Distance from centerline to via center:
    # signal_trace_half_width + gap + offset_from_gap
    via_offset_from_center = cpwg_spec.w_nm // 2 + cpwg_spec.gap_nm + fence_spec.offset_from_gap_nm

    positive_y_vias: list[Via] = []
    negative_y_vias: list[Via] = []

    for i in range(num_vias):
        # Position along the segment (start at half pitch, then every pitch)
        t = fence_spec.pitch_nm // 2 + i * fence_spec.pitch_nm
        if t > segment_length:
            break

        # Base position on the centerline
        base_x = start.x + int(t * unit_dx)
        base_y = start.y + int(t * unit_dy)

        # Positive Y side via
        pos_y_via = Via(
            position=PositionNM(
                base_x + int(via_offset_from_center * perp_dx),
                base_y + int(via_offset_from_center * perp_dy),
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
                base_x - int(via_offset_from_center * perp_dx),
                base_y - int(via_offset_from_center * perp_dy),
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
        CPWGResult containing the signal track and via fence tuples.
    """
    signal_track = generate_cpwg_horizontal(origin, cpwg_spec, direction)

    if fence_spec is None:
        return CPWGResult(
            signal_track=signal_track,
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
    )
    right_spec = CPWGSpec(
        w_nm=cpwg_spec.w_nm,
        gap_nm=cpwg_spec.gap_nm,
        length_nm=right_length_nm,
        layer=cpwg_spec.layer,
        net_id=cpwg_spec.net_id,
    )

    left_result = generate_cpwg_with_fence(center, left_spec, fence_spec, direction=-1)
    right_result = generate_cpwg_with_fence(center, right_spec, fence_spec, direction=1)

    return (left_result, right_result)
