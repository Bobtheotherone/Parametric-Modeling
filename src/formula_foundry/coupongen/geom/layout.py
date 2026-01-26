"""Layout Plan dataclasses - Single Source of Truth for Coupon Geometry.

This module defines the authoritative internal representation (IR) for coupon
layout geometry. All placements, trace segments, and port positions derive from
LayoutPlan to ensure topological consistency and eliminate duplicated geometry
math.

The LayoutPlan enforces a key invariant for F1 coupons:
    x_discontinuity_center_nm == x_left_connector_ref_nm + length_left_nm
    x_discontinuity_center_nm == x_right_connector_ref_nm - length_right_nm

This guarantees that trace segments connect without gaps and the SIG net
forms a single connected component.

All coordinates use integer nanometers (nm) for determinism and hashability.

Satisfies CP-2.1 (ECO-M1-ALIGN-0001).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .primitives import OriginMode, PositionNM

if TYPE_CHECKING:
    from ..resolve import ResolvedDesign
    from ..spec import CouponSpec
    from .launch import LaunchPlan


@dataclass(frozen=True, slots=True)
class PortPlan:
    """Port placement plan for a connector.

    Defines the reference position and signal pad anchor for a port
    (end-launch connector) in the canonical coordinate frame.

    Attributes:
        x_ref_nm: X coordinate of the port reference point (connector center)
                  in nanometers.
        y_ref_nm: Y coordinate of the port reference point (usually 0 for
                  centerline placement) in nanometers.
        signal_pad_x_nm: X coordinate of the signal pad connection point in nm.
                         This is where the transmission line connects (launch reference).
        signal_pad_y_nm: Y coordinate of the signal pad connection point in nm.
        footprint: Full footprint path as 'library:name'.
        rotation_mdeg: Rotation in millidegrees (0, 90000, 180000, 270000).
        side: Port side identifier ('left' or 'right').
    """

    x_ref_nm: int
    y_ref_nm: int
    signal_pad_x_nm: int
    signal_pad_y_nm: int
    footprint: str
    rotation_mdeg: int
    side: str

    def __post_init__(self) -> None:
        """Validate port plan invariants."""
        if self.side not in ("left", "right"):
            raise ValueError(f"Port side must be 'left' or 'right', got {self.side!r}")
        if self.rotation_mdeg not in (0, 90000, 180000, 270000):
            raise ValueError(f"Rotation must be 0, 90000, 180000, or 270000 mdeg, got {self.rotation_mdeg}")


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """Trace segment plan for a transmission line section.

    Defines a single horizontal trace segment in the canonical coordinate
    frame. All F0/F1 coupons use horizontal centerline traces (y=0).

    Attributes:
        x_start_nm: X coordinate of segment start in nanometers.
        x_end_nm: X coordinate of segment end in nanometers.
        y_nm: Y coordinate of the centerline in nanometers (typically 0).
        width_nm: Trace width in nanometers.
        layer: Copper layer name (e.g., "F.Cu", "B.Cu").
        net_name: Net name for electrical connectivity (e.g., "SIG").
        label: Human-readable label for the segment (e.g., "left", "right").
    """

    x_start_nm: int
    x_end_nm: int
    y_nm: int
    width_nm: int
    layer: str
    net_name: str
    label: str

    def __post_init__(self) -> None:
        """Validate segment plan invariants."""
        if self.width_nm <= 0:
            raise ValueError(f"Trace width must be positive, got {self.width_nm}")
        if self.x_end_nm < self.x_start_nm:
            raise ValueError(f"Segment end ({self.x_end_nm}) must be >= start ({self.x_start_nm})")

    @property
    def length_nm(self) -> int:
        """Computed length of the segment in nanometers."""
        return self.x_end_nm - self.x_start_nm


@dataclass(frozen=True, slots=True)
class LayoutPlan:
    """Authoritative layout plan for coupon geometry.

    This is the single source of truth for all geometry in a coupon.
    All placements and segments reference this plan — no duplicated geometry
    math elsewhere.

    Coordinate System (EDGE_L_CENTER):
    - x_board_left_edge_nm = 0 (origin convention)
    - y_centerline_nm = 0
    - +x to the right, +y upward

    Key Invariant for F1 coupons:
        x_disc_nm == left_port.signal_pad_x_nm + segment_left.length_nm
        x_disc_nm == right_port.signal_pad_x_nm - segment_right.length_nm

    This ensures trace endpoints coincide at the discontinuity center.

    Attributes:
        origin_mode: Coordinate frame origin mode.
        board_length_nm: Board length (X dimension) in nanometers.
        board_width_nm: Board width (Y dimension) in nanometers.
        board_corner_radius_nm: Board corner radius in nanometers.
        left_port: Left port placement plan.
        right_port: Right port placement plan.
        segments: Tuple of trace segment plans (ordered left-to-right).
        x_disc_nm: X coordinate of the discontinuity center in nanometers.
                   None for F0 coupons (no discontinuity).
        y_centerline_nm: Y coordinate of the signal centerline (typically 0).
        coupon_family: Coupon family identifier (e.g., "F0", "F1").
        launch_plans: Optional launch transition plans for left/right connectors.
    """

    origin_mode: OriginMode
    board_length_nm: int
    board_width_nm: int
    board_corner_radius_nm: int
    left_port: PortPlan
    right_port: PortPlan
    segments: tuple[SegmentPlan, ...]
    x_disc_nm: int | None
    y_centerline_nm: int
    coupon_family: str
    launch_plans: tuple[LaunchPlan, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate layout plan invariants."""
        # Board dimensions must be positive
        if self.board_length_nm <= 0:
            raise ValueError(f"Board length must be positive, got {self.board_length_nm}")
        if self.board_width_nm <= 0:
            raise ValueError(f"Board width must be positive, got {self.board_width_nm}")
        if self.board_corner_radius_nm < 0:
            raise ValueError(f"Board corner radius must be non-negative, got {self.board_corner_radius_nm}")

        # At least one segment is required
        if len(self.segments) == 0:
            raise ValueError("At least one trace segment is required")

        # For F1, validate the continuity invariant
        if self.x_disc_nm is not None and len(self.segments) >= 2:
            # Find left and right segments by label
            left_seg = next((s for s in self.segments if s.label == "left"), None)
            right_seg = next((s for s in self.segments if s.label == "right"), None)

            if left_seg is not None and right_seg is not None:
                # Left segment must end at discontinuity
                if left_seg.x_end_nm != self.x_disc_nm:
                    raise ValueError(f"Left segment end ({left_seg.x_end_nm}) must equal discontinuity X ({self.x_disc_nm})")
                # Right segment must start at discontinuity
                if right_seg.x_start_nm != self.x_disc_nm:
                    raise ValueError(
                        f"Right segment start ({right_seg.x_start_nm}) must equal discontinuity X ({self.x_disc_nm})"
                    )

    @property
    def x_board_left_edge_nm(self) -> int:
        """X coordinate of the left board edge (always 0 in EDGE_L_CENTER)."""
        return 0

    @property
    def x_board_right_edge_nm(self) -> int:
        """X coordinate of the right board edge."""
        return self.board_length_nm

    @property
    def y_board_top_edge_nm(self) -> int:
        """Y coordinate of the top board edge."""
        return self.board_width_nm // 2

    @property
    def y_board_bottom_edge_nm(self) -> int:
        """Y coordinate of the bottom board edge."""
        return -(self.board_width_nm // 2)

    @property
    def total_trace_length_nm(self) -> int:
        """Total length of all trace segments in nanometers."""
        return sum(seg.length_nm for seg in self.segments)

    @property
    def has_discontinuity(self) -> bool:
        """Whether this layout includes a discontinuity (via transition)."""
        return self.x_disc_nm is not None

    def get_segment_by_label(self, label: str) -> SegmentPlan | None:
        """Get a segment by its label.

        Args:
            label: Segment label (e.g., "left", "right", "through").

        Returns:
            The matching SegmentPlan, or None if not found.
        """
        return next((s for s in self.segments if s.label == label), None)

    def get_launch_plan(self, side: str) -> LaunchPlan | None:
        """Get launch plan by connector side."""
        return next((lp for lp in self.launch_plans if lp.side == side), None)

    def validate_connectivity(self) -> list[str]:
        """Validate that all segments are connected.

        Checks that consecutive segments share endpoints and that
        the first/last segments connect to ports.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        # Check segment connectivity
        for i in range(len(self.segments) - 1):
            curr = self.segments[i]
            next_seg = self.segments[i + 1]
            if curr.x_end_nm != next_seg.x_start_nm:
                errors.append(
                    f"Segment gap between '{curr.label}' end ({curr.x_end_nm}) "
                    f"and '{next_seg.label}' start ({next_seg.x_start_nm})"
                )

        # Check first segment connects to left port signal pad
        if self.segments:
            first_seg = self.segments[0]
            if first_seg.x_start_nm != self.left_port.signal_pad_x_nm:
                errors.append(
                    f"First segment start ({first_seg.x_start_nm}) doesn't match "
                    f"left port signal pad X ({self.left_port.signal_pad_x_nm})"
                )

            last_seg = self.segments[-1]
            if last_seg.x_end_nm != self.right_port.signal_pad_x_nm:
                errors.append(
                    f"Last segment end ({last_seg.x_end_nm}) doesn't match "
                    f"right port signal pad X ({self.right_port.signal_pad_x_nm})"
                )

        return errors


def create_f0_layout_plan(
    *,
    board_length_nm: int,
    board_width_nm: int,
    board_corner_radius_nm: int,
    left_port_x_nm: int,
    right_port_x_nm: int,
    trace_width_nm: int,
    trace_layer: str,
    footprint: str,
) -> LayoutPlan:
    """Create a LayoutPlan for an F0 (through-line) coupon.

    F0 coupons have a single continuous trace from left to right port
    with no discontinuity.

    Args:
        board_length_nm: Board length in nanometers.
        board_width_nm: Board width in nanometers.
        board_corner_radius_nm: Board corner radius in nanometers.
        left_port_x_nm: X position of left connector reference point.
        right_port_x_nm: X position of right connector reference point.
        trace_width_nm: Trace width in nanometers.
        trace_layer: Copper layer name.
        footprint: Connector footprint path.

    Returns:
        LayoutPlan for the F0 coupon.
    """
    from .footprint_meta import load_footprint_meta

    meta = load_footprint_meta(footprint)
    left_pad_x, left_pad_y = _transform_pad_position(
        anchor_x=left_port_x_nm,
        anchor_y=0,
        pad_offset_x=meta.signal_pad.center_x_nm,
        pad_offset_y=meta.signal_pad.center_y_nm,
        rotation_deg=0,
    )
    right_pad_x, right_pad_y = _transform_pad_position(
        anchor_x=right_port_x_nm,
        anchor_y=0,
        pad_offset_x=meta.signal_pad.center_x_nm,
        pad_offset_y=meta.signal_pad.center_y_nm,
        rotation_deg=180,
    )

    left_port = PortPlan(
        x_ref_nm=left_port_x_nm,
        y_ref_nm=0,
        signal_pad_x_nm=left_pad_x,
        signal_pad_y_nm=left_pad_y,
        footprint=meta.footprint_path,
        rotation_mdeg=0,
        side="left",
    )

    right_port = PortPlan(
        x_ref_nm=right_port_x_nm,
        y_ref_nm=0,
        signal_pad_x_nm=right_pad_x,
        signal_pad_y_nm=right_pad_y,
        footprint=meta.footprint_path,
        rotation_mdeg=180000,
        side="right",
    )

    # Single through segment
    through_segment = SegmentPlan(
        x_start_nm=left_pad_x,
        x_end_nm=right_pad_x,
        y_nm=left_pad_y,
        width_nm=trace_width_nm,
        layer=trace_layer,
        net_name="SIG",
        label="through",
    )

    return LayoutPlan(
        origin_mode=OriginMode.EDGE_L_CENTER,
        board_length_nm=board_length_nm,
        board_width_nm=board_width_nm,
        board_corner_radius_nm=board_corner_radius_nm,
        left_port=left_port,
        right_port=right_port,
        segments=(through_segment,),
        x_disc_nm=None,
        y_centerline_nm=0,
        coupon_family="F0_THROUGH_LINE",
    )


def create_f1_layout_plan(
    *,
    board_length_nm: int,
    board_width_nm: int,
    board_corner_radius_nm: int,
    left_port_x_nm: int,
    right_port_x_nm: int,
    left_length_nm: int,
    trace_width_nm: int,
    trace_layer: str,
    footprint: str,
) -> LayoutPlan:
    """Create a LayoutPlan for an F1 (via transition) coupon.

    F1 coupons have two trace segments connecting at a via transition
    (discontinuity). The right length is derived to ensure continuity:
        right_length_nm = right_port_x_nm - (left_port_x_nm + left_length_nm)

    This enforces the key invariant that both segments meet at the
    discontinuity center.

    Args:
        board_length_nm: Board length in nanometers.
        board_width_nm: Board width in nanometers.
        board_corner_radius_nm: Board corner radius in nanometers.
        left_port_x_nm: X position of left connector reference point.
        right_port_x_nm: X position of right connector reference point.
        left_length_nm: Length of the left trace segment.
        trace_width_nm: Trace width in nanometers.
        trace_layer: Copper layer name.
        footprint: Connector footprint path.

    Returns:
        LayoutPlan for the F1 coupon.

    Raises:
        ValueError: If left_length_nm results in discontinuity outside board.
    """
    from .footprint_meta import load_footprint_meta

    meta = load_footprint_meta(footprint)
    left_pad_x, left_pad_y = _transform_pad_position(
        anchor_x=left_port_x_nm,
        anchor_y=0,
        pad_offset_x=meta.signal_pad.center_x_nm,
        pad_offset_y=meta.signal_pad.center_y_nm,
        rotation_deg=0,
    )
    right_pad_x, right_pad_y = _transform_pad_position(
        anchor_x=right_port_x_nm,
        anchor_y=0,
        pad_offset_x=meta.signal_pad.center_x_nm,
        pad_offset_y=meta.signal_pad.center_y_nm,
        rotation_deg=180,
    )

    # Derive discontinuity position
    x_disc_nm = left_pad_x + left_length_nm

    # Derive right length to ensure continuity
    right_length_nm = right_pad_x - x_disc_nm

    if right_length_nm < 0:
        raise ValueError(f"Left length ({left_length_nm}) places discontinuity ({x_disc_nm}) beyond right port ({right_pad_x})")

    left_port = PortPlan(
        x_ref_nm=left_port_x_nm,
        y_ref_nm=0,
        signal_pad_x_nm=left_pad_x,
        signal_pad_y_nm=left_pad_y,
        footprint=meta.footprint_path,
        rotation_mdeg=0,
        side="left",
    )

    right_port = PortPlan(
        x_ref_nm=right_port_x_nm,
        y_ref_nm=0,
        signal_pad_x_nm=right_pad_x,
        signal_pad_y_nm=right_pad_y,
        footprint=meta.footprint_path,
        rotation_mdeg=180000,
        side="right",
    )

    # Left segment: from left port to discontinuity
    left_segment = SegmentPlan(
        x_start_nm=left_pad_x,
        x_end_nm=x_disc_nm,
        y_nm=left_pad_y,
        width_nm=trace_width_nm,
        layer=trace_layer,
        net_name="SIG",
        label="left",
    )

    # Right segment: from discontinuity to right port
    right_segment = SegmentPlan(
        x_start_nm=x_disc_nm,
        x_end_nm=right_pad_x,
        y_nm=right_pad_y,
        width_nm=trace_width_nm,
        layer=trace_layer,
        net_name="SIG",
        label="right",
    )

    return LayoutPlan(
        origin_mode=OriginMode.EDGE_L_CENTER,
        board_length_nm=board_length_nm,
        board_width_nm=board_width_nm,
        board_corner_radius_nm=board_corner_radius_nm,
        left_port=left_port,
        right_port=right_port,
        segments=(left_segment, right_segment),
        x_disc_nm=x_disc_nm,
        y_centerline_nm=0,
        coupon_family="F1_SINGLE_ENDED_VIA",
    )


def derive_right_length_nm(
    left_port_x_nm: int,
    right_port_x_nm: int,
    left_length_nm: int,
) -> int:
    """Derive the right trace length to ensure F1 continuity.

    Given the port positions and left trace length, compute the right
    trace length such that both segments meet at the discontinuity.

    Args:
        left_port_x_nm: X position of left connector.
        right_port_x_nm: X position of right connector.
        left_length_nm: Length of the left trace segment.

    Returns:
        The derived right trace length in nanometers.

    Raises:
        ValueError: If the derived length is negative.
    """
    x_disc_nm = left_port_x_nm + left_length_nm
    right_length_nm = right_port_x_nm - x_disc_nm

    if right_length_nm < 0:
        raise ValueError(
            f"Derived right length is negative ({right_length_nm}). "
            f"Left length ({left_length_nm}) is too long for the given port positions."
        )

    return right_length_nm


def compute_layout_plan(
    spec: CouponSpec,
    resolved: ResolvedDesign,  # noqa: ARG001 - reserved for future use
) -> LayoutPlan:
    """Compute LayoutPlan from CouponSpec and ResolvedDesign.

    This is the main entry point for computing layout geometry from a
    specification. It loads footprint metadata to get pad centers,
    enforces EDGE_L_CENTER origin, and for F1 coupons derives the right
    trace length from the continuity formula.

    For F1 coupons, the continuity formula is:
        xD = xL + length_left  (discontinuity position)
        length_right = xR - xD (derived to ensure continuity)

    where xL and xR are the signal pad X positions of the left and right
    connectors.

    Args:
        spec: The coupon specification with all geometry parameters.
        resolved: The resolved design (reserved for future extensions).

    Returns:
        LayoutPlan with all geometry computed and validated.

    Raises:
        FileNotFoundError: If footprint metadata cannot be loaded.
        ValueError: If geometry constraints cannot be satisfied.

    Satisfies CP-2.3 per ECO-M1-ALIGN-0001 Section 13.2.4.
    """
    from ..families import FAMILY_F0, FAMILY_F1
    from .footprint_meta import load_footprint_meta

    # Extract board dimensions
    board_length_nm = int(spec.board.outline.length_nm)
    board_width_nm = int(spec.board.outline.width_nm)
    board_corner_radius_nm = int(spec.board.outline.corner_radius_nm)

    # Load footprint metadata for left and right connectors
    left_footprint = spec.connectors.left.footprint
    right_footprint = spec.connectors.right.footprint

    left_meta = load_footprint_meta(left_footprint)
    right_meta = load_footprint_meta(right_footprint)

    # Get connector placement positions from spec
    left_connector_pos = spec.connectors.left.position_nm
    right_connector_pos = spec.connectors.right.position_nm
    left_rotation_deg = spec.connectors.left.rotation_deg
    right_rotation_deg = spec.connectors.right.rotation_deg

    # Compute signal pad center positions in board coordinates
    left_pad_center_x, left_pad_center_y = _transform_pad_position(
        anchor_x=int(left_connector_pos[0]),
        anchor_y=int(left_connector_pos[1]),
        pad_offset_x=left_meta.signal_pad.center_x_nm,
        pad_offset_y=left_meta.signal_pad.center_y_nm,
        rotation_deg=left_rotation_deg,
    )

    right_pad_center_x, right_pad_center_y = _transform_pad_position(
        anchor_x=int(right_connector_pos[0]),
        anchor_y=int(right_connector_pos[1]),
        pad_offset_x=right_meta.signal_pad.center_x_nm,
        pad_offset_y=right_meta.signal_pad.center_y_nm,
        rotation_deg=right_rotation_deg,
    )

    # Compute launch reference positions (connection point to CPWG)
    left_launch_x, left_launch_y = _transform_pad_position(
        anchor_x=int(left_connector_pos[0]),
        anchor_y=int(left_connector_pos[1]),
        pad_offset_x=left_meta.launch_reference.x_nm,
        pad_offset_y=left_meta.launch_reference.y_nm,
        rotation_deg=left_rotation_deg,
    )

    right_launch_x, right_launch_y = _transform_pad_position(
        anchor_x=int(right_connector_pos[0]),
        anchor_y=int(right_connector_pos[1]),
        pad_offset_x=right_meta.launch_reference.x_nm,
        pad_offset_y=right_meta.launch_reference.y_nm,
        rotation_deg=right_rotation_deg,
    )

    # Get trace parameters
    trace_width_nm = int(spec.transmission_line.w_nm)
    trace_layer = spec.transmission_line.layer
    gap_nm = int(spec.transmission_line.gap_nm)
    length_left_nm = int(spec.transmission_line.length_left_nm)

    # Build port plans
    left_port = PortPlan(
        x_ref_nm=int(left_connector_pos[0]),
        y_ref_nm=int(left_connector_pos[1]),
        signal_pad_x_nm=left_launch_x,
        signal_pad_y_nm=left_launch_y,
        footprint=left_meta.footprint_path,
        rotation_mdeg=left_rotation_deg * 1000,
        side="left",
    )

    right_port = PortPlan(
        x_ref_nm=int(right_connector_pos[0]),
        y_ref_nm=int(right_connector_pos[1]),
        signal_pad_x_nm=right_launch_x,
        signal_pad_y_nm=right_launch_y,
        footprint=right_meta.footprint_path,
        rotation_mdeg=right_rotation_deg * 1000,
        side="right",
    )

    from ..constraints.core import resolve_fab_limits
    from .launch import build_launch_plan

    fab_limits = resolve_fab_limits(spec)
    min_trace_width_nm = int(fab_limits.get("min_trace_width_nm", 0))
    min_gap_nm = int(fab_limits.get("min_gap_nm", 0))

    entry_layer = trace_layer
    exit_layer = "B.Cu" if trace_layer == "F.Cu" else "F.Cu"
    left_launch_layer = entry_layer
    right_launch_layer = exit_layer if spec.coupon_family == FAMILY_F1 else trace_layer

    left_launch = build_launch_plan(
        side="left",
        pad_center=PositionNM(left_pad_center_x, left_pad_center_y),
        launch_point=PositionNM(left_launch_x, left_launch_y),
        launch_direction_deg=left_meta.launch_reference.direction_deg,
        rotation_deg=left_rotation_deg,
        pad_size_x_nm=left_meta.signal_pad.size_x_nm,
        pad_size_y_nm=left_meta.signal_pad.size_y_nm,
        trace_width_nm=trace_width_nm,
        trace_layer=left_launch_layer,
        gap_nm=gap_nm,
        min_trace_width_nm=min_trace_width_nm,
        min_gap_nm=min_gap_nm,
        ground_via_fence=spec.transmission_line.ground_via_fence,
    )

    right_launch = build_launch_plan(
        side="right",
        pad_center=PositionNM(right_pad_center_x, right_pad_center_y),
        launch_point=PositionNM(right_launch_x, right_launch_y),
        launch_direction_deg=right_meta.launch_reference.direction_deg,
        rotation_deg=right_rotation_deg,
        pad_size_x_nm=right_meta.signal_pad.size_x_nm,
        pad_size_y_nm=right_meta.signal_pad.size_y_nm,
        trace_width_nm=trace_width_nm,
        trace_layer=right_launch_layer,
        gap_nm=gap_nm,
        min_trace_width_nm=min_trace_width_nm,
        min_gap_nm=min_gap_nm,
        ground_via_fence=spec.transmission_line.ground_via_fence,
    )

    launch_plans = (left_launch, right_launch)

    # Dispatch based on coupon family
    if spec.coupon_family == FAMILY_F0:
        return _compute_f0_layout(
            board_length_nm=board_length_nm,
            board_width_nm=board_width_nm,
            board_corner_radius_nm=board_corner_radius_nm,
            left_port=left_port,
            right_port=right_port,
            trace_width_nm=trace_width_nm,
            trace_layer=trace_layer,
            launch_plans=launch_plans,
        )
    elif spec.coupon_family == FAMILY_F1:
        return _compute_f1_layout(
            board_length_nm=board_length_nm,
            board_width_nm=board_width_nm,
            board_corner_radius_nm=board_corner_radius_nm,
            left_port=left_port,
            right_port=right_port,
            trace_width_nm=trace_width_nm,
            trace_layer=trace_layer,
            length_left_nm=length_left_nm,
            launch_plans=launch_plans,
        )
    else:
        raise ValueError(f"Unsupported coupon family: {spec.coupon_family}")


def _transform_pad_position(
    anchor_x: int,
    anchor_y: int,
    pad_offset_x: int,
    pad_offset_y: int,
    rotation_deg: int,
) -> tuple[int, int]:
    """Transform pad position from footprint-relative to board coordinates.

    Args:
        anchor_x: Connector anchor X position in board coordinates.
        anchor_y: Connector anchor Y position in board coordinates.
        pad_offset_x: Pad X offset relative to footprint anchor.
        pad_offset_y: Pad Y offset relative to footprint anchor.
        rotation_deg: Connector rotation in degrees.

    Returns:
        Tuple of (x, y) pad position in board coordinates.
    """
    # Apply rotation to the pad offset
    rad = math.radians(rotation_deg)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)

    # Rotate offset vector
    rotated_x = pad_offset_x * cos_r - pad_offset_y * sin_r
    rotated_y = pad_offset_x * sin_r + pad_offset_y * cos_r

    # Add to anchor position (round to int for determinism)
    return (
        anchor_x + round(rotated_x),
        anchor_y + round(rotated_y),
    )


def _compute_f0_layout(
    *,
    board_length_nm: int,
    board_width_nm: int,
    board_corner_radius_nm: int,
    left_port: PortPlan,
    right_port: PortPlan,
    trace_width_nm: int,
    trace_layer: str,
    launch_plans: tuple[LaunchPlan, ...] = (),
) -> LayoutPlan:
    """Compute layout for F0 (through-line) coupon.

    F0 coupons have a single continuous trace from left to right port.
    """
    # Single through segment from left signal pad to right signal pad
    through_segment = SegmentPlan(
        x_start_nm=left_port.signal_pad_x_nm,
        x_end_nm=right_port.signal_pad_x_nm,
        y_nm=left_port.signal_pad_y_nm,  # Assume centerline at y=0
        width_nm=trace_width_nm,
        layer=trace_layer,
        net_name="SIG",
        label="through",
    )

    return LayoutPlan(
        origin_mode=OriginMode.EDGE_L_CENTER,
        board_length_nm=board_length_nm,
        board_width_nm=board_width_nm,
        board_corner_radius_nm=board_corner_radius_nm,
        left_port=left_port,
        right_port=right_port,
        segments=(through_segment,),
        x_disc_nm=None,
        y_centerline_nm=left_port.signal_pad_y_nm,
        coupon_family="F0_CAL_THRU_LINE",
        launch_plans=launch_plans,
    )


def _compute_f1_layout(
    *,
    board_length_nm: int,
    board_width_nm: int,
    board_corner_radius_nm: int,
    left_port: PortPlan,
    right_port: PortPlan,
    trace_width_nm: int,
    trace_layer: str,
    length_left_nm: int,
    launch_plans: tuple[LaunchPlan, ...] = (),
) -> LayoutPlan:
    """Compute layout for F1 (via transition) coupon.

    F1 coupons have two trace segments meeting at a via discontinuity.
    The right length is derived from the continuity formula:
        xD = xL + length_left (discontinuity position)
        length_right = xR - xD (derived to ensure segments meet)

    where xL is the left signal pad X and xR is the right signal pad X.

    For proper via transition, the left segment uses the entry layer (e.g., F.Cu)
    and the right segment uses the exit layer (B.Cu). The signal via connects
    both layers, ensuring DRC clean (no via_dangling warnings).
    """
    # Discontinuity position: xD = xL + length_left
    x_disc_nm = left_port.signal_pad_x_nm + length_left_nm

    # Derived right length: xR - xD (the continuity formula)
    derived_right_length = right_port.signal_pad_x_nm - x_disc_nm

    if derived_right_length < 0:
        raise ValueError(
            f"Left length ({length_left_nm}) places discontinuity ({x_disc_nm}) "
            f"beyond right signal pad ({right_port.signal_pad_x_nm}). "
            f"Derived right length would be negative ({derived_right_length})."
        )

    # Determine entry and exit layers for via transition
    # Entry layer is from spec (typically F.Cu), exit layer is B.Cu for 4-layer boards
    entry_layer = trace_layer
    exit_layer = "B.Cu" if trace_layer == "F.Cu" else "F.Cu"

    # Left segment: from left signal pad to discontinuity (entry layer)
    left_segment = SegmentPlan(
        x_start_nm=left_port.signal_pad_x_nm,
        x_end_nm=x_disc_nm,
        y_nm=left_port.signal_pad_y_nm,
        width_nm=trace_width_nm,
        layer=entry_layer,
        net_name="SIG",
        label="left",
    )

    # Right segment: from discontinuity to right signal pad (exit layer)
    # This ensures the signal via connects traces on both layers
    right_segment = SegmentPlan(
        x_start_nm=x_disc_nm,
        x_end_nm=right_port.signal_pad_x_nm,
        y_nm=right_port.signal_pad_y_nm,
        width_nm=trace_width_nm,
        layer=exit_layer,
        net_name="SIG",
        label="right",
    )

    return LayoutPlan(
        origin_mode=OriginMode.EDGE_L_CENTER,
        board_length_nm=board_length_nm,
        board_width_nm=board_width_nm,
        board_corner_radius_nm=board_corner_radius_nm,
        left_port=left_port,
        right_port=right_port,
        segments=(left_segment, right_segment),
        x_disc_nm=x_disc_nm,
        y_centerline_nm=left_port.signal_pad_y_nm,
        coupon_family="F1_SINGLE_ENDED_VIA",
        launch_plans=launch_plans,
    )
