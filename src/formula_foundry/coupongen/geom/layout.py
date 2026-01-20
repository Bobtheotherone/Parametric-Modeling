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

from dataclasses import dataclass

from .primitives import OriginMode


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
        signal_pad_x_nm: X coordinate of the signal pad center in nanometers.
                         This is where the transmission line connects.
        signal_pad_y_nm: Y coordinate of the signal pad center in nanometers.
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
            raise ValueError(
                f"Rotation must be 0, 90000, 180000, or 270000 mdeg, got {self.rotation_mdeg}"
            )


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
            raise ValueError(
                f"Segment end ({self.x_end_nm}) must be >= start ({self.x_start_nm})"
            )

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

    def __post_init__(self) -> None:
        """Validate layout plan invariants."""
        # Board dimensions must be positive
        if self.board_length_nm <= 0:
            raise ValueError(f"Board length must be positive, got {self.board_length_nm}")
        if self.board_width_nm <= 0:
            raise ValueError(f"Board width must be positive, got {self.board_width_nm}")
        if self.board_corner_radius_nm < 0:
            raise ValueError(
                f"Board corner radius must be non-negative, got {self.board_corner_radius_nm}"
            )

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
                    raise ValueError(
                        f"Left segment end ({left_seg.x_end_nm}) must equal "
                        f"discontinuity X ({self.x_disc_nm})"
                    )
                # Right segment must start at discontinuity
                if right_seg.x_start_nm != self.x_disc_nm:
                    raise ValueError(
                        f"Right segment start ({right_seg.x_start_nm}) must equal "
                        f"discontinuity X ({self.x_disc_nm})"
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
    left_port = PortPlan(
        x_ref_nm=left_port_x_nm,
        y_ref_nm=0,
        signal_pad_x_nm=left_port_x_nm,  # Simplified: pad at ref point
        signal_pad_y_nm=0,
        footprint=footprint,
        rotation_mdeg=0,
        side="left",
    )

    right_port = PortPlan(
        x_ref_nm=right_port_x_nm,
        y_ref_nm=0,
        signal_pad_x_nm=right_port_x_nm,  # Simplified: pad at ref point
        signal_pad_y_nm=0,
        footprint=footprint,
        rotation_mdeg=180000,
        side="right",
    )

    # Single through segment
    through_segment = SegmentPlan(
        x_start_nm=left_port_x_nm,
        x_end_nm=right_port_x_nm,
        y_nm=0,
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
    # Derive discontinuity position
    x_disc_nm = left_port_x_nm + left_length_nm

    # Derive right length to ensure continuity
    right_length_nm = right_port_x_nm - x_disc_nm

    if right_length_nm < 0:
        raise ValueError(
            f"Left length ({left_length_nm}) places discontinuity ({x_disc_nm}) "
            f"beyond right port ({right_port_x_nm})"
        )

    left_port = PortPlan(
        x_ref_nm=left_port_x_nm,
        y_ref_nm=0,
        signal_pad_x_nm=left_port_x_nm,
        signal_pad_y_nm=0,
        footprint=footprint,
        rotation_mdeg=0,
        side="left",
    )

    right_port = PortPlan(
        x_ref_nm=right_port_x_nm,
        y_ref_nm=0,
        signal_pad_x_nm=right_port_x_nm,
        signal_pad_y_nm=0,
        footprint=footprint,
        rotation_mdeg=180000,
        side="right",
    )

    # Left segment: from left port to discontinuity
    left_segment = SegmentPlan(
        x_start_nm=left_port_x_nm,
        x_end_nm=x_disc_nm,
        y_nm=0,
        width_nm=trace_width_nm,
        layer=trace_layer,
        net_name="SIG",
        label="left",
    )

    # Right segment: from discontinuity to right port
    right_segment = SegmentPlan(
        x_start_nm=x_disc_nm,
        x_end_nm=right_port_x_nm,
        y_nm=0,
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
