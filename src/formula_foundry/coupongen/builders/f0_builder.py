"""Family F0 (Calibration Thru Line) coupon builder.

This module implements the feature composition pattern for F0 coupons:
  end-launch connector -> CPWG straight line -> end-launch connector

F0 coupons are calibration structures used to:
- Validate launch geometry
- Establish baseline insertion loss
- Calibrate measurement systems

The builder composes these features:
1. Board outline feature - defines PCB shape and dimensions
2. Port features - left and right end-launch connectors
3. Transmission line feature - CPWG segment connecting ports

No discontinuity feature is present (distinguishes F0 from F1).

Satisfies REQ-M1-006.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..families import FAMILY_F0, validate_family
from ..geom.cpwg import CPWGSpec
from ..geom.primitives import (
    BoardOutline,
    FootprintInstance,
    PositionNM,
    TrackSegment,
)
from ..spec import CouponSpec

if TYPE_CHECKING:
    from ..resolve import ResolvedDesign


@dataclass(frozen=True, slots=True)
class BoardOutlineFeature:
    """Board outline feature for F0 coupon.

    Defines the rectangular PCB shape with optional corner radius.
    Origin is at EDGE_L_CENTER (left edge, vertically centered).
    """

    width_nm: int
    length_nm: int
    corner_radius_nm: int

    def to_outline(self) -> BoardOutline:
        """Convert to BoardOutline primitive.

        Creates a rectangular board outline centered vertically (y=0).
        The origin is at the left edge center (EDGE_L_CENTER mode).
        """
        half_width = self.width_nm // 2
        return BoardOutline.rectangle(
            width_nm=self.length_nm,  # In EDGE_L_CENTER, length is the x dimension
            height_nm=self.width_nm,  # width is the y dimension
            corner_radius_nm=self.corner_radius_nm,
            origin=PositionNM(0, -half_width),  # Origin at left edge, centered
        )


@dataclass(frozen=True, slots=True)
class PortFeature:
    """Port feature representing an end-launch connector.

    Each port consists of a footprint instance placed at a specific
    position with a rotation. The footprint connects the external
    SMA connector to the on-board transmission line.

    Attributes:
        footprint: Full footprint path as 'library:name'.
        position: Footprint center position in nm.
        rotation_deg: Rotation in degrees.
        side: Port side ('left' or 'right').
    """

    footprint: str
    position: PositionNM
    rotation_deg: int
    side: str  # "left" or "right"

    @property
    def footprint_lib(self) -> str:
        """Extract library name from footprint path."""
        if ":" in self.footprint:
            return self.footprint.split(":")[0]
        return self.footprint

    @property
    def footprint_name(self) -> str:
        """Extract footprint name from footprint path."""
        if ":" in self.footprint:
            return self.footprint.split(":", 1)[1]
        return self.footprint

    @property
    def rotation_mdeg(self) -> int:
        """Rotation in millidegrees."""
        return self.rotation_deg * 1000

    def to_footprint_instance(self) -> FootprintInstance:
        """Convert to FootprintInstance primitive."""
        return FootprintInstance(
            reference=f"J_{self.side.upper()}",
            footprint_lib=self.footprint_lib,
            footprint_name=self.footprint_name,
            position=self.position,
            rotation_mdeg=self.rotation_mdeg,
            layer="F.Cu",
        )


@dataclass(frozen=True, slots=True)
class TransmissionLineFeature:
    """Transmission line feature for F0 coupon.

    Represents a CPWG (Coplanar Waveguide with Ground) transmission line
    segment connecting the left and right ports.

    For F0, the transmission line runs straight from the left connector
    to the right connector with no discontinuities.
    """

    w_nm: int
    gap_nm: int
    length_left_nm: int
    length_right_nm: int
    layer: str
    left_start_x_nm: int
    right_end_x_nm: int

    def get_cpwg_spec_left(self) -> CPWGSpec:
        """Get CPWG spec for left segment."""
        return CPWGSpec(
            w_nm=self.w_nm,
            gap_nm=self.gap_nm,
            length_nm=self.length_left_nm,
            layer=self.layer,
            net_id=1,
        )

    def get_cpwg_spec_right(self) -> CPWGSpec:
        """Get CPWG spec for right segment."""
        return CPWGSpec(
            w_nm=self.w_nm,
            gap_nm=self.gap_nm,
            length_nm=self.length_right_nm,
            layer=self.layer,
            net_id=1,
        )

    def to_track_segments(self) -> tuple[TrackSegment, TrackSegment]:
        """Generate track segments for the transmission line.

        Returns a tuple of (left_segment, right_segment).
        For F0, these segments meet at the center of the board.
        """
        # Calculate center point where segments meet
        center_x = self.left_start_x_nm + self.length_left_nm

        # Left segment: from left connector to center
        left_segment = TrackSegment(
            start=PositionNM(self.left_start_x_nm, 0),
            end=PositionNM(center_x, 0),
            width_nm=self.w_nm,
            layer=self.layer,
            net_id=1,
        )

        # Right segment: from center to right connector
        right_segment = TrackSegment(
            start=PositionNM(center_x, 0),
            end=PositionNM(self.right_end_x_nm, 0),
            width_nm=self.w_nm,
            layer=self.layer,
            net_id=1,
        )

        return (left_segment, right_segment)


@dataclass(frozen=True, slots=True)
class F0CouponComposition:
    """Complete F0 coupon feature composition.

    Contains all features needed to build an F0 calibration thru coupon:
    - Board outline
    - Left and right ports (connectors)
    - Transmission line
    """

    board_outline: BoardOutlineFeature
    left_port: PortFeature
    right_port: PortFeature
    transmission_line: TransmissionLineFeature

    @property
    def total_trace_length_nm(self) -> int:
        """Total transmission line length in nm."""
        return (
            self.transmission_line.length_left_nm
            + self.transmission_line.length_right_nm
        )


class F0CouponBuilder:
    """Builder for F0 (Calibration Thru Line) coupons.

    This builder implements the feature composition pattern:
    1. Extracts features from CouponSpec
    2. Validates F0-specific constraints
    3. Composes features into primitives
    4. Returns a complete F0CouponComposition

    Usage:
        builder = F0CouponBuilder(spec, resolved)
        composition = builder.build()
    """

    def __init__(self, spec: CouponSpec, resolved: ResolvedDesign) -> None:
        """Initialize the F0 coupon builder.

        Args:
            spec: Validated CouponSpec with coupon_family=F0_CAL_THRU_LINE.
            resolved: ResolvedDesign with all parameters in integer nm.

        Raises:
            ValueError: If spec is not an F0 coupon.
        """
        validate_family(spec)
        if spec.coupon_family != FAMILY_F0:
            raise ValueError(
                f"F0CouponBuilder requires coupon_family={FAMILY_F0!r}, "
                f"got {spec.coupon_family!r}"
            )
        self.spec = spec
        self.resolved = resolved

    def _build_board_outline(self) -> BoardOutlineFeature:
        """Extract board outline feature from spec."""
        return BoardOutlineFeature(
            width_nm=int(self.spec.board.outline.width_nm),
            length_nm=int(self.spec.board.outline.length_nm),
            corner_radius_nm=int(self.spec.board.outline.corner_radius_nm),
        )

    def _build_left_port(self) -> PortFeature:
        """Extract left port feature from spec."""
        conn = self.spec.connectors.left
        return PortFeature(
            footprint=conn.footprint,
            position=PositionNM(
                int(conn.position_nm[0]),
                int(conn.position_nm[1]),
            ),
            rotation_deg=conn.rotation_deg,
            side="left",
        )

    def _build_right_port(self) -> PortFeature:
        """Extract right port feature from spec."""
        conn = self.spec.connectors.right
        return PortFeature(
            footprint=conn.footprint,
            position=PositionNM(
                int(conn.position_nm[0]),
                int(conn.position_nm[1]),
            ),
            rotation_deg=conn.rotation_deg,
            side="right",
        )

    def _build_transmission_line(self) -> TransmissionLineFeature:
        """Extract transmission line feature from spec."""
        tl = self.spec.transmission_line
        left_conn = self.spec.connectors.left
        right_conn = self.spec.connectors.right

        return TransmissionLineFeature(
            w_nm=int(tl.w_nm),
            gap_nm=int(tl.gap_nm),
            length_left_nm=int(tl.length_left_nm),
            length_right_nm=int(tl.length_right_nm),
            layer=tl.layer,
            left_start_x_nm=int(left_conn.position_nm[0]),
            right_end_x_nm=int(right_conn.position_nm[0]),
        )

    def build(self) -> F0CouponComposition:
        """Build the complete F0 coupon composition.

        Returns:
            F0CouponComposition with all features extracted and validated.
        """
        return F0CouponComposition(
            board_outline=self._build_board_outline(),
            left_port=self._build_left_port(),
            right_port=self._build_right_port(),
            transmission_line=self._build_transmission_line(),
        )


def build_f0_coupon(spec: CouponSpec, resolved: ResolvedDesign) -> F0CouponComposition:
    """Convenience function to build an F0 coupon composition.

    Args:
        spec: Validated CouponSpec with coupon_family=F0_CAL_THRU_LINE.
        resolved: ResolvedDesign with all parameters in integer nm.

    Returns:
        F0CouponComposition with all features.

    Raises:
        ValueError: If spec is not an F0 coupon or validation fails.
    """
    builder = F0CouponBuilder(spec, resolved)
    return builder.build()
