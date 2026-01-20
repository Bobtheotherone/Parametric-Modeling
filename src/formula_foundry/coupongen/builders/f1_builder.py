"""Family F1 (Single-Ended Via Transition) coupon builder.

This module implements the feature composition pattern for F1 coupons:
  end-launch connector -> CPWG -> via transition -> CPWG -> end-launch connector

F1 coupons are single-ended via transition structures used to:
- Characterize via transition impedance discontinuities
- Measure via-induced reflections and insertion loss
- Validate antipad and return via geometries

The builder composes these features:
1. Board outline feature - defines PCB shape and dimensions
2. Port features - left and right end-launch connectors
3. Transmission line feature - CPWG segments on each side of the via
4. Discontinuity feature - via transition with signal via, return vias, antipads
5. Cutout features - plane cutouts/antipads on internal layers

Satisfies REQ-M1-007.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..families import FAMILY_F1, validate_family
from ..geom.cpwg import CPWGSpec
from ..geom.cutouts import (
    CircleAntipadSpec,
    CutoutShape,
    RoundRectAntipadSpec,
    SlotAntipadSpec,
    generate_circle_antipad,
    generate_roundrect_antipad,
    generate_slot_antipad,
)
from ..geom.primitives import (
    BoardOutline,
    FootprintInstance,
    Polygon,
    PositionNM,
    TrackSegment,
    Via,
)
from ..geom.via_patterns import (
    ReturnViaPattern,
    ReturnViaRingSpec,
    ReturnViaSpec,
    SignalViaSpec,
    ViaTransitionResult,
    generate_return_via_ring,
    generate_via_transition,
)
from ..spec import CouponSpec

if TYPE_CHECKING:
    from ..resolve import ResolvedDesign


@dataclass(frozen=True, slots=True)
class BoardOutlineFeature:
    """Board outline feature for F1 coupon.

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
    """Transmission line feature for F1 coupon.

    Represents the CPWG (Coplanar Waveguide with Ground) transmission line
    segments on each side of the via transition. For F1, there are two
    separate segments that connect the ports to the discontinuity center.

    Attributes:
        w_nm: Trace width in nanometers.
        gap_nm: Gap to coplanar ground in nanometers.
        length_left_nm: Length of left segment in nanometers.
        length_right_nm: Length of right segment in nanometers.
        layer: Copper layer name.
        left_start_x_nm: X position where left trace starts (connector position).
        right_end_x_nm: X position where right trace ends (connector position).
        discontinuity_x_nm: X position of the via transition center.
    """

    w_nm: int
    gap_nm: int
    length_left_nm: int
    length_right_nm: int
    layer: str
    left_start_x_nm: int
    right_end_x_nm: int
    discontinuity_x_nm: int

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
        For F1, segments connect the connectors to the via transition center.
        """
        # Left segment: from left connector to discontinuity
        left_segment = TrackSegment(
            start=PositionNM(self.left_start_x_nm, 0),
            end=PositionNM(self.discontinuity_x_nm, 0),
            width_nm=self.w_nm,
            layer=self.layer,
            net_id=1,
        )

        # Right segment: from discontinuity to right connector
        right_segment = TrackSegment(
            start=PositionNM(self.discontinuity_x_nm, 0),
            end=PositionNM(self.right_end_x_nm, 0),
            width_nm=self.w_nm,
            layer=self.layer,
            net_id=1,
        )

        return (left_segment, right_segment)


@dataclass(frozen=True, slots=True)
class SignalViaFeature:
    """Signal via feature for the via transition.

    Represents the primary signal via that transitions between layers.

    Attributes:
        drill_nm: Drill diameter in nanometers.
        diameter_nm: Via pad diameter in nanometers.
        pad_diameter_nm: Landing pad diameter in nanometers.
        position: Via center position in nanometers.
        layers: Tuple of layers the via connects.
        net_id: Net ID for the signal via.
    """

    drill_nm: int
    diameter_nm: int
    pad_diameter_nm: int
    position: PositionNM
    layers: tuple[str, str] = ("F.Cu", "B.Cu")
    net_id: int = 1

    def to_via(self) -> Via:
        """Convert to Via primitive."""
        return Via(
            position=self.position,
            diameter_nm=self.diameter_nm,
            drill_nm=self.drill_nm,
            layers=self.layers,
            net_id=self.net_id,
        )

    def to_signal_via_spec(self) -> SignalViaSpec:
        """Convert to SignalViaSpec for pattern generation."""
        return SignalViaSpec(
            drill_nm=self.drill_nm,
            diameter_nm=self.diameter_nm,
            pad_diameter_nm=self.pad_diameter_nm,
            layers=self.layers,
            net_id=self.net_id,
        )


@dataclass(frozen=True, slots=True)
class ReturnViasFeature:
    """Return vias feature for the via transition.

    Represents the return current vias arranged around the signal via
    to provide a low-inductance return path and control the via impedance.

    Attributes:
        pattern: Return via pattern type (RING, GRID, QUADRANT).
        count: Number of return vias (for RING pattern).
        radius_nm: Distance from signal via center to return via centers.
        drill_nm: Return via drill diameter in nanometers.
        diameter_nm: Return via pad diameter in nanometers.
        center: Center position (same as signal via) in nanometers.
        layers: Tuple of layers the return vias connect.
        net_id: Net ID for the return vias (typically ground).
    """

    pattern: str
    count: int
    radius_nm: int
    drill_nm: int
    diameter_nm: int
    center: PositionNM
    layers: tuple[str, str] = ("F.Cu", "B.Cu")
    net_id: int = 0  # Ground net

    def to_return_via_ring_spec(self) -> ReturnViaRingSpec:
        """Convert to ReturnViaRingSpec for pattern generation."""
        return ReturnViaRingSpec(
            pattern=ReturnViaPattern.RING,
            count=self.count,
            radius_nm=self.radius_nm,
            via=ReturnViaSpec(
                drill_nm=self.drill_nm,
                diameter_nm=self.diameter_nm,
                layers=self.layers,
                net_id=self.net_id,
            ),
        )

    def generate_vias(self) -> tuple[Via, ...]:
        """Generate return via primitives."""
        ring_spec = self.to_return_via_ring_spec()
        return generate_return_via_ring(self.center, ring_spec)


@dataclass(frozen=True, slots=True)
class AntipadFeature:
    """Antipad feature for a specific layer.

    Represents a copper-free area (cutout) around the signal via on a
    reference plane layer to control capacitance and impedance.

    Attributes:
        layer: Layer name (e.g., "In1.Cu", "In2.Cu").
        shape: Antipad shape type.
        center: Antipad center position in nanometers.
        r_nm: Radius for CIRCLE shape (optional).
        rx_nm: Half-width for ROUNDRECT shape (optional).
        ry_nm: Half-height for ROUNDRECT shape (optional).
        corner_nm: Corner radius for ROUNDRECT shape (optional).
    """

    layer: str
    shape: str
    center: PositionNM
    r_nm: int | None = None
    rx_nm: int | None = None
    ry_nm: int | None = None
    corner_nm: int | None = None

    def to_polygon(self) -> Polygon:
        """Generate antipad polygon primitive."""
        if self.shape == "CIRCLE":
            if self.r_nm is None:
                raise ValueError("r_nm required for CIRCLE antipad")
            spec = CircleAntipadSpec(
                shape=CutoutShape.CIRCLE,
                r_nm=self.r_nm,
                layer=self.layer,
            )
            return generate_circle_antipad(self.center, spec)

        elif self.shape == "ROUNDRECT":
            if self.rx_nm is None or self.ry_nm is None:
                raise ValueError("rx_nm and ry_nm required for ROUNDRECT antipad")
            corner = self.corner_nm if self.corner_nm is not None else 0
            spec = RoundRectAntipadSpec(
                shape=CutoutShape.ROUNDRECT,
                rx_nm=self.rx_nm,
                ry_nm=self.ry_nm,
                corner_nm=corner,
                layer=self.layer,
            )
            return generate_roundrect_antipad(self.center, spec)

        else:
            raise ValueError(f"Unsupported antipad shape: {self.shape}")


@dataclass(frozen=True, slots=True)
class PlaneCutoutFeature:
    """Plane cutout feature for a specific layer.

    Represents a slot-shaped or rectangular cutout in a reference plane,
    typically used for transition impedance tuning or thermal relief.

    Attributes:
        layer: Layer name (e.g., "In1.Cu", "In2.Cu").
        shape: Cutout shape type.
        center: Cutout center position in nanometers.
        length_nm: Cutout length in nanometers.
        width_nm: Cutout width in nanometers.
        rotation_deg: Rotation in degrees.
    """

    layer: str
    shape: str
    center: PositionNM
    length_nm: int
    width_nm: int
    rotation_deg: int

    def to_polygon(self) -> Polygon:
        """Generate cutout polygon primitive."""
        if self.shape == "SLOT":
            spec = SlotAntipadSpec(
                shape=CutoutShape.SLOT,
                length_nm=self.length_nm,
                width_nm=self.width_nm,
                rotation_mdeg=self.rotation_deg * 1000,
                layer=self.layer,
            )
            return generate_slot_antipad(self.center, spec)

        elif self.shape == "RECTANGLE":
            from ..geom.cutouts import RectangleAntipadSpec, generate_rectangle_antipad

            spec = RectangleAntipadSpec(
                shape=CutoutShape.RECTANGLE,
                width_nm=self.length_nm,
                height_nm=self.width_nm,
                layer=self.layer,
            )
            return generate_rectangle_antipad(self.center, spec)

        else:
            raise ValueError(f"Unsupported cutout shape: {self.shape}")


@dataclass(frozen=True, slots=True)
class DiscontinuityFeature:
    """Complete discontinuity feature for F1 via transition.

    Combines the signal via, return vias, antipads, and plane cutouts
    into a single cohesive feature representing the via transition.

    Attributes:
        signal_via: Signal via feature.
        return_vias: Return vias feature (optional).
        antipads: Tuple of antipad features per layer.
        plane_cutouts: Tuple of plane cutout features per layer.
        center: Discontinuity center position in nanometers.
    """

    signal_via: SignalViaFeature
    return_vias: ReturnViasFeature | None
    antipads: tuple[AntipadFeature, ...]
    plane_cutouts: tuple[PlaneCutoutFeature, ...]
    center: PositionNM

    def generate_via_transition(self) -> ViaTransitionResult:
        """Generate via transition geometry.

        Returns ViaTransitionResult with signal via and return vias.
        """
        signal_spec = self.signal_via.to_signal_via_spec()
        return_ring_spec = self.return_vias.to_return_via_ring_spec() if self.return_vias is not None else None
        return generate_via_transition(self.center, signal_spec, return_ring_spec)

    def generate_antipad_polygons(self) -> tuple[Polygon, ...]:
        """Generate all antipad polygons."""
        return tuple(ap.to_polygon() for ap in self.antipads)

    def generate_cutout_polygons(self) -> tuple[Polygon, ...]:
        """Generate all plane cutout polygons."""
        return tuple(pc.to_polygon() for pc in self.plane_cutouts)


@dataclass(frozen=True, slots=True)
class F1CouponComposition:
    """Complete F1 coupon feature composition.

    Contains all features needed to build an F1 single-ended via
    transition coupon:
    - Board outline
    - Left and right ports (connectors)
    - Transmission line (two segments)
    - Discontinuity (via transition with return vias, antipads, cutouts)
    """

    board_outline: BoardOutlineFeature
    left_port: PortFeature
    right_port: PortFeature
    transmission_line: TransmissionLineFeature
    discontinuity: DiscontinuityFeature

    @property
    def total_trace_length_nm(self) -> int:
        """Total transmission line length in nm."""
        return self.transmission_line.length_left_nm + self.transmission_line.length_right_nm

    @property
    def discontinuity_position(self) -> PositionNM:
        """Position of the via transition center."""
        return self.discontinuity.center

    @property
    def signal_via(self) -> Via:
        """Get the signal via primitive."""
        return self.discontinuity.signal_via.to_via()

    @property
    def return_vias(self) -> tuple[Via, ...]:
        """Get all return via primitives."""
        if self.discontinuity.return_vias is None:
            return ()
        return self.discontinuity.return_vias.generate_vias()

    @property
    def all_antipads(self) -> tuple[Polygon, ...]:
        """Get all antipad polygons."""
        return self.discontinuity.generate_antipad_polygons()

    @property
    def all_cutouts(self) -> tuple[Polygon, ...]:
        """Get all plane cutout polygons."""
        return self.discontinuity.generate_cutout_polygons()


class F1CouponBuilder:
    """Builder for F1 (Single-Ended Via Transition) coupons.

    This builder implements the feature composition pattern:
    1. Extracts features from CouponSpec
    2. Validates F1-specific constraints
    3. Composes features into primitives
    4. Returns a complete F1CouponComposition

    Usage:
        builder = F1CouponBuilder(spec, resolved)
        composition = builder.build()
    """

    def __init__(self, spec: CouponSpec, resolved: ResolvedDesign) -> None:
        """Initialize the F1 coupon builder.

        Args:
            spec: Validated CouponSpec with coupon_family=F1_SINGLE_ENDED_VIA.
            resolved: ResolvedDesign with all parameters in integer nm.

        Raises:
            ValueError: If spec is not an F1 coupon.
        """
        validate_family(spec)
        if spec.coupon_family != FAMILY_F1:
            raise ValueError(f"F1CouponBuilder requires coupon_family={FAMILY_F1!r}, got {spec.coupon_family!r}")
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

    def _calculate_discontinuity_center(self) -> PositionNM:
        """Calculate the via transition center position.

        The discontinuity is placed at the end of the left transmission
        line segment (which is also the start of the right segment).
        """
        left_conn_x = int(self.spec.connectors.left.position_nm[0])
        left_length = int(self.spec.transmission_line.length_left_nm)
        center_x = left_conn_x + left_length
        return PositionNM(center_x, 0)

    def _build_transmission_line(self, discontinuity_x_nm: int) -> TransmissionLineFeature:
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
            discontinuity_x_nm=discontinuity_x_nm,
        )

    def _build_signal_via(self, center: PositionNM) -> SignalViaFeature:
        """Extract signal via feature from spec."""
        disc = self.spec.discontinuity
        assert disc is not None  # Validated by F1 family check

        return SignalViaFeature(
            drill_nm=int(disc.signal_via.drill_nm),
            diameter_nm=int(disc.signal_via.diameter_nm),
            pad_diameter_nm=int(disc.signal_via.pad_diameter_nm),
            position=center,
        )

    def _build_return_vias(self, center: PositionNM) -> ReturnViasFeature | None:
        """Extract return vias feature from spec."""
        disc = self.spec.discontinuity
        assert disc is not None  # Validated by F1 family check

        if disc.return_vias is None:
            return None

        return ReturnViasFeature(
            pattern=disc.return_vias.pattern,
            count=disc.return_vias.count,
            radius_nm=int(disc.return_vias.radius_nm),
            drill_nm=int(disc.return_vias.via.drill_nm),
            diameter_nm=int(disc.return_vias.via.diameter_nm),
            center=center,
        )

    def _build_antipads(self, center: PositionNM) -> tuple[AntipadFeature, ...]:
        """Extract antipad features from spec."""
        disc = self.spec.discontinuity
        assert disc is not None  # Validated by F1 family check

        antipads: list[AntipadFeature] = []
        for layer, ap in disc.antipads.items():
            antipad = AntipadFeature(
                layer=layer,
                shape=ap.shape,
                center=center,
                r_nm=int(ap.r_nm) if ap.r_nm is not None else None,
                rx_nm=int(ap.rx_nm) if ap.rx_nm is not None else None,
                ry_nm=int(ap.ry_nm) if ap.ry_nm is not None else None,
                corner_nm=int(ap.corner_nm) if ap.corner_nm is not None else None,
            )
            antipads.append(antipad)

        return tuple(antipads)

    def _build_plane_cutouts(self, center: PositionNM) -> tuple[PlaneCutoutFeature, ...]:
        """Extract plane cutout features from spec."""
        disc = self.spec.discontinuity
        assert disc is not None  # Validated by F1 family check

        cutouts: list[PlaneCutoutFeature] = []
        for layer, pc in disc.plane_cutouts.items():
            cutout = PlaneCutoutFeature(
                layer=layer,
                shape=pc.shape,
                center=center,
                length_nm=int(pc.length_nm),
                width_nm=int(pc.width_nm),
                rotation_deg=pc.rotation_deg,
            )
            cutouts.append(cutout)

        return tuple(cutouts)

    def _build_discontinuity(self, center: PositionNM) -> DiscontinuityFeature:
        """Build complete discontinuity feature."""
        return DiscontinuityFeature(
            signal_via=self._build_signal_via(center),
            return_vias=self._build_return_vias(center),
            antipads=self._build_antipads(center),
            plane_cutouts=self._build_plane_cutouts(center),
            center=center,
        )

    def build(self) -> F1CouponComposition:
        """Build the complete F1 coupon composition.

        Returns:
            F1CouponComposition with all features extracted and validated.
        """
        # Calculate discontinuity position first
        disc_center = self._calculate_discontinuity_center()

        return F1CouponComposition(
            board_outline=self._build_board_outline(),
            left_port=self._build_left_port(),
            right_port=self._build_right_port(),
            transmission_line=self._build_transmission_line(disc_center.x),
            discontinuity=self._build_discontinuity(disc_center),
        )


def build_f1_coupon(spec: CouponSpec, resolved: ResolvedDesign) -> F1CouponComposition:
    """Convenience function to build an F1 coupon composition.

    Args:
        spec: Validated CouponSpec with coupon_family=F1_SINGLE_ENDED_VIA.
        resolved: ResolvedDesign with all parameters in integer nm.

    Returns:
        F1CouponComposition with all features.

    Raises:
        ValueError: If spec is not an F1 coupon or validation fails.
    """
    builder = F1CouponBuilder(spec, resolved)
    return builder.build()
