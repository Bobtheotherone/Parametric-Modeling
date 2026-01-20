"""Internal representation (IR) primitives for coupon geometry.

All coordinates are in integer nanometers (nm) to ensure:
- Determinism across platforms (no floating-point drift)
- Meaningful canonical hashing
- Exact clearance proofs

The canonical coordinate frame uses EDGE_L_CENTER origin:
- Origin at left board edge center
- +x direction to the right along coupon length
- +y upward (right-handed 2D)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class OriginMode(Enum):
    """Origin placement mode for the canonical coordinate frame."""

    EDGE_L_CENTER = "EDGE_L_CENTER"
    """Origin at left board edge center (default for all coupons)."""

    EDGE_R_CENTER = "EDGE_R_CENTER"
    """Origin at right board edge center."""

    CENTER = "CENTER"
    """Origin at board center."""

    BOTTOM_LEFT = "BOTTOM_LEFT"
    """Origin at bottom-left corner (KiCad default)."""


class PadShape(Enum):
    """Pad shape types matching KiCad pad shapes."""

    CIRCLE = "circle"
    RECT = "rect"
    ROUNDRECT = "roundrect"
    OVAL = "oval"
    TRAPEZOID = "trapezoid"
    CUSTOM = "custom"


class PolygonType(Enum):
    """Polygon usage type in the board."""

    COPPER_POUR = "copper_pour"
    """Filled copper zone."""

    CUTOUT = "cutout"
    """Copper-free area (antipad/clearance)."""

    KEEPOUT = "keepout"
    """Design rule area - no copper placement allowed."""


class RuleAreaType(Enum):
    """Rule area restriction types."""

    NO_TRACKS = "no_tracks"
    NO_VIAS = "no_vias"
    NO_COPPER_POUR = "no_copper_pour"
    NO_FOOTPRINTS = "no_footprints"


class TextLayer(Enum):
    """Text placement layer."""

    F_SILKSCREEN = "F.SilkS"
    B_SILKSCREEN = "B.SilkS"
    F_FAB = "F.Fab"
    B_FAB = "B.Fab"
    F_CU = "F.Cu"
    B_CU = "B.Cu"
    COMMENTS = "Cmts.User"


class TextJustify(Enum):
    """Text justification."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class PositionNM:
    """2D position in integer nanometers.

    Attributes:
        x: X coordinate in nanometers.
        y: Y coordinate in nanometers.
    """

    x: int
    y: int

    def __add__(self, other: PositionNM) -> PositionNM:
        return PositionNM(self.x + other.x, self.y + other.y)

    def __sub__(self, other: PositionNM) -> PositionNM:
        return PositionNM(self.x - other.x, self.y - other.y)

    def __neg__(self) -> PositionNM:
        return PositionNM(-self.x, -self.y)

    def scale(self, factor: int) -> PositionNM:
        """Scale position by an integer factor."""
        return PositionNM(self.x * factor, self.y * factor)

    def to_tuple(self) -> tuple[int, int]:
        """Return position as (x, y) tuple."""
        return (self.x, self.y)

    @classmethod
    def from_tuple(cls, xy: tuple[int, int]) -> PositionNM:
        """Create PositionNM from (x, y) tuple."""
        return cls(xy[0], xy[1])


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    """Coordinate frame transformation for converting between reference frames.

    The canonical frame for all coupons uses EDGE_L_CENTER:
    - Origin at left board edge center
    - +x to the right, +y upward

    Attributes:
        origin_mode: The origin placement mode.
        board_width_nm: Board width in nanometers.
        board_height_nm: Board height in nanometers.
    """

    origin_mode: OriginMode
    board_width_nm: int
    board_height_nm: int

    def to_kicad(self, pos: PositionNM) -> PositionNM:
        """Transform from canonical frame to KiCad coordinates.

        KiCad uses bottom-left origin with +y downward (screen coordinates).
        We convert from our right-handed +y-up system.
        """
        # First translate from our origin mode to bottom-left
        if self.origin_mode == OriginMode.EDGE_L_CENTER:
            # Our origin is at (0, board_height/2) in bottom-left coords
            x_bl = pos.x
            y_bl = self.board_height_nm // 2 + pos.y
        elif self.origin_mode == OriginMode.EDGE_R_CENTER:
            x_bl = self.board_width_nm + pos.x
            y_bl = self.board_height_nm // 2 + pos.y
        elif self.origin_mode == OriginMode.CENTER:
            x_bl = self.board_width_nm // 2 + pos.x
            y_bl = self.board_height_nm // 2 + pos.y
        elif self.origin_mode == OriginMode.BOTTOM_LEFT:
            x_bl = pos.x
            y_bl = pos.y
        else:
            raise ValueError(f"Unsupported origin mode: {self.origin_mode}")

        # KiCad uses +y downward, so flip y
        y_kicad = self.board_height_nm - y_bl
        return PositionNM(x_bl, y_kicad)

    def from_kicad(self, pos: PositionNM) -> PositionNM:
        """Transform from KiCad coordinates to canonical frame."""
        # KiCad uses +y downward, so flip y first
        y_bl = self.board_height_nm - pos.y
        x_bl = pos.x

        # Then translate from bottom-left to our origin mode
        if self.origin_mode == OriginMode.EDGE_L_CENTER:
            return PositionNM(x_bl, y_bl - self.board_height_nm // 2)
        elif self.origin_mode == OriginMode.EDGE_R_CENTER:
            return PositionNM(x_bl - self.board_width_nm, y_bl - self.board_height_nm // 2)
        elif self.origin_mode == OriginMode.CENTER:
            return PositionNM(x_bl - self.board_width_nm // 2, y_bl - self.board_height_nm // 2)
        elif self.origin_mode == OriginMode.BOTTOM_LEFT:
            return PositionNM(x_bl, y_bl)
        else:
            raise ValueError(f"Unsupported origin mode: {self.origin_mode}")


@dataclass(frozen=True, slots=True)
class Net:
    """Net (electrical connection) declaration.

    Attributes:
        id: Unique integer ID for the net (0 is typically unconnected).
        name: Net name string.
    """

    id: int
    name: str


@dataclass(frozen=True, slots=True)
class NetClass:
    """Net class for grouping nets with common design rules.

    Attributes:
        name: Net class name.
        description: Human-readable description.
        clearance_nm: Minimum clearance to other nets in nm.
        track_width_nm: Default track width in nm.
        via_diameter_nm: Default via pad diameter in nm.
        via_drill_nm: Default via drill diameter in nm.
        diff_pair_width_nm: Differential pair track width in nm (optional).
        diff_pair_gap_nm: Differential pair gap in nm (optional).
        net_names: List of net names belonging to this class.
    """

    name: str
    description: str
    clearance_nm: int
    track_width_nm: int
    via_diameter_nm: int
    via_drill_nm: int
    diff_pair_width_nm: int | None = None
    diff_pair_gap_nm: int | None = None
    net_names: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Pad:
    """Pad primitive within a footprint.

    Attributes:
        number: Pad number/name (e.g., "1", "2", "A1").
        shape: Pad shape type.
        position: Center position relative to footprint origin in nm.
        size_x_nm: Pad width in nm.
        size_y_nm: Pad height in nm.
        drill_nm: Drill diameter for through-hole pads (None for SMD).
        rotation_mdeg: Rotation in millidegrees.
        layers: Tuple of layer names the pad exists on.
        net_id: Net ID this pad is connected to (0 for unconnected).
        roundrect_ratio: Corner radius ratio for roundrect pads (0.0-0.5).
    """

    number: str
    shape: PadShape
    position: PositionNM
    size_x_nm: int
    size_y_nm: int
    drill_nm: int | None = None
    rotation_mdeg: int = 0
    layers: tuple[str, ...] = field(default_factory=lambda: ("F.Cu", "B.Cu", "*.Mask", "*.Paste"))
    net_id: int = 0
    roundrect_ratio: float = 0.25


@dataclass(frozen=True, slots=True)
class FootprintInstance:
    """Placed footprint instance on the board.

    Attributes:
        reference: Reference designator (e.g., "J1", "U1").
        footprint_lib: Footprint library name.
        footprint_name: Footprint name within the library.
        position: Footprint origin position in nm.
        rotation_mdeg: Rotation in millidegrees.
        layer: Placement layer ("F.Cu" or "B.Cu").
        pads: Tuple of pads in this footprint.
    """

    reference: str
    footprint_lib: str
    footprint_name: str
    position: PositionNM
    rotation_mdeg: int = 0
    layer: str = "F.Cu"
    pads: tuple[Pad, ...] = field(default_factory=tuple)

    @property
    def footprint_path(self) -> str:
        """Full footprint path as 'lib:name'."""
        return f"{self.footprint_lib}:{self.footprint_name}"


@dataclass(frozen=True, slots=True)
class TrackSegment:
    """Straight track segment on a copper layer.

    Attributes:
        start: Start position in nm.
        end: End position in nm.
        width_nm: Track width in nm.
        layer: Copper layer name.
        net_id: Net ID this track belongs to.
    """

    start: PositionNM
    end: PositionNM
    width_nm: int
    layer: str
    net_id: int = 0


@dataclass(frozen=True, slots=True)
class ArcTrack:
    """Arc-shaped track segment on a copper layer.

    Defined by start, mid (on arc), and end points.

    Attributes:
        start: Start position in nm.
        mid: Point on the arc (not center) in nm.
        end: End position in nm.
        width_nm: Track width in nm.
        layer: Copper layer name.
        net_id: Net ID this track belongs to.
    """

    start: PositionNM
    mid: PositionNM
    end: PositionNM
    width_nm: int
    layer: str
    net_id: int = 0


@dataclass(frozen=True, slots=True)
class Via:
    """Via connecting multiple copper layers.

    Attributes:
        position: Via center position in nm.
        diameter_nm: Via pad diameter in nm.
        drill_nm: Drill hole diameter in nm.
        layers: Tuple of layer names this via connects.
        net_id: Net ID this via belongs to.
        via_type: Via type ("through", "blind", "buried", "micro").
    """

    position: PositionNM
    diameter_nm: int
    drill_nm: int
    layers: tuple[str, str] = ("F.Cu", "B.Cu")
    net_id: int = 0
    via_type: str = "through"


@dataclass(frozen=True, slots=True)
class Polygon:
    """Polygon for copper pours, cutouts, or keepouts.

    Attributes:
        vertices: Tuple of vertex positions in nm (ordered, closed polygon).
        layer: Layer the polygon is on.
        polygon_type: Usage type (copper pour, cutout, etc.).
        net_id: Net ID for copper pour polygons.
        min_thickness_nm: Minimum fill thickness in nm.
        clearance_nm: Clearance to other nets in nm.
        fill_type: Fill pattern ("solid" or "hatch").
    """

    vertices: tuple[PositionNM, ...]
    layer: str
    polygon_type: PolygonType = PolygonType.COPPER_POUR
    net_id: int = 0
    min_thickness_nm: int = 100_000  # 0.1mm default
    clearance_nm: int = 200_000  # 0.2mm default
    fill_type: str = "solid"


@dataclass(frozen=True, slots=True)
class BoardOutline:
    """Board outline/edge cuts polygon.

    Attributes:
        vertices: Tuple of vertex positions in nm defining the outline.
        corner_radius_nm: Rounded corner radius in nm (0 for sharp corners).
    """

    vertices: tuple[PositionNM, ...]
    corner_radius_nm: int = 0

    @classmethod
    def rectangle(
        cls,
        width_nm: int,
        height_nm: int,
        corner_radius_nm: int = 0,
        origin: PositionNM | None = None,
    ) -> BoardOutline:
        """Create a rectangular board outline.

        Args:
            width_nm: Board width in nm.
            height_nm: Board height in nm.
            corner_radius_nm: Corner radius in nm.
            origin: Bottom-left corner position (default: (0, 0)).

        Returns:
            BoardOutline with rectangular vertices.
        """
        if origin is None:
            origin = PositionNM(0, 0)
        return cls(
            vertices=(
                origin,
                PositionNM(origin.x + width_nm, origin.y),
                PositionNM(origin.x + width_nm, origin.y + height_nm),
                PositionNM(origin.x, origin.y + height_nm),
            ),
            corner_radius_nm=corner_radius_nm,
        )


@dataclass(frozen=True, slots=True)
class Text:
    """Text annotation on the board.

    Attributes:
        content: Text string content.
        position: Text anchor position in nm.
        layer: Layer for the text.
        font_size_nm: Font height in nm.
        font_width_nm: Font stroke width in nm.
        rotation_mdeg: Text rotation in millidegrees.
        justify: Text justification.
        mirror: Whether text is mirrored.
    """

    content: str
    position: PositionNM
    layer: TextLayer = TextLayer.F_SILKSCREEN
    font_size_nm: int = 1_000_000  # 1mm default
    font_width_nm: int = 150_000  # 0.15mm default
    rotation_mdeg: int = 0
    justify: TextJustify = TextJustify.CENTER
    mirror: bool = False


@dataclass(frozen=True, slots=True)
class RuleArea:
    """Design rule area/keepout zone.

    Attributes:
        vertices: Tuple of vertex positions in nm defining the area.
        layer: Layer the rule area applies to (or "all").
        rule_types: Tuple of restriction types.
        name: Optional name for the rule area.
    """

    vertices: tuple[PositionNM, ...]
    layer: str
    rule_types: tuple[RuleAreaType, ...] = (RuleAreaType.NO_COPPER_POUR,)
    name: str = ""


def create_coordinate_frame(
    origin_mode: OriginMode | str,
    board_width_nm: int,
    board_height_nm: int,
) -> CoordinateFrame:
    """Create a coordinate frame for the given board dimensions.

    Args:
        origin_mode: Origin placement mode (enum or string).
        board_width_nm: Board width in nanometers.
        board_height_nm: Board height in nanometers.

    Returns:
        CoordinateFrame configured for the specified origin.
    """
    if isinstance(origin_mode, str):
        origin_mode = OriginMode(origin_mode)
    return CoordinateFrame(
        origin_mode=origin_mode,
        board_width_nm=board_width_nm,
        board_height_nm=board_height_nm,
    )


def transform_primitives_to_kicad(
    primitives: Sequence[TrackSegment | Via | FootprintInstance | Text],
    frame: CoordinateFrame,
) -> list[TrackSegment | Via | FootprintInstance | Text]:
    """Transform a sequence of primitives to KiCad coordinates.

    Args:
        primitives: Sequence of geometry primitives.
        frame: Coordinate frame for transformation.

    Returns:
        List of transformed primitives.
    """
    result: list[TrackSegment | Via | FootprintInstance | Text] = []

    for prim in primitives:
        if isinstance(prim, TrackSegment):
            result.append(
                TrackSegment(
                    start=frame.to_kicad(prim.start),
                    end=frame.to_kicad(prim.end),
                    width_nm=prim.width_nm,
                    layer=prim.layer,
                    net_id=prim.net_id,
                )
            )
        elif isinstance(prim, Via):
            result.append(
                Via(
                    position=frame.to_kicad(prim.position),
                    diameter_nm=prim.diameter_nm,
                    drill_nm=prim.drill_nm,
                    layers=prim.layers,
                    net_id=prim.net_id,
                    via_type=prim.via_type,
                )
            )
        elif isinstance(prim, FootprintInstance):
            result.append(
                FootprintInstance(
                    reference=prim.reference,
                    footprint_lib=prim.footprint_lib,
                    footprint_name=prim.footprint_name,
                    position=frame.to_kicad(prim.position),
                    rotation_mdeg=prim.rotation_mdeg,
                    layer=prim.layer,
                    pads=prim.pads,
                )
            )
        elif isinstance(prim, Text):
            result.append(
                Text(
                    content=prim.content,
                    position=frame.to_kicad(prim.position),
                    layer=prim.layer,
                    font_size_nm=prim.font_size_nm,
                    font_width_nm=prim.font_width_nm,
                    rotation_mdeg=prim.rotation_mdeg,
                    justify=prim.justify,
                    mirror=prim.mirror,
                )
            )
        else:
            # Pass through unchanged for types not explicitly handled
            result.append(prim)

    return result
