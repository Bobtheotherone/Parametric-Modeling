# SPDX-License-Identifier: MIT
"""Unit tests for coupongen geometry primitives module.

Tests the IR (Internal Representation) primitives used for coupon geometry:
- PositionNM: 2D position in nanometers
- CoordinateFrame: Coordinate frame transformations
- BoardOutline: Board edge cuts geometry
- Enums: OriginMode, PadShape, PolygonType, etc.
- Helper functions: half_width_nm, create_coordinate_frame

All geometry uses integer nanometers for determinism and hashability.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

# Direct import to avoid broken import chain in formula_foundry.__init__
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _load_module(name: str, path: Path) -> ModuleType:
    """Load a module from file with proper sys.modules registration."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {name} at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Register before exec to handle dataclasses
    spec.loader.exec_module(module)
    return module


# Load primitives module directly
_primitives = _load_module(
    "formula_foundry.coupongen.geom.primitives",
    _SRC_DIR / "formula_foundry" / "coupongen" / "geom" / "primitives.py",
)

# Import classes/functions from loaded module
PositionNM = _primitives.PositionNM
OriginMode = _primitives.OriginMode
PadShape = _primitives.PadShape
CoordinateFrame = _primitives.CoordinateFrame
BoardOutline = _primitives.BoardOutline
half_width_nm = _primitives.half_width_nm
create_coordinate_frame = _primitives.create_coordinate_frame
Net = _primitives.Net
NetClass = _primitives.NetClass
Via = _primitives.Via
TrackSegment = _primitives.TrackSegment
Text = _primitives.Text
TextLayer = _primitives.TextLayer


# =============================================================================
# PositionNM tests
# =============================================================================


class TestPositionNM:
    """Tests for PositionNM dataclass."""

    def test_create_position(self) -> None:
        """Create position with x and y coordinates."""
        pos = PositionNM(1000, 2000)
        assert pos.x == 1000
        assert pos.y == 2000

    def test_position_frozen(self) -> None:
        """Position should be immutable (frozen dataclass)."""
        pos = PositionNM(1000, 2000)
        with pytest.raises(AttributeError):
            pos.x = 5000  # type: ignore[misc]

    def test_add_positions(self) -> None:
        """Adding two positions returns sum of coordinates."""
        pos1 = PositionNM(1000, 2000)
        pos2 = PositionNM(500, 300)
        result = pos1 + pos2
        assert result.x == 1500
        assert result.y == 2300

    def test_subtract_positions(self) -> None:
        """Subtracting positions returns difference of coordinates."""
        pos1 = PositionNM(1000, 2000)
        pos2 = PositionNM(400, 500)
        result = pos1 - pos2
        assert result.x == 600
        assert result.y == 1500

    def test_negate_position(self) -> None:
        """Negating position returns negated coordinates."""
        pos = PositionNM(1000, -2000)
        result = -pos
        assert result.x == -1000
        assert result.y == 2000

    def test_scale_position(self) -> None:
        """Scaling position multiplies coordinates."""
        pos = PositionNM(100, 200)
        result = pos.scale(3)
        assert result.x == 300
        assert result.y == 600

    def test_to_tuple(self) -> None:
        """Convert position to (x, y) tuple."""
        pos = PositionNM(1000, 2000)
        result = pos.to_tuple()
        assert result == (1000, 2000)

    def test_from_tuple(self) -> None:
        """Create position from (x, y) tuple."""
        pos = PositionNM.from_tuple((1000, 2000))
        assert pos.x == 1000
        assert pos.y == 2000

    def test_position_equality(self) -> None:
        """Equal positions should compare equal."""
        pos1 = PositionNM(1000, 2000)
        pos2 = PositionNM(1000, 2000)
        assert pos1 == pos2

    def test_position_inequality(self) -> None:
        """Different positions should not be equal."""
        pos1 = PositionNM(1000, 2000)
        pos2 = PositionNM(1000, 3000)
        assert pos1 != pos2

    def test_position_hashable(self) -> None:
        """Position should be hashable for use in sets/dicts."""
        pos = PositionNM(1000, 2000)
        pos_set = {pos}
        assert pos in pos_set


# =============================================================================
# OriginMode enum tests
# =============================================================================


class TestOriginMode:
    """Tests for OriginMode enum."""

    def test_edge_l_center_exists(self) -> None:
        """EDGE_L_CENTER origin mode exists."""
        assert OriginMode.EDGE_L_CENTER.value == "EDGE_L_CENTER"

    def test_all_origin_modes(self) -> None:
        """All expected origin modes exist."""
        expected = {"EDGE_L_CENTER", "EDGE_R_CENTER", "CENTER", "BOTTOM_LEFT"}
        actual = {mode.value for mode in OriginMode}
        assert actual == expected


# =============================================================================
# PadShape enum tests
# =============================================================================


class TestPadShape:
    """Tests for PadShape enum."""

    def test_common_pad_shapes(self) -> None:
        """Common pad shapes are defined."""
        assert PadShape.CIRCLE.value == "circle"
        assert PadShape.RECT.value == "rect"
        assert PadShape.ROUNDRECT.value == "roundrect"
        assert PadShape.OVAL.value == "oval"


# =============================================================================
# CoordinateFrame tests
# =============================================================================


class TestCoordinateFrame:
    """Tests for CoordinateFrame transformations."""

    def test_create_frame_edge_l_center(self) -> None:
        """Create frame with EDGE_L_CENTER origin."""
        frame = CoordinateFrame(
            origin_mode=OriginMode.EDGE_L_CENTER,
            board_width_nm=80_000_000,
            board_height_nm=20_000_000,
        )
        assert frame.origin_mode == OriginMode.EDGE_L_CENTER
        assert frame.board_width_nm == 80_000_000
        assert frame.board_height_nm == 20_000_000

    def test_to_kicad_edge_l_center_origin(self) -> None:
        """Transform origin point from EDGE_L_CENTER to KiCad."""
        frame = CoordinateFrame(
            origin_mode=OriginMode.EDGE_L_CENTER,
            board_width_nm=80_000_000,
            board_height_nm=20_000_000,
        )
        # Origin (0,0) in EDGE_L_CENTER is at left edge, center vertically
        kicad_pos = frame.to_kicad(PositionNM(0, 0))
        # In KiCad: x=0, y should be half height (y inverted)
        assert kicad_pos.x == 0
        assert kicad_pos.y == 10_000_000

    def test_to_kicad_positive_y(self) -> None:
        """Transform positive y (upward) to KiCad coordinates."""
        frame = CoordinateFrame(
            origin_mode=OriginMode.EDGE_L_CENTER,
            board_width_nm=80_000_000,
            board_height_nm=20_000_000,
        )
        # Positive y in our frame is upward, which is negative y in KiCad
        kicad_pos = frame.to_kicad(PositionNM(0, 5_000_000))
        assert kicad_pos.y == 5_000_000  # Half height - 5mm from center

    def test_roundtrip_transformation(self) -> None:
        """Roundtrip transformation should return original position."""
        frame = CoordinateFrame(
            origin_mode=OriginMode.EDGE_L_CENTER,
            board_width_nm=80_000_000,
            board_height_nm=20_000_000,
        )
        original = PositionNM(10_000_000, 3_000_000)
        kicad = frame.to_kicad(original)
        back = frame.from_kicad(kicad)
        assert back == original

    def test_to_kicad_center_origin(self) -> None:
        """Transform from CENTER origin to KiCad."""
        frame = CoordinateFrame(
            origin_mode=OriginMode.CENTER,
            board_width_nm=80_000_000,
            board_height_nm=20_000_000,
        )
        # Origin (0,0) in CENTER is at board center
        kicad_pos = frame.to_kicad(PositionNM(0, 0))
        assert kicad_pos.x == 40_000_000
        assert kicad_pos.y == 10_000_000

    def test_to_kicad_bottom_left_origin(self) -> None:
        """Transform from BOTTOM_LEFT origin to KiCad."""
        frame = CoordinateFrame(
            origin_mode=OriginMode.BOTTOM_LEFT,
            board_width_nm=80_000_000,
            board_height_nm=20_000_000,
        )
        # Origin (0,0) in BOTTOM_LEFT
        kicad_pos = frame.to_kicad(PositionNM(0, 0))
        assert kicad_pos.x == 0
        assert kicad_pos.y == 20_000_000  # y is flipped


# =============================================================================
# BoardOutline tests
# =============================================================================


class TestBoardOutline:
    """Tests for BoardOutline dataclass."""

    def test_rectangle_default_origin(self) -> None:
        """Create rectangular outline with default origin."""
        outline = BoardOutline.rectangle(
            width_nm=80_000_000,
            height_nm=20_000_000,
        )
        assert len(outline.vertices) == 4
        assert outline.vertices[0] == PositionNM(0, 0)
        assert outline.vertices[1] == PositionNM(80_000_000, 0)
        assert outline.vertices[2] == PositionNM(80_000_000, 20_000_000)
        assert outline.vertices[3] == PositionNM(0, 20_000_000)

    def test_rectangle_with_corner_radius(self) -> None:
        """Create rectangular outline with corner radius."""
        outline = BoardOutline.rectangle(
            width_nm=80_000_000,
            height_nm=20_000_000,
            corner_radius_nm=2_000_000,
        )
        assert outline.corner_radius_nm == 2_000_000

    def test_rectangle_custom_origin(self) -> None:
        """Create rectangular outline with custom origin."""
        origin = PositionNM(10_000_000, 5_000_000)
        outline = BoardOutline.rectangle(
            width_nm=80_000_000,
            height_nm=20_000_000,
            origin=origin,
        )
        assert outline.vertices[0] == origin
        assert outline.vertices[1] == PositionNM(90_000_000, 5_000_000)


# =============================================================================
# Helper function tests
# =============================================================================


class TestHalfWidthNm:
    """Tests for half_width_nm helper function."""

    def test_even_width(self) -> None:
        """Even width divides exactly."""
        assert half_width_nm(1000) == 500

    def test_odd_width_rounds_up(self) -> None:
        """Odd width rounds up to preserve clearance."""
        assert half_width_nm(1001) == 501

    def test_minimum_width(self) -> None:
        """Width of 1 returns 1."""
        assert half_width_nm(1) == 1

    def test_zero_width_raises(self) -> None:
        """Zero width raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            half_width_nm(0)

    def test_negative_width_raises(self) -> None:
        """Negative width raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            half_width_nm(-100)


class TestCreateCoordinateFrame:
    """Tests for create_coordinate_frame factory function."""

    def test_create_from_enum(self) -> None:
        """Create frame using OriginMode enum."""
        frame = create_coordinate_frame(
            OriginMode.EDGE_L_CENTER,
            board_width_nm=80_000_000,
            board_height_nm=20_000_000,
        )
        assert frame.origin_mode == OriginMode.EDGE_L_CENTER

    def test_create_from_string(self) -> None:
        """Create frame using string origin mode."""
        frame = create_coordinate_frame(
            "EDGE_L_CENTER",
            board_width_nm=80_000_000,
            board_height_nm=20_000_000,
        )
        assert frame.origin_mode == OriginMode.EDGE_L_CENTER


# =============================================================================
# Net and NetClass tests
# =============================================================================


class TestNet:
    """Tests for Net dataclass."""

    def test_create_net(self) -> None:
        """Create a net with id and name."""
        net = Net(id=1, name="SIG")
        assert net.id == 1
        assert net.name == "SIG"

    def test_unconnected_net(self) -> None:
        """Net with id 0 is unconnected by convention."""
        net = Net(id=0, name="")
        assert net.id == 0


class TestNetClass:
    """Tests for NetClass dataclass."""

    def test_create_netclass(self) -> None:
        """Create a net class with design rules."""
        nc = NetClass(
            name="Signal",
            description="Signal nets",
            clearance_nm=200_000,
            track_width_nm=300_000,
            via_diameter_nm=600_000,
            via_drill_nm=300_000,
        )
        assert nc.name == "Signal"
        assert nc.clearance_nm == 200_000

    def test_netclass_with_nets(self) -> None:
        """Net class with associated net names."""
        nc = NetClass(
            name="Signal",
            description="Signal nets",
            clearance_nm=200_000,
            track_width_nm=300_000,
            via_diameter_nm=600_000,
            via_drill_nm=300_000,
            net_names=("SIG", "SIG_IN", "SIG_OUT"),
        )
        assert "SIG" in nc.net_names


# =============================================================================
# Via tests
# =============================================================================


class TestVia:
    """Tests for Via dataclass."""

    def test_create_via(self) -> None:
        """Create a through via."""
        via = Via(
            position=PositionNM(10_000_000, 0),
            diameter_nm=600_000,
            drill_nm=300_000,
        )
        assert via.diameter_nm == 600_000
        assert via.drill_nm == 300_000
        assert via.via_type == "through"

    def test_via_default_layers(self) -> None:
        """Via has default F.Cu to B.Cu layers."""
        via = Via(
            position=PositionNM(0, 0),
            diameter_nm=600_000,
            drill_nm=300_000,
        )
        assert via.layers == ("F.Cu", "B.Cu")


# =============================================================================
# TrackSegment tests
# =============================================================================


class TestTrackSegment:
    """Tests for TrackSegment dataclass."""

    def test_create_track(self) -> None:
        """Create a track segment."""
        track = TrackSegment(
            start=PositionNM(0, 0),
            end=PositionNM(10_000_000, 0),
            width_nm=300_000,
            layer="F.Cu",
        )
        assert track.width_nm == 300_000
        assert track.layer == "F.Cu"

    def test_track_with_net(self) -> None:
        """Track segment with net assignment."""
        track = TrackSegment(
            start=PositionNM(0, 0),
            end=PositionNM(10_000_000, 0),
            width_nm=300_000,
            layer="F.Cu",
            net_id=1,
        )
        assert track.net_id == 1


# =============================================================================
# Text tests
# =============================================================================


class TestText:
    """Tests for Text dataclass."""

    def test_create_text(self) -> None:
        """Create text annotation."""
        text = Text(
            content="TEST",
            position=PositionNM(10_000_000, 5_000_000),
        )
        assert text.content == "TEST"

    def test_text_default_layer(self) -> None:
        """Text has default silkscreen layer."""
        text = Text(
            content="TEST",
            position=PositionNM(0, 0),
        )
        assert text.layer == TextLayer.F_SILKSCREEN
