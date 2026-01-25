# SPDX-License-Identifier: MIT
"""Unit tests for coupongen geometry cutouts module.

Tests the antipad generation, rounded outline generation, and cutout shape
primitives used for plane clearances and board outlines.
"""

from __future__ import annotations

import importlib.util
import math
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


# Load primitives first - it has no dependencies
_primitives = _load_module(
    "formula_foundry.coupongen.geom.primitives",
    _SRC_DIR / "formula_foundry" / "coupongen" / "geom" / "primitives.py",
)

# Load cutouts - it depends on primitives which is now in sys.modules
_cutouts = _load_module(
    "formula_foundry.coupongen.geom.cutouts",
    _SRC_DIR / "formula_foundry" / "coupongen" / "geom" / "cutouts.py",
)

# Import from loaded modules
CircleAntipadSpec = _cutouts.CircleAntipadSpec
CutoutShape = _cutouts.CutoutShape
OutlineArc = _cutouts.OutlineArc
OutlineFeasibilityError = _cutouts.OutlineFeasibilityError
OutlineLine = _cutouts.OutlineLine
RectangleAntipadSpec = _cutouts.RectangleAntipadSpec
RoundedOutline = _cutouts.RoundedOutline
RoundRectAntipadSpec = _cutouts.RoundRectAntipadSpec
SlotAntipadSpec = _cutouts.SlotAntipadSpec
generate_antipad = _cutouts.generate_antipad
generate_circle_antipad = _cutouts.generate_circle_antipad
generate_multivia_antipad = _cutouts.generate_multivia_antipad
generate_plane_cutout_for_via = _cutouts.generate_plane_cutout_for_via
generate_rectangle_antipad = _cutouts.generate_rectangle_antipad
generate_rounded_outline = _cutouts.generate_rounded_outline
generate_roundrect_antipad = _cutouts.generate_roundrect_antipad
generate_slot_antipad = _cutouts.generate_slot_antipad
validate_rounded_outline_feasibility = _cutouts.validate_rounded_outline_feasibility

PolygonType = _primitives.PolygonType
PositionNM = _primitives.PositionNM


class TestCircleAntipadSpec:
    """Tests for CircleAntipadSpec dataclass."""

    def test_create_basic_spec(self) -> None:
        """Create a basic circle antipad spec."""
        spec = CircleAntipadSpec(
            shape=CutoutShape.CIRCLE,
            r_nm=500_000,
            layer="F.Cu",
        )
        assert spec.r_nm == 500_000
        assert spec.layer == "F.Cu"
        assert spec.segments == 32  # default

    def test_custom_segments(self) -> None:
        """Create spec with custom segment count."""
        spec = CircleAntipadSpec(
            shape=CutoutShape.CIRCLE,
            r_nm=100_000,
            layer="B.Cu",
            segments=64,
        )
        assert spec.segments == 64


class TestGenerateCircleAntipad:
    """Tests for generate_circle_antipad function."""

    def test_basic_circle_generation(self) -> None:
        """Generate a basic circle antipad."""
        center = PositionNM(1_000_000, 2_000_000)
        spec = CircleAntipadSpec(
            shape=CutoutShape.CIRCLE,
            r_nm=300_000,
            layer="F.Cu",
            segments=16,
        )
        polygon = generate_circle_antipad(center, spec)

        assert polygon.polygon_type == PolygonType.CUTOUT
        assert polygon.layer == "F.Cu"
        assert len(polygon.vertices) == 16

    def test_vertices_at_correct_radius(self) -> None:
        """Vertices should be at approximately the specified radius."""
        center = PositionNM(0, 0)
        radius = 500_000
        spec = CircleAntipadSpec(
            shape=CutoutShape.CIRCLE,
            r_nm=radius,
            layer="F.Cu",
            segments=8,
        )
        polygon = generate_circle_antipad(center, spec)

        for vertex in polygon.vertices:
            dist = math.sqrt(vertex.x**2 + vertex.y**2)
            # Allow 1nm tolerance due to integer rounding
            assert abs(dist - radius) < 2

    def test_negative_radius_raises(self) -> None:
        """Negative radius should raise ValueError."""
        center = PositionNM(0, 0)
        spec = CircleAntipadSpec(
            shape=CutoutShape.CIRCLE,
            r_nm=-100_000,
            layer="F.Cu",
        )
        with pytest.raises(ValueError, match="positive"):
            generate_circle_antipad(center, spec)

    def test_zero_radius_raises(self) -> None:
        """Zero radius should raise ValueError."""
        center = PositionNM(0, 0)
        spec = CircleAntipadSpec(
            shape=CutoutShape.CIRCLE,
            r_nm=0,
            layer="F.Cu",
        )
        with pytest.raises(ValueError, match="positive"):
            generate_circle_antipad(center, spec)

    def test_insufficient_segments_raises(self) -> None:
        """Less than 3 segments should raise ValueError."""
        center = PositionNM(0, 0)
        spec = CircleAntipadSpec(
            shape=CutoutShape.CIRCLE,
            r_nm=100_000,
            layer="F.Cu",
            segments=2,
        )
        with pytest.raises(ValueError, match="at least 3"):
            generate_circle_antipad(center, spec)


class TestGenerateRoundRectAntipad:
    """Tests for generate_roundrect_antipad function."""

    def test_basic_roundrect_generation(self) -> None:
        """Generate a basic rounded rectangle antipad."""
        center = PositionNM(1_000_000, 1_000_000)
        spec = RoundRectAntipadSpec(
            shape=CutoutShape.ROUNDRECT,
            rx_nm=500_000,
            ry_nm=300_000,
            corner_nm=50_000,
            layer="In1.Cu",
        )
        polygon = generate_roundrect_antipad(center, spec)

        assert polygon.polygon_type == PolygonType.CUTOUT
        assert polygon.layer == "In1.Cu"
        assert len(polygon.vertices) > 4  # Should have corner arcs

    def test_zero_corner_radius(self) -> None:
        """Zero corner radius should produce sharp corners."""
        center = PositionNM(0, 0)
        spec = RoundRectAntipadSpec(
            shape=CutoutShape.ROUNDRECT,
            rx_nm=100_000,
            ry_nm=50_000,
            corner_nm=0,
            layer="F.Cu",
            corner_segments=4,
        )
        polygon = generate_roundrect_antipad(center, spec)

        # With zero corner radius, we get 4 corners
        assert len(polygon.vertices) == 4

    def test_negative_dimensions_raise(self) -> None:
        """Negative rx_nm or ry_nm should raise ValueError."""
        center = PositionNM(0, 0)
        spec = RoundRectAntipadSpec(
            shape=CutoutShape.ROUNDRECT,
            rx_nm=-100_000,
            ry_nm=50_000,
            corner_nm=10_000,
            layer="F.Cu",
        )
        with pytest.raises(ValueError, match="positive"):
            generate_roundrect_antipad(center, spec)


class TestGenerateSlotAntipad:
    """Tests for generate_slot_antipad function."""

    def test_basic_slot_generation(self) -> None:
        """Generate a basic slot antipad."""
        center = PositionNM(500_000, 500_000)
        spec = SlotAntipadSpec(
            shape=CutoutShape.SLOT,
            length_nm=1_000_000,
            width_nm=300_000,
            rotation_mdeg=0,
            layer="F.Cu",
        )
        polygon = generate_slot_antipad(center, spec)

        assert polygon.polygon_type == PolygonType.CUTOUT
        assert polygon.layer == "F.Cu"

    def test_slot_with_rotation(self) -> None:
        """Slot with 45 degree rotation."""
        center = PositionNM(0, 0)
        spec = SlotAntipadSpec(
            shape=CutoutShape.SLOT,
            length_nm=1_000_000,
            width_nm=200_000,
            rotation_mdeg=45_000,  # 45 degrees
            layer="F.Cu",
        )
        polygon = generate_slot_antipad(center, spec)

        # The polygon should have vertices
        assert len(polygon.vertices) > 2

    def test_short_slot_becomes_circle(self) -> None:
        """Slot shorter than width should become a circle."""
        center = PositionNM(0, 0)
        spec = SlotAntipadSpec(
            shape=CutoutShape.SLOT,
            length_nm=200_000,
            width_nm=400_000,  # width > length
            rotation_mdeg=0,
            layer="F.Cu",
            end_segments=8,
        )
        polygon = generate_slot_antipad(center, spec)

        # Result should be circular (end_segments * 2)
        assert len(polygon.vertices) == 16

    def test_negative_length_raises(self) -> None:
        """Negative length should raise ValueError."""
        center = PositionNM(0, 0)
        spec = SlotAntipadSpec(
            shape=CutoutShape.SLOT,
            length_nm=-100_000,
            width_nm=50_000,
            rotation_mdeg=0,
            layer="F.Cu",
        )
        with pytest.raises(ValueError, match="positive"):
            generate_slot_antipad(center, spec)


class TestGenerateRectangleAntipad:
    """Tests for generate_rectangle_antipad function."""

    def test_basic_rectangle_generation(self) -> None:
        """Generate a basic rectangle antipad."""
        center = PositionNM(1_000_000, 1_000_000)
        spec = RectangleAntipadSpec(
            shape=CutoutShape.RECTANGLE,
            width_nm=400_000,
            height_nm=200_000,
            layer="F.Cu",
        )
        polygon = generate_rectangle_antipad(center, spec)

        assert polygon.polygon_type == PolygonType.CUTOUT
        assert polygon.layer == "F.Cu"
        assert len(polygon.vertices) == 4

    def test_rectangle_vertices_correct(self) -> None:
        """Rectangle vertices should be at correct positions."""
        center = PositionNM(0, 0)
        width = 400_000
        height = 200_000
        spec = RectangleAntipadSpec(
            shape=CutoutShape.RECTANGLE,
            width_nm=width,
            height_nm=height,
            layer="F.Cu",
        )
        polygon = generate_rectangle_antipad(center, spec)

        # Check that vertices span the expected dimensions
        xs = [v.x for v in polygon.vertices]
        ys = [v.y for v in polygon.vertices]
        assert max(xs) - min(xs) == width
        assert max(ys) - min(ys) == height

    def test_negative_width_raises(self) -> None:
        """Negative width should raise ValueError."""
        center = PositionNM(0, 0)
        spec = RectangleAntipadSpec(
            shape=CutoutShape.RECTANGLE,
            width_nm=-100_000,
            height_nm=50_000,
            layer="F.Cu",
        )
        with pytest.raises(ValueError, match="positive"):
            generate_rectangle_antipad(center, spec)


class TestGenerateAntipad:
    """Tests for the generate_antipad dispatcher function."""

    def test_dispatches_to_circle(self) -> None:
        """Dispatcher routes circle spec correctly."""
        center = PositionNM(0, 0)
        spec = CircleAntipadSpec(
            shape=CutoutShape.CIRCLE,
            r_nm=100_000,
            layer="F.Cu",
        )
        polygon = generate_antipad(center, spec)
        assert polygon.polygon_type == PolygonType.CUTOUT

    def test_dispatches_to_rectangle(self) -> None:
        """Dispatcher routes rectangle spec correctly."""
        center = PositionNM(0, 0)
        spec = RectangleAntipadSpec(
            shape=CutoutShape.RECTANGLE,
            width_nm=100_000,
            height_nm=50_000,
            layer="F.Cu",
        )
        polygon = generate_antipad(center, spec)
        assert len(polygon.vertices) == 4


class TestGeneratePlaneCutoutForVia:
    """Tests for generate_plane_cutout_for_via function."""

    def test_basic_via_cutout(self) -> None:
        """Generate a basic cutout around a via."""
        center = PositionNM(5_000_000, 5_000_000)
        polygon = generate_plane_cutout_for_via(
            via_center=center,
            via_diameter_nm=500_000,
            clearance_nm=200_000,
            layer="In1.Cu",
        )

        assert polygon.polygon_type == PolygonType.CUTOUT
        assert polygon.layer == "In1.Cu"

    def test_cutout_radius_includes_clearance(self) -> None:
        """Cutout should have radius = via_radius + clearance."""
        center = PositionNM(0, 0)
        via_diameter = 400_000
        clearance = 150_000
        expected_radius = via_diameter // 2 + clearance

        polygon = generate_plane_cutout_for_via(
            via_center=center,
            via_diameter_nm=via_diameter,
            clearance_nm=clearance,
            layer="F.Cu",
            segments=8,
        )

        # Check that vertices are at approximately expected radius
        for vertex in polygon.vertices:
            dist = math.sqrt(vertex.x**2 + vertex.y**2)
            assert abs(dist - expected_radius) < 2

    def test_rectangle_cutout_shape(self) -> None:
        """Generate rectangular cutout for via."""
        center = PositionNM(0, 0)
        polygon = generate_plane_cutout_for_via(
            via_center=center,
            via_diameter_nm=400_000,
            clearance_nm=100_000,
            layer="F.Cu",
            shape=CutoutShape.RECTANGLE,
        )

        assert len(polygon.vertices) == 4

    def test_negative_clearance_raises(self) -> None:
        """Negative clearance should raise ValueError."""
        center = PositionNM(0, 0)
        with pytest.raises(ValueError, match="non-negative"):
            generate_plane_cutout_for_via(
                via_center=center,
                via_diameter_nm=400_000,
                clearance_nm=-100_000,
                layer="F.Cu",
            )


class TestValidateRoundedOutlineFeasibility:
    """Tests for validate_rounded_outline_feasibility function."""

    def test_valid_parameters(self) -> None:
        """Valid parameters should not raise."""
        validate_rounded_outline_feasibility(
            width_nm=10_000_000,
            height_nm=5_000_000,
            corner_radius_nm=500_000,
        )

    def test_zero_width_raises(self) -> None:
        """Zero width should raise OutlineFeasibilityError."""
        with pytest.raises(OutlineFeasibilityError, match="Width must be positive"):
            validate_rounded_outline_feasibility(
                width_nm=0,
                height_nm=5_000_000,
                corner_radius_nm=100_000,
            )

    def test_negative_height_raises(self) -> None:
        """Negative height should raise OutlineFeasibilityError."""
        with pytest.raises(OutlineFeasibilityError, match="Height must be positive"):
            validate_rounded_outline_feasibility(
                width_nm=10_000_000,
                height_nm=-5_000_000,
                corner_radius_nm=100_000,
            )

    def test_negative_corner_radius_raises(self) -> None:
        """Negative corner radius should raise OutlineFeasibilityError."""
        with pytest.raises(OutlineFeasibilityError, match="non-negative"):
            validate_rounded_outline_feasibility(
                width_nm=10_000_000,
                height_nm=5_000_000,
                corner_radius_nm=-100_000,
            )

    def test_corner_radius_too_large_raises(self) -> None:
        """Corner radius > min(w,h)/2 should raise OutlineFeasibilityError."""
        with pytest.raises(OutlineFeasibilityError, match="exceeds maximum"):
            validate_rounded_outline_feasibility(
                width_nm=10_000_000,
                height_nm=4_000_000,
                corner_radius_nm=2_500_000,  # > 4_000_000 / 2
            )


class TestGenerateRoundedOutline:
    """Tests for generate_rounded_outline function."""

    def test_sharp_corner_rectangle(self) -> None:
        """Zero corner radius produces 4 line segments."""
        outline = generate_rounded_outline(
            x_left_nm=0,
            y_bottom_nm=0,
            width_nm=10_000_000,
            height_nm=5_000_000,
            corner_radius_nm=0,
        )

        assert isinstance(outline, RoundedOutline)
        assert len(outline.elements) == 4
        assert all(isinstance(e, OutlineLine) for e in outline.elements)
        assert outline.corner_radius_nm == 0

    def test_rounded_corner_rectangle(self) -> None:
        """Non-zero corner radius produces lines and arcs."""
        outline = generate_rounded_outline(
            x_left_nm=0,
            y_bottom_nm=0,
            width_nm=10_000_000,
            height_nm=5_000_000,
            corner_radius_nm=500_000,
        )

        # Should have 4 lines and 4 arcs = 8 elements
        assert len(outline.elements) == 8
        lines = [e for e in outline.elements if isinstance(e, OutlineLine)]
        arcs = [e for e in outline.elements if isinstance(e, OutlineArc)]
        assert len(lines) == 4
        assert len(arcs) == 4

    def test_outline_dimensions_stored(self) -> None:
        """Outline should store dimensions."""
        outline = generate_rounded_outline(
            x_left_nm=100_000,
            y_bottom_nm=200_000,
            width_nm=8_000_000,
            height_nm=4_000_000,
            corner_radius_nm=300_000,
        )

        assert outline.width_nm == 8_000_000
        assert outline.height_nm == 4_000_000
        assert outline.corner_radius_nm == 300_000

    def test_outline_continuity(self) -> None:
        """Outline elements should form a continuous closed loop."""
        outline = generate_rounded_outline(
            x_left_nm=0,
            y_bottom_nm=0,
            width_nm=10_000_000,
            height_nm=5_000_000,
            corner_radius_nm=500_000,
        )

        # Check that each element ends where the next begins
        for i in range(len(outline.elements)):
            current = outline.elements[i]
            next_elem = outline.elements[(i + 1) % len(outline.elements)]

            if isinstance(current, OutlineLine):
                current_end = current.end
            else:
                current_end = current.end

            if isinstance(next_elem, OutlineLine):
                next_start = next_elem.start
            else:
                next_start = next_elem.start

            assert current_end == next_start, f"Gap between element {i} and {i+1}"


class TestGenerateMultiviaAntipad:
    """Tests for generate_multivia_antipad function."""

    def test_single_via_becomes_circle(self) -> None:
        """Single via should produce circular antipad."""
        centers = (PositionNM(1_000_000, 1_000_000),)
        polygon = generate_multivia_antipad(
            via_centers=centers,
            via_diameter_nm=400_000,
            clearance_nm=100_000,
            layer="F.Cu",
        )

        assert polygon.polygon_type == PolygonType.CUTOUT

    def test_multiple_vias_produce_bounding_shape(self) -> None:
        """Multiple vias should produce bounding shape antipad."""
        centers = (
            PositionNM(0, 0),
            PositionNM(1_000_000, 0),
            PositionNM(500_000, 500_000),
        )
        polygon = generate_multivia_antipad(
            via_centers=centers,
            via_diameter_nm=300_000,
            clearance_nm=100_000,
            layer="In1.Cu",
        )

        assert polygon.polygon_type == PolygonType.CUTOUT
        assert polygon.layer == "In1.Cu"

    def test_empty_via_list_raises(self) -> None:
        """Empty via list should raise ValueError."""
        with pytest.raises(ValueError, match="not be empty"):
            generate_multivia_antipad(
                via_centers=(),
                via_diameter_nm=400_000,
                clearance_nm=100_000,
                layer="F.Cu",
            )


class TestCutoutShapeEnum:
    """Tests for CutoutShape enum values."""

    def test_all_shapes_defined(self) -> None:
        """All expected shapes should be defined."""
        assert CutoutShape.CIRCLE.value == "CIRCLE"
        assert CutoutShape.ROUNDRECT.value == "ROUNDRECT"
        assert CutoutShape.SLOT.value == "SLOT"
        assert CutoutShape.RECTANGLE.value == "RECTANGLE"
        assert CutoutShape.OBROUND.value == "OBROUND"

    def test_shape_count(self) -> None:
        """Should have 5 shape types."""
        assert len(CutoutShape) == 5
