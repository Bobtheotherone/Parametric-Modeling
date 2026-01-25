# SPDX-License-Identifier: MIT
"""Unit tests for coupongen geometry via patterns module.

Tests the via pattern generation functions used for signal vias, return via
rings, return via grids, and via transition structures.
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

# Load via_patterns - it depends on primitives which is now in sys.modules
_via_patterns = _load_module(
    "formula_foundry.coupongen.geom.via_patterns",
    _SRC_DIR / "formula_foundry" / "coupongen" / "geom" / "via_patterns.py",
)

# Import from loaded modules - primitives
PositionNM = _primitives.PositionNM
Via = _primitives.Via

# Import from loaded modules - via_patterns
ReturnViaGridSpec = _via_patterns.ReturnViaGridSpec
ReturnViaPattern = _via_patterns.ReturnViaPattern
ReturnViaRingSpec = _via_patterns.ReturnViaRingSpec
ReturnViaSpec = _via_patterns.ReturnViaSpec
SignalViaSpec = _via_patterns.SignalViaSpec
ViaTransitionResult = _via_patterns.ViaTransitionResult
calculate_minimum_return_via_radius = _via_patterns.calculate_minimum_return_via_radius
calculate_via_ring_circumference_clearance = _via_patterns.calculate_via_ring_circumference_clearance
generate_return_via_grid = _via_patterns.generate_return_via_grid
generate_return_via_quadrant = _via_patterns.generate_return_via_quadrant
generate_return_via_ring = _via_patterns.generate_return_via_ring
generate_signal_via = _via_patterns.generate_signal_via
generate_via_transition = _via_patterns.generate_via_transition
scale_component_nm = _via_patterns.scale_component_nm
segment_length_nm = _via_patterns.segment_length_nm
symmetric_offsets_nm = _via_patterns.symmetric_offsets_nm


class TestSignalViaSpec:
    """Tests for SignalViaSpec dataclass."""

    def test_create_basic_spec(self) -> None:
        """Create a basic signal via spec."""
        spec = SignalViaSpec(
            drill_nm=300_000,
            diameter_nm=600_000,
        )
        assert spec.drill_nm == 300_000
        assert spec.diameter_nm == 600_000
        assert spec.pad_diameter_nm is None
        assert spec.layers == ("F.Cu", "B.Cu")
        assert spec.net_id == 1

    def test_effective_pad_diameter_without_pad(self) -> None:
        """Without pad_diameter_nm, effective diameter equals diameter_nm."""
        spec = SignalViaSpec(
            drill_nm=300_000,
            diameter_nm=600_000,
        )
        assert spec.effective_pad_diameter_nm == 600_000

    def test_effective_pad_diameter_with_pad(self) -> None:
        """With pad_diameter_nm set, effective diameter uses it."""
        spec = SignalViaSpec(
            drill_nm=300_000,
            diameter_nm=600_000,
            pad_diameter_nm=800_000,
        )
        assert spec.effective_pad_diameter_nm == 800_000

    def test_custom_layers_and_net(self) -> None:
        """Create spec with custom layers and net."""
        spec = SignalViaSpec(
            drill_nm=200_000,
            diameter_nm=400_000,
            layers=("F.Cu", "In1.Cu"),
            net_id=42,
        )
        assert spec.layers == ("F.Cu", "In1.Cu")
        assert spec.net_id == 42


class TestReturnViaSpec:
    """Tests for ReturnViaSpec dataclass."""

    def test_create_basic_spec(self) -> None:
        """Create a basic return via spec."""
        spec = ReturnViaSpec(
            drill_nm=250_000,
            diameter_nm=500_000,
        )
        assert spec.drill_nm == 250_000
        assert spec.diameter_nm == 500_000
        assert spec.layers == ("F.Cu", "B.Cu")
        assert spec.net_id == 0  # GND default


class TestGenerateSignalVia:
    """Tests for generate_signal_via function."""

    def test_basic_signal_via(self) -> None:
        """Generate a basic signal via."""
        center = PositionNM(5_000_000, 3_000_000)
        spec = SignalViaSpec(
            drill_nm=300_000,
            diameter_nm=600_000,
            net_id=1,
        )
        via = generate_signal_via(center, spec)

        assert via.position == center
        assert via.drill_nm == 300_000
        assert via.diameter_nm == 600_000
        assert via.net_id == 1
        assert via.layers == ("F.Cu", "B.Cu")


class TestGenerateReturnViaRing:
    """Tests for generate_return_via_ring function."""

    def test_basic_ring(self) -> None:
        """Generate a basic return via ring."""
        center = PositionNM(2_000_000, 2_000_000)
        via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        ring_spec = ReturnViaRingSpec(
            pattern=ReturnViaPattern.RING,
            count=4,
            radius_nm=1_000_000,
            via=via_spec,
        )
        vias = generate_return_via_ring(center, ring_spec)

        assert len(vias) == 4
        assert all(isinstance(v, Via) for v in vias)

    def test_vias_at_correct_radius(self) -> None:
        """Vias should be at the specified radius."""
        center = PositionNM(0, 0)
        radius = 800_000
        via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        ring_spec = ReturnViaRingSpec(
            pattern=ReturnViaPattern.RING,
            count=8,
            radius_nm=radius,
            via=via_spec,
        )
        vias = generate_return_via_ring(center, ring_spec)

        for via in vias:
            dist = math.sqrt(via.position.x**2 + via.position.y**2)
            assert abs(dist - radius) < 2  # Allow 1nm rounding

    def test_ring_with_start_angle(self) -> None:
        """Ring with non-zero start angle should rotate pattern."""
        center = PositionNM(0, 0)
        via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        ring_spec = ReturnViaRingSpec(
            pattern=ReturnViaPattern.RING,
            count=4,
            radius_nm=1_000_000,
            via=via_spec,
            start_angle_mdeg=45_000,  # 45 degrees
        )
        vias = generate_return_via_ring(center, ring_spec)

        # First via should be at 45 degrees (positive x and y)
        first_via = vias[0]
        assert first_via.position.x > 0
        assert first_via.position.y > 0

    def test_zero_count_returns_empty(self) -> None:
        """Zero count should return empty tuple."""
        center = PositionNM(0, 0)
        via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        ring_spec = ReturnViaRingSpec(
            pattern=ReturnViaPattern.RING,
            count=0,
            radius_nm=1_000_000,
            via=via_spec,
        )
        vias = generate_return_via_ring(center, ring_spec)

        assert vias == ()

    def test_negative_radius_raises(self) -> None:
        """Negative radius should raise ValueError."""
        center = PositionNM(0, 0)
        via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        ring_spec = ReturnViaRingSpec(
            pattern=ReturnViaPattern.RING,
            count=4,
            radius_nm=-1_000_000,
            via=via_spec,
        )
        with pytest.raises(ValueError, match="positive"):
            generate_return_via_ring(center, ring_spec)

    def test_vias_have_correct_properties(self) -> None:
        """Vias should inherit properties from spec."""
        center = PositionNM(0, 0)
        via_spec = ReturnViaSpec(
            drill_nm=250_000,
            diameter_nm=500_000,
            layers=("F.Cu", "In1.Cu"),
            net_id=5,
        )
        ring_spec = ReturnViaRingSpec(
            pattern=ReturnViaPattern.RING,
            count=2,
            radius_nm=500_000,
            via=via_spec,
        )
        vias = generate_return_via_ring(center, ring_spec)

        for via in vias:
            assert via.drill_nm == 250_000
            assert via.diameter_nm == 500_000
            assert via.layers == ("F.Cu", "In1.Cu")
            assert via.net_id == 5


class TestGenerateReturnViaGrid:
    """Tests for generate_return_via_grid function."""

    def test_basic_grid(self) -> None:
        """Generate a basic 3x3 grid with center excluded."""
        center = PositionNM(5_000_000, 5_000_000)
        via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        grid_spec = ReturnViaGridSpec(
            pattern=ReturnViaPattern.GRID,
            rows=3,
            cols=3,
            row_pitch_nm=1_000_000,
            col_pitch_nm=1_000_000,
            via=via_spec,
            exclude_center=True,
        )
        vias = generate_return_via_grid(center, grid_spec)

        # 3x3 = 9 positions, minus center = 8
        assert len(vias) == 8

    def test_grid_without_center_exclusion(self) -> None:
        """Grid without center exclusion should have all positions."""
        center = PositionNM(0, 0)
        via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        grid_spec = ReturnViaGridSpec(
            pattern=ReturnViaPattern.GRID,
            rows=3,
            cols=3,
            row_pitch_nm=500_000,
            col_pitch_nm=500_000,
            via=via_spec,
            exclude_center=False,
        )
        vias = generate_return_via_grid(center, grid_spec)

        assert len(vias) == 9

    def test_even_grid_excludes_center(self) -> None:
        """Even grid should exclude the closest-to-center position."""
        center = PositionNM(0, 0)
        via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        grid_spec = ReturnViaGridSpec(
            pattern=ReturnViaPattern.GRID,
            rows=2,
            cols=2,
            row_pitch_nm=1_000_000,
            col_pitch_nm=1_000_000,
            via=via_spec,
            exclude_center=True,
        )
        vias = generate_return_via_grid(center, grid_spec)

        # 2x2 = 4 positions, center exclusion removes 1
        assert len(vias) == 3

    def test_zero_rows_returns_empty(self) -> None:
        """Zero rows should return empty tuple."""
        center = PositionNM(0, 0)
        via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        grid_spec = ReturnViaGridSpec(
            pattern=ReturnViaPattern.GRID,
            rows=0,
            cols=3,
            row_pitch_nm=500_000,
            col_pitch_nm=500_000,
            via=via_spec,
        )
        vias = generate_return_via_grid(center, grid_spec)

        assert vias == ()

    def test_negative_pitch_raises(self) -> None:
        """Negative pitch should raise ValueError."""
        center = PositionNM(0, 0)
        via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        grid_spec = ReturnViaGridSpec(
            pattern=ReturnViaPattern.GRID,
            rows=3,
            cols=3,
            row_pitch_nm=-500_000,
            col_pitch_nm=500_000,
            via=via_spec,
        )
        with pytest.raises(ValueError, match="positive"):
            generate_return_via_grid(center, grid_spec)


class TestGenerateReturnViaQuadrant:
    """Tests for generate_return_via_quadrant function."""

    def test_basic_quadrant(self) -> None:
        """Generate basic quadrant vias."""
        center = PositionNM(1_000_000, 1_000_000)
        via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        vias = generate_return_via_quadrant(center, 500_000, via_spec)

        assert len(vias) == 4

    def test_vias_in_all_quadrants(self) -> None:
        """Vias should be in all four quadrants relative to center."""
        center = PositionNM(0, 0)
        via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        vias = generate_return_via_quadrant(center, 500_000, via_spec)

        positions = [v.position for v in vias]
        # Q1: +x, +y
        assert any(p.x > 0 and p.y > 0 for p in positions)
        # Q2: -x, +y
        assert any(p.x < 0 and p.y > 0 for p in positions)
        # Q3: -x, -y
        assert any(p.x < 0 and p.y < 0 for p in positions)
        # Q4: +x, -y
        assert any(p.x > 0 and p.y < 0 for p in positions)

    def test_vias_at_correct_distance(self) -> None:
        """Vias should be at the specified radius."""
        center = PositionNM(0, 0)
        radius = 700_000
        via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        vias = generate_return_via_quadrant(center, radius, via_spec)

        for via in vias:
            dist = math.sqrt(via.position.x**2 + via.position.y**2)
            # Allow small rounding tolerance
            assert abs(dist - radius) < 2

    def test_negative_radius_raises(self) -> None:
        """Negative radius should raise ValueError."""
        center = PositionNM(0, 0)
        via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        with pytest.raises(ValueError, match="positive"):
            generate_return_via_quadrant(center, -500_000, via_spec)


class TestGenerateViaTransition:
    """Tests for generate_via_transition function."""

    def test_transition_with_return_ring(self) -> None:
        """Generate via transition with return via ring."""
        center = PositionNM(3_000_000, 3_000_000)
        signal_spec = SignalViaSpec(drill_nm=300_000, diameter_nm=600_000)
        return_via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        ring_spec = ReturnViaRingSpec(
            pattern=ReturnViaPattern.RING,
            count=6,
            radius_nm=1_000_000,
            via=return_via_spec,
        )

        result = generate_via_transition(center, signal_spec, ring_spec)

        assert isinstance(result, ViaTransitionResult)
        assert result.signal_via.position == center
        assert len(result.return_vias) == 6

    def test_transition_without_return_vias(self) -> None:
        """Generate via transition without return vias."""
        center = PositionNM(2_000_000, 2_000_000)
        signal_spec = SignalViaSpec(drill_nm=300_000, diameter_nm=600_000)

        result = generate_via_transition(center, signal_spec, None)

        assert result.signal_via.position == center
        assert result.return_vias == ()


class TestCalculateMinimumReturnViaRadius:
    """Tests for calculate_minimum_return_via_radius function."""

    def test_basic_calculation(self) -> None:
        """Calculate minimum radius with basic specs."""
        signal_spec = SignalViaSpec(drill_nm=300_000, diameter_nm=600_000)
        return_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)

        min_radius = calculate_minimum_return_via_radius(
            signal_spec, return_spec, clearance_nm=200_000
        )

        # Expected: signal_radius (300_000) + return_radius (200_000) + clearance (200_000)
        assert min_radius == 700_000

    def test_with_pad_diameter(self) -> None:
        """Calculate minimum radius with pad_diameter_nm set."""
        signal_spec = SignalViaSpec(
            drill_nm=300_000,
            diameter_nm=600_000,
            pad_diameter_nm=800_000,  # Larger pad
        )
        return_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)

        min_radius = calculate_minimum_return_via_radius(
            signal_spec, return_spec, clearance_nm=100_000
        )

        # Expected: pad_radius (400_000) + return_radius (200_000) + clearance (100_000)
        assert min_radius == 700_000


class TestCalculateViaRingCircumferenceClearance:
    """Tests for calculate_via_ring_circumference_clearance function."""

    def test_adequate_clearance(self) -> None:
        """Ring with adequate spacing should have positive clearance."""
        via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        ring_spec = ReturnViaRingSpec(
            pattern=ReturnViaPattern.RING,
            count=4,
            radius_nm=2_000_000,  # Large radius
            via=via_spec,
        )

        clearance = calculate_via_ring_circumference_clearance(ring_spec)

        assert clearance > 0

    def test_tight_clearance(self) -> None:
        """Ring with tight spacing may have small or zero clearance."""
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=600_000)
        ring_spec = ReturnViaRingSpec(
            pattern=ReturnViaPattern.RING,
            count=8,
            radius_nm=500_000,  # Small radius for many vias
            via=via_spec,
        )

        clearance = calculate_via_ring_circumference_clearance(ring_spec)

        # May be zero or negative (clamped to zero)
        assert clearance >= 0

    def test_single_via_ring(self) -> None:
        """Single via ring should return large clearance."""
        via_spec = ReturnViaSpec(drill_nm=200_000, diameter_nm=400_000)
        ring_spec = ReturnViaRingSpec(
            pattern=ReturnViaPattern.RING,
            count=1,
            radius_nm=500_000,
            via=via_spec,
        )

        clearance = calculate_via_ring_circumference_clearance(ring_spec)

        # Single via has clearance = 2 * radius
        assert clearance == 1_000_000


class TestSegmentLengthNm:
    """Tests for segment_length_nm helper function."""

    def test_horizontal_segment(self) -> None:
        """Horizontal segment length is absolute dx."""
        assert segment_length_nm(1_000_000, 0) == 1_000_000
        assert segment_length_nm(-500_000, 0) == 500_000

    def test_vertical_segment(self) -> None:
        """Vertical segment length is absolute dy."""
        assert segment_length_nm(0, 750_000) == 750_000
        assert segment_length_nm(0, -250_000) == 250_000

    def test_zero_length(self) -> None:
        """Zero length segment."""
        assert segment_length_nm(0, 0) == 0

    def test_diagonal_segment(self) -> None:
        """Diagonal segment using Pythagorean theorem."""
        # 3-4-5 triangle scaled
        length = segment_length_nm(3_000_000, 4_000_000)
        assert length == 5_000_000


class TestScaleComponentNm:
    """Tests for scale_component_nm helper function."""

    def test_positive_delta(self) -> None:
        """Positive delta scales correctly."""
        result = scale_component_nm(1_000_000, 500_000, 2_000_000)
        # 1_000_000 * 500_000 / 2_000_000 = 250_000
        assert result == 250_000

    def test_negative_delta(self) -> None:
        """Negative delta scales with correct sign."""
        result = scale_component_nm(-1_000_000, 500_000, 2_000_000)
        assert result == -250_000

    def test_zero_length_returns_zero(self) -> None:
        """Zero length should return zero."""
        result = scale_component_nm(1_000_000, 500_000, 0)
        assert result == 0


class TestSymmetricOffsetsNm:
    """Tests for symmetric_offsets_nm helper function."""

    def test_basic_offsets(self) -> None:
        """Generate symmetric offsets along a segment."""
        offsets = symmetric_offsets_nm(10_000_000, 2_000_000)

        assert len(offsets) >= 1
        assert all(isinstance(o, int) for o in offsets)

    def test_with_end_clearance(self) -> None:
        """Offsets with end clearance should stay within bounds."""
        length = 10_000_000
        clearance = 1_000_000
        offsets = symmetric_offsets_nm(length, 2_000_000, end_clearance_nm=clearance)

        for offset in offsets:
            assert offset >= clearance
            assert offset <= length - clearance

    def test_zero_length_returns_empty(self) -> None:
        """Zero length should return empty tuple."""
        offsets = symmetric_offsets_nm(0, 500_000)
        assert offsets == ()

    def test_negative_pitch_raises(self) -> None:
        """Negative pitch should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            symmetric_offsets_nm(10_000_000, -500_000)

    def test_negative_clearance_raises(self) -> None:
        """Negative end clearance should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            symmetric_offsets_nm(10_000_000, 500_000, end_clearance_nm=-100_000)


class TestReturnViaPatternEnum:
    """Tests for ReturnViaPattern enum."""

    def test_all_patterns_defined(self) -> None:
        """All expected patterns should be defined."""
        assert ReturnViaPattern.RING.value == "RING"
        assert ReturnViaPattern.GRID.value == "GRID"
        assert ReturnViaPattern.QUADRANT.value == "QUADRANT"

    def test_pattern_count(self) -> None:
        """Should have 3 pattern types."""
        assert len(ReturnViaPattern) == 3
