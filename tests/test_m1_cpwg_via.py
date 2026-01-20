"""Tests for CPWG and via pattern geometry generators.

These tests verify the deterministic geometry generation for:
- CPWG transmission line segments with ground via fencing
- Signal and return via patterns (ring, grid, quadrant)
- Antipad and plane cutout shapes
"""

from __future__ import annotations

import math

import pytest

from formula_foundry.coupongen.geom import (
    # Cutouts
    CircleAntipadSpec,
    # CPWG
    CPWGResult,
    CPWGSpec,
    CutoutShape,
    GroundViaFenceSpec,
    # Primitives
    PolygonType,
    PositionNM,
    RectangleAntipadSpec,
    # Via patterns
    ReturnViaGridSpec,
    ReturnViaPattern,
    ReturnViaRingSpec,
    ReturnViaSpec,
    RoundRectAntipadSpec,
    SignalViaSpec,
    SlotAntipadSpec,
    TrackSegment,
    Via,
    calculate_minimum_return_via_radius,
    calculate_via_ring_circumference_clearance,
    generate_antipad,
    generate_circle_antipad,
    generate_cpwg_horizontal,
    generate_cpwg_segment,
    generate_cpwg_with_fence,
    generate_ground_via_fence,
    generate_multivia_antipad,
    generate_plane_cutout_for_via,
    generate_rectangle_antipad,
    generate_return_via_grid,
    generate_return_via_quadrant,
    generate_return_via_ring,
    generate_roundrect_antipad,
    generate_signal_via,
    generate_slot_antipad,
    generate_symmetric_cpwg_pair,
    generate_via_transition,
)


class TestCPWGSegment:
    """Tests for basic CPWG segment generation."""

    def test_generate_cpwg_segment_creates_track(self) -> None:
        """Basic segment generation creates a TrackSegment."""
        start = PositionNM(0, 0)
        end = PositionNM(10_000_000, 0)  # 10mm
        spec = CPWGSpec(w_nm=300_000, gap_nm=180_000, length_nm=10_000_000)

        track = generate_cpwg_segment(start, end, spec)

        assert isinstance(track, TrackSegment)
        assert track.start == start
        assert track.end == end
        assert track.width_nm == 300_000
        assert track.layer == "F.Cu"
        assert track.net_id == 1

    def test_generate_cpwg_horizontal_positive_direction(self) -> None:
        """Horizontal CPWG in +x direction."""
        origin = PositionNM(5_000_000, 0)
        spec = CPWGSpec(w_nm=300_000, gap_nm=180_000, length_nm=25_000_000)

        track = generate_cpwg_horizontal(origin, spec, direction=1)

        assert track.start == origin
        assert track.end == PositionNM(30_000_000, 0)
        assert track.width_nm == 300_000

    def test_generate_cpwg_horizontal_negative_direction(self) -> None:
        """Horizontal CPWG in -x direction."""
        origin = PositionNM(30_000_000, 0)
        spec = CPWGSpec(w_nm=300_000, gap_nm=180_000, length_nm=25_000_000)

        track = generate_cpwg_horizontal(origin, spec, direction=-1)

        assert track.start == origin
        assert track.end == PositionNM(5_000_000, 0)

    def test_generate_cpwg_horizontal_invalid_direction_raises(self) -> None:
        """Invalid direction raises ValueError."""
        spec = CPWGSpec(w_nm=300_000, gap_nm=180_000, length_nm=10_000_000)

        with pytest.raises(ValueError, match="direction must be 1 or -1"):
            generate_cpwg_horizontal(PositionNM(0, 0), spec, direction=0)


class TestGroundViaFence:
    """Tests for ground via fence generation."""

    def test_generate_fence_creates_symmetric_vias(self) -> None:
        """Via fence creates vias on both sides of the trace."""
        start = PositionNM(0, 0)
        end = PositionNM(10_000_000, 0)  # 10mm
        cpwg_spec = CPWGSpec(w_nm=300_000, gap_nm=180_000, length_nm=10_000_000)
        fence_spec = GroundViaFenceSpec(
            pitch_nm=1_500_000,
            offset_from_gap_nm=800_000,
            drill_nm=300_000,
            diameter_nm=600_000,
        )

        pos_vias, neg_vias = generate_ground_via_fence(start, end, cpwg_spec, fence_spec)

        # Should have same number on each side
        assert len(pos_vias) == len(neg_vias)
        assert len(pos_vias) > 0

        # All vias should be Via instances
        for via in pos_vias:
            assert isinstance(via, Via)
            assert via.diameter_nm == 600_000
            assert via.drill_nm == 300_000

    def test_fence_via_positions_are_symmetric(self) -> None:
        """Vias should be symmetric about the centerline."""
        start = PositionNM(0, 0)
        end = PositionNM(10_000_000, 0)
        cpwg_spec = CPWGSpec(w_nm=300_000, gap_nm=180_000, length_nm=10_000_000)
        fence_spec = GroundViaFenceSpec(
            pitch_nm=2_000_000,
            offset_from_gap_nm=500_000,
            drill_nm=300_000,
            diameter_nm=600_000,
        )

        pos_vias, neg_vias = generate_ground_via_fence(start, end, cpwg_spec, fence_spec)

        # Each pair should have same x, opposite y
        for pos_via, neg_via in zip(pos_vias, neg_vias, strict=False):
            assert pos_via.position.x == neg_via.position.x
            assert pos_via.position.y == -neg_via.position.y

    def test_fence_via_count_depends_on_pitch(self) -> None:
        """Number of vias should depend on length and pitch."""
        start = PositionNM(0, 0)
        end = PositionNM(10_000_000, 0)
        cpwg_spec = CPWGSpec(w_nm=300_000, gap_nm=180_000, length_nm=10_000_000)

        # Larger pitch = fewer vias
        large_pitch = GroundViaFenceSpec(pitch_nm=5_000_000, offset_from_gap_nm=500_000, drill_nm=300_000, diameter_nm=600_000)
        small_pitch = GroundViaFenceSpec(pitch_nm=1_000_000, offset_from_gap_nm=500_000, drill_nm=300_000, diameter_nm=600_000)

        large_pos, _ = generate_ground_via_fence(start, end, cpwg_spec, large_pitch)
        small_pos, _ = generate_ground_via_fence(start, end, cpwg_spec, small_pitch)

        assert len(small_pos) > len(large_pos)

    def test_zero_length_segment_returns_empty(self) -> None:
        """Zero-length segment returns no vias."""
        pos = PositionNM(0, 0)
        cpwg_spec = CPWGSpec(w_nm=300_000, gap_nm=180_000, length_nm=0)
        fence_spec = GroundViaFenceSpec(pitch_nm=1_000_000, offset_from_gap_nm=500_000, drill_nm=300_000, diameter_nm=600_000)

        pos_vias, neg_vias = generate_ground_via_fence(pos, pos, cpwg_spec, fence_spec)

        assert pos_vias == ()
        assert neg_vias == ()

    def test_invalid_pitch_raises(self) -> None:
        """Zero or negative pitch raises ValueError."""
        start = PositionNM(0, 0)
        end = PositionNM(10_000_000, 0)
        cpwg_spec = CPWGSpec(w_nm=300_000, gap_nm=180_000, length_nm=10_000_000)
        fence_spec = GroundViaFenceSpec(pitch_nm=0, offset_from_gap_nm=500_000, drill_nm=300_000, diameter_nm=600_000)

        with pytest.raises(ValueError, match="pitch_nm must be positive"):
            generate_ground_via_fence(start, end, cpwg_spec, fence_spec)


class TestCPWGWithFence:
    """Tests for complete CPWG with fence generation."""

    def test_cpwg_with_fence_returns_result(self) -> None:
        """Complete CPWG generation returns CPWGResult."""
        origin = PositionNM(0, 0)
        cpwg_spec = CPWGSpec(w_nm=300_000, gap_nm=180_000, length_nm=10_000_000)
        fence_spec = GroundViaFenceSpec(pitch_nm=1_500_000, offset_from_gap_nm=800_000, drill_nm=300_000, diameter_nm=600_000)

        result = generate_cpwg_with_fence(origin, cpwg_spec, fence_spec)

        assert isinstance(result, CPWGResult)
        assert isinstance(result.signal_track, TrackSegment)
        assert len(result.fence_vias_positive_y) > 0
        assert len(result.fence_vias_negative_y) > 0

    def test_cpwg_without_fence_returns_empty_vias(self) -> None:
        """CPWG without fence spec returns empty via tuples."""
        origin = PositionNM(0, 0)
        cpwg_spec = CPWGSpec(w_nm=300_000, gap_nm=180_000, length_nm=10_000_000)

        result = generate_cpwg_with_fence(origin, cpwg_spec, fence_spec=None)

        assert result.fence_vias_positive_y == ()
        assert result.fence_vias_negative_y == ()


class TestSymmetricCPWGPair:
    """Tests for symmetric CPWG pair generation."""

    def test_symmetric_pair_extends_both_directions(self) -> None:
        """Symmetric pair creates tracks in both directions."""
        center = PositionNM(40_000_000, 0)  # 40mm
        cpwg_spec = CPWGSpec(w_nm=300_000, gap_nm=180_000, length_nm=0)

        left_result, right_result = generate_symmetric_cpwg_pair(
            center, cpwg_spec, left_length_nm=25_000_000, right_length_nm=25_000_000
        )

        # Left track should end at center
        assert left_result.signal_track.start == center
        assert left_result.signal_track.end.x < center.x

        # Right track should start at center
        assert right_result.signal_track.start == center
        assert right_result.signal_track.end.x > center.x


class TestSignalVia:
    """Tests for signal via generation."""

    def test_generate_signal_via_basic(self) -> None:
        """Basic signal via generation."""
        center = PositionNM(40_000_000, 0)
        spec = SignalViaSpec(drill_nm=300_000, diameter_nm=650_000)

        via = generate_signal_via(center, spec)

        assert isinstance(via, Via)
        assert via.position == center
        assert via.drill_nm == 300_000
        assert via.diameter_nm == 650_000
        assert via.net_id == 1

    def test_signal_via_with_pad_diameter(self) -> None:
        """Signal via with separate pad diameter."""
        spec = SignalViaSpec(drill_nm=300_000, diameter_nm=650_000, pad_diameter_nm=900_000, net_id=5)

        assert spec.effective_pad_diameter_nm == 900_000

    def test_signal_via_pad_diameter_fallback(self) -> None:
        """When pad_diameter_nm is None, use diameter_nm."""
        spec = SignalViaSpec(drill_nm=300_000, diameter_nm=650_000)

        assert spec.effective_pad_diameter_nm == 650_000


class TestReturnViaRing:
    """Tests for return via ring patterns."""

    def test_generate_ring_creates_correct_count(self) -> None:
        """Ring generates specified number of vias."""
        center = PositionNM(0, 0)
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=650_000)
        ring_spec = ReturnViaRingSpec(pattern=ReturnViaPattern.RING, count=4, radius_nm=1_700_000, via=via_spec)

        vias = generate_return_via_ring(center, ring_spec)

        assert len(vias) == 4

    def test_ring_vias_are_equidistant_from_center(self) -> None:
        """All vias in ring are same distance from center."""
        center = PositionNM(10_000_000, 10_000_000)
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=650_000)
        ring_spec = ReturnViaRingSpec(pattern=ReturnViaPattern.RING, count=6, radius_nm=1_700_000, via=via_spec)

        vias = generate_return_via_ring(center, ring_spec)

        for via in vias:
            dx = via.position.x - center.x
            dy = via.position.y - center.y
            distance = int(math.sqrt(dx * dx + dy * dy))
            # Allow small rounding error
            assert abs(distance - 1_700_000) < 100

    def test_ring_with_zero_count_returns_empty(self) -> None:
        """Zero count returns empty tuple."""
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=650_000)
        ring_spec = ReturnViaRingSpec(pattern=ReturnViaPattern.RING, count=0, radius_nm=1_700_000, via=via_spec)

        vias = generate_return_via_ring(PositionNM(0, 0), ring_spec)

        assert vias == ()

    def test_ring_start_angle_rotates_pattern(self) -> None:
        """Start angle rotates the entire pattern."""
        center = PositionNM(0, 0)
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=650_000)

        ring_0 = ReturnViaRingSpec(
            pattern=ReturnViaPattern.RING,
            count=4,
            radius_nm=1_000_000,
            via=via_spec,
            start_angle_mdeg=0,
        )
        ring_45 = ReturnViaRingSpec(
            pattern=ReturnViaPattern.RING,
            count=4,
            radius_nm=1_000_000,
            via=via_spec,
            start_angle_mdeg=45_000,
        )

        vias_0 = generate_return_via_ring(center, ring_0)
        vias_45 = generate_return_via_ring(center, ring_45)

        # First via at 0 degrees should be at (radius, 0)
        assert vias_0[0].position.x > 0
        assert abs(vias_0[0].position.y) < 100

        # First via at 45 degrees should be at diagonal
        assert vias_45[0].position.x > 0
        assert vias_45[0].position.y > 0

    def test_invalid_radius_raises(self) -> None:
        """Zero or negative radius raises ValueError."""
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=650_000)
        ring_spec = ReturnViaRingSpec(pattern=ReturnViaPattern.RING, count=4, radius_nm=0, via=via_spec)

        with pytest.raises(ValueError, match="radius_nm must be positive"):
            generate_return_via_ring(PositionNM(0, 0), ring_spec)


class TestReturnViaGrid:
    """Tests for return via grid patterns."""

    def test_generate_grid_basic(self) -> None:
        """Basic grid generation with center excluded."""
        center = PositionNM(0, 0)
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=650_000)
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

        # 3x3 = 9, minus center = 8
        assert len(vias) == 8

    def test_grid_without_center_exclusion(self) -> None:
        """Grid without center exclusion includes all positions."""
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=650_000)
        grid_spec = ReturnViaGridSpec(
            pattern=ReturnViaPattern.GRID,
            rows=3,
            cols=3,
            row_pitch_nm=1_000_000,
            col_pitch_nm=1_000_000,
            via=via_spec,
            exclude_center=False,
        )

        vias = generate_return_via_grid(PositionNM(0, 0), grid_spec)

        assert len(vias) == 9

    def test_grid_is_centered_on_position(self) -> None:
        """Grid is centered on the given position."""
        center = PositionNM(10_000_000, 10_000_000)
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=650_000)
        grid_spec = ReturnViaGridSpec(
            pattern=ReturnViaPattern.GRID,
            rows=3,
            cols=3,
            row_pitch_nm=1_000_000,
            col_pitch_nm=1_000_000,
            via=via_spec,
            exclude_center=False,
        )

        vias = generate_return_via_grid(center, grid_spec)

        # Calculate centroid
        avg_x = sum(v.position.x for v in vias) // len(vias)
        avg_y = sum(v.position.y for v in vias) // len(vias)

        assert abs(avg_x - center.x) < 100
        assert abs(avg_y - center.y) < 100

    def test_invalid_pitch_raises(self) -> None:
        """Zero or negative pitch raises ValueError."""
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=650_000)
        grid_spec = ReturnViaGridSpec(
            pattern=ReturnViaPattern.GRID,
            rows=3,
            cols=3,
            row_pitch_nm=0,
            col_pitch_nm=1_000_000,
            via=via_spec,
        )

        with pytest.raises(ValueError, match="pitch values must be positive"):
            generate_return_via_grid(PositionNM(0, 0), grid_spec)


class TestReturnViaQuadrant:
    """Tests for quadrant via pattern."""

    def test_quadrant_creates_four_vias(self) -> None:
        """Quadrant pattern creates exactly 4 vias."""
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=650_000)

        vias = generate_return_via_quadrant(PositionNM(0, 0), 1_000_000, via_spec)

        assert len(vias) == 4

    def test_quadrant_vias_are_in_all_quadrants(self) -> None:
        """One via in each quadrant."""
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=650_000)
        center = PositionNM(0, 0)

        vias = generate_return_via_quadrant(center, 1_000_000, via_spec)

        # Check we have vias in all four quadrants
        q1 = any(v.position.x > 0 and v.position.y > 0 for v in vias)
        q2 = any(v.position.x < 0 and v.position.y > 0 for v in vias)
        q3 = any(v.position.x < 0 and v.position.y < 0 for v in vias)
        q4 = any(v.position.x > 0 and v.position.y < 0 for v in vias)

        assert q1 and q2 and q3 and q4

    def test_invalid_radius_raises(self) -> None:
        """Zero or negative radius raises ValueError."""
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=650_000)

        with pytest.raises(ValueError, match="radius_nm must be positive"):
            generate_return_via_quadrant(PositionNM(0, 0), 0, via_spec)


class TestViaTransition:
    """Tests for complete via transition generation."""

    def test_via_transition_with_ring(self) -> None:
        """Via transition with return ring."""
        center = PositionNM(40_000_000, 0)
        signal_spec = SignalViaSpec(drill_nm=300_000, diameter_nm=650_000)
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=650_000)
        ring_spec = ReturnViaRingSpec(pattern=ReturnViaPattern.RING, count=4, radius_nm=1_700_000, via=via_spec)

        result = generate_via_transition(center, signal_spec, ring_spec)

        assert result.signal_via.position == center
        assert len(result.return_vias) == 4

    def test_via_transition_without_return_vias(self) -> None:
        """Via transition without return vias."""
        center = PositionNM(40_000_000, 0)
        signal_spec = SignalViaSpec(drill_nm=300_000, diameter_nm=650_000)

        result = generate_via_transition(center, signal_spec, return_ring_spec=None)

        assert result.signal_via.position == center
        assert result.return_vias == ()


class TestMinimumReturnViaRadius:
    """Tests for minimum radius calculation."""

    def test_calculate_minimum_radius(self) -> None:
        """Minimum radius includes clearance."""
        signal_spec = SignalViaSpec(drill_nm=300_000, diameter_nm=650_000, pad_diameter_nm=900_000)
        return_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=650_000)
        clearance = 200_000

        min_radius = calculate_minimum_return_via_radius(signal_spec, return_spec, clearance)

        # (900_000 / 2) + (650_000 / 2) + 200_000 = 450_000 + 325_000 + 200_000 = 975_000
        assert min_radius == 975_000


class TestViaRingClearance:
    """Tests for ring circumference clearance calculation."""

    def test_calculate_ring_clearance(self) -> None:
        """Calculate clearance between adjacent vias on ring."""
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=600_000)
        ring_spec = ReturnViaRingSpec(pattern=ReturnViaPattern.RING, count=4, radius_nm=1_000_000, via=via_spec)

        clearance = calculate_via_ring_circumference_clearance(ring_spec)

        # Circumference = 2 * pi * 1_000_000 ≈ 6_283_185
        # Arc length per via ≈ 1_570_796
        # Clearance = 1_570_796 - 600_000 = 970_796
        assert clearance > 0
        assert clearance < 1_000_000

    def test_single_via_returns_diameter(self) -> None:
        """Single via on ring returns diameter as 'clearance'."""
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=600_000)
        ring_spec = ReturnViaRingSpec(pattern=ReturnViaPattern.RING, count=1, radius_nm=1_000_000, via=via_spec)

        clearance = calculate_via_ring_circumference_clearance(ring_spec)

        assert clearance == 2_000_000  # 2 * radius


class TestCircleAntipad:
    """Tests for circular antipad generation."""

    def test_generate_circle_basic(self) -> None:
        """Basic circular antipad generation."""
        center = PositionNM(0, 0)
        spec = CircleAntipadSpec(shape=CutoutShape.CIRCLE, r_nm=1_000_000, layer="In1.Cu")

        polygon = generate_circle_antipad(center, spec)

        assert polygon.polygon_type == PolygonType.CUTOUT
        assert polygon.layer == "In1.Cu"
        assert len(polygon.vertices) == 32  # default segments

    def test_circle_vertices_are_equidistant(self) -> None:
        """All vertices are equidistant from center."""
        center = PositionNM(10_000_000, 10_000_000)
        spec = CircleAntipadSpec(shape=CutoutShape.CIRCLE, r_nm=500_000, layer="In1.Cu")

        polygon = generate_circle_antipad(center, spec)

        for vertex in polygon.vertices:
            dx = vertex.x - center.x
            dy = vertex.y - center.y
            distance = int(math.sqrt(dx * dx + dy * dy))
            assert abs(distance - 500_000) < 100

    def test_invalid_radius_raises(self) -> None:
        """Zero or negative radius raises ValueError."""
        spec = CircleAntipadSpec(shape=CutoutShape.CIRCLE, r_nm=0, layer="In1.Cu")

        with pytest.raises(ValueError, match="r_nm must be positive"):
            generate_circle_antipad(PositionNM(0, 0), spec)

    def test_too_few_segments_raises(self) -> None:
        """Fewer than 3 segments raises ValueError."""
        spec = CircleAntipadSpec(shape=CutoutShape.CIRCLE, r_nm=1_000_000, layer="In1.Cu", segments=2)

        with pytest.raises(ValueError, match="segments must be at least 3"):
            generate_circle_antipad(PositionNM(0, 0), spec)


class TestRoundRectAntipad:
    """Tests for rounded rectangle antipad generation."""

    def test_generate_roundrect_basic(self) -> None:
        """Basic rounded rectangle generation."""
        center = PositionNM(0, 0)
        spec = RoundRectAntipadSpec(
            shape=CutoutShape.ROUNDRECT,
            rx_nm=1_200_000,
            ry_nm=900_000,
            corner_nm=250_000,
            layer="In1.Cu",
        )

        polygon = generate_roundrect_antipad(center, spec)

        assert polygon.polygon_type == PolygonType.CUTOUT
        assert polygon.layer == "In1.Cu"
        assert len(polygon.vertices) > 4  # More than a simple rectangle

    def test_roundrect_with_zero_corner_is_rectangle(self) -> None:
        """Zero corner radius creates a rectangle (4 corners, 4 vertices per corner = 4)."""
        center = PositionNM(0, 0)
        spec = RoundRectAntipadSpec(
            shape=CutoutShape.ROUNDRECT,
            rx_nm=1_000_000,
            ry_nm=500_000,
            corner_nm=0,
            layer="In1.Cu",
            corner_segments=4,
        )

        polygon = generate_roundrect_antipad(center, spec)

        # Should have 4 corners with no arc vertices
        assert len(polygon.vertices) == 4

    def test_invalid_dimensions_raises(self) -> None:
        """Zero or negative dimensions raises ValueError."""
        spec = RoundRectAntipadSpec(shape=CutoutShape.ROUNDRECT, rx_nm=0, ry_nm=1_000_000, corner_nm=100_000, layer="In1.Cu")

        with pytest.raises(ValueError, match="rx_nm and ry_nm must be positive"):
            generate_roundrect_antipad(PositionNM(0, 0), spec)


class TestSlotAntipad:
    """Tests for slot-shaped antipad generation."""

    def test_generate_slot_basic(self) -> None:
        """Basic slot generation."""
        center = PositionNM(0, 0)
        spec = SlotAntipadSpec(
            shape=CutoutShape.SLOT,
            length_nm=3_000_000,
            width_nm=1_500_000,
            rotation_mdeg=0,
            layer="In1.Cu",
        )

        polygon = generate_slot_antipad(center, spec)

        assert polygon.polygon_type == PolygonType.CUTOUT
        assert polygon.layer == "In1.Cu"
        assert len(polygon.vertices) > 4

    def test_slot_rotation_changes_orientation(self) -> None:
        """Slot rotation changes vertex positions."""
        center = PositionNM(0, 0)
        spec_0 = SlotAntipadSpec(
            shape=CutoutShape.SLOT,
            length_nm=3_000_000,
            width_nm=1_000_000,
            rotation_mdeg=0,
            layer="In1.Cu",
        )
        spec_90 = SlotAntipadSpec(
            shape=CutoutShape.SLOT,
            length_nm=3_000_000,
            width_nm=1_000_000,
            rotation_mdeg=90_000,
            layer="In1.Cu",
        )

        poly_0 = generate_slot_antipad(center, spec_0)
        poly_90 = generate_slot_antipad(center, spec_90)

        # At 0 degrees, slot extends in x direction
        max_x_0 = max(v.x for v in poly_0.vertices)
        max_y_0 = max(v.y for v in poly_0.vertices)

        # At 90 degrees, slot extends in y direction
        max_x_90 = max(v.x for v in poly_90.vertices)
        max_y_90 = max(v.y for v in poly_90.vertices)

        # Rotated slot should have swapped extent
        assert max_x_0 > max_y_0
        assert max_y_90 > max_x_90

    def test_short_slot_becomes_circle(self) -> None:
        """Slot shorter than its width becomes circular."""
        center = PositionNM(0, 0)
        spec = SlotAntipadSpec(
            shape=CutoutShape.SLOT,
            length_nm=500_000,
            width_nm=1_000_000,  # Width > length
            rotation_mdeg=0,
            layer="In1.Cu",
        )

        polygon = generate_slot_antipad(center, spec)

        # Should still generate a valid polygon
        assert len(polygon.vertices) > 3

    def test_invalid_dimensions_raises(self) -> None:
        """Zero or negative dimensions raises ValueError."""
        spec = SlotAntipadSpec(
            shape=CutoutShape.SLOT,
            length_nm=0,
            width_nm=1_000_000,
            rotation_mdeg=0,
            layer="In1.Cu",
        )

        with pytest.raises(ValueError, match="length_nm and width_nm must be positive"):
            generate_slot_antipad(PositionNM(0, 0), spec)


class TestRectangleAntipad:
    """Tests for simple rectangle antipad generation."""

    def test_generate_rectangle_basic(self) -> None:
        """Basic rectangle generation."""
        center = PositionNM(0, 0)
        spec = RectangleAntipadSpec(shape=CutoutShape.RECTANGLE, width_nm=2_000_000, height_nm=1_000_000, layer="In1.Cu")

        polygon = generate_rectangle_antipad(center, spec)

        assert polygon.polygon_type == PolygonType.CUTOUT
        assert polygon.layer == "In1.Cu"
        assert len(polygon.vertices) == 4

    def test_rectangle_is_centered(self) -> None:
        """Rectangle is centered on given position."""
        center = PositionNM(10_000_000, 5_000_000)
        spec = RectangleAntipadSpec(shape=CutoutShape.RECTANGLE, width_nm=2_000_000, height_nm=1_000_000, layer="In1.Cu")

        polygon = generate_rectangle_antipad(center, spec)

        # Calculate centroid
        avg_x = sum(v.x for v in polygon.vertices) // 4
        avg_y = sum(v.y for v in polygon.vertices) // 4

        assert avg_x == center.x
        assert avg_y == center.y


class TestGenerateAntipad:
    """Tests for the generic antipad dispatcher."""

    def test_dispatch_circle(self) -> None:
        """Dispatcher handles circle spec."""
        spec = CircleAntipadSpec(shape=CutoutShape.CIRCLE, r_nm=1_000_000, layer="In1.Cu")

        polygon = generate_antipad(PositionNM(0, 0), spec)

        assert polygon.polygon_type == PolygonType.CUTOUT

    def test_dispatch_roundrect(self) -> None:
        """Dispatcher handles roundrect spec."""
        spec = RoundRectAntipadSpec(
            shape=CutoutShape.ROUNDRECT, rx_nm=1_000_000, ry_nm=500_000, corner_nm=100_000, layer="In1.Cu"
        )

        polygon = generate_antipad(PositionNM(0, 0), spec)

        assert polygon.polygon_type == PolygonType.CUTOUT

    def test_dispatch_slot(self) -> None:
        """Dispatcher handles slot spec."""
        spec = SlotAntipadSpec(shape=CutoutShape.SLOT, length_nm=2_000_000, width_nm=1_000_000, rotation_mdeg=0, layer="In1.Cu")

        polygon = generate_antipad(PositionNM(0, 0), spec)

        assert polygon.polygon_type == PolygonType.CUTOUT

    def test_dispatch_rectangle(self) -> None:
        """Dispatcher handles rectangle spec."""
        spec = RectangleAntipadSpec(shape=CutoutShape.RECTANGLE, width_nm=2_000_000, height_nm=1_000_000, layer="In1.Cu")

        polygon = generate_antipad(PositionNM(0, 0), spec)

        assert polygon.polygon_type == PolygonType.CUTOUT


class TestPlaneCutoutForVia:
    """Tests for via-centered plane cutout generation."""

    def test_generate_cutout_for_via(self) -> None:
        """Generate cutout centered on via."""
        via_center = PositionNM(40_000_000, 0)

        polygon = generate_plane_cutout_for_via(via_center, via_diameter_nm=650_000, clearance_nm=200_000, layer="In1.Cu")

        assert polygon.polygon_type == PolygonType.CUTOUT
        assert polygon.layer == "In1.Cu"

    def test_cutout_radius_includes_clearance(self) -> None:
        """Cutout radius is via radius plus clearance."""
        via_center = PositionNM(0, 0)
        via_diameter = 600_000
        clearance = 200_000

        polygon = generate_plane_cutout_for_via(via_center, via_diameter_nm=via_diameter, clearance_nm=clearance, layer="In1.Cu")

        # Check vertex distance from center
        for vertex in polygon.vertices:
            distance = int(math.sqrt(vertex.x**2 + vertex.y**2))
            expected = via_diameter // 2 + clearance  # 300_000 + 200_000 = 500_000
            assert abs(distance - expected) < 100

    def test_negative_clearance_raises(self) -> None:
        """Negative clearance raises ValueError."""
        with pytest.raises(ValueError, match="clearance_nm must be non-negative"):
            generate_plane_cutout_for_via(PositionNM(0, 0), via_diameter_nm=650_000, clearance_nm=-100, layer="In1.Cu")


class TestMultiviaAntipad:
    """Tests for multi-via antipad generation."""

    def test_single_via_returns_circle(self) -> None:
        """Single via falls back to circular cutout."""
        centers = (PositionNM(0, 0),)

        polygon = generate_multivia_antipad(centers, via_diameter_nm=600_000, clearance_nm=200_000, layer="In1.Cu")

        assert polygon.polygon_type == PolygonType.CUTOUT

    def test_multiple_vias_creates_bounding_shape(self) -> None:
        """Multiple vias create a bounding rounded rectangle."""
        centers = (
            PositionNM(0, 0),
            PositionNM(1_000_000, 0),
            PositionNM(1_000_000, 1_000_000),
            PositionNM(0, 1_000_000),
        )

        polygon = generate_multivia_antipad(centers, via_diameter_nm=600_000, clearance_nm=200_000, layer="In1.Cu")

        assert polygon.polygon_type == PolygonType.CUTOUT
        assert len(polygon.vertices) > 4  # Rounded rectangle has more than 4 vertices

    def test_empty_centers_raises(self) -> None:
        """Empty via centers raises ValueError."""
        with pytest.raises(ValueError, match="via_centers must not be empty"):
            generate_multivia_antipad((), via_diameter_nm=600_000, clearance_nm=200_000, layer="In1.Cu")


class TestDeterminism:
    """Tests verifying deterministic geometry generation."""

    def test_cpwg_is_deterministic(self) -> None:
        """CPWG generation produces identical results on repeated calls."""
        origin = PositionNM(5_000_000, 0)
        cpwg_spec = CPWGSpec(w_nm=300_000, gap_nm=180_000, length_nm=25_000_000)
        fence_spec = GroundViaFenceSpec(pitch_nm=1_500_000, offset_from_gap_nm=800_000, drill_nm=300_000, diameter_nm=600_000)

        result1 = generate_cpwg_with_fence(origin, cpwg_spec, fence_spec)
        result2 = generate_cpwg_with_fence(origin, cpwg_spec, fence_spec)

        assert result1.signal_track == result2.signal_track
        assert result1.fence_vias_positive_y == result2.fence_vias_positive_y
        assert result1.fence_vias_negative_y == result2.fence_vias_negative_y

    def test_via_ring_is_deterministic(self) -> None:
        """Via ring generation produces identical results on repeated calls."""
        center = PositionNM(40_000_000, 0)
        via_spec = ReturnViaSpec(drill_nm=300_000, diameter_nm=650_000)
        ring_spec = ReturnViaRingSpec(pattern=ReturnViaPattern.RING, count=6, radius_nm=1_700_000, via=via_spec)

        vias1 = generate_return_via_ring(center, ring_spec)
        vias2 = generate_return_via_ring(center, ring_spec)

        assert vias1 == vias2

    def test_antipad_is_deterministic(self) -> None:
        """Antipad generation produces identical results on repeated calls."""
        center = PositionNM(40_000_000, 0)
        spec = RoundRectAntipadSpec(
            shape=CutoutShape.ROUNDRECT,
            rx_nm=1_200_000,
            ry_nm=900_000,
            corner_nm=250_000,
            layer="In1.Cu",
        )

        poly1 = generate_roundrect_antipad(center, spec)
        poly2 = generate_roundrect_antipad(center, spec)

        assert poly1.vertices == poly2.vertices
