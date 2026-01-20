"""Tests for F1 (Single-Ended Via Transition) coupon builder.

These tests verify:
1. Feature composition correctly extracts board outline, ports, transmission line,
   and discontinuity features
2. F1 builder rejects non-F1 specs
3. Via transition geometry (signal via, return vias) is generated correctly
4. Antipad and plane cutout features are extracted and converted to polygons
5. Track segments connect to the discontinuity center correctly
6. F1 composition properties provide access to all geometry primitives

Satisfies REQ-M1-007.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.coupongen.builders.f1_builder import (
    AntipadFeature,
    BoardOutlineFeature,
    DiscontinuityFeature,
    F1CouponBuilder,
    F1CouponComposition,
    PlaneCutoutFeature,
    PortFeature,
    ReturnViasFeature,
    SignalViaFeature,
    TransmissionLineFeature,
    build_f1_coupon,
)
from formula_foundry.coupongen.families import FAMILY_F0, FAMILY_F1
from formula_foundry.coupongen.geom.primitives import (
    Polygon,
    PolygonType,
    PositionNM,
    Via,
)
from formula_foundry.coupongen.resolve import resolve
from formula_foundry.coupongen.spec import CouponSpec


def _base_spec_data() -> dict[str, Any]:
    """Create base spec data for testing."""
    return {
        "schema_version": 1,
        "units": "nm",
        "toolchain": {
            "kicad": {
                "version": "9.0.7",
                "docker_image": "kicad/kicad:9.0.7@sha256:deadbeef",
            }
        },
        "fab_profile": {"id": "oshpark_4layer", "overrides": {}},
        "stackup": {
            "copper_layers": 4,
            "thicknesses_nm": {
                "L1_to_L2": 180000,
                "L2_to_L3": 800000,
                "L3_to_L4": 180000,
            },
            "materials": {"er": 4.1, "loss_tangent": 0.02},
        },
        "board": {
            "outline": {
                "width_nm": 20000000,
                "length_nm": 80000000,
                "corner_radius_nm": 2000000,
            },
            "origin": {"mode": "EDGE_L_CENTER"},
            "text": {"coupon_id": "${COUPON_ID}", "include_manifest_hash": True},
        },
        "connectors": {
            "left": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [5000000, 0],
                "rotation_deg": 180,
            },
            "right": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [75000000, 0],
                "rotation_deg": 0,
            },
        },
        "transmission_line": {
            "type": "CPWG",
            "layer": "F.Cu",
            "w_nm": 300000,
            "gap_nm": 180000,
            "length_left_nm": 25000000,
            "length_right_nm": 25000000,
            "ground_via_fence": None,
        },
        "constraints": {
            "mode": "REJECT",
            "drc": {"must_pass": True, "severity": "all"},
            "symmetry": {"enforce": True},
            "allow_unconnected_copper": False,
        },
        "export": {
            "gerbers": {"enabled": True, "format": "gerbers"},
            "drill": {"enabled": True, "format": "excellon"},
            "outputs_dir": "artifacts/",
        },
    }


def _f0_spec_data() -> dict[str, Any]:
    """Create F0 spec data for testing."""
    data = _base_spec_data()
    data["coupon_family"] = FAMILY_F0
    data["discontinuity"] = None
    return data


def _f1_spec_data() -> dict[str, Any]:
    """Create F1 spec data for testing."""
    data = _base_spec_data()
    data["coupon_family"] = FAMILY_F1
    data["discontinuity"] = {
        "type": "VIA_TRANSITION",
        "signal_via": {
            "drill_nm": 300000,
            "diameter_nm": 650000,
            "pad_diameter_nm": 900000,
        },
        "antipads": {
            "In1.Cu": {
                "shape": "ROUNDRECT",
                "rx_nm": 1200000,
                "ry_nm": 900000,
                "corner_nm": 250000,
            },
            "In2.Cu": {
                "shape": "CIRCLE",
                "r_nm": 1100000,
            },
        },
        "return_vias": {
            "pattern": "RING",
            "count": 4,
            "radius_nm": 1700000,
            "via": {"drill_nm": 300000, "diameter_nm": 650000},
        },
        "plane_cutouts": {
            "In1.Cu": {
                "shape": "SLOT",
                "length_nm": 3000000,
                "width_nm": 1500000,
                "rotation_deg": 0,
            },
        },
    }
    return data


def _f1_spec_data_minimal() -> dict[str, Any]:
    """Create minimal F1 spec data (no antipads/cutouts/return vias)."""
    data = _base_spec_data()
    data["coupon_family"] = FAMILY_F1
    data["discontinuity"] = {
        "type": "VIA_TRANSITION",
        "signal_via": {
            "drill_nm": 300000,
            "diameter_nm": 650000,
            "pad_diameter_nm": 900000,
        },
        "antipads": {},
        "return_vias": None,
        "plane_cutouts": {},
    }
    return data


class TestBoardOutlineFeature:
    """Tests for BoardOutlineFeature."""

    def test_to_outline_conversion(self) -> None:
        """Test conversion to BoardOutline primitive."""
        feature = BoardOutlineFeature(
            width_nm=20000000,
            length_nm=80000000,
            corner_radius_nm=2000000,
        )
        outline = feature.to_outline()

        assert outline.corner_radius_nm == 2000000
        assert len(outline.vertices) == 4


class TestPortFeature:
    """Tests for PortFeature."""

    def test_to_footprint_instance_left(self) -> None:
        """Test conversion to FootprintInstance for left port."""
        feature = PortFeature(
            footprint="Coupongen_Connectors:SMA_EndLaunch_Generic",
            position=PositionNM(5000000, 0),
            rotation_deg=180,
            side="left",
        )
        instance = feature.to_footprint_instance()

        assert instance.reference == "J_LEFT"
        assert instance.footprint_lib == "Coupongen_Connectors"
        assert instance.footprint_name == "SMA_EndLaunch_Generic"
        assert instance.position == PositionNM(5000000, 0)
        assert instance.rotation_mdeg == 180000


class TestTransmissionLineFeature:
    """Tests for TransmissionLineFeature with discontinuity."""

    def test_track_segments_to_discontinuity(self) -> None:
        """Test track segments connect at discontinuity center."""
        feature = TransmissionLineFeature(
            w_nm=300000,
            gap_nm=180000,
            length_left_nm=25000000,
            length_right_nm=25000000,
            layer="F.Cu",
            left_start_x_nm=5000000,
            right_end_x_nm=75000000,
            discontinuity_x_nm=30000000,  # 5mm + 25mm
        )

        left_seg, right_seg = feature.to_track_segments()

        # Left segment: connector (5mm) to discontinuity (30mm)
        assert left_seg.start == PositionNM(5000000, 0)
        assert left_seg.end == PositionNM(30000000, 0)

        # Right segment: discontinuity (30mm) to connector (75mm)
        assert right_seg.start == PositionNM(30000000, 0)
        assert right_seg.end == PositionNM(75000000, 0)

        # Track continuity at discontinuity
        assert left_seg.end == right_seg.start


class TestSignalViaFeature:
    """Tests for SignalViaFeature."""

    def test_to_via_primitive(self) -> None:
        """Test conversion to Via primitive."""
        feature = SignalViaFeature(
            drill_nm=300000,
            diameter_nm=650000,
            pad_diameter_nm=900000,
            position=PositionNM(30000000, 0),
        )

        via = feature.to_via()

        assert via.position == PositionNM(30000000, 0)
        assert via.drill_nm == 300000
        assert via.diameter_nm == 650000
        assert via.net_id == 1  # Signal net

    def test_to_signal_via_spec(self) -> None:
        """Test conversion to SignalViaSpec."""
        feature = SignalViaFeature(
            drill_nm=300000,
            diameter_nm=650000,
            pad_diameter_nm=900000,
            position=PositionNM(30000000, 0),
        )

        spec = feature.to_signal_via_spec()

        assert spec.drill_nm == 300000
        assert spec.diameter_nm == 650000
        assert spec.pad_diameter_nm == 900000
        assert spec.net_id == 1


class TestReturnViasFeature:
    """Tests for ReturnViasFeature."""

    def test_generate_vias_ring_pattern(self) -> None:
        """Test return via generation with ring pattern."""
        feature = ReturnViasFeature(
            pattern="RING",
            count=4,
            radius_nm=1700000,
            drill_nm=300000,
            diameter_nm=650000,
            center=PositionNM(30000000, 0),
        )

        vias = feature.generate_vias()

        assert len(vias) == 4
        for via in vias:
            assert isinstance(via, Via)
            assert via.drill_nm == 300000
            assert via.diameter_nm == 650000
            assert via.net_id == 0  # Ground net


class TestAntipadFeature:
    """Tests for AntipadFeature."""

    def test_circle_antipad_to_polygon(self) -> None:
        """Test circle antipad conversion to polygon."""
        feature = AntipadFeature(
            layer="In1.Cu",
            shape="CIRCLE",
            center=PositionNM(30000000, 0),
            r_nm=1100000,
        )

        polygon = feature.to_polygon()

        assert isinstance(polygon, Polygon)
        assert polygon.layer == "In1.Cu"
        assert polygon.polygon_type == PolygonType.CUTOUT
        assert len(polygon.vertices) >= 8  # Circle approximation

    def test_roundrect_antipad_to_polygon(self) -> None:
        """Test roundrect antipad conversion to polygon."""
        feature = AntipadFeature(
            layer="In2.Cu",
            shape="ROUNDRECT",
            center=PositionNM(30000000, 0),
            rx_nm=1200000,
            ry_nm=900000,
            corner_nm=250000,
        )

        polygon = feature.to_polygon()

        assert isinstance(polygon, Polygon)
        assert polygon.layer == "In2.Cu"
        assert polygon.polygon_type == PolygonType.CUTOUT

    def test_circle_antipad_requires_radius(self) -> None:
        """Test circle antipad raises error without r_nm."""
        feature = AntipadFeature(
            layer="In1.Cu",
            shape="CIRCLE",
            center=PositionNM(30000000, 0),
        )

        with pytest.raises(ValueError, match="r_nm required"):
            feature.to_polygon()

    def test_roundrect_antipad_requires_dimensions(self) -> None:
        """Test roundrect antipad raises error without rx_nm/ry_nm."""
        feature = AntipadFeature(
            layer="In1.Cu",
            shape="ROUNDRECT",
            center=PositionNM(30000000, 0),
            rx_nm=1200000,
            # Missing ry_nm
        )

        with pytest.raises(ValueError, match="rx_nm and ry_nm required"):
            feature.to_polygon()


class TestPlaneCutoutFeature:
    """Tests for PlaneCutoutFeature."""

    def test_slot_cutout_to_polygon(self) -> None:
        """Test slot cutout conversion to polygon."""
        feature = PlaneCutoutFeature(
            layer="In1.Cu",
            shape="SLOT",
            center=PositionNM(30000000, 0),
            length_nm=3000000,
            width_nm=1500000,
            rotation_deg=0,
        )

        polygon = feature.to_polygon()

        assert isinstance(polygon, Polygon)
        assert polygon.layer == "In1.Cu"
        assert polygon.polygon_type == PolygonType.CUTOUT

    def test_rectangle_cutout_to_polygon(self) -> None:
        """Test rectangle cutout conversion to polygon."""
        feature = PlaneCutoutFeature(
            layer="In2.Cu",
            shape="RECTANGLE",
            center=PositionNM(30000000, 0),
            length_nm=2000000,
            width_nm=1000000,
            rotation_deg=0,
        )

        polygon = feature.to_polygon()

        assert isinstance(polygon, Polygon)
        assert polygon.layer == "In2.Cu"
        assert len(polygon.vertices) == 4  # Rectangle has 4 vertices


class TestDiscontinuityFeature:
    """Tests for DiscontinuityFeature."""

    def test_generate_via_transition(self) -> None:
        """Test via transition generation."""
        center = PositionNM(30000000, 0)
        signal_via = SignalViaFeature(
            drill_nm=300000,
            diameter_nm=650000,
            pad_diameter_nm=900000,
            position=center,
        )
        return_vias = ReturnViasFeature(
            pattern="RING",
            count=4,
            radius_nm=1700000,
            drill_nm=300000,
            diameter_nm=650000,
            center=center,
        )

        discontinuity = DiscontinuityFeature(
            signal_via=signal_via,
            return_vias=return_vias,
            antipads=(),
            plane_cutouts=(),
            center=center,
        )

        result = discontinuity.generate_via_transition()

        assert result.signal_via.position == center
        assert len(result.return_vias) == 4

    def test_generate_antipad_polygons(self) -> None:
        """Test antipad polygon generation."""
        center = PositionNM(30000000, 0)
        signal_via = SignalViaFeature(
            drill_nm=300000,
            diameter_nm=650000,
            pad_diameter_nm=900000,
            position=center,
        )
        antipad1 = AntipadFeature(
            layer="In1.Cu",
            shape="CIRCLE",
            center=center,
            r_nm=1100000,
        )
        antipad2 = AntipadFeature(
            layer="In2.Cu",
            shape="ROUNDRECT",
            center=center,
            rx_nm=1200000,
            ry_nm=900000,
            corner_nm=250000,
        )

        discontinuity = DiscontinuityFeature(
            signal_via=signal_via,
            return_vias=None,
            antipads=(antipad1, antipad2),
            plane_cutouts=(),
            center=center,
        )

        polygons = discontinuity.generate_antipad_polygons()

        assert len(polygons) == 2
        assert all(isinstance(p, Polygon) for p in polygons)


class TestF1CouponBuilder:
    """Tests for F1CouponBuilder."""

    def test_builder_accepts_f1_spec(self) -> None:
        """Test builder accepts F1 spec."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)

        builder = F1CouponBuilder(spec, resolved)
        composition = builder.build()

        assert isinstance(composition, F1CouponComposition)

    def test_builder_rejects_f0_spec(self) -> None:
        """Test builder rejects F0 spec."""
        spec = CouponSpec.model_validate(_f0_spec_data())
        resolved = resolve(spec)

        with pytest.raises(ValueError, match="F1CouponBuilder requires"):
            F1CouponBuilder(spec, resolved)

    def test_board_outline_extraction(self) -> None:
        """Test board outline feature extraction."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        assert composition.board_outline.width_nm == 20000000
        assert composition.board_outline.length_nm == 80000000
        assert composition.board_outline.corner_radius_nm == 2000000

    def test_port_extraction(self) -> None:
        """Test port feature extraction."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        # Left port
        assert composition.left_port.footprint == "Coupongen_Connectors:SMA_EndLaunch_Generic"
        assert composition.left_port.position == PositionNM(5000000, 0)
        assert composition.left_port.rotation_deg == 180
        assert composition.left_port.side == "left"

        # Right port
        assert composition.right_port.position == PositionNM(75000000, 0)
        assert composition.right_port.rotation_deg == 0
        assert composition.right_port.side == "right"

    def test_transmission_line_extraction(self) -> None:
        """Test transmission line feature extraction.

        CP-2.2: length_right_nm is derived from continuity formula:
        length_right = right_connector_x - discontinuity_x
                     = 75mm - (5mm + 25mm) = 45mm
        """
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        tl = composition.transmission_line
        assert tl.w_nm == 300000
        assert tl.gap_nm == 180000
        assert tl.length_left_nm == 25000000
        # CP-2.2: length_right is derived, not from spec
        # Derived: right_connector_x (75mm) - discontinuity_x (30mm) = 45mm
        assert tl.length_right_nm == 45000000
        assert tl.layer == "F.Cu"
        # Discontinuity at 5mm + 25mm = 30mm
        assert tl.discontinuity_x_nm == 30000000

    def test_discontinuity_extraction(self) -> None:
        """Test discontinuity feature extraction."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        disc = composition.discontinuity
        # Discontinuity center at left_connector + left_length
        assert disc.center == PositionNM(30000000, 0)

        # Signal via
        assert disc.signal_via.drill_nm == 300000
        assert disc.signal_via.diameter_nm == 650000
        assert disc.signal_via.pad_diameter_nm == 900000

        # Return vias
        assert disc.return_vias is not None
        assert disc.return_vias.pattern == "RING"
        assert disc.return_vias.count == 4
        assert disc.return_vias.radius_nm == 1700000

    def test_antipad_extraction(self) -> None:
        """Test antipad feature extraction."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        antipads = composition.discontinuity.antipads
        assert len(antipads) == 2

        # Find specific antipads by layer
        layers = {ap.layer for ap in antipads}
        assert "In1.Cu" in layers
        assert "In2.Cu" in layers

    def test_plane_cutout_extraction(self) -> None:
        """Test plane cutout feature extraction."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        cutouts = composition.discontinuity.plane_cutouts
        assert len(cutouts) == 1

        cutout = cutouts[0]
        assert cutout.layer == "In1.Cu"
        assert cutout.shape == "SLOT"
        assert cutout.length_nm == 3000000
        assert cutout.width_nm == 1500000

    def test_total_trace_length(self) -> None:
        """Test total trace length calculation.

        CP-2.2: length_right_nm is derived from continuity formula.
        Total trace = left (25mm) + derived_right (45mm) = 70mm
        """
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        # 25mm left + 45mm derived right = 70mm (full span between connectors)
        assert composition.total_trace_length_nm == 70000000

    def test_discontinuity_position(self) -> None:
        """Test discontinuity position accessor."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        # 5mm (left connector) + 25mm (left trace) = 30mm
        assert composition.discontinuity_position == PositionNM(30000000, 0)


class TestF1CompositionProperties:
    """Tests for F1CouponComposition property accessors."""

    def test_signal_via_property(self) -> None:
        """Test signal_via property returns Via primitive."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        via = composition.signal_via

        assert isinstance(via, Via)
        assert via.position == composition.discontinuity_position
        assert via.drill_nm == 300000
        assert via.diameter_nm == 650000

    def test_return_vias_property(self) -> None:
        """Test return_vias property returns Via tuple."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        vias = composition.return_vias

        assert len(vias) == 4
        assert all(isinstance(v, Via) for v in vias)

    def test_all_antipads_property(self) -> None:
        """Test all_antipads property returns Polygon tuple."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        antipads = composition.all_antipads

        assert len(antipads) == 2
        assert all(isinstance(p, Polygon) for p in antipads)
        assert all(p.polygon_type == PolygonType.CUTOUT for p in antipads)

    def test_all_cutouts_property(self) -> None:
        """Test all_cutouts property returns Polygon tuple."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        cutouts = composition.all_cutouts

        assert len(cutouts) == 1
        assert all(isinstance(p, Polygon) for p in cutouts)


class TestF1MinimalSpec:
    """Tests for F1 with minimal spec (no optional features)."""

    def test_minimal_spec_builds(self) -> None:
        """Test F1 builder with minimal spec (no antipads/cutouts/return vias)."""
        spec = CouponSpec.model_validate(_f1_spec_data_minimal())
        resolved = resolve(spec)

        composition = build_f1_coupon(spec, resolved)

        assert isinstance(composition, F1CouponComposition)

    def test_minimal_spec_no_return_vias(self) -> None:
        """Test minimal spec has no return vias."""
        spec = CouponSpec.model_validate(_f1_spec_data_minimal())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        assert composition.discontinuity.return_vias is None
        assert composition.return_vias == ()

    def test_minimal_spec_no_antipads(self) -> None:
        """Test minimal spec has no antipads."""
        spec = CouponSpec.model_validate(_f1_spec_data_minimal())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        assert len(composition.discontinuity.antipads) == 0
        assert composition.all_antipads == ()

    def test_minimal_spec_no_cutouts(self) -> None:
        """Test minimal spec has no plane cutouts."""
        spec = CouponSpec.model_validate(_f1_spec_data_minimal())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        assert len(composition.discontinuity.plane_cutouts) == 0
        assert composition.all_cutouts == ()


class TestF1CouponBuilderIntegration:
    """Integration tests for F1 builder with full pipeline."""

    def test_f1_composition_to_primitives(self) -> None:
        """Test F1 composition converts to primitives correctly."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        # Convert features to primitives
        outline = composition.board_outline.to_outline()
        left_fp = composition.left_port.to_footprint_instance()
        right_fp = composition.right_port.to_footprint_instance()
        left_track, right_track = composition.transmission_line.to_track_segments()

        # Verify primitive types
        from formula_foundry.coupongen.geom.primitives import (
            BoardOutline,
            FootprintInstance,
            TrackSegment,
        )

        assert isinstance(outline, BoardOutline)
        assert isinstance(left_fp, FootprintInstance)
        assert isinstance(right_fp, FootprintInstance)
        assert isinstance(left_track, TrackSegment)
        assert isinstance(right_track, TrackSegment)

    def test_f1_track_continuity_at_discontinuity(self) -> None:
        """Test F1 tracks meet at discontinuity center."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        left_track, right_track = composition.transmission_line.to_track_segments()

        # Left track end should equal right track start (at discontinuity)
        assert left_track.end == right_track.start

        # And should be at discontinuity center
        assert left_track.end == composition.discontinuity_position

        # Both on centerline (y=0)
        assert left_track.start.y == 0
        assert left_track.end.y == 0
        assert right_track.start.y == 0
        assert right_track.end.y == 0

    def test_f1_signal_via_at_discontinuity(self) -> None:
        """Test signal via is at discontinuity center."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        assert composition.signal_via.position == composition.discontinuity_position

    def test_f1_return_vias_around_signal_via(self) -> None:
        """Test return vias are positioned around signal via."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        center = composition.discontinuity_position
        radius = 1700000  # From spec

        for via in composition.return_vias:
            # Calculate distance from center
            dx = via.position.x - center.x
            dy = via.position.y - center.y
            distance = (dx**2 + dy**2) ** 0.5

            # Should be approximately at the specified radius
            assert abs(distance - radius) < 100  # Allow small rounding error


class TestF1GoldenSpecs:
    """Tests F1 builder with golden spec files."""

    @pytest.fixture
    def golden_spec_dir(self) -> Path:
        """Get golden spec directory."""
        return Path(__file__).parent / "golden_specs"

    @pytest.mark.parametrize("spec_num", range(1, 11))
    def test_f1_golden_specs_build(self, golden_spec_dir: Path, spec_num: int) -> None:
        """Test all F1 golden specs can be built."""
        spec_path = golden_spec_dir / f"f1_via_{spec_num:03d}.json"
        if not spec_path.exists():
            pytest.skip(f"Golden spec {spec_path.name} not found")

        data = json.loads(spec_path.read_text())
        spec = CouponSpec.model_validate(data)
        resolved = resolve(spec)

        # Should not raise
        composition = build_f1_coupon(spec, resolved)
        assert isinstance(composition, F1CouponComposition)


class TestF1RequirementCoverage:
    """Tests to verify REQ-M1-007 coverage."""

    def test_req_m1_007_end_launch_connectors(self) -> None:
        """REQ-M1-007: F1 has end-launch connectors at both ends."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        # Has left and right ports
        assert composition.left_port is not None
        assert composition.right_port is not None
        assert composition.left_port.side == "left"
        assert composition.right_port.side == "right"

    def test_req_m1_007_cpwg_segments(self) -> None:
        """REQ-M1-007: F1 has CPWG on both sides of via transition."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        # Has transmission line with left and right lengths
        assert composition.transmission_line.length_left_nm > 0
        assert composition.transmission_line.length_right_nm > 0

        # Can generate track segments
        left, right = composition.transmission_line.to_track_segments()
        assert left is not None
        assert right is not None

    def test_req_m1_007_via_transition(self) -> None:
        """REQ-M1-007: F1 has via transition (top to inner or top to bottom)."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        # Has signal via
        via = composition.signal_via
        assert via is not None
        assert via.layers == ("F.Cu", "B.Cu")  # Default is top to bottom

    def test_req_m1_007_antipads(self) -> None:
        """REQ-M1-007: F1 includes antipads/cutouts."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        # Has antipads
        assert len(composition.all_antipads) > 0

    def test_req_m1_007_return_vias(self) -> None:
        """REQ-M1-007: F1 includes return vias."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        # Has return vias
        assert len(composition.return_vias) > 0

    def test_req_m1_007_feature_composition_graph(self) -> None:
        """REQ-M1-007: F1 has full feature composition graph."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)
        composition = build_f1_coupon(spec, resolved)

        # Verify all required features are present
        assert composition.board_outline is not None
        assert composition.left_port is not None
        assert composition.right_port is not None
        assert composition.transmission_line is not None
        assert composition.discontinuity is not None
        assert composition.discontinuity.signal_via is not None
