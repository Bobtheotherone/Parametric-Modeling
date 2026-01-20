"""Tests for F0 (Calibration Thru Line) coupon builder.

These tests verify:
1. Feature composition correctly extracts board outline, ports, and transmission line
2. F0 builder rejects non-F0 specs
3. Track segments are generated correctly for F0 coupons
4. F0 composition properties are accurate

Satisfies REQ-M1-006.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.coupongen.builders.f0_builder import (
    BoardOutlineFeature,
    F0CouponBuilder,
    F0CouponComposition,
    PortFeature,
    TransmissionLineFeature,
    build_f0_coupon,
)
from formula_foundry.coupongen.families import FAMILY_F0, FAMILY_F1
from formula_foundry.coupongen.geom.primitives import PositionNM
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
        "antipads": {},
        "return_vias": {
            "pattern": "RING",
            "count": 4,
            "radius_nm": 1700000,
            "via": {"drill_nm": 300000, "diameter_nm": 650000},
        },
        "plane_cutouts": {},
    }
    return data


class TestBoardOutlineFeature:
    """Tests for BoardOutlineFeature."""

    def test_to_outline_conversion(self) -> None:
        """Test conversion to BoardOutline primitive."""
        feature = BoardOutlineFeature(
            width_nm=20000000,  # 20mm board width (y dimension)
            length_nm=80000000,  # 80mm board length (x dimension)
            corner_radius_nm=2000000,
        )
        outline = feature.to_outline()

        # BoardOutline uses vertices-based representation
        assert outline.corner_radius_nm == 2000000
        # Should have 4 vertices for rectangle
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
        # FootprintInstance uses separate lib/name fields
        assert instance.footprint_lib == "Coupongen_Connectors"
        assert instance.footprint_name == "SMA_EndLaunch_Generic"
        assert instance.footprint_path == "Coupongen_Connectors:SMA_EndLaunch_Generic"
        assert instance.position == PositionNM(5000000, 0)
        assert instance.rotation_mdeg == 180000  # 180 degrees in millidegrees
        assert instance.layer == "F.Cu"

    def test_to_footprint_instance_right(self) -> None:
        """Test conversion to FootprintInstance for right port."""
        feature = PortFeature(
            footprint="Coupongen_Connectors:SMA_EndLaunch_Generic",
            position=PositionNM(75000000, 0),
            rotation_deg=0,
            side="right",
        )
        instance = feature.to_footprint_instance()

        assert instance.reference == "J_RIGHT"
        assert instance.rotation_mdeg == 0

    def test_port_feature_properties(self) -> None:
        """Test PortFeature computed properties."""
        feature = PortFeature(
            footprint="Coupongen_Connectors:SMA_EndLaunch_Generic",
            position=PositionNM(5000000, 0),
            rotation_deg=90,
            side="left",
        )

        assert feature.footprint_lib == "Coupongen_Connectors"
        assert feature.footprint_name == "SMA_EndLaunch_Generic"
        assert feature.rotation_mdeg == 90000


class TestTransmissionLineFeature:
    """Tests for TransmissionLineFeature."""

    def test_cpwg_spec_generation(self) -> None:
        """Test CPWG spec generation for left and right segments."""
        feature = TransmissionLineFeature(
            w_nm=300000,
            gap_nm=180000,
            length_left_nm=25000000,
            length_right_nm=25000000,
            layer="F.Cu",
            left_start_x_nm=5000000,
            right_end_x_nm=75000000,
        )

        left_spec = feature.get_cpwg_spec_left()
        assert left_spec.w_nm == 300000
        assert left_spec.gap_nm == 180000
        assert left_spec.length_nm == 25000000
        assert left_spec.layer == "F.Cu"
        assert left_spec.net_id == 1

        right_spec = feature.get_cpwg_spec_right()
        assert right_spec.length_nm == 25000000

    def test_track_segments_generation(self) -> None:
        """Test track segment generation."""
        feature = TransmissionLineFeature(
            w_nm=300000,
            gap_nm=180000,
            length_left_nm=25000000,
            length_right_nm=25000000,
            layer="F.Cu",
            left_start_x_nm=5000000,
            right_end_x_nm=75000000,
        )

        left_seg, right_seg = feature.to_track_segments()

        # Left segment: 5mm to 30mm (5mm + 25mm)
        assert left_seg.start == PositionNM(5000000, 0)
        assert left_seg.end == PositionNM(30000000, 0)
        assert left_seg.width_nm == 300000
        assert left_seg.layer == "F.Cu"
        assert left_seg.net_id == 1

        # Right segment: 30mm to 75mm
        assert right_seg.start == PositionNM(30000000, 0)
        assert right_seg.end == PositionNM(75000000, 0)


class TestF0CouponBuilder:
    """Tests for F0CouponBuilder."""

    def test_builder_accepts_f0_spec(self) -> None:
        """Test builder accepts F0 spec."""
        spec = CouponSpec.model_validate(_f0_spec_data())
        resolved = resolve(spec)

        builder = F0CouponBuilder(spec, resolved)
        composition = builder.build()

        assert isinstance(composition, F0CouponComposition)

    def test_builder_rejects_f1_spec(self) -> None:
        """Test builder rejects F1 spec."""
        spec = CouponSpec.model_validate(_f1_spec_data())
        resolved = resolve(spec)

        with pytest.raises(ValueError, match="F0CouponBuilder requires"):
            F0CouponBuilder(spec, resolved)

    def test_board_outline_extraction(self) -> None:
        """Test board outline feature extraction."""
        spec = CouponSpec.model_validate(_f0_spec_data())
        resolved = resolve(spec)
        composition = build_f0_coupon(spec, resolved)

        assert composition.board_outline.width_nm == 20000000
        assert composition.board_outline.length_nm == 80000000
        assert composition.board_outline.corner_radius_nm == 2000000

    def test_port_extraction(self) -> None:
        """Test port feature extraction."""
        spec = CouponSpec.model_validate(_f0_spec_data())
        resolved = resolve(spec)
        composition = build_f0_coupon(spec, resolved)

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
        """Test transmission line feature extraction."""
        spec = CouponSpec.model_validate(_f0_spec_data())
        resolved = resolve(spec)
        composition = build_f0_coupon(spec, resolved)

        tl = composition.transmission_line
        assert tl.w_nm == 300000
        assert tl.gap_nm == 180000
        assert tl.length_left_nm == 25000000
        assert tl.length_right_nm == 25000000
        assert tl.layer == "F.Cu"

    def test_total_trace_length(self) -> None:
        """Test total trace length calculation."""
        spec = CouponSpec.model_validate(_f0_spec_data())
        resolved = resolve(spec)
        composition = build_f0_coupon(spec, resolved)

        # 25mm left + 25mm right = 50mm
        assert composition.total_trace_length_nm == 50000000


class TestF0CompositionWithGoldenSpecs:
    """Tests F0 builder with golden spec files."""

    @pytest.fixture
    def golden_spec_dir(self) -> Path:
        """Get golden spec directory."""
        return Path(__file__).parent / "golden_specs"

    def test_f0_golden_spec_001(self, golden_spec_dir: Path) -> None:
        """Test F0 builder with golden spec f0_cal_001."""
        spec_path = golden_spec_dir / "f0_cal_001.json"
        if not spec_path.exists():
            pytest.skip("Golden spec not found")

        data = json.loads(spec_path.read_text())
        spec = CouponSpec.model_validate(data)
        resolved = resolve(spec)

        composition = build_f0_coupon(spec, resolved)

        assert composition.board_outline.width_nm == 20000000
        assert composition.board_outline.length_nm == 80000000
        assert composition.left_port.side == "left"
        assert composition.right_port.side == "right"

    @pytest.mark.parametrize("spec_num", range(1, 11))
    def test_f0_golden_specs_build(self, golden_spec_dir: Path, spec_num: int) -> None:
        """Test all F0 golden specs can be built."""
        spec_path = golden_spec_dir / f"f0_cal_{spec_num:03d}.json"
        if not spec_path.exists():
            pytest.skip(f"Golden spec {spec_path.name} not found")

        data = json.loads(spec_path.read_text())
        spec = CouponSpec.model_validate(data)
        resolved = resolve(spec)

        # Should not raise
        composition = build_f0_coupon(spec, resolved)
        assert isinstance(composition, F0CouponComposition)


class TestF0CouponBuilderIntegration:
    """Integration tests for F0 builder with full pipeline."""

    def test_f0_composition_to_primitives(self) -> None:
        """Test F0 composition converts to primitives correctly."""
        spec = CouponSpec.model_validate(_f0_spec_data())
        resolved = resolve(spec)
        composition = build_f0_coupon(spec, resolved)

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

    def test_f0_track_continuity(self) -> None:
        """Test F0 tracks form continuous path."""
        spec = CouponSpec.model_validate(_f0_spec_data())
        resolved = resolve(spec)
        composition = build_f0_coupon(spec, resolved)

        left_track, right_track = composition.transmission_line.to_track_segments()

        # Left track end should equal right track start (continuity)
        assert left_track.end == right_track.start

        # Both on centerline (y=0)
        assert left_track.start.y == 0
        assert left_track.end.y == 0
        assert right_track.start.y == 0
        assert right_track.end.y == 0
