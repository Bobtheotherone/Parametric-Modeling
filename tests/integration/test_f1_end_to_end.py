# SPDX-License-Identifier: MIT
"""End-to-end integration tests for Family F1 (Single-Ended Via Transition).

This module tests the complete F1 coupon generation pipeline:
    spec -> resolve -> LayoutPlan -> BoardWriter -> .kicad_pcb

F1 coupons are via transition structures:
    end-launch -> CPWG -> via transition -> CPWG -> end-launch

These tests verify:
1. F1 specs load and validate correctly
2. LayoutPlan computes correct geometry (left/right segments, discontinuity)
3. BoardWriter generates valid KiCad board file with vias, antipads, cutouts
4. Generated board is deterministic (same spec -> same output)
5. Return vias and antipads are correctly placed
6. All golden specs produce valid boards

Satisfies REQ-M1-007.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from formula_foundry.coupongen.api import (
    generate_kicad,
    load_spec,
    resolve_spec,
)
from formula_foundry.coupongen.builders.f1_builder import (
    F1CouponComposition,
    build_f1_coupon,
)
from formula_foundry.coupongen.families import FAMILY_F1
from formula_foundry.coupongen.geom.layout import LayoutPlan
from formula_foundry.coupongen.kicad import (
    build_board_text,
    parse,
    write_board,
)
from formula_foundry.coupongen.kicad.board_writer import BoardWriter
from formula_foundry.coupongen.resolve import resolve
from formula_foundry.coupongen.spec import CouponSpec

# Test data directory
REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_SPECS_DIR = REPO_ROOT / "tests" / "golden_specs"


def _minimal_f1_spec_data() -> dict[str, Any]:
    """Create minimal valid F1 spec data with via transition."""
    return {
        "schema_version": 1,
        "coupon_family": FAMILY_F1,
        "units": "nm",
        "toolchain": {
            "kicad": {
                "version": "9.0.7",
                "docker_image": "kicad/kicad:9.0.7@sha256:0000000000000000000000000000000000000000000000000000000000000001",
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
            "text": {"coupon_id": "F1-TEST-001", "include_manifest_hash": True},
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
            # length_right_nm is derived from continuity formula for F1
        },
        "discontinuity": {
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


def _minimal_f1_spec_data_no_return_vias() -> dict[str, Any]:
    """Create F1 spec data without return vias."""
    data = _minimal_f1_spec_data()
    data["discontinuity"]["return_vias"] = None
    data["discontinuity"]["antipads"] = {}
    data["discontinuity"]["plane_cutouts"] = {}
    return data


@pytest.fixture
def f1_spec() -> CouponSpec:
    """Create a validated F1 CouponSpec."""
    return CouponSpec.model_validate(_minimal_f1_spec_data())


@pytest.fixture
def f1_spec_minimal() -> CouponSpec:
    """Create a validated F1 CouponSpec without optional features."""
    return CouponSpec.model_validate(_minimal_f1_spec_data_no_return_vias())


def _collect_f1_golden_specs() -> list[Path]:
    """Collect all F1 golden spec files."""
    return sorted(GOLDEN_SPECS_DIR.glob("f1_via_*.yaml"))


class TestF1SpecValidation:
    """Tests for F1 spec loading and validation."""

    def test_f1_spec_loads_successfully(self, f1_spec: CouponSpec) -> None:
        """F1 spec should load and validate successfully."""
        assert f1_spec.coupon_family == FAMILY_F1
        assert f1_spec.discontinuity is not None

    def test_f1_spec_has_required_fields(self, f1_spec: CouponSpec) -> None:
        """F1 spec should have all required fields."""
        assert f1_spec.schema_version == 1
        assert f1_spec.board is not None
        assert f1_spec.connectors is not None
        assert f1_spec.transmission_line is not None
        assert f1_spec.constraints is not None
        assert f1_spec.discontinuity is not None

    def test_f1_spec_discontinuity_type(self, f1_spec: CouponSpec) -> None:
        """F1 spec discontinuity should be VIA_TRANSITION."""
        assert f1_spec.discontinuity is not None
        assert f1_spec.discontinuity.type == "VIA_TRANSITION"

    def test_f1_spec_connectors_configured(self, f1_spec: CouponSpec) -> None:
        """F1 spec should have left and right connectors."""
        assert f1_spec.connectors.left is not None
        assert f1_spec.connectors.right is not None
        assert f1_spec.connectors.left.footprint == "Coupongen_Connectors:SMA_EndLaunch_Generic"
        assert f1_spec.connectors.right.footprint == "Coupongen_Connectors:SMA_EndLaunch_Generic"

    def test_f1_spec_has_signal_via(self, f1_spec: CouponSpec) -> None:
        """F1 spec should have signal via parameters."""
        assert f1_spec.discontinuity is not None
        sig_via = f1_spec.discontinuity.signal_via
        assert sig_via.drill_nm == 300000
        assert sig_via.diameter_nm == 650000
        assert sig_via.pad_diameter_nm == 900000

    def test_f1_spec_has_return_vias(self, f1_spec: CouponSpec) -> None:
        """F1 spec should have return vias configured."""
        assert f1_spec.discontinuity is not None
        ret_vias = f1_spec.discontinuity.return_vias
        assert ret_vias is not None
        assert ret_vias.pattern == "RING"
        assert ret_vias.count == 4
        assert ret_vias.radius_nm == 1700000

    def test_f1_spec_has_antipads(self, f1_spec: CouponSpec) -> None:
        """F1 spec should have antipads on internal layers."""
        assert f1_spec.discontinuity is not None
        antipads = f1_spec.discontinuity.antipads
        assert "In1.Cu" in antipads
        assert "In2.Cu" in antipads


class TestF1Resolver:
    """Tests for F1 spec resolution to ResolvedDesign."""

    def test_f1_resolves_to_design(self, f1_spec: CouponSpec) -> None:
        """F1 spec should resolve to a ResolvedDesign."""
        resolved = resolve(f1_spec)
        assert resolved is not None
        assert resolved.coupon_family == FAMILY_F1

    def test_f1_resolved_has_layout_plan(self, f1_spec: CouponSpec) -> None:
        """Resolved F1 design should have a LayoutPlan attached."""
        resolved = resolve(f1_spec)
        assert resolved.layout_plan is not None
        assert isinstance(resolved.layout_plan, LayoutPlan)

    def test_f1_layout_plan_has_discontinuity(self, f1_spec: CouponSpec) -> None:
        """F1 LayoutPlan should have a discontinuity (x_disc_nm is not None)."""
        resolved = resolve(f1_spec)
        layout = resolved.layout_plan
        assert layout is not None
        assert layout.x_disc_nm is not None
        assert layout.has_discontinuity

    def test_f1_layout_plan_has_two_segments(self, f1_spec: CouponSpec) -> None:
        """F1 LayoutPlan should have two segments (left and right)."""
        resolved = resolve(f1_spec)
        layout = resolved.layout_plan
        assert layout is not None
        assert len(layout.segments) == 2

        labels = {seg.label for seg in layout.segments}
        assert "left" in labels
        assert "right" in labels

    def test_f1_layout_plan_segments_meet_at_discontinuity(self, f1_spec: CouponSpec) -> None:
        """F1 segments should meet at the discontinuity center."""
        resolved = resolve(f1_spec)
        layout = resolved.layout_plan
        assert layout is not None
        assert layout.x_disc_nm is not None

        left_seg = layout.get_segment_by_label("left")
        right_seg = layout.get_segment_by_label("right")
        assert left_seg is not None
        assert right_seg is not None

        # Left segment ends at discontinuity
        assert left_seg.x_end_nm == layout.x_disc_nm
        # Right segment starts at discontinuity
        assert right_seg.x_start_nm == layout.x_disc_nm

    def test_f1_layout_plan_derived_right_length(self, f1_spec: CouponSpec) -> None:
        """F1 right length should be derived from continuity formula."""
        resolved = resolve(f1_spec)
        assert resolved.length_right_nm is not None

        # Verify continuity: left_x + left_length + right_length = right_x
        layout = resolved.layout_plan
        assert layout is not None
        left_pad_x = layout.left_port.signal_pad_x_nm
        right_pad_x = layout.right_port.signal_pad_x_nm
        left_length = f1_spec.transmission_line.length_left_nm

        # Derived right length = right_pad_x - (left_pad_x + left_length)
        derived_right = right_pad_x - (left_pad_x + left_length)
        assert resolved.length_right_nm == derived_right


class TestF1Builder:
    """Tests for F1 builder composition."""

    def test_f1_builder_creates_composition(self, f1_spec: CouponSpec) -> None:
        """F1 builder should create an F1CouponComposition."""
        resolved = resolve(f1_spec)
        composition = build_f1_coupon(f1_spec, resolved)
        assert isinstance(composition, F1CouponComposition)

    def test_f1_composition_has_all_features(self, f1_spec: CouponSpec) -> None:
        """F1 composition should have all features."""
        resolved = resolve(f1_spec)
        composition = build_f1_coupon(f1_spec, resolved)

        assert composition.board_outline is not None
        assert composition.left_port is not None
        assert composition.right_port is not None
        assert composition.transmission_line is not None
        assert composition.discontinuity is not None

    def test_f1_composition_has_signal_via(self, f1_spec: CouponSpec) -> None:
        """F1 composition should have signal via at discontinuity."""
        resolved = resolve(f1_spec)
        composition = build_f1_coupon(f1_spec, resolved)

        via = composition.signal_via
        assert via is not None
        assert via.position == composition.discontinuity_position

    def test_f1_composition_has_return_vias(self, f1_spec: CouponSpec) -> None:
        """F1 composition should have return vias around signal via."""
        resolved = resolve(f1_spec)
        composition = build_f1_coupon(f1_spec, resolved)

        return_vias = composition.return_vias
        assert len(return_vias) == 4  # From spec

    def test_f1_composition_has_antipads(self, f1_spec: CouponSpec) -> None:
        """F1 composition should have antipad polygons."""
        resolved = resolve(f1_spec)
        composition = build_f1_coupon(f1_spec, resolved)

        antipads = composition.all_antipads
        assert len(antipads) == 2  # In1.Cu and In2.Cu

    def test_f1_composition_has_cutouts(self, f1_spec: CouponSpec) -> None:
        """F1 composition should have plane cutout polygons."""
        resolved = resolve(f1_spec)
        composition = build_f1_coupon(f1_spec, resolved)

        cutouts = composition.all_cutouts
        assert len(cutouts) == 1  # In1.Cu slot

    def test_f1_composition_minimal_no_return_vias(self, f1_spec_minimal: CouponSpec) -> None:
        """F1 minimal composition should work without return vias."""
        resolved = resolve(f1_spec_minimal)
        composition = build_f1_coupon(f1_spec_minimal, resolved)

        assert composition.return_vias == ()
        assert composition.all_antipads == ()
        assert composition.all_cutouts == ()


class TestF1BoardWriter:
    """Tests for F1 board file generation."""

    def test_f1_board_writer_creates_valid_board(self, f1_spec: CouponSpec) -> None:
        """F1 BoardWriter should create a valid board S-expression."""
        resolved = resolve(f1_spec)
        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        assert isinstance(board, list)
        assert board[0] == "kicad_pcb"

    def test_f1_board_has_two_signal_track_segments(self, f1_spec: CouponSpec) -> None:
        """F1 board should have exactly two signal track segments (left and right).

        The board also contains ground ring segments for return via connectivity,
        but these are on net 2 (GND). Signal traces are on net 1 (SIG).
        """
        resolved = resolve(f1_spec)
        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        # Get all segments
        all_segments = [e for e in board if isinstance(e, list) and e[0] == "segment"]

        # Filter for signal segments (net 1)
        def get_net(seg: list) -> int:
            for elem in seg:
                if isinstance(elem, list) and elem[0] == "net":
                    return elem[1]
            return -1

        signal_segments = [s for s in all_segments if get_net(s) == 1]
        assert len(signal_segments) == 2, f"Expected 2 signal segments, got {len(signal_segments)}"

        # Ground ring segments should also be present (4 vias × 2 layers = 8 segments)
        gnd_segments = [s for s in all_segments if get_net(s) == 2]
        assert len(gnd_segments) == 8, f"Expected 8 ground ring segments, got {len(gnd_segments)}"

    def test_f1_board_has_signal_via(self, f1_spec: CouponSpec) -> None:
        """F1 board should have a signal via at the discontinuity."""
        resolved = resolve(f1_spec)
        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        vias = [e for e in board if isinstance(e, list) and e[0] == "via"]
        # 1 signal via + 4 return vias = 5 total
        assert len(vias) >= 1

    def test_f1_board_has_return_vias(self, f1_spec: CouponSpec) -> None:
        """F1 board should have return vias around the signal via."""
        resolved = resolve(f1_spec)
        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        vias = [e for e in board if isinstance(e, list) and e[0] == "via"]
        # 1 signal via + 4 return vias = 5 total
        assert len(vias) == 5

    def test_f1_board_has_two_footprints(self, f1_spec: CouponSpec) -> None:
        """F1 board should have exactly two footprints (left and right connectors)."""
        resolved = resolve(f1_spec)
        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        footprints = [e for e in board if isinstance(e, list) and e[0] == "footprint"]
        assert len(footprints) == 2

    def test_f1_board_has_board_outline(self, f1_spec: CouponSpec) -> None:
        """F1 board should have a board outline on Edge.Cuts."""
        resolved = resolve(f1_spec)
        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        gr_rects = [e for e in board if isinstance(e, list) and e[0] == "gr_rect"]
        assert len(gr_rects) == 1

    def test_f1_board_has_antipads_as_zones(self, f1_spec: CouponSpec) -> None:
        """F1 board should have antipads as keepout zones."""
        resolved = resolve(f1_spec)
        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        zones = [e for e in board if isinstance(e, list) and e[0] == "zone"]
        # 2 antipads + 1 cutout = 3 zones
        assert len(zones) >= 2

    def test_f1_board_minimal_no_return_vias(self, f1_spec_minimal: CouponSpec) -> None:
        """F1 minimal board should have only signal via."""
        resolved = resolve(f1_spec_minimal)
        writer = BoardWriter(f1_spec_minimal, resolved)
        board = writer.build_board()

        vias = [e for e in board if isinstance(e, list) and e[0] == "via"]
        assert len(vias) == 1  # Only signal via


class TestF1BoardDeterminism:
    """Tests for F1 board generation determinism."""

    def test_f1_board_text_deterministic(self, f1_spec: CouponSpec) -> None:
        """Same F1 spec should produce identical board text."""
        resolved = resolve(f1_spec)
        text1 = build_board_text(f1_spec, resolved)
        text2 = build_board_text(f1_spec, resolved)
        assert text1 == text2

    def test_f1_write_board_deterministic(self, f1_spec: CouponSpec, tmp_path: Path) -> None:
        """Same F1 spec should produce identical board files."""
        resolved = resolve(f1_spec)
        board_path1 = write_board(f1_spec, resolved, tmp_path / "run1")
        board_path2 = write_board(f1_spec, resolved, tmp_path / "run2")

        content1 = board_path1.read_text(encoding="utf-8")
        content2 = board_path2.read_text(encoding="utf-8")
        assert content1 == content2

    def test_f1_board_parseable(self, f1_spec: CouponSpec, tmp_path: Path) -> None:
        """Generated F1 board should be parseable S-expression."""
        resolved = resolve(f1_spec)
        board_path = write_board(f1_spec, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")
        parsed = parse(content)
        assert parsed[0] == "kicad_pcb"


class TestF1ViaGeometry:
    """Tests for F1 via geometry correctness."""

    def test_f1_signal_via_at_discontinuity(self, f1_spec: CouponSpec) -> None:
        """F1 signal via should be at the discontinuity center."""
        resolved = resolve(f1_spec)
        layout = resolved.layout_plan
        assert layout is not None
        assert layout.x_disc_nm is not None

        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        vias = [e for e in board if isinstance(e, list) and e[0] == "via"]
        assert len(vias) >= 1

        # First via should be signal via at discontinuity
        signal_via = vias[0]
        at_elem = [e for e in signal_via if isinstance(e, list) and e[0] == "at"][0]
        via_x_mm = float(at_elem[1])
        via_x_nm = int(via_x_mm * 1_000_000)

        # Should be at discontinuity position (within tolerance)
        assert abs(via_x_nm - layout.x_disc_nm) < 1000  # 1um tolerance

    def test_f1_return_vias_around_signal_via(self, f1_spec: CouponSpec) -> None:
        """F1 return vias should be at correct radius from signal via."""
        resolved = resolve(f1_spec)
        layout = resolved.layout_plan
        assert layout is not None
        assert layout.x_disc_nm is not None

        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        vias = [e for e in board if isinstance(e, list) and e[0] == "via"]
        assert len(vias) == 5  # 1 signal + 4 return

        # Get expected radius from spec
        expected_radius_nm = f1_spec.discontinuity.return_vias.radius_nm

        # Check return vias (skip first which is signal via)
        center_x_nm = layout.x_disc_nm
        center_y_nm = layout.y_centerline_nm

        for return_via in vias[1:]:
            at_elem = [e for e in return_via if isinstance(e, list) and e[0] == "at"][0]
            via_x_mm = float(at_elem[1])
            via_y_mm = float(at_elem[2])
            via_x_nm = int(via_x_mm * 1_000_000)
            via_y_nm = int(via_y_mm * 1_000_000)

            # Calculate distance from center
            dx = via_x_nm - center_x_nm
            dy = via_y_nm - center_y_nm
            distance = (dx**2 + dy**2) ** 0.5

            # Should be at expected radius (within tolerance)
            assert abs(distance - expected_radius_nm) < 1000  # 1um tolerance


def _get_segment_net(seg: list) -> int:
    """Extract net ID from a segment S-expression."""
    for elem in seg:
        if isinstance(elem, list) and elem[0] == "net":
            return elem[1]
    return -1


def _get_segment_layer(seg: list) -> str:
    """Extract layer name from a segment S-expression."""
    for elem in seg:
        if isinstance(elem, list) and elem[0] == "layer":
            return elem[1]
    return ""


def _get_signal_segments(board: list) -> list:
    """Extract signal trace segments (net 1) from a board S-expression."""
    all_segments = [e for e in board if isinstance(e, list) and e[0] == "segment"]
    return [s for s in all_segments if _get_segment_net(s) == 1]


class TestF1TrackGeometry:
    """Tests for F1 track geometry correctness."""

    def test_f1_signal_tracks_on_correct_layers(self, f1_spec: CouponSpec) -> None:
        """F1 signal tracks should be on correct layers for via transition.

        For F1 via transitions:
        - Left signal segment is on entry layer (F.Cu)
        - Right signal segment is on exit layer (B.Cu)

        This ensures the signal via connects traces on both layers, eliminating
        via_dangling DRC warnings.
        """
        resolved = resolve(f1_spec)
        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        signal_segments = _get_signal_segments(board)
        assert len(signal_segments) == 2, "F1 should have exactly 2 signal segments"

        # Collect layers from signal segments
        layers = {_get_segment_layer(s) for s in signal_segments}

        # F1 via transition: left on F.Cu, right on B.Cu
        # Both layers must be present for proper via connectivity
        assert "F.Cu" in layers, "Missing F.Cu signal segment (left/entry)"
        assert "B.Cu" in layers, "Missing B.Cu signal segment (right/exit)"

    def test_f1_signal_tracks_have_correct_width(self, f1_spec: CouponSpec) -> None:
        """F1 signal tracks should have the correct width from spec."""
        resolved = resolve(f1_spec)
        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        signal_segments = _get_signal_segments(board)
        assert len(signal_segments) == 2

        for segment in signal_segments:
            width_elem = [e for e in segment if isinstance(e, list) and e[0] == "width"][0]
            # Width in mm = 300000 nm / 1000000 = 0.3 mm
            assert width_elem[1] == "0.3", f"Signal segment width should be 0.3mm"

    def test_f1_signal_tracks_on_signal_net(self, f1_spec: CouponSpec) -> None:
        """F1 signal tracks should be on the SIG net (net 1)."""
        resolved = resolve(f1_spec)
        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        signal_segments = _get_signal_segments(board)
        assert len(signal_segments) == 2

        # Verify all signal segments are on net 1 (by definition of filter)
        for segment in signal_segments:
            net_elem = [e for e in segment if isinstance(e, list) and e[0] == "net"][0]
            assert net_elem[1] == 1

    def test_f1_signal_tracks_meet_at_discontinuity(self, f1_spec: CouponSpec) -> None:
        """F1 signal tracks should meet at the discontinuity center."""
        resolved = resolve(f1_spec)
        layout = resolved.layout_plan
        assert layout is not None
        assert layout.x_disc_nm is not None

        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        signal_segments = _get_signal_segments(board)
        assert len(signal_segments) == 2, "F1 should have exactly 2 signal segments"

        # Get endpoints from signal segments only
        endpoints = []
        for segment in signal_segments:
            start_elem = [e for e in segment if isinstance(e, list) and e[0] == "start"][0]
            end_elem = [e for e in segment if isinstance(e, list) and e[0] == "end"][0]
            endpoints.append(float(start_elem[1]))
            endpoints.append(float(end_elem[1]))

        # Convert discontinuity x to mm
        disc_x_mm = layout.x_disc_nm / 1_000_000

        # At least one endpoint should be at discontinuity (from each segment)
        at_disc = [x for x in endpoints if abs(x - disc_x_mm) < 0.001]
        assert len(at_disc) == 2, "Both signal segments should meet at discontinuity"


class TestF1GoldenSpecs:
    """Tests for F1 golden specification files."""

    def test_f1_golden_specs_exist(self) -> None:
        """At least 10 F1 golden specs should exist."""
        f1_specs = _collect_f1_golden_specs()
        assert len(f1_specs) >= 10, f"Expected >= 10 F1 specs, found {len(f1_specs)}"

    @pytest.mark.parametrize(
        "spec_path",
        _collect_f1_golden_specs(),  # Test ALL F1 golden specs (no sampling)
        ids=lambda p: p.stem,
    )
    def test_f1_golden_spec_loads(self, spec_path: Path) -> None:
        """Each F1 golden spec should load successfully."""
        with open(spec_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        spec = CouponSpec.model_validate(data)
        assert spec.coupon_family == FAMILY_F1
        assert spec.discontinuity is not None

    @pytest.mark.parametrize(
        "spec_path",
        _collect_f1_golden_specs(),  # Test ALL F1 golden specs (no sampling)
        ids=lambda p: p.stem,
    )
    def test_f1_golden_spec_resolves(self, spec_path: Path) -> None:
        """Each F1 golden spec should resolve to a valid design."""
        with open(spec_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        spec = CouponSpec.model_validate(data)
        resolved = resolve(spec)

        assert resolved is not None
        assert resolved.layout_plan is not None
        assert resolved.layout_plan.has_discontinuity

    @pytest.mark.parametrize(
        "spec_path",
        _collect_f1_golden_specs(),  # Test ALL F1 golden specs (no sampling)
        ids=lambda p: p.stem,
    )
    def test_f1_golden_spec_generates_board(self, spec_path: Path, tmp_path: Path) -> None:
        """Each F1 golden spec should generate a valid board file."""
        with open(spec_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        spec = CouponSpec.model_validate(data)
        resolved = resolve(spec)
        board_path = write_board(spec, resolved, tmp_path)

        assert board_path.exists()
        content = board_path.read_text(encoding="utf-8")
        assert content.strip().startswith("(kicad_pcb")
        assert content.count("(") == content.count(")")


class TestF1EndToEndPipeline:
    """End-to-end integration tests for F1 pipeline."""

    def test_f1_full_pipeline(self, f1_spec: CouponSpec, tmp_path: Path) -> None:
        """Test complete F1 pipeline: spec -> resolve -> board."""
        # Step 1: Resolve spec
        resolved = resolve(f1_spec)
        assert resolved is not None

        # Step 2: Verify LayoutPlan
        layout = resolved.layout_plan
        assert layout is not None
        assert len(layout.segments) == 2
        assert layout.has_discontinuity

        # Step 3: Build F1 composition
        composition = build_f1_coupon(f1_spec, resolved)
        assert isinstance(composition, F1CouponComposition)

        # Step 4: Generate board
        board_path = write_board(f1_spec, resolved, tmp_path)
        assert board_path.exists()

        # Step 5: Verify board contents
        content = board_path.read_text(encoding="utf-8")
        parsed = parse(content)

        # Should have expected elements
        all_segments = [e for e in parsed if isinstance(e, list) and e[0] == "segment"]
        footprints = [e for e in parsed if isinstance(e, list) and e[0] == "footprint"]
        vias = [e for e in parsed if isinstance(e, list) and e[0] == "via"]
        zones = [e for e in parsed if isinstance(e, list) and e[0] == "zone"]

        # F1 boards have:
        # - 2 signal trace segments (net 1: left on F.Cu, right on B.Cu)
        # - 8 ground ring segments (net 2: 4 vias × 2 layers for return via connectivity)
        signal_segments = _get_signal_segments(parsed)
        assert len(signal_segments) == 2, "F1 should have exactly 2 signal trace segments"
        assert len(all_segments) == 10, "F1 should have 10 total segments (2 signal + 8 ground ring)"
        assert len(footprints) == 2, "F1 should have exactly 2 footprints"
        assert len(vias) == 5, "F1 should have 5 vias (1 signal + 4 return)"
        assert len(zones) >= 2, "F1 should have antipads/cutouts as zones"

    def test_f1_api_pipeline(self, tmp_path: Path) -> None:
        """Test F1 pipeline using the public API functions."""
        # Create spec file
        spec_data = _minimal_f1_spec_data()
        spec_file = tmp_path / "f1_test.json"
        spec_file.write_text(json.dumps(spec_data), encoding="utf-8")

        # Load and validate spec
        spec = load_spec(spec_file)
        assert spec.coupon_family == FAMILY_F1

        # Resolve spec
        resolved = resolve_spec(spec)
        assert resolved is not None

        # Generate KiCad project
        project = generate_kicad(resolved, spec, tmp_path / "output")
        assert project.board_path.exists()

        # Verify board file
        content = project.board_path.read_text(encoding="utf-8")
        assert "(kicad_pcb" in content
        assert "(via" in content  # Should have vias

    def test_f1_geometry_matches_layout_plan(self, f1_spec: CouponSpec) -> None:
        """Board geometry should exactly match LayoutPlan."""
        resolved = resolve(f1_spec)
        layout = resolved.layout_plan
        assert layout is not None

        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        # Get board outline from S-expr
        gr_rects = [e for e in board if isinstance(e, list) and e[0] == "gr_rect"]
        assert len(gr_rects) == 1
        rect = gr_rects[0]

        # Get start/end coords (in mm)
        start_elem = [e for e in rect if isinstance(e, list) and e[0] == "start"][0]
        end_elem = [e for e in rect if isinstance(e, list) and e[0] == "end"][0]

        # Convert back to nm for comparison
        start_x_nm = int(float(start_elem[1]) * 1_000_000)
        end_x_nm = int(float(end_elem[1]) * 1_000_000)

        # Should match LayoutPlan board dimensions
        assert start_x_nm == layout.x_board_left_edge_nm
        assert end_x_nm == layout.x_board_right_edge_nm

        # Check signal via at discontinuity
        vias = [e for e in board if isinstance(e, list) and e[0] == "via"]
        signal_via = vias[0]
        at_elem = [e for e in signal_via if isinstance(e, list) and e[0] == "at"][0]
        via_x_nm = int(float(at_elem[1]) * 1_000_000)

        assert layout.x_disc_nm is not None
        assert abs(via_x_nm - layout.x_disc_nm) < 1000  # 1um tolerance

    def test_f1_connectivity_invariant(self, f1_spec: CouponSpec) -> None:
        """F1 should maintain connectivity invariant: segments meet at via."""
        resolved = resolve(f1_spec)
        layout = resolved.layout_plan
        assert layout is not None

        # Verify connectivity invariant
        errors = layout.validate_connectivity()
        assert errors == [], f"Connectivity errors: {errors}"

        # Verify left segment end == discontinuity == right segment start
        left_seg = layout.get_segment_by_label("left")
        right_seg = layout.get_segment_by_label("right")
        assert left_seg is not None
        assert right_seg is not None
        assert layout.x_disc_nm is not None

        assert left_seg.x_end_nm == layout.x_disc_nm
        assert right_seg.x_start_nm == layout.x_disc_nm
        assert left_seg.x_end_nm == right_seg.x_start_nm


class TestF1RequirementCoverage:
    """Tests to verify REQ-M1-007 coverage for F1 coupons."""

    def test_req_m1_007_end_launch_connectors(self, f1_spec: CouponSpec) -> None:
        """REQ-M1-007: F1 has end-launch connectors at both ends."""
        resolved = resolve(f1_spec)
        composition = build_f1_coupon(f1_spec, resolved)

        assert composition.left_port is not None
        assert composition.right_port is not None
        assert composition.left_port.side == "left"
        assert composition.right_port.side == "right"

    def test_req_m1_007_cpwg_on_both_sides(self, f1_spec: CouponSpec) -> None:
        """REQ-M1-007: F1 has CPWG on both sides of via transition."""
        resolved = resolve(f1_spec)
        composition = build_f1_coupon(f1_spec, resolved)

        tl = composition.transmission_line
        assert tl.length_left_nm > 0
        assert tl.length_right_nm > 0

        # Verify track segments
        left_track, right_track = tl.to_track_segments()
        assert left_track is not None
        assert right_track is not None

    def test_req_m1_007_via_transition(self, f1_spec: CouponSpec) -> None:
        """REQ-M1-007: F1 has via transition (top to inner or top to bottom)."""
        resolved = resolve(f1_spec)
        composition = build_f1_coupon(f1_spec, resolved)

        via = composition.signal_via
        assert via is not None
        assert via.layers == ("F.Cu", "B.Cu")  # Top to bottom

    def test_req_m1_007_antipads_cutouts(self, f1_spec: CouponSpec) -> None:
        """REQ-M1-007: F1 includes antipads/cutouts."""
        resolved = resolve(f1_spec)
        composition = build_f1_coupon(f1_spec, resolved)

        antipads = composition.all_antipads
        assert len(antipads) > 0

        cutouts = composition.all_cutouts
        assert len(cutouts) > 0

    def test_req_m1_007_return_vias(self, f1_spec: CouponSpec) -> None:
        """REQ-M1-007: F1 includes return vias."""
        resolved = resolve(f1_spec)
        composition = build_f1_coupon(f1_spec, resolved)

        return_vias = composition.return_vias
        assert len(return_vias) > 0
