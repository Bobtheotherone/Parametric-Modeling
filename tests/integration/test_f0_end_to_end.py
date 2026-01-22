# SPDX-License-Identifier: MIT
"""End-to-end integration tests for Family F0 (Calibration Thru Line).

This module tests the complete F0 coupon generation pipeline:
    spec -> resolve -> LayoutPlan -> BoardWriter -> .kicad_pcb

F0 coupons are calibration structures:
    end-launch connector -> CPWG straight line -> end-launch connector

These tests verify:
1. F0 specs load and validate correctly
2. LayoutPlan computes correct geometry (single through-segment)
3. BoardWriter generates valid KiCad board file
4. Generated board is deterministic (same spec -> same output)
5. All golden specs produce valid boards

Satisfies REQ-M1-006.
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
from formula_foundry.coupongen.builders.f0_builder import (
    F0CouponComposition,
    build_f0_coupon,
)
from formula_foundry.coupongen.families import FAMILY_F0
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


def _minimal_f0_spec_data() -> dict[str, Any]:
    """Create minimal valid F0 spec data."""
    return {
        "schema_version": 1,
        "coupon_family": FAMILY_F0,
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
            "text": {"coupon_id": "F0-TEST-001", "include_manifest_hash": True},
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
        },
        "discontinuity": None,
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


@pytest.fixture
def f0_spec() -> CouponSpec:
    """Create a validated F0 CouponSpec."""
    return CouponSpec.model_validate(_minimal_f0_spec_data())


def _collect_f0_golden_specs() -> list[Path]:
    """Collect all F0 golden spec files."""
    return sorted(GOLDEN_SPECS_DIR.glob("f0_cal_*.yaml"))


class TestF0SpecValidation:
    """Tests for F0 spec loading and validation."""

    def test_f0_spec_loads_successfully(self, f0_spec: CouponSpec) -> None:
        """F0 spec should load and validate successfully."""
        assert f0_spec.coupon_family == FAMILY_F0
        assert f0_spec.discontinuity is None

    def test_f0_spec_has_required_fields(self, f0_spec: CouponSpec) -> None:
        """F0 spec should have all required fields."""
        assert f0_spec.schema_version == 1
        assert f0_spec.board is not None
        assert f0_spec.connectors is not None
        assert f0_spec.transmission_line is not None
        assert f0_spec.constraints is not None

    def test_f0_spec_connectors_configured(self, f0_spec: CouponSpec) -> None:
        """F0 spec should have left and right connectors."""
        assert f0_spec.connectors.left is not None
        assert f0_spec.connectors.right is not None
        assert f0_spec.connectors.left.footprint == "Coupongen_Connectors:SMA_EndLaunch_Generic"
        assert f0_spec.connectors.right.footprint == "Coupongen_Connectors:SMA_EndLaunch_Generic"

    def test_f0_spec_transmission_line_has_both_lengths(self, f0_spec: CouponSpec) -> None:
        """F0 spec should have both left and right lengths specified."""
        tl = f0_spec.transmission_line
        assert tl.length_left_nm is not None
        assert tl.length_right_nm is not None


class TestF0Resolver:
    """Tests for F0 spec resolution to ResolvedDesign."""

    def test_f0_resolves_to_design(self, f0_spec: CouponSpec) -> None:
        """F0 spec should resolve to a ResolvedDesign."""
        resolved = resolve(f0_spec)
        assert resolved is not None
        assert resolved.coupon_family == FAMILY_F0

    def test_f0_resolved_has_layout_plan(self, f0_spec: CouponSpec) -> None:
        """Resolved F0 design should have a LayoutPlan attached."""
        resolved = resolve(f0_spec)
        assert resolved.layout_plan is not None
        assert isinstance(resolved.layout_plan, LayoutPlan)

    def test_f0_layout_plan_has_no_discontinuity(self, f0_spec: CouponSpec) -> None:
        """F0 LayoutPlan should have no discontinuity (x_disc_nm is None)."""
        resolved = resolve(f0_spec)
        layout = resolved.layout_plan
        assert layout is not None
        assert layout.x_disc_nm is None
        assert not layout.has_discontinuity

    def test_f0_layout_plan_has_single_through_segment(self, f0_spec: CouponSpec) -> None:
        """F0 LayoutPlan should have a single 'through' segment."""
        resolved = resolve(f0_spec)
        layout = resolved.layout_plan
        assert layout is not None
        assert len(layout.segments) == 1
        assert layout.segments[0].label == "through"

    def test_f0_layout_plan_segment_spans_connectors(self, f0_spec: CouponSpec) -> None:
        """F0 through segment should span from left to right signal pads."""
        resolved = resolve(f0_spec)
        layout = resolved.layout_plan
        assert layout is not None

        through_seg = layout.segments[0]
        assert through_seg.x_start_nm == layout.left_port.signal_pad_x_nm
        assert through_seg.x_end_nm == layout.right_port.signal_pad_x_nm

    def test_f0_layout_plan_total_trace_length(self, f0_spec: CouponSpec) -> None:
        """F0 total trace length should equal through segment length."""
        resolved = resolve(f0_spec)
        layout = resolved.layout_plan
        assert layout is not None

        through_seg = layout.segments[0]
        assert layout.total_trace_length_nm == through_seg.length_nm


class TestF0Builder:
    """Tests for F0 builder composition."""

    def test_f0_builder_creates_composition(self, f0_spec: CouponSpec) -> None:
        """F0 builder should create an F0CouponComposition."""
        resolved = resolve(f0_spec)
        composition = build_f0_coupon(f0_spec, resolved)
        assert isinstance(composition, F0CouponComposition)

    def test_f0_composition_has_all_features(self, f0_spec: CouponSpec) -> None:
        """F0 composition should have board outline, ports, and transmission line."""
        resolved = resolve(f0_spec)
        composition = build_f0_coupon(f0_spec, resolved)

        assert composition.board_outline is not None
        assert composition.left_port is not None
        assert composition.right_port is not None
        assert composition.transmission_line is not None

    def test_f0_composition_total_trace_length(self, f0_spec: CouponSpec) -> None:
        """F0 composition total_trace_length should equal spec values."""
        resolved = resolve(f0_spec)
        composition = build_f0_coupon(f0_spec, resolved)

        expected = f0_spec.transmission_line.length_left_nm + f0_spec.transmission_line.length_right_nm
        assert composition.total_trace_length_nm == expected


class TestF0BoardWriter:
    """Tests for F0 board file generation."""

    def test_f0_board_writer_creates_valid_board(self, f0_spec: CouponSpec) -> None:
        """F0 BoardWriter should create a valid board S-expression."""
        resolved = resolve(f0_spec)
        writer = BoardWriter(f0_spec, resolved)
        board = writer.build_board()

        assert isinstance(board, list)
        assert board[0] == "kicad_pcb"

    def test_f0_board_has_single_track_segment(self, f0_spec: CouponSpec) -> None:
        """F0 board should have exactly one track segment (through-line)."""
        resolved = resolve(f0_spec)
        writer = BoardWriter(f0_spec, resolved)
        board = writer.build_board()

        segments = [e for e in board if isinstance(e, list) and e[0] == "segment"]
        assert len(segments) == 1

    def test_f0_board_has_two_footprints(self, f0_spec: CouponSpec) -> None:
        """F0 board should have exactly two footprints (left and right connectors)."""
        resolved = resolve(f0_spec)
        writer = BoardWriter(f0_spec, resolved)
        board = writer.build_board()

        footprints = [e for e in board if isinstance(e, list) and e[0] == "footprint"]
        assert len(footprints) == 2

    def test_f0_board_has_no_vias(self, f0_spec: CouponSpec) -> None:
        """F0 board should have no vias (no discontinuity)."""
        resolved = resolve(f0_spec)
        writer = BoardWriter(f0_spec, resolved)
        board = writer.build_board()

        vias = [e for e in board if isinstance(e, list) and e[0] == "via"]
        assert len(vias) == 0

    def test_f0_board_has_board_outline(self, f0_spec: CouponSpec) -> None:
        """F0 board should have a board outline on Edge.Cuts."""
        resolved = resolve(f0_spec)
        writer = BoardWriter(f0_spec, resolved)
        board = writer.build_board()

        gr_rects = [e for e in board if isinstance(e, list) and e[0] == "gr_rect"]
        assert len(gr_rects) == 1


class TestF0BoardDeterminism:
    """Tests for F0 board generation determinism."""

    def test_f0_board_text_deterministic(self, f0_spec: CouponSpec) -> None:
        """Same F0 spec should produce identical board text."""
        resolved = resolve(f0_spec)
        text1 = build_board_text(f0_spec, resolved)
        text2 = build_board_text(f0_spec, resolved)
        assert text1 == text2

    def test_f0_write_board_deterministic(self, f0_spec: CouponSpec, tmp_path: Path) -> None:
        """Same F0 spec should produce identical board files."""
        resolved = resolve(f0_spec)
        board_path1 = write_board(f0_spec, resolved, tmp_path / "run1")
        board_path2 = write_board(f0_spec, resolved, tmp_path / "run2")

        content1 = board_path1.read_text(encoding="utf-8")
        content2 = board_path2.read_text(encoding="utf-8")
        assert content1 == content2

    def test_f0_board_parseable(self, f0_spec: CouponSpec, tmp_path: Path) -> None:
        """Generated F0 board should be parseable S-expression."""
        resolved = resolve(f0_spec)
        board_path = write_board(f0_spec, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")
        parsed = parse(content)
        assert parsed[0] == "kicad_pcb"


class TestF0TrackGeometry:
    """Tests for F0 track geometry correctness."""

    def test_f0_track_on_correct_layer(self, f0_spec: CouponSpec) -> None:
        """F0 track should be on the specified layer."""
        resolved = resolve(f0_spec)
        writer = BoardWriter(f0_spec, resolved)
        board = writer.build_board()

        segments = [e for e in board if isinstance(e, list) and e[0] == "segment"]
        assert len(segments) == 1

        segment = segments[0]
        layer_elem = [e for e in segment if isinstance(e, list) and e[0] == "layer"][0]
        assert layer_elem[1] == "F.Cu"

    def test_f0_track_has_correct_width(self, f0_spec: CouponSpec) -> None:
        """F0 track should have the correct width from spec."""
        resolved = resolve(f0_spec)
        writer = BoardWriter(f0_spec, resolved)
        board = writer.build_board()

        segments = [e for e in board if isinstance(e, list) and e[0] == "segment"]
        segment = segments[0]

        width_elem = [e for e in segment if isinstance(e, list) and e[0] == "width"][0]
        # Width in mm = 300000 nm / 1000000 = 0.3 mm
        assert width_elem[1] == "0.3"

    def test_f0_track_on_signal_net(self, f0_spec: CouponSpec) -> None:
        """F0 track should be on the SIG net (net 1)."""
        resolved = resolve(f0_spec)
        writer = BoardWriter(f0_spec, resolved)
        board = writer.build_board()

        segments = [e for e in board if isinstance(e, list) and e[0] == "segment"]
        segment = segments[0]

        net_elem = [e for e in segment if isinstance(e, list) and e[0] == "net"][0]
        assert net_elem[1] == 1


class TestF0GoldenSpecs:
    """Tests for F0 golden specification files."""

    def test_f0_golden_specs_exist(self) -> None:
        """At least 10 F0 golden specs should exist."""
        f0_specs = _collect_f0_golden_specs()
        assert len(f0_specs) >= 10, f"Expected >= 10 F0 specs, found {len(f0_specs)}"

    @pytest.mark.parametrize(
        "spec_path",
        _collect_f0_golden_specs()[:5],  # Test first 5 for speed
        ids=lambda p: p.stem,
    )
    def test_f0_golden_spec_loads(self, spec_path: Path) -> None:
        """Each F0 golden spec should load successfully."""
        with open(spec_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        spec = CouponSpec.model_validate(data)
        assert spec.coupon_family == FAMILY_F0
        assert spec.discontinuity is None

    @pytest.mark.parametrize(
        "spec_path",
        _collect_f0_golden_specs()[:5],
        ids=lambda p: p.stem,
    )
    def test_f0_golden_spec_resolves(self, spec_path: Path) -> None:
        """Each F0 golden spec should resolve to a valid design."""
        with open(spec_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        spec = CouponSpec.model_validate(data)
        resolved = resolve(spec)

        assert resolved is not None
        assert resolved.layout_plan is not None
        assert not resolved.layout_plan.has_discontinuity

    @pytest.mark.parametrize(
        "spec_path",
        _collect_f0_golden_specs()[:5],
        ids=lambda p: p.stem,
    )
    def test_f0_golden_spec_generates_board(self, spec_path: Path, tmp_path: Path) -> None:
        """Each F0 golden spec should generate a valid board file."""
        with open(spec_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        spec = CouponSpec.model_validate(data)
        resolved = resolve(spec)
        board_path = write_board(spec, resolved, tmp_path)

        assert board_path.exists()
        content = board_path.read_text(encoding="utf-8")
        assert content.strip().startswith("(kicad_pcb")
        assert content.count("(") == content.count(")")


class TestF0EndToEndPipeline:
    """End-to-end integration tests for F0 pipeline."""

    def test_f0_full_pipeline(self, f0_spec: CouponSpec, tmp_path: Path) -> None:
        """Test complete F0 pipeline: spec -> resolve -> board."""
        # Step 1: Resolve spec
        resolved = resolve(f0_spec)
        assert resolved is not None

        # Step 2: Verify LayoutPlan
        layout = resolved.layout_plan
        assert layout is not None
        assert len(layout.segments) == 1
        assert layout.segments[0].label == "through"

        # Step 3: Build F0 composition
        composition = build_f0_coupon(f0_spec, resolved)
        assert isinstance(composition, F0CouponComposition)

        # Step 4: Generate board
        board_path = write_board(f0_spec, resolved, tmp_path)
        assert board_path.exists()

        # Step 5: Verify board contents
        content = board_path.read_text(encoding="utf-8")
        parsed = parse(content)

        # Should have expected elements
        segments = [e for e in parsed if isinstance(e, list) and e[0] == "segment"]
        footprints = [e for e in parsed if isinstance(e, list) and e[0] == "footprint"]
        vias = [e for e in parsed if isinstance(e, list) and e[0] == "via"]

        assert len(segments) == 1, "F0 should have exactly 1 track segment"
        assert len(footprints) == 2, "F0 should have exactly 2 footprints"
        assert len(vias) == 0, "F0 should have no vias"

    def test_f0_api_pipeline(self, tmp_path: Path) -> None:
        """Test F0 pipeline using the public API functions."""
        # Create spec file
        spec_data = _minimal_f0_spec_data()
        spec_file = tmp_path / "f0_test.json"
        spec_file.write_text(json.dumps(spec_data), encoding="utf-8")

        # Load and validate spec
        spec = load_spec(spec_file)
        assert spec.coupon_family == FAMILY_F0

        # Resolve spec
        resolved = resolve_spec(spec)
        assert resolved is not None

        # Generate KiCad project
        project = generate_kicad(resolved, spec, tmp_path / "output")
        assert project.board_path.exists()

        # Verify board file
        content = project.board_path.read_text(encoding="utf-8")
        assert "(kicad_pcb" in content

    def test_f0_geometry_matches_layout_plan(self, f0_spec: CouponSpec) -> None:
        """Board geometry should exactly match LayoutPlan."""
        resolved = resolve(f0_spec)
        layout = resolved.layout_plan
        assert layout is not None

        writer = BoardWriter(f0_spec, resolved)
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
        start_y_nm = int(float(start_elem[2]) * 1_000_000)
        end_x_nm = int(float(end_elem[1]) * 1_000_000)
        end_y_nm = int(float(end_elem[2]) * 1_000_000)

        # Should match LayoutPlan board dimensions
        assert start_x_nm == layout.x_board_left_edge_nm
        assert end_x_nm == layout.x_board_right_edge_nm

        # Get track segment
        segments = [e for e in board if isinstance(e, list) and e[0] == "segment"]
        assert len(segments) == 1
        segment = segments[0]

        # Get start/end of track
        seg_start = [e for e in segment if isinstance(e, list) and e[0] == "start"][0]
        seg_end = [e for e in segment if isinstance(e, list) and e[0] == "end"][0]

        track_start_x_nm = int(float(seg_start[1]) * 1_000_000)
        track_end_x_nm = int(float(seg_end[1]) * 1_000_000)

        # Should match LayoutPlan segment
        through_seg = layout.segments[0]
        assert abs(track_start_x_nm - through_seg.x_start_nm) < 1000  # 1um tolerance
        assert abs(track_end_x_nm - through_seg.x_end_nm) < 1000
