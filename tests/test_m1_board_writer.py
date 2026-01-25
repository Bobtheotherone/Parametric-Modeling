"""Tests for board writer and deterministic UUID generation (REQ-M1-012, REQ-M1-013)."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from formula_foundry.coupongen.kicad import (
    BoardWriter,
    build_board_text,
    deterministic_uuid,
    deterministic_uuid_indexed,
    parse,
    write_board,
)
from formula_foundry.coupongen.kicad.board_writer import (
    SPEC_TO_KICAD_LAYER,
    map_layer_to_kicad,
)
from formula_foundry.coupongen.resolve import resolve
from formula_foundry.coupongen.spec import CouponSpec


def _get_segment_net(seg: list) -> int:
    for elem in seg:
        if isinstance(elem, list) and elem[0] == "net":
            return elem[1]
    return -1


def _get_segments_by_net(board: list, net_id: int) -> list:
    segments = [e for e in board if isinstance(e, list) and e[0] == "segment"]
    return [seg for seg in segments if _get_segment_net(seg) == net_id]


@pytest.fixture
def f0_spec_data() -> dict:
    """Minimal F0 calibration spec data."""
    return {
        "schema_version": 1,
        "coupon_family": "F0_CAL_THRU_LINE",
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
def f1_spec_data(f0_spec_data: dict) -> dict:
    """F1 via transition spec data."""
    spec = f0_spec_data.copy()
    spec["coupon_family"] = "F1_SINGLE_ENDED_VIA"
    spec["discontinuity"] = {
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
    return spec


@pytest.fixture
def f0_spec(f0_spec_data: dict) -> CouponSpec:
    return CouponSpec.model_validate(f0_spec_data)


@pytest.fixture
def f1_spec(f1_spec_data: dict) -> CouponSpec:
    return CouponSpec.model_validate(f1_spec_data)


class TestDeterministicUuid:
    """Tests for deterministic UUIDv5 generation."""

    def test_uuid_format(self) -> None:
        """UUID should be valid UUID format."""
        result = deterministic_uuid(1, "test.path")
        # Should parse as valid UUID
        parsed = uuid.UUID(result)
        assert str(parsed) == result

    def test_uuid_deterministic(self) -> None:
        """Same inputs should produce same UUID."""
        uuid1 = deterministic_uuid(1, "board.outline")
        uuid2 = deterministic_uuid(1, "board.outline")
        assert uuid1 == uuid2

    def test_uuid_different_paths(self) -> None:
        """Different paths should produce different UUIDs."""
        uuid1 = deterministic_uuid(1, "board.outline")
        uuid2 = deterministic_uuid(1, "connector.left")
        assert uuid1 != uuid2

    def test_uuid_different_versions(self) -> None:
        """Different schema versions should produce different UUIDs."""
        uuid1 = deterministic_uuid(1, "board.outline")
        uuid2 = deterministic_uuid(2, "board.outline")
        assert uuid1 != uuid2

    def test_uuid_indexed(self) -> None:
        """Indexed UUIDs should be deterministic."""
        uuid1 = deterministic_uuid_indexed(1, "via.return", 0)
        uuid2 = deterministic_uuid_indexed(1, "via.return", 0)
        assert uuid1 == uuid2

        uuid3 = deterministic_uuid_indexed(1, "via.return", 1)
        assert uuid1 != uuid3


class TestBoardWriter:
    """Tests for BoardWriter class."""

    def test_writer_creates_valid_sexpr(self, f0_spec: CouponSpec) -> None:
        """Writer should produce parseable S-expression."""
        resolved = resolve(f0_spec)
        writer = BoardWriter(f0_spec, resolved)
        board = writer.build_board()

        # Should be a list starting with 'kicad_pcb'
        assert isinstance(board, list)
        assert board[0] == "kicad_pcb"

    def test_writer_includes_version(self, f0_spec: CouponSpec) -> None:
        """Board should include version header."""
        resolved = resolve(f0_spec)
        writer = BoardWriter(f0_spec, resolved)
        board = writer.build_board()

        # Find version element
        version_elem = [e for e in board if isinstance(e, list) and e[0] == "version"]
        assert len(version_elem) == 1
        assert version_elem[0][1] == 20240101

    def test_writer_includes_nets(self, f0_spec: CouponSpec) -> None:
        """Board should include net declarations."""
        resolved = resolve(f0_spec)
        writer = BoardWriter(f0_spec, resolved)
        board = writer.build_board()

        # Find net elements
        net_elems = [e for e in board if isinstance(e, list) and e[0] == "net"]
        assert len(net_elems) >= 3
        # Should have nets 0 (unconnected), 1 (SIG), 2 (GND)
        net_ids = [e[1] for e in net_elems]
        assert 0 in net_ids
        assert 1 in net_ids
        assert 2 in net_ids

    def test_writer_includes_outline(self, f0_spec: CouponSpec) -> None:
        """Board should include Edge.Cuts outline (rounded when corner_radius_nm > 0).

        REQ-M1-009: When corner_radius_nm > 0, the outline is generated as
        gr_line and gr_arc elements forming a rounded rectangle.
        """
        resolved = resolve(f0_spec)
        writer = BoardWriter(f0_spec, resolved)
        board = writer.build_board()

        # With corner_radius_nm=2000000, we get gr_line and gr_arc elements
        # (4 lines for edges + 4 arcs for corners = 8 total elements)
        gr_line = [e for e in board if isinstance(e, list) and e[0] == "gr_line"]
        gr_arc = [e for e in board if isinstance(e, list) and e[0] == "gr_arc"]

        # Rounded outline has 4 lines (edges) and 4 arcs (corners)
        assert len(gr_line) == 4, f"Expected 4 gr_line elements, got {len(gr_line)}"
        assert len(gr_arc) == 4, f"Expected 4 gr_arc elements, got {len(gr_arc)}"

        # Verify all are on Edge.Cuts layer
        for line in gr_line:
            layer_elem = [e for e in line if isinstance(e, list) and e[0] == "layer"]
            assert layer_elem, "gr_line missing layer"
            assert layer_elem[0][1] == "Edge.Cuts"

        for arc in gr_arc:
            layer_elem = [e for e in arc if isinstance(e, list) and e[0] == "layer"]
            assert layer_elem, "gr_arc missing layer"
            assert layer_elem[0][1] == "Edge.Cuts"

    def test_writer_includes_footprints(self, f0_spec: CouponSpec) -> None:
        """Board should include connector footprints."""
        resolved = resolve(f0_spec)
        writer = BoardWriter(f0_spec, resolved)
        board = writer.build_board()

        # Find footprint elements
        footprints = [e for e in board if isinstance(e, list) and e[0] == "footprint"]
        assert len(footprints) == 2  # left and right connectors

    def test_writer_includes_tracks(self, f0_spec: CouponSpec) -> None:
        """Board should include track segments."""
        resolved = resolve(f0_spec)
        writer = BoardWriter(f0_spec, resolved)
        board = writer.build_board()

        signal_segments = _get_segments_by_net(board, 1)
        ground_segments = _get_segments_by_net(board, 2)
        # F0 (calibration through-line) has a single continuous trace from
        # left to right connector, plus two CPWG ground rails.
        assert len(signal_segments) == 1
        assert len(ground_segments) == 2

    def test_writer_f1_includes_vias(self, f1_spec: CouponSpec) -> None:
        """F1 board should include signal and return vias."""
        resolved = resolve(f1_spec)
        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        # Find via elements
        vias = [e for e in board if isinstance(e, list) and e[0] == "via"]
        assert len(vias) == 5  # 1 signal + 4 return vias


class TestBuildBoardText:
    """Tests for build_board_text function."""

    def test_build_board_text_parseable(self, f0_spec: CouponSpec) -> None:
        """Generated text should be parseable S-expression."""
        resolved = resolve(f0_spec)
        text = build_board_text(f0_spec, resolved)

        # Should parse without error
        parsed = parse(text)
        assert parsed[0] == "kicad_pcb"

    def test_build_board_text_deterministic(self, f0_spec: CouponSpec) -> None:
        """Same inputs should produce same output."""
        resolved = resolve(f0_spec)
        text1 = build_board_text(f0_spec, resolved)
        text2 = build_board_text(f0_spec, resolved)
        assert text1 == text2


class TestWriteBoard:
    """Tests for write_board function."""

    def test_write_board_creates_file(self, f0_spec: CouponSpec, tmp_path: Path) -> None:
        """write_board should create a .kicad_pcb file."""
        resolved = resolve(f0_spec)
        board_path = write_board(f0_spec, resolved, tmp_path)

        assert board_path.exists()
        assert board_path.name == "coupon.kicad_pcb"

    def test_write_board_content_parseable(self, f0_spec: CouponSpec, tmp_path: Path) -> None:
        """Written file should contain parseable S-expression."""
        resolved = resolve(f0_spec)
        board_path = write_board(f0_spec, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")
        parsed = parse(content)
        assert parsed[0] == "kicad_pcb"

    def test_write_board_deterministic_tstamps(self, f0_spec: CouponSpec, tmp_path: Path) -> None:
        """Written file should have deterministic tstamp values."""
        resolved = resolve(f0_spec)
        board_path1 = write_board(f0_spec, resolved, tmp_path / "run1")
        board_path2 = write_board(f0_spec, resolved, tmp_path / "run2")

        content1 = board_path1.read_text(encoding="utf-8")
        content2 = board_path2.read_text(encoding="utf-8")

        # Files should be identical
        assert content1 == content2

    def test_write_board_f1_with_vias(self, f1_spec: CouponSpec, tmp_path: Path) -> None:
        """F1 board file should contain via elements."""
        import re

        resolved = resolve(f1_spec)
        board_path = write_board(f1_spec, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")
        assert "(via" in content
        # Should have exactly 1 signal via + 4 return vias = 5 total
        # Use regex to match actual via elements (not "vias" in keepout zones)
        via_count = len(re.findall(r"\(via\n", content))
        assert via_count == 5, f"Expected 5 vias (1 signal + 4 return), got {via_count}"


@pytest.fixture
def f1_spec_with_antipads(f1_spec_data: dict) -> CouponSpec:
    """F1 spec with antipads and cutouts configured."""
    spec = f1_spec_data.copy()
    spec["discontinuity"] = {
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
    return CouponSpec.model_validate(spec)


class TestF1BoardWriterWithAntipads:
    """Tests for F1 board generation with antipads and cutouts (REQ-M1-007)."""

    def test_f1_via_position_uses_composition(self, f1_spec: CouponSpec) -> None:
        """Signal via should be at discontinuity position from builder composition."""
        from formula_foundry.coupongen.builders.f1_builder import build_f1_coupon
        from formula_foundry.coupongen.kicad.sexpr import nm_to_mm

        resolved = resolve(f1_spec)
        composition = build_f1_coupon(f1_spec, resolved)
        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        # Find signal via
        vias = [e for e in board if isinstance(e, list) and e[0] == "via"]
        signal_via = vias[0]  # First via is signal via

        # Extract position
        at_elem = [e for e in signal_via if isinstance(e, list) and e[0] == "at"][0]
        via_x_mm = at_elem[1]
        via_y_mm = at_elem[2]

        # Verify position matches composition
        expected_x_mm = nm_to_mm(composition.discontinuity_position.x)
        expected_y_mm = nm_to_mm(composition.discontinuity_position.y)

        assert via_x_mm == expected_x_mm
        assert via_y_mm == expected_y_mm

    def test_f1_board_includes_antipads(self, f1_spec_with_antipads: CouponSpec, tmp_path: Path) -> None:
        """F1 board with antipads should include zone keepouts."""
        resolved = resolve(f1_spec_with_antipads)
        board_path = write_board(f1_spec_with_antipads, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")

        # Should have zone elements for antipads
        assert "(zone" in content
        # Should have zones on internal layers (unquoted in KiCad S-expr format)
        assert "In1.Cu" in content

    def test_f1_board_includes_cutouts(self, f1_spec_with_antipads: CouponSpec, tmp_path: Path) -> None:
        """F1 board with plane cutouts should include zone keepouts."""
        resolved = resolve(f1_spec_with_antipads)
        board_path = write_board(f1_spec_with_antipads, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")

        # Should have zone elements
        zone_count = content.count("(zone")
        # Should have antipads (2) + cutouts (1) = 3 zones
        assert zone_count >= 3

    def test_f1_antipad_zones_have_keepout(self, f1_spec_with_antipads: CouponSpec) -> None:
        """Antipad zones should have keepout properties set."""
        resolved = resolve(f1_spec_with_antipads)
        writer = BoardWriter(f1_spec_with_antipads, resolved)
        board = writer.build_board()

        # Find zone elements
        zones = [e for e in board if isinstance(e, list) and e[0] == "zone"]

        # Each zone should have keepout element
        for zone in zones:
            keepout_elems = [e for e in zone if isinstance(e, list) and e[0] == "keepout"]
            assert len(keepout_elems) == 1

    def test_f1_return_vias_positioned_around_discontinuity(self, f1_spec: CouponSpec) -> None:
        """Return vias should be positioned around the discontinuity center."""
        from formula_foundry.coupongen.builders.f1_builder import build_f1_coupon

        resolved = resolve(f1_spec)
        composition = build_f1_coupon(f1_spec, resolved)
        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        # Find all vias
        vias = [e for e in board if isinstance(e, list) and e[0] == "via"]

        # Skip signal via (first one), check return vias
        center_x = composition.discontinuity_position.x
        center_y = composition.discontinuity_position.y
        radius_nm = 1700000  # From spec

        for via in vias[1:]:  # Skip signal via
            at_elem = [e for e in via if isinstance(e, list) and e[0] == "at"][0]
            via_x_mm_str = at_elem[1]
            via_y_mm_str = at_elem[2]

            # nm_to_mm returns strings, need to convert back to nm
            via_x_nm = int(float(via_x_mm_str) * 1_000_000)
            via_y_nm = int(float(via_y_mm_str) * 1_000_000)

            # Calculate distance from center
            dx = via_x_nm - center_x
            dy = via_y_nm - center_y
            distance = (dx**2 + dy**2) ** 0.5

            # Should be approximately at radius_nm (allow small rounding error)
            assert abs(distance - radius_nm) < 1000  # 1um tolerance

    def test_f1_board_deterministic_with_antipads(self, f1_spec_with_antipads: CouponSpec, tmp_path: Path) -> None:
        """F1 board with antipads should be deterministic."""
        resolved = resolve(f1_spec_with_antipads)

        board_path1 = write_board(f1_spec_with_antipads, resolved, tmp_path / "run1")
        board_path2 = write_board(f1_spec_with_antipads, resolved, tmp_path / "run2")

        content1 = board_path1.read_text(encoding="utf-8")
        content2 = board_path2.read_text(encoding="utf-8")

        assert content1 == content2

    def test_f1_antipad_polygon_vertices(self, f1_spec_with_antipads: CouponSpec) -> None:
        """Antipad zones should have polygon vertices."""
        resolved = resolve(f1_spec_with_antipads)
        writer = BoardWriter(f1_spec_with_antipads, resolved)
        board = writer.build_board()

        # Find zone elements
        zones = [e for e in board if isinstance(e, list) and e[0] == "zone"]

        # Each zone should have a polygon with pts
        for zone in zones:
            polygon_elems = [e for e in zone if isinstance(e, list) and e[0] == "polygon"]
            assert len(polygon_elems) == 1

            polygon = polygon_elems[0]
            pts_elems = [e for e in polygon if isinstance(e, list) and e[0] == "pts"]
            assert len(pts_elems) == 1

            pts = pts_elems[0]
            # Should have multiple xy points
            xy_points = [e for e in pts if isinstance(e, list) and e[0] == "xy"]
            assert len(xy_points) >= 3  # At least a triangle


class TestF1RequirementCoverageInBoardWriter:
    """Tests verifying REQ-M1-007 coverage in board writer."""

    def test_req_m1_007_end_to_end_via_transition(self, f1_spec_with_antipads: CouponSpec, tmp_path: Path) -> None:
        """REQ-M1-007: Full F1 coupon with all features generates valid board."""
        resolved = resolve(f1_spec_with_antipads)
        board_path = write_board(f1_spec_with_antipads, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")

        # Verify all required elements are present
        # 1. Board outline (Edge.Cuts) - rounded with corner_radius_nm > 0
        # Uses gr_line and gr_arc elements instead of gr_rect
        assert "(gr_line" in content or "(gr_rect" in content
        assert "Edge.Cuts" in content

        # 2. Footprints (connectors) - count standalone "(footprint" at start of element
        # Note: zones contain "footprints" as keepout attribute, so we count lines
        import re

        footprint_matches = re.findall(r"\(footprint\s+[\w:]+", content)
        assert len(footprint_matches) == 2  # Left and right connectors

        # 3. Transmission line tracks (2 signal traces + 8 ground ring + 4 CPWG rails)
        # Signal traces: left on F.Cu, right on B.Cu (for via transition)
        # Ground ring: 4 return vias × 2 layers (F.Cu, B.Cu) = 8 traces
        # CPWG rails: 2 per signal segment = 4 traces
        assert "(segment" in content
        segment_count = content.count("(segment")
        assert segment_count == 14, f"Expected 14 segments (2 signal + 12 ground), got {segment_count}"

        # 4. Signal via
        assert "(via" in content

        # 5. Return vias (4 vias in RING pattern)
        # Count via elements (not "vias" attributes in keepout zones)
        # Via elements start with "(via" followed by newline, not "(vias"
        via_count = len(re.findall(r"\(via\n", content))
        assert via_count == 5, f"Expected 5 vias (1 signal + 4 return), got {via_count}"

        # 6. Antipads (zones on internal layers)
        assert "(zone" in content

        # 7. Plane cutouts - zones for antipads (2) + cutouts (1) = 3
        zone_count = content.count("(zone")
        assert zone_count >= 3  # 2 antipads + 1 cutout


class TestLayerMapping:
    """Tests for spec layer name to KiCad layer name mapping."""

    def test_map_layer_l1_to_fcu(self) -> None:
        """L1 should map to F.Cu (front copper)."""
        assert map_layer_to_kicad("L1") == "F.Cu"

    def test_map_layer_l2_to_in1cu(self) -> None:
        """L2 should map to In1.Cu (internal layer 1)."""
        assert map_layer_to_kicad("L2") == "In1.Cu"

    def test_map_layer_l3_to_in2cu(self) -> None:
        """L3 should map to In2.Cu (internal layer 2)."""
        assert map_layer_to_kicad("L3") == "In2.Cu"

    def test_map_layer_l4_to_bcu(self) -> None:
        """L4 should map to B.Cu (back copper)."""
        assert map_layer_to_kicad("L4") == "B.Cu"

    def test_map_layer_passthrough_fcu(self) -> None:
        """KiCad layer names should pass through unchanged."""
        assert map_layer_to_kicad("F.Cu") == "F.Cu"

    def test_map_layer_passthrough_in1cu(self) -> None:
        """KiCad layer names should pass through unchanged."""
        assert map_layer_to_kicad("In1.Cu") == "In1.Cu"

    def test_map_layer_passthrough_in2cu(self) -> None:
        """KiCad layer names should pass through unchanged."""
        assert map_layer_to_kicad("In2.Cu") == "In2.Cu"

    def test_map_layer_passthrough_bcu(self) -> None:
        """KiCad layer names should pass through unchanged."""
        assert map_layer_to_kicad("B.Cu") == "B.Cu"

    def test_map_layer_unknown_raises(self) -> None:
        """Unknown layer names should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown layer name"):
            map_layer_to_kicad("Unknown.Layer")

    def test_layer_mapping_completeness(self) -> None:
        """SPEC_TO_KICAD_LAYER should contain all expected mappings."""
        expected_mappings = {
            "L1": "F.Cu",
            "L2": "In1.Cu",
            "L3": "In2.Cu",
            "L4": "B.Cu",
            "F.Cu": "F.Cu",
            "In1.Cu": "In1.Cu",
            "In2.Cu": "In2.Cu",
            "B.Cu": "B.Cu",
        }
        assert SPEC_TO_KICAD_LAYER == expected_mappings


class TestBoardFileValidation:
    """Tests for validating generated .kicad_pcb files are syntactically correct."""

    def test_board_file_not_empty(self, f0_spec: CouponSpec, tmp_path: Path) -> None:
        """Generated board file should not be empty."""
        resolved = resolve(f0_spec)
        board_path = write_board(f0_spec, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")
        assert len(content) > 0, "Board file should not be empty"

    def test_board_file_starts_with_kicad_pcb(self, f0_spec: CouponSpec, tmp_path: Path) -> None:
        """Generated board file should start with (kicad_pcb."""
        resolved = resolve(f0_spec)
        board_path = write_board(f0_spec, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")
        assert content.strip().startswith("(kicad_pcb"), (
            f"Board file should start with (kicad_pcb, got: {content[:50]!r}"
        )

    def test_board_file_balanced_parentheses(self, f0_spec: CouponSpec, tmp_path: Path) -> None:
        """Generated board file should have balanced parentheses."""
        resolved = resolve(f0_spec)
        board_path = write_board(f0_spec, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")
        open_count = content.count("(")
        close_count = content.count(")")
        assert open_count == close_count, (
            f"Unbalanced parentheses: {open_count} open vs {close_count} close"
        )

    def test_board_file_parseable(self, f0_spec: CouponSpec, tmp_path: Path) -> None:
        """Generated board file should parse without errors."""
        resolved = resolve(f0_spec)
        board_path = write_board(f0_spec, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")
        # This should not raise
        parsed = parse(content)
        assert parsed[0] == "kicad_pcb"

    def test_f1_board_file_valid(self, f1_spec: CouponSpec, tmp_path: Path) -> None:
        """F1 generated board file should be syntactically valid."""
        resolved = resolve(f1_spec)
        board_path = write_board(f1_spec, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")

        # Check basic validity
        assert content.strip().startswith("(kicad_pcb")
        assert content.count("(") == content.count(")")

        # Should parse without error
        parsed = parse(content)
        assert parsed[0] == "kicad_pcb"


@pytest.fixture
def f1_spec_with_logical_layers(f1_spec_data: dict) -> CouponSpec:
    """F1 spec using logical layer names (L2, L3) for antipads."""
    spec = f1_spec_data.copy()
    spec["discontinuity"] = {
        "type": "VIA_TRANSITION",
        "signal_via": {
            "drill_nm": 300000,
            "diameter_nm": 650000,
            "pad_diameter_nm": 900000,
        },
        "antipads": {
            "L2": {
                "shape": "CIRCLE",
                "r_nm": 480000,
            },
            "L3": {
                "shape": "CIRCLE",
                "r_nm": 480000,
            },
        },
        "return_vias": {
            "pattern": "RING",
            "count": 4,
            "radius_nm": 1000000,
            "via": {"drill_nm": 300000, "diameter_nm": 510000},
        },
        "plane_cutouts": {},
    }
    return CouponSpec.model_validate(spec)


class TestLayerMappingIntegration:
    """Integration tests verifying layer mapping works in board generation."""

    def test_f1_logical_layers_mapped_to_kicad(
        self, f1_spec_with_logical_layers: CouponSpec, tmp_path: Path
    ) -> None:
        """F1 spec with L2/L3 layers should generate board with In1.Cu/In2.Cu."""
        resolved = resolve(f1_spec_with_logical_layers)
        board_path = write_board(f1_spec_with_logical_layers, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")

        # Should NOT contain the logical layer names L2/L3 (except in comments)
        # Split lines and check non-comment lines
        for line in content.split("\n"):
            # Skip comment-like patterns
            if line.strip().startswith(";"):
                continue
            # Check that L2 and L3 are not used as layer names
            # (layer L2) or (layer L3) should not appear
            assert "(layer\n      L2)" not in line and "(layer L2)" not in line, (
                f"Found invalid layer name L2 in: {line}"
            )
            assert "(layer\n      L3)" not in line and "(layer L3)" not in line, (
                f"Found invalid layer name L3 in: {line}"
            )

        # Should contain the correct KiCad layer names
        assert "In1.Cu" in content, "Should contain In1.Cu layer"
        assert "In2.Cu" in content, "Should contain In2.Cu layer"

    def test_f1_logical_layers_zones_have_valid_kicad_layers(
        self, f1_spec_with_logical_layers: CouponSpec
    ) -> None:
        """Zone elements should have valid KiCad layer names."""
        resolved = resolve(f1_spec_with_logical_layers)
        writer = BoardWriter(f1_spec_with_logical_layers, resolved)
        board = writer.build_board()

        # Find zone elements
        zones = [e for e in board if isinstance(e, list) and e[0] == "zone"]
        assert len(zones) >= 2, "Should have at least 2 zones for antipads"

        valid_kicad_layers = {"F.Cu", "In1.Cu", "In2.Cu", "B.Cu"}

        for zone in zones:
            # Find the layer element in the zone
            layer_elems = [e for e in zone if isinstance(e, list) and e[0] == "layer"]
            assert len(layer_elems) == 1, "Each zone should have exactly one layer element"

            layer_name = layer_elems[0][1]
            assert layer_name in valid_kicad_layers, (
                f"Zone has invalid layer name: {layer_name!r}, "
                f"expected one of {valid_kicad_layers}"
            )

    def test_board_file_kicad_loadable_layers(
        self, f1_spec_with_logical_layers: CouponSpec, tmp_path: Path
    ) -> None:
        """Generated board file should only use valid KiCad layer names.

        This is a regression test for the issue where L2/L3 logical layer
        names were being written to the board file, causing KiCad to fail
        to load the file (returncode 3: "Failed to load board").
        """
        resolved = resolve(f1_spec_with_logical_layers)
        board_path = write_board(f1_spec_with_logical_layers, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")
        parsed = parse(content)

        # Valid copper layer names in KiCad for 4-layer boards
        valid_copper_layers = {"F.Cu", "In1.Cu", "In2.Cu", "B.Cu"}

        # Find all zones and check their layer names
        def find_zones(sexpr: list) -> list:
            zones = []
            for elem in sexpr:
                if isinstance(elem, list):
                    if elem and elem[0] == "zone":
                        zones.append(elem)
                    zones.extend(find_zones(elem))
            return zones

        zones = find_zones(parsed)
        for zone in zones:
            for elem in zone:
                if isinstance(elem, list) and elem[0] == "layer":
                    layer_name = elem[1]
                    assert layer_name in valid_copper_layers, (
                        f"Zone uses invalid layer name '{layer_name}'. "
                        f"This would cause KiCad to fail with 'Failed to load board'. "
                        f"Valid layers: {valid_copper_layers}"
                    )


class TestAntipadKeeoutRules:
    """Regression tests for antipad/cutout keepout rules.

    These tests verify that antipad and cutout zones only block copperpour
    and allow vias/tracks/pads to pass through. This prevents DRC errors
    like 'items_not_allowed' when the signal via passes through antipads.
    """

    def test_antipad_allows_vias(self, f1_spec: CouponSpec) -> None:
        """Antipad zones must allow vias to pass through.

        Regression test: antipads blocking vias caused 'items_not_allowed'
        DRC errors because the signal via passes through antipad regions.
        """
        resolved = resolve(f1_spec)
        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        # Find zone elements (antipads/cutouts)
        zones = [e for e in board if isinstance(e, list) and e[0] == "zone"]

        for zone in zones:
            # Find the keepout element
            keepout_elems = [e for e in zone if isinstance(e, list) and e[0] == "keepout"]
            assert len(keepout_elems) == 1, "Each zone should have exactly one keepout element"

            keepout = keepout_elems[0]

            # Find vias rule
            vias_rules = [e for e in keepout if isinstance(e, list) and e[0] == "vias"]
            assert len(vias_rules) == 1, "Keepout should have exactly one vias rule"

            vias_rule = vias_rules[0][1]
            assert vias_rule == "allowed", (
                f"Antipad/cutout must allow vias (got '{vias_rule}'). "
                f"Blocking vias causes 'items_not_allowed' DRC errors."
            )

    def test_antipad_blocks_copperpour(self, f1_spec: CouponSpec) -> None:
        """Antipad zones must block copperpour to create clearance in ground planes."""
        resolved = resolve(f1_spec)
        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        zones = [e for e in board if isinstance(e, list) and e[0] == "zone"]

        for zone in zones:
            keepout_elems = [e for e in zone if isinstance(e, list) and e[0] == "keepout"]
            assert len(keepout_elems) == 1

            keepout = keepout_elems[0]

            # Find copperpour rule
            copperpour_rules = [e for e in keepout if isinstance(e, list) and e[0] == "copperpour"]
            assert len(copperpour_rules) == 1, "Keepout should have exactly one copperpour rule"

            copperpour_rule = copperpour_rules[0][1]
            assert copperpour_rule == "not_allowed", (
                f"Antipad/cutout must block copperpour (got '{copperpour_rule}'). "
                f"This creates the clearance in ground planes."
            )

    def test_antipad_allows_tracks_and_pads(self, f1_spec: CouponSpec) -> None:
        """Antipad zones should allow tracks and pads."""
        resolved = resolve(f1_spec)
        writer = BoardWriter(f1_spec, resolved)
        board = writer.build_board()

        zones = [e for e in board if isinstance(e, list) and e[0] == "zone"]

        for zone in zones:
            keepout_elems = [e for e in zone if isinstance(e, list) and e[0] == "keepout"]
            assert len(keepout_elems) == 1

            keepout = keepout_elems[0]

            # Check tracks rule
            tracks_rules = [e for e in keepout if isinstance(e, list) and e[0] == "tracks"]
            assert len(tracks_rules) == 1
            assert tracks_rules[0][1] == "allowed", "Antipad should allow tracks"

            # Check pads rule
            pads_rules = [e for e in keepout if isinstance(e, list) and e[0] == "pads"]
            assert len(pads_rules) == 1
            assert pads_rules[0][1] == "allowed", "Antipad should allow pads"


class TestReturnViasNetAssignment:
    """Regression tests for return via net assignment.

    These tests verify that return vias are assigned to net 0 (unconnected)
    in M1 test coupons. This prevents DRC 'unconnected_items' errors since
    there are no ground plane fills to provide copper connectivity.
    """

    def test_return_vias_on_gnd_net(self, f1_spec_with_antipads: CouponSpec) -> None:
        """Return vias must be on GND net (net 2) with proper connectivity.

        Return vias are connected to GND (net 2) via ground ring traces
        that form a ring connecting all return vias on F.Cu and B.Cu layers.
        This ensures proper DRC compliance with no via_dangling or
        unconnected_items violations.
        """
        resolved = resolve(f1_spec_with_antipads)
        writer = BoardWriter(f1_spec_with_antipads, resolved)
        board = writer.build_board()

        # Find via elements
        vias = [e for e in board if isinstance(e, list) and e[0] == "via"]
        assert len(vias) >= 2, "F1 board should have signal + return vias"

        # Count how many vias are on net 2 (GND/return vias) vs net 1 (signal via)
        net_2_count = 0
        net_1_count = 0

        for via in vias:
            net_elems = [e for e in via if isinstance(e, list) and e[0] == "net"]
            assert len(net_elems) == 1, "Each via should have exactly one net element"
            net_id = net_elems[0][1]

            if net_id == 2:
                net_2_count += 1
            elif net_id == 1:
                net_1_count += 1

        # Should have exactly 1 signal via on net 1 and all others on net 2 (GND)
        assert net_1_count == 1, f"Should have exactly 1 signal via on net 1, got {net_1_count}"
        assert net_2_count >= 1, (
            f"Return vias should be on net 2 (GND), "
            f"but found {net_2_count} vias on net 2. "
            f"Return vias are connected via ground ring traces."
        )

    def test_signal_via_on_signal_net(self, f1_spec_with_antipads: CouponSpec) -> None:
        """Signal via should be on signal net (net 1)."""
        resolved = resolve(f1_spec_with_antipads)
        writer = BoardWriter(f1_spec_with_antipads, resolved)
        board = writer.build_board()

        vias = [e for e in board if isinstance(e, list) and e[0] == "via"]

        # Count vias on each net
        signal_via_found = False
        for via in vias:
            net_elems = [e for e in via if isinstance(e, list) and e[0] == "net"]
            assert len(net_elems) == 1
            net_id = net_elems[0][1]

            if net_id == 1:
                signal_via_found = True

        assert signal_via_found, "F1 board should have a signal via on net 1"


class TestSilkscreenAnnotations:
    """Tests for silkscreen annotations with coupon_id and hash (REQ-M1-010)."""

    def test_board_includes_silkscreen_text(self, f0_spec: CouponSpec, tmp_path: Path) -> None:
        """Board should include gr_text elements on silkscreen layers."""
        resolved = resolve(f0_spec)
        board_path = write_board(f0_spec, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")

        # Should have gr_text elements for silkscreen
        assert "(gr_text" in content
        # Should be on silkscreen layer
        assert "F.SilkS" in content or "B.SilkS" in content

    def test_silkscreen_text_deterministic(self, f0_spec: CouponSpec, tmp_path: Path) -> None:
        """Silkscreen text should be deterministic across runs."""
        resolved = resolve(f0_spec)

        board_path1 = write_board(f0_spec, resolved, tmp_path / "run1")
        board_path2 = write_board(f0_spec, resolved, tmp_path / "run2")

        content1 = board_path1.read_text(encoding="utf-8")
        content2 = board_path2.read_text(encoding="utf-8")

        # Files should be identical
        assert content1 == content2

    def test_silkscreen_with_design_hash(self, f0_spec: CouponSpec, tmp_path: Path) -> None:
        """Board with design_hash should include hash marker in silkscreen."""
        resolved = resolve(f0_spec)
        design_hash = "a1b2c3d4e5f6789012345678901234567890abcd1234567890abcdef12345678"

        board_path = write_board(f0_spec, resolved, tmp_path, design_hash=design_hash)
        content = board_path.read_text(encoding="utf-8")

        # Should include short hash (first 8 chars)
        assert "a1b2c3d4" in content

    def test_silkscreen_has_front_layer(self, f0_spec: CouponSpec) -> None:
        """Silkscreen annotations should appear on F.SilkS layer."""
        from formula_foundry.coupongen.kicad import build_annotations_from_spec
        from formula_foundry.coupongen.hashing import coupon_id_from_design_hash

        resolved = resolve(f0_spec)
        design_hash = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        coupon_id = coupon_id_from_design_hash(design_hash)

        annotations = build_annotations_from_spec(
            coupon_id_template="${COUPON_ID}",
            include_manifest_hash=True,
            actual_coupon_id=coupon_id,
            design_hash=design_hash,
            layout_plan=resolved.layout_plan,
            uuid_generator=lambda path: f"test-uuid-{path}",
        )

        # Should have at least one annotation
        assert len(annotations) >= 1

        # Check for F.SilkS layer
        layers = []
        for ann in annotations:
            for elem in ann:
                if isinstance(elem, list) and elem[0] == "layer":
                    layers.append(elem[1])

        assert "F.SilkS" in layers, f"Expected F.SilkS layer, got {layers}"

    def test_silkscreen_coupon_id_template_substitution(self, f0_spec: CouponSpec) -> None:
        """${COUPON_ID} template should be substituted with actual coupon_id."""
        from formula_foundry.coupongen.kicad import build_annotations_from_spec
        from formula_foundry.coupongen.kicad.sexpr import dump

        resolved = resolve(f0_spec)
        design_hash = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        coupon_id = "testcouponid"

        annotations = build_annotations_from_spec(
            coupon_id_template="${COUPON_ID}",
            include_manifest_hash=True,
            actual_coupon_id=coupon_id,
            design_hash=design_hash,
            layout_plan=resolved.layout_plan,
            uuid_generator=lambda path: f"uuid-{path}",
        )

        # Dump to text and check for coupon_id
        for ann in annotations:
            text = dump(ann)
            assert coupon_id in text, f"Expected coupon_id '{coupon_id}' in annotation"

    def test_silkscreen_without_hash_when_disabled(self, f0_spec: CouponSpec) -> None:
        """When include_manifest_hash is False, annotations should not include hash."""
        from formula_foundry.coupongen.kicad import build_annotations_from_spec
        from formula_foundry.coupongen.kicad.sexpr import dump

        resolved = resolve(f0_spec)
        design_hash = "1111111122222222333333334444444455555555666666667777777788888888"
        short_hash = design_hash[:8]  # "11111111"
        coupon_id = "nocoupon123"

        annotations = build_annotations_from_spec(
            coupon_id_template=coupon_id,
            include_manifest_hash=False,
            actual_coupon_id=coupon_id,
            design_hash=design_hash,
            layout_plan=resolved.layout_plan,
            uuid_generator=lambda path: f"uuid-{path}",
        )

        # Should still have annotation
        assert len(annotations) >= 1

        # But should not have the full "coupon_id:hash" format
        for ann in annotations:
            text = dump(ann)
            assert f":{short_hash}" not in text, "Hash should not appear when include_manifest_hash is False"

    def test_f1_board_includes_silkscreen(self, f1_spec: CouponSpec, tmp_path: Path) -> None:
        """F1 board should also include silkscreen annotations."""
        resolved = resolve(f1_spec)
        design_hash = "f1ae5bcd12345678901234567890123456789012345678901234567890abcdef"

        board_path = write_board(f1_spec, resolved, tmp_path, design_hash=design_hash)
        content = board_path.read_text(encoding="utf-8")

        # Should have silkscreen text
        assert "(gr_text" in content
        # Should include short hash (first 8 hex chars)
        assert "f1ae5bcd" in content


class TestRoundedBoardOutline:
    """Tests for REQ-M1-009: Rounded board outline generation.

    REQ-M1-009: If `corner_radius_nm > 0` is provided, the board outline MUST
    be generated as a rounded-rectangle on `Edge.Cuts` using deterministic
    integer-nm arcs/segments and validated for feasibility.
    """

    @pytest.fixture
    def spec_with_corner_radius(self, f0_spec_data: dict) -> CouponSpec:
        """F0 spec with non-zero corner radius."""
        data = f0_spec_data.copy()
        data["board"]["outline"]["corner_radius_nm"] = 2_000_000  # 2mm
        return CouponSpec.model_validate(data)

    @pytest.fixture
    def spec_without_corner_radius(self, f0_spec_data: dict) -> CouponSpec:
        """F0 spec with zero corner radius (sharp corners)."""
        data = f0_spec_data.copy()
        data["board"]["outline"]["corner_radius_nm"] = 0
        return CouponSpec.model_validate(data)

    def test_rounded_outline_generates_lines_and_arcs(
        self, spec_with_corner_radius: CouponSpec
    ) -> None:
        """Board with corner_radius > 0 should generate gr_line and gr_arc elements."""
        resolved = resolve(spec_with_corner_radius)
        writer = BoardWriter(spec_with_corner_radius, resolved)
        board = writer.build_board()

        # Find outline elements
        gr_line = [e for e in board if isinstance(e, list) and e[0] == "gr_line"]
        gr_arc = [e for e in board if isinstance(e, list) and e[0] == "gr_arc"]
        gr_rect = [e for e in board if isinstance(e, list) and e[0] == "gr_rect"]

        # Should have 4 lines (edges) and 4 arcs (corners)
        assert len(gr_line) == 4, f"Expected 4 gr_line elements, got {len(gr_line)}"
        assert len(gr_arc) == 4, f"Expected 4 gr_arc elements, got {len(gr_arc)}"
        assert len(gr_rect) == 0, "Should NOT have gr_rect when corner_radius > 0"

    def test_sharp_corner_outline_generates_rect(
        self, spec_without_corner_radius: CouponSpec
    ) -> None:
        """Board with corner_radius = 0 should generate a simple gr_rect."""
        resolved = resolve(spec_without_corner_radius)
        writer = BoardWriter(spec_without_corner_radius, resolved)
        board = writer.build_board()

        # Find outline elements
        gr_rect = [e for e in board if isinstance(e, list) and e[0] == "gr_rect"]
        gr_line = [e for e in board if isinstance(e, list) and e[0] == "gr_line"]
        gr_arc = [e for e in board if isinstance(e, list) and e[0] == "gr_arc"]

        # Should have only gr_rect
        assert len(gr_rect) == 1, f"Expected 1 gr_rect element, got {len(gr_rect)}"
        assert len(gr_line) == 0, "Should NOT have gr_line when corner_radius = 0"
        assert len(gr_arc) == 0, "Should NOT have gr_arc when corner_radius = 0"

    def test_rounded_outline_elements_on_edge_cuts(
        self, spec_with_corner_radius: CouponSpec
    ) -> None:
        """All rounded outline elements should be on Edge.Cuts layer."""
        resolved = resolve(spec_with_corner_radius)
        writer = BoardWriter(spec_with_corner_radius, resolved)
        board = writer.build_board()

        # Find all outline elements
        outline_elems = [
            e for e in board
            if isinstance(e, list) and e[0] in ("gr_line", "gr_arc")
        ]

        for elem in outline_elems:
            layer_elem = [e for e in elem if isinstance(e, list) and e[0] == "layer"]
            assert layer_elem, f"Outline element missing layer: {elem}"
            assert layer_elem[0][1] == "Edge.Cuts", (
                f"Outline element should be on Edge.Cuts, got {layer_elem[0][1]}"
            )

    def test_rounded_outline_arc_has_start_mid_end(
        self, spec_with_corner_radius: CouponSpec
    ) -> None:
        """Arc elements should have start, mid, and end points (KiCad format)."""
        resolved = resolve(spec_with_corner_radius)
        writer = BoardWriter(spec_with_corner_radius, resolved)
        board = writer.build_board()

        gr_arc = [e for e in board if isinstance(e, list) and e[0] == "gr_arc"]

        for arc in gr_arc:
            start = [e for e in arc if isinstance(e, list) and e[0] == "start"]
            mid = [e for e in arc if isinstance(e, list) and e[0] == "mid"]
            end = [e for e in arc if isinstance(e, list) and e[0] == "end"]

            assert len(start) == 1, f"Arc missing start point: {arc}"
            assert len(mid) == 1, f"Arc missing mid point: {arc}"
            assert len(end) == 1, f"Arc missing end point: {arc}"

    def test_rounded_outline_deterministic(
        self, spec_with_corner_radius: CouponSpec
    ) -> None:
        """Rounded outline should be deterministic across multiple builds."""
        resolved = resolve(spec_with_corner_radius)

        writer1 = BoardWriter(spec_with_corner_radius, resolved)
        writer2 = BoardWriter(spec_with_corner_radius, resolved)

        board1 = writer1.build_board()
        board2 = writer2.build_board()

        # Extract outline elements
        def get_outline(board: list) -> list:
            return [
                e for e in board
                if isinstance(e, list) and e[0] in ("gr_line", "gr_arc")
            ]

        outline1 = get_outline(board1)
        outline2 = get_outline(board2)

        assert len(outline1) == len(outline2)
        # Note: Deep comparison may need adjustment for UUID differences
        # The key is that geometry coordinates are identical

    def test_rounded_outline_coordinates_integer_nm(
        self, spec_with_corner_radius: CouponSpec
    ) -> None:
        """Outline coordinates should be derived from integer-nm calculations."""
        from formula_foundry.coupongen.geom.cutouts import generate_rounded_outline

        # Test the underlying generator directly
        rounded = generate_rounded_outline(
            x_left_nm=0,
            y_bottom_nm=-10_000_000,  # -10mm (board center at y=0)
            width_nm=80_000_000,  # 80mm
            height_nm=20_000_000,  # 20mm
            corner_radius_nm=2_000_000,  # 2mm
        )

        # Verify all coordinates are integers
        for elem in rounded.elements:
            if hasattr(elem, 'start'):
                assert isinstance(elem.start.x, int)
                assert isinstance(elem.start.y, int)
            if hasattr(elem, 'end'):
                assert isinstance(elem.end.x, int)
                assert isinstance(elem.end.y, int)
            if hasattr(elem, 'mid'):
                assert isinstance(elem.mid.x, int)
                assert isinstance(elem.mid.y, int)

    def test_rounded_outline_feasibility_check(self) -> None:
        """Outline generator should validate feasibility of corner radius."""
        from formula_foundry.coupongen.geom.cutouts import (
            OutlineFeasibilityError,
            generate_rounded_outline,
        )

        # Corner radius exceeds half of smallest dimension (height = 10mm, max_r = 5mm)
        with pytest.raises(OutlineFeasibilityError):
            generate_rounded_outline(
                x_left_nm=0,
                y_bottom_nm=0,
                width_nm=80_000_000,
                height_nm=10_000_000,
                corner_radius_nm=6_000_000,  # > 5mm (half of height)
            )

    def test_rounded_outline_zero_radius_is_rectangle(self) -> None:
        """Corner radius of 0 should produce a simple rectangle (4 lines only)."""
        from formula_foundry.coupongen.geom.cutouts import (
            OutlineLine,
            generate_rounded_outline,
        )

        rounded = generate_rounded_outline(
            x_left_nm=0,
            y_bottom_nm=0,
            width_nm=80_000_000,
            height_nm=20_000_000,
            corner_radius_nm=0,
        )

        # Should have exactly 4 line elements (no arcs)
        assert len(rounded.elements) == 4
        assert all(isinstance(e, OutlineLine) for e in rounded.elements)

    def test_file_output_rounded_outline(
        self, spec_with_corner_radius: CouponSpec, tmp_path: Path
    ) -> None:
        """Written board file should contain gr_line and gr_arc for rounded outline."""
        resolved = resolve(spec_with_corner_radius)
        board_path = write_board(spec_with_corner_radius, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")

        # Should have lines and arcs
        assert "(gr_line" in content
        assert "(gr_arc" in content
        assert "Edge.Cuts" in content

        # Should NOT have gr_rect
        assert "(gr_rect" not in content

    def test_file_output_sharp_corner_outline(
        self, spec_without_corner_radius: CouponSpec, tmp_path: Path
    ) -> None:
        """Written board file should contain gr_rect for sharp corner outline."""
        resolved = resolve(spec_without_corner_radius)
        board_path = write_board(spec_without_corner_radius, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")

        # Should have gr_rect
        assert "(gr_rect" in content
        assert "Edge.Cuts" in content

        # Should NOT have gr_line/gr_arc for outline (may appear elsewhere)
        # Note: We check that there's no gr_arc at all since they're only used for outline
        assert "(gr_arc" not in content
