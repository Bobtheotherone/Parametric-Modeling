"""Tests for board writer and deterministic UUID generation (REQ-M1-012, REQ-M1-013)."""

from __future__ import annotations

import json
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
from formula_foundry.coupongen.resolve import resolve
from formula_foundry.coupongen.spec import CouponSpec


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
        """Board should include Edge.Cuts outline."""
        resolved = resolve(f0_spec)
        writer = BoardWriter(f0_spec, resolved)
        board = writer.build_board()

        # Find gr_rect element
        gr_rect = [e for e in board if isinstance(e, list) and e[0] == "gr_rect"]
        assert len(gr_rect) == 1

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

        # Find segment elements
        segments = [e for e in board if isinstance(e, list) and e[0] == "segment"]
        assert len(segments) == 2  # left and right tracks

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

    def test_write_board_creates_file(
        self, f0_spec: CouponSpec, tmp_path: Path
    ) -> None:
        """write_board should create a .kicad_pcb file."""
        resolved = resolve(f0_spec)
        board_path = write_board(f0_spec, resolved, tmp_path)

        assert board_path.exists()
        assert board_path.name == "coupon.kicad_pcb"

    def test_write_board_content_parseable(
        self, f0_spec: CouponSpec, tmp_path: Path
    ) -> None:
        """Written file should contain parseable S-expression."""
        resolved = resolve(f0_spec)
        board_path = write_board(f0_spec, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")
        parsed = parse(content)
        assert parsed[0] == "kicad_pcb"

    def test_write_board_deterministic_tstamps(
        self, f0_spec: CouponSpec, tmp_path: Path
    ) -> None:
        """Written file should have deterministic tstamp values."""
        resolved = resolve(f0_spec)
        board_path1 = write_board(f0_spec, resolved, tmp_path / "run1")
        board_path2 = write_board(f0_spec, resolved, tmp_path / "run2")

        content1 = board_path1.read_text(encoding="utf-8")
        content2 = board_path2.read_text(encoding="utf-8")

        # Files should be identical
        assert content1 == content2

    def test_write_board_f1_with_vias(
        self, f1_spec: CouponSpec, tmp_path: Path
    ) -> None:
        """F1 board file should contain via elements."""
        resolved = resolve(f1_spec)
        board_path = write_board(f1_spec, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")
        assert "(via" in content
        # Should have signal via and return vias
        via_count = content.count("(via")
        assert via_count == 5  # 1 signal + 4 return


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

    def test_f1_board_includes_antipads(
        self, f1_spec_with_antipads: CouponSpec, tmp_path: Path
    ) -> None:
        """F1 board with antipads should include zone keepouts."""
        resolved = resolve(f1_spec_with_antipads)
        board_path = write_board(f1_spec_with_antipads, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")

        # Should have zone elements for antipads
        assert "(zone" in content
        # Should have zones on internal layers (unquoted in KiCad S-expr format)
        assert "In1.Cu" in content

    def test_f1_board_includes_cutouts(
        self, f1_spec_with_antipads: CouponSpec, tmp_path: Path
    ) -> None:
        """F1 board with plane cutouts should include zone keepouts."""
        resolved = resolve(f1_spec_with_antipads)
        board_path = write_board(f1_spec_with_antipads, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")

        # Should have zone elements
        zone_count = content.count("(zone")
        # Should have antipads (2) + cutouts (1) = 3 zones
        assert zone_count >= 3

    def test_f1_antipad_zones_have_keepout(
        self, f1_spec_with_antipads: CouponSpec
    ) -> None:
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

    def test_f1_return_vias_positioned_around_discontinuity(
        self, f1_spec: CouponSpec
    ) -> None:
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
            distance = (dx ** 2 + dy ** 2) ** 0.5

            # Should be approximately at radius_nm (allow small rounding error)
            assert abs(distance - radius_nm) < 1000  # 1um tolerance

    def test_f1_board_deterministic_with_antipads(
        self, f1_spec_with_antipads: CouponSpec, tmp_path: Path
    ) -> None:
        """F1 board with antipads should be deterministic."""
        resolved = resolve(f1_spec_with_antipads)

        board_path1 = write_board(f1_spec_with_antipads, resolved, tmp_path / "run1")
        board_path2 = write_board(f1_spec_with_antipads, resolved, tmp_path / "run2")

        content1 = board_path1.read_text(encoding="utf-8")
        content2 = board_path2.read_text(encoding="utf-8")

        assert content1 == content2

    def test_f1_antipad_polygon_vertices(
        self, f1_spec_with_antipads: CouponSpec
    ) -> None:
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

    def test_req_m1_007_end_to_end_via_transition(
        self, f1_spec_with_antipads: CouponSpec, tmp_path: Path
    ) -> None:
        """REQ-M1-007: Full F1 coupon with all features generates valid board."""
        resolved = resolve(f1_spec_with_antipads)
        board_path = write_board(f1_spec_with_antipads, resolved, tmp_path)

        content = board_path.read_text(encoding="utf-8")

        # Verify all required elements are present
        # 1. Board outline (Edge.Cuts)
        assert "(gr_rect" in content
        assert "Edge.Cuts" in content

        # 2. Footprints (connectors) - count standalone "(footprint" at start of element
        # Note: zones contain "footprints" as keepout attribute, so we count lines
        import re
        footprint_matches = re.findall(r'\(footprint\s+[\w:]+', content)
        assert len(footprint_matches) == 2  # Left and right connectors

        # 3. Transmission line tracks
        assert "(segment" in content
        segment_count = content.count("(segment")
        assert segment_count == 2  # Left and right tracks

        # 4. Signal via
        assert "(via" in content

        # 5. Return vias (4 vias in RING pattern)
        # Count via elements (not "vias" attributes in keepout zones)
        # Via elements start with "(via" followed by newline, not "(vias"
        via_count = len(re.findall(r'\(via\n', content))
        assert via_count == 5  # 1 signal + 4 return

        # 6. Antipads (zones on internal layers)
        assert "(zone" in content

        # 7. Plane cutouts - zones for antipads (2) + cutouts (1) = 3
        zone_count = content.count("(zone")
        assert zone_count >= 3  # 2 antipads + 1 cutout
