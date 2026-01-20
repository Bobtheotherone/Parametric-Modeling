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
