from __future__ import annotations

from pathlib import Path

from formula_foundry.coupongen.kicad import BackendA, deterministic_uuid
from formula_foundry.coupongen.paths import FOOTPRINT_LIB_DIR
from formula_foundry.coupongen.resolve import resolve
from formula_foundry.coupongen.spec import CouponSpec


def _example_spec_data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "coupon_family": "F1_SINGLE_ENDED_VIA",
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


def test_backend_a_exists_and_writes_board(tmp_path: Path) -> None:
    spec = CouponSpec.model_validate(_example_spec_data())
    resolved = resolve(spec)
    backend = BackendA()

    board_path = backend.write_board(spec, resolved, tmp_path)

    assert board_path.exists()
    assert "(kicad_pcb" in board_path.read_text(encoding="utf-8")


def test_deterministic_uuid_generation() -> None:
    uuid_a = deterministic_uuid(1, "board.outline")
    uuid_b = deterministic_uuid(1, "board.outline")
    uuid_c = deterministic_uuid(1, "connector.left")

    assert uuid_a == uuid_b
    assert uuid_a != uuid_c


def test_footprints_are_vendored() -> None:
    spec = CouponSpec.model_validate(_example_spec_data())
    footprint = spec.connectors.left.footprint
    lib, name = footprint.split(":")
    footprint_path = FOOTPRINT_LIB_DIR / f"{lib}.pretty" / f"{name}.kicad_mod"

    assert footprint_path.exists()
