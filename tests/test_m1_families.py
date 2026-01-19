from __future__ import annotations

from pathlib import Path
from typing import Any

from formula_foundry.coupongen.api import generate_kicad, validate_spec
from formula_foundry.coupongen.spec import CouponSpec


def _base_spec_data() -> dict[str, Any]:
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
    data = _base_spec_data()
    data["coupon_family"] = "F0_CAL_THRU_LINE"
    data["discontinuity"] = None
    return data


def _f1_spec_data() -> dict[str, Any]:
    data = _base_spec_data()
    data["coupon_family"] = "F1_SINGLE_ENDED_VIA"
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


def test_family_f0_builds(tmp_path: Path) -> None:
    spec = CouponSpec.model_validate(_f0_spec_data())
    evaluation = validate_spec(spec, out_dir=tmp_path)
    project = generate_kicad(evaluation.resolved, evaluation.spec, tmp_path)

    assert project.board_path.exists()


def test_family_f1_builds(tmp_path: Path) -> None:
    spec = CouponSpec.model_validate(_f1_spec_data())
    evaluation = validate_spec(spec, out_dir=tmp_path)
    project = generate_kicad(evaluation.resolved, evaluation.spec, tmp_path)

    assert project.board_path.exists()
