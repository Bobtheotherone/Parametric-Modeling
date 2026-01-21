from __future__ import annotations

from typing import Any

import pytest

from formula_foundry.coupongen.constraints import (
    ConstraintViolation,
    constraint_proof_payload,
    enforce_constraints,
    evaluate_constraints,
)
from formula_foundry.coupongen.spec import CouponSpec


def _example_spec_data() -> dict[str, Any]:
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
            # CP-2.2: For F1, length_right_nm is derived from continuity.
            # With left connector at 5mm, right at 75mm (70mm span),
            # and length_left=35mm, discontinuity is at 40mm,
            # so derived length_right = 75 - 40 = 35mm (symmetric).
            "length_left_nm": 35000000,
            "length_right_nm": 35000000,  # Must match derived value for F1
            "ground_via_fence": {
                "enabled": True,
                "pitch_nm": 1500000,
                "offset_from_gap_nm": 800000,
                "via": {"drill_nm": 300000, "diameter_nm": 600000},
            },
        },
        "discontinuity": {
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


def test_constraint_tiers_exist() -> None:
    spec = CouponSpec.model_validate(_example_spec_data())
    proof = evaluate_constraints(spec)

    assert set(proof.tiers.keys()) == {"T0", "T1", "T2", "T3", "T4"}


def test_reject_mode_reports_constraint_ids() -> None:
    data = _example_spec_data()
    data["transmission_line"]["w_nm"] = 10
    data["constraints"]["mode"] = "REJECT"
    spec = CouponSpec.model_validate(data)

    with pytest.raises(ConstraintViolation) as exc:
        enforce_constraints(spec)

    ids = {result.constraint_id for result in exc.value.violations}
    assert "T0_TRACE_WIDTH_MIN" in ids


def test_repair_mode_emits_repair_map_and_distance() -> None:
    data = _example_spec_data()
    data["transmission_line"]["w_nm"] = 10
    data["constraints"]["mode"] = "REPAIR"
    spec = CouponSpec.model_validate(data)

    evaluation = enforce_constraints(spec)

    assert evaluation.repair_info is not None
    assert evaluation.repair_info.repair_map["transmission_line.w_nm"]["after"] >= 100000
    assert evaluation.repair_info.repair_distance > 0


def test_constraint_proof_schema() -> None:
    spec = CouponSpec.model_validate(_example_spec_data())
    proof = evaluate_constraints(spec)
    payload = constraint_proof_payload(proof)

    assert payload["passed"] is True
    assert payload["tiers"]["T0"] == [result.constraint_id for result in proof.tiers["T0"]]
    for entry in payload["constraints"]:
        assert {"id", "description", "tier", "value", "limit", "margin", "passed"} <= set(entry)
