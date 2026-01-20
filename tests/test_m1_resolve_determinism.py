from __future__ import annotations

import pytest

from formula_foundry.coupongen.resolve import design_hash, resolve, resolved_design_canonical_json
from formula_foundry.coupongen.spec import CouponSpec
from formula_foundry.substrate import sha256_bytes


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
            "antipads": {
                "L2": {
                    "shape": "ROUNDRECT",
                    "rx_nm": 1200000,
                    "ry_nm": 900000,
                    "corner_nm": 250000,
                },
                "L3": {"shape": "CIRCLE", "r_nm": 1100000},
            },
            "return_vias": {
                "pattern": "RING",
                "count": 4,
                "radius_nm": 1700000,
                "via": {"drill_nm": 300000, "diameter_nm": 650000},
            },
            "plane_cutouts": {
                "L2": {
                    "shape": "SLOT",
                    "length_nm": 3000000,
                    "width_nm": 1500000,
                    "rotation_deg": 0,
                }
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


def _example_spec_data_reordered() -> dict[str, object]:
    original = _example_spec_data()
    connectors = original["connectors"]
    discontinuity = original["discontinuity"]
    return {
        "coupon_family": original["coupon_family"],
        "schema_version": original["schema_version"],
        "units": original["units"],
        "constraints": original["constraints"],
        "export": original["export"],
        "board": original["board"],
        "stackup": original["stackup"],
        "toolchain": original["toolchain"],
        "fab_profile": original["fab_profile"],
        "transmission_line": original["transmission_line"],
        "connectors": {"right": connectors["right"], "left": connectors["left"]},
        "discontinuity": {
            "return_vias": discontinuity["return_vias"],
            "signal_via": discontinuity["signal_via"],
            "plane_cutouts": discontinuity["plane_cutouts"],
            "antipads": discontinuity["antipads"],
            "type": discontinuity["type"],
        },
    }


def test_resolve_emits_integer_nm_and_groups() -> None:
    """Resolve must emit integer nanometer parameters and dimensionless groups."""
    spec = CouponSpec.model_validate(_example_spec_data())
    resolved = resolve(spec)

    assert resolved.parameters_nm["board.outline.width_nm"] == 20000000
    assert resolved.derived_features["board_area_nm2"] == 20000000 * 80000000
    assert resolved.derived_features["trace_total_length_nm"] == 50000000
    assert resolved.dimensionless_groups["cpwg_w_over_gap"] == pytest.approx(300000 / 180000)

    for value in resolved.parameters_nm.values():
        assert isinstance(value, int)
    for value in resolved.derived_features.values():
        assert isinstance(value, int)


def test_design_hash_is_stable() -> None:
    """Design hash must be stable across multiple resolves of the same spec."""
    spec = CouponSpec.model_validate(_example_spec_data())
    resolved_a = resolve(spec)
    resolved_b = resolve(spec)

    assert design_hash(resolved_a) == design_hash(resolved_b)


def test_resolved_design_canonical_json_is_stable() -> None:
    """Canonical JSON must be byte-identical regardless of input key order."""
    spec_a = CouponSpec.model_validate(_example_spec_data())
    spec_b = CouponSpec.model_validate(_example_spec_data_reordered())
    resolved_a = resolve(spec_a)
    resolved_b = resolve(spec_b)

    assert resolved_design_canonical_json(resolved_a) == resolved_design_canonical_json(resolved_b)


def test_design_hash_matches_canonical_json_sha256() -> None:
    """Design hash must be sha256 of the canonical JSON representation."""
    spec = CouponSpec.model_validate(_example_spec_data())
    resolved = resolve(spec)

    canonical = resolved_design_canonical_json(resolved)
    expected = sha256_bytes(canonical.encode("utf-8"))
    assert design_hash(resolved) == expected
