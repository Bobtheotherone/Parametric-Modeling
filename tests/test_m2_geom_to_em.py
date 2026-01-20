from __future__ import annotations

from typing import Any

from formula_foundry.coupongen.hashing import coupon_id_from_design_hash
from formula_foundry.coupongen.resolve import design_hash, resolve
from formula_foundry.coupongen.spec import CouponSpec
from formula_foundry.openems.convert import build_simulation_spec, simulation_canonical_json
from formula_foundry.openems.geometry import build_geometry_spec, geometry_canonical_json


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
            "length_left_nm": 25000000,
            "length_right_nm": 25000000,
            "ground_via_fence": None,
        },
        "discontinuity": {
            "type": "VIA_TRANSITION",
            "signal_via": {
                "drill_nm": 300000,
                "diameter_nm": 650000,
                "pad_diameter_nm": 900000,
            },
            "antipads": {},
            "return_vias": None,
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


def _manifest_for(spec: CouponSpec) -> dict[str, Any]:
    resolved = resolve(spec)
    design_hash_value = design_hash(resolved)
    return {
        "design_hash": design_hash_value,
        "coupon_id": coupon_id_from_design_hash(design_hash_value),
        "coupon_family": spec.coupon_family,
        "stackup": spec.stackup.model_dump(mode="json"),
    }


def test_build_geometry_spec_extracts_layers_and_discontinuity() -> None:
    spec = CouponSpec.model_validate(_example_spec_data())
    resolved = resolve(spec)
    manifest = _manifest_for(spec)

    geometry = build_geometry_spec(resolved, manifest)

    assert geometry.board.width_nm == spec.board.outline.width_nm
    assert geometry.board.length_nm == spec.board.outline.length_nm
    assert geometry.layers[0].id == "L1"
    assert geometry.layers[0].z_nm == 0
    assert geometry.layers[1].z_nm == spec.stackup.thicknesses_nm["L1_to_L2"]
    assert geometry.layers[2].z_nm == spec.stackup.thicknesses_nm["L1_to_L2"] + spec.stackup.thicknesses_nm["L2_to_L3"]
    assert geometry.layers[3].z_nm == sum(spec.stackup.thicknesses_nm.values())
    assert geometry.discontinuity is not None
    assert geometry.discontinuity.parameters_nm["signal_via.drill_nm"] == spec.discontinuity.signal_via.drill_nm

    first = geometry_canonical_json(geometry)
    second = geometry_canonical_json(geometry)
    assert first == second


def test_build_simulation_spec_ports_and_materials() -> None:
    spec = CouponSpec.model_validate(_example_spec_data())
    resolved = resolve(spec)
    manifest = _manifest_for(spec)

    simulation = build_simulation_spec(resolved, manifest)

    assert simulation.geometry_ref.design_hash == manifest["design_hash"]
    assert simulation.ports[0].position_nm[0] == spec.connectors.left.position_nm[0]
    assert simulation.ports[0].direction == "x"
    assert simulation.ports[1].position_nm[0] == spec.connectors.right.position_nm[0]
    assert simulation.ports[1].direction == "-x"
    assert simulation.materials.dielectrics[0].epsilon_r == spec.stackup.materials.er
    assert simulation.materials.dielectrics[0].loss_tangent == spec.stackup.materials.loss_tangent

    sim_payload = simulation_canonical_json(simulation)
    assert sim_payload == simulation_canonical_json(simulation)
