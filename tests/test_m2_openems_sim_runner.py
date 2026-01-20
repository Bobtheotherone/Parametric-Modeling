from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

from formula_foundry.coupongen.hashing import coupon_id_from_design_hash
from formula_foundry.coupongen.resolve import design_hash, resolve
from formula_foundry.coupongen.spec import CouponSpec
from formula_foundry.openems import OpenEMSRunner, SimulationRunner
from formula_foundry.openems.convert import build_simulation_spec
from formula_foundry.openems.geometry import GeometrySpec, build_geometry_spec
from formula_foundry.openems.spec import SimulationSpec
from formula_foundry.substrate import canonical_json_dumps, sha256_bytes


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR)


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


def _build_inputs() -> tuple[SimulationSpec, GeometrySpec]:
    spec = CouponSpec.model_validate(_example_spec_data())
    resolved = resolve(spec)
    manifest = _manifest_for(spec)
    geometry = build_geometry_spec(resolved, manifest)
    simulation = build_simulation_spec(resolved, manifest)
    return simulation, geometry


def test_sim_runner_stub_cache_and_manifest_hash(tmp_path: Path) -> None:
    simulation, geometry = _build_inputs()
    runner = SimulationRunner(mode="stub")
    output_dir = tmp_path / "run"

    result_first = runner.run(simulation, geometry, output_dir=output_dir)
    assert result_first.cache_hit is False
    assert result_first.manifest_path.exists()

    sparam_path = result_first.outputs_dir / "sparams.s2p"
    assert sparam_path.exists()
    content_first = sparam_path.read_text(encoding="utf-8")

    manifest_payload = json.loads(result_first.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["simulation_hash"] == result_first.simulation_hash
    expected_manifest_hash = sha256_bytes(canonical_json_dumps(manifest_payload).encode("utf-8"))
    assert result_first.manifest_hash == expected_manifest_hash

    rel_path = (Path(simulation.output.outputs_dir) / "sparams.s2p").as_posix()
    assert rel_path in result_first.output_hashes

    result_second = runner.run(simulation, geometry, output_dir=output_dir)
    assert result_second.cache_hit is True
    assert result_second.manifest_hash == result_first.manifest_hash
    content_second = sparam_path.read_text(encoding="utf-8")
    assert content_second == content_first


def test_sim_runner_cli_invokes_openems(tmp_path: Path) -> None:
    simulation, geometry = _build_inputs()
    simulation = simulation.model_copy(
        update={
            "output": simulation.output.model_copy(update={"outputs_dir": "cli_outputs/"}),
        }
    )

    stub = tmp_path / "openEMS"
    _write_executable(
        stub,
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf \"%s\\n\" \"$@\" > args.txt\n"
        "echo 'stub-touchstone' > sparams.s2p\n",
    )
    runner = SimulationRunner(
        mode="cli",
        openems_runner=OpenEMSRunner(mode="local", openems_bin=str(stub)),
    )
    result = runner.run(
        simulation,
        geometry,
        output_dir=tmp_path / "run_cli",
        openems_args=["--flag", "value"],
    )
    assert result.cache_hit is False
    args_path = result.outputs_dir / "args.txt"
    assert args_path.exists()
    args_text = args_path.read_text(encoding="utf-8")
    assert "--flag" in args_text
    assert "value" in args_text
    assert (result.outputs_dir / "sparams.s2p").exists()
