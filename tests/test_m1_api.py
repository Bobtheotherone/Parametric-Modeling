from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from formula_foundry.coupongen.api import export_fab, generate_kicad, load_spec, run_drc, validate_spec


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


class _FakeRunner:
    def run_drc(self, board_path: Path, report_path: Path) -> subprocess.CompletedProcess[str]:
        report_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return _completed_process()

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        (out_dir / "F.Cu.gbr").write_text("G04 test*", encoding="utf-8")
        return _completed_process()

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        (out_dir / "drill.drl").write_text("M48", encoding="utf-8")
        return _completed_process()


def _completed_process() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")


def test_python_api_contract(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(_example_spec_data()), encoding="utf-8")

    spec = load_spec(spec_path)
    evaluation = validate_spec(spec, out_dir=tmp_path)
    project = generate_kicad(evaluation.resolved, evaluation.spec, tmp_path)

    assert project.board_path.exists()
    assert (tmp_path / "resolved_design.json").exists()
    assert (tmp_path / "constraint_proof.json").exists()

    runner = _FakeRunner()
    report = run_drc(project.board_path, spec.toolchain.kicad, runner=runner)
    hashes = export_fab(project.board_path, tmp_path / "fab", spec.toolchain.kicad, runner=runner)

    assert report.report_path.exists()
    assert "gerbers/F.Cu.gbr" in hashes
    assert "drill/drill.drl" in hashes
