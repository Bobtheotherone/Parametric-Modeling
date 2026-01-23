from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import jsonschema

from formula_foundry.coupongen.api import build_coupon
from formula_foundry.coupongen.hashing import canonical_hash_export_text
from formula_foundry.coupongen.manifest import toolchain_hash
from formula_foundry.coupongen.spec import CouponSpec

_SCHEMA_PATH = Path(__file__).parent.parent / "coupongen" / "schemas" / "manifest.schema.json"


class _FakeRunner:
    def run_drc(self, board_path: Path, report_path: Path) -> subprocess.CompletedProcess[str]:
        report_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return _completed_process()

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        (out_dir / "F.Cu.gbr").write_text("G04 Created*\nX0Y0D02*\n", encoding="utf-8")
        return _completed_process()

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        (out_dir / "drill.drl").write_text("M48\n", encoding="utf-8")
        return _completed_process()


def _completed_process() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")


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


def test_manifest_required_fields(tmp_path: Path) -> None:
    spec = CouponSpec.model_validate(_example_spec_data())
    result = build_coupon(
        spec,
        out_root=tmp_path,
        mode="docker",
        runner=_FakeRunner(),
        kicad_cli_version="9.0.7",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "coupon_family",
        "design_hash",
        "coupon_id",
        "resolved_design",
        "derived_features",
        "dimensionless_groups",
        "fab_profile",
        "stackup",
        "toolchain",
        "toolchain_hash",
        "exports",
        "verification",
        "lineage",
    }
    assert required <= set(manifest.keys())
    assert manifest["design_hash"] == result.design_hash
    assert manifest["coupon_id"] == result.coupon_id
    assert manifest["toolchain"]["docker"]["image_ref"].startswith("kicad/kicad:9.0.7@sha256:")
    assert manifest["toolchain"]["kicad"]["version"] == "9.0.7"
    assert manifest["toolchain"]["mode"] == "docker"
    assert manifest["toolchain"]["kicad"]["cli_version_output"] == "9.0.7"
    assert manifest["toolchain_hash"] == toolchain_hash(manifest["toolchain"])

    # Verify DRC summary is present per Section 13.5.1
    assert "summary" in manifest["verification"]["drc"]
    assert "canonical_hash" in manifest["verification"]["drc"]
    drc_summary = manifest["verification"]["drc"]["summary"]
    assert "violations" in drc_summary
    assert "warnings" in drc_summary
    assert "exclusions" in drc_summary

    exports = {entry["path"]: entry["hash"] for entry in manifest["exports"]}
    # Export paths include fab/ prefix (files are under output_dir/fab/)
    expected_exports = {
        "fab/gerbers/F.Cu.gbr": canonical_hash_export_text("G04 Created*\nX0Y0D02*\n"),
        "fab/drill/drill.drl": canonical_hash_export_text("M48\n"),
    }
    assert exports == expected_exports


def test_outputs_keyed_by_design_hash(tmp_path: Path) -> None:
    spec = CouponSpec.model_validate(_example_spec_data())
    result = build_coupon(
        spec,
        out_root=tmp_path,
        mode="docker",
        runner=_FakeRunner(),
        kicad_cli_version="9.0.7",
    )

    folder_name = result.output_dir.name
    assert result.design_hash in folder_name
    assert result.coupon_id in folder_name


def test_manifest_validates_against_schema(tmp_path: Path) -> None:
    """Verify that generated manifests validate against manifest.schema.json."""
    spec = CouponSpec.model_validate(_example_spec_data())
    result = build_coupon(
        spec,
        out_root=tmp_path,
        mode="docker",
        runner=_FakeRunner(),
        kicad_cli_version="9.0.7",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))

    # Should not raise ValidationError
    jsonschema.validate(instance=manifest, schema=schema)
