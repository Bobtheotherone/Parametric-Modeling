from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from formula_foundry.coupongen.api import build_coupon
from formula_foundry.coupongen.spec import CouponSpec


class _CountingRunner:
    def __init__(self) -> None:
        self.drc_calls = 0
        self.gerber_calls = 0
        self.drill_calls = 0

    def run_drc(self, board_path: Path, report_path: Path) -> subprocess.CompletedProcess[str]:
        self.drc_calls += 1
        report_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return _completed_process()

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        self.gerber_calls += 1
        (out_dir / "F.Cu.gbr").write_text("G04 Cached*\nX0Y0D02*\n", encoding="utf-8")
        return _completed_process()

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        self.drill_calls += 1
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


def test_cache_toolchain_hash_behavior(tmp_path: Path) -> None:
    runner = _CountingRunner()
    spec = CouponSpec.model_validate(_example_spec_data())

    result_a = build_coupon(spec, out_root=tmp_path, runner=runner)
    assert result_a.cache_hit is False
    assert runner.drc_calls == 1
    assert runner.gerber_calls == 1
    assert runner.drill_calls == 1

    result_b = build_coupon(spec, out_root=tmp_path, runner=runner)
    assert result_b.cache_hit is True
    assert runner.drc_calls == 1
    assert runner.gerber_calls == 1
    assert runner.drill_calls == 1

    modified = _example_spec_data()
    modified["toolchain"]["kicad"]["docker_image"] = "kicad/kicad:9.0.7@sha256:feedbeef"
    spec_modified = CouponSpec.model_validate(modified)
    result_c = build_coupon(spec_modified, out_root=tmp_path, runner=runner)
    assert result_c.cache_hit is False
    assert runner.drc_calls == 2
    assert runner.gerber_calls == 2
    assert runner.drill_calls == 2
