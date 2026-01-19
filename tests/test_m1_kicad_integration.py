from __future__ import annotations

import json
import subprocess
from pathlib import Path

from formula_foundry.coupongen import build_coupon, load_spec

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SPECS_DIR = ROOT / "tests" / "golden_specs"


class _FakeRunner:
    def run_drc(self, board_path: Path, report_path: Path) -> subprocess.CompletedProcess[str]:
        report_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return _completed_process()

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        (out_dir / "F.Cu.gbr").write_text("G04 Export*\nX0Y0D02*\n", encoding="utf-8")
        return _completed_process()

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        (out_dir / "drill.drl").write_text("M48\n", encoding="utf-8")
        return _completed_process()


def _completed_process() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")


def _golden_specs() -> list[Path]:
    patterns = ("*.json", "*.yaml", "*.yml")
    specs: list[Path] = []
    for pattern in patterns:
        specs.extend(sorted(GOLDEN_SPECS_DIR.glob(pattern)))
    return sorted(specs)


def test_drc_clean_and_exports_in_pinned_toolchain(tmp_path: Path) -> None:
    runner = _FakeRunner()
    for spec_path in _golden_specs():
        spec = load_spec(spec_path)
        assert "@sha256:" in spec.toolchain.kicad.docker_image
        result = build_coupon(
            spec,
            out_root=tmp_path,
            mode="docker",
            runner=runner,
            kicad_cli_version="9.0.7",
        )
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest["toolchain"]["docker_image"] == spec.toolchain.kicad.docker_image
        assert manifest["verification"]["drc"]["returncode"] == 0
        assert manifest["exports"]
