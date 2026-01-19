from __future__ import annotations

import subprocess
from pathlib import Path

from formula_foundry.coupongen.api import export_fab
from formula_foundry.coupongen.spec import KicadToolchain


class _FakeRunner:
    def run_drc(self, board_path: Path, report_path: Path) -> subprocess.CompletedProcess[str]:
        report_path.write_text("{}", encoding="utf-8")
        return _completed_process()

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        (out_dir / "F.Cu.gbr").write_text("G04 Created by fake*\nX0Y0D02*\n", encoding="utf-8")
        return _completed_process()

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        (out_dir / "drill.drl").write_text("M48\n", encoding="utf-8")
        return _completed_process()


def _completed_process() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")


def test_export_outputs_and_hashes(tmp_path: Path) -> None:
    board_path = tmp_path / "coupon.kicad_pcb"
    board_path.write_text("(kicad_pcb)", encoding="utf-8")
    toolchain = KicadToolchain(version="9.0.7", docker_image="kicad/kicad:9.0.7@sha256:deadbeef")

    hashes = export_fab(
        board_path,
        tmp_path / "fab",
        toolchain,
        runner=_FakeRunner(),
    )

    assert "gerbers/F.Cu.gbr" in hashes
    assert "drill/drill.drl" in hashes
    for digest in hashes.values():
        assert len(digest) == 64
