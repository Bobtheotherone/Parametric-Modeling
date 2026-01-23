from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from formula_foundry.coupongen.api import export_fab
from formula_foundry.coupongen.export import KicadExportError
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


class _FailingGerberRunner:
    """Runner that fails on Gerber export (simulates permission error)."""

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        # Return non-zero exit code without writing files
        return subprocess.CompletedProcess(
            args=["kicad-cli", "pcb", "export", "gerbers"],
            returncode=1,
            stdout="",
            stderr="Error: Permission denied writing to /workspace/fab/gerbers",
        )

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        (out_dir / "drill.drl").write_text("M48\n", encoding="utf-8")
        return _completed_process()


class _SilentFailGerberRunner:
    """Runner that returns success but doesn't write files (simulates permission issue)."""

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        # Return success but don't write any files (simulates silent permission failure)
        return _completed_process()

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        (out_dir / "drill.drl").write_text("M48\n", encoding="utf-8")
        return _completed_process()


class _FailingDrillRunner:
    """Runner that fails on drill export."""

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        (out_dir / "F.Cu.gbr").write_text("G04 Created by fake*\nX0Y0D02*\n", encoding="utf-8")
        return _completed_process()

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        # Return non-zero exit code without writing files
        return subprocess.CompletedProcess(
            args=["kicad-cli", "pcb", "export", "drill"],
            returncode=1,
            stdout="",
            stderr="Error: Permission denied writing to /workspace/fab/drill",
        )


class _SilentFailDrillRunner:
    """Runner that returns success on drill but doesn't write files."""

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        (out_dir / "F.Cu.gbr").write_text("G04 Created by fake*\nX0Y0D02*\n", encoding="utf-8")
        return _completed_process()

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        # Return success but don't write any files
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


# ============================================================================
# Tests for fail-fast behavior on export failures
# ============================================================================


def test_export_fails_fast_on_gerber_error(tmp_path: Path) -> None:
    """Export must raise KicadExportError on Gerber export failure.

    This tests the fail-fast behavior when kicad-cli returns non-zero.
    Previously, the code would silently continue and produce empty exports.
    """
    board_path = tmp_path / "coupon.kicad_pcb"
    board_path.write_text("(kicad_pcb)", encoding="utf-8")
    toolchain = KicadToolchain(version="9.0.7", docker_image="kicad/kicad:9.0.7@sha256:deadbeef")

    with pytest.raises(KicadExportError) as exc_info:
        export_fab(
            board_path,
            tmp_path / "fab",
            toolchain,
            runner=_FailingGerberRunner(),
        )

    # Verify error message contains useful diagnostic info
    error_msg = str(exc_info.value)
    assert "Gerber" in error_msg
    assert "Permission denied" in error_msg or "failed" in error_msg.lower()


def test_export_fails_fast_on_silent_gerber_failure(tmp_path: Path) -> None:
    """Export must detect when Gerbers weren't written despite success code.

    This catches the case where the container user can't write to the
    bind-mounted directory but kicad-cli doesn't report an error.
    """
    board_path = tmp_path / "coupon.kicad_pcb"
    board_path.write_text("(kicad_pcb)", encoding="utf-8")
    toolchain = KicadToolchain(version="9.0.7", docker_image="kicad/kicad:9.0.7@sha256:deadbeef")

    with pytest.raises(KicadExportError) as exc_info:
        export_fab(
            board_path,
            tmp_path / "fab",
            toolchain,
            runner=_SilentFailGerberRunner(),
        )

    # Verify error message mentions missing files
    error_msg = str(exc_info.value)
    assert "no gerber files" in error_msg.lower() or "were created" in error_msg.lower()
    assert "permission" in error_msg.lower()


def test_export_fails_fast_on_drill_error(tmp_path: Path) -> None:
    """Export must raise KicadExportError on drill export failure."""
    board_path = tmp_path / "coupon.kicad_pcb"
    board_path.write_text("(kicad_pcb)", encoding="utf-8")
    toolchain = KicadToolchain(version="9.0.7", docker_image="kicad/kicad:9.0.7@sha256:deadbeef")

    with pytest.raises(KicadExportError) as exc_info:
        export_fab(
            board_path,
            tmp_path / "fab",
            toolchain,
            runner=_FailingDrillRunner(),
        )

    # Verify error message contains useful diagnostic info
    error_msg = str(exc_info.value)
    assert "drill" in error_msg.lower()


def test_export_fails_fast_on_silent_drill_failure(tmp_path: Path) -> None:
    """Export must detect when drill files weren't written despite success code."""
    board_path = tmp_path / "coupon.kicad_pcb"
    board_path.write_text("(kicad_pcb)", encoding="utf-8")
    toolchain = KicadToolchain(version="9.0.7", docker_image="kicad/kicad:9.0.7@sha256:deadbeef")

    with pytest.raises(KicadExportError) as exc_info:
        export_fab(
            board_path,
            tmp_path / "fab",
            toolchain,
            runner=_SilentFailDrillRunner(),
        )

    # Verify error message mentions missing files
    error_msg = str(exc_info.value)
    assert "no drill files" in error_msg.lower() or "not created" in error_msg.lower()


def test_kicad_export_error_attributes() -> None:
    """KicadExportError must include diagnostic information."""
    error = KicadExportError(
        message="Test error",
        command=["kicad-cli", "pcb", "export", "gerbers"],
        returncode=1,
        stdout="output",
        stderr="error details",
    )

    assert error.command == ["kicad-cli", "pcb", "export", "gerbers"]
    assert error.returncode == 1
    assert error.stdout == "output"
    assert error.stderr == "error details"

    # Verify all info is in the error message
    error_str = str(error)
    assert "Test error" in error_str
    assert "Exit code: 1" in error_str
    assert "stdout: output" in error_str
    assert "stderr: error details" in error_str
