"""KiCad DRC tests for golden specs.

REQ-M1-025: CI must prove DRC-clean boards and export completeness for all
golden specs using the pinned KiCad toolchain.

This module tests that:
- All golden specs can run DRC with the pinned KiCad Docker image
- DRC invocation uses correct flags (severity-all, exit-code-violations, JSON output)
- Docker mode correctly mounts workdirs and uses digest-pinned images
- DRC reports are generated in the expected format

IMPORTANT: These tests use fake runners to avoid actually invoking KiCad
during CI. The real KiCad DRC is tested via integration tests that require
the KiCad Docker image.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.coupongen import (
    KicadCliRunner,
    build_drc_args,
    load_spec,
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SPECS_DIR = ROOT / "tests" / "golden_specs"


class _FakeDrcRunner:
    """Fake KiCad CLI runner that simulates DRC without invoking KiCad.

    This allows testing the DRC pipeline without requiring the KiCad Docker image.
    Returns configurable DRC results for testing success and failure paths.
    """

    def __init__(
        self,
        *,
        returncode: int = 0,
        violations: list[dict[str, Any]] | None = None,
    ) -> None:
        self.returncode = returncode
        self.violations = violations or []
        self.calls: list[tuple[Path, Path]] = []

    def run_drc(self, board_path: Path, report_path: Path) -> subprocess.CompletedProcess[str]:
        """Simulate DRC execution and write a fake report."""
        self.calls.append((board_path, report_path))

        report = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "source": str(board_path),
            "violations": self.violations,
            "unconnected_items": [],
            "schematic_parity": [],
            "coordinate_units": "mm",
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        return subprocess.CompletedProcess(
            args=["kicad-cli", "pcb", "drc"],
            returncode=self.returncode,
            stdout="",
            stderr="" if self.returncode == 0 else "DRC violations found",
        )

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        """Simulate Gerber export."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "F.Cu.gbr").write_text("G04 Fake Gerber*\nX0Y0D02*\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        """Simulate drill file export."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "drill.drl").write_text("M48\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")


def _golden_specs() -> list[Path]:
    """Collect all golden spec files."""
    patterns = ("*.json", "*.yaml", "*.yml")
    specs: list[Path] = []
    for pattern in patterns:
        specs.extend(sorted(GOLDEN_SPECS_DIR.glob(pattern)))
    return sorted(specs)


class TestKicadCliRunnerModes:
    """Tests for KiCad CLI runner mode configuration.

    REQ-M1-025: CI must prove DRC-clean boards using the pinned KiCad toolchain.
    """

    def test_local_mode_command_structure(self, tmp_path: Path) -> None:
        """Local mode should use kicad-cli directly."""
        runner = KicadCliRunner(mode="local")
        cmd = runner.build_command(["pcb", "drc", "board.kicad_pcb"], workdir=tmp_path)

        assert cmd[0] == "kicad-cli"
        assert "pcb" in cmd
        assert "drc" in cmd
        assert "board.kicad_pcb" in cmd

    def test_docker_mode_command_structure(self, tmp_path: Path) -> None:
        """REQ-M1-025: Docker mode should use pinned digest image."""
        docker_image = "kicad/kicad:9.0.7@sha256:abc123def456"
        runner = KicadCliRunner(mode="docker", docker_image=docker_image)
        cmd = runner.build_command(["pcb", "drc", "board.kicad_pcb"], workdir=tmp_path)

        assert cmd[0] == "docker"
        assert "run" in cmd
        assert "--rm" in cmd
        assert docker_image in cmd
        assert "-v" in cmd
        # Verify workdir mount
        mount_idx = cmd.index("-v")
        mount_arg = cmd[mount_idx + 1]
        assert str(tmp_path) in mount_arg
        assert "/workspace" in mount_arg

    def test_docker_mode_requires_image(self, tmp_path: Path) -> None:
        """Docker mode must require a docker_image."""
        runner = KicadCliRunner(mode="docker", docker_image=None)
        with pytest.raises(ValueError, match="docker_image is required"):
            runner.build_command(["pcb", "drc"], workdir=tmp_path)


class TestBuildDrcArgs:
    """Tests for DRC argument construction.

    REQ-M1-025: CI must prove DRC-clean boards using correct invocation flags.
    """

    def test_drc_args_default_severity_all(self, tmp_path: Path) -> None:
        """DRC should default to all severity for M1.

        Per REQ-M1-016 and DESIGN_DOCUMENT.md: M1 uses --severity-all to catch
        all DRC violations including warnings for thorough quality checks.
        """
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report)

        assert "--severity-all" in args

    def test_drc_args_severity_all_option(self, tmp_path: Path) -> None:
        """DRC should support all severity levels when requested."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report, severity="all")

        assert "--severity-all" in args

    def test_drc_args_severity_warning_option(self, tmp_path: Path) -> None:
        """DRC should support warning severity level."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report, severity="warning")

        assert "--severity-warning" in args

    def test_drc_args_include_exit_code_violations(self, tmp_path: Path) -> None:
        """DRC should return non-zero exit on violations."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report)

        assert "--exit-code-violations" in args

    def test_drc_args_use_json_format(self, tmp_path: Path) -> None:
        """DRC should output JSON format for machine parsing."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report)

        assert "--format" in args
        format_idx = args.index("--format")
        assert args[format_idx + 1] == "json"

    def test_drc_args_specify_output_path(self, tmp_path: Path) -> None:
        """DRC should write report to specified path."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report)

        assert "--output" in args
        output_idx = args.index("--output")
        assert args[output_idx + 1] == str(report)


class TestGoldenSpecsDrcCompatibility:
    """Tests verifying golden specs are compatible with DRC toolchain.

    REQ-M1-025: CI must prove DRC-clean boards and export completeness for all
    golden specs using the pinned KiCad toolchain.
    """

    def test_golden_specs_exist(self) -> None:
        """Verify golden specs directory contains expected specs."""
        specs = _golden_specs()
        assert len(specs) >= 20, f"Expected at least 20 golden specs, found {len(specs)}"

    def test_golden_specs_specify_pinned_docker_image(self) -> None:
        """REQ-M1-025: All golden specs must use digest-pinned Docker images."""
        specs = _golden_specs()
        for spec_path in specs:
            spec = load_spec(spec_path)
            docker_image = spec.toolchain.kicad.docker_image
            assert "@sha256:" in docker_image, (
                f"Golden spec {spec_path.name} must use digest-pinned Docker image, got: {docker_image}"
            )

    def test_golden_specs_kicad_version_pinned(self) -> None:
        """REQ-M1-025: All golden specs must pin KiCad version."""
        specs = _golden_specs()
        for spec_path in specs:
            spec = load_spec(spec_path)
            version = spec.toolchain.kicad.version
            assert version, f"Golden spec {spec_path.name} must specify KiCad version"
            # Version should be a semantic version string
            parts = version.split(".")
            assert len(parts) >= 2, f"Invalid version format: {version}"

    @pytest.mark.parametrize("spec_path", _golden_specs(), ids=lambda p: p.name)
    def test_golden_spec_drc_clean_with_fake_runner(self, spec_path: Path, tmp_path: Path) -> None:
        """REQ-M1-025: Each golden spec should be DRC-clean.

        This test uses a fake runner to verify the DRC pipeline works correctly.
        Real KiCad DRC is tested separately with Docker.
        """
        from formula_foundry.coupongen import build_coupon

        spec = load_spec(spec_path)
        runner = _FakeDrcRunner(returncode=0)

        result = build_coupon(
            spec,
            out_root=tmp_path,
            mode="docker",
            runner=runner,
            kicad_cli_version="9.0.7",
        )

        # Verify DRC was called
        assert len(runner.calls) == 1
        board_path, report_path = runner.calls[0]
        assert board_path.suffix == ".kicad_pcb"
        assert report_path.suffix == ".json"

        # Verify manifest records DRC success
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest["verification"]["drc"]["returncode"] == 0


class TestDrcFailureHandling:
    """Tests for DRC failure scenarios."""

    def test_drc_failure_raises_when_must_pass(self, tmp_path: Path) -> None:
        """DRC failure should raise when constraints.drc.must_pass is True."""
        from formula_foundry.coupongen import build_coupon
        from formula_foundry.coupongen.spec import CouponSpec

        spec_data = {
            "schema_version": 1,
            "coupon_family": "F0_CAL_THRU_LINE",
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
                "w_nm": 250000,
                "gap_nm": 180000,
                "length_left_nm": 24000000,
                "length_right_nm": 24000000,
                "ground_via_fence": None,
            },
            "discontinuity": None,
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
        spec = CouponSpec.model_validate(spec_data)
        runner = _FakeDrcRunner(
            returncode=1,
            violations=[{"type": "clearance", "severity": "error", "description": "Test violation"}],
        )

        with pytest.raises(RuntimeError, match="KiCad DRC failed"):
            build_coupon(
                spec,
                out_root=tmp_path,
                mode="docker",
                runner=runner,
                kicad_cli_version="9.0.7",
            )

    def test_drc_report_json_structure(self, tmp_path: Path) -> None:
        """DRC report should have expected JSON structure."""
        runner = _FakeDrcRunner(returncode=0)
        board_path = tmp_path / "test.kicad_pcb"
        report_path = tmp_path / "drc.json"
        board_path.write_text("(kicad_pcb)", encoding="utf-8")

        runner.run_drc(board_path, report_path)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert "violations" in report
        assert "unconnected_items" in report
        assert isinstance(report["violations"], list)
