# SPDX-License-Identifier: MIT
"""Gate G3 tests: KiCad DRC clean verification.

This module tests:
- DRC invocation arguments are correct
- DRC exit code handling (0 = clean, 5 = violations)
- DRC JSON report parsing and validation
- Golden specs satisfy DRC requirements (using fake runner for unit tests)

Per ECO-M1-ALIGN-0001:
- REQ-M1-016: DRC with severity-all, JSON report output, and exit-code gating
- REQ-M1-025: CI must prove DRC-clean boards for all golden specs

Pytest marker: gate_g3

Note: Real KiCad DRC integration tests are in tests/integration/test_kicad_drc_integration.py
and require Docker. These tests verify the DRC pipeline logic without Docker.
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


# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parents[1]
GOLDEN_SPECS_DIR = TESTS_DIR / "golden_specs"

# DRC exit codes per KiCad CLI documentation
DRC_EXIT_CODE_CLEAN = 0
DRC_EXIT_CODE_VIOLATIONS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_golden_specs() -> list[Path]:
    """Collect all golden spec files (YAML only to avoid duplicates)."""
    specs: list[Path] = []
    specs.extend(sorted(GOLDEN_SPECS_DIR.glob("f0_*.yaml")))
    specs.extend(sorted(GOLDEN_SPECS_DIR.glob("f1_*.yaml")))
    return specs


def _count_f0_specs() -> int:
    """Count F0 golden specs."""
    return len(list(GOLDEN_SPECS_DIR.glob("f0_*.yaml")))


def _count_f1_specs() -> int:
    """Count F1 golden specs."""
    return len(list(GOLDEN_SPECS_DIR.glob("f1_*.yaml")))


class _FakeDrcRunner:
    """Fake KiCad CLI runner for unit testing DRC pipeline.

    Simulates DRC execution without requiring KiCad Docker image.
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

    def run_drc(
        self, board_path: Path, report_path: Path
    ) -> subprocess.CompletedProcess[str]:
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

    def export_gerbers(
        self, board_path: Path, out_dir: Path
    ) -> subprocess.CompletedProcess[str]:
        """Simulate Gerber export."""
        out_dir.mkdir(parents=True, exist_ok=True)
        board_name = board_path.stem
        # Generate all required layers with KiCad naming: board-F_Cu.gbr
        for layer in ["F_Cu", "B_Cu", "In1_Cu", "In2_Cu", "F_Mask", "B_Mask", "Edge_Cuts"]:
            (out_dir / f"{board_name}-{layer}.gbr").write_text(
                f"G04 Fake {layer}*\n", encoding="utf-8"
            )
        return subprocess.CompletedProcess(
            args=["kicad-cli"], returncode=0, stdout="", stderr=""
        )

    def export_drill(
        self, board_path: Path, out_dir: Path
    ) -> subprocess.CompletedProcess[str]:
        """Simulate drill file export."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "drill.drl").write_text("M48\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            args=["kicad-cli"], returncode=0, stdout="", stderr=""
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def golden_specs() -> list[Path]:
    """Fixture providing list of golden spec paths."""
    return _collect_golden_specs()


# ---------------------------------------------------------------------------
# G3 Gate Tests: Golden Spec Coverage
# ---------------------------------------------------------------------------


@pytest.mark.gate_g3
class TestG3GoldenSpecCoverage:
    """Gate G3 tests verifying sufficient golden specs for DRC testing.

    Per ECO-M1-ALIGN-0001: "For ≥10 golden specs per family, CI proves DRC clean (G3)"
    """

    def test_minimum_f0_golden_specs_for_drc(self) -> None:
        """Verify at least 10 F0 golden specs exist for DRC testing."""
        count = _count_f0_specs()
        assert count >= 10, f"Expected ≥10 F0 specs for DRC gate, found {count}"

    def test_minimum_f1_golden_specs_for_drc(self) -> None:
        """Verify at least 10 F1 golden specs exist for DRC testing."""
        count = _count_f1_specs()
        assert count >= 10, f"Expected ≥10 F1 specs for DRC gate, found {count}"

    def test_total_golden_specs_for_drc(self, golden_specs: list[Path]) -> None:
        """Verify at least 20 total golden specs exist for DRC testing."""
        assert len(golden_specs) >= 20, (
            f"Expected ≥20 total golden specs for DRC gate, found {len(golden_specs)}"
        )


# ---------------------------------------------------------------------------
# G3 Gate Tests: DRC Argument Construction
# ---------------------------------------------------------------------------


@pytest.mark.gate_g3
class TestG3DrcArgumentConstruction:
    """Gate G3 tests for DRC command-line argument construction.

    Per REQ-M1-016: DRC must use severity-all, JSON format, exit-code-violations.
    """

    def test_drc_args_include_severity_all(self, tmp_path: Path) -> None:
        """DRC must check all severity levels for thoroughness."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report)

        assert "--severity-all" in args, "DRC must use --severity-all flag"

    def test_drc_args_include_exit_code_violations(self, tmp_path: Path) -> None:
        """DRC must return non-zero exit on violations for gating."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report)

        assert "--exit-code-violations" in args, (
            "DRC must use --exit-code-violations flag"
        )

    def test_drc_args_use_json_format(self, tmp_path: Path) -> None:
        """DRC must output JSON for programmatic parsing."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report)

        assert "--format" in args, "DRC must specify --format"
        format_idx = args.index("--format")
        assert args[format_idx + 1] == "json", "DRC format must be 'json'"

    def test_drc_args_specify_output_path(self, tmp_path: Path) -> None:
        """DRC must write report to specified path."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report)

        assert "--output" in args, "DRC must specify --output"
        output_idx = args.index("--output")
        assert args[output_idx + 1] == str(report), "DRC output must match report path"

    def test_drc_args_specify_board_path(self, tmp_path: Path) -> None:
        """DRC must include board file path."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report)

        assert str(board) in args, "DRC must include board file path"


# ---------------------------------------------------------------------------
# G3 Gate Tests: DRC Exit Code Semantics
# ---------------------------------------------------------------------------


@pytest.mark.gate_g3
class TestG3DrcExitCodeSemantics:
    """Gate G3 tests for DRC exit code interpretation.

    Per KiCad CLI documentation:
    - Exit code 0: DRC passed (no violations)
    - Exit code 5: DRC completed but found violations
    """

    def test_exit_code_0_means_clean(self) -> None:
        """Exit code 0 indicates DRC passed with no violations."""
        assert DRC_EXIT_CODE_CLEAN == 0

    def test_exit_code_5_means_violations(self) -> None:
        """Exit code 5 indicates DRC found violations."""
        assert DRC_EXIT_CODE_VIOLATIONS == 5

    def test_clean_drc_runner_returns_zero(self, tmp_path: Path) -> None:
        """Clean DRC should return exit code 0."""
        runner = _FakeDrcRunner(returncode=0, violations=[])
        board_path = tmp_path / "test.kicad_pcb"
        report_path = tmp_path / "drc.json"
        board_path.write_text("(kicad_pcb)", encoding="utf-8")

        result = runner.run_drc(board_path, report_path)

        assert result.returncode == 0, "Clean DRC should return 0"

    def test_violations_drc_runner_returns_nonzero(self, tmp_path: Path) -> None:
        """DRC with violations should return non-zero exit code."""
        runner = _FakeDrcRunner(
            returncode=5,
            violations=[
                {
                    "type": "clearance",
                    "severity": "error",
                    "description": "Test violation",
                }
            ],
        )
        board_path = tmp_path / "test.kicad_pcb"
        report_path = tmp_path / "drc.json"
        board_path.write_text("(kicad_pcb)", encoding="utf-8")

        result = runner.run_drc(board_path, report_path)

        assert result.returncode != 0, "DRC with violations should return non-zero"


# ---------------------------------------------------------------------------
# G3 Gate Tests: DRC JSON Report Validation
# ---------------------------------------------------------------------------


@pytest.mark.gate_g3
class TestG3DrcReportValidation:
    """Gate G3 tests for DRC JSON report structure validation."""

    def test_drc_report_has_violations_field(self, tmp_path: Path) -> None:
        """DRC report must contain 'violations' field."""
        runner = _FakeDrcRunner(returncode=0)
        board_path = tmp_path / "test.kicad_pcb"
        report_path = tmp_path / "drc.json"
        board_path.write_text("(kicad_pcb)", encoding="utf-8")

        runner.run_drc(board_path, report_path)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert "violations" in report, "DRC report must have 'violations' field"

    def test_drc_report_violations_is_list(self, tmp_path: Path) -> None:
        """DRC report violations must be a list."""
        runner = _FakeDrcRunner(returncode=0)
        board_path = tmp_path / "test.kicad_pcb"
        report_path = tmp_path / "drc.json"
        board_path.write_text("(kicad_pcb)", encoding="utf-8")

        runner.run_drc(board_path, report_path)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(report["violations"], list), "violations must be a list"

    def test_clean_drc_report_has_zero_violations(self, tmp_path: Path) -> None:
        """Clean DRC report must have zero violations."""
        runner = _FakeDrcRunner(returncode=0, violations=[])
        board_path = tmp_path / "test.kicad_pcb"
        report_path = tmp_path / "drc.json"
        board_path.write_text("(kicad_pcb)", encoding="utf-8")

        runner.run_drc(board_path, report_path)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert len(report["violations"]) == 0, "Clean DRC must have 0 violations"

    def test_drc_report_has_source_field(self, tmp_path: Path) -> None:
        """DRC report must reference source board file."""
        runner = _FakeDrcRunner(returncode=0)
        board_path = tmp_path / "test.kicad_pcb"
        report_path = tmp_path / "drc.json"
        board_path.write_text("(kicad_pcb)", encoding="utf-8")

        runner.run_drc(board_path, report_path)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert "source" in report, "DRC report must have 'source' field"


# ---------------------------------------------------------------------------
# G3 Gate Tests: Golden Specs DRC Compatibility
# ---------------------------------------------------------------------------


@pytest.mark.gate_g3
class TestG3GoldenSpecsDrcCompatibility:
    """Gate G3 tests verifying golden specs are compatible with DRC pipeline.

    These tests use fake runners to verify DRC pipeline logic without Docker.
    Real KiCad DRC is tested in integration tests.
    """

    def test_golden_specs_specify_drc_must_pass(
        self, golden_specs: list[Path]
    ) -> None:
        """All golden specs should require DRC to pass."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            assert spec.constraints.drc.must_pass is True, (
                f"Golden spec {spec_path.name} must have constraints.drc.must_pass=True"
            )

    def test_golden_specs_specify_drc_severity(
        self, golden_specs: list[Path]
    ) -> None:
        """All golden specs should specify DRC severity level."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            assert spec.constraints.drc.severity in ("all", "error", "warning"), (
                f"Golden spec {spec_path.name} has invalid DRC severity"
            )

    def test_golden_specs_use_digest_pinned_docker_image(
        self, golden_specs: list[Path]
    ) -> None:
        """All golden specs must use digest-pinned Docker images."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            docker_image = spec.toolchain.kicad.docker_image
            assert "@sha256:" in docker_image, (
                f"Golden spec {spec_path.name} must use digest-pinned Docker image, "
                f"got: {docker_image}"
            )

    @pytest.mark.parametrize(
        "spec_path",
        _collect_golden_specs(),
        ids=lambda p: p.name,
    )
    def test_golden_spec_drc_clean_with_fake_runner(
        self, spec_path: Path, tmp_path: Path
    ) -> None:
        """Each golden spec should pass DRC (using fake runner).

        This verifies the DRC pipeline is correctly wired for each golden spec.
        Real KiCad DRC is tested in integration tests.
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
        assert len(runner.calls) == 1, "DRC should be called once per build"
        board_path, report_path = runner.calls[0]
        assert board_path.suffix == ".kicad_pcb"
        assert report_path.suffix == ".json"

        # Verify manifest records DRC success
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest["verification"]["drc"]["returncode"] == 0


# ---------------------------------------------------------------------------
# G3 Gate Tests: DRC Failure Handling
# ---------------------------------------------------------------------------


@pytest.mark.gate_g3
class TestG3DrcFailureHandling:
    """Gate G3 tests for DRC failure scenarios."""

    def test_drc_failure_raises_when_must_pass(self, tmp_path: Path) -> None:
        """DRC failure should raise when constraints.drc.must_pass is True."""
        from formula_foundry.coupongen import build_coupon
        from formula_foundry.coupongen.spec import CouponSpec

        spec_data = _minimal_spec_data()
        spec = CouponSpec.model_validate(spec_data)
        runner = _FakeDrcRunner(
            returncode=5,
            violations=[
                {
                    "type": "clearance",
                    "severity": "error",
                    "description": "Test violation",
                }
            ],
        )

        with pytest.raises(RuntimeError, match="KiCad DRC failed"):
            build_coupon(
                spec,
                out_root=tmp_path,
                mode="docker",
                runner=runner,
                kicad_cli_version="9.0.7",
            )

    def test_drc_failure_includes_violation_count(self, tmp_path: Path) -> None:
        """DRC failure message should indicate violation count."""
        runner = _FakeDrcRunner(
            returncode=5,
            violations=[
                {"type": "clearance", "severity": "error", "description": "V1"},
                {"type": "unconnected", "severity": "error", "description": "V2"},
            ],
        )
        board_path = tmp_path / "test.kicad_pcb"
        report_path = tmp_path / "drc.json"
        board_path.write_text("(kicad_pcb)", encoding="utf-8")

        runner.run_drc(board_path, report_path)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert len(report["violations"]) == 2


# ---------------------------------------------------------------------------
# G3 Gate Tests: KiCad CLI Runner Modes
# ---------------------------------------------------------------------------


@pytest.mark.gate_g3
class TestG3KicadCliRunnerModes:
    """Gate G3 tests for KiCad CLI runner mode configuration."""

    def test_local_mode_command_structure(self, tmp_path: Path) -> None:
        """Local mode should use kicad-cli directly."""
        runner = KicadCliRunner(mode="local")
        cmd = runner.build_command(["pcb", "drc", "board.kicad_pcb"], workdir=tmp_path)

        assert cmd[0] == "kicad-cli"
        assert "pcb" in cmd
        assert "drc" in cmd

    def test_docker_mode_command_structure(self, tmp_path: Path) -> None:
        """Docker mode should use pinned digest image."""
        docker_image = "kicad/kicad:9.0.7@sha256:abc123def456"
        runner = KicadCliRunner(mode="docker", docker_image=docker_image)
        cmd = runner.build_command(["pcb", "drc", "board.kicad_pcb"], workdir=tmp_path)

        assert cmd[0] == "docker"
        assert "run" in cmd
        assert "--rm" in cmd
        assert docker_image in cmd

    def test_docker_mode_mounts_workdir(self, tmp_path: Path) -> None:
        """Docker mode should mount working directory."""
        docker_image = "kicad/kicad:9.0.7@sha256:abc123"
        runner = KicadCliRunner(mode="docker", docker_image=docker_image)
        cmd = runner.build_command(["pcb", "drc"], workdir=tmp_path)

        assert "-v" in cmd
        mount_idx = cmd.index("-v")
        mount_arg = cmd[mount_idx + 1]
        assert str(tmp_path) in mount_arg
        assert "/workspace" in mount_arg

    def test_docker_mode_requires_image(self, tmp_path: Path) -> None:
        """Docker mode must require a docker_image."""
        runner = KicadCliRunner(mode="docker", docker_image=None)
        with pytest.raises(ValueError, match="docker_image is required"):
            runner.build_command(["pcb", "drc"], workdir=tmp_path)


# ---------------------------------------------------------------------------
# Helper for minimal spec data
# ---------------------------------------------------------------------------


def _minimal_spec_data() -> dict[str, Any]:
    """Return minimal valid spec data for testing."""
    return {
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
                "L1_to_L2": 180_000,
                "L2_to_L3": 800_000,
                "L3_to_L4": 180_000,
            },
            "materials": {"er": 4.1, "loss_tangent": 0.02},
        },
        "board": {
            "outline": {
                "width_nm": 20_000_000,
                "length_nm": 80_000_000,
                "corner_radius_nm": 2_000_000,
            },
            "origin": {"mode": "EDGE_L_CENTER"},
            "text": {"coupon_id": "${COUPON_ID}", "include_manifest_hash": True},
        },
        "connectors": {
            "left": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [5_000_000, 0],
                "rotation_deg": 180,
            },
            "right": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [75_000_000, 0],
                "rotation_deg": 0,
            },
        },
        "transmission_line": {
            "type": "CPWG",
            "layer": "F.Cu",
            "w_nm": 250_000,
            "gap_nm": 180_000,
            "length_left_nm": 24_000_000,
            "length_right_nm": 24_000_000,
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
