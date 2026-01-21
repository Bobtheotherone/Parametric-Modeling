"""Tests for CP-3.5 Pipeline Integration.

This module tests the integration of ConstraintEngine into the build pipeline:
- validate_spec_with_engine() uses ConstraintEngine (Tier 0-3)
- build_coupon_with_engine() uses ConstraintEngine for validation
- CLI validate command uses ConstraintEngine by default
- CLI build command uses ConstraintEngine by default
- Build flow: validate/repair (Tier0-3) -> generate -> DRC -> export

CP-3.5: Wire ConstraintEngine into build pipeline
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml  # type: ignore[import-untyped]

from formula_foundry.coupongen import cli_main
from formula_foundry.coupongen.api import (
    BuildResult,
    ValidationResult,
    build_coupon_with_engine,
    load_spec,
    validate_spec_with_engine,
)
from formula_foundry.coupongen.constraints.tiers import ConstraintViolationError
from formula_foundry.coupongen.spec import CouponSpec


def _example_spec_data() -> dict[str, Any]:
    """Return a valid example CouponSpec for testing."""
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
            # CP-3.3: For F1 continuity, length_right must equal:
            #   right_x - (left_x + length_left) = 75M - (5M + 35M) = 35M
            # Using symmetric lengths with total span = 70M
            "length_left_nm": 35000000,
            "length_right_nm": 35000000,
            "ground_via_fence": {
                "enabled": True,
                "pitch_nm": 1500000,
                "offset_from_gap_nm": 800000,
                "via": {"drill_nm": 300000, "diameter_nm": 600000},
            },
        },
        "discontinuity": {
            "type": "VIA_TRANSITION",
            "signal_via": {
                "drill_nm": 300000,
                "diameter_nm": 650000,
                "pad_diameter_nm": 900000,
            },
            "antipads": {
                "L2": {
                    "shape": "ROUNDRECT",
                    "rx_nm": 1200000,
                    "ry_nm": 900000,
                    "corner_nm": 250000,
                },
                "L3": {"shape": "CIRCLE", "r_nm": 1100000},
            },
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


def _invalid_spec_data() -> dict[str, Any]:
    """Return an invalid spec with constraint violations."""
    data = _example_spec_data()
    data["transmission_line"]["w_nm"] = 50000  # Below fab minimum
    data["transmission_line"]["gap_nm"] = 50000  # Below fab minimum
    return data


class _FakeRunner:
    """Fake KiCad runner for testing."""

    def run_drc(self, board_path: Path, report_path: Path) -> subprocess.CompletedProcess[str]:
        report_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "F.Cu.gbr").write_text("G04 test*", encoding="utf-8")
        return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "drill.drl").write_text("M48", encoding="utf-8")
        return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")


class TestValidateSpecWithEngine:
    """Tests for validate_spec_with_engine() API."""

    def test_valid_spec_returns_validation_result(self, tmp_path: Path) -> None:
        """CP-3.5: validate_spec_with_engine returns ValidationResult for valid spec."""
        spec = CouponSpec.model_validate(_example_spec_data())

        result = validate_spec_with_engine(spec, out_dir=tmp_path)

        assert isinstance(result, ValidationResult)
        assert result.proof.passed is True
        assert result.was_repaired is False
        assert result.resolved is not None

    def test_valid_spec_writes_outputs(self, tmp_path: Path) -> None:
        """CP-3.5: validate_spec_with_engine writes resolved_design.json and constraint_proof.json."""
        spec = CouponSpec.model_validate(_example_spec_data())

        validate_spec_with_engine(spec, out_dir=tmp_path)

        assert (tmp_path / "resolved_design.json").exists()
        assert (tmp_path / "constraint_proof.json").exists()

    def test_invalid_spec_reject_mode_raises(self, tmp_path: Path) -> None:
        """CP-3.5: validate_spec_with_engine raises ConstraintViolationError in REJECT mode."""
        spec = CouponSpec.model_validate(_invalid_spec_data())

        with pytest.raises(ConstraintViolationError) as exc_info:
            validate_spec_with_engine(spec, out_dir=tmp_path, mode="REJECT")

        assert "T0_TRACE_WIDTH_MIN" in exc_info.value.constraint_ids

    def test_invalid_spec_repair_mode_repairs(self, tmp_path: Path) -> None:
        """CP-3.5: validate_spec_with_engine repairs spec in REPAIR mode."""
        data = _invalid_spec_data()
        data["constraints"]["mode"] = "REPAIR"
        spec = CouponSpec.model_validate(data)

        result = validate_spec_with_engine(spec, out_dir=tmp_path, mode="REPAIR")

        assert result.proof.passed is True
        assert result.was_repaired is True

    def test_repair_mode_writes_repair_map(self, tmp_path: Path) -> None:
        """CP-3.5: REPAIR mode writes repair_map.json when repairs are made."""
        data = _invalid_spec_data()
        data["constraints"]["mode"] = "REPAIR"
        spec = CouponSpec.model_validate(data)

        validate_spec_with_engine(spec, out_dir=tmp_path, mode="REPAIR")

        assert (tmp_path / "repair_map.json").exists()

    def test_proof_includes_all_tiers(self, tmp_path: Path) -> None:
        """CP-3.5: Proof includes constraints from all tiers (T0-T3)."""
        spec = CouponSpec.model_validate(_example_spec_data())

        result = validate_spec_with_engine(spec, out_dir=tmp_path)

        assert "T0" in result.proof.tiers
        assert "T1" in result.proof.tiers
        assert "T2" in result.proof.tiers
        assert "T3" in result.proof.tiers


class TestBuildCouponWithEngine:
    """Tests for build_coupon_with_engine() API."""

    def test_valid_spec_builds_successfully(self, tmp_path: Path) -> None:
        """CP-3.5: build_coupon_with_engine builds coupon for valid spec."""
        spec = CouponSpec.model_validate(_example_spec_data())
        runner = _FakeRunner()

        result = build_coupon_with_engine(
            spec,
            out_root=tmp_path,
            runner=runner,
        )

        assert isinstance(result, BuildResult)
        assert result.output_dir.exists()
        assert result.manifest_path.exists()
        assert result.cache_hit is False

    def test_build_writes_validation_outputs(self, tmp_path: Path) -> None:
        """CP-3.5: build_coupon_with_engine writes validation outputs."""
        spec = CouponSpec.model_validate(_example_spec_data())
        runner = _FakeRunner()

        result = build_coupon_with_engine(
            spec,
            out_root=tmp_path,
            runner=runner,
        )

        assert (result.output_dir / "resolved_design.json").exists()
        assert (result.output_dir / "constraint_proof.json").exists()

    def test_build_runs_drc_and_export(self, tmp_path: Path) -> None:
        """CP-3.5: build_coupon_with_engine runs DRC and export."""
        spec = CouponSpec.model_validate(_example_spec_data())
        runner = _FakeRunner()

        result = build_coupon_with_engine(
            spec,
            out_root=tmp_path,
            runner=runner,
        )

        # Check that fab outputs exist
        fab_dir = result.output_dir / "fab"
        assert fab_dir.exists()

    def test_invalid_spec_reject_mode_raises(self, tmp_path: Path) -> None:
        """CP-3.5: build_coupon_with_engine raises ConstraintViolationError in REJECT mode."""
        spec = CouponSpec.model_validate(_invalid_spec_data())
        runner = _FakeRunner()

        with pytest.raises(ConstraintViolationError) as exc_info:
            build_coupon_with_engine(
                spec,
                out_root=tmp_path,
                constraint_mode="REJECT",
                runner=runner,
            )

        assert "T0_TRACE_WIDTH_MIN" in exc_info.value.constraint_ids

    def test_repair_mode_builds_repaired_spec(self, tmp_path: Path) -> None:
        """CP-3.5: build_coupon_with_engine builds repaired spec in REPAIR mode."""
        data = _invalid_spec_data()
        data["constraints"]["mode"] = "REPAIR"
        spec = CouponSpec.model_validate(data)
        runner = _FakeRunner()

        result = build_coupon_with_engine(
            spec,
            out_root=tmp_path,
            constraint_mode="REPAIR",
            runner=runner,
        )

        assert isinstance(result, BuildResult)
        assert result.output_dir.exists()


class TestCLIValidateCommand:
    """Tests for CLI validate command with ConstraintEngine."""

    def test_validate_uses_engine_by_default(self, tmp_path: Path) -> None:
        """CP-3.5: validate command uses ConstraintEngine by default."""
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(_example_spec_data()), encoding="utf-8")

        with patch("sys.stdout.write") as mock_stdout:
            exit_code = cli_main.main(["validate", str(spec_path), "--out", str(tmp_path)])

        assert exit_code == 0
        call_args = mock_stdout.call_args[0][0]
        output = json.loads(call_args.strip())
        assert output["engine"] is True
        assert output["status"] == "valid"

    def test_validate_legacy_mode(self, tmp_path: Path) -> None:
        """CP-3.5: validate --legacy uses legacy constraint system."""
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(_example_spec_data()), encoding="utf-8")

        with patch("sys.stdout.write") as mock_stdout:
            exit_code = cli_main.main(["validate", str(spec_path), "--out", str(tmp_path), "--legacy"])

        assert exit_code == 0
        call_args = mock_stdout.call_args[0][0]
        output = json.loads(call_args.strip())
        assert output["engine"] is False

    def test_validate_invalid_spec_returns_error(self, tmp_path: Path) -> None:
        """CP-3.5: validate returns error exit code for invalid spec in REJECT mode."""
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(_invalid_spec_data()), encoding="utf-8")

        with patch("sys.stderr.write") as mock_stderr:
            exit_code = cli_main.main(["validate", str(spec_path), "--out", str(tmp_path)])

        assert exit_code == 1
        call_args = mock_stderr.call_args[0][0]
        output = json.loads(call_args.strip())
        assert output["status"] == "invalid"
        assert "T0_TRACE_WIDTH_MIN" in output["constraint_ids"]

    def test_validate_repair_mode(self, tmp_path: Path) -> None:
        """CP-3.5: validate --constraint-mode REPAIR repairs invalid spec."""
        data = _invalid_spec_data()
        data["constraints"]["mode"] = "REPAIR"
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(data), encoding="utf-8")

        with patch("sys.stdout.write") as mock_stdout:
            exit_code = cli_main.main([
                "validate", str(spec_path),
                "--out", str(tmp_path),
                "--constraint-mode", "REPAIR",
            ])

        assert exit_code == 0
        call_args = mock_stdout.call_args[0][0]
        output = json.loads(call_args.strip())
        assert output["was_repaired"] is True


class TestCLIBuildCommand:
    """Tests for CLI build command with ConstraintEngine."""

    def test_build_uses_engine_by_default(self, tmp_path: Path) -> None:
        """CP-3.5: build command uses ConstraintEngine by default."""
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(_example_spec_data()), encoding="utf-8")

        mock_result = BuildResult(
            output_dir=tmp_path / "output",
            design_hash="abc123",
            coupon_id="coupon",
            manifest_path=tmp_path / "output" / "manifest.json",
            cache_hit=False,
            toolchain_hash="toolchain123",
        )

        with (
            patch("formula_foundry.coupongen.cli_main.build_coupon_with_engine", return_value=mock_result),
            patch("sys.stdout.write"),
        ):
            exit_code = cli_main.main(["build", str(spec_path), "--out", str(tmp_path)])

        assert exit_code == 0

    def test_build_legacy_mode(self, tmp_path: Path) -> None:
        """CP-3.5: build --legacy uses legacy build_coupon function."""
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(_example_spec_data()), encoding="utf-8")

        mock_result = BuildResult(
            output_dir=tmp_path / "output",
            design_hash="abc123",
            coupon_id="coupon",
            manifest_path=tmp_path / "output" / "manifest.json",
            cache_hit=False,
            toolchain_hash="toolchain123",
        )

        with (
            patch("formula_foundry.coupongen.cli_main.build_coupon", return_value=mock_result) as mock_build,
            patch("sys.stdout.write"),
        ):
            exit_code = cli_main.main(["build", str(spec_path), "--out", str(tmp_path), "--legacy"])

        assert exit_code == 0
        mock_build.assert_called_once()

    def test_build_invalid_spec_returns_error(self, tmp_path: Path) -> None:
        """CP-3.5: build returns error exit code for invalid spec in REJECT mode."""
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(_invalid_spec_data()), encoding="utf-8")

        with patch("sys.stderr.write") as mock_stderr:
            exit_code = cli_main.main(["build", str(spec_path), "--out", str(tmp_path)])

        assert exit_code == 1
        call_args = mock_stderr.call_args[0][0]
        output = json.loads(call_args.strip())
        assert output["status"] == "constraint_violation"


class TestBuildFlowOrder:
    """Tests verifying the build flow order: validate/repair (Tier0-3) -> generate -> DRC -> export."""

    def test_build_flow_validates_before_generate(self, tmp_path: Path) -> None:
        """CP-3.5: Build flow validates constraints before generating KiCad project."""
        spec = CouponSpec.model_validate(_invalid_spec_data())
        runner = _FakeRunner()

        # REJECT mode should fail before generate
        with pytest.raises(ConstraintViolationError):
            build_coupon_with_engine(
                spec,
                out_root=tmp_path,
                constraint_mode="REJECT",
                runner=runner,
            )

        # No board file should be created
        assert not any(tmp_path.rglob("*.kicad_pcb"))

    def test_repair_flow_repairs_then_generates(self, tmp_path: Path) -> None:
        """CP-3.5: REPAIR mode repairs spec then generates valid KiCad project."""
        data = _invalid_spec_data()
        data["constraints"]["mode"] = "REPAIR"
        spec = CouponSpec.model_validate(data)
        runner = _FakeRunner()

        result = build_coupon_with_engine(
            spec,
            out_root=tmp_path,
            constraint_mode="REPAIR",
            runner=runner,
        )

        # Board file should be created after repair
        assert any(result.output_dir.rglob("*.kicad_pcb"))
