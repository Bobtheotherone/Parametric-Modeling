"""Tests for REQ-M1-022: Public Python API in formula_foundry.__init__.py.

Verifies that the top-level formula_foundry package exports the required functions
with proper type hints for orchestration by M6-M8:
- load_spec(): Load CouponSpec from YAML/JSON file
- resolve(): Resolve CouponSpec to ResolvedDesign
- generate_kicad(): Generate KiCad board files
- run_drc(): Run KiCad DRC check
- export_fab(): Export Gerber and drill files
"""

from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path
from typing import get_type_hints

import pytest
import yaml  # type: ignore[import-untyped]

import formula_foundry


def _example_spec_data() -> dict[str, object]:
    """Return a valid F1 coupon spec for testing."""
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
    """Fake KiCad CLI runner for testing without real KiCad installation."""

    def run_drc(self, board_path: Path, report_path: Path) -> subprocess.CompletedProcess[str]:
        report_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        (out_dir / "F.Cu.gbr").write_text("G04 test*", encoding="utf-8")
        return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        (out_dir / "drill.drl").write_text("M48", encoding="utf-8")
        return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")


class TestTopLevelAPIExports:
    """Test that required API functions are exported at top level."""

    def test_load_spec_exported(self) -> None:
        """load_spec function is exported from formula_foundry."""
        assert hasattr(formula_foundry, "load_spec")
        assert callable(formula_foundry.load_spec)

    def test_resolve_exported(self) -> None:
        """resolve function is exported from formula_foundry."""
        assert hasattr(formula_foundry, "resolve")
        assert callable(formula_foundry.resolve)

    def test_generate_kicad_exported(self) -> None:
        """generate_kicad function is exported from formula_foundry."""
        assert hasattr(formula_foundry, "generate_kicad")
        assert callable(formula_foundry.generate_kicad)

    def test_run_drc_exported(self) -> None:
        """run_drc function is exported from formula_foundry."""
        assert hasattr(formula_foundry, "run_drc")
        assert callable(formula_foundry.run_drc)

    def test_export_fab_exported(self) -> None:
        """export_fab function is exported from formula_foundry."""
        assert hasattr(formula_foundry, "export_fab")
        assert callable(formula_foundry.export_fab)

    def test_all_exports_in_dunder_all(self) -> None:
        """All required functions are in __all__."""
        required = ["load_spec", "resolve", "generate_kicad", "run_drc", "export_fab"]
        for name in required:
            assert name in formula_foundry.__all__, f"{name} not in __all__"


class TestTopLevelAPITypeHints:
    """Test that API functions have proper type hints for M6-M8 orchestration."""

    def test_load_spec_type_hints(self) -> None:
        """load_spec has type hints: (Path) -> CouponSpec."""
        sig = inspect.signature(formula_foundry.load_spec)
        params = list(sig.parameters.keys())
        assert "path" in params
        # Return type should be CouponSpec
        hints = get_type_hints(formula_foundry.load_spec)
        assert "return" in hints
        assert hints["return"].__name__ == "CouponSpec"

    def test_resolve_type_hints(self) -> None:
        """resolve has type hints: (CouponSpec) -> ResolvedDesign."""
        sig = inspect.signature(formula_foundry.resolve)
        params = list(sig.parameters.keys())
        assert "spec" in params
        hints = get_type_hints(formula_foundry.resolve)
        assert "return" in hints
        assert hints["return"].__name__ == "ResolvedDesign"

    def test_generate_kicad_type_hints(self) -> None:
        """generate_kicad has type hints with ResolvedDesign, CouponSpec, Path."""
        sig = inspect.signature(formula_foundry.generate_kicad)
        params = list(sig.parameters.keys())
        assert "resolved" in params
        assert "spec" in params
        assert "out_dir" in params
        hints = get_type_hints(formula_foundry.generate_kicad)
        assert "return" in hints
        assert hints["return"].__name__ == "KiCadProjectPaths"

    def test_run_drc_type_hints(self) -> None:
        """run_drc has type hints with Path, KicadToolchain."""
        sig = inspect.signature(formula_foundry.run_drc)
        params = list(sig.parameters.keys())
        assert "board_path" in params
        assert "toolchain" in params
        hints = get_type_hints(formula_foundry.run_drc)
        assert "return" in hints
        assert hints["return"].__name__ == "DrcReport"

    def test_export_fab_type_hints(self) -> None:
        """export_fab has type hints with Path, Path, KicadToolchain."""
        sig = inspect.signature(formula_foundry.export_fab)
        params = list(sig.parameters.keys())
        assert "board_path" in params
        assert "out_dir" in params
        assert "toolchain" in params
        hints = get_type_hints(formula_foundry.export_fab)
        assert "return" in hints


class TestTopLevelAPITypeExports:
    """Test that result types are also exported for type checking."""

    def test_coupon_spec_exported(self) -> None:
        """CouponSpec type is exported."""
        assert hasattr(formula_foundry, "CouponSpec")

    def test_resolved_design_exported(self) -> None:
        """ResolvedDesign type is exported."""
        assert hasattr(formula_foundry, "ResolvedDesign")

    def test_kicad_project_paths_exported(self) -> None:
        """KiCadProjectPaths type is exported."""
        assert hasattr(formula_foundry, "KiCadProjectPaths")

    def test_drc_report_exported(self) -> None:
        """DrcReport type is exported."""
        assert hasattr(formula_foundry, "DrcReport")

    def test_build_result_exported(self) -> None:
        """BuildResult type is exported."""
        assert hasattr(formula_foundry, "BuildResult")

    def test_constraint_types_exported(self) -> None:
        """Constraint result types are exported."""
        assert hasattr(formula_foundry, "ConstraintEvaluation")
        assert hasattr(formula_foundry, "ConstraintProof")
        assert hasattr(formula_foundry, "ConstraintResult")
        assert hasattr(formula_foundry, "ConstraintViolation")
        assert hasattr(formula_foundry, "RepairInfo")


class TestTopLevelAPIWorkflow:
    """Test the complete API workflow using top-level imports."""

    def test_full_workflow_with_top_level_api(self, tmp_path: Path) -> None:
        """Complete workflow using formula_foundry.* imports."""
        # Write spec file
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(_example_spec_data()), encoding="utf-8")

        # Step 1: Load spec
        spec = formula_foundry.load_spec(spec_path)
        assert isinstance(spec, formula_foundry.CouponSpec)

        # Step 2: Resolve spec
        resolved = formula_foundry.resolve(spec)
        assert isinstance(resolved, formula_foundry.ResolvedDesign)

        # Step 3: Generate KiCad
        project = formula_foundry.generate_kicad(resolved, spec, tmp_path)
        assert isinstance(project, formula_foundry.KiCadProjectPaths)
        assert project.board_path.exists()

        # Step 4: Run DRC
        runner = _FakeRunner()
        report = formula_foundry.run_drc(project.board_path, spec.toolchain.kicad, runner=runner)
        assert isinstance(report, formula_foundry.DrcReport)
        assert report.report_path.exists()
        assert report.returncode == 0

        # Step 5: Export fab
        fab_dir = tmp_path / "fab"
        hashes = formula_foundry.export_fab(
            project.board_path, fab_dir, spec.toolchain.kicad, runner=runner
        )
        assert isinstance(hashes, dict)
        assert len(hashes) > 0
        assert "gerbers/F.Cu.gbr" in hashes
        assert "drill/drill.drl" in hashes

    def test_validate_spec_also_exported(self, tmp_path: Path) -> None:
        """validate_spec convenience function is also exported."""
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(_example_spec_data()), encoding="utf-8")

        spec = formula_foundry.load_spec(spec_path)
        evaluation = formula_foundry.validate_spec(spec, out_dir=tmp_path)
        assert isinstance(evaluation, formula_foundry.ConstraintEvaluation)
        assert (tmp_path / "resolved_design.json").exists()
        assert (tmp_path / "constraint_proof.json").exists()

    def test_build_coupon_also_exported(self, tmp_path: Path) -> None:
        """build_coupon convenience function is also exported."""
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(_example_spec_data()), encoding="utf-8")

        spec = formula_foundry.load_spec(spec_path)
        runner = _FakeRunner()

        result = formula_foundry.build_coupon(
            spec, out_root=tmp_path, runner=runner, kicad_cli_version="9.0.7"
        )
        assert isinstance(result, formula_foundry.BuildResult)
        assert result.manifest_path.exists()


class TestAPIDocstrings:
    """Test that the module and functions have docstrings."""

    def test_module_docstring(self) -> None:
        """Module has a docstring with usage example."""
        assert formula_foundry.__doc__ is not None
        assert "load_spec" in formula_foundry.__doc__
        assert "resolve" in formula_foundry.__doc__
        assert "generate_kicad" in formula_foundry.__doc__
        assert "run_drc" in formula_foundry.__doc__
        assert "export_fab" in formula_foundry.__doc__
