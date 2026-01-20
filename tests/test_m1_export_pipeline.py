"""Tests for the export pipeline module.

Verifies REQ-M1-019 and REQ-M1-020:
- REQ-M1-019: All output directories must be keyed by design_hash and coupon_id;
              re-running build must not create divergent outputs.
- REQ-M1-020: The build pipeline must implement caching keyed by
              design_hash + toolchain_hash and must be deterministic
              when cache hits occur.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from formula_foundry.coupongen.export import (
    CacheKey,
    ExportPipeline,
    ExportResult,
    compute_cache_key,
    is_cache_valid,
    run_export_pipeline,
)
from formula_foundry.coupongen.spec import CouponSpec


class _CountingRunner:
    """Test runner that counts calls to each method."""

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


class TestCacheKey:
    """Tests for CacheKey class."""

    def test_combined_hash_is_deterministic(self) -> None:
        key1 = CacheKey(design_hash="abc123", toolchain_hash="def456")
        key2 = CacheKey(design_hash="abc123", toolchain_hash="def456")
        assert key1.combined_hash == key2.combined_hash

    def test_combined_hash_differs_for_different_keys(self) -> None:
        key1 = CacheKey(design_hash="abc123", toolchain_hash="def456")
        key2 = CacheKey(design_hash="abc123", toolchain_hash="xyz789")
        assert key1.combined_hash != key2.combined_hash

    def test_matches_manifest(self) -> None:
        key = CacheKey(design_hash="abc123", toolchain_hash="def456")
        manifest = {"design_hash": "abc123", "toolchain_hash": "def456"}
        assert key.matches(manifest)

    def test_does_not_match_different_design_hash(self) -> None:
        key = CacheKey(design_hash="abc123", toolchain_hash="def456")
        manifest = {"design_hash": "different", "toolchain_hash": "def456"}
        assert not key.matches(manifest)

    def test_does_not_match_different_toolchain_hash(self) -> None:
        key = CacheKey(design_hash="abc123", toolchain_hash="def456")
        manifest = {"design_hash": "abc123", "toolchain_hash": "different"}
        assert not key.matches(manifest)


class TestExportPipeline:
    """Tests for ExportPipeline class."""

    def test_run_returns_export_result(self, tmp_path: Path) -> None:
        """Test that running the pipeline returns an ExportResult."""
        runner = _CountingRunner()
        spec = CouponSpec.model_validate(_example_spec_data())
        pipeline = ExportPipeline(out_root=tmp_path, runner=runner, kicad_cli_version="9.0.7")

        result = pipeline.run(spec)

        assert isinstance(result, ExportResult)
        assert result.output_dir.exists()
        assert result.manifest_path.exists()
        assert result.cache_hit is False
        assert result.design_hash
        assert result.coupon_id
        assert result.toolchain_hash

    def test_output_dir_keyed_by_design_hash_req_m1_019(self, tmp_path: Path) -> None:
        """REQ-M1-019: Output directory must be keyed by design_hash and coupon_id."""
        runner = _CountingRunner()
        spec = CouponSpec.model_validate(_example_spec_data())
        pipeline = ExportPipeline(out_root=tmp_path, runner=runner, kicad_cli_version="9.0.7")

        result = pipeline.run(spec)

        folder_name = result.output_dir.name
        assert result.design_hash in folder_name
        assert result.coupon_id in folder_name

    def test_cache_hit_on_second_run_req_m1_020(self, tmp_path: Path) -> None:
        """REQ-M1-020: Cache must hit on second run with same spec."""
        runner = _CountingRunner()
        spec = CouponSpec.model_validate(_example_spec_data())
        pipeline = ExportPipeline(out_root=tmp_path, runner=runner, kicad_cli_version="9.0.7")

        result_a = pipeline.run(spec)
        assert result_a.cache_hit is False
        assert runner.drc_calls == 1
        assert runner.gerber_calls == 1
        assert runner.drill_calls == 1

        result_b = pipeline.run(spec)
        assert result_b.cache_hit is True
        assert runner.drc_calls == 1  # No additional calls
        assert runner.gerber_calls == 1
        assert runner.drill_calls == 1

    def test_cache_miss_on_toolchain_change_req_m1_020(self, tmp_path: Path) -> None:
        """REQ-M1-020: Cache must miss when toolchain changes."""
        runner = _CountingRunner()
        spec = CouponSpec.model_validate(_example_spec_data())
        pipeline = ExportPipeline(out_root=tmp_path, runner=runner, kicad_cli_version="9.0.7")

        result_a = pipeline.run(spec)
        assert result_a.cache_hit is False
        assert runner.drc_calls == 1

        # Change toolchain
        modified = _example_spec_data()
        modified["toolchain"]["kicad"]["docker_image"] = "kicad/kicad:9.0.7@sha256:feedbeef"
        spec_modified = CouponSpec.model_validate(modified)

        result_b = pipeline.run(spec_modified)
        assert result_b.cache_hit is False
        assert runner.drc_calls == 2  # New DRC run

    def test_deterministic_outputs_req_m1_019(self, tmp_path: Path) -> None:
        """REQ-M1-019: Re-running build must not create divergent outputs."""
        runner = _CountingRunner()
        spec = CouponSpec.model_validate(_example_spec_data())
        pipeline = ExportPipeline(out_root=tmp_path, runner=runner, kicad_cli_version="9.0.7")

        result_a = pipeline.run(spec)
        result_b = pipeline.run(spec)

        # Same output directory
        assert result_a.output_dir == result_b.output_dir
        assert result_a.design_hash == result_b.design_hash
        assert result_a.coupon_id == result_b.coupon_id
        assert result_a.toolchain_hash == result_b.toolchain_hash


class TestRunExportPipeline:
    """Tests for the convenience function."""

    def test_functional_interface(self, tmp_path: Path) -> None:
        """Test the functional interface works correctly."""
        runner = _CountingRunner()
        spec = CouponSpec.model_validate(_example_spec_data())

        result = run_export_pipeline(spec, out_root=tmp_path, runner=runner, kicad_cli_version="9.0.7")

        assert isinstance(result, ExportResult)
        assert result.output_dir.exists()


class TestComputeCacheKey:
    """Tests for compute_cache_key function."""

    def test_returns_cache_key(self) -> None:
        """Test that compute_cache_key returns a CacheKey."""
        spec = CouponSpec.model_validate(_example_spec_data())
        key = compute_cache_key(spec, kicad_cli_version="9.0.7")

        assert isinstance(key, CacheKey)
        assert key.design_hash
        assert key.toolchain_hash

    def test_deterministic(self) -> None:
        """Test that compute_cache_key is deterministic."""
        spec = CouponSpec.model_validate(_example_spec_data())
        key1 = compute_cache_key(spec, kicad_cli_version="9.0.7")
        key2 = compute_cache_key(spec, kicad_cli_version="9.0.7")

        assert key1.design_hash == key2.design_hash
        assert key1.toolchain_hash == key2.toolchain_hash


class TestIsCacheValid:
    """Tests for is_cache_valid function."""

    def test_returns_false_when_no_cache(self, tmp_path: Path) -> None:
        """Test that is_cache_valid returns False when no cache exists."""
        spec = CouponSpec.model_validate(_example_spec_data())
        assert is_cache_valid(spec, tmp_path, kicad_cli_version="9.0.7") is False

    def test_returns_true_after_build(self, tmp_path: Path) -> None:
        """Test that is_cache_valid returns True after building."""
        runner = _CountingRunner()
        spec = CouponSpec.model_validate(_example_spec_data())

        run_export_pipeline(spec, out_root=tmp_path, runner=runner, kicad_cli_version="9.0.7")
        assert is_cache_valid(spec, tmp_path, kicad_cli_version="9.0.7") is True

    def test_returns_false_after_toolchain_change(self, tmp_path: Path) -> None:
        """Test that is_cache_valid returns False when toolchain changes."""
        runner = _CountingRunner()
        spec = CouponSpec.model_validate(_example_spec_data())

        run_export_pipeline(spec, out_root=tmp_path, runner=runner, kicad_cli_version="9.0.7")

        # Change toolchain
        modified = _example_spec_data()
        modified["toolchain"]["kicad"]["docker_image"] = "kicad/kicad:9.0.7@sha256:feedbeef"
        spec_modified = CouponSpec.model_validate(modified)

        assert is_cache_valid(spec_modified, tmp_path, kicad_cli_version="9.0.7") is False
