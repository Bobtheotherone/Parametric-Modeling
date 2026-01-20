"""Tests for CP-4.3 GPU pipeline integration.

This module tests the integration of GPU filter into the build-batch pipeline:
- GPU Tier0-2 filter on candidates
- Parameter mapping from u vectors to CouponSpec
- CuPy/CUDA version recording in manifest
- NumPy fallback when CuPy unavailable
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from formula_foundry.coupongen.constraints.gpu_filter import (
    BatchFilterResult,
    FamilyF1ParameterSpace,
    RepairMeta,
    batch_filter,
    is_gpu_available,
)
from formula_foundry.coupongen.param_mapping import (
    apply_params_to_spec,
    batch_u_to_specs_f1,
    get_f1_parameter_space,
    u_to_spec_f1,
    u_to_spec_params_f1,
)
from formula_foundry.coupongen.spec import CouponSpec


def _minimal_f1_spec_dict() -> dict[str, Any]:
    """Return a minimal F1 spec template for testing."""
    return {
        "schema_version": 1,
        "coupon_family": "F1",
        "units": "nm",
        "toolchain": {
            "kicad": {
                "version": "9.0.7",
                "docker_image": "kicad/kicad:9.0.7@sha256:test",
            }
        },
        "fab_profile": {"id": "generic"},
        "stackup": {
            "copper_layers": 4,
            "thicknesses_nm": {
                "copper_top": 35_000,
                "core": 1_000_000,
                "prepreg": 200_000,
                "copper_inner1": 35_000,
                "copper_inner2": 35_000,
                "copper_bottom": 35_000,
            },
            "materials": {"er": 4.5, "loss_tangent": 0.02},
        },
        "board": {
            "outline": {
                "width_nm": 20_000_000,
                "length_nm": 100_000_000,
                "corner_radius_nm": 1_000_000,
            },
            "origin": {"mode": "center"},
            "text": {"coupon_id": "TEST", "include_manifest_hash": True},
        },
        "connectors": {
            "left": {
                "footprint": "test:SMA",
                "position_nm": [5_000_000, 0],
                "rotation_deg": 0,
            },
            "right": {
                "footprint": "test:SMA",
                "position_nm": [95_000_000, 0],
                "rotation_deg": 180,
            },
        },
        "transmission_line": {
            "type": "cpwg",
            "layer": "F.Cu",
            "w_nm": 200_000,
            "gap_nm": 150_000,
            "length_left_nm": 20_000_000,
            "length_right_nm": 20_000_000,
            "ground_via_fence": {
                "enabled": True,
                "pitch_nm": 1_500_000,
                "offset_from_gap_nm": 500_000,
                "via": {"drill_nm": 300_000, "diameter_nm": 600_000},
            },
        },
        "discontinuity": {
            "type": "single_via",
            "signal_via": {
                "drill_nm": 300_000,
                "diameter_nm": 600_000,
                "pad_diameter_nm": 900_000,
            },
            "return_vias": {
                "pattern": "ring",
                "count": 8,
                "radius_nm": 2_000_000,
                "via": {"drill_nm": 300_000, "diameter_nm": 600_000},
            },
        },
        "constraints": {
            "mode": "REPAIR",
            "drc": {"must_pass": True, "severity": "error"},
            "symmetry": {"enforce": True},
            "allow_unconnected_copper": False,
        },
        "export": {
            "gerbers": {"enabled": True, "format": "RS274X"},
            "drill": {"enabled": True, "format": "excellon"},
            "outputs_dir": "outputs",
        },
    }


class TestParameterMapping:
    """Test parameter mapping from u vectors to spec parameters."""

    def test_get_f1_parameter_space(self) -> None:
        """Should return the F1 parameter space."""
        space = get_f1_parameter_space()
        assert isinstance(space, FamilyF1ParameterSpace)
        assert space.dimension == 19

    def test_u_to_spec_params_f1(self) -> None:
        """Should convert u vector to physical parameters."""
        u = np.ones(19) * 0.5  # All midpoints
        params = u_to_spec_params_f1(u)

        assert "trace_width_nm" in params
        assert "board_length_nm" in params
        assert "signal_drill_nm" in params

        # All values should be integers
        for val in params.values():
            assert isinstance(val, int)

    def test_apply_params_to_spec(self) -> None:
        """Should apply parameters to spec template."""
        spec_dict = _minimal_f1_spec_dict()
        spec = CouponSpec.model_validate(spec_dict)

        params = {
            "trace_width_nm": 300_000,
            "trace_gap_nm": 180_000,
            "board_width_nm": 25_000_000,
        }

        new_spec = apply_params_to_spec(spec, params)

        assert new_spec.transmission_line.w_nm == 300_000
        assert new_spec.transmission_line.gap_nm == 180_000
        assert new_spec.board.outline.width_nm == 25_000_000

    def test_u_to_spec_f1(self) -> None:
        """Should convert u vector to full CouponSpec."""
        spec_dict = _minimal_f1_spec_dict()
        spec_template = CouponSpec.model_validate(spec_dict)

        # Create a valid u vector (midpoints)
        u = np.ones(19) * 0.5

        new_spec = u_to_spec_f1(u, spec_template)

        assert isinstance(new_spec, CouponSpec)
        # Values should differ from template
        assert new_spec.transmission_line.w_nm != spec_template.transmission_line.w_nm

    def test_batch_u_to_specs_f1(self) -> None:
        """Should convert batch of u vectors to specs."""
        spec_dict = _minimal_f1_spec_dict()
        spec_template = CouponSpec.model_validate(spec_dict)

        u_batch = np.random.rand(5, 19)
        specs = batch_u_to_specs_f1(u_batch, spec_template)

        assert len(specs) == 5
        for spec in specs:
            assert isinstance(spec, CouponSpec)


class TestGPUFilterIntegration:
    """Test GPU filter integration with pipeline."""

    def test_batch_filter_with_good_candidates(self) -> None:
        """GPU filter should pass well-formed candidates."""
        # Create candidates with known-good values
        space = FamilyF1ParameterSpace()
        u_batch = np.ones((10, space.dimension)) * 0.5

        result = batch_filter(u_batch, mode="REPAIR", seed=42, use_gpu=False)

        assert isinstance(result, BatchFilterResult)
        assert result.n_feasible > 0

    def test_batch_filter_repair_mode(self) -> None:
        """REPAIR mode should produce more feasible candidates than REJECT."""
        np.random.seed(42)
        u_batch = np.random.rand(100, 19)

        result_reject = batch_filter(u_batch, mode="REJECT", seed=42, use_gpu=False)
        result_repair = batch_filter(u_batch, mode="REPAIR", seed=42, use_gpu=False)

        assert result_repair.n_feasible >= result_reject.n_feasible

    def test_batch_filter_deterministic_with_seed(self) -> None:
        """Same seed should produce same results."""
        u_batch = np.random.rand(50, 19)

        result1 = batch_filter(u_batch, mode="REPAIR", seed=12345, use_gpu=False)
        result2 = batch_filter(u_batch, mode="REPAIR", seed=12345, use_gpu=False)

        np.testing.assert_array_equal(result1.mask, result2.mask)
        np.testing.assert_array_equal(result1.u_repaired, result2.u_repaired)


class TestCLIBuildBatch:
    """Test build-batch CLI command (without actually running KiCad)."""

    def test_build_batch_cli_arguments(self) -> None:
        """build-batch should accept GPU-related arguments."""
        from formula_foundry.coupongen.cli_main import build_parser

        parser = build_parser()

        # Parse with GPU-related flags
        args = parser.parse_args([
            "build-batch",
            "spec.yaml",
            "--u", "u.npy",
            "--out", "output",
            "--no-gpu",
            "--profile", "generic",
            "--seed", "42",
            "--constraint-mode", "REPAIR",
            "--skip-filter",
        ])

        assert args.command == "build-batch"
        assert args.no_gpu is True
        assert args.profile == "generic"
        assert args.seed == 42
        assert args.constraint_mode == "REPAIR"
        assert args.skip_filter is True

    def test_build_batch_default_arguments(self) -> None:
        """build-batch should have sensible defaults."""
        from formula_foundry.coupongen.cli_main import build_parser

        parser = build_parser()

        args = parser.parse_args([
            "build-batch",
            "spec.yaml",
            "--u", "u.npy",
            "--out", "output",
        ])

        assert args.no_gpu is False
        assert args.profile == "generic"
        assert args.seed == 0
        assert args.constraint_mode == "REPAIR"
        assert args.skip_filter is False


class TestManifestGPUMetadata:
    """Test GPU metadata in manifest."""

    def test_gpu_metadata_structure(self) -> None:
        """GPU metadata should have expected structure."""
        from formula_foundry.coupongen.cli_main import _add_gpu_metadata_to_manifest

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Write minimal manifest
            manifest = {
                "schema_version": 1,
                "design_hash": "test",
                "toolchain": {"kicad": {"version": "9.0.7"}},
            }
            f.write(json.dumps(manifest))
            f.flush()
            manifest_path = Path(f.name)

        try:
            _add_gpu_metadata_to_manifest(
                manifest_path,
                cupy_version="13.0.0",
                cuda_version="12010",
            )

            # Read back and verify
            updated_manifest = json.loads(manifest_path.read_text())

            assert "gpu" in updated_manifest["toolchain"]
            assert updated_manifest["toolchain"]["gpu"]["used"] is True
            assert updated_manifest["toolchain"]["gpu"]["cupy_version"] == "13.0.0"
            assert updated_manifest["toolchain"]["gpu"]["cuda_runtime_version"] == "12010"
        finally:
            manifest_path.unlink()


class TestGPUAvailability:
    """Test GPU availability handling."""

    def test_is_gpu_available_returns_bool(self) -> None:
        """is_gpu_available should return a boolean."""
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_batch_filter_works_without_gpu(self) -> None:
        """batch_filter should work when GPU is disabled."""
        u_batch = np.random.rand(20, 19)
        result = batch_filter(u_batch, mode="REPAIR", seed=0, use_gpu=False)

        assert isinstance(result, BatchFilterResult)
        assert result.n_candidates == 20


@pytest.mark.skipif(not is_gpu_available(), reason="CuPy not available")
class TestWithGPU:
    """Tests requiring GPU/CuPy availability."""

    def test_gpu_and_cpu_results_match(self) -> None:
        """GPU and CPU should produce same results in REJECT mode."""
        np.random.seed(42)
        u_batch = np.random.rand(100, 19)

        result_cpu = batch_filter(u_batch, mode="REJECT", seed=42, use_gpu=False)
        result_gpu = batch_filter(u_batch, mode="REJECT", seed=42, use_gpu=True)

        np.testing.assert_array_equal(result_cpu.mask, result_gpu.mask)

    def test_gpu_cupy_version_available(self) -> None:
        """Should be able to get CuPy version when GPU available."""
        import cupy as cp

        version = cp.__version__
        assert isinstance(version, str)
        assert len(version) > 0


class TestEndToEndPipeline:
    """End-to-end tests for the GPU-integrated pipeline (mocked KiCad)."""

    @patch("formula_foundry.coupongen.cli_main.build_coupon_with_engine")
    @patch("formula_foundry.coupongen.cli_main.load_spec")
    def test_build_batch_runs_gpu_filter(
        self,
        mock_load_spec: MagicMock,
        mock_build_coupon: MagicMock,
    ) -> None:
        """build-batch should run GPU filter on candidates."""
        from formula_foundry.coupongen.cli_main import _run_build_batch

        # Setup mocks
        spec_dict = _minimal_f1_spec_dict()
        mock_load_spec.return_value = CouponSpec.model_validate(spec_dict)

        mock_result = MagicMock()
        mock_result.design_hash = "test_hash"
        mock_result.coupon_id = "test_id"
        mock_result.output_dir = Path("/tmp/test")
        mock_result.cache_hit = False
        mock_result.manifest_path = Path("/tmp/test/manifest.json")
        mock_build_coupon.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create mock spec template file
            spec_path = tmppath / "spec.yaml"
            spec_path.write_text("# mock")

            # Create u vectors file
            u_batch = np.random.rand(5, 19)
            u_path = tmppath / "u.npy"
            np.save(u_path, u_batch)

            # Create output directory
            out_path = tmppath / "output"

            # Create mock args
            args = MagicMock()
            args.spec_template = spec_path
            args.u = u_path
            args.out = out_path
            args.mode = "local"
            args.toolchain_image = ""
            args.limit = 2
            args.skip_filter = False
            args.no_gpu = True  # Force CPU for testing
            args.profile = "generic"
            args.seed = 42
            args.constraint_mode = "REPAIR"

            # Run
            result = _run_build_batch(args)

            # Verify filter metadata was written
            assert (out_path / "filter_metadata.json").exists()
            assert (out_path / "mask.npy").exists()
            assert (out_path / "u_repaired.npy").exists()

            filter_meta = json.loads((out_path / "filter_metadata.json").read_text())
            assert "n_candidates" in filter_meta
            assert "n_feasible" in filter_meta
            assert filter_meta["use_gpu"] is False  # We forced CPU
            assert filter_meta["seed"] == 42

    @patch("formula_foundry.coupongen.cli_main.build_coupon_with_engine")
    @patch("formula_foundry.coupongen.cli_main.load_spec")
    def test_build_batch_skip_filter(
        self,
        mock_load_spec: MagicMock,
        mock_build_coupon: MagicMock,
    ) -> None:
        """build-batch with --skip-filter should not run GPU filter."""
        from formula_foundry.coupongen.cli_main import _run_build_batch

        # Setup mocks
        spec_dict = _minimal_f1_spec_dict()
        mock_load_spec.return_value = CouponSpec.model_validate(spec_dict)

        mock_result = MagicMock()
        mock_result.design_hash = "test_hash"
        mock_result.coupon_id = "test_id"
        mock_result.output_dir = Path("/tmp/test")
        mock_result.cache_hit = False
        mock_result.manifest_path = Path("/tmp/test/manifest.json")
        mock_build_coupon.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create mock spec template file
            spec_path = tmppath / "spec.yaml"
            spec_path.write_text("# mock")

            # Create u vectors file
            u_batch = np.random.rand(3, 19)
            u_path = tmppath / "u.npy"
            np.save(u_path, u_batch)

            out_path = tmppath / "output"

            # Create mock args with skip_filter
            args = MagicMock()
            args.spec_template = spec_path
            args.u = u_path
            args.out = out_path
            args.mode = "local"
            args.toolchain_image = ""
            args.limit = 2
            args.skip_filter = True
            args.no_gpu = True
            args.profile = "generic"
            args.seed = 0
            args.constraint_mode = "REPAIR"

            # Run
            result = _run_build_batch(args)

            # Verify filter metadata was NOT written (skipped)
            assert not (out_path / "filter_metadata.json").exists()
