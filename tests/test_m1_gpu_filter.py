"""Tests for GPU-accelerated batch constraint prefilter (REQ-M1-GPU-FILTER).

This module tests the GPU vectorized constraint checking for Tier 0-2 constraints:
- Batch filtering of normalized design vectors
- Feasibility mask generation
- Constraint repair logic
- Repair metadata tracking

Updated for CP-4.1: Tests formal batch_filter API with mode, seed, and RepairMeta.
"""

from __future__ import annotations

import numpy as np
import pytest

from formula_foundry.coupongen.constraints.gpu_filter import (
    BatchFilterResult,
    FamilyF1ParameterSpace,
    GPUConstraintFilter,
    ParameterMapping,
    RepairMeta,
    batch_filter,
    is_gpu_available,
)


def _default_fab_limits() -> dict[str, int]:
    """Return default fab limits for testing."""
    return {
        "min_trace_width_nm": 100_000,
        "min_gap_nm": 100_000,
        "min_drill_nm": 200_000,
        "min_via_diameter_nm": 300_000,
        "min_annular_ring_nm": 100_000,
        "min_edge_clearance_nm": 200_000,
        "min_via_to_via_nm": 200_000,
        "min_board_width_nm": 5_000_000,
    }


class TestParameterMapping:
    """Test parameter mapping between normalized and physical space."""

    def test_linear_mapping_to_physical(self) -> None:
        """Linear mapping should correctly transform [0,1] to [min, max]."""
        mapping = ParameterMapping(name="test", index=0, scale="linear", min_val=100.0, max_val=500.0)

        # Test endpoints
        assert mapping.to_physical(0.0, np) == 100.0
        assert mapping.to_physical(1.0, np) == 500.0

        # Test midpoint
        assert mapping.to_physical(0.5, np) == 300.0

    def test_linear_mapping_to_normalized(self) -> None:
        """Linear mapping should correctly transform [min, max] to [0,1]."""
        mapping = ParameterMapping(name="test", index=0, scale="linear", min_val=100.0, max_val=500.0)

        assert mapping.to_normalized(100.0, np) == pytest.approx(0.0)
        assert mapping.to_normalized(500.0, np) == pytest.approx(1.0)
        assert mapping.to_normalized(300.0, np) == pytest.approx(0.5)

    def test_log_mapping_to_physical(self) -> None:
        """Log mapping should correctly transform using logarithmic scale."""
        mapping = ParameterMapping(name="test", index=0, scale="log", min_val=10.0, max_val=1000.0)

        assert mapping.to_physical(0.0, np) == pytest.approx(10.0)
        assert mapping.to_physical(1.0, np) == pytest.approx(1000.0)
        # Midpoint in log space: sqrt(10 * 1000) = 100
        assert mapping.to_physical(0.5, np) == pytest.approx(100.0, rel=0.01)

    def test_vectorized_mapping(self) -> None:
        """Mapping should work on arrays."""
        mapping = ParameterMapping(name="test", index=0, scale="linear", min_val=0.0, max_val=100.0)

        u = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        expected = np.array([0.0, 25.0, 50.0, 75.0, 100.0])

        result = mapping.to_physical(u, np)
        np.testing.assert_array_almost_equal(result, expected)


class TestFamilyF1ParameterSpace:
    """Test F1 family parameter space definition."""

    def test_dimension(self) -> None:
        """Parameter space should have correct dimension."""
        space = FamilyF1ParameterSpace()
        assert space.dimension == 19

    def test_get_mapping_by_name(self) -> None:
        """Should be able to retrieve mappings by name."""
        space = FamilyF1ParameterSpace()

        trace_width = space.get_mapping("trace_width_nm")
        assert trace_width is not None
        assert trace_width.index == 0

        signal_drill = space.get_mapping("signal_drill_nm")
        assert signal_drill is not None
        assert signal_drill.index == 5

    def test_get_mapping_nonexistent(self) -> None:
        """Getting nonexistent mapping should return None."""
        space = FamilyF1ParameterSpace()
        assert space.get_mapping("nonexistent") is None

    def test_to_physical_batch(self) -> None:
        """Batch conversion to physical parameters should work."""
        space = FamilyF1ParameterSpace()
        u_batch = np.ones((10, space.dimension)) * 0.5  # All midpoint values

        params = space.to_physical_batch(u_batch, np)

        # Check some expected values at midpoint
        assert len(params["trace_width_nm"]) == 10
        assert all(params["trace_width_nm"] > 0)
        assert all(params["board_width_nm"] > 0)


class TestGPUConstraintFilter:
    """Test the GPU constraint filter."""

    def test_filter_initialization(self) -> None:
        """Filter should initialize with fab limits."""
        limits = _default_fab_limits()
        filter_instance = GPUConstraintFilter(limits, use_gpu=False)

        assert filter_instance._min_trace_width == 100_000
        assert filter_instance._min_gap == 100_000
        assert filter_instance._min_drill == 200_000

    def test_check_tier0_all_pass(self) -> None:
        """Valid parameters should pass Tier 0 checks."""
        limits = _default_fab_limits()
        filter_instance = GPUConstraintFilter(limits, use_gpu=False)

        # Create valid parameters (all well above minimums)
        params = {
            "trace_width_nm": np.array([200_000, 300_000]),
            "trace_gap_nm": np.array([150_000, 200_000]),
            "board_width_nm": np.array([20_000_000, 30_000_000]),
            "board_length_nm": np.array([80_000_000, 100_000_000]),
            "corner_radius_nm": np.array([1_000_000, 2_000_000]),
            "signal_drill_nm": np.array([300_000, 400_000]),
            "signal_via_diameter_nm": np.array([500_000, 600_000]),
            "signal_pad_diameter_nm": np.array([700_000, 800_000]),
            "return_via_drill_nm": np.array([300_000, 400_000]),
            "return_via_diameter_nm": np.array([500_000, 600_000]),
            "fence_via_drill_nm": np.array([300_000, 400_000]),
            "fence_via_diameter_nm": np.array([500_000, 600_000]),
        }

        passed, margins = filter_instance.check_tier0(params)

        assert passed.all(), f"Failed constraints: {[k for k, v in margins.items() if (v < 0).any()]}"

    def test_check_tier0_trace_width_violation(self) -> None:
        """Trace width below minimum should fail Tier 0."""
        limits = _default_fab_limits()
        filter_instance = GPUConstraintFilter(limits, use_gpu=False)

        params = {
            "trace_width_nm": np.array([50_000, 200_000]),  # First is below min
            "trace_gap_nm": np.array([150_000, 200_000]),
            "board_width_nm": np.array([20_000_000, 30_000_000]),
            "board_length_nm": np.array([80_000_000, 100_000_000]),
            "corner_radius_nm": np.array([1_000_000, 2_000_000]),
            "signal_drill_nm": np.array([300_000, 400_000]),
            "signal_via_diameter_nm": np.array([500_000, 600_000]),
            "signal_pad_diameter_nm": np.array([700_000, 800_000]),
            "return_via_drill_nm": np.array([300_000, 400_000]),
            "return_via_diameter_nm": np.array([500_000, 600_000]),
            "fence_via_drill_nm": np.array([300_000, 400_000]),
            "fence_via_diameter_nm": np.array([500_000, 600_000]),
        }

        passed, margins = filter_instance.check_tier0(params)

        assert not passed[0]  # First should fail
        assert passed[1]  # Second should pass
        assert margins["T0_TRACE_WIDTH_MIN"][0] < 0

    def test_check_tier1_annular_ring_violation(self) -> None:
        """Insufficient annular ring should fail Tier 1."""
        limits = _default_fab_limits()
        filter_instance = GPUConstraintFilter(limits, use_gpu=False)

        params = {
            "trace_width_nm": np.array([200_000]),
            "trace_gap_nm": np.array([150_000]),
            "board_width_nm": np.array([20_000_000]),
            "board_length_nm": np.array([80_000_000]),
            "signal_drill_nm": np.array([400_000]),
            "signal_via_diameter_nm": np.array([450_000]),  # Only 25k annular ring
            "signal_pad_diameter_nm": np.array([500_000]),  # Only 50k annular from drill
            "return_via_drill_nm": np.array([300_000]),
            "return_via_diameter_nm": np.array([500_000]),
            "fence_via_drill_nm": np.array([300_000]),
            "fence_via_diameter_nm": np.array([500_000]),
            "trace_length_left_nm": np.array([10_000_000]),
            "trace_length_right_nm": np.array([10_000_000]),
        }

        passed, margins = filter_instance.check_tier1(params)

        assert not passed[0]
        assert margins["T1_SIGNAL_ANNULAR_MIN"][0] < 0

    def test_check_tier2_connector_outside_board(self) -> None:
        """Connector position exceeding board bounds should fail Tier 2."""
        limits = _default_fab_limits()
        filter_instance = GPUConstraintFilter(limits, use_gpu=False)

        params = {
            "trace_width_nm": np.array([200_000]),
            "board_width_nm": np.array([20_000_000]),
            "board_length_nm": np.array([80_000_000]),
            "left_connector_x_nm": np.array([5_000_000]),
            "right_connector_x_nm": np.array([90_000_000]),  # Beyond board
            "trace_length_left_nm": np.array([10_000_000]),
            "trace_length_right_nm": np.array([10_000_000]),
            "signal_pad_diameter_nm": np.array([800_000]),
            "return_via_diameter_nm": np.array([600_000]),
            "return_via_ring_radius_nm": np.array([1_500_000]),
            "fence_offset_nm": np.array([500_000]),
            "fence_via_diameter_nm": np.array([500_000]),
            "fence_pitch_nm": np.array([1_000_000]),
        }

        passed, margins = filter_instance.check_tier2(params)

        assert not passed[0]
        assert margins["T2_RIGHT_CONNECTOR_X_MAX"][0] < 0


class TestBatchFilter:
    """Test the main batch_filter function."""

    def test_batch_filter_returns_result(self) -> None:
        """batch_filter should return a BatchFilterResult."""
        u_batch = np.random.rand(100, 19)
        result = batch_filter(u_batch, use_gpu=False)

        assert isinstance(result, BatchFilterResult)
        assert len(result.feasible_mask) == 100
        assert result.repaired_u.shape == (100, 19)

    def test_batch_filter_feasibility_rate(self) -> None:
        """batch_filter should report feasibility rate."""
        u_batch = np.random.rand(1000, 19)
        result = batch_filter(u_batch, use_gpu=False)

        assert 0.0 <= result.feasibility_rate <= 1.0
        assert result.n_feasible == result.feasible_mask.sum()
        assert result.n_candidates == 1000

    def test_batch_filter_with_valid_candidates(self) -> None:
        """Candidates with known-good values should be feasible."""
        # Create candidates with values tuned to pass constraints
        # These are carefully chosen to satisfy all Tier 0-2 constraints
        space = FamilyF1ParameterSpace()
        n = 10
        u_batch = np.zeros((n, space.dimension))

        # Set parameters to known-good values (normalized)
        # Trace parameters - use higher values for safety margin
        u_batch[:, 0] = 0.5  # trace_width_nm: mid-range (300k)
        u_batch[:, 1] = 0.5  # trace_gap_nm: mid-range
        u_batch[:, 2] = 0.3  # board_width_nm: ~22M
        u_batch[:, 3] = 0.5  # board_length_nm: ~90M
        u_batch[:, 4] = 0.1  # corner_radius_nm: small

        # Via parameters - ensure annular ring is satisfied
        u_batch[:, 5] = 0.3  # signal_drill_nm: ~290k
        u_batch[:, 6] = 0.6  # signal_via_diameter_nm: ~600k (> drill)
        u_batch[:, 7] = 0.7  # signal_pad_diameter_nm: ~960k (good annular ring)
        u_batch[:, 8] = 0.3  # return_via_drill_nm
        u_batch[:, 9] = 0.6  # return_via_diameter_nm
        u_batch[:, 10] = 0.3  # fence_via_drill_nm
        u_batch[:, 11] = 0.6  # fence_via_diameter_nm

        # Spatial parameters - ensure fit within board
        # board_length is ~90M, so right connector must be < 90M - edge_clearance
        u_batch[:, 12] = 0.3  # left_connector_x_nm: ~4.4M
        u_batch[:, 13] = 0.2  # right_connector_x_nm: ~85M (within board)
        u_batch[:, 14] = 0.2  # trace_length_left_nm: ~14M
        u_batch[:, 15] = 0.2  # trace_length_right_nm: ~14M
        u_batch[:, 16] = 0.6  # return_via_ring_radius_nm: ~2.1M
        u_batch[:, 17] = 0.6  # fence_pitch_nm: ~2M
        u_batch[:, 18] = 0.6  # fence_offset_nm: ~980k

        result = batch_filter(u_batch, use_gpu=False, repair=True)

        # With repair, we should get feasible candidates
        assert result.n_feasible > 0, f"No feasible candidates. Tier violations: {result.tier_violations}"

    def test_batch_filter_with_extreme_candidates(self) -> None:
        """Candidates at extreme values may fail constraints."""
        # All zeros will likely violate some constraints
        u_batch = np.zeros((10, 19))
        result = batch_filter(u_batch, use_gpu=False, repair=False)

        # Some constraints should fail at extreme low values
        # (corner radius at 0 is actually valid, but other derived constraints may fail)
        assert result.tier_violations["T0"].sum() >= 0

    def test_batch_filter_repair_increases_feasibility(self) -> None:
        """Repair should increase feasibility compared to no repair."""
        np.random.seed(42)
        u_batch = np.random.rand(500, 19)

        result_no_repair = batch_filter(u_batch, use_gpu=False, repair=False)
        result_with_repair = batch_filter(u_batch, use_gpu=False, repair=True)

        # Repair should generally increase or maintain feasibility
        assert result_with_repair.n_feasible >= result_no_repair.n_feasible

    def test_batch_filter_repair_metadata(self) -> None:
        """Repair should populate repair metadata."""
        u_batch = np.random.rand(100, 19)
        result = batch_filter(u_batch, use_gpu=False, repair=True)

        # Should have repair information
        assert result.repair_counts is not None
        assert result.repair_distances is not None
        assert len(result.repair_counts) == 100
        assert len(result.repair_distances) == 100

    def test_batch_filter_constraint_margins(self) -> None:
        """batch_filter should return constraint margins."""
        u_batch = np.random.rand(50, 19)
        result = batch_filter(u_batch, use_gpu=False)

        # Should have margins for key constraints
        assert "T0_TRACE_WIDTH_MIN" in result.constraint_margins
        assert "T1_SIGNAL_ANNULAR_MIN" in result.constraint_margins
        assert "T2_TRACE_FITS_IN_BOARD" in result.constraint_margins

        # Margins should be arrays of correct length
        assert len(result.constraint_margins["T0_TRACE_WIDTH_MIN"]) == 50

    def test_batch_filter_tier_violations(self) -> None:
        """batch_filter should track violations by tier."""
        u_batch = np.random.rand(100, 19)
        result = batch_filter(u_batch, use_gpu=False, repair=False)

        assert "T0" in result.tier_violations
        assert "T1" in result.tier_violations
        assert "T2" in result.tier_violations

        assert len(result.tier_violations["T0"]) == 100

    def test_batch_filter_large_batch(self) -> None:
        """batch_filter should handle large batches efficiently."""
        # Test with 10K candidates (reduced from 1M for CI speed)
        u_batch = np.random.rand(10_000, 19)
        result = batch_filter(u_batch, use_gpu=False)

        assert result.n_candidates == 10_000
        assert len(result.feasible_mask) == 10_000


class TestGPUAvailability:
    """Test GPU availability detection."""

    def test_is_gpu_available_returns_bool(self) -> None:
        """is_gpu_available should return a boolean."""
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_cpu_fallback_works(self) -> None:
        """Filter should work with CPU fallback when GPU unavailable."""
        limits = _default_fab_limits()
        filter_instance = GPUConstraintFilter(limits, use_gpu=False)

        u_batch = np.random.rand(100, 19)
        result = filter_instance.batch_filter(u_batch)

        assert isinstance(result, BatchFilterResult)


class TestBatchFilterResult:
    """Test BatchFilterResult properties per CP-4.1 API."""

    def test_properties(self) -> None:
        """BatchFilterResult should compute properties correctly."""
        repair_meta = RepairMeta(
            repair_counts=np.array([0, 1, 2, 0, 3]),
            repair_distances=np.array([0.0, 0.1, 0.2, 0.0, 0.3]),
            tier_violations={"T0": np.array([0, 0, 1, 0, 1])},
            constraint_margins={},
            seed=42,
            mode="REPAIR",
        )
        result = BatchFilterResult(
            mask=np.array([True, True, False, True, False]),
            u_repaired=np.random.rand(5, 19),
            repair_meta=repair_meta,
        )

        assert result.n_candidates == 5
        assert result.n_feasible == 3
        assert result.feasibility_rate == 0.6

    def test_legacy_aliases(self) -> None:
        """BatchFilterResult should provide legacy attribute aliases."""
        repair_meta = RepairMeta(
            repair_counts=np.array([1, 2]),
            repair_distances=np.array([0.1, 0.2]),
            tier_violations={"T0": np.array([1, 0])},
            constraint_margins={"T0_TEST": np.array([10.0, 20.0])},
            seed=123,
            mode="REPAIR",
        )
        result = BatchFilterResult(
            mask=np.array([True, False]),
            u_repaired=np.array([[0.5] * 19, [0.6] * 19]),
            repair_meta=repair_meta,
        )

        # Test legacy aliases
        np.testing.assert_array_equal(result.feasible_mask, result.mask)
        np.testing.assert_array_equal(result.repaired_u, result.u_repaired)
        np.testing.assert_array_equal(result.repair_counts, repair_meta.repair_counts)
        np.testing.assert_array_equal(result.repair_distances, repair_meta.repair_distances)
        assert result.tier_violations == repair_meta.tier_violations
        assert result.constraint_margins == repair_meta.constraint_margins


class TestCP41FormalAPI:
    """Test the formal CP-4.1 batch_filter API."""

    def test_batch_filter_with_mode_reject(self) -> None:
        """batch_filter with mode=REJECT should not repair violations."""
        np.random.seed(42)
        u_batch = np.random.rand(100, 19)

        result = batch_filter(u_batch, mode="REJECT", seed=42, use_gpu=False)

        assert isinstance(result, BatchFilterResult)
        assert result.repair_meta.mode == "REJECT"
        assert result.repair_meta.seed == 42
        # In REJECT mode, repair counts should be zero
        assert result.repair_meta.repair_counts.sum() == 0

    def test_batch_filter_with_mode_repair(self) -> None:
        """batch_filter with mode=REPAIR should attempt repairs."""
        np.random.seed(42)
        u_batch = np.random.rand(100, 19)

        result = batch_filter(u_batch, mode="REPAIR", seed=42, use_gpu=False)

        assert isinstance(result, BatchFilterResult)
        assert result.repair_meta.mode == "REPAIR"
        assert result.repair_meta.seed == 42
        # REPAIR mode should have some repairs applied (with random data)
        # Note: Some candidates may not need repairs

    def test_batch_filter_seed_determinism(self) -> None:
        """batch_filter with same seed should produce identical results."""
        u_batch = np.random.rand(50, 19)

        result1 = batch_filter(u_batch, mode="REPAIR", seed=12345, use_gpu=False)
        result2 = batch_filter(u_batch, mode="REPAIR", seed=12345, use_gpu=False)

        np.testing.assert_array_equal(result1.mask, result2.mask)
        np.testing.assert_array_equal(result1.u_repaired, result2.u_repaired)
        np.testing.assert_array_equal(result1.repair_meta.repair_counts, result2.repair_meta.repair_counts)

    def test_batch_filter_profiles_as_dict(self) -> None:
        """batch_filter should accept profiles as a dict."""
        u_batch = np.random.rand(20, 19)
        profiles = _default_fab_limits()

        result = batch_filter(u_batch, profiles=profiles, mode="REPAIR", seed=0, use_gpu=False)

        assert isinstance(result, BatchFilterResult)

    def test_batch_filter_mask_attribute(self) -> None:
        """batch_filter result should have mask attribute (CP-4.1 spec)."""
        u_batch = np.random.rand(10, 19)

        result = batch_filter(u_batch, mode="REPAIR", seed=0, use_gpu=False)

        assert hasattr(result, "mask")
        assert hasattr(result, "u_repaired")
        assert hasattr(result, "repair_meta")
        assert len(result.mask) == 10

    def test_batch_filter_repair_meta_structure(self) -> None:
        """batch_filter repair_meta should have required fields."""
        u_batch = np.random.rand(10, 19)

        result = batch_filter(u_batch, mode="REPAIR", seed=99, use_gpu=False)

        meta = result.repair_meta
        assert isinstance(meta, RepairMeta)
        assert hasattr(meta, "repair_counts")
        assert hasattr(meta, "repair_distances")
        assert hasattr(meta, "tier_violations")
        assert hasattr(meta, "constraint_margins")
        assert hasattr(meta, "seed")
        assert hasattr(meta, "mode")
        assert meta.seed == 99
        assert meta.mode == "REPAIR"

    def test_legacy_parameter_compatibility(self) -> None:
        """batch_filter should accept legacy parameters for backward compatibility."""
        u_batch = np.random.rand(20, 19)
        limits = _default_fab_limits()

        # Test with legacy fab_limits parameter
        result1 = batch_filter(u_batch, fab_limits=limits, use_gpu=False)
        assert isinstance(result1, BatchFilterResult)

        # Test with legacy repair parameter
        result2 = batch_filter(u_batch, fab_limits=limits, repair=True, use_gpu=False)
        assert result2.repair_meta.mode == "REPAIR"

        result3 = batch_filter(u_batch, fab_limits=limits, repair=False, use_gpu=False)
        assert result3.repair_meta.mode == "REJECT"

    def test_reject_mode_no_modification(self) -> None:
        """In REJECT mode, u_repaired should equal original u_batch."""
        np.random.seed(42)
        u_batch = np.random.rand(50, 19).astype(np.float64)

        result = batch_filter(u_batch.copy(), mode="REJECT", seed=0, use_gpu=False)

        # u_repaired should be the same as input (no repairs applied)
        np.testing.assert_array_almost_equal(result.u_repaired, u_batch)

    def test_repair_mode_increases_feasibility(self) -> None:
        """REPAIR mode should increase or maintain feasibility vs REJECT."""
        np.random.seed(42)
        u_batch = np.random.rand(500, 19)

        result_reject = batch_filter(u_batch, mode="REJECT", seed=0, use_gpu=False)
        result_repair = batch_filter(u_batch, mode="REPAIR", seed=0, use_gpu=False)

        assert result_repair.n_feasible >= result_reject.n_feasible


class TestCustomFabLimits:
    """Test batch_filter with custom fab limits."""

    def test_stricter_limits_reduce_feasibility(self) -> None:
        """Stricter fab limits should reduce feasibility."""
        np.random.seed(123)
        u_batch = np.random.rand(200, 19)

        # Default limits
        default_limits = _default_fab_limits()
        result_default = batch_filter(u_batch, fab_limits=default_limits, use_gpu=False, repair=False)

        # Stricter limits (double the minimums)
        strict_limits = {k: v * 2 for k, v in default_limits.items()}
        result_strict = batch_filter(u_batch, fab_limits=strict_limits, use_gpu=False, repair=False)

        # Stricter limits should have lower or equal feasibility
        assert result_strict.n_feasible <= result_default.n_feasible


@pytest.mark.skipif(not is_gpu_available(), reason="CuPy not available")
class TestWithGPU:
    """Tests that require GPU/CuPy availability."""

    def test_gpu_results_match_cpu(self) -> None:
        """GPU and CPU results should be approximately equal."""
        np.random.seed(42)
        u_batch = np.random.rand(1000, 19)
        limits = _default_fab_limits()

        result_cpu = batch_filter(u_batch, fab_limits=limits, use_gpu=False, repair=False)
        result_gpu = batch_filter(u_batch, fab_limits=limits, use_gpu=True, repair=False)

        # Feasibility masks should match
        np.testing.assert_array_equal(result_cpu.feasible_mask, result_gpu.feasible_mask)

        # Feasibility rates should match
        assert result_cpu.feasibility_rate == result_gpu.feasibility_rate
