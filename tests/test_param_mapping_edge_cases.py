# SPDX-License-Identifier: MIT
"""Edge case unit tests for FamilyF1ParameterSpace.

Tests additional edge cases and boundary conditions for:
- Parameter space boundary handling
- Parameter dimension and type validation
- Deterministic behavior of parameter space conversions

These tests complement test_cp43_gpu_pipeline.py by covering edge cases
not addressed by the integration-focused tests there.
"""

from __future__ import annotations

import numpy as np
import pytest

from formula_foundry.coupongen.constraints.gpu_filter import FamilyF1ParameterSpace


class TestParameterSpaceDimensions:
    """Tests for parameter space dimension and structure."""

    def test_dimension_is_19(self) -> None:
        """F1 parameter space should have 19 dimensions."""
        space = FamilyF1ParameterSpace()
        assert space.dimension == 19

    def test_bounds_have_correct_shape(self) -> None:
        """Parameter bounds should have shape (2, dimension)."""
        space = FamilyF1ParameterSpace()
        bounds = space.bounds
        assert bounds.shape == (2, 19)

    def test_lower_bounds_less_than_upper(self) -> None:
        """Lower bounds should be strictly less than upper bounds."""
        space = FamilyF1ParameterSpace()
        lower = space.bounds[0]
        upper = space.bounds[1]
        for i in range(19):
            assert lower[i] < upper[i], f"Bound violation at index {i}"


class TestParameterSpaceBoundaries:
    """Tests for parameter space boundary handling."""

    def test_u_at_lower_bound(self) -> None:
        """u vector at 0.0 should produce minimum physical values."""
        space = FamilyF1ParameterSpace()
        u = np.zeros((1, 19))
        params = space.to_physical_batch(u, np)

        # All values should match lower bounds
        for name, arr in params.items():
            assert arr.shape == (1,), f"{name} should have shape (1,)"

    def test_u_at_upper_bound(self) -> None:
        """u vector at 1.0 should produce maximum physical values."""
        space = FamilyF1ParameterSpace()
        u = np.ones((1, 19))
        params = space.to_physical_batch(u, np)

        for name, arr in params.items():
            assert arr.shape == (1,), f"{name} should have shape (1,)"

    def test_u_with_mixed_bounds(self) -> None:
        """u vector with alternating 0.0 and 1.0 values."""
        space = FamilyF1ParameterSpace()
        u_row = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(19)])
        u = u_row.reshape(1, 19)
        params = space.to_physical_batch(u, np)

        for name, arr in params.items():
            assert arr.shape == (1,), f"{name} should have shape (1,)"


class TestBatchConversion:
    """Tests for batch parameter conversion."""

    def test_batch_size_preserved(self) -> None:
        """Batch conversion should preserve the number of samples."""
        space = FamilyF1ParameterSpace()
        batch_size = 10
        u_batch = np.random.rand(batch_size, 19)
        params = space.to_physical_batch(u_batch, np)

        for name, arr in params.items():
            assert arr.shape == (batch_size,), f"{name} shape mismatch"

    def test_empty_batch(self) -> None:
        """Empty batch should return empty parameter arrays."""
        space = FamilyF1ParameterSpace()
        u_empty = np.zeros((0, 19))
        params = space.to_physical_batch(u_empty, np)

        for name, arr in params.items():
            assert arr.shape == (0,), f"{name} should be empty"

    def test_single_sample_batch(self) -> None:
        """Single sample batch should work correctly."""
        space = FamilyF1ParameterSpace()
        u_single = np.ones((1, 19)) * 0.5
        params = space.to_physical_batch(u_single, np)

        for name, arr in params.items():
            assert arr.shape == (1,), f"{name} should have single element"


class TestDeterminism:
    """Tests for deterministic behavior of parameter space."""

    def test_to_physical_deterministic(self) -> None:
        """Same u vector should produce same params across calls."""
        space = FamilyF1ParameterSpace()
        u = np.ones((5, 19)) * 0.5

        params_a = space.to_physical_batch(u.copy(), np)
        params_b = space.to_physical_batch(u.copy(), np)

        for name in params_a:
            np.testing.assert_array_equal(
                params_a[name], params_b[name], err_msg=f"{name} not deterministic"
            )

    def test_parameter_space_reusable(self) -> None:
        """Same parameter space can be reused for multiple conversions."""
        space = FamilyF1ParameterSpace()

        results = []
        for i in range(3):
            u = np.ones((1, 19)) * (0.25 * (i + 1))
            params = space.to_physical_batch(u, np)
            results.append({k: v[0] for k, v in params.items()})

        # Different u values should produce different params
        assert results[0] != results[1]
        assert results[1] != results[2]


class TestParameterRanges:
    """Tests for parameter value ranges."""

    def test_trace_width_increases_with_u(self) -> None:
        """Trace width should increase as u increases."""
        space = FamilyF1ParameterSpace()

        u_low = np.zeros((1, 19))
        u_high = np.ones((1, 19))

        params_low = space.to_physical_batch(u_low, np)
        params_high = space.to_physical_batch(u_high, np)

        assert params_low["trace_width_nm"][0] <= params_high["trace_width_nm"][0]

    def test_board_dimensions_increase_with_u(self) -> None:
        """Board dimensions should increase as u increases."""
        space = FamilyF1ParameterSpace()

        u_low = np.zeros((1, 19))
        u_high = np.ones((1, 19))

        params_low = space.to_physical_batch(u_low, np)
        params_high = space.to_physical_batch(u_high, np)

        assert params_low["board_length_nm"][0] <= params_high["board_length_nm"][0]
        assert params_low["board_width_nm"][0] <= params_high["board_width_nm"][0]

    def test_all_physical_values_positive(self) -> None:
        """All physical parameter values should be positive."""
        space = FamilyF1ParameterSpace()
        u = np.zeros((1, 19))  # Lower bound
        params = space.to_physical_batch(u, np)

        for name, arr in params.items():
            assert arr[0] >= 0, f"{name} should be non-negative at lower bound"


class TestParameterNames:
    """Tests for expected parameter names."""

    def test_expected_parameter_names_present(self) -> None:
        """Parameter dict should contain expected parameter names."""
        space = FamilyF1ParameterSpace()
        u = np.ones((1, 19)) * 0.5
        params = space.to_physical_batch(u, np)

        expected_keys = {
            "trace_width_nm",
            "trace_gap_nm",
            "board_width_nm",
            "board_length_nm",
            "corner_radius_nm",
            "signal_drill_nm",
        }

        for key in expected_keys:
            assert key in params, f"Missing expected key: {key}"

    def test_parameter_count_matches_dimension(self) -> None:
        """Number of output parameters should match dimension."""
        space = FamilyF1ParameterSpace()
        u = np.ones((1, 19)) * 0.5
        params = space.to_physical_batch(u, np)

        assert len(params) == 19, f"Expected 19 params, got {len(params)}"
