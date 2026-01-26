# SPDX-License-Identifier: MIT
"""Unit tests for param_mapping module main entry points.

Tests the main functions in param_mapping.py which map normalized u vectors
to CouponSpec instances for the F1 family. These functions are essential for
batch coupon generation and GPU-accelerated design space exploration.

This complements test_param_mapping_edge_cases.py which focuses on
FamilyF1ParameterSpace edge cases, while these tests focus on the
higher-level mapping functions:
  - get_f1_parameter_space
  - u_to_spec_params_f1
  - apply_params_to_spec
  - u_to_spec_f1
  - batch_u_to_specs_f1

Per CP-4.3: Parameter mapping must be deterministic and consistent
with the GPU filter's parameter space definition.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_f1_spec_template_dict() -> dict[str, Any]:
    """Return a minimal valid F1 spec dictionary for use as template."""
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
            "thicknesses_nm": {"core": 800_000, "prepreg": 180_000},
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
            "w_nm": 300_000,
            "gap_nm": 180_000,
            "length_left_nm": 25_000_000,
            "length_right_nm": 25_000_000,
            "ground_via_fence": {
                "enabled": True,
                "pitch_nm": 1_500_000,
                "offset_from_gap_nm": 800_000,
                "via": {"drill_nm": 300_000, "diameter_nm": 600_000},
            },
        },
        "discontinuity": {
            "type": "VIA_TRANSITION",
            "signal_via": {
                "drill_nm": 300_000,
                "diameter_nm": 650_000,
                "pad_diameter_nm": 900_000,
            },
            "antipads": {
                "L2": {
                    "shape": "ROUNDRECT",
                    "rx_nm": 1_200_000,
                    "ry_nm": 900_000,
                    "corner_nm": 250_000,
                },
            },
            "return_vias": {
                "pattern": "RING",
                "count": 4,
                "radius_nm": 1_700_000,
                "via": {"drill_nm": 300_000, "diameter_nm": 650_000},
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


@pytest.fixture
def f1_spec_template(valid_f1_spec_template_dict: dict[str, Any]):
    """Return a validated CouponSpec instance from the template dict."""
    from formula_foundry.coupongen.spec import CouponSpec

    return CouponSpec.model_validate(valid_f1_spec_template_dict)


# =============================================================================
# get_f1_parameter_space tests
# =============================================================================


class TestGetF1ParameterSpace:
    """Tests for get_f1_parameter_space function."""

    def test_returns_f1_parameter_space(self) -> None:
        """Function should return FamilyF1ParameterSpace instance."""
        from formula_foundry.coupongen.constraints.gpu_filter import FamilyF1ParameterSpace
        from formula_foundry.coupongen.param_mapping import get_f1_parameter_space

        result = get_f1_parameter_space()
        assert isinstance(result, FamilyF1ParameterSpace)

    def test_returns_new_instance_each_call(self) -> None:
        """Each call should return a fresh instance."""
        from formula_foundry.coupongen.param_mapping import get_f1_parameter_space

        space1 = get_f1_parameter_space()
        space2 = get_f1_parameter_space()
        assert space1 is not space2

    def test_parameter_space_has_correct_dimension(self) -> None:
        """Returned space should have dimension 19."""
        from formula_foundry.coupongen.param_mapping import get_f1_parameter_space

        space = get_f1_parameter_space()
        assert space.dimension == 19


# =============================================================================
# u_to_spec_params_f1 tests
# =============================================================================


class TestUToSpecParamsF1:
    """Tests for u_to_spec_params_f1 function."""

    def test_returns_dict_of_ints(self) -> None:
        """Should return dict mapping param names to int values."""
        from formula_foundry.coupongen.param_mapping import u_to_spec_params_f1

        u = np.ones(19) * 0.5
        params = u_to_spec_params_f1(u)

        assert isinstance(params, dict)
        for key, value in params.items():
            assert isinstance(key, str)
            assert isinstance(value, int), f"{key} should be int, got {type(value)}"

    def test_expected_parameter_names(self) -> None:
        """Should include expected parameter names."""
        from formula_foundry.coupongen.param_mapping import u_to_spec_params_f1

        u = np.ones(19) * 0.5
        params = u_to_spec_params_f1(u)

        expected_keys = {
            "trace_width_nm",
            "trace_gap_nm",
            "board_width_nm",
            "board_length_nm",
            "corner_radius_nm",
        }
        for key in expected_keys:
            assert key in params, f"Missing expected key: {key}"

    def test_parameter_count(self) -> None:
        """Should return 19 parameters."""
        from formula_foundry.coupongen.param_mapping import u_to_spec_params_f1

        u = np.ones(19) * 0.5
        params = u_to_spec_params_f1(u)

        assert len(params) == 19

    def test_deterministic_output(self) -> None:
        """Same u vector should produce same params."""
        from formula_foundry.coupongen.param_mapping import u_to_spec_params_f1

        u = np.ones(19) * 0.75
        params1 = u_to_spec_params_f1(u.copy())
        params2 = u_to_spec_params_f1(u.copy())

        assert params1 == params2

    def test_all_values_positive(self) -> None:
        """All physical values should be positive."""
        from formula_foundry.coupongen.param_mapping import u_to_spec_params_f1

        u = np.zeros(19)  # Lower bounds
        params = u_to_spec_params_f1(u)

        for key, value in params.items():
            assert value >= 0, f"{key} should be non-negative"

    def test_accepts_custom_param_space(self) -> None:
        """Should accept custom parameter space."""
        from formula_foundry.coupongen.constraints.gpu_filter import FamilyF1ParameterSpace
        from formula_foundry.coupongen.param_mapping import u_to_spec_params_f1

        custom_space = FamilyF1ParameterSpace()
        u = np.ones(19) * 0.5
        params = u_to_spec_params_f1(u, param_space=custom_space)

        assert len(params) == 19


# =============================================================================
# apply_params_to_spec tests
# =============================================================================


class TestApplyParamsToSpec:
    """Tests for apply_params_to_spec function."""

    def test_returns_couponspec(self, f1_spec_template) -> None:
        """Should return a CouponSpec instance."""
        from formula_foundry.coupongen.param_mapping import apply_params_to_spec
        from formula_foundry.coupongen.spec import CouponSpec

        params = {"trace_width_nm": 500_000}
        result = apply_params_to_spec(f1_spec_template, params)

        assert isinstance(result, CouponSpec)

    def test_applies_trace_width(self, f1_spec_template) -> None:
        """Should apply trace_width_nm to transmission_line.w_nm."""
        from formula_foundry.coupongen.param_mapping import apply_params_to_spec

        params = {"trace_width_nm": 500_000}
        result = apply_params_to_spec(f1_spec_template, params)

        assert result.transmission_line.w_nm == 500_000

    def test_applies_trace_gap(self, f1_spec_template) -> None:
        """Should apply trace_gap_nm to transmission_line.gap_nm."""
        from formula_foundry.coupongen.param_mapping import apply_params_to_spec

        params = {"trace_gap_nm": 250_000}
        result = apply_params_to_spec(f1_spec_template, params)

        assert result.transmission_line.gap_nm == 250_000

    def test_applies_board_dimensions(self, f1_spec_template) -> None:
        """Should apply board dimensions."""
        from formula_foundry.coupongen.param_mapping import apply_params_to_spec

        params = {
            "board_width_nm": 25_000_000,
            "board_length_nm": 100_000_000,
        }
        result = apply_params_to_spec(f1_spec_template, params)

        assert result.board.outline.width_nm == 25_000_000
        assert result.board.outline.length_nm == 100_000_000

    def test_applies_corner_radius(self, f1_spec_template) -> None:
        """Should apply corner_radius_nm."""
        from formula_foundry.coupongen.param_mapping import apply_params_to_spec

        params = {"corner_radius_nm": 3_000_000}
        result = apply_params_to_spec(f1_spec_template, params)

        assert result.board.outline.corner_radius_nm == 3_000_000

    def test_applies_signal_via_params(self, f1_spec_template) -> None:
        """Should apply signal via parameters."""
        from formula_foundry.coupongen.param_mapping import apply_params_to_spec

        params = {
            "signal_drill_nm": 400_000,
            "signal_via_diameter_nm": 800_000,
            "signal_pad_diameter_nm": 1_000_000,
        }
        result = apply_params_to_spec(f1_spec_template, params)

        assert result.discontinuity.signal_via.drill_nm == 400_000
        assert result.discontinuity.signal_via.diameter_nm == 800_000
        assert result.discontinuity.signal_via.pad_diameter_nm == 1_000_000

    def test_applies_fence_params(self, f1_spec_template) -> None:
        """Should apply fence via parameters."""
        from formula_foundry.coupongen.param_mapping import apply_params_to_spec

        params = {
            "fence_pitch_nm": 2_000_000,
            "fence_offset_nm": 900_000,
            "fence_via_drill_nm": 350_000,
            "fence_via_diameter_nm": 700_000,
        }
        result = apply_params_to_spec(f1_spec_template, params)

        fence = result.transmission_line.ground_via_fence
        assert fence.pitch_nm == 2_000_000
        assert fence.offset_from_gap_nm == 900_000
        assert fence.via.drill_nm == 350_000
        assert fence.via.diameter_nm == 700_000

    def test_applies_trace_lengths(self, f1_spec_template) -> None:
        """Should apply trace length parameters."""
        from formula_foundry.coupongen.param_mapping import apply_params_to_spec

        params = {
            "trace_length_left_nm": 30_000_000,
            "trace_length_right_nm": 35_000_000,
        }
        result = apply_params_to_spec(f1_spec_template, params)

        assert result.transmission_line.length_left_nm == 30_000_000
        assert result.transmission_line.length_right_nm == 35_000_000

    def test_returns_validated_spec(self, f1_spec_template) -> None:
        """Result should be a validated CouponSpec."""
        from formula_foundry.coupongen.param_mapping import apply_params_to_spec

        params = {"trace_width_nm": 400_000}
        result = apply_params_to_spec(f1_spec_template, params)

        # Should not raise - spec is valid
        _ = result.model_dump()

    def test_preserves_unchanged_fields(self, f1_spec_template) -> None:
        """Fields not in params should remain unchanged."""
        from formula_foundry.coupongen.param_mapping import apply_params_to_spec

        original_family = f1_spec_template.coupon_family
        original_stackup = f1_spec_template.stackup.copper_layers

        params = {"trace_width_nm": 400_000}
        result = apply_params_to_spec(f1_spec_template, params)

        assert result.coupon_family == original_family
        assert result.stackup.copper_layers == original_stackup

    def test_empty_params_returns_copy(self, f1_spec_template) -> None:
        """Empty params dict should return equivalent spec."""
        from formula_foundry.coupongen.param_mapping import apply_params_to_spec

        params: dict[str, int] = {}
        result = apply_params_to_spec(f1_spec_template, params)

        # Core fields should match
        assert result.transmission_line.w_nm == f1_spec_template.transmission_line.w_nm
        assert result.board.outline.width_nm == f1_spec_template.board.outline.width_nm


# =============================================================================
# u_to_spec_f1 tests
# =============================================================================


class TestUToSpecF1:
    """Tests for u_to_spec_f1 function."""

    def test_returns_couponspec(self, f1_spec_template) -> None:
        """Should return a CouponSpec instance."""
        from formula_foundry.coupongen.param_mapping import u_to_spec_f1
        from formula_foundry.coupongen.spec import CouponSpec

        u = np.ones(19) * 0.5
        result = u_to_spec_f1(u, f1_spec_template)

        assert isinstance(result, CouponSpec)

    def test_different_u_produces_different_spec(self, f1_spec_template) -> None:
        """Different u vectors should produce different specs."""
        from formula_foundry.coupongen.param_mapping import u_to_spec_f1

        u1 = np.zeros(19)
        u2 = np.ones(19)

        spec1 = u_to_spec_f1(u1, f1_spec_template)
        spec2 = u_to_spec_f1(u2, f1_spec_template)

        # At least some parameters should differ
        assert spec1.transmission_line.w_nm != spec2.transmission_line.w_nm

    def test_deterministic_mapping(self, f1_spec_template) -> None:
        """Same u should produce same spec."""
        from formula_foundry.coupongen.param_mapping import u_to_spec_f1

        u = np.ones(19) * 0.3

        spec1 = u_to_spec_f1(u.copy(), f1_spec_template)
        spec2 = u_to_spec_f1(u.copy(), f1_spec_template)

        assert spec1.transmission_line.w_nm == spec2.transmission_line.w_nm
        assert spec1.transmission_line.gap_nm == spec2.transmission_line.gap_nm
        assert spec1.board.outline.width_nm == spec2.board.outline.width_nm

    def test_preserves_family(self, f1_spec_template) -> None:
        """Coupon family should be preserved."""
        from formula_foundry.coupongen.param_mapping import u_to_spec_f1

        u = np.ones(19) * 0.5
        result = u_to_spec_f1(u, f1_spec_template)

        assert result.coupon_family == "F1_SINGLE_ENDED_VIA"

    def test_accepts_custom_param_space(self, f1_spec_template) -> None:
        """Should accept custom parameter space."""
        from formula_foundry.coupongen.constraints.gpu_filter import FamilyF1ParameterSpace
        from formula_foundry.coupongen.param_mapping import u_to_spec_f1

        custom_space = FamilyF1ParameterSpace()
        u = np.ones(19) * 0.5
        result = u_to_spec_f1(u, f1_spec_template, param_space=custom_space)

        assert result.coupon_family == "F1_SINGLE_ENDED_VIA"


# =============================================================================
# batch_u_to_specs_f1 tests
# =============================================================================


class TestBatchUToSpecsF1:
    """Tests for batch_u_to_specs_f1 function."""

    def test_returns_list_of_specs(self, f1_spec_template) -> None:
        """Should return list of CouponSpec instances."""
        from formula_foundry.coupongen.param_mapping import batch_u_to_specs_f1
        from formula_foundry.coupongen.spec import CouponSpec

        u_batch = np.random.rand(5, 19)
        result = batch_u_to_specs_f1(u_batch, f1_spec_template)

        assert isinstance(result, list)
        assert len(result) == 5
        for spec in result:
            assert isinstance(spec, CouponSpec)

    def test_batch_size_preserved(self, f1_spec_template) -> None:
        """Number of output specs should match batch size."""
        from formula_foundry.coupongen.param_mapping import batch_u_to_specs_f1

        for batch_size in [1, 3, 10]:
            u_batch = np.random.rand(batch_size, 19)
            result = batch_u_to_specs_f1(u_batch, f1_spec_template)
            assert len(result) == batch_size

    def test_empty_batch(self, f1_spec_template) -> None:
        """Empty batch should return empty list."""
        from formula_foundry.coupongen.param_mapping import batch_u_to_specs_f1

        u_batch = np.zeros((0, 19))
        result = batch_u_to_specs_f1(u_batch, f1_spec_template)

        assert result == []

    def test_each_spec_distinct_for_distinct_u(self, f1_spec_template) -> None:
        """Distinct u vectors should produce distinct specs."""
        from formula_foundry.coupongen.param_mapping import batch_u_to_specs_f1

        # Create batch with clearly different u values
        u_batch = np.array(
            [
                [0.0] * 19,
                [0.5] * 19,
                [1.0] * 19,
            ]
        )
        result = batch_u_to_specs_f1(u_batch, f1_spec_template)

        widths = [spec.transmission_line.w_nm for spec in result]
        # All three should be different
        assert len(set(widths)) == 3

    def test_deterministic_batch_processing(self, f1_spec_template) -> None:
        """Same batch should produce same results."""
        from formula_foundry.coupongen.param_mapping import batch_u_to_specs_f1

        u_batch = np.random.rand(3, 19)
        result1 = batch_u_to_specs_f1(u_batch.copy(), f1_spec_template)
        result2 = batch_u_to_specs_f1(u_batch.copy(), f1_spec_template)

        for s1, s2 in zip(result1, result2, strict=False):
            assert s1.transmission_line.w_nm == s2.transmission_line.w_nm
            assert s1.transmission_line.gap_nm == s2.transmission_line.gap_nm

    def test_accepts_custom_param_space(self, f1_spec_template) -> None:
        """Should accept custom parameter space."""
        from formula_foundry.coupongen.constraints.gpu_filter import FamilyF1ParameterSpace
        from formula_foundry.coupongen.param_mapping import batch_u_to_specs_f1

        custom_space = FamilyF1ParameterSpace()
        u_batch = np.random.rand(3, 19)
        result = batch_u_to_specs_f1(u_batch, f1_spec_template, param_space=custom_space)

        assert len(result) == 3


# =============================================================================
# Integration/workflow tests
# =============================================================================


class TestParamMappingWorkflow:
    """Integration tests for the full param mapping workflow."""

    def test_round_trip_workflow(self, f1_spec_template) -> None:
        """Test complete workflow from u to spec."""
        from formula_foundry.coupongen.param_mapping import (
            get_f1_parameter_space,
            u_to_spec_f1,
        )

        # Get parameter space
        param_space = get_f1_parameter_space()

        # Create normalized u vector
        u = np.ones(param_space.dimension) * 0.5

        # Convert to spec
        spec = u_to_spec_f1(u, f1_spec_template, param_space)

        # Verify spec is valid and has expected structure
        assert spec.coupon_family == "F1_SINGLE_ENDED_VIA"
        assert spec.transmission_line.w_nm > 0
        assert spec.transmission_line.gap_nm > 0
        assert spec.board.outline.width_nm > 0

    def test_batch_workflow_consistency(self, f1_spec_template) -> None:
        """Batch processing should be consistent with single processing."""
        from formula_foundry.coupongen.param_mapping import (
            batch_u_to_specs_f1,
            get_f1_parameter_space,
            u_to_spec_f1,
        )

        param_space = get_f1_parameter_space()
        u_batch = np.array(
            [
                [0.25] * 19,
                [0.75] * 19,
            ]
        )

        # Process as batch
        batch_results = batch_u_to_specs_f1(u_batch, f1_spec_template, param_space)

        # Process individually
        single_results = [u_to_spec_f1(u_batch[i], f1_spec_template, param_space) for i in range(len(u_batch))]

        # Results should match
        for batch_spec, single_spec in zip(batch_results, single_results, strict=False):
            assert batch_spec.transmission_line.w_nm == single_spec.transmission_line.w_nm
            assert batch_spec.transmission_line.gap_nm == single_spec.transmission_line.gap_nm
