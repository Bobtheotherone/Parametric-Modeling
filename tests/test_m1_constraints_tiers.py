"""Tests for the tiered constraint system (REQ-M1-008, REQ-M1-009).

This module tests the hierarchical constraint validation:
- Tier 0: Parameter bounds validation
- Tier 1: Derived scalar constraints
- Tier 2: Analytic spatial constraints
- Tier 3: Exact geometry collision detection
- REJECT mode with constraint IDs and reasons
"""

from __future__ import annotations

from typing import Any

import pytest

from formula_foundry.coupongen.constraints.tiers import (
    ConstraintResult,
    ConstraintViolationError,
    Tier0Checker,
    Tier1Checker,
    Tier2Checker,
    Tier3Checker,
    TieredConstraintSystem,
    evaluate_tiered_constraints,
)
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
            # CP-2.2: For F1, length_right_nm is derived from continuity.
            # With left connector at 5mm, right at 75mm (70mm span),
            # and length_left=35mm, discontinuity is at 40mm,
            # so derived length_right = 75 - 40 = 35mm (symmetric).
            "length_left_nm": 35000000,
            "length_right_nm": 35000000,  # Must match derived value for F1
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


class TestTier0ParameterBounds:
    """Test Tier 0 parameter bounds validation."""

    def test_valid_spec_passes_tier0(self) -> None:
        """Valid spec should pass all Tier 0 constraints."""
        spec = CouponSpec.model_validate(_example_spec_data())
        limits = _default_fab_limits()
        checker = Tier0Checker()

        results = checker.check(spec, limits)

        assert all(r.passed for r in results), [r for r in results if not r.passed]
        assert all(r.tier == "T0" for r in results)

    def test_trace_width_below_minimum_fails(self) -> None:
        """Trace width below fab minimum should fail."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # Below 100_000 min
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier0Checker()

        results = checker.check(spec, limits)

        width_result = next(r for r in results if r.constraint_id == "T0_TRACE_WIDTH_MIN")
        assert not width_result.passed
        assert width_result.value == 50_000
        assert width_result.limit == 100_000
        assert width_result.margin == -50_000

    def test_trace_gap_below_minimum_fails(self) -> None:
        """CPWG gap below fab minimum should fail."""
        data = _example_spec_data()
        data["transmission_line"]["gap_nm"] = 50_000  # Below 100_000 min
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier0Checker()

        results = checker.check(spec, limits)

        gap_result = next(r for r in results if r.constraint_id == "T0_TRACE_GAP_MIN")
        assert not gap_result.passed

    def test_signal_via_drill_below_minimum_fails(self) -> None:
        """Signal via drill below fab minimum should fail."""
        data = _example_spec_data()
        data["discontinuity"]["signal_via"]["drill_nm"] = 100_000  # Below 200_000 min
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier0Checker()

        results = checker.check(spec, limits)

        drill_result = next(r for r in results if r.constraint_id == "T0_SIGNAL_DRILL_MIN")
        assert not drill_result.passed

    def test_corner_radius_exceeds_max_fails(self) -> None:
        """Corner radius exceeding half the min dimension should fail."""
        data = _example_spec_data()
        data["board"]["outline"]["corner_radius_nm"] = 15_000_000  # Exceeds 10_000_000 (half of 20_000_000 width)
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier0Checker()

        results = checker.check(spec, limits)

        corner_result = next(r for r in results if r.constraint_id == "T0_CORNER_RADIUS_MAX")
        assert not corner_result.passed


class TestTier1DerivedScalars:
    """Test Tier 1 derived scalar constraints."""

    def test_valid_spec_passes_tier1(self) -> None:
        """Valid spec should pass all Tier 1 constraints."""
        spec = CouponSpec.model_validate(_example_spec_data())
        limits = _default_fab_limits()
        checker = Tier1Checker()

        results = checker.check(spec, limits)

        assert all(r.passed for r in results), [r for r in results if not r.passed]
        assert all(r.tier == "T1" for r in results)

    def test_annular_ring_too_small_fails(self) -> None:
        """Signal via annular ring below minimum should fail."""
        data = _example_spec_data()
        # Set pad barely larger than drill: annular ring = (500_000 - 400_000) / 2 = 50_000 < 100_000 min
        data["discontinuity"]["signal_via"]["drill_nm"] = 400_000
        data["discontinuity"]["signal_via"]["diameter_nm"] = 450_000
        data["discontinuity"]["signal_via"]["pad_diameter_nm"] = 500_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier1Checker()

        results = checker.check(spec, limits)

        annular_result = next(r for r in results if r.constraint_id == "T1_SIGNAL_ANNULAR_MIN")
        assert not annular_result.passed
        assert annular_result.value == 50_000
        assert annular_result.limit == 100_000

    def test_extreme_board_aspect_ratio_fails(self) -> None:
        """Board with extreme aspect ratio should fail."""
        data = _example_spec_data()
        data["board"]["outline"]["length_nm"] = 500_000_000  # 500mm length
        data["board"]["outline"]["width_nm"] = 10_000_000  # 10mm width = 50:1 ratio
        # Adjust connector positions for the longer board
        data["connectors"]["right"]["position_nm"] = [490_000_000, 0]
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier1Checker()

        results = checker.check(spec, limits)

        aspect_result = next(r for r in results if r.constraint_id == "T1_BOARD_ASPECT_RATIO_MAX")
        assert not aspect_result.passed

    def test_copper_to_edge_clearance_fails(self) -> None:
        """Copper features too close to edge should fail (Section 13.3.2)."""
        data = _example_spec_data()
        # Make board very narrow so copper is too close to edge
        data["board"]["outline"]["width_nm"] = 2_000_000  # 2mm width
        # With trace_width=300000, gap=180000, fence_offset=800000, fence_via_radius=300000
        # Extent from center = 150000 + 180000 + 800000 + 300000 = 1430000 nm
        # Available clearance = 1000000 - 1430000 = -430000 nm (negative = fails)
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier1Checker()

        results = checker.check(spec, limits)

        clearance_result = next(r for r in results if r.constraint_id == "T1_COPPER_TO_EDGE_CLEARANCE")
        assert not clearance_result.passed

    def test_fence_pitch_too_small_fails(self) -> None:
        """Fence pitch smaller than via + spacing should fail (Section 13.3.2)."""
        data = _example_spec_data()
        # Set fence pitch smaller than via diameter + spacing
        data["transmission_line"]["ground_via_fence"]["pitch_nm"] = 500_000  # Too small
        # Via diameter = 600_000, min_via_to_via = 200_000, so min pitch = 800_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier1Checker()

        results = checker.check(spec, limits)

        pitch_result = next(r for r in results if r.constraint_id == "T1_FENCE_PITCH_MIN")
        assert not pitch_result.passed


class TestTier2AnalyticSpatial:
    """Test Tier 2 analytic spatial constraints."""

    def test_valid_spec_passes_tier2(self) -> None:
        """Valid spec should pass all Tier 2 constraints."""
        spec = CouponSpec.model_validate(_example_spec_data())
        limits = _default_fab_limits()
        checker = Tier2Checker()

        results = checker.check(spec, limits)

        assert all(r.passed for r in results), [r for r in results if not r.passed]
        assert all(r.tier == "T2" for r in results)

    def test_connector_outside_board_fails(self) -> None:
        """Connector position outside board bounds should fail."""
        data = _example_spec_data()
        data["connectors"]["right"]["position_nm"] = [100_000_000, 0]  # Beyond 80_000_000 board length
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier2Checker()

        results = checker.check(spec, limits)

        x_max_result = next(r for r in results if r.constraint_id == "T2_RIGHT_CONNECTOR_X_MAX")
        assert not x_max_result.passed

    def test_traces_exceed_available_length_fails(self) -> None:
        """Traces that don't fit between connectors should fail.

        CP-2.2 note: For F1 coupons, length_right is derived from continuity.
        This test explicitly provides length_right_nm to test the T2 constraint
        directly without relying on the derived value. This tests the constraint
        logic when the user explicitly specifies both lengths (deprecated for F1,
        but still validated in T2 if provided).
        """
        data = _example_spec_data()
        # Available length: 75_000_000 - 5_000_000 = 70_000_000nm
        # Set explicit lengths that exceed the available space
        data["transmission_line"]["length_left_nm"] = 40_000_000
        data["transmission_line"]["length_right_nm"] = 40_000_000  # Total 80M > 70M available
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier2Checker()

        results = checker.check(spec, limits)

        fit_result = next(r for r in results if r.constraint_id == "T2_TRACE_FITS_IN_BOARD")
        assert not fit_result.passed


class TestTier3GeometryCollision:
    """Test Tier 3 geometry collision detection."""

    def test_valid_spec_passes_tier3(self) -> None:
        """Valid spec should pass all Tier 3 constraints."""
        spec = CouponSpec.model_validate(_example_spec_data())
        limits = _default_fab_limits()
        checker = Tier3Checker()

        results = checker.check(spec, limits)

        assert all(r.passed for r in results), [r for r in results if not r.passed]
        assert all(r.tier == "T3" for r in results)

    def test_return_vias_overlap_fails(self) -> None:
        """Return vias that overlap should fail."""
        data = _example_spec_data()
        # 8 vias at small radius with large diameter will overlap
        data["discontinuity"]["return_vias"]["count"] = 8
        data["discontinuity"]["return_vias"]["radius_nm"] = 800_000  # Very small radius
        data["discontinuity"]["return_vias"]["via"]["diameter_nm"] = 600_000  # Large via
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier3Checker()

        results = checker.check(spec, limits)

        overlap_result = next(r for r in results if r.constraint_id == "T3_RETURN_VIA_RING_NO_OVERLAP")
        assert not overlap_result.passed

    def test_asymmetric_traces_with_symmetry_enforced_fails(self) -> None:
        """Asymmetric traces with symmetry enforced should fail."""
        data = _example_spec_data()
        # With connectors at 5M and 75M (70M span), set asymmetric left length.
        # For F1, derived right = 75M - (5M + left) = 70M - left
        # If left = 20M, derived right = 50M (asymmetric)
        data["transmission_line"]["length_left_nm"] = 20_000_000
        # Don't set length_right_nm - let it be derived (50M, asymmetric)
        del data["transmission_line"]["length_right_nm"]
        data["constraints"]["symmetry"]["enforce"] = True
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier3Checker()

        results = checker.check(spec, limits)

        symmetry_result = next(r for r in results if r.constraint_id == "T3_TRACE_SYMMETRY")
        assert not symmetry_result.passed

    def test_symmetric_traces_with_symmetry_enforced_passes(self) -> None:
        """Symmetric traces with symmetry enforced should pass."""
        data = _example_spec_data()
        # With connectors at 5M and 75M (70M span), symmetric means:
        # left = right = 35M (discontinuity at 40M, middle of the span)
        data["transmission_line"]["length_left_nm"] = 35_000_000
        # For F1, derived right = 75M - (5M + 35M) = 35M (symmetric)
        del data["transmission_line"]["length_right_nm"]
        data["constraints"]["symmetry"]["enforce"] = True
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier3Checker()

        results = checker.check(spec, limits)

        symmetry_result = next(r for r in results if r.constraint_id == "T3_TRACE_SYMMETRY")
        assert symmetry_result.passed


class TestTieredConstraintSystem:
    """Test the full tiered constraint system."""

    def test_all_tiers_evaluated(self) -> None:
        """System should evaluate all four tiers."""
        spec = CouponSpec.model_validate(_example_spec_data())
        limits = _default_fab_limits()

        proof = evaluate_tiered_constraints(spec, limits)

        assert set(proof.tiers.keys()) == {"T0", "T1", "T2", "T3"}
        # All tiers should have at least one constraint
        assert len(proof.tiers["T0"]) > 0
        assert len(proof.tiers["T1"]) > 0
        assert len(proof.tiers["T2"]) > 0
        assert len(proof.tiers["T3"]) > 0

    def test_valid_spec_passes_all_tiers(self) -> None:
        """Valid spec should pass all tiers."""
        spec = CouponSpec.model_validate(_example_spec_data())
        limits = _default_fab_limits()

        proof = evaluate_tiered_constraints(spec, limits)

        assert proof.passed
        assert proof.first_failure_tier is None
        assert len(proof.get_failures()) == 0

    def test_fail_fast_stops_at_first_tier_failure(self) -> None:
        """fail_fast mode should stop at first tier with failures."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # T0 violation
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        proof = evaluate_tiered_constraints(spec, limits, fail_fast=True)

        assert not proof.passed
        assert proof.first_failure_tier == "T0"
        # Should not have evaluated higher tiers (or they should be empty)
        # Actually fail_fast stops iteration, so T1+ may be empty
        assert len(proof.tiers["T1"]) == 0

    def test_without_fail_fast_evaluates_all_tiers(self) -> None:
        """Without fail_fast, all tiers should be evaluated even on failure."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # T0 violation
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        proof = evaluate_tiered_constraints(spec, limits, fail_fast=False)

        assert not proof.passed
        assert proof.first_failure_tier == "T0"
        # Should have evaluated all tiers
        assert len(proof.tiers["T1"]) > 0
        assert len(proof.tiers["T2"]) > 0


class TestRejectMode:
    """Test REJECT mode with constraint IDs and reasons (REQ-M1-009)."""

    def test_reject_mode_raises_with_constraint_ids(self) -> None:
        """REJECT mode should raise ConstraintViolationError with constraint IDs."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # Violation
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        system = TieredConstraintSystem()

        with pytest.raises(ConstraintViolationError) as exc_info:
            system.enforce(spec, limits, mode="REJECT")

        error = exc_info.value
        assert "T0_TRACE_WIDTH_MIN" in error.constraint_ids
        assert error.tier == "T0"
        assert len(error.violations) >= 1

    def test_reject_mode_error_includes_reason(self) -> None:
        """ConstraintViolationError message should include reasons."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        system = TieredConstraintSystem()

        with pytest.raises(ConstraintViolationError) as exc_info:
            system.enforce(spec, limits, mode="REJECT")

        error_message = str(exc_info.value)
        assert "T0_TRACE_WIDTH_MIN" in error_message
        assert "value=" in error_message or "below minimum" in error_message.lower()

    def test_reject_mode_multiple_violations(self) -> None:
        """REJECT mode should report all violations."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # T0 violation
        data["transmission_line"]["gap_nm"] = 50_000  # T0 violation
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        system = TieredConstraintSystem()

        with pytest.raises(ConstraintViolationError) as exc_info:
            system.enforce(spec, limits, mode="REJECT")

        error = exc_info.value
        assert "T0_TRACE_WIDTH_MIN" in error.constraint_ids
        assert "T0_TRACE_GAP_MIN" in error.constraint_ids
        assert len(error.violations) >= 2


class TestConstraintProofSerialization:
    """Test constraint proof serialization."""

    def test_proof_to_dict(self) -> None:
        """Proof should serialize to a valid dictionary."""
        spec = CouponSpec.model_validate(_example_spec_data())
        limits = _default_fab_limits()

        proof = evaluate_tiered_constraints(spec, limits)
        payload = proof.to_dict()

        assert "passed" in payload
        assert payload["passed"] is True
        assert "tiers" in payload
        assert "constraints" in payload
        assert "first_failure_tier" in payload

        # Check constraint entry structure
        for entry in payload["constraints"]:
            assert "id" in entry
            assert "description" in entry
            assert "tier" in entry
            assert "value" in entry
            assert "limit" in entry
            assert "margin" in entry
            assert "passed" in entry
            assert "reason" in entry

    def test_failed_proof_to_dict(self) -> None:
        """Failed proof should serialize with failure information."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        proof = evaluate_tiered_constraints(spec, limits)
        payload = proof.to_dict()

        assert payload["passed"] is False
        assert payload["first_failure_tier"] == "T0"

        # Find the failed constraint
        failed_entries = [e for e in payload["constraints"] if not e["passed"]]
        assert len(failed_entries) >= 1


class TestConstraintResultDetails:
    """Test ConstraintResult details and properties."""

    def test_constraint_result_has_all_fields(self) -> None:
        """ConstraintResult should have all required fields."""
        result = ConstraintResult(
            constraint_id="TEST_CONSTRAINT",
            description="A test constraint",
            tier="T0",
            value=50.0,
            limit=100.0,
            margin=-50.0,
            passed=False,
            reason="Value is too small",
        )

        assert result.constraint_id == "TEST_CONSTRAINT"
        assert result.description == "A test constraint"
        assert result.tier == "T0"
        assert result.value == 50.0
        assert result.limit == 100.0
        assert result.margin == -50.0
        assert result.passed is False
        assert result.reason == "Value is too small"

    def test_constraint_result_immutable(self) -> None:
        """ConstraintResult should be immutable (frozen dataclass)."""
        result = ConstraintResult(
            constraint_id="TEST",
            description="Test",
            tier="T0",
            value=50.0,
            limit=100.0,
            margin=-50.0,
            passed=False,
        )

        with pytest.raises(AttributeError):
            result.passed = True  # type: ignore[misc]
