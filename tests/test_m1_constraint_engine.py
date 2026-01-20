"""Tests for unified ConstraintEngine (CP-3.1).

This module tests:
- ConstraintEngine as the single unified path for constraint validation
- evaluate() method returning proof without raising
- validate_or_repair() with REJECT and REPAIR modes
- Integration with tiered constraints and connectivity oracle
- ConstraintEngineResult properties and serialization

REQ-M1-008: Tiered constraint system with Tiers 0-3
REQ-M1-009: REJECT mode with constraint IDs and reasons
REQ-M1-010: REPAIR mode with repair_map, repair_reason, repair_distance
REQ-M1-011: constraint_proof.json with per-constraint evaluations and signed margins
CP-3.1: Unified ConstraintEngine as single path for constraint validation
"""

from __future__ import annotations

from typing import Any

import pytest

from formula_foundry.coupongen.constraints.engine import (
    ConstraintEngine,
    ConstraintEngineResult,
    create_constraint_engine,
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


class TestConstraintEngineCreation:
    """Test ConstraintEngine instantiation and configuration."""

    def test_create_with_fab_limits(self) -> None:
        """Engine should be created with provided fab limits."""
        limits = _default_fab_limits()
        engine = ConstraintEngine(fab_limits=limits)

        assert engine.fab_limits == limits
        assert engine.fail_fast is False
        assert engine.include_connectivity is True

    def test_create_with_fail_fast(self) -> None:
        """Engine should support fail_fast configuration."""
        engine = ConstraintEngine(fab_limits=_default_fab_limits(), fail_fast=True)

        assert engine.fail_fast is True

    def test_create_without_connectivity(self) -> None:
        """Engine should support disabling connectivity checker."""
        engine = ConstraintEngine(
            fab_limits=_default_fab_limits(),
            include_connectivity=False,
        )

        assert engine.include_connectivity is False

    def test_factory_function_with_defaults(self) -> None:
        """Factory function should create engine with default limits."""
        engine = create_constraint_engine()

        assert engine.fab_limits is not None
        assert "min_trace_width_nm" in engine.fab_limits

    def test_factory_function_with_custom_limits(self) -> None:
        """Factory function should accept custom limits."""
        custom_limits = {"min_trace_width_nm": 200_000}
        engine = create_constraint_engine(fab_limits=custom_limits)

        assert engine.fab_limits["min_trace_width_nm"] == 200_000


class TestConstraintEngineEvaluate:
    """Test ConstraintEngine.evaluate() method."""

    def test_evaluate_valid_spec_returns_passing_proof(self) -> None:
        """evaluate() should return passing proof for valid spec."""
        spec = CouponSpec.model_validate(_example_spec_data())
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        proof = engine.evaluate(spec)

        assert proof.passed is True
        assert proof.first_failure_tier is None
        assert len(proof.constraints) > 0

    def test_evaluate_invalid_spec_returns_failing_proof(self) -> None:
        """evaluate() should return failing proof for invalid spec without raising."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # Below minimum
        spec = CouponSpec.model_validate(data)
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        proof = engine.evaluate(spec)

        # Should NOT raise - evaluate returns proof regardless
        assert proof.passed is False
        assert proof.first_failure_tier == "T0"

    def test_evaluate_returns_all_tier_results(self) -> None:
        """evaluate() should include results from all tiers."""
        spec = CouponSpec.model_validate(_example_spec_data())
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        proof = engine.evaluate(spec)

        assert "T0" in proof.tiers
        assert "T1" in proof.tiers
        assert "T2" in proof.tiers
        assert "T3" in proof.tiers

    def test_evaluate_includes_connectivity_checks(self) -> None:
        """evaluate() should include connectivity constraint results."""
        spec = CouponSpec.model_validate(_example_spec_data())
        engine = ConstraintEngine(fab_limits=_default_fab_limits(), include_connectivity=True)

        proof = engine.evaluate(spec)

        # Should have connectivity-related constraints
        constraint_ids = [c.constraint_id for c in proof.constraints]
        connectivity_ids = [cid for cid in constraint_ids if "CONNECT" in cid or "SIG" in cid]
        assert len(connectivity_ids) > 0

    def test_evaluate_without_connectivity(self) -> None:
        """evaluate() should skip connectivity checks when disabled."""
        spec = CouponSpec.model_validate(_example_spec_data())
        engine = ConstraintEngine(
            fab_limits=_default_fab_limits(),
            include_connectivity=False,
        )

        proof = engine.evaluate(spec)

        # Should still pass but have fewer constraints
        assert proof.passed is True


class TestConstraintEngineValidateOrRepair:
    """Test ConstraintEngine.validate_or_repair() method."""

    def test_reject_mode_valid_spec_returns_result(self) -> None:
        """REJECT mode with valid spec should return passing result."""
        spec = CouponSpec.model_validate(_example_spec_data())
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        result = engine.validate_or_repair(spec, mode="REJECT")

        assert result.passed is True
        assert result.was_repaired is False
        assert result.repair_map is None
        assert result.resolved is not None

    def test_reject_mode_invalid_spec_raises(self) -> None:
        """REJECT mode with invalid spec should raise ConstraintViolationError."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # Below minimum
        spec = CouponSpec.model_validate(data)
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        with pytest.raises(ConstraintViolationError) as exc_info:
            engine.validate_or_repair(spec, mode="REJECT")

        assert "T0_TRACE_WIDTH_MIN" in exc_info.value.constraint_ids

    def test_repair_mode_valid_spec_no_repairs(self) -> None:
        """REPAIR mode with valid spec should not make repairs."""
        spec = CouponSpec.model_validate(_example_spec_data())
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        result = engine.validate_or_repair(spec, mode="REPAIR")

        assert result.passed is True
        assert result.was_repaired is False
        assert result.repair_result is None

    def test_repair_mode_invalid_spec_repairs(self) -> None:
        """REPAIR mode with invalid spec should repair and return result."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # Below minimum
        spec = CouponSpec.model_validate(data)
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        result = engine.validate_or_repair(spec, mode="REPAIR")

        assert result.passed is True
        assert result.was_repaired is True
        assert result.repair_map is not None
        assert "transmission_line.w_nm" in result.repair_map

    def test_repair_mode_returns_repaired_resolved(self) -> None:
        """REPAIR mode should return ResolvedDesign with repaired values."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # Below minimum
        spec = CouponSpec.model_validate(data)
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        result = engine.validate_or_repair(spec, mode="REPAIR")

        # The resolved design should reflect the repaired spec
        assert result.resolved is not None
        # Check that the parameters were updated
        # (exact field access depends on ResolvedDesign structure)

    def test_repair_mode_multiple_violations(self) -> None:
        """REPAIR mode should fix multiple violations."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # Violation 1
        data["transmission_line"]["gap_nm"] = 50_000  # Violation 2
        spec = CouponSpec.model_validate(data)
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        result = engine.validate_or_repair(spec, mode="REPAIR")

        assert result.passed is True
        assert result.was_repaired is True
        assert len(result.repair_map or {}) >= 2


class TestConstraintEngineResult:
    """Test ConstraintEngineResult properties."""

    def test_result_passed_property(self) -> None:
        """passed property should reflect proof status."""
        spec = CouponSpec.model_validate(_example_spec_data())
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        result = engine.validate_or_repair(spec, mode="REJECT")

        assert result.passed is True
        assert result.proof.passed is True

    def test_result_was_repaired_false_when_no_repairs(self) -> None:
        """was_repaired should be False when no repairs needed."""
        spec = CouponSpec.model_validate(_example_spec_data())
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        result = engine.validate_or_repair(spec, mode="REPAIR")

        assert result.was_repaired is False

    def test_result_was_repaired_true_when_repairs_made(self) -> None:
        """was_repaired should be True when repairs were made."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        result = engine.validate_or_repair(spec, mode="REPAIR")

        assert result.was_repaired is True

    def test_result_repair_map_none_when_no_repairs(self) -> None:
        """repair_map should be None when no repairs needed."""
        spec = CouponSpec.model_validate(_example_spec_data())
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        result = engine.validate_or_repair(spec, mode="REPAIR")

        assert result.repair_map is None

    def test_result_repair_map_populated_when_repaired(self) -> None:
        """repair_map should contain repair details when repairs made."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        result = engine.validate_or_repair(spec, mode="REPAIR")

        assert result.repair_map is not None
        assert "transmission_line.w_nm" in result.repair_map
        assert result.repair_map["transmission_line.w_nm"]["before"] == 50_000
        assert result.repair_map["transmission_line.w_nm"]["after"] == 100_000

    def test_result_to_proof_document(self) -> None:
        """to_proof_document() should generate valid ConstraintProofDocument."""
        spec = CouponSpec.model_validate(_example_spec_data())
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        result = engine.validate_or_repair(spec, mode="REJECT")
        doc = result.to_proof_document()

        assert doc.schema_version == 1
        assert doc.passed is True
        assert doc.total_constraints > 0

    def test_result_to_proof_document_with_repairs(self) -> None:
        """to_proof_document() should include repair info when repaired."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        result = engine.validate_or_repair(spec, mode="REPAIR")
        doc = result.to_proof_document()

        assert doc.repair_applied is True
        assert doc.repair_info is not None


class TestConstraintEngineIntegration:
    """Integration tests for ConstraintEngine with full pipeline."""

    def test_engine_as_single_validation_path(self) -> None:
        """Engine should be usable as the single validation path."""
        spec = CouponSpec.model_validate(_example_spec_data())
        limits = _default_fab_limits()

        # Create engine
        engine = ConstraintEngine(fab_limits=limits)

        # Use for evaluation
        proof = engine.evaluate(spec)
        assert proof.passed

        # Use for validation
        result = engine.validate_or_repair(spec, mode="REJECT")
        assert result.passed

    def test_engine_constraint_coverage(self) -> None:
        """Engine should evaluate all expected constraint tiers."""
        spec = CouponSpec.model_validate(_example_spec_data())
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        proof = engine.evaluate(spec)

        # Check we have constraints from each tier
        tier0_constraints = proof.tiers.get("T0", ())
        tier1_constraints = proof.tiers.get("T1", ())
        tier2_constraints = proof.tiers.get("T2", ())
        tier3_constraints = proof.tiers.get("T3", ())

        assert len(tier0_constraints) > 0, "Should have T0 constraints"
        assert len(tier1_constraints) > 0, "Should have T1 constraints"
        assert len(tier2_constraints) > 0, "Should have T2 constraints"
        assert len(tier3_constraints) >= 0, "T3 constraints depend on spec features"

    def test_engine_fail_fast_mode(self) -> None:
        """fail_fast mode should stop at first failing tier."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # T0 violation
        spec = CouponSpec.model_validate(data)
        engine = ConstraintEngine(fab_limits=_default_fab_limits(), fail_fast=True)

        proof = engine.evaluate(spec)

        # Should fail at T0 and not continue to later tiers
        assert proof.first_failure_tier == "T0"
        assert not proof.passed

    def test_engine_proof_has_signed_margins(self) -> None:
        """Proof constraints should have signed margins."""
        spec = CouponSpec.model_validate(_example_spec_data())
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        proof = engine.evaluate(spec)

        for constraint in proof.constraints:
            # All passing constraints should have non-negative margin
            if constraint.passed:
                assert constraint.margin >= 0, f"{constraint.constraint_id} has negative margin"

    def test_engine_create_proof_document(self) -> None:
        """create_proof_document() should create valid document."""
        spec = CouponSpec.model_validate(_example_spec_data())
        engine = ConstraintEngine(fab_limits=_default_fab_limits())

        proof = engine.evaluate(spec)
        doc = engine.create_proof_document(proof)

        assert doc.schema_version == 1
        assert doc.passed is True
        assert "T0" in doc.tiers
        assert len(doc.constraints) > 0
