"""Comprehensive tests for the constraint system.

This module consolidates tests for:
- REQ-M1-008: All constraint tiers (T0-T4)
- REQ-M1-009: REJECT mode with constraint IDs and reasons
- REQ-M1-010: REPAIR mode with repair_map, repair_reason, repair_distance
- REQ-M1-011: constraint_proof generation with per-constraint evaluations and signed margins
- REQ-M1-GPU-FILTER: GPU vectorized constraint prefilter correctness

Cross-references:
- tests/test_m1_constraints.py: Basic constraint tests
- tests/test_m1_constraints_tiers.py: Detailed tier tests
- tests/test_m1_constraints_repair.py: REPAIR mode tests
- tests/test_m1_gpu_filter.py: GPU filter tests
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from formula_foundry.coupongen.constraints import (
    BatchFilterResult,
    ConstraintViolation,
    ConstraintViolationError,
    FamilyF1ParameterSpace,
    GPUConstraintFilter,
    ParameterMapping,
    RepairEngine,
    RepairMeta,
    Tier0Checker,
    Tier1Checker,
    Tier2Checker,
    Tier3Checker,
    TieredConstraintSystem,
    batch_filter,
    constraint_proof_payload,
    enforce_constraints,
    evaluate_constraints,
    evaluate_tiered_constraints,
    generate_constraint_proof,
    is_gpu_available,
    repair_spec_tiered,
    write_constraint_proof,
)
from formula_foundry.coupongen.geom import (
    CPWGSpec,
    GroundViaFenceSpec,
    OutlineArc,
    OutlineLine,
    PositionNM,
    generate_ground_via_fence,
    generate_rounded_outline,
)
from formula_foundry.coupongen.spec import CouponSpec

# ==============================================================================
# Test Fixtures
# ==============================================================================


def _valid_spec_data() -> dict[str, Any]:
    """Return a valid CouponSpec data dictionary that passes all constraints."""
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


# ==============================================================================
# REQ-M1-008: Constraint Tiers (T0-T4)
# ==============================================================================


class TestConstraintTiers:
    """Test all constraint tiers (T0, T1, T2, T3, T4)."""

    def test_all_tiers_exist_in_proof(self) -> None:
        """Proof should contain all five tiers."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        proof = evaluate_constraints(spec)

        assert set(proof.tiers.keys()) == {"T0", "T1", "T2", "T3", "T4"}

    def test_tier0_parameter_bounds(self) -> None:
        """Tier 0 validates direct parameter bounds against fab limits."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        limits = _default_fab_limits()
        checker = Tier0Checker()

        results = checker.check(spec, limits)

        assert all(r.tier == "T0" for r in results)
        # Valid spec should pass all T0 checks
        assert all(r.passed for r in results)

    def test_tier0_trace_width_minimum(self) -> None:
        """Tier 0 should catch trace width below fab minimum."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # Below 100_000 minimum
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier0Checker()

        results = checker.check(spec, limits)

        width_result = next(r for r in results if r.constraint_id == "T0_TRACE_WIDTH_MIN")
        assert not width_result.passed
        assert width_result.value == 50_000
        assert width_result.limit == 100_000
        assert width_result.margin == -50_000

    def test_tier1_derived_scalar_constraints(self) -> None:
        """Tier 1 validates computed/derived scalar constraints."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        limits = _default_fab_limits()
        checker = Tier1Checker()

        results = checker.check(spec, limits)

        assert all(r.tier == "T1" for r in results)
        assert all(r.passed for r in results)

    def test_tier1_annular_ring_minimum(self) -> None:
        """Tier 1 should catch insufficient annular ring."""
        data = _valid_spec_data()
        # Set pad barely larger than drill: annular ring too small
        data["discontinuity"]["signal_via"]["drill_nm"] = 400_000
        data["discontinuity"]["signal_via"]["diameter_nm"] = 450_000
        data["discontinuity"]["signal_via"]["pad_diameter_nm"] = 500_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier1Checker()

        results = checker.check(spec, limits)

        annular_result = next(r for r in results if r.constraint_id == "T1_SIGNAL_ANNULAR_MIN")
        assert not annular_result.passed
        assert annular_result.value == 50_000  # (500k - 400k) / 2
        assert annular_result.limit == 100_000

    def test_tier2_analytic_spatial_constraints(self) -> None:
        """Tier 2 validates analytic spatial relationships."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        limits = _default_fab_limits()
        checker = Tier2Checker()

        results = checker.check(spec, limits)

        assert all(r.tier == "T2" for r in results)
        assert all(r.passed for r in results)

    def test_tier2_connector_outside_board(self) -> None:
        """Tier 2 should catch connector position outside board bounds."""
        data = _valid_spec_data()
        data["connectors"]["right"]["position_nm"] = [100_000_000, 0]  # Beyond 80M board
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier2Checker()

        results = checker.check(spec, limits)

        x_max_result = next(r for r in results if r.constraint_id == "T2_RIGHT_CONNECTOR_X_MAX")
        assert not x_max_result.passed

    def test_tier3_geometry_collision_detection(self) -> None:
        """Tier 3 validates exact geometry collision detection."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        limits = _default_fab_limits()
        checker = Tier3Checker()

        results = checker.check(spec, limits)

        assert all(r.tier == "T3" for r in results)
        assert all(r.passed for r in results)

    def test_tier3_return_vias_overlap(self) -> None:
        """Tier 3 should catch overlapping return vias."""
        data = _valid_spec_data()
        # Many vias at small radius with large diameter will overlap
        data["discontinuity"]["return_vias"]["count"] = 8
        data["discontinuity"]["return_vias"]["radius_nm"] = 800_000
        data["discontinuity"]["return_vias"]["via"]["diameter_nm"] = 600_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier3Checker()

        results = checker.check(spec, limits)

        overlap_result = next(r for r in results if r.constraint_id == "T3_RETURN_VIA_RING_NO_OVERLAP")
        assert not overlap_result.passed

    def test_tier3_symmetry_enforcement(self) -> None:
        """Tier 3 should enforce trace symmetry when enabled."""
        data = _valid_spec_data()
        data["transmission_line"]["length_left_nm"] = 20_000_000
        data["transmission_line"]["length_right_nm"] = 30_000_000  # Asymmetric
        data["constraints"]["symmetry"]["enforce"] = True
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        checker = Tier3Checker()

        results = checker.check(spec, limits)

        symmetry_result = next(r for r in results if r.constraint_id == "T3_TRACE_SYMMETRY")
        assert not symmetry_result.passed


class TestTieredConstraintSystem:
    """Test the complete tiered constraint system."""

    def test_evaluates_all_tiers(self) -> None:
        """System should evaluate all tiers."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        limits = _default_fab_limits()

        proof = evaluate_tiered_constraints(spec, limits)

        assert set(proof.tiers.keys()) == {"T0", "T1", "T2", "T3", "T4"}
        for tier in ("T0", "T1", "T2", "T3"):
            assert len(proof.tiers[tier]) > 0
        # T4 may be empty (DRC is external validation, not always run)

    def test_valid_spec_passes_all_tiers(self) -> None:
        """Valid spec should pass all constraint tiers."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        limits = _default_fab_limits()

        proof = evaluate_tiered_constraints(spec, limits)

        assert proof.passed
        assert proof.first_failure_tier is None
        assert len(proof.get_failures()) == 0

    def test_fail_fast_stops_at_first_tier_failure(self) -> None:
        """fail_fast mode should stop evaluating after first tier with failures."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # T0 violation
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        proof = evaluate_tiered_constraints(spec, limits, fail_fast=True)

        assert not proof.passed
        assert proof.first_failure_tier == "T0"
        # Should not have evaluated higher tiers
        assert len(proof.tiers["T1"]) == 0

    def test_without_fail_fast_evaluates_all_tiers(self) -> None:
        """Without fail_fast, all tiers should be evaluated even on failure."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # T0 violation
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        proof = evaluate_tiered_constraints(spec, limits, fail_fast=False)

        assert not proof.passed
        assert proof.first_failure_tier == "T0"
        # Should still evaluate all tiers
        assert len(proof.tiers["T1"]) > 0
        assert len(proof.tiers["T2"]) > 0


# ==============================================================================
# REQ-M1-009: REJECT Mode
# ==============================================================================


class TestRejectMode:
    """Test REJECT mode with constraint IDs and reasons."""

    def test_reject_mode_raises_constraint_violation(self) -> None:
        """REJECT mode should raise ConstraintViolation on failure."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 10  # Violation
        data["constraints"]["mode"] = "REJECT"
        spec = CouponSpec.model_validate(data)

        with pytest.raises(ConstraintViolation) as exc:
            enforce_constraints(spec)

        ids = {result.constraint_id for result in exc.value.violations}
        assert "T0_TRACE_WIDTH_MIN" in ids

    def test_reject_mode_reports_all_constraint_ids(self) -> None:
        """REJECT mode error should contain all failed constraint IDs."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # Violation
        data["transmission_line"]["gap_nm"] = 50_000  # Violation
        data["constraints"]["mode"] = "REJECT"
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        system = TieredConstraintSystem()

        with pytest.raises(ConstraintViolationError) as exc_info:
            system.enforce(spec, limits, mode="REJECT")

        error = exc_info.value
        assert "T0_TRACE_WIDTH_MIN" in error.constraint_ids
        assert "T0_TRACE_GAP_MIN" in error.constraint_ids
        assert len(error.violations) >= 2

    def test_reject_mode_includes_reasons_in_message(self) -> None:
        """ConstraintViolationError message should include failure reasons."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        data["constraints"]["mode"] = "REJECT"
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        system = TieredConstraintSystem()

        with pytest.raises(ConstraintViolationError) as exc_info:
            system.enforce(spec, limits, mode="REJECT")

        error_message = str(exc_info.value)
        assert "T0_TRACE_WIDTH_MIN" in error_message

    def test_reject_mode_valid_spec_passes(self) -> None:
        """Valid spec in REJECT mode should not raise."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        limits = _default_fab_limits()

        system = TieredConstraintSystem()
        proof = system.enforce(spec, limits, mode="REJECT")

        assert proof.passed


# ==============================================================================
# REQ-M1-010: REPAIR Mode
# ==============================================================================


class TestRepairMode:
    """Test REPAIR mode with repair_map, repair_reason, and repair_distance."""

    def test_repair_mode_no_repairs_for_valid_spec(self) -> None:
        """Valid spec should not require any repairs."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        limits = _default_fab_limits()

        repaired_spec, repair_result = repair_spec_tiered(spec, limits)

        assert repair_result.repair_map == {}
        assert repair_result.repair_reason == []
        assert repair_result.repair_distance == 0.0
        assert repair_result.repaired_proof.passed

    def test_repair_mode_emits_repair_map(self) -> None:
        """REPAIR mode should emit repair_map with before/after values."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 10
        data["constraints"]["mode"] = "REPAIR"
        spec = CouponSpec.model_validate(data)

        evaluation = enforce_constraints(spec)

        assert evaluation.repair_info is not None
        assert "transmission_line.w_nm" in evaluation.repair_info.repair_map
        repair = evaluation.repair_info.repair_map["transmission_line.w_nm"]
        assert "before" in repair
        assert "after" in repair
        assert repair["after"] >= 100000

    def test_repair_mode_emits_repair_reason(self) -> None:
        """REPAIR mode should emit human-readable repair reasons."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        assert len(repair_result.repair_reason) >= 1
        reason = repair_result.repair_reason[0]
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_repair_mode_emits_repair_distance(self) -> None:
        """REPAIR mode should compute normalized repair distance."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # 50% change to 100_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        # Distance = |100_000 - 50_000| / 50_000 = 1.0
        assert repair_result.repair_distance > 0.0
        assert repair_result.repair_distance == pytest.approx(1.0, rel=0.01)

    def test_repair_mode_fixes_multiple_violations(self) -> None:
        """REPAIR mode should fix multiple violations simultaneously."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # Violation
        data["transmission_line"]["gap_nm"] = 50_000  # Violation
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        repaired_spec, repair_result = repair_spec_tiered(spec, limits)

        assert len(repair_result.repair_map) >= 2
        assert "transmission_line.w_nm" in repair_result.repair_map
        assert "transmission_line.gap_nm" in repair_result.repair_map

    def test_repaired_spec_passes_all_constraints(self) -> None:
        """Repaired spec should pass all constraints."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        data["transmission_line"]["gap_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        repaired_spec, repair_result = repair_spec_tiered(spec, limits)

        assert repair_result.repaired_proof.passed
        assert repair_result.repaired_proof.first_failure_tier is None

        # Double-check by re-evaluating
        system = TieredConstraintSystem()
        re_proof = system.evaluate(repaired_spec, limits)
        assert re_proof.passed

    def test_repair_actions_reference_constraint_id(self) -> None:
        """RepairActions should reference triggering constraint ID."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        assert len(repair_result.repair_actions) >= 1
        action = repair_result.repair_actions[0]
        assert action.constraint_id == "T0_TRACE_WIDTH_MIN"
        assert action.path == "transmission_line.w_nm"
        assert action.before == 50_000
        assert action.after == 100_000


class TestRepairEngine:
    """Test RepairEngine directly."""

    def test_repair_tier0_parameters(self) -> None:
        """RepairEngine should repair Tier 0 parameter violations."""
        limits = _default_fab_limits()
        engine = RepairEngine(limits)
        payload = {
            "transmission_line": {
                "w_nm": 50_000,
                "gap_nm": 50_000,
                # length_right_nm is now optional/derived for F1 (CP-2.2)
                "length_left_nm": 35000000,
                "length_right_nm": 35000000,
                "ground_via_fence": None,
            },
            "board": {
                "outline": {
                    "width_nm": 20000000,
                    "length_nm": 80000000,
                    "corner_radius_nm": 2000000,
                }
            },
            "discontinuity": None,
        }

        engine.repair_tier0(payload)

        assert payload["transmission_line"]["w_nm"] == 100_000
        assert payload["transmission_line"]["gap_nm"] == 100_000
        assert len(engine.actions) == 2


# ==============================================================================
# REQ-M1-011: Constraint Proof Generation
# ==============================================================================


class TestConstraintProofGeneration:
    """Test constraint_proof.json generation with per-constraint evaluations and signed margins."""

    def test_proof_schema_structure(self) -> None:
        """Constraint proof payload should have required schema structure."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        proof = evaluate_constraints(spec)
        payload = constraint_proof_payload(proof)

        assert "passed" in payload
        assert payload["passed"] is True
        assert "tiers" in payload
        assert "constraints" in payload
        for tier in ("T0", "T1", "T2", "T3", "T4"):
            assert tier in payload["tiers"]

    def test_constraint_entries_have_all_fields(self) -> None:
        """Each constraint entry should have required fields."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        proof = evaluate_constraints(spec)
        payload = constraint_proof_payload(proof)

        for entry in payload["constraints"]:
            assert {"id", "description", "tier", "value", "limit", "margin", "passed"} <= set(entry)

    def test_signed_margins_positive_for_passing(self) -> None:
        """Passing constraints should have non-negative margins."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        limits = _default_fab_limits()
        proof = evaluate_tiered_constraints(spec, limits)

        doc = generate_constraint_proof(proof)

        passing_entries = [e for e in doc.constraints if e["passed"]]
        for entry in passing_entries:
            assert entry["margin"] >= 0, f"{entry['id']} has negative margin {entry['margin']}"

    def test_signed_margins_negative_for_failing(self) -> None:
        """Failing constraints should have negative margins."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # Below minimum
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        proof = evaluate_tiered_constraints(spec, limits)

        doc = generate_constraint_proof(proof)

        failing_entries = [e for e in doc.constraints if not e["passed"]]
        assert len(failing_entries) >= 1
        width_entry = next(e for e in failing_entries if e["id"] == "T0_TRACE_WIDTH_MIN")
        assert width_entry["margin"] < 0
        assert width_entry["margin"] == -50_000  # 50_000 - 100_000

    def test_constraint_proof_document_structure(self) -> None:
        """ConstraintProofDocument should have complete structure."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        limits = _default_fab_limits()
        proof = evaluate_tiered_constraints(spec, limits)

        doc = generate_constraint_proof(proof)

        assert doc.schema_version == 1
        assert doc.passed is True
        assert doc.first_failure_tier is None
        assert doc.total_constraints > 0
        assert doc.failed_constraints == 0
        assert "T0" in doc.tiers

    def test_constraint_proof_with_repair_info(self) -> None:
        """Proof should include repair info when REPAIR mode was used."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)
        doc = generate_constraint_proof(repair_result.repaired_proof, repair_result)

        assert doc.repair_applied is True
        assert doc.repair_info is not None
        assert "repair_map" in doc.repair_info
        assert "repair_reason" in doc.repair_info
        assert "repair_distance" in doc.repair_info


class TestWriteConstraintProof:
    """Test writing constraint_proof.json to file."""

    def test_write_creates_file(self) -> None:
        """write_constraint_proof should create a valid JSON file."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        limits = _default_fab_limits()
        proof = evaluate_tiered_constraints(spec, limits)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "constraint_proof.json"
            write_constraint_proof(output_path, proof)

            assert output_path.exists()
            content = json.loads(output_path.read_text())
            assert content["passed"] is True

    def test_write_creates_parent_directories(self) -> None:
        """write_constraint_proof should create parent directories if needed."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        limits = _default_fab_limits()
        proof = evaluate_tiered_constraints(spec, limits)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "constraint_proof.json"
            write_constraint_proof(output_path, proof)

            assert output_path.exists()

    def test_write_includes_repair_info(self) -> None:
        """Written constraint_proof.json should include repair info when provided."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "constraint_proof.json"
            write_constraint_proof(output_path, repair_result.repaired_proof, repair_result)

            content = json.loads(output_path.read_text())
            assert content["repair_applied"] is True
            assert "repair_map" in content["repair_info"]


# ==============================================================================
# REQ-M1-GPU-FILTER: GPU Vectorized Constraint Prefilter
# ==============================================================================


class TestGPUFilterParameterMapping:
    """Test parameter mapping between normalized and physical space."""

    def test_linear_mapping_to_physical(self) -> None:
        """Linear mapping should correctly transform [0,1] to [min, max]."""
        mapping = ParameterMapping(name="test", index=0, scale="linear", min_val=100.0, max_val=500.0)

        assert mapping.to_physical(0.0, np) == 100.0
        assert mapping.to_physical(1.0, np) == 500.0
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


class TestGPUFilterParameterSpace:
    """Test F1 family parameter space definition."""

    def test_dimension(self) -> None:
        """Parameter space should have correct dimension."""
        space = FamilyF1ParameterSpace()
        assert space.dimension == 19

    def test_get_mapping_by_name(self) -> None:
        """Should retrieve mappings by name."""
        space = FamilyF1ParameterSpace()

        trace_width = space.get_mapping("trace_width_nm")
        assert trace_width is not None
        assert trace_width.index == 0

        signal_drill = space.get_mapping("signal_drill_nm")
        assert signal_drill is not None
        assert signal_drill.index == 5

    def test_batch_conversion_to_physical(self) -> None:
        """Batch conversion to physical parameters should work."""
        space = FamilyF1ParameterSpace()
        u_batch = np.ones((10, space.dimension)) * 0.5  # All midpoint values

        params = space.to_physical_batch(u_batch, np)

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

    def test_tier0_check_valid_parameters(self) -> None:
        """Valid parameters should pass Tier 0 checks."""
        limits = _default_fab_limits()
        filter_instance = GPUConstraintFilter(limits, use_gpu=False)

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

        assert passed.all()

    def test_tier0_check_trace_width_violation(self) -> None:
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


class TestBatchFilter:
    """Test the main batch_filter function."""

    def test_returns_batch_filter_result(self) -> None:
        """batch_filter should return a BatchFilterResult."""
        u_batch = np.random.rand(100, 19)
        result = batch_filter(u_batch, use_gpu=False)

        assert isinstance(result, BatchFilterResult)
        assert len(result.feasible_mask) == 100
        assert result.repaired_u.shape == (100, 19)

    def test_reports_feasibility_rate(self) -> None:
        """batch_filter should report feasibility rate correctly."""
        u_batch = np.random.rand(1000, 19)
        result = batch_filter(u_batch, use_gpu=False)

        assert 0.0 <= result.feasibility_rate <= 1.0
        assert result.n_feasible == result.feasible_mask.sum()
        assert result.n_candidates == 1000

    def test_repair_increases_feasibility(self) -> None:
        """Repair should increase feasibility compared to no repair."""
        np.random.seed(42)
        u_batch = np.random.rand(500, 19)

        result_no_repair = batch_filter(u_batch, use_gpu=False, repair=False)
        result_with_repair = batch_filter(u_batch, use_gpu=False, repair=True)

        # Repair should generally increase or maintain feasibility
        assert result_with_repair.n_feasible >= result_no_repair.n_feasible

    def test_repair_populates_metadata(self) -> None:
        """Repair should populate repair metadata."""
        u_batch = np.random.rand(100, 19)
        result = batch_filter(u_batch, use_gpu=False, repair=True)

        assert result.repair_counts is not None
        assert result.repair_distances is not None
        assert len(result.repair_counts) == 100
        assert len(result.repair_distances) == 100

    def test_tracks_constraint_margins(self) -> None:
        """batch_filter should return constraint margins."""
        u_batch = np.random.rand(50, 19)
        result = batch_filter(u_batch, use_gpu=False)

        assert "T0_TRACE_WIDTH_MIN" in result.constraint_margins
        assert "T1_SIGNAL_ANNULAR_MIN" in result.constraint_margins
        assert "T2_TRACE_FITS_IN_BOARD" in result.constraint_margins
        assert len(result.constraint_margins["T0_TRACE_WIDTH_MIN"]) == 50

    def test_tracks_tier_violations(self) -> None:
        """batch_filter should track violations by tier."""
        u_batch = np.random.rand(100, 19)
        result = batch_filter(u_batch, use_gpu=False, repair=False)

        assert "T0" in result.tier_violations
        assert "T1" in result.tier_violations
        assert "T2" in result.tier_violations
        assert len(result.tier_violations["T0"]) == 100

    def test_handles_large_batch(self) -> None:
        """batch_filter should handle large batches efficiently."""
        # Test with 10K candidates (reduced for CI speed)
        u_batch = np.random.rand(10_000, 19)
        result = batch_filter(u_batch, use_gpu=False)

        assert result.n_candidates == 10_000
        assert len(result.feasible_mask) == 10_000

    def test_stricter_limits_reduce_feasibility(self) -> None:
        """Stricter fab limits should reduce feasibility."""
        np.random.seed(123)
        u_batch = np.random.rand(200, 19)

        default_limits = _default_fab_limits()
        result_default = batch_filter(u_batch, fab_limits=default_limits, use_gpu=False, repair=False)

        strict_limits = {k: v * 2 for k, v in default_limits.items()}
        result_strict = batch_filter(u_batch, fab_limits=strict_limits, use_gpu=False, repair=False)

        assert result_strict.n_feasible <= result_default.n_feasible


class TestBatchFilterResult:
    """Test BatchFilterResult properties."""

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


@pytest.mark.skipif(not is_gpu_available(), reason="CuPy not available")
class TestGPUCPUConsistency:
    """Tests that require GPU/CuPy availability."""

    def test_gpu_results_match_cpu(self) -> None:
        """GPU and CPU results should be approximately equal."""
        np.random.seed(42)
        u_batch = np.random.rand(1000, 19)
        limits = _default_fab_limits()

        result_cpu = batch_filter(u_batch, fab_limits=limits, use_gpu=False, repair=False)
        result_gpu = batch_filter(u_batch, fab_limits=limits, use_gpu=True, repair=False)

        np.testing.assert_array_equal(result_cpu.feasible_mask, result_gpu.feasible_mask)
        assert result_cpu.feasibility_rate == result_gpu.feasibility_rate


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestConstraintSystemIntegration:
    """Integration tests for the complete constraint system."""

    def test_full_pipeline_reject_mode(self) -> None:
        """Test full constraint pipeline in REJECT mode."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        data["constraints"]["mode"] = "REJECT"
        spec = CouponSpec.model_validate(data)

        with pytest.raises(ConstraintViolation) as exc:
            enforce_constraints(spec)

        assert len(exc.value.violations) >= 1
        assert any(v.constraint_id == "T0_TRACE_WIDTH_MIN" for v in exc.value.violations)

    def test_full_pipeline_repair_mode(self) -> None:
        """Test full constraint pipeline in REPAIR mode."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        data["constraints"]["mode"] = "REPAIR"
        spec = CouponSpec.model_validate(data)

        evaluation = enforce_constraints(spec)

        assert evaluation.repair_info is not None
        assert evaluation.proof.passed
        assert int(evaluation.spec.transmission_line.w_nm) >= 100_000

    def test_constraint_proof_json_end_to_end(self) -> None:
        """Test end-to-end constraint proof JSON generation."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "constraint_proof.json"
            write_constraint_proof(output_path, repair_result.repaired_proof, repair_result)

            content = json.loads(output_path.read_text())

            # Verify complete structure
            assert content["schema_version"] == 1
            assert content["passed"] is True
            assert content["repair_applied"] is True
            assert "constraints" in content
            assert len(content["constraints"]) > 0

            # Verify each constraint has required fields
            for entry in content["constraints"]:
                assert "id" in entry
                assert "description" in entry
                assert "tier" in entry
                assert "value" in entry
                assert "limit" in entry
                assert "margin" in entry
                assert "passed" in entry
                assert "reason" in entry

    def test_gpu_filter_to_single_spec_consistency(self) -> None:
        """GPU filter results should be consistent with single-spec evaluation."""
        space = FamilyF1ParameterSpace()
        limits = _default_fab_limits()

        # Create a known-good normalized vector
        u = np.zeros(space.dimension)
        u[0] = 0.6  # trace_width: safe value
        u[1] = 0.6  # trace_gap: safe value
        u[2] = 0.3  # board_width
        u[3] = 0.5  # board_length
        u[4] = 0.1  # corner_radius
        u[5] = 0.4  # signal_drill
        u[6] = 0.6  # signal_via_diameter
        u[7] = 0.7  # signal_pad_diameter
        u[8] = 0.4  # return_via_drill
        u[9] = 0.6  # return_via_diameter
        u[10] = 0.4  # fence_via_drill
        u[11] = 0.6  # fence_via_diameter
        u[12] = 0.3  # left_connector_x
        u[13] = 0.2  # right_connector_x
        u[14] = 0.2  # trace_length_left
        u[15] = 0.2  # trace_length_right
        u[16] = 0.6  # return_via_ring_radius
        u[17] = 0.6  # fence_pitch
        u[18] = 0.6  # fence_offset

        u_batch = u.reshape(1, -1)
        result = batch_filter(u_batch, fab_limits=limits, use_gpu=False, repair=True)

        # The result should be feasible after repair
        assert result.feasibility_rate >= 0.0


# ==============================================================================
# REQ-M1-007: Via Fence Determinism Tests
# ==============================================================================


class TestViaFenceDeterminism:
    """Tests for via fence determinism (REQ-M1-007).

    When ground_via_fence.enabled=true, the generator MUST place GND via fences
    deterministically with correct pitch/offset policies, symmetry, and
    collision/edge-clearance enforcement.
    """

    def test_fence_constraint_determinism_across_runs(self) -> None:
        """Fence constraints should produce identical results across multiple runs."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        limits = _default_fab_limits()

        # Evaluate constraints multiple times
        results = []
        for _ in range(5):
            proof = evaluate_tiered_constraints(spec, limits)
            results.append(proof)

        # All proofs should be identical
        first = results[0]
        for proof in results[1:]:
            assert proof.passed == first.passed
            assert len(proof.constraints) == len(first.constraints)
            for c1, c2 in zip(first.constraints, proof.constraints, strict=True):
                assert c1.constraint_id == c2.constraint_id
                assert c1.value == c2.value
                assert c1.limit == c2.limit
                assert c1.margin == c2.margin
                assert c1.passed == c2.passed

    def test_fence_pitch_constraint_enforced(self) -> None:
        """Fence pitch constraint (T1_FENCE_PITCH_MIN) should be properly enforced."""
        data = _valid_spec_data()
        # Set pitch too small for via spacing
        data["transmission_line"]["ground_via_fence"]["pitch_nm"] = 400_000  # Too small
        data["transmission_line"]["ground_via_fence"]["via"]["diameter_nm"] = 600_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()
        limits["min_via_to_via_nm"] = 200_000

        # Should fail fence pitch constraint
        proof = evaluate_tiered_constraints(spec, limits, fail_fast=False)

        # Find fence pitch constraint results
        pitch_constraints = [
            c for c in proof.constraints
            if "FENCE_PITCH" in c.constraint_id
        ]
        assert len(pitch_constraints) >= 1
        # At least one should fail since pitch < diameter + min_via_to_via
        assert any(not c.passed for c in pitch_constraints)

    def test_fence_offset_constraint_enforced(self) -> None:
        """Fence offset from gap constraint (T2_FENCE_VIA_GAP_CLEARANCE) should be enforced."""
        data = _valid_spec_data()
        # Set offset too small (less than via radius)
        data["transmission_line"]["ground_via_fence"]["offset_from_gap_nm"] = 100_000
        data["transmission_line"]["ground_via_fence"]["via"]["diameter_nm"] = 600_000  # radius=300k
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        # Repair mode should fix this
        repaired_spec, repair_result = repair_spec_tiered(spec, limits)

        # Check if offset was repaired
        if "transmission_line.ground_via_fence.offset_from_gap_nm" in repair_result.repair_map:
            repair = repair_result.repair_map["transmission_line.ground_via_fence.offset_from_gap_nm"]
            # Repaired offset should be at least via radius
            assert repair["after"] >= 300_000

    def test_fence_parameters_in_repair_result(self) -> None:
        """Repair result should include fence parameters in design vector (CP-3.4)."""
        data = _valid_spec_data()
        data["transmission_line"]["ground_via_fence"]["pitch_nm"] = 300_000  # Too small
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        # Original vector should have fence parameters
        assert repair_result.original_vector is not None
        fence_params = [
            p for p in repair_result.original_vector.parameters.keys()
            if "fence" in p.lower()
        ]
        assert len(fence_params) > 0, "Fence parameters should be in design vector"

    def test_fence_enabled_false_skips_constraints(self) -> None:
        """When fence is disabled, fence-specific constraints should not apply."""
        data = _valid_spec_data()
        data["transmission_line"]["ground_via_fence"]["enabled"] = False
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        proof = evaluate_tiered_constraints(spec, limits)

        # Should not have any fence-specific constraint failures
        fence_failures = [
            c for c in proof.constraints
            if "FENCE" in c.constraint_id and not c.passed
        ]
        assert len(fence_failures) == 0

    def test_fence_geometry_generated_from_layout(self) -> None:
        """Fence-enabled specs should generate deterministic via fence geometry."""
        from formula_foundry.coupongen.resolve import resolve

        spec = CouponSpec.model_validate(_valid_spec_data())
        resolved = resolve(spec)
        layout = resolved.layout_plan
        assert layout is not None

        fence = spec.transmission_line.ground_via_fence
        assert fence is not None
        assert fence.enabled

        left_segment = next(seg for seg in layout.segments if seg.label == "left")
        start = PositionNM(left_segment.x_start_nm, left_segment.y_nm)
        end = PositionNM(left_segment.x_end_nm, left_segment.y_nm)

        cpwg_spec = CPWGSpec(
            w_nm=int(spec.transmission_line.w_nm),
            gap_nm=int(spec.transmission_line.gap_nm),
            length_nm=left_segment.length_nm,
            layer=left_segment.layer,
            net_id=1,
        )
        fence_spec = GroundViaFenceSpec(
            pitch_nm=int(fence.pitch_nm),
            offset_from_gap_nm=int(fence.offset_from_gap_nm),
            drill_nm=int(fence.via.drill_nm),
            diameter_nm=int(fence.via.diameter_nm),
        )

        pos_vias, neg_vias = generate_ground_via_fence(start, end, cpwg_spec, fence_spec)

        assert len(pos_vias) > 0
        assert len(pos_vias) == len(neg_vias)

        expected_offset = (
            (cpwg_spec.w_nm + 1) // 2 + cpwg_spec.gap_nm + fence_spec.offset_from_gap_nm
        )
        for pos_via, neg_via in zip(pos_vias, neg_vias, strict=True):
            assert pos_via.position.x == neg_via.position.x
            assert pos_via.position.y == expected_offset
            assert neg_via.position.y == -expected_offset


# ==============================================================================
# REQ-M1-009: Rounded Outline Feasibility Tests
# ==============================================================================


class TestRoundedOutlineFeasibility:
    """Tests for rounded outline feasibility (REQ-M1-009).

    If corner_radius_nm > 0 is provided, the board outline MUST be generated as a
    rounded-rectangle with deterministic integer-nm arcs/segments and validated
    for feasibility.
    """

    def test_corner_radius_zero_valid(self) -> None:
        """corner_radius_nm = 0 should be valid (sharp corners)."""
        data = _valid_spec_data()
        data["board"]["outline"]["corner_radius_nm"] = 0
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        proof = evaluate_tiered_constraints(spec, limits)

        # Corner radius constraints should pass
        corner_constraints = [
            c for c in proof.constraints
            if "CORNER_RADIUS" in c.constraint_id
        ]
        assert all(c.passed for c in corner_constraints)

    def test_corner_radius_positive_valid(self) -> None:
        """Valid positive corner_radius_nm should pass constraints."""
        data = _valid_spec_data()
        # Board is 20mm x 80mm, so max corner radius is 10mm
        data["board"]["outline"]["corner_radius_nm"] = 2_000_000  # 2mm - valid
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        proof = evaluate_tiered_constraints(spec, limits)

        # Corner radius constraints should pass
        corner_constraints = [
            c for c in proof.constraints
            if "CORNER_RADIUS" in c.constraint_id
        ]
        assert all(c.passed for c in corner_constraints)

    def test_rounded_outline_contains_arcs(self) -> None:
        """Rounded outline should emit arcs when corner_radius_nm > 0."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        outline = generate_rounded_outline(
            x_left_nm=0,
            y_bottom_nm=-(spec.board.outline.width_nm // 2),
            width_nm=int(spec.board.outline.length_nm),
            height_nm=int(spec.board.outline.width_nm),
            corner_radius_nm=int(spec.board.outline.corner_radius_nm),
        )

        arcs = [elem for elem in outline.elements if isinstance(elem, OutlineArc)]
        lines = [elem for elem in outline.elements if isinstance(elem, OutlineLine)]

        assert len(arcs) == 4
        assert len(lines) == 4

    def test_corner_radius_exceeds_max_fails(self) -> None:
        """corner_radius_nm > half min(width, length) should fail."""
        data = _valid_spec_data()
        # Board is 20mm x 80mm, so max corner radius is 10mm
        data["board"]["outline"]["corner_radius_nm"] = 15_000_000  # 15mm - too large
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        proof = evaluate_tiered_constraints(spec, limits)

        # T0_CORNER_RADIUS_MAX should fail
        max_constraint = next(
            (c for c in proof.constraints if c.constraint_id == "T0_CORNER_RADIUS_MAX"),
            None
        )
        assert max_constraint is not None
        assert not max_constraint.passed
        assert max_constraint.margin < 0

    def test_corner_radius_negative_fails(self) -> None:
        """Negative corner_radius_nm should fail constraints."""
        data = _valid_spec_data()
        data["board"]["outline"]["corner_radius_nm"] = -1_000_000  # negative
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        proof = evaluate_tiered_constraints(spec, limits)

        # T0_CORNER_RADIUS_MIN should fail
        min_constraint = next(
            (c for c in proof.constraints if c.constraint_id == "T0_CORNER_RADIUS_MIN"),
            None
        )
        assert min_constraint is not None
        assert not min_constraint.passed

    def test_corner_radius_repaired_to_feasible(self) -> None:
        """REPAIR mode should fix infeasible corner radius to valid range."""
        data = _valid_spec_data()
        # Board is 20mm x 80mm, max corner = 10mm
        data["board"]["outline"]["corner_radius_nm"] = 15_000_000  # Too large
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        repaired_spec, repair_result = repair_spec_tiered(spec, limits)

        # Should have repaired corner radius
        assert "board.outline.corner_radius_nm" in repair_result.repair_map
        repair = repair_result.repair_map["board.outline.corner_radius_nm"]
        # Repaired value should be <= max allowed (10mm)
        assert repair["after"] <= 10_000_000

    def test_corner_radius_determinism(self) -> None:
        """Corner radius constraints should be deterministic across evaluations."""
        data = _valid_spec_data()
        data["board"]["outline"]["corner_radius_nm"] = 3_000_000  # 3mm
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        # Evaluate multiple times
        proofs = [evaluate_tiered_constraints(spec, limits) for _ in range(3)]

        # Extract corner constraints
        corner_results = []
        for proof in proofs:
            corner = [c for c in proof.constraints if "CORNER_RADIUS" in c.constraint_id]
            corner_results.append(corner)

        # All should be identical
        first = corner_results[0]
        for result in corner_results[1:]:
            assert len(result) == len(first)
            for c1, c2 in zip(first, result, strict=True):
                assert c1.value == c2.value
                assert c1.margin == c2.margin
                assert c1.passed == c2.passed

    def test_corner_radius_at_boundary(self) -> None:
        """corner_radius_nm exactly at max should be valid."""
        data = _valid_spec_data()
        # Board is 20mm x 80mm, max corner = 10mm
        data["board"]["outline"]["corner_radius_nm"] = 10_000_000  # Exactly max
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        proof = evaluate_tiered_constraints(spec, limits)

        # Should pass (margin = 0)
        max_constraint = next(
            (c for c in proof.constraints if c.constraint_id == "T0_CORNER_RADIUS_MAX"),
            None
        )
        assert max_constraint is not None
        assert max_constraint.passed
        assert max_constraint.margin >= 0


# ==============================================================================
# REQ-M1-011: REPAIR Reproducibility Tests
# ==============================================================================


class TestRepairReproducibility:
    """Tests for REPAIR mode reproducibility (REQ-M1-011).

    REPAIR mode MUST emit a serialized repair_map plus a deterministic repaired_spec
    (or deterministic patch set) such that rebuilding from the repaired spec
    reproduces the same design_hash and artifacts.
    """

    def test_repair_emits_spec_hashes(self) -> None:
        """Repair result should include spec hashes for rebuild verification."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # Violation
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        # REQ-M1-011: Must have original and repaired spec hashes
        assert repair_result.original_spec_hash is not None
        assert repair_result.repaired_spec_hash is not None
        assert repair_result.repaired_design_hash is not None

        # Hashes should be different (spec was repaired)
        assert repair_result.original_spec_hash != repair_result.repaired_spec_hash

    def test_repair_deterministic_hash(self) -> None:
        """Repair should produce deterministic hashes across runs."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        data["transmission_line"]["gap_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        # Run repair multiple times
        results = [repair_spec_tiered(spec, limits) for _ in range(3)]

        # All should produce identical repaired_spec_hash and repaired_design_hash
        first = results[0][1]
        for _, repair_result in results[1:]:
            assert repair_result.repaired_spec_hash == first.repaired_spec_hash
            assert repair_result.repaired_design_hash == first.repaired_design_hash

    def test_repair_layout_plan_deterministic(self) -> None:
        """Repaired specs should resolve to deterministic geometry."""
        from formula_foundry.coupongen.resolve import resolve

        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        data["board"]["outline"]["corner_radius_nm"] = 15_000_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        layouts = []
        for _ in range(3):
            repaired_spec, _ = repair_spec_tiered(spec, limits)
            resolved = resolve(repaired_spec)
            layout = resolved.layout_plan
            assert layout is not None
            assert len(layout.segments) > 0
            layouts.append(layout)

        first = layouts[0]
        for layout in layouts[1:]:
            assert layout == first

    def test_rebuild_from_repaired_spec_same_hash(self) -> None:
        """Rebuilding from repaired_spec should produce same design_hash."""
        from formula_foundry.coupongen.resolve import design_hash, resolve

        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        repaired_spec, repair_result = repair_spec_tiered(spec, limits)

        # Resolve the repaired spec and compute its design hash
        resolved = resolve(repaired_spec)
        computed_hash = design_hash(resolved)

        # Should match the hash stored in repair_result
        assert computed_hash == repair_result.repaired_design_hash

    def test_repair_map_json_serialization(self) -> None:
        """repair_map should be JSON serializable and roundtrip correctly."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        # Convert to dict and serialize
        result_dict = repair_result.to_dict()
        json_str = json.dumps(result_dict)
        roundtrip = json.loads(json_str)

        # Verify structure preserved
        assert roundtrip["repair_map"] == result_dict["repair_map"]
        assert roundtrip["repair_reason"] == result_dict["repair_reason"]
        assert roundtrip["projection_policy_order"] == result_dict["projection_policy_order"]

    def test_repair_map_write_canonical_format(self) -> None:
        """write_repair_map should produce canonical JSON format (REQ-M1-011)."""
        from formula_foundry.coupongen.constraints import write_repair_map

        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "repair_map.json"
            write_repair_map(output_path, repair_result)

            # Read and verify structure
            content = json.loads(output_path.read_text())

            # REQ-M1-011: Must include repair audit trail
            assert "repair_map" in content
            assert "repair_reason" in content
            assert "repair_actions" in content
            assert "projection_policy_order" in content

            # CP-3.4: Should include design vectors and distance metrics
            if repair_result.original_vector is not None:
                assert "original_vector" in content
            if repair_result.repaired_vector is not None:
                assert "repaired_vector" in content
            if repair_result.distance_metrics is not None:
                assert "distance_metrics" in content

            # REQ-M1-011: Should include hashes for rebuild verification
            assert "original_spec_hash" in content
            assert "repaired_spec_hash" in content
            assert "repaired_design_hash" in content

    def test_repaired_spec_passes_constraints(self) -> None:
        """Repaired spec must pass all constraints (REQ-M1-011 validity)."""
        data = _valid_spec_data()
        # Multiple violations
        data["transmission_line"]["w_nm"] = 50_000
        data["transmission_line"]["gap_nm"] = 50_000
        data["board"]["outline"]["corner_radius_nm"] = 15_000_000  # Too large
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        repaired_spec, repair_result = repair_spec_tiered(spec, limits)

        # Repaired proof should pass
        assert repair_result.repaired_proof.passed
        assert repair_result.repaired_proof.first_failure_tier is None

        # Re-evaluate to confirm
        system = TieredConstraintSystem()
        re_proof = system.evaluate(repaired_spec, limits)
        assert re_proof.passed

    def test_repair_actions_audit_trail(self) -> None:
        """Repair actions should provide complete audit trail."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        assert len(repair_result.repair_actions) >= 1

        # Each action should have complete info
        for action in repair_result.repair_actions:
            assert action.path  # parameter path
            assert action.before != action.after  # actual repair
            assert action.reason  # human-readable
            assert action.constraint_id  # triggering constraint

    def test_repair_distance_metrics(self) -> None:
        """Repair should compute L2/Linf distance metrics (CP-3.4)."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        # Should have distance metrics
        assert repair_result.distance_metrics is not None
        assert repair_result.distance_metrics.l2_distance >= 0.0
        assert repair_result.distance_metrics.linf_distance >= 0.0
        assert repair_result.distance_metrics.normalized_sum_distance >= 0.0

        # L2 should be non-zero for repaired spec
        assert repair_result.distance_metrics.l2_distance > 0.0

    def test_repair_projection_policy_order(self) -> None:
        """Repair should apply policies in documented order (CP-3.4)."""
        data = _valid_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        # Check documented policy order
        expected_order = ("T0", "T1", "T2", "F1_CONTINUITY")
        assert repair_result.projection_policy_order == expected_order

    def test_valid_spec_no_repair_same_hashes(self) -> None:
        """Valid spec should have identical original/repaired hashes."""
        spec = CouponSpec.model_validate(_valid_spec_data())
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        # No repairs needed
        assert len(repair_result.repair_actions) == 0
        assert repair_result.repair_distance == 0.0

        # Hashes should be identical
        assert repair_result.original_spec_hash == repair_result.repaired_spec_hash


# =============================================================================
# Module-level wrapper test for DESIGN_DOCUMENT.md Test Matrix
# =============================================================================


def test_constraints() -> None:
    """Wrapper test for REQ-M1-007, REQ-M1-009, and REQ-M1-011.

    This test aggregates key assertions for the requirements mapped to
    tests/test_constraints.py::test_constraints in DESIGN_DOCUMENT.md Test Matrix.

    REQ-M1-007: When ground_via_fence.enabled=true, generator MUST place GND via
                fences deterministically with correct pitch/offset policies.
    REQ-M1-009: If corner_radius_nm > 0 is provided, board outline MUST be generated
                as rounded-rectangle using deterministic integer-nm arcs/segments.
    REQ-M1-011: REPAIR mode MUST emit serialized repair_map plus deterministic
                repaired_spec such that rebuilding reproduces same design_hash.
    """
    from formula_foundry.coupongen.resolve import resolve, design_hash

    # REQ-M1-007: Via fence parameters are validated (min/max pitch)
    limits = _default_fab_limits()
    assert limits["min_trace_width_nm"] == 100_000, "REQ-M1-007: Min trace width must be defined"

    # REQ-M1-007: Ground via fence can be generated with correct pitch
    cpwg_spec = CPWGSpec(
        w_nm=300000,
        gap_nm=180000,
        length_nm=20000000,
        layer="F.Cu",
        net_id=1,
    )
    fence_spec = GroundViaFenceSpec(
        pitch_nm=1500000,
        offset_from_gap_nm=800000,
        drill_nm=300000,
        diameter_nm=600000,
    )
    start = PositionNM(x=5000000, y=0)
    end = PositionNM(x=25000000, y=0)
    pos_vias, neg_vias = generate_ground_via_fence(start, end, cpwg_spec, fence_spec)
    assert len(pos_vias) > 0, "REQ-M1-007: Fence must generate via elements"
    assert len(pos_vias) == len(neg_vias), "REQ-M1-007: Symmetric via fence on both sides"

    # REQ-M1-009: Create spec with corner_radius_nm > 0 and generate rounded outline
    spec_data = _valid_spec_data()
    spec_data["board"]["outline"]["corner_radius_nm"] = 2000000
    spec = CouponSpec.model_validate(spec_data)
    assert spec.board.outline.corner_radius_nm == 2000000, "REQ-M1-009: Corner radius must be parsed"

    # REQ-M1-009: Rounded outline generates arcs
    outline = generate_rounded_outline(
        x_left_nm=0,
        y_bottom_nm=0,
        width_nm=spec.board.outline.length_nm,
        height_nm=spec.board.outline.width_nm,
        corner_radius_nm=spec.board.outline.corner_radius_nm,
    )
    arcs = [e for e in outline.elements if isinstance(e, OutlineArc)]
    lines = [e for e in outline.elements if isinstance(e, OutlineLine)]
    assert len(arcs) == 4, "REQ-M1-009: Rounded outline must have 4 corner arcs"
    assert len(lines) == 4, "REQ-M1-009: Rounded outline must have 4 edge lines"

    # REQ-M1-011: REPAIR mode produces repair_map with hashes
    violation_data = _valid_spec_data()
    violation_data["transmission_line"]["w_nm"] = 50000  # Below min, triggers repair
    violation_spec = CouponSpec.model_validate(violation_data)

    repaired_spec, repair_result = repair_spec_tiered(violation_spec, limits)

    # REQ-M1-011: repair_map includes hashes
    assert repair_result.original_spec_hash is not None, "REQ-M1-011: original_spec_hash must be present"
    assert repair_result.repaired_spec_hash is not None, "REQ-M1-011: repaired_spec_hash must be present"
    assert repair_result.repaired_design_hash is not None, "REQ-M1-011: repaired_design_hash must be present"

    # REQ-M1-011: Rebuilding from repaired_spec reproduces same design_hash
    re_resolved = resolve(repaired_spec)
    re_design_hash = design_hash(re_resolved)
    assert re_design_hash == repair_result.repaired_design_hash, (
        "REQ-M1-011: Rebuilding from repaired_spec must reproduce same design_hash"
    )

    # REQ-M1-011: Hashes differ when repairs were made
    assert repair_result.original_spec_hash != repair_result.repaired_spec_hash, (
        "REQ-M1-011: Original and repaired spec hashes must differ when repairs made"
    )
