"""Tests for tiered REPAIR mode and constraint_proof.json generation.

This module tests:
- REQ-M1-010: REPAIR mode projecting infeasible specs into feasible space
  with repair_map, repair_reason, and repair_distance
- REQ-M1-011: constraint_proof.json with per-constraint evaluations and
  signed margins
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.coupongen.constraints.repair import (
    RepairAction,
    RepairEngine,
    generate_constraint_proof,
    repair_spec_tiered,
    write_constraint_proof,
)
from formula_foundry.coupongen.constraints.tiers import (
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
            "length_left_nm": 25000000,
            "length_right_nm": 25000000,
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


class TestRepairModeTiered:
    """Test REPAIR mode for tiered constraint system (REQ-M1-010)."""

    def test_valid_spec_returns_no_repairs(self) -> None:
        """Valid spec should not require any repairs."""
        spec = CouponSpec.model_validate(_example_spec_data())
        limits = _default_fab_limits()

        repaired_spec, repair_result = repair_spec_tiered(spec, limits)

        assert repair_result.repair_map == {}
        assert repair_result.repair_reason == []
        assert repair_result.repair_distance == 0.0
        assert repair_result.repair_actions == ()
        assert repair_result.repaired_proof.passed

    def test_repair_trace_width_violation(self) -> None:
        """Repair should fix trace width below minimum."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # Below 100_000 min
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        repaired_spec, repair_result = repair_spec_tiered(spec, limits)

        assert "transmission_line.w_nm" in repair_result.repair_map
        assert repair_result.repair_map["transmission_line.w_nm"]["before"] == 50_000
        assert repair_result.repair_map["transmission_line.w_nm"]["after"] == 100_000
        assert repair_result.repair_distance > 0.0
        assert int(repaired_spec.transmission_line.w_nm) == 100_000

    def test_repair_gap_violation(self) -> None:
        """Repair should fix CPWG gap below minimum."""
        data = _example_spec_data()
        data["transmission_line"]["gap_nm"] = 50_000  # Below 100_000 min
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        repaired_spec, repair_result = repair_spec_tiered(spec, limits)

        assert "transmission_line.gap_nm" in repair_result.repair_map
        assert repair_result.repair_map["transmission_line.gap_nm"]["after"] == 100_000
        assert int(repaired_spec.transmission_line.gap_nm) == 100_000

    def test_repair_signal_via_drill_violation(self) -> None:
        """Repair should fix signal via drill below minimum."""
        data = _example_spec_data()
        data["discontinuity"]["signal_via"]["drill_nm"] = 100_000  # Below 200_000 min
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        repaired_spec, repair_result = repair_spec_tiered(spec, limits)

        assert "discontinuity.signal_via.drill_nm" in repair_result.repair_map
        repair_info = repair_result.repair_map["discontinuity.signal_via.drill_nm"]
        assert repair_info["before"] == 100_000
        assert repair_info["after"] == 200_000

    def test_repair_multiple_violations(self) -> None:
        """Repair should fix multiple violations simultaneously."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # Violation
        data["transmission_line"]["gap_nm"] = 50_000  # Violation
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        repaired_spec, repair_result = repair_spec_tiered(spec, limits)

        assert len(repair_result.repair_map) >= 2
        assert "transmission_line.w_nm" in repair_result.repair_map
        assert "transmission_line.gap_nm" in repair_result.repair_map
        assert len(repair_result.repair_reason) >= 2

    def test_repair_reason_list_populated(self) -> None:
        """Repair should provide human-readable reasons."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        assert len(repair_result.repair_reason) >= 1
        reason = repair_result.repair_reason[0]
        assert "50000" in reason or "100000" in reason  # Should mention values
        assert isinstance(reason, str)

    def test_repair_distance_normalized(self) -> None:
        """Repair distance should be normalized relative to original values."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000  # 50% change to 100_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        # Distance = |100_000 - 50_000| / 50_000 = 1.0
        assert repair_result.repair_distance == pytest.approx(1.0, rel=0.01)

    def test_repair_actions_include_constraint_id(self) -> None:
        """Repair actions should reference the constraint that triggered them."""
        data = _example_spec_data()
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

    def test_repair_result_to_dict(self) -> None:
        """RepairResult should serialize to dictionary properly."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)
        result_dict = repair_result.to_dict()

        assert "repair_map" in result_dict
        assert "repair_reason" in result_dict
        assert "repair_distance" in result_dict
        assert "repair_actions" in result_dict
        assert "original_passed" in result_dict
        assert "repaired_passed" in result_dict
        assert result_dict["original_passed"] is False
        assert result_dict["repaired_passed"] is True

    def test_repaired_spec_passes_all_constraints(self) -> None:
        """Repaired spec should pass all constraints."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        data["transmission_line"]["gap_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        repaired_spec, repair_result = repair_spec_tiered(spec, limits)

        # Verify the repaired proof passes
        assert repair_result.repaired_proof.passed
        assert repair_result.repaired_proof.first_failure_tier is None

        # Double-check by re-evaluating
        system = TieredConstraintSystem()
        re_proof = system.evaluate(repaired_spec, limits)
        assert re_proof.passed


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
                "length_left_nm": 25000000,
                "length_right_nm": 25000000,
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

    def test_repair_tier1_diameter_less_than_drill(self) -> None:
        """RepairEngine should fix diameter when less than or equal to drill."""
        limits = _default_fab_limits()
        engine = RepairEngine(limits)
        payload = {
            "transmission_line": {
                "length_left_nm": 25000000,
                "length_right_nm": 25000000,
                "ground_via_fence": None,
            },
            "discontinuity": {
                "signal_via": {
                    "drill_nm": 200_000,
                    "diameter_nm": 200_000,  # Same as drill - needs to be larger
                    "pad_diameter_nm": 300_000,
                }
            },
        }

        engine.repair_tier1(payload)

        # Diameter should be raised: drill + min_annular = 200_000 + 100_000 = 300_000
        assert payload["discontinuity"]["signal_via"]["diameter_nm"] == 300_000

    def test_repair_tier1_pad_annular_ring(self) -> None:
        """RepairEngine should fix pad when annular ring is too small."""
        limits = _default_fab_limits()
        engine = RepairEngine(limits)
        payload = {
            "transmission_line": {
                "length_left_nm": 25000000,
                "length_right_nm": 25000000,
                "ground_via_fence": None,
            },
            "discontinuity": {
                "signal_via": {
                    "drill_nm": 200_000,
                    "diameter_nm": 400_000,  # OK - larger than drill
                    "pad_diameter_nm": 410_000,  # Too small: needs diameter + 2*annular
                }
            },
        }

        engine.repair_tier1(payload)

        # Pad should be raised: diameter + 2*annular = 400_000 + 200_000 = 600_000
        assert payload["discontinuity"]["signal_via"]["pad_diameter_nm"] == 600_000


class TestConstraintProofDocument:
    """Test constraint_proof.json generation (REQ-M1-011)."""

    def test_generate_constraint_proof_structure(self) -> None:
        """Generated proof should have required schema structure."""
        spec = CouponSpec.model_validate(_example_spec_data())
        limits = _default_fab_limits()
        proof = evaluate_tiered_constraints(spec, limits)

        doc = generate_constraint_proof(proof)

        assert doc.schema_version == 1
        assert doc.passed is True
        assert doc.first_failure_tier is None
        assert doc.total_constraints > 0
        assert doc.failed_constraints == 0
        assert "T0" in doc.tiers
        assert "T1" in doc.tiers
        assert "T2" in doc.tiers
        assert "T3" in doc.tiers

    def test_constraint_entries_have_signed_margins(self) -> None:
        """Each constraint entry should have a signed margin."""
        spec = CouponSpec.model_validate(_example_spec_data())
        limits = _default_fab_limits()
        proof = evaluate_tiered_constraints(spec, limits)

        doc = generate_constraint_proof(proof)

        for entry in doc.constraints:
            assert "id" in entry
            assert "description" in entry
            assert "tier" in entry
            assert "value" in entry
            assert "limit" in entry
            assert "margin" in entry  # Signed margin
            assert "passed" in entry
            assert "reason" in entry

    def test_signed_margins_positive_for_passing(self) -> None:
        """Passing constraints should have non-negative margins."""
        spec = CouponSpec.model_validate(_example_spec_data())
        limits = _default_fab_limits()
        proof = evaluate_tiered_constraints(spec, limits)

        doc = generate_constraint_proof(proof)

        passing_entries = [e for e in doc.constraints if e["passed"]]
        for entry in passing_entries:
            assert entry["margin"] >= 0, f"{entry['id']} has negative margin {entry['margin']}"

    def test_signed_margins_negative_for_failing(self) -> None:
        """Failing constraints should have negative margins."""
        data = _example_spec_data()
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

    def test_constraint_proof_with_repair_info(self) -> None:
        """Proof should include repair info when REPAIR mode was used."""
        data = _example_spec_data()
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

    def test_to_dict_structure(self) -> None:
        """to_dict should produce valid dictionary structure."""
        spec = CouponSpec.model_validate(_example_spec_data())
        limits = _default_fab_limits()
        proof = evaluate_tiered_constraints(spec, limits)
        doc = generate_constraint_proof(proof)

        result = doc.to_dict()

        assert isinstance(result, dict)
        assert result["schema_version"] == 1
        assert result["passed"] is True
        assert isinstance(result["tiers"], dict)
        assert isinstance(result["constraints"], list)

    def test_to_json_produces_valid_json(self) -> None:
        """to_json should produce valid JSON string."""
        spec = CouponSpec.model_validate(_example_spec_data())
        limits = _default_fab_limits()
        proof = evaluate_tiered_constraints(spec, limits)
        doc = generate_constraint_proof(proof)

        json_str = doc.to_json()
        parsed = json.loads(json_str)

        assert parsed["passed"] is True
        assert "constraints" in parsed


class TestWriteConstraintProof:
    """Test writing constraint_proof.json to file."""

    def test_write_constraint_proof_creates_file(self) -> None:
        """write_constraint_proof should create a JSON file."""
        spec = CouponSpec.model_validate(_example_spec_data())
        limits = _default_fab_limits()
        proof = evaluate_tiered_constraints(spec, limits)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "constraint_proof.json"
            write_constraint_proof(output_path, proof)

            assert output_path.exists()
            content = json.loads(output_path.read_text())
            assert content["passed"] is True

    def test_write_constraint_proof_creates_directories(self) -> None:
        """write_constraint_proof should create parent directories if needed."""
        spec = CouponSpec.model_validate(_example_spec_data())
        limits = _default_fab_limits()
        proof = evaluate_tiered_constraints(spec, limits)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "constraint_proof.json"
            write_constraint_proof(output_path, proof)

            assert output_path.exists()

    def test_write_constraint_proof_with_repair(self) -> None:
        """write_constraint_proof should include repair info when provided."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "constraint_proof.json"
            write_constraint_proof(output_path, repair_result.repaired_proof, repair_result)

            content = json.loads(output_path.read_text())
            assert content["repair_applied"] is True
            assert content["repair_info"] is not None
            assert "repair_map" in content["repair_info"]

    def test_constraint_proof_json_schema_compliance(self) -> None:
        """Generated constraint_proof.json should match expected schema."""
        data = _example_spec_data()
        data["transmission_line"]["w_nm"] = 50_000
        spec = CouponSpec.model_validate(data)
        limits = _default_fab_limits()

        _, repair_result = repair_spec_tiered(spec, limits)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "constraint_proof.json"
            write_constraint_proof(output_path, repair_result.repaired_proof, repair_result)

            content = json.loads(output_path.read_text())

            # Verify required top-level fields
            assert "schema_version" in content
            assert "passed" in content
            assert "first_failure_tier" in content
            assert "total_constraints" in content
            assert "failed_constraints" in content
            assert "tiers" in content
            assert "constraints" in content
            assert "repair_applied" in content

            # Verify constraint entry fields
            for entry in content["constraints"]:
                assert "id" in entry
                assert "description" in entry
                assert "tier" in entry
                assert "value" in entry
                assert "limit" in entry
                assert "margin" in entry
                assert "passed" in entry
                assert "reason" in entry


class TestRepairAction:
    """Test RepairAction dataclass."""

    def test_repair_action_fields(self) -> None:
        """RepairAction should have all required fields."""
        action = RepairAction(
            path="transmission_line.w_nm",
            before=50_000,
            after=100_000,
            reason="Trace width raised to fab minimum",
            constraint_id="T0_TRACE_WIDTH_MIN",
        )

        assert action.path == "transmission_line.w_nm"
        assert action.before == 50_000
        assert action.after == 100_000
        assert action.reason == "Trace width raised to fab minimum"
        assert action.constraint_id == "T0_TRACE_WIDTH_MIN"

    def test_repair_action_immutable(self) -> None:
        """RepairAction should be immutable (frozen dataclass)."""
        action = RepairAction(
            path="test",
            before=1,
            after=2,
            reason="test",
            constraint_id="TEST",
        )

        with pytest.raises(AttributeError):
            action.before = 999  # type: ignore[misc]
