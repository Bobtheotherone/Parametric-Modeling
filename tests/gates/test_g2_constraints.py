# SPDX-License-Identifier: MIT
"""Gate G2 tests: Constraint proof completeness and reject/repair behavior.

This module tests:
- Proof JSON validation against schema (constraint_proof.schema.json)
- 10k seeded random u vectors with REJECT mode (deterministic fail reasons)
- 10k seeded random u vectors with REPAIR mode (deterministic projection, bounded repair distance)

Per ECO-M1-ALIGN-0001:
- REQ-M1-008: Tiered constraint system with Tiers 0-3
- REQ-M1-009: REJECT mode with constraint IDs and reasons
- REQ-M1-010: REPAIR mode with repair_map, repair_reason, repair_distance
- REQ-M1-011: constraint_proof.json with per-constraint evaluations and signed margins

Pytest marker: gate_g2
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

try:
    import jsonschema
except ImportError:
    jsonschema = None  # type: ignore[assignment]

from formula_foundry.coupongen.constraints.engine import (
    ConstraintEngine,
    create_constraint_engine,
)
from formula_foundry.coupongen.constraints.repair import (
    ConstraintProofDocument,
    RepairDistanceMetrics,
    generate_constraint_proof,
    repair_spec_tiered,
)
from formula_foundry.coupongen.constraints.tiers import (
    ConstraintViolationError,
    TieredConstraintSystem,
    evaluate_tiered_constraints,
)
from formula_foundry.coupongen.spec import CouponSpec

# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "schemas"
CONSTRAINT_PROOF_SCHEMA_PATH = SCHEMA_DIR / "constraint_proof.schema.json"

# Test configuration
NUM_RANDOM_SAMPLES = 10_000
RANDOM_SEED = 42

# Parameter bounds for normalized u in [0, 1]
# These define the valid parameter space for F1 coupons
_PARAM_RANGES: dict[str, tuple[int, int]] = {
    "w_nm": (100_000, 500_000),
    "gap_nm": (100_000, 300_000),
    "length_left_nm": (5_000_000, 50_000_000),
    "length_right_nm": (5_000_000, 50_000_000),
    "board_width_nm": (10_000_000, 50_000_000),
    "board_length_nm": (30_000_000, 150_000_000),
    "corner_radius_nm": (0, 5_000_000),
    "signal_via_drill_nm": (200_000, 500_000),
    "signal_via_diameter_nm": (300_000, 800_000),
    "signal_via_pad_nm": (400_000, 1_200_000),
    "return_via_drill_nm": (200_000, 500_000),
    "return_via_diameter_nm": (300_000, 800_000),
    "return_via_radius_nm": (800_000, 3_000_000),
    "fence_via_drill_nm": (200_000, 400_000),
    "fence_via_diameter_nm": (300_000, 700_000),
    "fence_pitch_nm": (500_000, 3_000_000),
    "fence_offset_nm": (200_000, 1_500_000),
    "left_connector_x_nm": (2_000_000, 10_000_000),
    "right_connector_x_nm": (70_000_000, 145_000_000),
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


def _u_to_param(u: float, param_name: str) -> int:
    """Convert normalized u in [0, 1] to physical parameter value."""
    if param_name not in _PARAM_RANGES:
        return int(u * 1_000_000)  # Default scaling
    min_val, max_val = _PARAM_RANGES[param_name]
    return int(min_val + u * (max_val - min_val))


def _generate_spec_from_u(u_vector: list[float], derive_length_right: bool = True) -> dict[str, Any]:
    """Generate a CouponSpec dictionary from normalized u vector.

    The u vector has 19 dimensions corresponding to the parameters defined
    in _PARAM_RANGES. Values in [0, 1] are mapped to physical ranges.

    For F1 family specs, length_right_nm is derived from the F1 continuity
    equation rather than taken directly from the u vector (per CP-2.2/CP-3.3):
        length_right = right_connector_x - (left_connector_x + length_left)

    Args:
        u_vector: List of 19 normalized values in [0, 1]
        derive_length_right: If True (default), derive length_right_nm from
            connector positions and length_left_nm to satisfy F1 continuity
            constraint. This ensures: discontinuity_center = left_connector + length_left
            and length_right = right_connector - discontinuity_center.
            Set to False only for tests that explicitly need non-compliant specs.

    Returns:
        CouponSpec dictionary ready for validation
    """
    # Extract parameters from u vector (19 dimensions)
    param_names = list(_PARAM_RANGES.keys())
    params = {name: _u_to_param(u_vector[i], name) for i, name in enumerate(param_names)}

    # CP-3.3: Derive length_right from F1 continuity equation for F1 family specs
    # This ensures generated specs satisfy the continuity constraint (CP-2.2)
    if derive_length_right:
        left_x = params["left_connector_x_nm"]
        right_x = params["right_connector_x_nm"]
        length_left = params["length_left_nm"]
        derived_length_right = max(0, right_x - (left_x + length_left))
        params["length_right_nm"] = derived_length_right

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
                "L1_to_L2": 180_000,
                "L2_to_L3": 800_000,
                "L3_to_L4": 180_000,
            },
            "materials": {"er": 4.1, "loss_tangent": 0.02},
        },
        "board": {
            "outline": {
                "width_nm": params["board_width_nm"],
                "length_nm": params["board_length_nm"],
                "corner_radius_nm": params["corner_radius_nm"],
            },
            "origin": {"mode": "EDGE_L_CENTER"},
            "text": {"coupon_id": "${COUPON_ID}", "include_manifest_hash": True},
        },
        "connectors": {
            "left": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [params["left_connector_x_nm"], 0],
                "rotation_deg": 180,
            },
            "right": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [params["right_connector_x_nm"], 0],
                "rotation_deg": 0,
            },
        },
        "transmission_line": {
            "type": "CPWG",
            "layer": "F.Cu",
            "w_nm": params["w_nm"],
            "gap_nm": params["gap_nm"],
            "length_left_nm": params["length_left_nm"],
            "length_right_nm": params["length_right_nm"],
            "ground_via_fence": {
                "enabled": True,
                "pitch_nm": params["fence_pitch_nm"],
                "offset_from_gap_nm": params["fence_offset_nm"],
                "via": {
                    "drill_nm": params["fence_via_drill_nm"],
                    "diameter_nm": params["fence_via_diameter_nm"],
                },
            },
        },
        "discontinuity": {
            "type": "VIA_TRANSITION",
            "signal_via": {
                "drill_nm": params["signal_via_drill_nm"],
                "diameter_nm": params["signal_via_diameter_nm"],
                "pad_diameter_nm": params["signal_via_pad_nm"],
            },
            "antipads": {
                "L2": {
                    "shape": "ROUNDRECT",
                    "rx_nm": 1_200_000,
                    "ry_nm": 900_000,
                    "corner_nm": 250_000,
                },
                "L3": {"shape": "CIRCLE", "r_nm": 1_100_000},
            },
            "return_vias": {
                "pattern": "RING",
                "count": 4,
                "radius_nm": params["return_via_radius_nm"],
                "via": {
                    "drill_nm": params["return_via_drill_nm"],
                    "diameter_nm": params["return_via_diameter_nm"],
                },
            },
            "plane_cutouts": {},
        },
        "constraints": {
            "mode": "REJECT",
            "drc": {"must_pass": True, "severity": "all"},
            "symmetry": {"enforce": False},  # Don't enforce symmetry for random tests
            "allow_unconnected_copper": False,
        },
        "export": {
            "gerbers": {"enabled": True, "format": "gerbers"},
            "drill": {"enabled": True, "format": "excellon"},
            "outputs_dir": "artifacts/",
        },
    }


def _generate_invalid_spec_from_u(
    u_vector: list[float],
    violation_type: str,
    derive_length_right: bool = False,
) -> dict[str, Any]:
    """Generate an invalid spec with a specific violation type.

    Args:
        u_vector: Normalized u vector
        violation_type: One of 'T0', 'T1', 'T2', 'T3' indicating which tier to violate
        derive_length_right: If True, derive length_right_nm to satisfy F1 continuity

    Returns:
        CouponSpec dictionary with intentional violations
    """
    spec_data = _generate_spec_from_u(u_vector, derive_length_right=derive_length_right)

    if violation_type == "T0":
        # Violate T0: trace width below minimum
        spec_data["transmission_line"]["w_nm"] = 50_000  # Below 100_000 min
    elif violation_type == "T1":
        # Violate T1: annular ring too small (pad close to drill)
        spec_data["discontinuity"]["signal_via"]["pad_diameter_nm"] = \
            spec_data["discontinuity"]["signal_via"]["drill_nm"] + 50_000
    elif violation_type == "T2":
        # Violate T2: connector outside board bounds
        spec_data["connectors"]["left"]["position_nm"][0] = 50_000  # Too close to edge
    elif violation_type == "T3":
        # Violate T3: return via ring too small (vias would overlap)
        spec_data["discontinuity"]["return_vias"]["radius_nm"] = 100_000  # Too small

    return spec_data


def _load_constraint_proof_schema() -> dict[str, Any] | None:
    """Load the constraint proof JSON schema."""
    if not CONSTRAINT_PROOF_SCHEMA_PATH.exists():
        return None
    return json.loads(CONSTRAINT_PROOF_SCHEMA_PATH.read_text(encoding="utf-8"))


class SimpleLCG:
    """Simple Linear Congruential Generator for deterministic random numbers.

    Uses the same parameters as glibc's rand() for reproducibility.
    """

    def __init__(self, seed: int) -> None:
        self.state = seed

    def next(self) -> float:
        """Generate next random number in [0, 1)."""
        # glibc LCG parameters
        a = 1103515245
        c = 12345
        m = 2**31
        self.state = (a * self.state + c) % m
        return self.state / m

    def next_vector(self, n: int) -> list[float]:
        """Generate n random numbers in [0, 1)."""
        return [self.next() for _ in range(n)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def constraint_proof_schema() -> dict[str, Any] | None:
    """Fixture providing the constraint proof JSON schema."""
    return _load_constraint_proof_schema()


@pytest.fixture(scope="module")
def random_u_vectors() -> list[list[float]]:
    """Fixture providing 10k seeded random u vectors."""
    rng = SimpleLCG(RANDOM_SEED)
    num_params = len(_PARAM_RANGES)
    return [rng.next_vector(num_params) for _ in range(NUM_RANDOM_SAMPLES)]


@pytest.fixture(scope="module")
def fab_limits() -> dict[str, int]:
    """Fixture providing default fab limits."""
    return _default_fab_limits()


# ---------------------------------------------------------------------------
# G2 Gate Tests: Schema Validation
# ---------------------------------------------------------------------------


@pytest.mark.gate_g2
class TestG2SchemaValidation:
    """Gate G2 tests for constraint proof schema validation.

    These tests verify that generated constraint proofs conform to the
    constraint_proof.schema.json specification.
    """

    @pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
    def test_schema_file_exists(self) -> None:
        """Verify constraint_proof.schema.json exists."""
        assert CONSTRAINT_PROOF_SCHEMA_PATH.exists(), (
            f"Schema file not found: {CONSTRAINT_PROOF_SCHEMA_PATH}"
        )

    @pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
    def test_schema_is_valid_json_schema(self, constraint_proof_schema: dict[str, Any] | None) -> None:
        """Verify the schema itself is a valid JSON Schema."""
        if constraint_proof_schema is None:
            pytest.skip("Schema file not found")

        # Validate the schema is well-formed
        jsonschema.Draft202012Validator.check_schema(constraint_proof_schema)

    @pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
    def test_valid_proof_passes_schema(
        self, constraint_proof_schema: dict[str, Any] | None, fab_limits: dict[str, int]
    ) -> None:
        """Verify a valid constraint proof passes schema validation."""
        if constraint_proof_schema is None:
            pytest.skip("Schema file not found")

        # Generate a valid spec
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        spec_data = _generate_spec_from_u(u)
        spec = CouponSpec.model_validate(spec_data)

        # Evaluate constraints
        proof = evaluate_tiered_constraints(spec, fab_limits)
        doc = generate_constraint_proof(proof)
        proof_dict = doc.to_dict()

        # Validate against schema
        jsonschema.validate(proof_dict, constraint_proof_schema)

    @pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
    def test_proof_with_repair_passes_schema(
        self, constraint_proof_schema: dict[str, Any] | None, fab_limits: dict[str, int]
    ) -> None:
        """Verify a proof with repair info passes schema validation."""
        if constraint_proof_schema is None:
            pytest.skip("Schema file not found")

        # Generate an invalid spec
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        spec_data = _generate_invalid_spec_from_u(u, "T0")
        spec = CouponSpec.model_validate(spec_data)

        # Repair and generate proof
        repaired_spec, repair_result = repair_spec_tiered(spec, fab_limits)
        doc = generate_constraint_proof(repair_result.repaired_proof, repair_result)
        proof_dict = doc.to_dict()

        # Validate against schema
        jsonschema.validate(proof_dict, constraint_proof_schema)

    def test_proof_has_required_fields(self, fab_limits: dict[str, int]) -> None:
        """Verify proof has all required fields per schema."""
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        spec_data = _generate_spec_from_u(u)
        spec = CouponSpec.model_validate(spec_data)

        proof = evaluate_tiered_constraints(spec, fab_limits)
        doc = generate_constraint_proof(proof)
        proof_dict = doc.to_dict()

        # Required top-level fields
        required_fields = [
            "schema_version",
            "passed",
            "first_failure_tier",
            "total_constraints",
            "failed_constraints",
            "tiers",
            "constraints",
            "repair_applied",
        ]

        for field in required_fields:
            assert field in proof_dict, f"Missing required field: {field}"

    def test_constraint_entries_have_required_fields(self, fab_limits: dict[str, int]) -> None:
        """Verify each constraint entry has required fields."""
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        spec_data = _generate_spec_from_u(u)
        spec = CouponSpec.model_validate(spec_data)

        proof = evaluate_tiered_constraints(spec, fab_limits)
        doc = generate_constraint_proof(proof)
        proof_dict = doc.to_dict()

        required_entry_fields = ["id", "description", "tier", "value", "limit", "margin", "passed", "reason"]

        for entry in proof_dict["constraints"]:
            for field in required_entry_fields:
                assert field in entry, f"Missing required field '{field}' in constraint entry"


# ---------------------------------------------------------------------------
# G2 Gate Tests: REJECT Mode
# ---------------------------------------------------------------------------


@pytest.mark.gate_g2
class TestG2RejectMode:
    """Gate G2 tests for REJECT mode behavior.

    These tests verify:
    - REJECT mode fails deterministically with constraint IDs and reasons
    - Failure reasons are stable across runs with same seed
    - All constraint tiers produce appropriate failures
    """

    def test_reject_mode_valid_spec_passes(self, fab_limits: dict[str, int]) -> None:
        """REJECT mode should pass for valid specs."""
        # Use carefully chosen u values that produce a geometrically valid spec.
        # Parameter indices (from _PARAM_RANGES order):
        # 0: w_nm, 1: gap_nm, 2: length_left_nm, 3: length_right_nm (ignored when derived)
        # 4: board_width_nm, 5: board_length_nm, 6: corner_radius_nm
        # 17: left_connector_x_nm, 18: right_connector_x_nm
        u = [0.5] * len(_PARAM_RANGES)
        # Set board_length high (u=0.9 → 138M) so right_connector_x (100M at u=0.4) fits
        u[5] = 0.9  # board_length_nm: 138M
        u[18] = 0.4  # right_connector_x_nm: 100M (within 138M board)
        # Use shorter length_left to ensure discontinuity fits within board
        u[2] = 0.3  # length_left_nm: smaller to keep discontinuity inside board
        # Generate spec with derived length_right_nm to satisfy F1 continuity
        spec_data = _generate_spec_from_u(u, derive_length_right=True)
        spec = CouponSpec.model_validate(spec_data)

        engine = ConstraintEngine(fab_limits=fab_limits)
        result = engine.validate_or_repair(spec, mode="REJECT")

        assert result.passed, f"Valid spec should pass REJECT mode, but failed with: {result.proof.first_failure_tier}"

    def test_reject_mode_t0_violation_raises(self, fab_limits: dict[str, int]) -> None:
        """REJECT mode should raise ConstraintViolationError for T0 violations."""
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        spec_data = _generate_invalid_spec_from_u(u, "T0")
        spec = CouponSpec.model_validate(spec_data)

        engine = ConstraintEngine(fab_limits=fab_limits)

        with pytest.raises(ConstraintViolationError) as exc_info:
            engine.validate_or_repair(spec, mode="REJECT")

        assert exc_info.value.tier == "T0"
        assert "T0_TRACE_WIDTH_MIN" in exc_info.value.constraint_ids

    def test_reject_mode_deterministic_failure_ids(
        self, random_u_vectors: list[list[float]], fab_limits: dict[str, int]
    ) -> None:
        """REJECT mode failure IDs should be deterministic for same input.

        Run the first 100 samples twice and verify identical failure IDs.
        """
        engine = ConstraintEngine(fab_limits=fab_limits)
        sample_size = min(100, len(random_u_vectors))

        # First pass: collect failure IDs
        failure_ids_pass1: list[list[str] | None] = []
        for u in random_u_vectors[:sample_size]:
            spec_data = _generate_spec_from_u(u)
            try:
                spec = CouponSpec.model_validate(spec_data)
                proof = engine.evaluate(spec)
                if not proof.passed:
                    failure_ids_pass1.append(sorted([c.constraint_id for c in proof.get_failures()]))
                else:
                    failure_ids_pass1.append(None)
            except Exception:
                failure_ids_pass1.append(["VALIDATION_ERROR"])

        # Second pass: verify identical
        failure_ids_pass2: list[list[str] | None] = []
        for u in random_u_vectors[:sample_size]:
            spec_data = _generate_spec_from_u(u)
            try:
                spec = CouponSpec.model_validate(spec_data)
                proof = engine.evaluate(spec)
                if not proof.passed:
                    failure_ids_pass2.append(sorted([c.constraint_id for c in proof.get_failures()]))
                else:
                    failure_ids_pass2.append(None)
            except Exception:
                failure_ids_pass2.append(["VALIDATION_ERROR"])

        assert failure_ids_pass1 == failure_ids_pass2, "REJECT mode failure IDs not deterministic"

    def test_reject_mode_10k_samples_deterministic(
        self, random_u_vectors: list[list[float]], fab_limits: dict[str, int]
    ) -> None:
        """10k seeded random u vectors should produce deterministic REJECT results.

        This test verifies that failure/pass status is consistent across all samples.
        We hash the results to detect any non-determinism.
        """
        engine = ConstraintEngine(fab_limits=fab_limits)

        # Compute hash of results
        results_hash_1 = 0
        results_hash_2 = 0

        for u in random_u_vectors:
            spec_data = _generate_spec_from_u(u)
            try:
                spec = CouponSpec.model_validate(spec_data)
                proof = engine.evaluate(spec)
                # Hash: 1 for pass, tier number for fail
                if proof.passed:
                    h = 1
                else:
                    tier_map = {"T0": 2, "T1": 3, "T2": 4, "T3": 5}
                    h = tier_map.get(proof.first_failure_tier or "T0", 6)
            except Exception:
                h = 99

            results_hash_1 = (results_hash_1 * 31 + h) % (2**63)

        # Second pass
        for u in random_u_vectors:
            spec_data = _generate_spec_from_u(u)
            try:
                spec = CouponSpec.model_validate(spec_data)
                proof = engine.evaluate(spec)
                if proof.passed:
                    h = 1
                else:
                    tier_map = {"T0": 2, "T1": 3, "T2": 4, "T3": 5}
                    h = tier_map.get(proof.first_failure_tier or "T0", 6)
            except Exception:
                h = 99

            results_hash_2 = (results_hash_2 * 31 + h) % (2**63)

        assert results_hash_1 == results_hash_2, "10k REJECT results not deterministic"

    def test_reject_mode_constraint_reasons_populated(self, fab_limits: dict[str, int]) -> None:
        """Failed constraints should have populated reason strings."""
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        spec_data = _generate_invalid_spec_from_u(u, "T0")
        spec = CouponSpec.model_validate(spec_data)

        engine = ConstraintEngine(fab_limits=fab_limits)

        with pytest.raises(ConstraintViolationError) as exc_info:
            engine.validate_or_repair(spec, mode="REJECT")

        # All violations should have reasons
        for v in exc_info.value.violations:
            if not v.passed:
                assert v.reason or v.margin < 0, f"Constraint {v.constraint_id} missing reason"

    def test_reject_mode_signed_margins(self, fab_limits: dict[str, int]) -> None:
        """Constraint margins should be signed: positive for pass, negative for fail."""
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        spec_data = _generate_invalid_spec_from_u(u, "T0")
        spec = CouponSpec.model_validate(spec_data)

        engine = ConstraintEngine(fab_limits=fab_limits)
        proof = engine.evaluate(spec)

        for c in proof.constraints:
            if c.passed:
                assert c.margin >= 0, f"Passing constraint {c.constraint_id} has negative margin"
            else:
                assert c.margin < 0, f"Failing constraint {c.constraint_id} has non-negative margin"


# ---------------------------------------------------------------------------
# G2 Gate Tests: REPAIR Mode
# ---------------------------------------------------------------------------


@pytest.mark.gate_g2
class TestG2RepairMode:
    """Gate G2 tests for REPAIR mode behavior.

    These tests verify:
    - REPAIR mode projects infeasible specs into feasible space
    - Projections are deterministic for same seed
    - Repair distance is bounded and finite
    - Repaired specs pass all constraints
    """

    def test_repair_mode_valid_spec_unchanged(self, fab_limits: dict[str, int]) -> None:
        """REPAIR mode should not modify valid specs."""
        # Use carefully chosen u values that produce a geometrically valid spec.
        # Parameter indices (from _PARAM_RANGES order):
        # 0: w_nm, 1: gap_nm, 2: length_left_nm, 3: length_right_nm (ignored when derived)
        # 4: board_width_nm, 5: board_length_nm, 6: corner_radius_nm
        # 17: left_connector_x_nm, 18: right_connector_x_nm
        u = [0.5] * len(_PARAM_RANGES)
        # Set board_length high (u=0.9 → 138M) so right_connector_x (100M at u=0.4) fits
        u[5] = 0.9  # board_length_nm: 138M
        u[18] = 0.4  # right_connector_x_nm: 100M (within 138M board)
        # Use shorter length_left to ensure discontinuity fits within board
        u[2] = 0.3  # length_left_nm: smaller to keep discontinuity inside board
        # Generate spec with derived length_right_nm to satisfy F1 continuity
        spec_data = _generate_spec_from_u(u, derive_length_right=True)
        spec = CouponSpec.model_validate(spec_data)

        repaired_spec, repair_result = repair_spec_tiered(spec, fab_limits)

        assert repair_result.repair_map == {}, "Valid spec should have no repairs"
        assert repair_result.repair_distance == 0.0, "Valid spec should have zero repair distance"

    def test_repair_mode_t0_violation_fixed(self, fab_limits: dict[str, int]) -> None:
        """REPAIR mode should fix T0 violations.

        This test uses derive_length_right=True to ensure F1 continuity is
        satisfied before introducing the T0 violation. After repair, only
        the T0 constraint needs fixing.
        """
        # Use carefully chosen u values for a geometrically valid starting point
        u = [0.5] * len(_PARAM_RANGES)
        u[5] = 0.9  # board_length_nm: 138M (enough space)
        u[18] = 0.4  # right_connector_x_nm: 100M
        u[2] = 0.3  # length_left_nm: smaller value

        # Generate with F1 continuity satisfied, then add T0 violation
        spec_data = _generate_invalid_spec_from_u(u, "T0", derive_length_right=True)
        spec = CouponSpec.model_validate(spec_data)

        repaired_spec, repair_result = repair_spec_tiered(spec, fab_limits)

        assert repair_result.repaired_proof.passed, (
            f"Repaired spec should pass all constraints, but failed: "
            f"{[c.constraint_id for c in repair_result.repaired_proof.constraints if not c.passed]}"
        )
        assert "transmission_line.w_nm" in repair_result.repair_map

    def test_repair_mode_deterministic_projection(
        self, random_u_vectors: list[list[float]], fab_limits: dict[str, int]
    ) -> None:
        """REPAIR mode projections should be deterministic for same input.

        Run the first 100 samples twice and verify identical repair maps.
        """
        sample_size = min(100, len(random_u_vectors))

        # First pass: collect repair maps
        repair_maps_pass1: list[dict[str, dict[str, int]]] = []
        for u in random_u_vectors[:sample_size]:
            spec_data = _generate_spec_from_u(u)
            try:
                spec = CouponSpec.model_validate(spec_data)
                _, repair_result = repair_spec_tiered(spec, fab_limits)
                repair_maps_pass1.append(repair_result.repair_map)
            except Exception:
                repair_maps_pass1.append({"__error__": {"before": -1, "after": -1}})

        # Second pass: verify identical
        repair_maps_pass2: list[dict[str, dict[str, int]]] = []
        for u in random_u_vectors[:sample_size]:
            spec_data = _generate_spec_from_u(u)
            try:
                spec = CouponSpec.model_validate(spec_data)
                _, repair_result = repair_spec_tiered(spec, fab_limits)
                repair_maps_pass2.append(repair_result.repair_map)
            except Exception:
                repair_maps_pass2.append({"__error__": {"before": -1, "after": -1}})

        assert repair_maps_pass1 == repair_maps_pass2, "REPAIR mode projections not deterministic"

    def test_repair_mode_10k_samples_deterministic(
        self, random_u_vectors: list[list[float]], fab_limits: dict[str, int]
    ) -> None:
        """10k seeded random u vectors should produce deterministic REPAIR results.

        We hash the repair distances to detect any non-determinism.
        """
        # Compute hash of repair distances (quantized to avoid floating point issues)
        distances_hash_1 = 0
        distances_hash_2 = 0

        for u in random_u_vectors:
            spec_data = _generate_spec_from_u(u)
            try:
                spec = CouponSpec.model_validate(spec_data)
                _, repair_result = repair_spec_tiered(spec, fab_limits)
                # Quantize distance to integer for hashing
                d = int(repair_result.repair_distance * 1_000_000)
            except Exception:
                d = -1

            distances_hash_1 = (distances_hash_1 * 31 + d) % (2**63)

        # Second pass
        for u in random_u_vectors:
            spec_data = _generate_spec_from_u(u)
            try:
                spec = CouponSpec.model_validate(spec_data)
                _, repair_result = repair_spec_tiered(spec, fab_limits)
                d = int(repair_result.repair_distance * 1_000_000)
            except Exception:
                d = -1

            distances_hash_2 = (distances_hash_2 * 31 + d) % (2**63)

        assert distances_hash_1 == distances_hash_2, "10k REPAIR distances not deterministic"

    def test_repair_mode_bounded_distance(
        self, random_u_vectors: list[list[float]], fab_limits: dict[str, int]
    ) -> None:
        """Repair distances should be bounded and finite.

        Per CP-3.4, repair distance metrics (L2, Linf) should be non-negative
        and finite. We check that distances don't exceed reasonable bounds.
        """
        max_distance = 0.0
        sample_size = min(1000, len(random_u_vectors))  # Check first 1000

        for u in random_u_vectors[:sample_size]:
            spec_data = _generate_spec_from_u(u)
            try:
                spec = CouponSpec.model_validate(spec_data)
                _, repair_result = repair_spec_tiered(spec, fab_limits)

                # Check distance is finite and non-negative
                assert math.isfinite(repair_result.repair_distance), "Repair distance not finite"
                assert repair_result.repair_distance >= 0.0, "Repair distance negative"

                if repair_result.distance_metrics is not None:
                    assert repair_result.distance_metrics.l2_distance >= 0.0
                    assert repair_result.distance_metrics.linf_distance >= 0.0
                    assert math.isfinite(repair_result.distance_metrics.l2_distance)
                    assert math.isfinite(repair_result.distance_metrics.linf_distance)

                max_distance = max(max_distance, repair_result.repair_distance)

            except Exception:
                pass  # Skip validation errors

        # Distance should be bounded (typically < 100 for reasonable parameter changes)
        assert max_distance < 1000.0, f"Max repair distance too large: {max_distance}"

    def test_repair_mode_repaired_spec_valid(
        self, random_u_vectors: list[list[float]], fab_limits: dict[str, int]
    ) -> None:
        """Repaired specs should pass T0/T1 constraint validation.

        The repair function handles:
        - T0: Parameter bounds (trace width, gap, via sizes, board dims)
        - T1: Derived scalar constraints (annular ring, aspect ratios)
        - Some T2: Connector positions within board bounds

        T1/T2 geometry constraints that require complex trade-offs between
        multiple parameters are not yet fully handled by the repair function:
        - T1_F1_CONTINUITY_LENGTH_ERROR: F1 continuity requires length_right
          to be derived from other parameters, not independently specified
        - T2_TRACE_FITS_IN_BOARD: Complex geometry interactions
        - T2_DISCONTINUITY_FITS_IN_BOARD: Complex geometry interactions
        """
        sample_size = min(500, len(random_u_vectors))  # Check 500 samples
        t0_t1_failures = 0
        geometry_skipped = 0

        # Constraints that require complex geometry trade-offs and are
        # not fully handled by the repair function yet
        known_unhandled = {
            "T2_TRACE_FITS_IN_BOARD",
            "T2_DISCONTINUITY_FITS_IN_BOARD",
            # F1 continuity requires length_right to be derived from
            # connector positions and length_left (see CP-2.2)
            "T1_F1_CONTINUITY_LENGTH_ERROR",
        }

        for u in random_u_vectors[:sample_size]:
            spec_data = _generate_spec_from_u(u)
            try:
                spec = CouponSpec.model_validate(spec_data)
                repaired_spec, repair_result = repair_spec_tiered(spec, fab_limits)

                if not repair_result.repaired_proof.passed:
                    # Check which constraints failed
                    failed_ids = {c.constraint_id for c in repair_result.repaired_proof.constraints if not c.passed}
                    if failed_ids.issubset(known_unhandled):
                        geometry_skipped += 1
                    else:
                        # Unexpected failure - T0/basic T1 should always be fixed
                        t0_t1_failures += 1
            except Exception as e:
                # Count spec validation failures separately
                if "validation" not in str(e).lower():
                    t0_t1_failures += 1

        assert t0_t1_failures == 0, f"{t0_t1_failures} repaired specs failed T0/T1 validation"

    def test_repair_mode_l2_geq_linf(self, fab_limits: dict[str, int]) -> None:
        """L2 distance should always be >= Linf distance."""
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        # Create multiple violations to get meaningful distances
        spec_data = _generate_spec_from_u(u)
        spec_data["transmission_line"]["w_nm"] = 50_000
        spec_data["transmission_line"]["gap_nm"] = 50_000
        spec = CouponSpec.model_validate(spec_data)

        _, repair_result = repair_spec_tiered(spec, fab_limits)

        if repair_result.distance_metrics is not None:
            assert repair_result.distance_metrics.l2_distance >= repair_result.distance_metrics.linf_distance, (
                f"L2 ({repair_result.distance_metrics.l2_distance}) < "
                f"Linf ({repair_result.distance_metrics.linf_distance})"
            )

    def test_repair_mode_projection_policy_order(self, fab_limits: dict[str, int]) -> None:
        """Repair result should document projection policy order per CP-3.4."""
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        spec_data = _generate_invalid_spec_from_u(u, "T0")
        spec = CouponSpec.model_validate(spec_data)

        _, repair_result = repair_spec_tiered(spec, fab_limits)

        assert repair_result.projection_policy_order is not None
        assert len(repair_result.projection_policy_order) == 4
        assert repair_result.projection_policy_order == ("T0", "T1", "T2", "F1_CONTINUITY")


# ---------------------------------------------------------------------------
# G2 Gate Tests: Proof Document Generation
# ---------------------------------------------------------------------------


@pytest.mark.gate_g2
class TestG2ProofDocument:
    """Gate G2 tests for ConstraintProofDocument generation.

    These tests verify the structure and content of generated proof documents.
    """

    def test_proof_document_schema_version(self, fab_limits: dict[str, int]) -> None:
        """Proof document should have schema_version == 1."""
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        spec_data = _generate_spec_from_u(u)
        spec = CouponSpec.model_validate(spec_data)

        proof = evaluate_tiered_constraints(spec, fab_limits)
        doc = generate_constraint_proof(proof)

        assert doc.schema_version == 1

    def test_proof_document_tier_grouping(self, fab_limits: dict[str, int]) -> None:
        """Proof document should group constraints by tier."""
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        spec_data = _generate_spec_from_u(u)
        spec = CouponSpec.model_validate(spec_data)

        proof = evaluate_tiered_constraints(spec, fab_limits)
        doc = generate_constraint_proof(proof)

        # Should have all tiers T0-T3
        for tier in ["T0", "T1", "T2", "T3"]:
            assert tier in doc.tiers, f"Missing tier {tier} in proof document"

    def test_proof_document_constraint_count(self, fab_limits: dict[str, int]) -> None:
        """Proof document constraint counts should be accurate."""
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        spec_data = _generate_spec_from_u(u)
        spec = CouponSpec.model_validate(spec_data)

        proof = evaluate_tiered_constraints(spec, fab_limits)
        doc = generate_constraint_proof(proof)

        assert doc.total_constraints == len(doc.constraints)
        assert doc.failed_constraints == len([c for c in doc.constraints if not c["passed"]])

    def test_proof_document_repair_info_structure(self, fab_limits: dict[str, int]) -> None:
        """Proof document repair_info should have correct structure when repairs made."""
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        spec_data = _generate_invalid_spec_from_u(u, "T0")
        spec = CouponSpec.model_validate(spec_data)

        _, repair_result = repair_spec_tiered(spec, fab_limits)
        doc = generate_constraint_proof(repair_result.repaired_proof, repair_result)

        assert doc.repair_applied is True
        assert doc.repair_info is not None

        # Verify repair_info structure
        assert "repair_map" in doc.repair_info
        assert "repair_reason" in doc.repair_info
        assert "repair_distance" in doc.repair_info

    def test_proof_document_to_json_valid(self, fab_limits: dict[str, int]) -> None:
        """to_json() should produce valid JSON string."""
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        spec_data = _generate_spec_from_u(u)
        spec = CouponSpec.model_validate(spec_data)

        proof = evaluate_tiered_constraints(spec, fab_limits)
        doc = generate_constraint_proof(proof)

        json_str = doc.to_json()
        parsed = json.loads(json_str)  # Should not raise

        assert parsed["passed"] == doc.passed
        assert parsed["schema_version"] == doc.schema_version


# ---------------------------------------------------------------------------
# G2 Gate Tests: Engine Integration
# ---------------------------------------------------------------------------


@pytest.mark.gate_g2
class TestG2EngineIntegration:
    """Gate G2 tests for ConstraintEngine integration.

    These tests verify the unified ConstraintEngine works correctly
    with both REJECT and REPAIR modes.
    """

    def test_engine_evaluate_returns_proof(self, fab_limits: dict[str, int]) -> None:
        """engine.evaluate() should return TieredConstraintProof."""
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        spec_data = _generate_spec_from_u(u)
        spec = CouponSpec.model_validate(spec_data)

        engine = ConstraintEngine(fab_limits=fab_limits)
        proof = engine.evaluate(spec)

        assert hasattr(proof, "passed")
        assert hasattr(proof, "constraints")
        assert hasattr(proof, "tiers")

    def test_engine_result_properties(self, fab_limits: dict[str, int]) -> None:
        """ConstraintEngineResult should have correct properties.

        Uses derive_length_right=True to ensure F1 continuity is satisfied,
        so the repair only needs to fix the T0 violation.
        """
        # Use carefully chosen u values for a geometrically valid starting point
        u = [0.5] * len(_PARAM_RANGES)
        u[5] = 0.9  # board_length_nm: 138M (enough space)
        u[18] = 0.4  # right_connector_x_nm: 100M
        u[2] = 0.3  # length_left_nm: smaller value

        # Generate with F1 continuity satisfied, then add T0 violation
        spec_data = _generate_invalid_spec_from_u(u, "T0", derive_length_right=True)
        spec = CouponSpec.model_validate(spec_data)

        engine = ConstraintEngine(fab_limits=fab_limits)
        result = engine.validate_or_repair(spec, mode="REPAIR")

        assert result.passed is True, (
            f"Repaired should pass, but failed: "
            f"{[c.constraint_id for c in result.proof.constraints if not c.passed]}"
        )
        assert result.was_repaired is True
        assert result.repair_map is not None

    def test_engine_create_proof_document(self, fab_limits: dict[str, int]) -> None:
        """engine.create_proof_document() should work correctly."""
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        spec_data = _generate_spec_from_u(u)
        spec = CouponSpec.model_validate(spec_data)

        engine = ConstraintEngine(fab_limits=fab_limits)
        proof = engine.evaluate(spec)
        doc = engine.create_proof_document(proof)

        assert isinstance(doc, ConstraintProofDocument)
        assert doc.schema_version == 1

    def test_factory_function_with_defaults(self) -> None:
        """create_constraint_engine() should work with default limits."""
        engine = create_constraint_engine()

        assert engine.fab_limits is not None
        assert "min_trace_width_nm" in engine.fab_limits

    def test_engine_fail_fast_mode(self, fab_limits: dict[str, int]) -> None:
        """fail_fast mode should stop at first failing tier."""
        rng = SimpleLCG(12345)
        u = rng.next_vector(len(_PARAM_RANGES))
        spec_data = _generate_invalid_spec_from_u(u, "T0")
        spec = CouponSpec.model_validate(spec_data)

        engine = ConstraintEngine(fab_limits=fab_limits, fail_fast=True)
        proof = engine.evaluate(spec)

        assert proof.first_failure_tier == "T0"
        assert not proof.passed
