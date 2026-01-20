"""REPAIR mode implementation for constraint system.

This module implements REPAIR mode that projects infeasible specs into
feasible space with full auditability through repair_map, repair_reason,
and repair_distance.

It also provides constraint_proof.json generation with per-constraint
evaluations and signed margins.

REQ-M1-010: REPAIR mode must project infeasible specs into feasible space
            with auditable repair_map, repair_reason list, and repair_distance.
REQ-M1-011: Every generated design must emit a constraint_proof.json with
            per-constraint evaluations and signed margins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .tiers import (
    ConstraintResult,
    ConstraintTier,
    TieredConstraintProof,
    TieredConstraintSystem,
    _TIERS,
)

if TYPE_CHECKING:
    from ..spec import CouponSpec


@dataclass(frozen=True, slots=True)
class RepairAction:
    """A single repair action applied to a spec parameter.

    Attributes:
        path: Dotted path to the repaired parameter (e.g., "transmission_line.w_nm")
        before: Original value before repair
        after: Repaired value
        reason: Human-readable explanation of why the repair was needed
        constraint_id: ID of the constraint that triggered this repair
    """

    path: str
    before: int
    after: int
    reason: str
    constraint_id: str


@dataclass(frozen=True)
class RepairResult:
    """Result of a REPAIR mode projection.

    Attributes:
        repair_map: Mapping from parameter paths to {before, after} dicts
        repair_reason: List of human-readable repair explanations
        repair_distance: Normalized total distance of all repairs
        repair_actions: Detailed list of all repair actions taken
        original_proof: Constraint proof before repair
        repaired_proof: Constraint proof after repair (should pass)
    """

    repair_map: dict[str, dict[str, int]]
    repair_reason: list[str]
    repair_distance: float
    repair_actions: tuple[RepairAction, ...]
    original_proof: TieredConstraintProof
    repaired_proof: TieredConstraintProof

    def to_dict(self) -> dict[str, Any]:
        """Convert repair result to a dictionary for serialization."""
        return {
            "repair_map": self.repair_map,
            "repair_reason": self.repair_reason,
            "repair_distance": self.repair_distance,
            "repair_actions": [
                {
                    "path": a.path,
                    "before": a.before,
                    "after": a.after,
                    "reason": a.reason,
                    "constraint_id": a.constraint_id,
                }
                for a in self.repair_actions
            ],
            "original_passed": self.original_proof.passed,
            "repaired_passed": self.repaired_proof.passed,
        }


@dataclass
class ConstraintProofDocument:
    """A constraint proof document for JSON serialization.

    This is the canonical format for constraint_proof.json files,
    containing per-constraint evaluations with signed margins.

    Attributes:
        schema_version: Version of the constraint proof schema
        passed: Overall pass/fail status
        first_failure_tier: First tier with failures (None if passed)
        total_constraints: Total number of constraints evaluated
        failed_constraints: Number of failed constraints
        tiers: Constraint IDs grouped by tier
        constraints: Full per-constraint evaluation details
        repair_applied: Whether REPAIR mode was applied
        repair_info: Repair details if REPAIR was applied
    """

    schema_version: int = 1
    passed: bool = True
    first_failure_tier: ConstraintTier | None = None
    total_constraints: int = 0
    failed_constraints: int = 0
    tiers: dict[ConstraintTier, list[str]] = field(default_factory=dict)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    repair_applied: bool = False
    repair_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "first_failure_tier": self.first_failure_tier,
            "total_constraints": self.total_constraints,
            "failed_constraints": self.failed_constraints,
            "tiers": self.tiers,
            "constraints": self.constraints,
            "repair_applied": self.repair_applied,
        }
        if self.repair_info is not None:
            result["repair_info"] = self.repair_info
        return result

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def write_to_file(self, path: Path | str) -> None:
        """Write constraint proof to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")


def _compute_repair_distance(repair_actions: list[RepairAction]) -> float:
    """Compute normalized repair distance from all repair actions.

    The distance is the sum of relative changes:
        sum(|after - before| / max(|before|, 1))

    This normalizes by the original value to make the distance
    comparable across parameters with different scales.

    Args:
        repair_actions: List of repair actions taken

    Returns:
        Total normalized repair distance
    """
    total = 0.0
    for action in repair_actions:
        before = float(action.before)
        after = float(action.after)
        denom = max(abs(before), 1.0)
        total += abs(after - before) / denom
    return total


def _get_nested_value(data: dict[str, Any], path: str) -> Any:
    """Get a nested value from a dictionary using dotted path notation."""
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _set_nested_value(data: dict[str, Any], path: str, value: Any) -> None:
    """Set a nested value in a dictionary using dotted path notation."""
    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


class RepairEngine:
    """Engine for projecting infeasible specs into feasible space.

    The repair engine applies a series of repair strategies to fix
    constraint violations, starting from the lowest tier and working up.
    """

    def __init__(self, fab_limits: dict[str, int]) -> None:
        """Initialize the repair engine.

        Args:
            fab_limits: Dictionary of fab capability limits in nm
        """
        self.fab_limits = fab_limits
        self.actions: list[RepairAction] = []

    def _record(
        self, path: str, before: int, after: int, reason: str, constraint_id: str
    ) -> int:
        """Record a repair action and return the repaired value."""
        if before != after:
            self.actions.append(
                RepairAction(
                    path=path,
                    before=before,
                    after=after,
                    reason=reason,
                    constraint_id=constraint_id,
                )
            )
        return after

    def _clamp_min(
        self,
        path: str,
        value: int,
        limit: int,
        reason_template: str,
        constraint_id: str,
    ) -> int:
        """Clamp a value to a minimum and record the repair if needed."""
        if value >= limit:
            return value
        reason = reason_template.format(value=value, limit=limit)
        return self._record(path, value, limit, reason, constraint_id)

    def _clamp_max(
        self,
        path: str,
        value: int,
        limit: int,
        reason_template: str,
        constraint_id: str,
    ) -> int:
        """Clamp a value to a maximum and record the repair if needed."""
        if value <= limit:
            return value
        reason = reason_template.format(value=value, limit=limit)
        return self._record(path, value, limit, reason, constraint_id)

    def repair_tier0(self, payload: dict[str, Any]) -> None:
        """Apply Tier 0 repairs: parameter bounds."""
        # Trace width minimum
        tl = payload["transmission_line"]
        min_trace = self.fab_limits.get("min_trace_width_nm", 100_000)
        tl["w_nm"] = self._clamp_min(
            "transmission_line.w_nm",
            int(tl["w_nm"]),
            min_trace,
            "Trace width {value}nm raised to fab minimum {limit}nm",
            "T0_TRACE_WIDTH_MIN",
        )

        # Trace gap minimum
        min_gap = self.fab_limits.get("min_gap_nm", 100_000)
        tl["gap_nm"] = self._clamp_min(
            "transmission_line.gap_nm",
            int(tl["gap_nm"]),
            min_gap,
            "CPWG gap {value}nm raised to fab minimum {limit}nm",
            "T0_TRACE_GAP_MIN",
        )

        # Board outline constraints
        board = payload["board"]["outline"]
        min_board_dim = self.fab_limits.get("min_board_width_nm", 5_000_000)

        board["width_nm"] = self._clamp_min(
            "board.outline.width_nm",
            int(board["width_nm"]),
            min_board_dim,
            "Board width {value}nm raised to fab minimum {limit}nm",
            "T0_BOARD_WIDTH_MIN",
        )

        board["length_nm"] = self._clamp_min(
            "board.outline.length_nm",
            int(board["length_nm"]),
            min_board_dim,
            "Board length {value}nm raised to fab minimum {limit}nm",
            "T0_BOARD_LENGTH_MIN",
        )

        # Corner radius bounds
        max_corner = min(int(board["width_nm"]), int(board["length_nm"])) // 2
        corner = int(board["corner_radius_nm"])

        if corner < 0:
            board["corner_radius_nm"] = self._record(
                "board.outline.corner_radius_nm",
                corner,
                0,
                f"Corner radius {corner}nm raised to minimum 0nm",
                "T0_CORNER_RADIUS_MIN",
            )
            corner = 0

        if corner > max_corner:
            board["corner_radius_nm"] = self._record(
                "board.outline.corner_radius_nm",
                corner,
                max_corner,
                f"Corner radius {corner}nm reduced to max {max_corner}nm",
                "T0_CORNER_RADIUS_MAX",
            )

        # Via/drill constraints
        if payload.get("discontinuity") is not None:
            disc = payload["discontinuity"]
            signal_via = disc["signal_via"]
            min_drill = self.fab_limits.get("min_drill_nm", 200_000)
            min_via_dia = self.fab_limits.get("min_via_diameter_nm", 300_000)

            signal_via["drill_nm"] = self._clamp_min(
                "discontinuity.signal_via.drill_nm",
                int(signal_via["drill_nm"]),
                min_drill,
                "Signal via drill {value}nm raised to fab minimum {limit}nm",
                "T0_SIGNAL_DRILL_MIN",
            )

            signal_via["diameter_nm"] = self._clamp_min(
                "discontinuity.signal_via.diameter_nm",
                int(signal_via["diameter_nm"]),
                min_via_dia,
                "Signal via diameter {value}nm raised to fab minimum {limit}nm",
                "T0_SIGNAL_VIA_DIAMETER_MIN",
            )

            signal_via["pad_diameter_nm"] = self._clamp_min(
                "discontinuity.signal_via.pad_diameter_nm",
                int(signal_via["pad_diameter_nm"]),
                min_via_dia,
                "Signal via pad diameter {value}nm raised to fab minimum {limit}nm",
                "T0_SIGNAL_PAD_DIAMETER_MIN",
            )

            # Return vias
            if disc.get("return_vias") is not None:
                return_via = disc["return_vias"]["via"]

                return_via["drill_nm"] = self._clamp_min(
                    "discontinuity.return_vias.via.drill_nm",
                    int(return_via["drill_nm"]),
                    min_drill,
                    "Return via drill {value}nm raised to fab minimum {limit}nm",
                    "T0_RETURN_VIA_DRILL_MIN",
                )

                return_via["diameter_nm"] = self._clamp_min(
                    "discontinuity.return_vias.via.diameter_nm",
                    int(return_via["diameter_nm"]),
                    min_via_dia,
                    "Return via diameter {value}nm raised to fab minimum {limit}nm",
                    "T0_RETURN_VIA_DIAMETER_MIN",
                )

        # Ground via fence constraints
        fence = payload["transmission_line"].get("ground_via_fence")
        if fence is not None and fence.get("enabled", False):
            min_drill = self.fab_limits.get("min_drill_nm", 200_000)
            min_via_dia = self.fab_limits.get("min_via_diameter_nm", 300_000)

            fence["via"]["drill_nm"] = self._clamp_min(
                "transmission_line.ground_via_fence.via.drill_nm",
                int(fence["via"]["drill_nm"]),
                min_drill,
                "Fence via drill {value}nm raised to fab minimum {limit}nm",
                "T0_FENCE_VIA_DRILL_MIN",
            )

            fence["via"]["diameter_nm"] = self._clamp_min(
                "transmission_line.ground_via_fence.via.diameter_nm",
                int(fence["via"]["diameter_nm"]),
                min_via_dia,
                "Fence via diameter {value}nm raised to fab minimum {limit}nm",
                "T0_FENCE_VIA_DIAMETER_MIN",
            )

    def repair_tier1(self, payload: dict[str, Any]) -> None:
        """Apply Tier 1 repairs: derived scalar constraints."""
        min_annular = self.fab_limits.get("min_annular_ring_nm", 100_000)

        # Signal via annular ring
        if payload.get("discontinuity") is not None:
            disc = payload["discontinuity"]
            signal_via = disc["signal_via"]

            # Ensure diameter > drill
            drill = int(signal_via["drill_nm"])
            diameter = int(signal_via["diameter_nm"])
            if diameter <= drill:
                new_diameter = drill + min_annular
                signal_via["diameter_nm"] = self._record(
                    "discontinuity.signal_via.diameter_nm",
                    diameter,
                    new_diameter,
                    f"Via diameter {diameter}nm must be > drill {drill}nm",
                    "T1_SIGNAL_VIA_DIAMETER_GT_DRILL",
                )
                diameter = new_diameter

            # Ensure pad >= diameter with annular ring
            pad = int(signal_via["pad_diameter_nm"])
            min_pad = diameter + 2 * min_annular
            if pad < min_pad:
                signal_via["pad_diameter_nm"] = self._record(
                    "discontinuity.signal_via.pad_diameter_nm",
                    pad,
                    min_pad,
                    f"Signal via pad {pad}nm raised for annular ring to {min_pad}nm",
                    "T1_SIGNAL_ANNULAR_MIN",
                )

            # Return via annular ring
            if disc.get("return_vias") is not None:
                return_via = disc["return_vias"]["via"]
                r_drill = int(return_via["drill_nm"])
                r_diameter = int(return_via["diameter_nm"])
                min_r_diameter = r_drill + 2 * min_annular
                if r_diameter < min_r_diameter:
                    return_via["diameter_nm"] = self._record(
                        "discontinuity.return_vias.via.diameter_nm",
                        r_diameter,
                        min_r_diameter,
                        f"Return via diameter {r_diameter}nm raised for annular ring to {min_r_diameter}nm",
                        "T1_RETURN_ANNULAR_MIN",
                    )

        # Ground fence via annular ring
        fence = payload["transmission_line"].get("ground_via_fence")
        if fence is not None and fence.get("enabled", False):
            f_drill = int(fence["via"]["drill_nm"])
            f_diameter = int(fence["via"]["diameter_nm"])
            min_f_diameter = f_drill + 2 * min_annular
            if f_diameter < min_f_diameter:
                fence["via"]["diameter_nm"] = self._record(
                    "transmission_line.ground_via_fence.via.diameter_nm",
                    f_diameter,
                    min_f_diameter,
                    f"Fence via diameter {f_diameter}nm raised for annular ring to {min_f_diameter}nm",
                    "T1_FENCE_ANNULAR_MIN",
                )

        # Trace length constraints
        tl = payload["transmission_line"]
        if int(tl["length_left_nm"]) < 1:
            tl["length_left_nm"] = self._record(
                "transmission_line.length_left_nm",
                int(tl["length_left_nm"]),
                1,
                "Left trace length must be positive",
                "T1_TRACE_LEFT_POSITIVE",
            )
        if int(tl["length_right_nm"]) < 1:
            tl["length_right_nm"] = self._record(
                "transmission_line.length_right_nm",
                int(tl["length_right_nm"]),
                1,
                "Right trace length must be positive",
                "T1_TRACE_RIGHT_POSITIVE",
            )

    def repair_tier2(self, payload: dict[str, Any]) -> None:
        """Apply Tier 2 repairs: analytic spatial constraints."""
        board_length = int(payload["board"]["outline"]["length_nm"])
        min_edge_clearance = self.fab_limits.get("min_edge_clearance_nm", 200_000)

        # Connector positions
        left_pos = payload["connectors"]["left"]["position_nm"]
        right_pos = payload["connectors"]["right"]["position_nm"]

        # Left connector X position
        left_x = int(left_pos[0])
        if left_x < min_edge_clearance:
            left_pos[0] = self._record(
                "connectors.left.position_nm[0]",
                left_x,
                min_edge_clearance,
                f"Left connector X {left_x}nm moved for edge clearance to {min_edge_clearance}nm",
                "T2_LEFT_CONNECTOR_X_MIN",
            )

        max_left_x = board_length - min_edge_clearance
        if left_x > max_left_x:
            left_pos[0] = self._record(
                "connectors.left.position_nm[0]",
                left_x,
                max_left_x,
                f"Left connector X {left_x}nm reduced to fit board at {max_left_x}nm",
                "T2_LEFT_CONNECTOR_X_MAX",
            )

        # Right connector X position
        right_x = int(right_pos[0])
        if right_x < min_edge_clearance:
            right_pos[0] = self._record(
                "connectors.right.position_nm[0]",
                right_x,
                min_edge_clearance,
                f"Right connector X {right_x}nm moved for edge clearance to {min_edge_clearance}nm",
                "T2_RIGHT_CONNECTOR_X_MIN",
            )

        max_right_x = board_length - min_edge_clearance
        if right_x > max_right_x:
            right_pos[0] = self._record(
                "connectors.right.position_nm[0]",
                right_x,
                max_right_x,
                f"Right connector X {right_x}nm reduced to fit board at {max_right_x}nm",
                "T2_RIGHT_CONNECTOR_X_MAX",
            )

        # Return via ring radius (ensure clearance from signal via)
        if payload.get("discontinuity") is not None:
            disc = payload["discontinuity"]
            if disc.get("return_vias") is not None:
                signal_pad = int(disc["signal_via"]["pad_diameter_nm"])
                return_via_dia = int(disc["return_vias"]["via"]["diameter_nm"])
                return_radius = int(disc["return_vias"]["radius_nm"])
                min_via_to_via = self.fab_limits.get("min_via_to_via_nm", 200_000)
                required_radius = signal_pad // 2 + return_via_dia // 2 + min_via_to_via

                if return_radius < required_radius:
                    disc["return_vias"]["radius_nm"] = self._record(
                        "discontinuity.return_vias.radius_nm",
                        return_radius,
                        required_radius,
                        f"Return via ring radius {return_radius}nm raised for signal via clearance to {required_radius}nm",
                        "T2_RETURN_VIA_RING_RADIUS",
                    )

        # Ground via fence spacing
        fence = payload["transmission_line"].get("ground_via_fence")
        if fence is not None and fence.get("enabled", False):
            fence_via_dia = int(fence["via"]["diameter_nm"])
            fence_offset = int(fence["offset_from_gap_nm"])
            fence_pitch = int(fence["pitch_nm"])
            min_via_to_via = self.fab_limits.get("min_via_to_via_nm", 200_000)

            # Fence via must not encroach on gap
            min_offset = fence_via_dia // 2
            if fence_offset < min_offset:
                fence["offset_from_gap_nm"] = self._record(
                    "transmission_line.ground_via_fence.offset_from_gap_nm",
                    fence_offset,
                    min_offset,
                    f"Fence via offset {fence_offset}nm raised for gap clearance to {min_offset}nm",
                    "T2_FENCE_VIA_GAP_CLEARANCE",
                )

            # Fence pitch must allow vias without overlap
            min_pitch = fence_via_dia + min_via_to_via
            if fence_pitch < min_pitch:
                fence["pitch_nm"] = self._record(
                    "transmission_line.ground_via_fence.pitch_nm",
                    fence_pitch,
                    min_pitch,
                    f"Fence pitch {fence_pitch}nm raised to prevent via overlap to {min_pitch}nm",
                    "T2_FENCE_PITCH_MIN",
                )

    def get_repair_result(
        self,
        original_proof: TieredConstraintProof,
        repaired_proof: TieredConstraintProof,
    ) -> RepairResult:
        """Build the final repair result from accumulated actions."""
        repair_map: dict[str, dict[str, int]] = {}
        repair_reason: list[str] = []

        for action in self.actions:
            repair_map[action.path] = {"before": action.before, "after": action.after}
            repair_reason.append(action.reason)

        repair_distance = _compute_repair_distance(self.actions)

        return RepairResult(
            repair_map=repair_map,
            repair_reason=repair_reason,
            repair_distance=repair_distance,
            repair_actions=tuple(self.actions),
            original_proof=original_proof,
            repaired_proof=repaired_proof,
        )


def repair_spec_tiered(
    spec: CouponSpec,
    fab_limits: dict[str, int],
) -> tuple[CouponSpec, RepairResult]:
    """Project an infeasible spec into feasible space using tiered repair.

    This function applies repairs in tier order (T0 -> T1 -> T2), ensuring
    that lower-tier repairs are applied before higher-tier ones. This is
    important because higher-tier constraints often depend on lower-tier
    parameter values.

    Args:
        spec: The CouponSpec to repair
        fab_limits: Dictionary of fab capability limits in nm

    Returns:
        Tuple of (repaired_spec, repair_result)
    """
    from ..spec import CouponSpec

    # Get original proof
    system = TieredConstraintSystem()
    original_proof = system.evaluate(spec, fab_limits)

    # If already valid, return as-is
    if original_proof.passed:
        return spec, RepairResult(
            repair_map={},
            repair_reason=[],
            repair_distance=0.0,
            repair_actions=(),
            original_proof=original_proof,
            repaired_proof=original_proof,
        )

    # Get mutable payload
    payload = spec.model_dump(mode="json")

    # Create repair engine and apply repairs
    engine = RepairEngine(fab_limits)
    engine.repair_tier0(payload)
    engine.repair_tier1(payload)
    engine.repair_tier2(payload)

    # Validate repaired spec
    repaired_spec = CouponSpec.model_validate(payload)
    repaired_proof = system.evaluate(repaired_spec, fab_limits)

    return repaired_spec, engine.get_repair_result(original_proof, repaired_proof)


def generate_constraint_proof(
    proof: TieredConstraintProof,
    repair_result: RepairResult | None = None,
) -> ConstraintProofDocument:
    """Generate a constraint proof document from a tiered proof.

    This creates the canonical constraint_proof.json format with
    per-constraint evaluations and signed margins.

    Args:
        proof: The tiered constraint proof to document
        repair_result: Optional repair result if REPAIR mode was used

    Returns:
        ConstraintProofDocument ready for serialization
    """
    constraints_list: list[dict[str, Any]] = []
    for c in proof.constraints:
        constraints_list.append({
            "id": c.constraint_id,
            "description": c.description,
            "tier": c.tier,
            "value": c.value,
            "limit": c.limit,
            "margin": c.margin,  # Signed margin
            "passed": c.passed,
            "reason": c.reason,
        })

    tiers_dict: dict[ConstraintTier, list[str]] = {}
    for tier in _TIERS:
        tier_constraints = proof.tiers.get(tier, ())
        tiers_dict[tier] = [c.constraint_id for c in tier_constraints]

    failed_count = len([c for c in proof.constraints if not c.passed])

    repair_info = None
    if repair_result is not None and repair_result.repair_actions:
        repair_info = repair_result.to_dict()

    return ConstraintProofDocument(
        schema_version=1,
        passed=proof.passed,
        first_failure_tier=proof.first_failure_tier,
        total_constraints=len(proof.constraints),
        failed_constraints=failed_count,
        tiers=tiers_dict,
        constraints=constraints_list,
        repair_applied=repair_result is not None and bool(repair_result.repair_actions),
        repair_info=repair_info,
    )


def write_constraint_proof(
    path: Path | str,
    proof: TieredConstraintProof,
    repair_result: RepairResult | None = None,
) -> None:
    """Write a constraint proof to a JSON file.

    Args:
        path: Path to write the constraint_proof.json file
        proof: The tiered constraint proof to document
        repair_result: Optional repair result if REPAIR mode was used
    """
    doc = generate_constraint_proof(proof, repair_result)
    doc.write_to_file(path)
