"""REPAIR mode implementation for constraint system.

This module implements REPAIR mode that projects infeasible specs into
feasible space with full auditability through repair_map, repair_reason,
and repair_distance.

It also provides constraint_proof.json and repair_map.json generation with
per-constraint evaluations and signed margins.

REQ-M1-010: REPAIR mode must project infeasible specs into feasible space
            with auditable repair_map, repair_reason list, and repair_distance.
REQ-M1-011: Every generated design must emit a constraint_proof.json with
            per-constraint evaluations and signed margins.

CP-3.4: Enhanced REPAIR mode with audit trail including:
        - repair_map.json with original/repaired design vectors
        - L2 and Linf distance metrics in normalized space
        - F1 continuity clamping (length_right >= 0)
        - Documented projection policy order

Projection Policy Order (documented per CP-3.4):
    1. T0: Parameter bounds (clamping to fab minimums/maximums)
    2. T1: Derived scalar constraints (annular ring, diameter relationships)
    3. T2: Spatial constraints (connector positions, via clearances)

    F1 Continuity: After applying all repairs, ensure length_right >= 0.
    This guarantees valid topology for F1_SINGLE_ENDED_VIA family.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from formula_foundry.substrate import canonical_json_dumps, sha256_bytes

from .tiers import (
    _TIERS,
    ConstraintTier,
    TieredConstraintProof,
    TieredConstraintSystem,
)

# Parameter normalization bounds for F1 family (used for distance metrics)
# These match the bounds in gpu_filter.py FamilyF1ParameterSpace
_PARAM_BOUNDS: dict[str, tuple[int, int]] = {
    "transmission_line.w_nm": (100_000, 500_000),
    "transmission_line.gap_nm": (100_000, 300_000),
    "transmission_line.length_left_nm": (5_000_000, 50_000_000),
    "transmission_line.length_right_nm": (0, 50_000_000),  # 0 for F1 continuity
    "board.outline.width_nm": (10_000_000, 50_000_000),
    "board.outline.length_nm": (30_000_000, 150_000_000),
    "board.outline.corner_radius_nm": (0, 5_000_000),
    "discontinuity.signal_via.drill_nm": (200_000, 500_000),
    "discontinuity.signal_via.diameter_nm": (300_000, 800_000),
    "discontinuity.signal_via.pad_diameter_nm": (400_000, 1_200_000),
    "discontinuity.return_vias.via.drill_nm": (200_000, 500_000),
    "discontinuity.return_vias.via.diameter_nm": (300_000, 800_000),
    "discontinuity.return_vias.radius_nm": (800_000, 3_000_000),
    "transmission_line.ground_via_fence.via.drill_nm": (200_000, 400_000),
    "transmission_line.ground_via_fence.via.diameter_nm": (300_000, 700_000),
    "transmission_line.ground_via_fence.pitch_nm": (500_000, 3_000_000),
    "transmission_line.ground_via_fence.offset_from_gap_nm": (200_000, 1_500_000),
    "connectors.left.position_nm[0]": (2_000_000, 10_000_000),
    "connectors.right.position_nm[0]": (70_000_000, 145_000_000),
}

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
class RepairDistanceMetrics:
    """Distance metrics for repair in normalized space (CP-3.4).

    Attributes:
        l2_distance: L2 (Euclidean) norm of repair vector in normalized space
        linf_distance: L-infinity (max) norm of repair vector in normalized space
        normalized_sum_distance: Sum of normalized relative changes (original metric)
    """

    l2_distance: float
    linf_distance: float
    normalized_sum_distance: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "l2_distance": self.l2_distance,
            "linf_distance": self.linf_distance,
            "normalized_sum_distance": self.normalized_sum_distance,
        }


@dataclass(frozen=True)
class DesignVector:
    """Design vector representation for repair audit trail (CP-3.4).

    Stores both raw parameter values and normalized [0,1] values
    for comparison and distance calculations.

    Attributes:
        parameters: Dict mapping parameter paths to values (in nm)
        normalized: Dict mapping parameter paths to normalized [0,1] values
    """

    parameters: dict[str, int]
    normalized: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "parameters": self.parameters,
            "normalized": self.normalized,
        }


@dataclass(frozen=True)
class RepairResult:
    """Result of a REPAIR mode projection.

    Attributes:
        repair_map: Mapping from parameter paths to {before, after} dicts
        repair_reason: List of human-readable repair explanations
        repair_distance: Normalized total distance of all repairs (legacy)
        repair_actions: Detailed list of all repair actions taken
        original_proof: Constraint proof before repair
        repaired_proof: Constraint proof after repair (should pass)
        original_vector: Original design vector (CP-3.4)
        repaired_vector: Repaired design vector (CP-3.4)
        distance_metrics: L2/Linf distance metrics in normalized space (CP-3.4)
        projection_policy_order: List of tiers in repair order (CP-3.4)
        original_spec_hash: SHA256 hash of canonical original spec (REQ-M1-011)
        repaired_spec_hash: SHA256 hash of canonical repaired spec (REQ-M1-011)
        repaired_design_hash: Design hash from repaired spec (REQ-M1-011)
    """

    repair_map: dict[str, dict[str, int]]
    repair_reason: list[str]
    repair_distance: float
    repair_actions: tuple[RepairAction, ...]
    original_proof: TieredConstraintProof
    repaired_proof: TieredConstraintProof
    # CP-3.4 additions
    original_vector: DesignVector | None = None
    repaired_vector: DesignVector | None = None
    distance_metrics: RepairDistanceMetrics | None = None
    projection_policy_order: tuple[str, ...] = ("T0", "T1", "T2", "F1_CONTINUITY")
    # REQ-M1-011: Hashes for rebuild verification
    original_spec_hash: str | None = None
    repaired_spec_hash: str | None = None
    repaired_design_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert repair result to a dictionary for serialization."""
        result: dict[str, Any] = {
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
            "projection_policy_order": list(self.projection_policy_order),
        }

        # CP-3.4: Include design vectors and distance metrics if available
        if self.original_vector is not None:
            result["original_vector"] = self.original_vector.to_dict()
        if self.repaired_vector is not None:
            result["repaired_vector"] = self.repaired_vector.to_dict()
        if self.distance_metrics is not None:
            result["distance_metrics"] = self.distance_metrics.to_dict()

        # REQ-M1-011: Include spec hashes for rebuild verification
        if self.original_spec_hash is not None:
            result["original_spec_hash"] = self.original_spec_hash
        if self.repaired_spec_hash is not None:
            result["repaired_spec_hash"] = self.repaired_spec_hash
        if self.repaired_design_hash is not None:
            result["repaired_design_hash"] = self.repaired_design_hash

        return result


@dataclass
class CategoryMarginSummary:
    """Summary of margin statistics for a constraint category (CP-3.2).

    Attributes:
        min_margin_nm: Minimum margin in nanometers across all constraints in this category
        min_margin_constraint_id: ID of the constraint with the minimum margin
        constraint_count: Total number of constraints in this category
        failed_count: Number of failed constraints in this category
        passed_count: Number of passed constraints in this category
        average_margin_nm: Average margin in nanometers across all constraints
    """

    min_margin_nm: int | None = None
    min_margin_constraint_id: str | None = None
    constraint_count: int = 0
    failed_count: int = 0
    passed_count: int = 0
    average_margin_nm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {}
        if self.min_margin_nm is not None:
            result["min_margin_nm"] = self.min_margin_nm
        if self.min_margin_constraint_id is not None:
            result["min_margin_constraint_id"] = self.min_margin_constraint_id
        result["constraint_count"] = self.constraint_count
        result["failed_count"] = self.failed_count
        result["passed_count"] = self.passed_count
        if self.average_margin_nm is not None:
            result["average_margin_nm"] = self.average_margin_nm
        return result


@dataclass
class FailingConstraintsSummary:
    """Summary of all failing constraints (CP-3.2).

    Attributes:
        total_failures: Total number of failing constraints
        must_pass_failures: Number of failing constraints where must_pass=true
        failures_by_tier: Count of failures grouped by tier
        failures_by_category: Count of failures grouped by category
        constraint_ids: List of all failing constraint IDs
        failure_details: Detailed information for each failing constraint
    """

    total_failures: int = 0
    must_pass_failures: int = 0
    failures_by_tier: dict[str, int] = field(default_factory=dict)
    failures_by_category: dict[str, int] = field(default_factory=dict)
    constraint_ids: list[str] = field(default_factory=list)
    failure_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_failures": self.total_failures,
            "must_pass_failures": self.must_pass_failures,
            "failures_by_tier": self.failures_by_tier,
            "failures_by_category": self.failures_by_category,
            "constraint_ids": self.constraint_ids,
            "failure_details": self.failure_details,
        }


@dataclass
class RepairSummary:
    """Summary of repairs applied when REPAIR mode was used (CP-3.2).

    Attributes:
        repair_applied: Whether any repairs were applied
        total_repairs: Total number of parameter repairs made
        repairs_by_tier: Count of repairs triggered by constraints in each tier
        repaired_parameter_paths: List of all parameter paths that were repaired
        total_distance_nm: Sum of absolute repair distances in nanometers
        max_single_repair_nm: Largest single repair distance in nanometers
        normalized_repair_distance: Normalized repair distance (L2 norm)
        original_failures: Number of constraint failures before repair
        remaining_failures: Number of constraint failures after repair
        projection_policy_order: Order of projection policies applied
    """

    repair_applied: bool = False
    total_repairs: int = 0
    repairs_by_tier: dict[str, int] = field(default_factory=dict)
    repaired_parameter_paths: list[str] = field(default_factory=list)
    total_distance_nm: int = 0
    max_single_repair_nm: int = 0
    normalized_repair_distance: float = 0.0
    original_failures: int = 0
    remaining_failures: int = 0
    projection_policy_order: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "repair_applied": self.repair_applied,
            "total_repairs": self.total_repairs,
            "repairs_by_tier": self.repairs_by_tier,
            "repaired_parameter_paths": self.repaired_parameter_paths,
            "total_distance_nm": self.total_distance_nm,
            "max_single_repair_nm": self.max_single_repair_nm,
            "normalized_repair_distance": self.normalized_repair_distance,
            "original_failures": self.original_failures,
            "remaining_failures": self.remaining_failures,
            "projection_policy_order": self.projection_policy_order,
        }


@dataclass
class ConstraintProofDocument:
    """A constraint proof document for JSON serialization (CP-3.2 M1 compliant).

    This is the canonical format for constraint_proof.json files,
    containing per-constraint evaluations with signed margins and
    summary sections per M1 requirements.

    Attributes:
        schema_version: Version of the constraint proof schema
        passed: Overall pass/fail status
        first_failure_tier: First tier with failures (None if passed)
        total_constraints: Total number of constraints evaluated
        failed_constraints: Number of failed constraints
        tiers: Constraint IDs grouped by tier
        constraints: Full per-constraint evaluation details
        min_margin_by_category: Minimum margin per constraint category (CP-3.2)
        failing_constraints_summary: Summary of all failing constraints (CP-3.2)
        repair_summary: Summary of repairs applied (CP-3.2)
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
    min_margin_by_category: dict[str, CategoryMarginSummary] = field(default_factory=dict)
    failing_constraints_summary: FailingConstraintsSummary | None = None
    repair_summary: RepairSummary | None = None
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
        # Add min_margin_by_category if populated (CP-3.2)
        if self.min_margin_by_category:
            result["min_margin_by_category"] = {
                cat: summary.to_dict() for cat, summary in self.min_margin_by_category.items()
            }
        # Add failing_constraints_summary if there are failures (CP-3.2)
        if self.failing_constraints_summary is not None:
            result["failing_constraints_summary"] = self.failing_constraints_summary.to_dict()
        # Add repair_summary if repairs were applied (CP-3.2)
        if self.repair_summary is not None:
            result["repair_summary"] = self.repair_summary.to_dict()
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


def _normalize_value(path: str, value: int) -> float:
    """Normalize a parameter value to [0, 1] range (CP-3.4).

    Args:
        path: Parameter path (e.g., "transmission_line.w_nm")
        value: Raw value in nm

    Returns:
        Normalized value in [0, 1]
    """
    bounds = _PARAM_BOUNDS.get(path)
    if bounds is None:
        # Unknown parameter, return as-is scaled by 1e-6
        return float(value) / 1_000_000.0

    min_val, max_val = bounds
    range_val = max_val - min_val
    if range_val <= 0:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / range_val))


def _compute_distance_metrics(repair_actions: list[RepairAction]) -> RepairDistanceMetrics:
    """Compute L2 and Linf distance metrics in normalized space (CP-3.4).

    Args:
        repair_actions: List of repair actions taken

    Returns:
        RepairDistanceMetrics with L2, Linf, and normalized sum distances
    """
    if not repair_actions:
        return RepairDistanceMetrics(
            l2_distance=0.0,
            linf_distance=0.0,
            normalized_sum_distance=0.0,
        )

    # Compute normalized differences for each action
    normalized_diffs: list[float] = []
    for action in repair_actions:
        # Normalize before and after values
        norm_before = _normalize_value(action.path, action.before)
        norm_after = _normalize_value(action.path, action.after)
        normalized_diffs.append(abs(norm_after - norm_before))

    # L2 (Euclidean) distance
    l2_distance = math.sqrt(sum(d * d for d in normalized_diffs))

    # Linf (max) distance
    linf_distance = max(normalized_diffs) if normalized_diffs else 0.0

    # Normalized sum (original metric)
    normalized_sum = _compute_repair_distance(repair_actions)

    return RepairDistanceMetrics(
        l2_distance=l2_distance,
        linf_distance=linf_distance,
        normalized_sum_distance=normalized_sum,
    )


def _extract_design_vector(payload: dict[str, Any]) -> DesignVector:
    """Extract design vector from spec payload (CP-3.4).

    Args:
        payload: Spec payload dictionary

    Returns:
        DesignVector with parameters and normalized values
    """
    parameters: dict[str, int] = {}
    normalized: dict[str, float] = {}

    # Extract transmission line parameters
    tl = payload.get("transmission_line", {})
    if tl:
        for key in ["w_nm", "gap_nm", "length_left_nm", "length_right_nm"]:
            if key in tl:
                path = f"transmission_line.{key}"
                val = int(tl[key])
                parameters[path] = val
                normalized[path] = _normalize_value(path, val)

        # Ground via fence parameters
        fence = tl.get("ground_via_fence")
        if fence and fence.get("enabled", False):
            for key in ["pitch_nm", "offset_from_gap_nm"]:
                if key in fence:
                    path = f"transmission_line.ground_via_fence.{key}"
                    val = int(fence[key])
                    parameters[path] = val
                    normalized[path] = _normalize_value(path, val)
            via = fence.get("via", {})
            for key in ["drill_nm", "diameter_nm"]:
                if key in via:
                    path = f"transmission_line.ground_via_fence.via.{key}"
                    val = int(via[key])
                    parameters[path] = val
                    normalized[path] = _normalize_value(path, val)

    # Extract board outline parameters
    board = payload.get("board", {}).get("outline", {})
    if board:
        for key in ["width_nm", "length_nm", "corner_radius_nm"]:
            if key in board:
                path = f"board.outline.{key}"
                val = int(board[key])
                parameters[path] = val
                normalized[path] = _normalize_value(path, val)

    # Extract discontinuity parameters
    disc = payload.get("discontinuity")
    if disc:
        signal_via = disc.get("signal_via", {})
        for key in ["drill_nm", "diameter_nm", "pad_diameter_nm"]:
            if key in signal_via:
                path = f"discontinuity.signal_via.{key}"
                val = int(signal_via[key])
                parameters[path] = val
                normalized[path] = _normalize_value(path, val)

        return_vias = disc.get("return_vias")
        if return_vias:
            if "radius_nm" in return_vias:
                path = "discontinuity.return_vias.radius_nm"
                val = int(return_vias["radius_nm"])
                parameters[path] = val
                normalized[path] = _normalize_value(path, val)
            via = return_vias.get("via", {})
            for key in ["drill_nm", "diameter_nm"]:
                if key in via:
                    path = f"discontinuity.return_vias.via.{key}"
                    val = int(via[key])
                    parameters[path] = val
                    normalized[path] = _normalize_value(path, val)

    # Extract connector positions
    connectors = payload.get("connectors", {})
    for side in ["left", "right"]:
        conn = connectors.get(side, {})
        pos = conn.get("position_nm")
        if pos and len(pos) >= 1:
            path = f"connectors.{side}.position_nm[0]"
            val = int(pos[0])
            parameters[path] = val
            normalized[path] = _normalize_value(path, val)

    return DesignVector(parameters=parameters, normalized=normalized)


def _build_repair_map(
    actions: list[RepairAction],
    original_vector: DesignVector | None,
    repaired_vector: DesignVector | None,
) -> dict[str, dict[str, int]]:
    """Build a deterministic repair map from original/repaired vectors when possible."""
    merged: dict[str, dict[str, int]] = {}

    if original_vector is not None and repaired_vector is not None:
        original_params = original_vector.parameters
        repaired_params = repaired_vector.parameters
        for path in sorted(set(original_params) | set(repaired_params)):
            before = original_params.get(path)
            after = repaired_params.get(path)
            if before is None or after is None:
                continue
            if before != after:
                merged[path] = {"before": before, "after": after}

    if actions:
        action_map: dict[str, dict[str, int]] = {}
        for action in actions:
            entry = action_map.get(action.path)
            if entry is None:
                action_map[action.path] = {"before": action.before, "after": action.after}
            else:
                entry["after"] = action.after
        for path, entry in action_map.items():
            if path not in merged:
                merged[path] = entry

    return {path: merged[path] for path in sorted(merged)}


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

    def _record(self, path: str, before: int, after: int, reason: str, constraint_id: str) -> int:
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

        # Fence pitch constraint (Section 13.3.2)
        if fence is not None and fence.get("enabled", False):
            fence_pitch = int(fence["pitch_nm"])
            fence_via_dia = int(fence["via"]["diameter_nm"])
            min_via_to_via = self.fab_limits.get("min_via_to_via_nm", 200_000)
            min_fence_pitch = fence_via_dia + min_via_to_via

            if fence_pitch < min_fence_pitch:
                fence["pitch_nm"] = self._record(
                    "transmission_line.ground_via_fence.pitch_nm",
                    fence_pitch,
                    min_fence_pitch,
                    f"Fence pitch {fence_pitch}nm raised for via spacing to {min_fence_pitch}nm",
                    "T1_FENCE_PITCH_MIN",
                )

        # Copper-to-edge clearance (Section 13.3.2)
        # If copper is too close to edge, we can widen the board
        if payload.get("board") is not None and payload["board"].get("outline") is not None:
            board = payload["board"]["outline"]
            board_width = int(board["width_nm"])
            trace_width = int(tl["w_nm"])
            trace_gap = int(tl["gap_nm"])
            min_edge_clearance = self.fab_limits.get("min_edge_clearance_nm", 200_000)

            # Calculate copper extent from centerline
            trace_extent_from_center = trace_width // 2 + trace_gap
            if fence is not None and fence.get("enabled", False):
                fence_offset = int(fence["offset_from_gap_nm"])
                fence_via_radius = int(fence["via"]["diameter_nm"]) // 2
                trace_extent_from_center = trace_width // 2 + trace_gap + fence_offset + fence_via_radius

            # Required board half-width = trace extent + edge clearance
            required_half_width = trace_extent_from_center + min_edge_clearance
            required_board_width = required_half_width * 2

            if board_width < required_board_width:
                board["width_nm"] = self._record(
                    "board.outline.width_nm",
                    board_width,
                    required_board_width,
                    f"Board width {board_width}nm raised for copper-to-edge clearance to {required_board_width}nm",
                    "T1_COPPER_TO_EDGE_CLEARANCE",
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

        # Discontinuity copper-to-edge clearance (X direction)
        # Ensure board length accommodates signal/return vias near the right edge.
        disc = payload.get("discontinuity")
        connectors = payload.get("connectors", {})
        left_conn = connectors.get("left", {})
        left_pos = left_conn.get("position_nm", [])
        tl = payload.get("transmission_line", {})
        if disc is not None and len(left_pos) >= 1 and "length_left_nm" in tl:
            x_disc = int(left_pos[0]) + int(tl.get("length_left_nm", 0))
            signal_pad_radius = int(disc["signal_via"]["pad_diameter_nm"]) // 2
            required_length = x_disc + signal_pad_radius + min_edge_clearance
            constraint_id = "T3_VIA_COPPER_TO_RIGHT_EDGE"

            if disc.get("return_vias") is not None:
                return_ring_radius = int(disc["return_vias"]["radius_nm"])
                return_via_radius = int(disc["return_vias"]["via"]["diameter_nm"]) // 2
                return_extent = return_ring_radius + return_via_radius
                required_length = max(required_length, x_disc + return_extent + min_edge_clearance)
                constraint_id = "T3_RETURN_VIA_COPPER_TO_EDGE_X"

            if board_length < required_length:
                payload["board"]["outline"]["length_nm"] = self._record(
                    "board.outline.length_nm",
                    board_length,
                    required_length,
                    f"Board length {board_length}nm extended to {required_length}nm for discontinuity edge clearance",
                    constraint_id,
                )
                board_length = required_length

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

    def repair_f1_continuity(self, payload: dict[str, Any]) -> None:
        """Apply F1 continuity repair: derive length_right from continuity (CP-3.3/CP-3.4).

        For the F1_SINGLE_ENDED_VIA family, the right trace length must be
        derived from the continuity equation to ensure valid topology.

        The F1 topology constraint is:
            x_discontinuity_center = x_left_connector + length_left
            x_discontinuity_center = x_right_connector - length_right

        Therefore:
            length_right_derived = x_right_connector - (x_left_connector + length_left)

        This repair step computes the derived length_right and sets it
        to enforce F1 continuity. If the derived value is negative,
        it clamps to 0 to maintain valid topology.

        Args:
            payload: Spec payload to repair
        """
        tl = payload.get("transmission_line")
        if tl is None:
            return

        # Check if this is F1 family (has discontinuity with VIA_TRANSITION)
        disc = payload.get("discontinuity")
        if disc is None or disc.get("type") != "VIA_TRANSITION":
            return

        # Get connector positions and length_left
        connectors = payload.get("connectors", {})
        left_conn = connectors.get("left", {})
        right_conn = connectors.get("right", {})
        left_pos = left_conn.get("position_nm", [])
        right_pos = right_conn.get("position_nm", [])

        if len(left_pos) < 1 or len(right_pos) < 1:
            return

        left_x = int(left_pos[0])
        right_x = int(right_pos[0])
        length_left = int(tl.get("length_left_nm", 0))
        length_right_specified = int(tl.get("length_right_nm", 0))

        # Compute derived length_right from F1 continuity equation
        # length_right_derived = right_x - (left_x + length_left)
        x_discontinuity = left_x + length_left
        length_right_derived = right_x - x_discontinuity

        # If derived length_right is < 1, we need to reduce length_left to make room
        # T1 constraint requires trace length >= 1, so we need length_right >= 1
        # This means: right_x - (left_x + length_left) >= 1
        #            => length_left <= right_x - left_x - 1
        if length_right_derived < 1:
            # Maximum allowed length_left to ensure length_right >= 1
            max_length_left = right_x - left_x - 1
            if max_length_left < 1:
                # Both connectors are too close - use minimum lengths for both
                max_length_left = 1

            if length_left > max_length_left:
                tl["length_left_nm"] = self._record(
                    "transmission_line.length_left_nm",
                    length_left,
                    max_length_left,
                    f"Left trace length reduced from {length_left}nm to {max_length_left}nm "
                    f"to ensure F1 continuity (derived length_right was {length_right_derived}nm)",
                    "F1_CONTINUITY_LENGTH_LEFT",
                )
                length_left = max_length_left

            # Recompute derived length_right with updated length_left
            x_discontinuity = left_x + length_left
            length_right_derived = right_x - x_discontinuity

            # Final clamp to ensure minimum value (should be >= 1 now)
            if length_right_derived < 1:
                length_right_derived = 1

        # Record the length_right repair if the value changed
        if length_right_specified != length_right_derived:
            tl["length_right_nm"] = self._record(
                "transmission_line.length_right_nm",
                length_right_specified,
                length_right_derived,
                f"Right trace length set to {length_right_derived}nm for F1 continuity "
                f"(derived from right_x={right_x} - (left_x={left_x} + length_left={length_left}))",
                "F1_CONTINUITY_LENGTH_RIGHT",
            )

    def get_repair_result(
        self,
        original_proof: TieredConstraintProof,
        repaired_proof: TieredConstraintProof,
        original_vector: DesignVector | None = None,
        repaired_vector: DesignVector | None = None,
        *,
        original_spec_hash: str | None = None,
        repaired_spec_hash: str | None = None,
        repaired_design_hash: str | None = None,
    ) -> RepairResult:
        """Build the final repair result from accumulated actions (CP-3.4/REQ-M1-011).

        Args:
            original_proof: Constraint proof before repair
            repaired_proof: Constraint proof after repair
            original_vector: Original design vector (CP-3.4)
            repaired_vector: Repaired design vector (CP-3.4)
            original_spec_hash: SHA256 of canonical original spec (REQ-M1-011)
            repaired_spec_hash: SHA256 of canonical repaired spec (REQ-M1-011)
            repaired_design_hash: Design hash from repaired spec (REQ-M1-011)

        Returns:
            RepairResult with full audit trail including hashes for rebuild verification
        """
        repair_map = _build_repair_map(self.actions, original_vector, repaired_vector)
        repair_reason = [action.reason for action in self.actions]

        repair_distance = _compute_repair_distance(self.actions)

        # CP-3.4: Compute L2/Linf distance metrics
        distance_metrics = _compute_distance_metrics(self.actions)

        return RepairResult(
            repair_map=repair_map,
            repair_reason=repair_reason,
            repair_distance=repair_distance,
            repair_actions=tuple(self.actions),
            original_proof=original_proof,
            repaired_proof=repaired_proof,
            original_vector=original_vector,
            repaired_vector=repaired_vector,
            distance_metrics=distance_metrics,
            projection_policy_order=("T0", "T1", "T2", "F1_CONTINUITY"),
            original_spec_hash=original_spec_hash,
            repaired_spec_hash=repaired_spec_hash,
            repaired_design_hash=repaired_design_hash,
        )


def repair_spec_tiered(
    spec: CouponSpec,
    fab_limits: dict[str, int],
) -> tuple[CouponSpec, RepairResult]:
    """Project an infeasible spec into feasible space using tiered repair.

    This function applies repairs in tier order (T0 -> T1 -> T2 -> F1_CONTINUITY),
    ensuring that lower-tier repairs are applied before higher-tier ones. This is
    important because higher-tier constraints often depend on lower-tier
    parameter values.

    Projection Policy Order (CP-3.4):
        1. T0: Parameter bounds (clamping to fab minimums/maximums)
        2. T1: Derived scalar constraints (annular ring, diameter relationships)
        3. T2: Spatial constraints (connector positions, via clearances)
        4. F1_CONTINUITY: Ensure length_right >= 0 for valid F1 topology

    Args:
        spec: The CouponSpec to repair
        fab_limits: Dictionary of fab capability limits in nm

    Returns:
        Tuple of (repaired_spec, repair_result) with CP-3.4/REQ-M1-011 audit trail:
        - original_vector: Design vector before repair
        - repaired_vector: Design vector after repair
        - distance_metrics: L2/Linf distances in normalized space
        - projection_policy_order: Order of repair tiers applied
        - original_spec_hash: SHA256 of canonical original spec (REQ-M1-011)
        - repaired_spec_hash: SHA256 of canonical repaired spec (REQ-M1-011)
        - repaired_design_hash: Design hash for rebuild verification (REQ-M1-011)
    """
    from ..resolve import design_hash, resolve
    from ..spec import CouponSpec

    # Get original proof
    system = TieredConstraintSystem()
    original_proof = system.evaluate(spec, fab_limits)

    # Get mutable payload and extract original design vector (CP-3.4)
    payload = spec.model_dump(mode="json")
    original_vector = _extract_design_vector(payload)

    # Compute original spec hash (REQ-M1-011)
    original_spec_hash = sha256_bytes(canonical_json_dumps(payload).encode("utf-8"))

    # If already valid, return as-is with empty metrics and matching hashes
    if original_proof.passed:
        resolved = resolve(spec)
        spec_design_hash = design_hash(resolved)
        return spec, RepairResult(
            repair_map={},
            repair_reason=[],
            repair_distance=0.0,
            repair_actions=(),
            original_proof=original_proof,
            repaired_proof=original_proof,
            original_vector=original_vector,
            repaired_vector=original_vector,  # No change
            distance_metrics=RepairDistanceMetrics(
                l2_distance=0.0,
                linf_distance=0.0,
                normalized_sum_distance=0.0,
            ),
            projection_policy_order=("T0", "T1", "T2", "F1_CONTINUITY"),
            original_spec_hash=original_spec_hash,
            repaired_spec_hash=original_spec_hash,  # Same as original - no repairs
            repaired_design_hash=spec_design_hash,
        )

    # Create repair engine and apply repairs in policy order
    engine = RepairEngine(fab_limits)

    # T0: Parameter bounds
    engine.repair_tier0(payload)

    # T1: Derived scalar constraints
    engine.repair_tier1(payload)

    # T2: Spatial constraints
    engine.repair_tier2(payload)

    # F1_CONTINUITY: Ensure length_right >= 0 for F1 topology (CP-3.4)
    engine.repair_f1_continuity(payload)

    # Extract repaired design vector (CP-3.4)
    repaired_vector = _extract_design_vector(payload)

    # Validate repaired spec
    repaired_spec = CouponSpec.model_validate(payload)
    repaired_proof = system.evaluate(repaired_spec, fab_limits)

    # Compute repaired spec hash and design hash (REQ-M1-011)
    repaired_payload = repaired_spec.model_dump(mode="json")
    repaired_spec_hash = sha256_bytes(canonical_json_dumps(repaired_payload).encode("utf-8"))
    repaired_resolved = resolve(repaired_spec)
    repaired_design_hash = design_hash(repaired_resolved)

    return repaired_spec, engine.get_repair_result(
        original_proof,
        repaired_proof,
        original_vector=original_vector,
        repaired_vector=repaired_vector,
        original_spec_hash=original_spec_hash,
        repaired_spec_hash=repaired_spec_hash,
        repaired_design_hash=repaired_design_hash,
    )


def _infer_category_from_constraint_id(constraint_id: str) -> str:
    """Infer constraint category from constraint ID (CP-3.2).

    Maps constraint IDs to categories based on naming conventions.

    Args:
        constraint_id: Constraint ID like "T0_TRACE_WIDTH_MIN"

    Returns:
        Category string like "FABRICATION"
    """
    constraint_id_upper = constraint_id.upper()

    # TOPOLOGY constraints
    if any(kw in constraint_id_upper for kw in [
        "CONNECT", "SIG", "TOPOLOGY", "CONTINUITY", "NET"
    ]):
        return "TOPOLOGY"

    # SPACING constraints
    if any(kw in constraint_id_upper for kw in [
        "CLEARANCE", "SPACING", "GAP", "EDGE", "PITCH"
    ]):
        return "SPACING"

    # GEOMETRY constraints
    if any(kw in constraint_id_upper for kw in [
        "OVERLAP", "COLLISION", "SYMMETRY", "FITS", "COVERAGE", "RADIUS", "ASPECT"
    ]):
        return "GEOMETRY"

    # ELECTRICAL constraints
    if any(kw in constraint_id_upper for kw in [
        "IMPEDANCE", "CURRENT", "POWER", "RESISTANCE"
    ]):
        return "ELECTRICAL"

    # MATERIAL constraints
    if any(kw in constraint_id_upper for kw in [
        "MATERIAL", "ER", "LOSS", "DIELECTRIC"
    ]):
        return "MATERIAL"

    # Default to FABRICATION for parameter bounds
    return "FABRICATION"


def _infer_severity_from_constraint(constraint_id: str, passed: bool) -> str:
    """Infer severity level from constraint (CP-3.2).

    Args:
        constraint_id: Constraint ID
        passed: Whether constraint passed

    Returns:
        Severity string: "ERROR", "WARNING", or "INFO"
    """
    # Tier 0 constraints are always ERROR
    if constraint_id.startswith("T0_"):
        return "ERROR"
    # Tier 1 constraints are ERROR for critical ones
    if constraint_id.startswith("T1_"):
        return "ERROR"
    # Tier 2-3 constraints are ERROR for safety-critical ones
    if constraint_id.startswith(("T2_", "T3_")):
        return "ERROR"
    # Default to ERROR
    return "ERROR"


def generate_constraint_proof(
    proof: TieredConstraintProof,
    repair_result: RepairResult | None = None,
) -> ConstraintProofDocument:
    """Generate a constraint proof document from a tiered proof (CP-3.2 M1 compliant).

    This creates the canonical constraint_proof.json format with
    per-constraint evaluations and signed margins, plus summary sections
    required by M1:
    - min_margin_by_category: Minimum margin per constraint category
    - failing_constraints_summary: Summary of all failing constraints
    - repair_summary: Summary of repairs applied (if REPAIR mode used)

    Args:
        proof: The tiered constraint proof to document
        repair_result: Optional repair result if REPAIR mode was used

    Returns:
        ConstraintProofDocument ready for serialization
    """
    # Build constraint entries with all M1-required fields
    constraints_list: list[dict[str, Any]] = []

    # Track statistics for summaries
    category_stats: dict[str, dict[str, Any]] = {}  # category -> {margins, constraints, failures}
    failed_constraints: list[dict[str, Any]] = []

    for c in proof.constraints:
        # Infer category from constraint ID if not available
        category = _infer_category_from_constraint_id(c.constraint_id)
        severity = _infer_severity_from_constraint(c.constraint_id, c.passed)
        # Most constraints should pass for validity (must_pass=True)
        must_pass = severity == "ERROR"

        # Convert margin to integer nanometers (CP-3.2 M1 requirement)
        margin_nm = int(round(c.margin))

        constraint_entry: dict[str, Any] = {
            "id": c.constraint_id,
            "description": c.description,
            "tier": c.tier,
            "category": category,
            "value": c.value,
            "limit": c.limit,
            "margin": c.margin,  # Signed margin (float)
            "margin_nm": margin_nm,  # Signed margin in nm (CP-3.2 M1)
            "passed": c.passed,
            "severity": severity,
            "must_pass": must_pass,
            "reason": c.reason if c.reason else "",  # Always include reason
        }

        constraints_list.append(constraint_entry)

        # Update category statistics
        if category not in category_stats:
            category_stats[category] = {
                "margins": [],
                "constraint_ids": [],
                "failed_count": 0,
                "passed_count": 0,
            }

        category_stats[category]["margins"].append((margin_nm, c.constraint_id))
        category_stats[category]["constraint_ids"].append(c.constraint_id)
        if c.passed:
            category_stats[category]["passed_count"] += 1
        else:
            category_stats[category]["failed_count"] += 1

            # Track failure details
            failed_constraints.append({
                "id": c.constraint_id,
                "tier": c.tier,
                "category": category,
                "margin_nm": margin_nm,
                "severity": severity,
                "must_pass": must_pass,
                "reason": c.reason if c.reason else f"Margin {margin_nm}nm < 0",
            })

    # Build min_margin_by_category summary (CP-3.2)
    min_margin_by_category: dict[str, CategoryMarginSummary] = {}
    for category, stats in category_stats.items():
        margins = stats["margins"]
        if margins:
            min_margin, min_constraint_id = min(margins, key=lambda x: x[0])
            avg_margin = sum(m[0] for m in margins) / len(margins)

            min_margin_by_category[category] = CategoryMarginSummary(
                min_margin_nm=min_margin,
                min_margin_constraint_id=min_constraint_id,
                constraint_count=len(margins),
                failed_count=stats["failed_count"],
                passed_count=stats["passed_count"],
                average_margin_nm=avg_margin,
            )

    # Build tiers dict
    tiers_dict: dict[ConstraintTier, list[str]] = {}
    for tier in _TIERS:
        tier_constraints = proof.tiers.get(tier, ())
        tiers_dict[tier] = [c.constraint_id for c in tier_constraints]

    failed_count = len(failed_constraints)

    # Build failing_constraints_summary (CP-3.2)
    failing_constraints_summary = None
    if failed_count > 0:
        failures_by_tier: dict[str, int] = {}
        failures_by_category: dict[str, int] = {}
        must_pass_failures = 0

        for fc in failed_constraints:
            tier = fc["tier"]
            cat = fc["category"]
            failures_by_tier[tier] = failures_by_tier.get(tier, 0) + 1
            failures_by_category[cat] = failures_by_category.get(cat, 0) + 1
            if fc["must_pass"]:
                must_pass_failures += 1

        failing_constraints_summary = FailingConstraintsSummary(
            total_failures=failed_count,
            must_pass_failures=must_pass_failures,
            failures_by_tier=failures_by_tier,
            failures_by_category=failures_by_category,
            constraint_ids=[fc["id"] for fc in failed_constraints],
            failure_details=failed_constraints,
        )

    # Build repair_summary (CP-3.2)
    repair_summary = None
    repair_info = None
    if repair_result is not None and repair_result.repair_actions:
        repair_info = repair_result.to_dict()

        # Calculate repair statistics
        total_distance_nm = sum(abs(a.after - a.before) for a in repair_result.repair_actions)
        max_single_repair_nm = max(
            abs(a.after - a.before) for a in repair_result.repair_actions
        ) if repair_result.repair_actions else 0

        # Group repairs by tier (from constraint_id)
        repairs_by_tier: dict[str, int] = {}
        for action in repair_result.repair_actions:
            tier = action.constraint_id.split("_")[0] if "_" in action.constraint_id else "T0"
            repairs_by_tier[tier] = repairs_by_tier.get(tier, 0) + 1

        # Get original and remaining failure counts
        original_failures = len([c for c in repair_result.original_proof.constraints if not c.passed])
        remaining_failures = len([c for c in repair_result.repaired_proof.constraints if not c.passed])

        repair_summary = RepairSummary(
            repair_applied=True,
            total_repairs=len(repair_result.repair_actions),
            repairs_by_tier=repairs_by_tier,
            repaired_parameter_paths=list(repair_result.repair_map.keys()),
            total_distance_nm=total_distance_nm,
            max_single_repair_nm=max_single_repair_nm,
            normalized_repair_distance=(
                repair_result.distance_metrics.l2_distance
                if repair_result.distance_metrics else repair_result.repair_distance
            ),
            original_failures=original_failures,
            remaining_failures=remaining_failures,
            projection_policy_order=list(repair_result.projection_policy_order),
        )

    return ConstraintProofDocument(
        schema_version=1,
        passed=proof.passed,
        first_failure_tier=proof.first_failure_tier,
        total_constraints=len(proof.constraints),
        failed_constraints=failed_count,
        tiers=tiers_dict,
        constraints=constraints_list,
        min_margin_by_category=min_margin_by_category,
        failing_constraints_summary=failing_constraints_summary,
        repair_summary=repair_summary,
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


def write_repair_map(
    path: Path | str,
    repair_result: RepairResult,
) -> None:
    """Write a repair_map.json file with full audit trail (CP-3.4/REQ-M1-011).

    The repair_map.json contains:
    - original_vector: Design vector before repair
    - repaired_vector: Design vector after repair
    - repair_actions: Per-constraint repair actions
    - distance_metrics: L2, Linf distances in normalized space
    - projection_policy_order: Order of repair tiers applied
    - original_spec_hash: SHA256 hash of canonical original spec (REQ-M1-011)
    - repaired_spec_hash: SHA256 hash of canonical repaired spec (REQ-M1-011)
    - repaired_design_hash: Design hash from repaired spec (REQ-M1-011)

    REQ-M1-011: The repair_map.json is written in canonical JSON format to ensure
    deterministic serialization. The repaired_spec_hash and repaired_design_hash
    enable verification that rebuilding from repaired_spec.json reproduces the
    same design_hash and artifacts.

    Args:
        path: Path to write the repair_map.json file
        repair_result: The repair result with audit trail
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    content = repair_result.to_dict()
    # Use canonical JSON for deterministic serialization (REQ-M1-011)
    path.write_text(canonical_json_dumps(content) + "\n", encoding="utf-8")
