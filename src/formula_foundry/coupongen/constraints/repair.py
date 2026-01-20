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

        return result


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
        """Apply F1 continuity repair: ensure length_right >= 0 (CP-3.4).

        For the F1_SINGLE_ENDED_VIA family, the right trace length must be
        non-negative to ensure valid topology. This is a post-repair step
        that clamps length_right to 0 if it would otherwise be negative.

        The F1 topology constraint is:
            x_discontinuity_center = x_left_connector + length_left
            x_discontinuity_center = x_right_connector - length_right

        If length_right would be negative (due to connector positions or
        length_left being too large), we clamp it to 0 to maintain valid
        topology.

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

        length_right = int(tl.get("length_right_nm", 0))
        if length_right < 0:
            tl["length_right_nm"] = self._record(
                "transmission_line.length_right_nm",
                length_right,
                0,
                f"Right trace length {length_right}nm clamped to 0 for F1 continuity",
                "F1_CONTINUITY_LENGTH_RIGHT",
            )

    def get_repair_result(
        self,
        original_proof: TieredConstraintProof,
        repaired_proof: TieredConstraintProof,
        original_vector: DesignVector | None = None,
        repaired_vector: DesignVector | None = None,
    ) -> RepairResult:
        """Build the final repair result from accumulated actions (CP-3.4 enhanced).

        Args:
            original_proof: Constraint proof before repair
            repaired_proof: Constraint proof after repair
            original_vector: Original design vector (CP-3.4)
            repaired_vector: Repaired design vector (CP-3.4)

        Returns:
            RepairResult with full audit trail
        """
        repair_map: dict[str, dict[str, int]] = {}
        repair_reason: list[str] = []

        for action in self.actions:
            repair_map[action.path] = {"before": action.before, "after": action.after}
            repair_reason.append(action.reason)

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
        Tuple of (repaired_spec, repair_result) with CP-3.4 audit trail:
        - original_vector: Design vector before repair
        - repaired_vector: Design vector after repair
        - distance_metrics: L2/Linf distances in normalized space
        - projection_policy_order: Order of repair tiers applied
    """
    from ..spec import CouponSpec

    # Get original proof
    system = TieredConstraintSystem()
    original_proof = system.evaluate(spec, fab_limits)

    # Get mutable payload and extract original design vector (CP-3.4)
    payload = spec.model_dump(mode="json")
    original_vector = _extract_design_vector(payload)

    # If already valid, return as-is with empty metrics
    if original_proof.passed:
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

    return repaired_spec, engine.get_repair_result(
        original_proof,
        repaired_proof,
        original_vector=original_vector,
        repaired_vector=repaired_vector,
    )


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
        constraints_list.append(
            {
                "id": c.constraint_id,
                "description": c.description,
                "tier": c.tier,
                "value": c.value,
                "limit": c.limit,
                "margin": c.margin,  # Signed margin
                "passed": c.passed,
                "reason": c.reason,
            }
        )

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


def write_repair_map(
    path: Path | str,
    repair_result: RepairResult,
) -> None:
    """Write a repair_map.json file with full audit trail (CP-3.4).

    The repair_map.json contains:
    - original_vector: Design vector before repair
    - repaired_vector: Design vector after repair
    - repair_actions: Per-constraint repair actions
    - distance_metrics: L2, Linf distances in normalized space
    - projection_policy_order: Order of repair tiers applied

    Args:
        path: Path to write the repair_map.json file
        repair_result: The repair result with audit trail
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    content = repair_result.to_dict()
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")
