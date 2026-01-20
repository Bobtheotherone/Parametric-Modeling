from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..fab_profiles import FabCapabilityProfile, get_fab_limits, load_fab_profile
from ..resolve import ResolvedDesign, resolve
from ..spec import CouponSpec

ConstraintTier = Literal["T0", "T1", "T2", "T3", "T4"]

_TIERS: tuple[ConstraintTier, ...] = ("T0", "T1", "T2", "T3", "T4")


@dataclass(frozen=True)
class ConstraintResult:
    constraint_id: str
    description: str
    tier: ConstraintTier
    value: float
    limit: float
    margin: float
    passed: bool


@dataclass(frozen=True)
class ConstraintProof:
    constraints: tuple[ConstraintResult, ...]
    tiers: dict[ConstraintTier, tuple[ConstraintResult, ...]]
    passed: bool


@dataclass(frozen=True)
class RepairInfo:
    repair_map: dict[str, dict[str, int]]
    repair_reason: list[str]
    repair_distance: float


@dataclass(frozen=True)
class ConstraintEvaluation:
    spec: CouponSpec
    resolved: ResolvedDesign
    proof: ConstraintProof
    repair_info: RepairInfo | None


class ConstraintViolation(ValueError):
    def __init__(self, violations: list[ConstraintResult]) -> None:
        self.violations = tuple(violations)
        message = "Constraint violations: " + ", ".join(result.constraint_id for result in violations)
        super().__init__(message)


def evaluate_constraints(spec: CouponSpec) -> ConstraintProof:
    limits = resolve_fab_limits(spec)
    results: list[ConstraintResult] = []

    results.append(
        _min_constraint(
            "T0_TRACE_WIDTH_MIN",
            "Trace width must exceed fab minimum.",
            tier="T0",
            value=int(spec.transmission_line.w_nm),
            limit=limits["min_trace_width_nm"],
        )
    )
    results.append(
        _min_constraint(
            "T0_TRACE_GAP_MIN",
            "CPWG gap must exceed fab minimum.",
            tier="T0",
            value=int(spec.transmission_line.gap_nm),
            limit=limits["min_gap_nm"],
        )
    )

    if spec.discontinuity is not None:
        signal_via = spec.discontinuity.signal_via
        results.append(
            _min_constraint(
                "T0_SIGNAL_DRILL_MIN",
                "Signal via drill must exceed fab minimum.",
                tier="T0",
                value=int(signal_via.drill_nm),
                limit=limits["min_drill_nm"],
            )
        )
        annular_ring = int(signal_via.pad_diameter_nm) - int(signal_via.drill_nm)
        results.append(
            _min_constraint(
                "T1_SIGNAL_ANNULAR_MIN",
                "Signal via annular ring must exceed fab minimum.",
                tier="T1",
                value=annular_ring,
                limit=limits["min_annular_ring_nm"],
            )
        )

    proof = _build_proof(results)
    return proof


def enforce_constraints(spec: CouponSpec) -> ConstraintEvaluation:
    proof = evaluate_constraints(spec)
    if proof.passed:
        return ConstraintEvaluation(spec=spec, resolved=resolve(spec), proof=proof, repair_info=None)
    if spec.constraints.mode == "REJECT":
        violations = [result for result in proof.constraints if not result.passed]
        raise ConstraintViolation(violations)
    repaired_spec, repair_info = repair_spec(spec)
    repaired_proof = evaluate_constraints(repaired_spec)
    return ConstraintEvaluation(
        spec=repaired_spec,
        resolved=resolve(repaired_spec),
        proof=repaired_proof,
        repair_info=repair_info,
    )


def constraint_proof_payload(proof: ConstraintProof) -> dict[str, Any]:
    return {
        "passed": proof.passed,
        "tiers": {tier: [result.constraint_id for result in proof.tiers[tier]] for tier in _TIERS},
        "constraints": [
            {
                "id": result.constraint_id,
                "description": result.description,
                "tier": result.tier,
                "value": result.value,
                "limit": result.limit,
                "margin": result.margin,
                "passed": result.passed,
            }
            for result in proof.constraints
        ],
    }


def repair_spec(spec: CouponSpec) -> tuple[CouponSpec, RepairInfo]:
    limits = resolve_fab_limits(spec)
    payload = spec.model_dump(mode="json")

    repair_map: dict[str, dict[str, int]] = {}
    repair_reason: list[str] = []

    def record(path: str, before: int, after: int, reason: str) -> None:
        if before == after:
            return
        repair_map[path] = {"before": before, "after": after}
        repair_reason.append(reason)

    def clamp_min(path: str, value: int, limit: int, reason: str) -> int:
        updated = value if value >= limit else limit
        record(path, value, updated, reason)
        return updated

    tl = payload["transmission_line"]
    tl["w_nm"] = clamp_min(
        "transmission_line.w_nm",
        int(tl["w_nm"]),
        limits["min_trace_width_nm"],
        "trace width raised to fab minimum",
    )
    tl["gap_nm"] = clamp_min(
        "transmission_line.gap_nm",
        int(tl["gap_nm"]),
        limits["min_gap_nm"],
        "trace gap raised to fab minimum",
    )

    if payload.get("discontinuity") is not None:
        signal_via = payload["discontinuity"]["signal_via"]
        signal_via["drill_nm"] = clamp_min(
            "discontinuity.signal_via.drill_nm",
            int(signal_via["drill_nm"]),
            limits["min_drill_nm"],
            "signal via drill raised to fab minimum",
        )
        pad_before = int(signal_via["pad_diameter_nm"])
        pad_after = max(pad_before, signal_via["drill_nm"] + limits["min_annular_ring_nm"])
        record(
            "discontinuity.signal_via.pad_diameter_nm",
            pad_before,
            pad_after,
            "signal via pad raised to satisfy annular ring minimum",
        )
        signal_via["pad_diameter_nm"] = pad_after

    repaired_spec = CouponSpec.model_validate(payload)
    repair_distance = _compute_repair_distance(repair_map)
    return repaired_spec, RepairInfo(repair_map=repair_map, repair_reason=repair_reason, repair_distance=repair_distance)


def _compute_repair_distance(repair_map: dict[str, dict[str, int]]) -> float:
    total = 0.0
    for change in repair_map.values():
        before = float(change["before"])
        after = float(change["after"])
        denom = max(abs(before), 1.0)
        total += abs(after - before) / denom
    return total


def resolve_fab_limits(spec: CouponSpec) -> dict[str, int]:
    """Resolve fab limits from profile ID and any overrides.

    Attempts to load the fab profile by ID. If the profile exists, uses its
    constraints as the base and applies any overrides from the spec.
    If the profile doesn't exist, falls back to conservative defaults.

    Args:
        spec: CouponSpec with fab_profile.id and optional overrides

    Returns:
        Dictionary of constraint limits in nanometers
    """
    overrides = spec.fab_profile.overrides or {}

    # Try to load the profile by ID
    try:
        profile = load_fab_profile(spec.fab_profile.id)
        base_limits = get_fab_limits(profile)
    except FileNotFoundError:
        # Fall back to conservative defaults if profile not found
        base_limits = {
            "min_trace_width_nm": 100_000,
            "min_gap_nm": 100_000,
            "min_drill_nm": 100_000,
            "min_annular_ring_nm": 100_000,
            "min_via_diameter_nm": 300_000,
            "min_edge_clearance_nm": 200_000,
            "min_soldermask_expansion_nm": 50_000,
            "min_soldermask_web_nm": 100_000,
            "min_silkscreen_width_nm": 150_000,
            "min_silkscreen_clearance_nm": 125_000,
        }

    # Apply overrides
    return {key: int(overrides.get(key, base_limits.get(key, 100_000))) for key in base_limits}


def resolve_fab_limits_from_profile(profile: FabCapabilityProfile) -> dict[str, int]:
    """Resolve fab limits directly from a loaded FabCapabilityProfile.

    Args:
        profile: A loaded FabCapabilityProfile instance

    Returns:
        Dictionary of constraint limits in nanometers
    """
    return get_fab_limits(profile)


def _min_constraint(
    constraint_id: str,
    description: str,
    *,
    tier: ConstraintTier,
    value: int,
    limit: int,
) -> ConstraintResult:
    margin = float(value - limit)
    return ConstraintResult(
        constraint_id=constraint_id,
        description=description,
        tier=tier,
        value=float(value),
        limit=float(limit),
        margin=margin,
        passed=margin >= 0,
    )


def _build_proof(results: list[ConstraintResult]) -> ConstraintProof:
    tiers: dict[ConstraintTier, list[ConstraintResult]] = {tier: [] for tier in _TIERS}
    for result in results:
        tiers[result.tier].append(result)
    frozen_tiers = {tier: tuple(items) for tier, items in tiers.items()}
    passed = all(result.passed for result in results)
    return ConstraintProof(constraints=tuple(results), tiers=frozen_tiers, passed=passed)
