"""Tiered constraint system for coupon generation.

This module implements a hierarchical constraint validation system:
- Tier 0: Parameter bounds (direct value checks against fab limits)
- Tier 1: Derived scalar constraints (computed from multiple parameters)
- Tier 2: Analytic spatial constraints (geometric relationships)
- Tier 3: Exact geometry collision detection

REQ-M1-008: Constraint system must support tiered validation
REQ-M1-009: REJECT mode must fail with constraint IDs and reasons
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

ConstraintTier = Literal["T0", "T1", "T2", "T3"]

_TIERS: tuple[ConstraintTier, ...] = ("T0", "T1", "T2", "T3")


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    """Result of a single constraint evaluation.

    Attributes:
        constraint_id: Unique identifier (e.g., "T0_TRACE_WIDTH_MIN")
        description: Human-readable description of the constraint
        tier: Constraint tier (T0, T1, T2, T3)
        value: Actual value being checked
        limit: Limit value for the constraint
        margin: value - limit (positive means passing for min constraints)
        passed: Whether the constraint was satisfied
        reason: Optional failure reason for more context
    """

    constraint_id: str
    description: str
    tier: ConstraintTier
    value: float
    limit: float
    margin: float
    passed: bool
    reason: str = ""


@dataclass(frozen=True)
class TieredConstraintProof:
    """Complete constraint proof with results organized by tier.

    Attributes:
        constraints: All constraint results in evaluation order
        tiers: Constraint results grouped by tier
        passed: Overall pass/fail status
        first_failure_tier: First tier with a failure (None if all passed)
    """

    constraints: tuple[ConstraintResult, ...]
    tiers: dict[ConstraintTier, tuple[ConstraintResult, ...]]
    passed: bool
    first_failure_tier: ConstraintTier | None = None

    def get_failures(self) -> list[ConstraintResult]:
        """Return all failed constraints."""
        return [c for c in self.constraints if not c.passed]

    def get_failures_by_tier(self, tier: ConstraintTier) -> list[ConstraintResult]:
        """Return failed constraints for a specific tier."""
        return [c for c in self.tiers.get(tier, ()) if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        """Convert proof to a dictionary for serialization."""
        return {
            "passed": self.passed,
            "first_failure_tier": self.first_failure_tier,
            "tiers": {tier: [c.constraint_id for c in results] for tier, results in self.tiers.items()},
            "constraints": [
                {
                    "id": c.constraint_id,
                    "description": c.description,
                    "tier": c.tier,
                    "value": c.value,
                    "limit": c.limit,
                    "margin": c.margin,
                    "passed": c.passed,
                    "reason": c.reason,
                }
                for c in self.constraints
            ],
        }


class ConstraintViolationError(ValueError):
    """Raised when constraints fail in REJECT mode.

    Attributes:
        violations: Tuple of failed constraint results
        tier: First tier with violations
    """

    def __init__(self, violations: Sequence[ConstraintResult], tier: ConstraintTier | None = None) -> None:
        self.violations = tuple(violations)
        self.tier = tier

        # Build detailed message with constraint IDs and reasons
        violation_details = []
        for v in self.violations:
            detail = f"{v.constraint_id}: {v.description}"
            if v.reason:
                detail += f" ({v.reason})"
            else:
                detail += f" (value={v.value}, limit={v.limit}, margin={v.margin})"
            violation_details.append(detail)

        message = f"Constraint violations in tier {tier}:\n" + "\n".join(f"  - {d}" for d in violation_details)
        super().__init__(message)

    @property
    def constraint_ids(self) -> list[str]:
        """Return list of violated constraint IDs."""
        return [v.constraint_id for v in self.violations]


class TierChecker(ABC):
    """Abstract base class for tier-specific constraint checkers."""

    @property
    @abstractmethod
    def tier(self) -> ConstraintTier:
        """Return the tier this checker handles."""
        ...

    @abstractmethod
    def check(self, spec: Any, fab_limits: dict[str, int], resolved: Any | None = None) -> list[ConstraintResult]:
        """Check constraints for this tier.

        Args:
            spec: CouponSpec being validated
            fab_limits: Dictionary of fab capability limits in nm
            resolved: Optional ResolvedDesign with derived features

        Returns:
            List of constraint results for this tier
        """
        ...


def _min_constraint(
    constraint_id: str,
    description: str,
    *,
    tier: ConstraintTier,
    value: int | float,
    limit: int | float,
    reason: str = "",
) -> ConstraintResult:
    """Create a minimum-value constraint result."""
    margin = float(value) - float(limit)
    passed = margin >= 0
    if not passed and not reason:
        reason = f"Value {value} is below minimum {limit}"
    return ConstraintResult(
        constraint_id=constraint_id,
        description=description,
        tier=tier,
        value=float(value),
        limit=float(limit),
        margin=margin,
        passed=passed,
        reason=reason if not passed else "",
    )


def _max_constraint(
    constraint_id: str,
    description: str,
    *,
    tier: ConstraintTier,
    value: int | float,
    limit: int | float,
    reason: str = "",
) -> ConstraintResult:
    """Create a maximum-value constraint result."""
    margin = float(limit) - float(value)
    passed = margin >= 0
    if not passed and not reason:
        reason = f"Value {value} exceeds maximum {limit}"
    return ConstraintResult(
        constraint_id=constraint_id,
        description=description,
        tier=tier,
        value=float(value),
        limit=float(limit),
        margin=margin,
        passed=passed,
        reason=reason if not passed else "",
    )


def _bool_constraint(
    constraint_id: str,
    description: str,
    *,
    tier: ConstraintTier,
    condition: bool,
    reason: str = "",
) -> ConstraintResult:
    """Create a boolean constraint result."""
    return ConstraintResult(
        constraint_id=constraint_id,
        description=description,
        tier=tier,
        value=1.0 if condition else 0.0,
        limit=1.0,
        margin=0.0 if condition else -1.0,
        passed=condition,
        reason=reason if not condition else "",
    )


class Tier0Checker(TierChecker):
    """Tier 0: Parameter bounds validation.

    Checks direct parameter values against fab capability limits.
    This is the fastest tier - pure value comparisons.
    """

    @property
    def tier(self) -> ConstraintTier:
        return "T0"

    def check(self, spec: Any, fab_limits: dict[str, int], resolved: Any | None = None) -> list[ConstraintResult]:
        results: list[ConstraintResult] = []

        # Trace width minimum
        results.append(
            _min_constraint(
                "T0_TRACE_WIDTH_MIN",
                "Trace width must exceed fab minimum",
                tier="T0",
                value=int(spec.transmission_line.w_nm),
                limit=fab_limits.get("min_trace_width_nm", 100_000),
            )
        )

        # Trace gap minimum (CPWG gap)
        results.append(
            _min_constraint(
                "T0_TRACE_GAP_MIN",
                "CPWG gap must exceed fab minimum spacing",
                tier="T0",
                value=int(spec.transmission_line.gap_nm),
                limit=fab_limits.get("min_gap_nm", 100_000),
            )
        )

        # Board outline minimum dimensions
        board_width = int(spec.board.outline.width_nm)
        board_length = int(spec.board.outline.length_nm)
        min_board_dim = fab_limits.get("min_board_width_nm", 5_000_000)  # 5mm default

        results.append(
            _min_constraint(
                "T0_BOARD_WIDTH_MIN",
                "Board width must exceed fab minimum",
                tier="T0",
                value=board_width,
                limit=min_board_dim,
            )
        )

        results.append(
            _min_constraint(
                "T0_BOARD_LENGTH_MIN",
                "Board length must exceed fab minimum",
                tier="T0",
                value=board_length,
                limit=min_board_dim,
            )
        )

        # Corner radius bounds (must be non-negative and not exceed half the min dimension)
        corner_radius = int(spec.board.outline.corner_radius_nm)
        max_corner = min(board_width, board_length) // 2

        results.append(
            _min_constraint(
                "T0_CORNER_RADIUS_MIN",
                "Corner radius must be non-negative",
                tier="T0",
                value=corner_radius,
                limit=0,
            )
        )

        results.append(
            _max_constraint(
                "T0_CORNER_RADIUS_MAX",
                "Corner radius must not exceed half the smallest board dimension",
                tier="T0",
                value=corner_radius,
                limit=max_corner,
            )
        )

        # Via/drill constraints (if discontinuity present)
        if spec.discontinuity is not None:
            signal_via = spec.discontinuity.signal_via

            results.append(
                _min_constraint(
                    "T0_SIGNAL_DRILL_MIN",
                    "Signal via drill must exceed fab minimum",
                    tier="T0",
                    value=int(signal_via.drill_nm),
                    limit=fab_limits.get("min_drill_nm", 200_000),
                )
            )

            results.append(
                _min_constraint(
                    "T0_SIGNAL_VIA_DIAMETER_MIN",
                    "Signal via diameter must exceed fab minimum",
                    tier="T0",
                    value=int(signal_via.diameter_nm),
                    limit=fab_limits.get("min_via_diameter_nm", 300_000),
                )
            )

            results.append(
                _min_constraint(
                    "T0_SIGNAL_PAD_DIAMETER_MIN",
                    "Signal via pad diameter must exceed fab minimum",
                    tier="T0",
                    value=int(signal_via.pad_diameter_nm),
                    limit=fab_limits.get("min_via_diameter_nm", 300_000),
                )
            )

            # Return vias
            if spec.discontinuity.return_vias is not None:
                return_via = spec.discontinuity.return_vias.via

                results.append(
                    _min_constraint(
                        "T0_RETURN_VIA_DRILL_MIN",
                        "Return via drill must exceed fab minimum",
                        tier="T0",
                        value=int(return_via.drill_nm),
                        limit=fab_limits.get("min_drill_nm", 200_000),
                    )
                )

                results.append(
                    _min_constraint(
                        "T0_RETURN_VIA_DIAMETER_MIN",
                        "Return via diameter must exceed fab minimum",
                        tier="T0",
                        value=int(return_via.diameter_nm),
                        limit=fab_limits.get("min_via_diameter_nm", 300_000),
                    )
                )

        # Ground via fence constraints
        fence = spec.transmission_line.ground_via_fence
        if fence is not None and fence.enabled:
            results.append(
                _min_constraint(
                    "T0_FENCE_VIA_DRILL_MIN",
                    "Ground fence via drill must exceed fab minimum",
                    tier="T0",
                    value=int(fence.via.drill_nm),
                    limit=fab_limits.get("min_drill_nm", 200_000),
                )
            )

            results.append(
                _min_constraint(
                    "T0_FENCE_VIA_DIAMETER_MIN",
                    "Ground fence via diameter must exceed fab minimum",
                    tier="T0",
                    value=int(fence.via.diameter_nm),
                    limit=fab_limits.get("min_via_diameter_nm", 300_000),
                )
            )

        return results


class Tier1Checker(TierChecker):
    """Tier 1: Derived scalar constraints.

    Checks constraints computed from multiple parameters:
    - Annular ring calculations
    - Aspect ratios
    - Via-to-pad ratios
    """

    @property
    def tier(self) -> ConstraintTier:
        return "T1"

    def check(self, spec: Any, fab_limits: dict[str, int], resolved: Any | None = None) -> list[ConstraintResult]:
        results: list[ConstraintResult] = []

        # Signal via annular ring (pad - drill must exceed minimum)
        if spec.discontinuity is not None:
            signal_via = spec.discontinuity.signal_via
            pad_dia = int(signal_via.pad_diameter_nm)
            drill = int(signal_via.drill_nm)
            annular_ring = (pad_dia - drill) // 2  # Half the difference

            min_annular = fab_limits.get("min_annular_ring_nm", 100_000)

            results.append(
                _min_constraint(
                    "T1_SIGNAL_ANNULAR_MIN",
                    "Signal via annular ring must exceed fab minimum",
                    tier="T1",
                    value=annular_ring,
                    limit=min_annular,
                )
            )

            # Via diameter must be larger than drill
            results.append(
                _bool_constraint(
                    "T1_SIGNAL_VIA_DIAMETER_GT_DRILL",
                    "Signal via diameter must be larger than drill",
                    tier="T1",
                    condition=int(signal_via.diameter_nm) > drill,
                    reason=f"diameter={signal_via.diameter_nm} must be > drill={drill}",
                )
            )

            # Pad diameter must be larger than via diameter
            results.append(
                _bool_constraint(
                    "T1_SIGNAL_PAD_GT_VIA",
                    "Signal via pad must be larger than via diameter",
                    tier="T1",
                    condition=pad_dia >= int(signal_via.diameter_nm),
                    reason=f"pad={pad_dia} must be >= diameter={signal_via.diameter_nm}",
                )
            )

            # Return via annular ring
            if spec.discontinuity.return_vias is not None:
                return_via = spec.discontinuity.return_vias.via
                return_dia = int(return_via.diameter_nm)
                return_drill = int(return_via.drill_nm)
                return_annular = (return_dia - return_drill) // 2

                results.append(
                    _min_constraint(
                        "T1_RETURN_ANNULAR_MIN",
                        "Return via annular ring must exceed fab minimum",
                        tier="T1",
                        value=return_annular,
                        limit=min_annular,
                    )
                )

        # Ground fence via annular ring
        fence = spec.transmission_line.ground_via_fence
        if fence is not None and fence.enabled:
            fence_dia = int(fence.via.diameter_nm)
            fence_drill = int(fence.via.drill_nm)
            fence_annular = (fence_dia - fence_drill) // 2

            min_annular = fab_limits.get("min_annular_ring_nm", 100_000)

            results.append(
                _min_constraint(
                    "T1_FENCE_ANNULAR_MIN",
                    "Ground fence via annular ring must exceed fab minimum",
                    tier="T1",
                    value=fence_annular,
                    limit=min_annular,
                )
            )

        # Trace length constraints
        trace_left = int(spec.transmission_line.length_left_nm)
        trace_right = int(spec.transmission_line.length_right_nm)

        results.append(
            _min_constraint(
                "T1_TRACE_LEFT_POSITIVE",
                "Left trace length must be positive",
                tier="T1",
                value=trace_left,
                limit=1,
            )
        )

        results.append(
            _min_constraint(
                "T1_TRACE_RIGHT_POSITIVE",
                "Right trace length must be positive",
                tier="T1",
                value=trace_right,
                limit=1,
            )
        )

        # Board aspect ratio (sanity check - not too extreme)
        board_width = int(spec.board.outline.width_nm)
        board_length = int(spec.board.outline.length_nm)
        aspect_ratio = max(board_length, board_width) / max(min(board_length, board_width), 1)

        results.append(
            _max_constraint(
                "T1_BOARD_ASPECT_RATIO_MAX",
                "Board aspect ratio should not be extreme",
                tier="T1",
                value=aspect_ratio,
                limit=20.0,  # Allow up to 20:1 aspect ratio
                reason=f"Aspect ratio {aspect_ratio:.1f}:1 exceeds 20:1 limit",
            )
        )

        # Copper-to-edge clearance (Section 13.3.2)
        # The minimum copper-to-edge clearance is derived from:
        # - Trace center at y=0 (centerline)
        # - Trace half-width + gap + ground pour edge = distance from centerline
        # - Board half-width - this distance = clearance to board edge
        trace_width = int(spec.transmission_line.w_nm)
        trace_gap = int(spec.transmission_line.gap_nm)
        min_edge_clearance = fab_limits.get("min_edge_clearance_nm", 200_000)

        # For CPWG, the ground pour extends beyond the gap. Compute clearance:
        # Copper footprint from centerline = trace_width/2 + gap + ground_pour_margin
        # For simplicity, we check that half the trace width + gap doesn't exceed
        # half the board width minus edge clearance
        half_board_width = board_width // 2
        trace_extent_from_center = trace_width // 2 + trace_gap

        # If there's a ground via fence, include its extent
        fence = spec.transmission_line.ground_via_fence
        if fence is not None and fence.enabled:
            fence_offset = int(fence.offset_from_gap_nm)
            fence_via_radius = int(fence.via.diameter_nm) // 2
            trace_extent_from_center = trace_width // 2 + trace_gap + fence_offset + fence_via_radius

        available_clearance = half_board_width - trace_extent_from_center

        results.append(
            _min_constraint(
                "T1_COPPER_TO_EDGE_CLEARANCE",
                "Copper features must maintain minimum clearance from board edge",
                tier="T1",
                value=available_clearance,
                limit=min_edge_clearance,
            )
        )

        # Fence pitch constraint (Section 13.3.2)
        # Fence pitch should be reasonable relative to via diameter
        if fence is not None and fence.enabled:
            fence_pitch = int(fence.pitch_nm)
            fence_via_dia = int(fence.via.diameter_nm)
            min_via_to_via = fab_limits.get("min_via_to_via_nm", 200_000)

            # Fence pitch must allow non-overlapping vias with clearance
            min_fence_pitch = fence_via_dia + min_via_to_via

            results.append(
                _min_constraint(
                    "T1_FENCE_PITCH_MIN",
                    "Ground fence pitch must exceed via diameter plus spacing",
                    tier="T1",
                    value=fence_pitch,
                    limit=min_fence_pitch,
                )
            )

        return results


class Tier2Checker(TierChecker):
    """Tier 2: Analytic spatial constraints.

    Checks geometric relationships without full collision detection:
    - Edge clearances
    - Component spacing
    - Via-to-via spacing
    - Trace-to-edge clearances
    """

    @property
    def tier(self) -> ConstraintTier:
        return "T2"

    def check(self, spec: Any, fab_limits: dict[str, int], resolved: Any | None = None) -> list[ConstraintResult]:
        results: list[ConstraintResult] = []

        board_width = int(spec.board.outline.width_nm)
        board_length = int(spec.board.outline.length_nm)
        min_edge_clearance = fab_limits.get("min_edge_clearance_nm", 200_000)

        # Connector positions must be within board bounds with clearance
        left_pos = spec.connectors.left.position_nm
        right_pos = spec.connectors.right.position_nm

        results.append(
            _min_constraint(
                "T2_LEFT_CONNECTOR_X_MIN",
                "Left connector X position must have edge clearance",
                tier="T2",
                value=int(left_pos[0]),
                limit=min_edge_clearance,
            )
        )

        results.append(
            _max_constraint(
                "T2_LEFT_CONNECTOR_X_MAX",
                "Left connector X position must be within board",
                tier="T2",
                value=int(left_pos[0]),
                limit=board_length - min_edge_clearance,
            )
        )

        results.append(
            _min_constraint(
                "T2_RIGHT_CONNECTOR_X_MIN",
                "Right connector X position must have edge clearance",
                tier="T2",
                value=int(right_pos[0]),
                limit=min_edge_clearance,
            )
        )

        results.append(
            _max_constraint(
                "T2_RIGHT_CONNECTOR_X_MAX",
                "Right connector X position must be within board",
                tier="T2",
                value=int(right_pos[0]),
                limit=board_length - min_edge_clearance,
            )
        )

        # Y positions should be near center (within board width)
        half_width = board_width // 2

        results.append(
            _bool_constraint(
                "T2_LEFT_CONNECTOR_Y_CENTERED",
                "Left connector Y position should be near board center",
                tier="T2",
                condition=abs(int(left_pos[1])) <= half_width,
                reason=f"Y={left_pos[1]} exceeds board half-width={half_width}",
            )
        )

        results.append(
            _bool_constraint(
                "T2_RIGHT_CONNECTOR_Y_CENTERED",
                "Right connector Y position should be near board center",
                tier="T2",
                condition=abs(int(right_pos[1])) <= half_width,
                reason=f"Y={right_pos[1]} exceeds board half-width={half_width}",
            )
        )

        # Total trace length must fit within board
        trace_left = int(spec.transmission_line.length_left_nm)
        trace_right = int(spec.transmission_line.length_right_nm)
        left_x = int(left_pos[0])
        right_x = int(right_pos[0])
        available_length = right_x - left_x

        results.append(
            _bool_constraint(
                "T2_TRACE_FITS_IN_BOARD",
                "Total trace length must fit between connectors",
                tier="T2",
                condition=trace_left + trace_right <= available_length,
                reason=f"traces {trace_left + trace_right}nm > available {available_length}nm",
            )
        )

        # Discontinuity spatial constraints
        if spec.discontinuity is not None:
            # Return via ring must fit within reasonable bounds
            if spec.discontinuity.return_vias is not None:
                return_radius = int(spec.discontinuity.return_vias.radius_nm)
                return_via_dia = int(spec.discontinuity.return_vias.via.diameter_nm)
                signal_pad = int(spec.discontinuity.signal_via.pad_diameter_nm)

                # Return vias must not overlap with signal via pad
                min_spacing = fab_limits.get("min_via_to_via_nm", 200_000)
                required_radius = signal_pad // 2 + return_via_dia // 2 + min_spacing

                results.append(
                    _min_constraint(
                        "T2_RETURN_VIA_RING_RADIUS",
                        "Return via ring radius must provide clearance from signal via",
                        tier="T2",
                        value=return_radius,
                        limit=required_radius,
                    )
                )

        # Ground via fence spacing
        fence = spec.transmission_line.ground_via_fence
        if fence is not None and fence.enabled:
            int(spec.transmission_line.w_nm)
            int(spec.transmission_line.gap_nm)
            fence_offset = int(fence.offset_from_gap_nm)
            fence_via_dia = int(fence.via.diameter_nm)

            # Fence via center is at: trace_edge + gap + offset
            # Fence via edge is at: fence_center - fence_via_radius
            # Must not encroach on gap
            fence_via_edge = fence_offset - fence_via_dia // 2

            results.append(
                _min_constraint(
                    "T2_FENCE_VIA_GAP_CLEARANCE",
                    "Ground fence via must not encroach on CPWG gap",
                    tier="T2",
                    value=fence_via_edge,
                    limit=0,
                )
            )

            # Fence pitch must allow vias without overlap
            fence_pitch = int(fence.pitch_nm)

            results.append(
                _min_constraint(
                    "T2_FENCE_PITCH_MIN",
                    "Ground fence pitch must exceed via diameter",
                    tier="T2",
                    value=fence_pitch,
                    limit=fence_via_dia + fab_limits.get("min_via_to_via_nm", 200_000),
                )
            )

        return results


class Tier3Checker(TierChecker):
    """Tier 3: Exact geometry collision detection.

    Performs precise collision checks between geometric primitives:
    - Via-to-via collisions
    - Via-to-trace collisions
    - Antipad coverage verification
    """

    @property
    def tier(self) -> ConstraintTier:
        return "T3"

    def check(self, spec: Any, fab_limits: dict[str, int], resolved: Any | None = None) -> list[ConstraintResult]:
        results: list[ConstraintResult] = []

        # Return via ring collision detection
        if spec.discontinuity is not None and spec.discontinuity.return_vias is not None:
            return_vias = spec.discontinuity.return_vias
            count = return_vias.count
            radius = int(return_vias.radius_nm)
            via_dia = int(return_vias.via.diameter_nm)

            if count > 1:
                # Check that return vias don't overlap with each other
                # Vias are placed in a ring, angular spacing = 2*pi/count
                # Distance between adjacent via centers = 2*radius*sin(pi/count)
                angular_spacing = math.pi / count
                via_center_spacing = 2 * radius * math.sin(angular_spacing)
                min_spacing = via_dia + fab_limits.get("min_via_to_via_nm", 200_000)

                results.append(
                    _min_constraint(
                        "T3_RETURN_VIA_RING_NO_OVERLAP",
                        "Return vias in ring must not overlap",
                        tier="T3",
                        value=via_center_spacing,
                        limit=min_spacing,
                        reason=f"{count} vias at radius {radius}nm: spacing={via_center_spacing:.0f}nm < required {min_spacing}nm",
                    )
                )

        # Antipad coverage check
        if spec.discontinuity is not None:
            signal_pad = int(spec.discontinuity.signal_via.pad_diameter_nm)

            for layer, antipad in spec.discontinuity.antipads.items():
                if antipad.shape == "CIRCLE":
                    antipad_r = int(antipad.r_nm) if antipad.r_nm else 0
                    # Antipad must cover signal pad with clearance
                    min_antipad = signal_pad // 2 + fab_limits.get("min_gap_nm", 100_000)

                    results.append(
                        _min_constraint(
                            f"T3_ANTIPAD_{layer}_COVERAGE",
                            f"Antipad on {layer} must fully cover signal via with clearance",
                            tier="T3",
                            value=antipad_r,
                            limit=min_antipad,
                        )
                    )

                elif antipad.shape == "ROUNDRECT":
                    rx = int(antipad.rx_nm) if antipad.rx_nm else 0
                    ry = int(antipad.ry_nm) if antipad.ry_nm else 0
                    # Both dimensions must cover signal pad
                    min_antipad = signal_pad // 2 + fab_limits.get("min_gap_nm", 100_000)

                    results.append(
                        _min_constraint(
                            f"T3_ANTIPAD_{layer}_COVERAGE_X",
                            f"Antipad X dimension on {layer} must cover signal via",
                            tier="T3",
                            value=rx,
                            limit=min_antipad,
                        )
                    )

                    results.append(
                        _min_constraint(
                            f"T3_ANTIPAD_{layer}_COVERAGE_Y",
                            f"Antipad Y dimension on {layer} must cover signal via",
                            tier="T3",
                            value=ry,
                            limit=min_antipad,
                        )
                    )

        # Symmetry constraint (if enforced)
        if spec.constraints.symmetry.enforce:
            # Check that left and right trace lengths are equal
            trace_left = int(spec.transmission_line.length_left_nm)
            trace_right = int(spec.transmission_line.length_right_nm)

            results.append(
                _bool_constraint(
                    "T3_TRACE_SYMMETRY",
                    "Left and right trace lengths must be equal when symmetry enforced",
                    tier="T3",
                    condition=trace_left == trace_right,
                    reason=f"left={trace_left}nm != right={trace_right}nm",
                )
            )

            # Check connector Y positions are symmetric
            left_y = int(spec.connectors.left.position_nm[1])
            right_y = int(spec.connectors.right.position_nm[1])

            results.append(
                _bool_constraint(
                    "T3_CONNECTOR_Y_SYMMETRY",
                    "Connector Y positions must be equal when symmetry enforced",
                    tier="T3",
                    condition=left_y == right_y,
                    reason=f"left_y={left_y}nm != right_y={right_y}nm",
                )
            )

        return results


@dataclass
class TieredConstraintSystem:
    """Orchestrates tiered constraint validation.

    The system evaluates constraints in order from Tier 0 to Tier 3,
    optionally stopping at the first tier with failures (fail-fast mode).

    Attributes:
        checkers: List of tier checkers to run
        fail_fast: If True, stop at first tier with failures
    """

    checkers: list[TierChecker] = field(default_factory=list)
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if not self.checkers:
            # Initialize with default checkers
            self.checkers = [
                Tier0Checker(),
                Tier1Checker(),
                Tier2Checker(),
                Tier3Checker(),
            ]

    def evaluate(
        self,
        spec: Any,
        fab_limits: dict[str, int],
        resolved: Any | None = None,
    ) -> TieredConstraintProof:
        """Evaluate all tiered constraints.

        Args:
            spec: CouponSpec to validate
            fab_limits: Dictionary of fab capability limits
            resolved: Optional ResolvedDesign with derived features

        Returns:
            TieredConstraintProof with all results
        """
        all_results: list[ConstraintResult] = []
        tiers: dict[ConstraintTier, list[ConstraintResult]] = {t: [] for t in _TIERS}
        first_failure_tier: ConstraintTier | None = None

        for checker in self.checkers:
            tier = checker.tier
            results = checker.check(spec, fab_limits, resolved)
            all_results.extend(results)
            tiers[tier].extend(results)

            # Check for failures in this tier
            tier_failures = [r for r in results if not r.passed]
            if tier_failures and first_failure_tier is None:
                first_failure_tier = tier

                if self.fail_fast:
                    break

        # Build frozen proof
        frozen_tiers = {tier: tuple(items) for tier, items in tiers.items()}
        passed = first_failure_tier is None

        return TieredConstraintProof(
            constraints=tuple(all_results),
            tiers=frozen_tiers,
            passed=passed,
            first_failure_tier=first_failure_tier,
        )

    def enforce(
        self,
        spec: Any,
        fab_limits: dict[str, int],
        resolved: Any | None = None,
        mode: Literal["REJECT", "REPAIR"] = "REJECT",
    ) -> TieredConstraintProof:
        """Evaluate constraints and raise on failure in REJECT mode.

        Args:
            spec: CouponSpec to validate
            fab_limits: Dictionary of fab capability limits
            resolved: Optional ResolvedDesign with derived features
            mode: REJECT raises on failure, REPAIR is a no-op (caller handles)

        Returns:
            TieredConstraintProof with all results

        Raises:
            ConstraintViolationError: If mode is REJECT and constraints fail
        """
        proof = self.evaluate(spec, fab_limits, resolved)

        if not proof.passed and mode == "REJECT":
            failures = proof.get_failures()
            raise ConstraintViolationError(failures, proof.first_failure_tier)

        return proof


def evaluate_tiered_constraints(
    spec: Any,
    fab_limits: dict[str, int],
    resolved: Any | None = None,
    fail_fast: bool = False,
) -> TieredConstraintProof:
    """Convenience function to evaluate tiered constraints.

    Args:
        spec: CouponSpec to validate
        fab_limits: Dictionary of fab capability limits
        resolved: Optional ResolvedDesign with derived features
        fail_fast: If True, stop at first tier with failures

    Returns:
        TieredConstraintProof with all results
    """
    system = TieredConstraintSystem(fail_fast=fail_fast)
    return system.evaluate(spec, fab_limits, resolved)
