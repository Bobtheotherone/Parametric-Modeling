"""Constraint primitives and result types per Section 13.3.1.

This module defines the core primitive types for the constraint system:
- Constraint: A declarative constraint definition with id, tier, category, etc.
- ConstraintResult: The result of evaluating a single constraint
- ConstraintContext: Evaluation context providing access to spec, limits, and resolved values

These primitives form the foundation for the tiered constraint validation system
and support both REJECT and REPAIR modes with full auditability.

REQ-M1-008: Constraint system must support tiered validation with primitives
REQ-M1-009: REJECT mode must fail with constraint IDs and reasons
REQ-M1-010: REPAIR mode must project infeasible specs with auditable repair info
REQ-M1-011: constraint_proof.json with per-constraint evaluations and signed margins
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from ..fab_profiles import FabCapabilityProfile
    from ..resolve import ResolvedDesign
    from ..spec import CouponSpec


class ConstraintTier(str, Enum):
    """Constraint evaluation tier.

    Constraints are organized into tiers by computational complexity:
    - T0: Parameter bounds (direct value checks against fab limits)
    - T1: Derived scalar constraints (computed from multiple parameters)
    - T2: Analytic spatial constraints (geometric relationships)
    - T3: Exact geometry collision detection
    - T4: External validation (DRC, etc.)
    """

    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"


class ConstraintCategory(str, Enum):
    """Constraint category for grouping related constraints.

    Categories help organize constraints by their functional area:
    - FABRICATION: Fab capability limits (trace width, drill size, etc.)
    - GEOMETRY: Geometric validity (no overlaps, connectivity, etc.)
    - TOPOLOGY: Topological correctness (net connectivity, etc.)
    - SPACING: Clearance and spacing requirements
    - MATERIAL: Material property constraints
    - ELECTRICAL: Electrical design rules
    """

    FABRICATION = "FABRICATION"
    GEOMETRY = "GEOMETRY"
    TOPOLOGY = "TOPOLOGY"
    SPACING = "SPACING"
    MATERIAL = "MATERIAL"
    ELECTRICAL = "ELECTRICAL"


class ConstraintSeverity(str, Enum):
    """Severity level for constraint violations.

    - ERROR: Constraint must pass; violation blocks build
    - WARNING: Constraint should pass; violation generates warning
    - INFO: Informational constraint; violation is logged
    """

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


ConstraintTierLiteral = Literal["T0", "T1", "T2", "T3", "T4"]
ConstraintCategoryLiteral = Literal[
    "FABRICATION", "GEOMETRY", "TOPOLOGY", "SPACING", "MATERIAL", "ELECTRICAL"
]
ConstraintSeverityLiteral = Literal["ERROR", "WARNING", "INFO"]


@dataclass(frozen=True, slots=True)
class Constraint:
    """A declarative constraint definition.

    Constraints define rules that must be satisfied for a valid design.
    Each constraint has a unique ID, tier, category, and evaluation expression.

    Attributes:
        id: Unique constraint identifier (e.g., "T0_TRACE_WIDTH_MIN")
        tier: Evaluation tier (T0-T4) determining when the constraint is checked
        category: Functional category for grouping related constraints
        description: Human-readable description of the constraint
        expr: Expression or callable that evaluates the constraint
        severity: Severity level for violations (ERROR, WARNING, INFO)
        must_pass: Whether this constraint must pass for the design to be valid
        repairable: Whether this constraint can be auto-repaired in REPAIR mode
        repair_strategy: Optional strategy hint for REPAIR mode
        metadata: Optional metadata for extensibility
    """

    id: str
    tier: ConstraintTierLiteral
    category: ConstraintCategoryLiteral
    description: str
    expr: str | Callable[["ConstraintContext"], "ConstraintResult"]
    severity: ConstraintSeverityLiteral = "ERROR"
    must_pass: bool = True
    repairable: bool = False
    repair_strategy: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate constraint definition."""
        if not self.id:
            msg = "Constraint id cannot be empty"
            raise ValueError(msg)
        if not self.description:
            msg = "Constraint description cannot be empty"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Convert constraint to dictionary for serialization."""
        result: dict[str, Any] = {
            "id": self.id,
            "tier": self.tier,
            "category": self.category,
            "description": self.description,
            "severity": self.severity,
            "must_pass": self.must_pass,
            "repairable": self.repairable,
        }
        if self.repair_strategy:
            result["repair_strategy"] = self.repair_strategy
        if self.metadata:
            result["metadata"] = self.metadata
        # expr is not serialized as it may be a callable
        if isinstance(self.expr, str):
            result["expr"] = self.expr
        return result


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    """Result of evaluating a single constraint.

    Captures the evaluation outcome including the actual and limit values,
    the signed margin, and whether the constraint passed.

    Attributes:
        constraint_id: ID of the constraint that was evaluated
        description: Human-readable description of the constraint
        tier: Tier of the constraint (T0-T4)
        category: Category of the constraint
        value: Actual value being checked
        limit: Limit value for the constraint
        margin: Signed margin (value - limit for min constraints, limit - value for max)
        passed: Whether the constraint was satisfied
        severity: Severity level of the constraint
        must_pass: Whether this constraint must pass for validity
        reason: Optional detailed failure reason
        repair_hint: Optional hint for auto-repair in REPAIR mode
    """

    constraint_id: str
    description: str
    tier: ConstraintTierLiteral
    category: ConstraintCategoryLiteral
    value: float
    limit: float
    margin: float
    passed: bool
    severity: ConstraintSeverityLiteral = "ERROR"
    must_pass: bool = True
    reason: str = ""
    repair_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        result: dict[str, Any] = {
            "id": self.constraint_id,
            "description": self.description,
            "tier": self.tier,
            "category": self.category,
            "value": self.value,
            "limit": self.limit,
            "margin": self.margin,
            "passed": self.passed,
            "severity": self.severity,
            "must_pass": self.must_pass,
        }
        if self.reason:
            result["reason"] = self.reason
        if self.repair_hint:
            result["repair_hint"] = self.repair_hint
        return result

    @classmethod
    def from_constraint(
        cls,
        constraint: Constraint,
        *,
        value: float,
        limit: float,
        margin: float,
        passed: bool,
        reason: str = "",
        repair_hint: str = "",
    ) -> "ConstraintResult":
        """Create a result from a constraint definition."""
        return cls(
            constraint_id=constraint.id,
            description=constraint.description,
            tier=constraint.tier,
            category=constraint.category,
            value=value,
            limit=limit,
            margin=margin,
            passed=passed,
            severity=constraint.severity,
            must_pass=constraint.must_pass,
            reason=reason,
            repair_hint=repair_hint,
        )


@dataclass
class ConstraintContext:
    """Evaluation context for constraint checking.

    Provides access to the spec, fab limits, resolved design, and other
    context needed during constraint evaluation.

    Attributes:
        spec: The CouponSpec being validated
        fab_limits: Dictionary of fab capability limits in nm
        fab_profile: Optional loaded FabCapabilityProfile
        resolved: Optional ResolvedDesign with derived features
        mode: Constraint mode (REJECT or REPAIR)
        cache: Optional cache for expensive computations
    """

    spec: "CouponSpec"
    fab_limits: dict[str, int]
    fab_profile: "FabCapabilityProfile | None" = None
    resolved: "ResolvedDesign | None" = None
    mode: Literal["REJECT", "REPAIR"] = "REJECT"
    cache: dict[str, Any] = field(default_factory=dict)

    def get_spec_value(self, path: str) -> Any:
        """Get a value from the spec using dotted path notation.

        Args:
            path: Dotted path like "transmission_line.w_nm"

        Returns:
            The value at the path, or None if not found
        """
        parts = path.split(".")
        current: Any = self.spec
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def get_fab_limit(self, key: str, default: int = 0) -> int:
        """Get a fab limit value.

        Args:
            key: Limit key like "min_trace_width_nm"
            default: Default value if key not found

        Returns:
            The limit value in nm
        """
        return self.fab_limits.get(key, default)

    def get_cached(self, key: str) -> Any | None:
        """Get a cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        return self.cache.get(key)

    def set_cached(self, key: str, value: Any) -> None:
        """Set a cached value.

        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = value


def create_min_constraint_result(
    constraint: Constraint,
    *,
    value: int | float,
    limit: int | float,
    reason_template: str = "",
) -> ConstraintResult:
    """Create a minimum-value constraint result.

    For minimum constraints, margin = value - limit, and passed = margin >= 0.

    Args:
        constraint: The constraint definition
        value: Actual value being checked
        limit: Minimum limit value
        reason_template: Optional template for failure reason

    Returns:
        ConstraintResult with computed margin and pass status
    """
    margin = float(value) - float(limit)
    passed = margin >= 0
    reason = ""
    if not passed:
        if reason_template:
            reason = reason_template.format(value=value, limit=limit, margin=margin)
        else:
            reason = f"Value {value} is below minimum {limit}"
    repair_hint = f"Increase to at least {limit}" if not passed else ""

    return ConstraintResult.from_constraint(
        constraint,
        value=float(value),
        limit=float(limit),
        margin=margin,
        passed=passed,
        reason=reason,
        repair_hint=repair_hint,
    )


def create_max_constraint_result(
    constraint: Constraint,
    *,
    value: int | float,
    limit: int | float,
    reason_template: str = "",
) -> ConstraintResult:
    """Create a maximum-value constraint result.

    For maximum constraints, margin = limit - value, and passed = margin >= 0.

    Args:
        constraint: The constraint definition
        value: Actual value being checked
        limit: Maximum limit value
        reason_template: Optional template for failure reason

    Returns:
        ConstraintResult with computed margin and pass status
    """
    margin = float(limit) - float(value)
    passed = margin >= 0
    reason = ""
    if not passed:
        if reason_template:
            reason = reason_template.format(value=value, limit=limit, margin=margin)
        else:
            reason = f"Value {value} exceeds maximum {limit}"
    repair_hint = f"Decrease to at most {limit}" if not passed else ""

    return ConstraintResult.from_constraint(
        constraint,
        value=float(value),
        limit=float(limit),
        margin=margin,
        passed=passed,
        reason=reason,
        repair_hint=repair_hint,
    )


def create_bool_constraint_result(
    constraint: Constraint,
    *,
    condition: bool,
    reason: str = "",
) -> ConstraintResult:
    """Create a boolean constraint result.

    For boolean constraints, margin is 0 if passed, -1 if failed.

    Args:
        constraint: The constraint definition
        condition: Boolean condition to check
        reason: Failure reason if condition is False

    Returns:
        ConstraintResult with boolean outcome
    """
    return ConstraintResult.from_constraint(
        constraint,
        value=1.0 if condition else 0.0,
        limit=1.0,
        margin=0.0 if condition else -1.0,
        passed=condition,
        reason=reason if not condition else "",
    )


def create_equality_constraint_result(
    constraint: Constraint,
    *,
    value: int | float,
    expected: int | float,
    tolerance: float = 0.0,
    reason_template: str = "",
) -> ConstraintResult:
    """Create an equality constraint result.

    For equality constraints, margin = tolerance - abs(value - expected),
    and passed = margin >= 0.

    Args:
        constraint: The constraint definition
        value: Actual value
        expected: Expected value
        tolerance: Allowed tolerance (default 0 for exact match)
        reason_template: Optional template for failure reason

    Returns:
        ConstraintResult with equality outcome
    """
    diff = abs(float(value) - float(expected))
    margin = float(tolerance) - diff
    passed = margin >= 0
    reason = ""
    if not passed:
        if reason_template:
            reason = reason_template.format(value=value, expected=expected, diff=diff)
        else:
            reason = f"Value {value} differs from expected {expected} by {diff}"

    return ConstraintResult.from_constraint(
        constraint,
        value=float(value),
        limit=float(expected),
        margin=margin,
        passed=passed,
        reason=reason,
    )
