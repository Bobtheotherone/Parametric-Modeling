"""Constraint system for coupon generation.

This package provides:
1. Core constraint evaluation (backward compatible with existing API)
2. Tiered constraint validation system with four tiers:
   - Tier 0: Parameter bounds (direct value checks against fab limits)
   - Tier 1: Derived scalar constraints (computed from multiple parameters)
   - Tier 2: Analytic spatial constraints (geometric relationships)
   - Tier 3: Exact geometry collision detection

The system supports REJECT mode which fails with constraint IDs and reasons.

REQ-M1-008: Tiered constraint system with Tiers 0-3
REQ-M1-009: REJECT mode with constraint IDs and reasons
"""

# Re-export core constraint types and functions for backward compatibility
from .core import (
    ConstraintEvaluation,
    ConstraintProof,
    ConstraintResult,
    ConstraintTier,
    ConstraintViolation,
    RepairInfo,
    constraint_proof_payload,
    enforce_constraints,
    evaluate_constraints,
    repair_spec,
    resolve_fab_limits,
    resolve_fab_limits_from_profile,
)

# Export tiered constraint system (new in M1-CONSTRAINTS-TIERS)
from .tiers import (
    ConstraintViolationError,
    Tier0Checker,
    Tier1Checker,
    Tier2Checker,
    Tier3Checker,
    TieredConstraintProof,
    TieredConstraintSystem,
    evaluate_tiered_constraints,
)
# Import ConstraintResult from tiers as TieredConstraintResult to avoid name collision
from .tiers import ConstraintResult as TieredConstraintResult

__all__ = [
    # Core constraint types (backward compatible)
    "ConstraintEvaluation",
    "ConstraintProof",
    "ConstraintResult",
    "ConstraintTier",
    "ConstraintViolation",
    "RepairInfo",
    # Core constraint functions (backward compatible)
    "constraint_proof_payload",
    "enforce_constraints",
    "evaluate_constraints",
    "repair_spec",
    "resolve_fab_limits",
    "resolve_fab_limits_from_profile",
    # Tiered constraint system (new)
    "ConstraintViolationError",
    "Tier0Checker",
    "Tier1Checker",
    "Tier2Checker",
    "Tier3Checker",
    "TieredConstraintProof",
    "TieredConstraintResult",
    "TieredConstraintSystem",
    "evaluate_tiered_constraints",
]
