"""Constraint system for coupon generation.

This package provides:
1. Constraint primitives: Declarative constraint definitions with id, tier,
   category, description, expr, severity, and must_pass fields (Section 13.3.1)
2. Core constraint evaluation (backward compatible with existing API)
3. Tiered constraint validation system with four tiers:
   - Tier 0: Parameter bounds (direct value checks against fab limits)
   - Tier 1: Derived scalar constraints (computed from multiple parameters)
   - Tier 2: Analytic spatial constraints (geometric relationships)
   - Tier 3: Exact geometry collision detection
4. REPAIR mode that projects infeasible specs into feasible space with
   auditable repair_map, repair_reason, and repair_distance
5. constraint_proof.json generation with per-constraint evaluations and
   signed margins

The system supports REJECT mode which fails with constraint IDs and reasons,
and REPAIR mode which projects infeasible specs into feasible space.

REQ-M1-008: Tiered constraint system with Tiers 0-3
REQ-M1-009: REJECT mode with constraint IDs and reasons
REQ-M1-010: REPAIR mode with repair_map, repair_reason, repair_distance
REQ-M1-011: constraint_proof.json with per-constraint evaluations and signed margins
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

# Export GPU-accelerated batch filtering (new in M1-GPU-FILTER)
# Updated for CP-4.1 formal API with mode, seed, and RepairMeta
from .gpu_filter import (
    BatchFilterResult,
    FabProfiles,
    FamilyF1ParameterSpace,
    GPUConstraintFilter,
    ParameterMapping,
    RepairMeta,
    batch_filter,
    is_gpu_available,
)

# Export REPAIR mode and constraint proof generation (new in M1-CONSTRAINTS-REPAIR)
from .repair import (
    ConstraintProofDocument,
    RepairAction,
    RepairEngine,
    RepairResult,
    generate_constraint_proof,
    repair_spec_tiered,
    write_constraint_proof,
)

# Import ConstraintResult from tiers as TieredConstraintResult to avoid name collision
from .tiers import ConstraintResult as TieredConstraintResult

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

# Export constraint primitives (Section 13.3.1)
from .primitives import (
    Constraint,
    ConstraintCategory,
    ConstraintCategoryLiteral,
    ConstraintContext,
    ConstraintSeverity,
    ConstraintSeverityLiteral,
    ConstraintTierLiteral,
    create_bool_constraint_result,
    create_equality_constraint_result,
    create_max_constraint_result,
    create_min_constraint_result,
)
from .primitives import ConstraintResult as PrimitiveConstraintResult
from .primitives import ConstraintTier as PrimitiveConstraintTier

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
    # REPAIR mode and constraint proof generation (new)
    "ConstraintProofDocument",
    "RepairAction",
    "RepairEngine",
    "RepairResult",
    "generate_constraint_proof",
    "repair_spec_tiered",
    "write_constraint_proof",
    # GPU-accelerated batch filtering (new, updated for CP-4.1)
    "BatchFilterResult",
    "FabProfiles",
    "FamilyF1ParameterSpace",
    "GPUConstraintFilter",
    "ParameterMapping",
    "RepairMeta",
    "batch_filter",
    "is_gpu_available",
    # Constraint primitives (Section 13.3.1)
    "Constraint",
    "ConstraintCategory",
    "ConstraintCategoryLiteral",
    "ConstraintContext",
    "ConstraintSeverity",
    "ConstraintSeverityLiteral",
    "ConstraintTierLiteral",
    "PrimitiveConstraintResult",
    "PrimitiveConstraintTier",
    "create_bool_constraint_result",
    "create_equality_constraint_result",
    "create_max_constraint_result",
    "create_min_constraint_result",
]
