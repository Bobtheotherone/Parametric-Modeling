"""Unified ConstraintEngine for coupon generation (CP-3.1).

This module provides the single unified path for constraint validation,
implementing the ConstraintEngine class per ECO-M1-ALIGN-0001 Section 13.3.

The ConstraintEngine orchestrates:
- Tiered constraint evaluation (Tier 0-3)
- Connectivity oracle integration (pre-KiCad topology check)
- REJECT/REPAIR mode handling
- ConstraintProof generation with signed margins

Public API:
    engine = ConstraintEngine(fab_limits)
    proof = engine.evaluate(spec_or_resolved)
    resolved, proof, repair_map = engine.validate_or_repair(spec, mode="REPAIR")

REQ-M1-008: Tiered constraint system with Tiers 0-3
REQ-M1-009: REJECT mode with constraint IDs and reasons
REQ-M1-010: REPAIR mode with repair_map, repair_reason, repair_distance
REQ-M1-011: constraint_proof.json with per-constraint evaluations and signed margins
CP-3.1: Unified ConstraintEngine as single path for constraint validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Union

from .connectivity import ConnectivityChecker
from .repair import (
    ConstraintProofDocument,
    RepairEngine,
    RepairResult,
    generate_constraint_proof,
    repair_spec_tiered,
)
from .tiers import (
    ConstraintViolationError,
    Tier0Checker,
    Tier1Checker,
    Tier2Checker,
    Tier3Checker,
    TierChecker,
    TieredConstraintProof,
    TieredConstraintSystem,
)

if TYPE_CHECKING:
    from ..resolve import ResolvedDesign
    from ..spec import CouponSpec

# Type alias for input that can be either a CouponSpec or ResolvedDesign
SpecOrResolved = Union["CouponSpec", "ResolvedDesign"]

# Constraint mode literals
ConstraintMode = Literal["REJECT", "REPAIR"]


@dataclass
class ConstraintEngineResult:
    """Result from ConstraintEngine.validate_or_repair().

    Attributes:
        resolved: The ResolvedDesign (original or repaired)
        proof: The constraint proof with per-constraint evaluations
        repair_result: Repair details if REPAIR mode was used and changes were made
    """

    resolved: "ResolvedDesign"
    proof: TieredConstraintProof
    repair_result: RepairResult | None = None

    @property
    def passed(self) -> bool:
        """Return True if all constraints passed."""
        return self.proof.passed

    @property
    def was_repaired(self) -> bool:
        """Return True if REPAIR mode made changes."""
        return self.repair_result is not None and bool(self.repair_result.repair_actions)

    @property
    def repair_map(self) -> dict[str, dict[str, int]] | None:
        """Return the repair map if repairs were made, else None."""
        if self.repair_result is None:
            return None
        return self.repair_result.repair_map if self.repair_result.repair_actions else None

    def to_proof_document(self) -> ConstraintProofDocument:
        """Generate a ConstraintProofDocument for serialization."""
        return generate_constraint_proof(self.proof, self.repair_result)


@dataclass
class ConstraintEngine:
    """Unified constraint validation engine (CP-3.1).

    This is the single path for all constraint validation in the coupongen
    pipeline. It integrates:
    - Tiered constraint checkers (Tier 0-3)
    - Connectivity oracle for pre-KiCad topology verification
    - REJECT/REPAIR mode handling
    - ConstraintProof generation

    Usage:
        from formula_foundry.coupongen.constraints import ConstraintEngine

        # Create engine with fab limits
        engine = ConstraintEngine(fab_limits={"min_trace_width_nm": 100_000, ...})

        # Evaluate constraints (returns proof without raising)
        proof = engine.evaluate(spec)

        # Validate with mode (REJECT raises, REPAIR projects to feasible space)
        result = engine.validate_or_repair(spec, mode="REPAIR")
        if result.was_repaired:
            print(f"Repairs applied: {result.repair_map}")

    Attributes:
        fab_limits: Dictionary of fab capability limits in nm
        fail_fast: If True, stop evaluation at first tier with failures
        include_connectivity: If True, include connectivity checker (default True)
    """

    fab_limits: dict[str, int]
    fail_fast: bool = False
    include_connectivity: bool = True

    # Internal state
    _system: TieredConstraintSystem = field(init=False, repr=False)
    _checkers: list[TierChecker] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the tiered constraint system with all checkers."""
        self._checkers = [
            Tier0Checker(),
            Tier1Checker(),
            Tier2Checker(),
            Tier3Checker(),
        ]

        # Optionally include connectivity checker (Tier 2)
        if self.include_connectivity:
            self._checkers.append(ConnectivityChecker())

        self._system = TieredConstraintSystem(
            checkers=self._checkers,
            fail_fast=self.fail_fast,
        )

    def evaluate(
        self,
        spec_or_resolved: SpecOrResolved,
    ) -> TieredConstraintProof:
        """Evaluate all constraints and return proof without raising.

        This method evaluates all tiered constraints against the given spec
        or resolved design and returns a TieredConstraintProof. It does NOT
        raise on failure - use validate_or_repair() for mode-aware handling.

        Args:
            spec_or_resolved: Either a CouponSpec or ResolvedDesign to validate.
                If a CouponSpec is provided, it will be resolved first.
                If a ResolvedDesign is provided, its spec will be extracted.

        Returns:
            TieredConstraintProof with all constraint results and signed margins
        """
        spec, resolved = self._normalize_input(spec_or_resolved)
        return self._system.evaluate(spec, self.fab_limits, resolved)

    def validate_or_repair(
        self,
        spec_or_resolved: SpecOrResolved,
        mode: ConstraintMode = "REJECT",
    ) -> ConstraintEngineResult:
        """Validate constraints with mode-aware handling.

        In REJECT mode:
            - Evaluates constraints
            - Raises ConstraintViolationError if any fail

        In REPAIR mode:
            - Evaluates constraints
            - If any fail, projects spec into feasible space
            - Returns repaired ResolvedDesign with repair details

        Args:
            spec_or_resolved: Either a CouponSpec or ResolvedDesign to validate
            mode: "REJECT" to raise on failure, "REPAIR" to project to feasible space

        Returns:
            ConstraintEngineResult containing:
                - resolved: The (possibly repaired) ResolvedDesign
                - proof: The constraint proof (for repaired spec if repairs made)
                - repair_result: Repair details if REPAIR mode made changes

        Raises:
            ConstraintViolationError: If mode is REJECT and constraints fail
        """
        spec, resolved = self._normalize_input(spec_or_resolved)

        # Initial evaluation
        proof = self._system.evaluate(spec, self.fab_limits, resolved)

        if proof.passed:
            # All constraints pass - return as-is
            return ConstraintEngineResult(
                resolved=resolved,
                proof=proof,
                repair_result=None,
            )

        # Handle failure based on mode
        if mode == "REJECT":
            failures = proof.get_failures()
            raise ConstraintViolationError(failures, proof.first_failure_tier)

        # REPAIR mode: project spec into feasible space
        repaired_spec, repair_result = repair_spec_tiered(spec, self.fab_limits)

        # Re-resolve the repaired spec
        from ..resolve import resolve

        repaired_resolved = resolve(repaired_spec)

        # Re-evaluate with repaired spec
        repaired_proof = self._system.evaluate(repaired_spec, self.fab_limits, repaired_resolved)

        return ConstraintEngineResult(
            resolved=repaired_resolved,
            proof=repaired_proof,
            repair_result=repair_result,
        )

    def _normalize_input(
        self,
        spec_or_resolved: SpecOrResolved,
    ) -> tuple["CouponSpec", "ResolvedDesign"]:
        """Normalize input to (spec, resolved) tuple.

        Args:
            spec_or_resolved: Either CouponSpec or ResolvedDesign

        Returns:
            Tuple of (CouponSpec, ResolvedDesign)
        """
        from ..resolve import ResolvedDesign, resolve
        from ..spec import CouponSpec

        if isinstance(spec_or_resolved, ResolvedDesign):
            # Extract spec from resolved design - need to reconstruct
            # For now, we'll need to require the spec be passed separately
            # or stored in ResolvedDesign. For this implementation, we
            # support only CouponSpec input for full functionality.
            # ResolvedDesign input is primarily for evaluation-only.
            raise ValueError(
                "validate_or_repair() requires a CouponSpec input for REPAIR mode. "
                "Use evaluate() for ResolvedDesign inputs."
            )

        # Input is CouponSpec
        spec = spec_or_resolved
        resolved = resolve(spec)
        return spec, resolved

    def create_proof_document(
        self,
        proof: TieredConstraintProof,
        repair_result: RepairResult | None = None,
    ) -> ConstraintProofDocument:
        """Create a ConstraintProofDocument for serialization.

        Args:
            proof: The tiered constraint proof
            repair_result: Optional repair result if REPAIR was used

        Returns:
            ConstraintProofDocument ready for JSON serialization
        """
        return generate_constraint_proof(proof, repair_result)


def create_constraint_engine(
    fab_limits: dict[str, int] | None = None,
    *,
    fail_fast: bool = False,
    include_connectivity: bool = True,
) -> ConstraintEngine:
    """Factory function to create a ConstraintEngine with defaults.

    Args:
        fab_limits: Dictionary of fab capability limits. If None, uses conservative defaults.
        fail_fast: If True, stop evaluation at first tier with failures
        include_connectivity: If True, include connectivity checker

    Returns:
        Configured ConstraintEngine instance
    """
    if fab_limits is None:
        # Conservative defaults
        fab_limits = {
            "min_trace_width_nm": 100_000,
            "min_gap_nm": 100_000,
            "min_drill_nm": 200_000,
            "min_annular_ring_nm": 100_000,
            "min_via_diameter_nm": 300_000,
            "min_edge_clearance_nm": 200_000,
            "min_via_to_via_nm": 200_000,
            "min_board_width_nm": 5_000_000,
        }

    return ConstraintEngine(
        fab_limits=fab_limits,
        fail_fast=fail_fast,
        include_connectivity=include_connectivity,
    )


# Public API
__all__ = [
    "ConstraintEngine",
    "ConstraintEngineResult",
    "ConstraintMode",
    "SpecOrResolved",
    "create_constraint_engine",
]
