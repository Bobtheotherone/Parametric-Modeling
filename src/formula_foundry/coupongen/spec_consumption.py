"""Spec consumption tracking for coupongen.

This module re-exports the spec consumption functionality from the core modules
for convenience when working within the coupongen package.

Satisfies REQ-M1-001:
    - The generator MUST track and emit spec consumption (consumed paths,
      expected paths, unused provided paths)
    - MUST fail in strict mode if any provided field is unused or any
      expected field is unconsumed.

For implementation details, see:
    - formula_foundry.resolve.types: SpecConsumption model
    - formula_foundry.resolve.consumption: Build and enforcement functions
"""

from formula_foundry.resolve.consumption import (
    SpecConsumptionError,
    build_spec_consumption,
    collect_provided_paths,
    enforce_spec_consumption,
    get_consumed_paths,
    get_expected_paths,
)
from formula_foundry.resolve.types import SpecConsumption

__all__ = [
    "SpecConsumption",
    "SpecConsumptionError",
    "build_spec_consumption",
    "collect_provided_paths",
    "enforce_spec_consumption",
    "get_consumed_paths",
    "get_expected_paths",
]
