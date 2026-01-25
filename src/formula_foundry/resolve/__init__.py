"""Resolve module for derived features, dimensionless groups, and spec consumption.

This module provides:
- Computation of physics-relevant dimensionless groups for CPWG/via/fence/launch
  structures used in equation discovery (REQ-M1-015)
- Spec consumption tracking to detect unused/unconsumed paths (REQ-M1-001)

Satisfies REQ-M1-001 and REQ-M1-015.
"""

from formula_foundry.resolve.consumption import (
    SpecConsumptionError,
    build_spec_consumption,
    collect_provided_paths,
    enforce_spec_consumption,
    get_consumed_paths,
    get_expected_paths,
)
from formula_foundry.resolve.derived_groups import (
    compute_cpwg_groups,
    compute_fence_groups,
    compute_launch_groups,
    compute_stackup_groups,
    compute_via_groups,
)
from formula_foundry.resolve.types import SpecConsumption

__all__ = [
    # Spec consumption (REQ-M1-001)
    "SpecConsumption",
    "SpecConsumptionError",
    "build_spec_consumption",
    "collect_provided_paths",
    "enforce_spec_consumption",
    "get_consumed_paths",
    "get_expected_paths",
    # Derived groups (REQ-M1-015)
    "compute_cpwg_groups",
    "compute_via_groups",
    "compute_fence_groups",
    "compute_launch_groups",
    "compute_stackup_groups",
]
