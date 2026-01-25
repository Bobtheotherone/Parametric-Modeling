"""Resolve module for derived features and dimensionless groups.

This module provides the computation of physics-relevant dimensionless groups
for CPWG/via/fence/launch structures used in equation discovery.

Satisfies REQ-M1-015.
"""

from formula_foundry.resolve.derived_groups import (
    compute_cpwg_groups,
    compute_fence_groups,
    compute_launch_groups,
    compute_stackup_groups,
    compute_via_groups,
)

__all__ = [
    "compute_cpwg_groups",
    "compute_via_groups",
    "compute_fence_groups",
    "compute_launch_groups",
    "compute_stackup_groups",
]
