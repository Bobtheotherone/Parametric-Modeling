"""Spec consumption tracking and enforcement for CouponSpec resolution.

This module provides functions to:
1. Collect provided paths from a CouponSpec
2. Define expected paths per coupon family
3. Build a SpecConsumption summary
4. Enforce strict mode (fail on unused provided or unconsumed expected)

Satisfies REQ-M1-001:
    - The generator MUST track and emit spec consumption (consumed paths,
      expected paths, unused provided paths)
    - MUST fail in strict mode if any provided field is unused or any
      expected field is unconsumed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from formula_foundry.resolve.expectations import (
    expected_paths_for_family,
    is_optional_path,
)
from formula_foundry.resolve.types import SpecConsumption

if TYPE_CHECKING:
    from formula_foundry.coupongen.spec import CouponSpec


class SpecConsumptionError(Exception):
    """Raised when spec consumption validation fails in strict mode."""

    def __init__(
        self,
        message: str,
        unused_provided: frozenset[str],
        unconsumed_expected: frozenset[str],
    ) -> None:
        super().__init__(message)
        self.unused_provided = unused_provided
        self.unconsumed_expected = unconsumed_expected


def get_expected_paths(coupon_family: str) -> frozenset[str]:
    """Get the expected paths for a given coupon family.

    Args:
        coupon_family: The coupon family identifier (e.g., "F0", "F1").

    Returns:
        Frozenset of expected dot-delimited paths.

    """
    return expected_paths_for_family(coupon_family)


def collect_provided_paths(spec: CouponSpec) -> frozenset[str]:
    """Collect all provided paths from a CouponSpec.

    This walks the spec's model_dump output and collects all leaf paths
    that have non-None values.

    Args:
        spec: The CouponSpec to analyze.

    Returns:
        Frozenset of dot-delimited paths for all provided values.
    """
    payload = spec.model_dump(mode="json")
    paths: set[str] = set()
    _walk_and_collect(payload, paths, prefix="")
    return frozenset(paths)


def _walk_and_collect(
    obj: Any,
    paths: set[str],
    *,
    prefix: str,
) -> None:
    """Recursively walk an object and collect leaf paths.

    Args:
        obj: The object to walk (dict, list, or scalar).
        paths: Set to add discovered paths to.
        prefix: Current path prefix (dot-delimited).
    """
    if obj is None:
        return

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (Mapping, list)):
                _walk_and_collect(value, paths, prefix=path)
            elif value is not None:
                paths.add(path)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            path = f"{prefix}[{idx}]"
            if isinstance(item, (Mapping, list)):
                _walk_and_collect(item, paths, prefix=path)
            elif item is not None:
                paths.add(path)


def get_consumed_paths(spec: CouponSpec) -> frozenset[str]:
    """Determine which paths are actually consumed during resolution.

    This returns the union of expected paths (for the family) and any
    optional paths that are actually provided in the spec.

    Args:
        spec: The CouponSpec being resolved.

    Returns:
        Frozenset of paths that are consumed during resolution.
    """
    family = spec.coupon_family
    expected = get_expected_paths(family)
    provided = collect_provided_paths(spec)

    # Consumed = expected paths that are provided + optional paths that are provided
    consumed: set[str] = set()

    # Add expected paths that are provided
    for path in expected:
        if path in provided or _is_path_prefix_in(path, provided):
            consumed.add(path)

    # Add optional paths that are provided
    for path in provided:
        if _matches_optional_pattern(path):
            consumed.add(path)

    # Special handling for dynamic keys (stackup thicknesses, antipads, etc.)
    consumed.update(_collect_dynamic_consumed_paths(spec, provided))

    return frozenset(consumed)


def _is_path_prefix_in(path: str, paths: frozenset[str]) -> bool:
    """Check if a path or any of its descendants exists in paths."""
    prefix = path + "."
    list_prefix = path + "["
    return any(p == path or p.startswith(prefix) or p.startswith(list_prefix) for p in paths)


def _matches_optional_pattern(path: str) -> bool:
    """Check if a path matches any optional pattern."""
    return is_optional_path(path)


def _collect_dynamic_consumed_paths(
    spec: CouponSpec,
    provided: frozenset[str],
) -> set[str]:
    """Collect consumed paths for dynamic/nested structures.

    Args:
        spec: The CouponSpec being resolved.
        provided: The set of provided paths.

    Returns:
        Set of dynamic paths that should be marked as consumed.
    """
    consumed: set[str] = set()

    # Stackup thicknesses (all provided thickness keys are consumed)
    for key in spec.stackup.thicknesses_nm:
        path = f"stackup.thicknesses_nm.{key}"
        if path in provided:
            consumed.add(path)

    # Ground via fence (if enabled, all fence paths are consumed)
    fence = spec.transmission_line.ground_via_fence
    if fence is not None:
        fence_paths = [
            "transmission_line.ground_via_fence.enabled",
            "transmission_line.ground_via_fence.pitch_nm",
            "transmission_line.ground_via_fence.offset_from_gap_nm",
            "transmission_line.ground_via_fence.via.drill_nm",
            "transmission_line.ground_via_fence.via.diameter_nm",
        ]
        for path in fence_paths:
            if path in provided:
                consumed.add(path)

    # Discontinuity sub-structures (if present)
    if spec.discontinuity is not None:
        disc = spec.discontinuity

        # Antipads (dynamic keys)
        for key in disc.antipads:
            for subkey in ["shape", "rx_nm", "ry_nm", "corner_nm", "r_nm"]:
                path = f"discontinuity.antipads.{key}.{subkey}"
                if path in provided:
                    consumed.add(path)

        # Plane cutouts (dynamic keys)
        for key in disc.plane_cutouts:
            for subkey in ["shape", "length_nm", "width_nm", "rotation_deg"]:
                path = f"discontinuity.plane_cutouts.{key}.{subkey}"
                if path in provided:
                    consumed.add(path)

        # Return vias (if present)
        if disc.return_vias is not None:
            return_via_paths = [
                "discontinuity.return_vias.pattern",
                "discontinuity.return_vias.count",
                "discontinuity.return_vias.radius_nm",
                "discontinuity.return_vias.via.drill_nm",
                "discontinuity.return_vias.via.diameter_nm",
            ]
            for path in return_via_paths:
                if path in provided:
                    consumed.add(path)

    return consumed


def build_spec_consumption(spec: CouponSpec) -> SpecConsumption:
    """Build a SpecConsumption summary for a CouponSpec.

    This analyzes the spec and returns a SpecConsumption model with:
    - consumed_paths: Paths that are consumed during resolution
    - expected_paths: Paths that are expected for the coupon family
    - provided_paths: Paths that are provided in the input spec

    Args:
        spec: The CouponSpec to analyze.

    Returns:
        SpecConsumption summary.
    """
    expected = get_expected_paths(spec.coupon_family)
    provided = collect_provided_paths(spec)
    consumed = get_consumed_paths(spec)

    return SpecConsumption(
        consumed_paths=consumed,
        expected_paths=expected,
        provided_paths=provided,
    )


def enforce_spec_consumption(consumption: SpecConsumption) -> None:
    """Enforce strict mode: fail if any paths are unused or unconsumed.

    This raises SpecConsumptionError if:
    - Any provided paths are not consumed (unused provided)
    - Any expected paths are not consumed (unconsumed expected)

    Args:
        consumption: The SpecConsumption summary to validate.

    Raises:
        SpecConsumptionError: If consumption validation fails.
    """
    unused = consumption.unused_provided_paths
    unconsumed = consumption.unconsumed_expected_paths

    if unused or unconsumed:
        parts = []
        if unused:
            parts.append(f"unused provided paths: {sorted(unused)}")
        if unconsumed:
            parts.append(f"unconsumed expected paths: {sorted(unconsumed)}")

        message = "Spec consumption validation failed: " + "; ".join(parts)
        raise SpecConsumptionError(message, unused, unconsumed)
