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

from formula_foundry.spec.consumption import SpecConsumption

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


# Expected paths for F0 family (simple through-line)
_EXPECTED_PATHS_F0 = frozenset(
    {
        "schema_version",
        "coupon_family",
        "units",
        "toolchain.kicad.version",
        "toolchain.kicad.docker_image",
        "fab_profile.id",
        "stackup.copper_layers",
        "stackup.materials.er",
        "stackup.materials.loss_tangent",
        "board.outline.width_nm",
        "board.outline.length_nm",
        "board.outline.corner_radius_nm",
        "board.origin.mode",
        "board.text.coupon_id",
        "board.text.include_manifest_hash",
        "connectors.left.footprint",
        "connectors.left.position_nm",
        "connectors.left.rotation_deg",
        "connectors.right.footprint",
        "connectors.right.position_nm",
        "connectors.right.rotation_deg",
        "transmission_line.type",
        "transmission_line.layer",
        "transmission_line.w_nm",
        "transmission_line.gap_nm",
        "transmission_line.length_left_nm",
        "transmission_line.length_right_nm",
        "constraints.mode",
        "constraints.drc.must_pass",
        "constraints.drc.severity",
        "constraints.symmetry.enforce",
        "constraints.allow_unconnected_copper",
        "export.gerbers.enabled",
        "export.gerbers.format",
        "export.drill.enabled",
        "export.drill.format",
        "export.outputs_dir",
    }
)

# Expected paths for F1 family (via discontinuity)
# Note: Use explicit parentheses to ensure correct operator precedence
# (| has higher precedence than - for frozenset)
_EXPECTED_PATHS_F1 = (
    _EXPECTED_PATHS_F0
    | frozenset(
        {
            # Discontinuity is required for F1
            "discontinuity.type",
            "discontinuity.signal_via.drill_nm",
            "discontinuity.signal_via.diameter_nm",
            "discontinuity.signal_via.pad_diameter_nm",
            # Return vias are optional but if present, these are expected
            "discontinuity.return_vias.pattern",
            "discontinuity.return_vias.count",
            "discontinuity.return_vias.radius_nm",
            "discontinuity.return_vias.via.drill_nm",
            "discontinuity.return_vias.via.diameter_nm",
        }
    )
) - frozenset(
    {
        # F1 derives length_right_nm, so it's not expected to be provided
        "transmission_line.length_right_nm",
    }
)

# Optional paths that, if provided, are consumed
_OPTIONAL_PATHS_COMMON = frozenset(
    {
        # Ground via fence (optional for both families)
        "transmission_line.ground_via_fence.enabled",
        "transmission_line.ground_via_fence.pitch_nm",
        "transmission_line.ground_via_fence.offset_from_gap_nm",
        "transmission_line.ground_via_fence.via.drill_nm",
        "transmission_line.ground_via_fence.via.diameter_nm",
        # Stackup thicknesses (dynamic keys)
        "stackup.thicknesses_nm.core",
        "stackup.thicknesses_nm.prepreg",
        "stackup.thicknesses_nm.copper",
        "stackup.thicknesses_nm.soldermask",
        # Fab profile overrides
        "fab_profile.overrides",
        # Antipads and plane cutouts (dynamic keys)
        "discontinuity.antipads",
        "discontinuity.plane_cutouts",
    }
)


def get_expected_paths(coupon_family: str) -> frozenset[str]:
    """Get the expected paths for a given coupon family.

    Args:
        coupon_family: The coupon family identifier (e.g., "F0", "F1").

    Returns:
        Frozenset of expected dot-delimited paths.

    Raises:
        ValueError: If the coupon family is not recognized.
    """
    if coupon_family == "F0":
        return _EXPECTED_PATHS_F0
    elif coupon_family == "F1":
        return _EXPECTED_PATHS_F1
    else:
        # Unknown family - use F0 as base and log warning
        return _EXPECTED_PATHS_F0


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
    for p in paths:
        if p == path or p.startswith(prefix):
            return True
    return False


def _matches_optional_pattern(path: str) -> bool:
    """Check if a path matches any optional pattern."""
    # Check exact match
    if path in _OPTIONAL_PATHS_COMMON:
        return True

    # Check prefix patterns for dynamic keys
    dynamic_prefixes = [
        "stackup.thicknesses_nm.",
        "fab_profile.overrides.",
        "discontinuity.antipads.",
        "discontinuity.plane_cutouts.",
    ]
    for prefix in dynamic_prefixes:
        if path.startswith(prefix):
            return True

    return False


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
