from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from formula_foundry.resolve.consumption import (
    build_spec_consumption,
    enforce_spec_consumption,
)
from formula_foundry.spec.consumption import SpecConsumption
from formula_foundry.substrate import canonical_json_dumps, sha256_bytes

from .families import FAMILY_F1
from .spec import CouponSpec

if TYPE_CHECKING:
    from .geom.layout import LayoutPlan


class ResolvedDesign(BaseModel):
    """Resolved design with computed geometry and derived parameters.

    This dataclass holds all resolved parameters from a CouponSpec,
    including the computed LayoutPlan which is the single source of truth
    for all geometry. For F1 coupons, length_right_nm is derived from
    the continuity formula to ensure trace segments connect at the
    discontinuity center.

    The layout_plan field is excluded from serialization/hashing as it is
    computed deterministically from the parameters. This ensures the design_hash
    remains stable and only depends on input parameters.

    Satisfies CP-2.4 per ECO-M1-ALIGN-0001.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    schema_version: int = Field(..., ge=1)
    coupon_family: str = Field(..., min_length=1)
    units: Literal["nm"] = "nm"
    parameters_nm: dict[str, int]
    derived_features: dict[str, int]
    dimensionless_groups: dict[str, float]
    # Derived length_right_nm for F1 coupons (ensures continuity per CP-2.2)
    # This IS serialized and included in the design hash
    length_right_nm: int | None = Field(default=None)
    spec_consumption: SpecConsumption | None = Field(default=None, exclude=True)

    # LayoutPlan is the single source of truth for geometry (CP-2.1)
    # Stored as a private attribute, excluded from serialization/hashing
    # as it's computed deterministically from parameters
    _layout_plan: LayoutPlan | None = None

    @property
    def layout_plan(self) -> LayoutPlan | None:
        """Get the computed LayoutPlan (single source of truth for geometry)."""
        return self._layout_plan

    def with_layout_plan(self, layout_plan: LayoutPlan) -> ResolvedDesign:
        """Return a copy with the layout_plan set.

        Since layout_plan is excluded from serialization, this method allows
        setting it after construction while maintaining immutability.
        """
        new_obj = self.model_copy()
        object.__setattr__(new_obj, "_layout_plan", layout_plan)
        return new_obj

    def get_spec_consumption_summary(self) -> dict[str, list[str]] | None:
        """Get the spec consumption summary for manifest emission.

        Returns:
            Dictionary with sorted lists of consumed/expected/unused paths,
            or None if spec_consumption was not computed.

        Satisfies REQ-M1-001 and REQ-M1-013: Spec consumption summary
        emitted in resolved outputs and manifest.
        """
        if self.spec_consumption is None:
            return None
        return self.spec_consumption.to_summary_dict()


def resolve(spec: CouponSpec, *, strict: bool = False) -> ResolvedDesign:
    """Resolve a CouponSpec to a ResolvedDesign with computed geometry.

    This function resolves all parameters from the spec and computes the
    LayoutPlan as the single source of truth for all geometry. For F1
    coupons, length_right_nm is derived from the continuity formula to
    ensure trace segments connect at the discontinuity center.

    The continuity formula for F1 coupons is:
        xD = xL + length_left  (discontinuity position)
        length_right = xR - xD (derived to ensure continuity)

    where xL and xR are the signal pad X positions of the left and right
    connectors.

    Args:
        spec: The coupon specification with all geometry parameters.
        strict: If True, raise on any unused provided or unconsumed expected paths.

    Returns:
        ResolvedDesign with all parameters resolved and LayoutPlan computed.

    Satisfies CP-2.4 per ECO-M1-ALIGN-0001.
    """
    from .geom.layout import compute_layout_plan

    payload = spec.model_dump(mode="json")
    parameters_nm = _collect_length_parameters(payload)

    # Create a preliminary ResolvedDesign (without LayoutPlan) for computing
    # derived features that don't depend on the layout
    preliminary_resolved = ResolvedDesign(
        schema_version=spec.schema_version,
        coupon_family=spec.coupon_family,
        parameters_nm=parameters_nm,
        derived_features={},  # Will be computed after LayoutPlan
        dimensionless_groups={},  # Will be computed after LayoutPlan
        length_right_nm=None,
    )

    # Compute LayoutPlan - the single source of truth for geometry (CP-2.1)
    layout_plan = compute_layout_plan(spec, preliminary_resolved)

    # For F1 coupons, derive length_right_nm from the LayoutPlan (CP-2.2)
    length_right_nm: int | None = None
    if spec.coupon_family == FAMILY_F1:
        # Get the right segment from the layout plan
        right_segment = layout_plan.get_segment_by_label("right")
        if right_segment is not None:
            length_right_nm = right_segment.length_nm

    # Now compute derived features with access to LayoutPlan and derived length_right_nm
    derived_features = _build_derived_features(spec, layout_plan, length_right_nm)
    dimensionless_groups = _build_dimensionless_groups(spec)

    spec_consumption = build_spec_consumption(spec)
    if strict:
        enforce_spec_consumption(spec_consumption)

    # Create the final ResolvedDesign with layout_plan attached
    resolved = ResolvedDesign(
        schema_version=spec.schema_version,
        coupon_family=spec.coupon_family,
        parameters_nm=parameters_nm,
        derived_features=derived_features,
        dimensionless_groups=dimensionless_groups,
        length_right_nm=length_right_nm,
        spec_consumption=spec_consumption,
    )

    # Attach the layout_plan (excluded from serialization/hashing)
    return resolved.with_layout_plan(layout_plan)


def resolved_design_canonical_json(resolved: ResolvedDesign) -> str:
    payload = resolved.model_dump(mode="json")
    return canonical_json_dumps(payload)


def design_hash(resolved: ResolvedDesign) -> str:
    canonical = resolved_design_canonical_json(resolved)
    return sha256_bytes(canonical.encode("utf-8"))


def _collect_length_parameters(payload: Mapping[str, Any]) -> dict[str, int]:
    params: dict[str, int] = {}
    _walk_mapping(payload, params, prefix="")
    return params


def _walk_mapping(payload: Mapping[str, Any], params: dict[str, int], *, prefix: str) -> None:
    for key, value in payload.items():
        path = f"{prefix}{key}"
        if key.endswith("_nm"):
            _collect_nm_values(path, value, params)
            continue
        if isinstance(value, Mapping):
            _walk_mapping(value, params, prefix=f"{path}.")
        elif isinstance(value, list):
            _walk_list(value, params, prefix=path)


def _walk_list(items: list[Any], params: dict[str, int], *, prefix: str) -> None:
    for idx, item in enumerate(items):
        path = f"{prefix}[{idx}]"
        if isinstance(item, Mapping):
            _walk_mapping(item, params, prefix=f"{path}.")
        elif isinstance(item, list):
            _walk_list(item, params, prefix=path)


def _collect_nm_values(path: str, value: Any, params: dict[str, int]) -> None:
    if value is None:
        return
    if isinstance(value, Mapping):
        for sub_key, sub_value in value.items():
            if sub_value is None:
                continue
            params[f"{path}.{sub_key}"] = _coerce_int(sub_value)
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            if item is None:
                continue
            params[f"{path}[{idx}]"] = _coerce_int(item)
        return
    params[path] = _coerce_int(value)


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("Length parameters must be integers, not booleans.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError(f"Length parameter must be an integer nanometer value, got {value!r}.")


def _build_derived_features(
    spec: CouponSpec,
    layout_plan: LayoutPlan,
    derived_length_right_nm: int | None,
) -> dict[str, int]:
    """Build derived features using LayoutPlan as the source of truth.

    Combines the comprehensive derived features from formula_foundry.derive
    with LayoutPlan-derived values to ensure geometry math is not duplicated.

    For F1 coupons, uses the derived length_right_nm from the LayoutPlan
    to compute total trace length, ensuring geometry math is not duplicated.

    Satisfies REQ-M1-015: Derived features include CPWG/via/fence/launch-relevant
    features and emit deterministically in manifest.json.

    Args:
        spec: The coupon specification.
        layout_plan: The computed LayoutPlan (single source of truth for geometry).
        derived_length_right_nm: The derived right length for F1 coupons, or None.

    Returns:
        Dictionary of derived features, sorted by key for deterministic output.
    """
    from formula_foundry.derive import compute_derived_features

    # Get comprehensive derived features from the derive module
    derived = compute_derived_features(spec, derived_length_right_nm)

    # Override trace_total_length_nm with LayoutPlan's value (source of truth)
    # This ensures no geometry math is duplicated (CP-2.1)
    derived["trace_total_length_nm"] = layout_plan.total_trace_length_nm

    # Ensure sorted output for deterministic JSON emission
    return dict(sorted(derived.items()))


def _build_dimensionless_groups(spec: CouponSpec) -> dict[str, float]:
    """Build comprehensive dimensionless groups for equation discovery.

    Delegates to formula_foundry.derive.compute_dimensionless_groups which
    provides CPWG/via/fence/launch-relevant dimensionless groups.

    Satisfies REQ-M1-015: Derived groups include CPWG/via/fence/launch-relevant
    dimensionless groups and emit deterministically in manifest.json.

    Args:
        spec: The coupon specification.

    Returns:
        Dictionary of dimensionless groups, sorted by key for deterministic output.
    """
    from formula_foundry.derive import compute_dimensionless_groups

    return compute_dimensionless_groups(spec)
