from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from formula_foundry.substrate import canonical_json_dumps, sha256_bytes

from .spec import CouponSpec


class ResolvedDesign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(..., ge=1)
    coupon_family: str = Field(..., min_length=1)
    units: Literal["nm"] = "nm"
    parameters_nm: dict[str, int]
    derived_features: dict[str, int]
    dimensionless_groups: dict[str, float]


def resolve(spec: CouponSpec) -> ResolvedDesign:
    payload = spec.model_dump(mode="json")
    parameters_nm = _collect_length_parameters(payload)
    derived_features = _build_derived_features(spec)
    dimensionless_groups = _build_dimensionless_groups(spec)
    return ResolvedDesign(
        schema_version=spec.schema_version,
        coupon_family=spec.coupon_family,
        parameters_nm=parameters_nm,
        derived_features=derived_features,
        dimensionless_groups=dimensionless_groups,
    )


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


def _build_derived_features(spec: CouponSpec) -> dict[str, int]:
    width = int(spec.board.outline.width_nm)
    length = int(spec.board.outline.length_nm)
    trace_left = int(spec.transmission_line.length_left_nm)
    trace_right = int(spec.transmission_line.length_right_nm)
    derived: dict[str, int] = {
        "board_area_nm2": width * length,
        "trace_total_length_nm": trace_left + trace_right,
    }
    if spec.discontinuity is not None:
        pad = int(spec.discontinuity.signal_via.pad_diameter_nm)
        drill = int(spec.discontinuity.signal_via.drill_nm)
        derived["signal_via_annular_ring_nm"] = pad - drill
    return derived


def _build_dimensionless_groups(spec: CouponSpec) -> dict[str, float]:
    width = int(spec.board.outline.width_nm)
    length = int(spec.board.outline.length_nm)
    w_nm = int(spec.transmission_line.w_nm)
    gap_nm = int(spec.transmission_line.gap_nm)
    groups: dict[str, float] = {
        "board_aspect_ratio": _safe_ratio(length, width),
        "cpwg_w_over_gap": _safe_ratio(w_nm, gap_nm),
    }
    if spec.discontinuity is not None:
        pad = int(spec.discontinuity.signal_via.pad_diameter_nm)
        drill = int(spec.discontinuity.signal_via.drill_nm)
        groups["signal_via_pad_over_drill"] = _safe_ratio(pad, drill)
    return groups


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
