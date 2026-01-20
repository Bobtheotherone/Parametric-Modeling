from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from formula_foundry.coupongen.resolve import ResolvedDesign, design_hash as compute_design_hash
from formula_foundry.coupongen.units import LengthNM
from formula_foundry.substrate import canonical_json_dumps

_LAYER_THICKNESS_RE = re.compile(r"^L(\d+)_to_L(\d+)$")


class _SpecBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoardOutlineSpec(_SpecBase):
    width_nm: LengthNM
    length_nm: LengthNM
    corner_radius_nm: LengthNM


class StackupMaterialsSpec(_SpecBase):
    er: float = Field(..., gt=0)
    loss_tangent: float = Field(..., ge=0)


class StackupSpec(_SpecBase):
    copper_layers: int = Field(..., ge=1)
    thicknesses_nm: dict[str, LengthNM]
    materials: StackupMaterialsSpec


class LayerSpec(_SpecBase):
    id: str = Field(..., min_length=1)
    z_nm: LengthNM
    role: Literal["signal", "ground"]


class TransmissionLineSpec(_SpecBase):
    type: str = Field(..., min_length=1)
    layer: str = Field(..., min_length=1)
    w_nm: LengthNM
    gap_nm: LengthNM
    length_left_nm: LengthNM
    length_right_nm: LengthNM


class DiscontinuitySpec(_SpecBase):
    type: str = Field(..., min_length=1)
    parameters_nm: dict[str, LengthNM] = Field(default_factory=dict)


class GeometrySpec(_SpecBase):
    schema_version: int = Field(1, ge=1)
    design_hash: str = Field(..., min_length=1)
    coupon_family: str = Field(..., min_length=1)
    units: Literal["nm"] = "nm"
    origin: str = Field("EDGE_L_CENTER", min_length=1)
    board: BoardOutlineSpec
    stackup: StackupSpec
    layers: list[LayerSpec] = Field(..., min_length=1)
    transmission_line: TransmissionLineSpec
    discontinuity: DiscontinuitySpec | None = None
    parameters_nm: dict[str, int] = Field(default_factory=dict)
    derived_features: dict[str, int] = Field(default_factory=dict)
    dimensionless_groups: dict[str, float] = Field(default_factory=dict)


def build_geometry_spec(
    resolved: ResolvedDesign,
    manifest: Mapping[str, Any],
    *,
    signal_layer_id: str = "L1",
    transmission_line_type: str = "CPWG",
    transmission_line_layer: str = "F.Cu",
    discontinuity_type: str = "VIA_TRANSITION",
) -> GeometrySpec:
    params = resolved.parameters_nm
    board = BoardOutlineSpec(
        width_nm=_require_param(params, "board.outline.width_nm"),
        length_nm=_require_param(params, "board.outline.length_nm"),
        corner_radius_nm=_require_param(params, "board.outline.corner_radius_nm"),
    )
    stackup = _load_stackup(manifest)
    layers = _build_layers(stackup, signal_layer_id=signal_layer_id)
    transmission_line = TransmissionLineSpec(
        type=transmission_line_type,
        layer=transmission_line_layer,
        w_nm=_require_param(params, "transmission_line.w_nm"),
        gap_nm=_require_param(params, "transmission_line.gap_nm"),
        length_left_nm=_require_param(params, "transmission_line.length_left_nm"),
        length_right_nm=_require_param(params, "transmission_line.length_right_nm"),
    )
    discontinuity_params = _extract_discontinuity_params(params)
    discontinuity = None
    if discontinuity_params:
        discontinuity = DiscontinuitySpec(
            type=discontinuity_type,
            parameters_nm=discontinuity_params,
        )
    design_hash_value = _resolve_design_hash(resolved, manifest)
    coupon_family = _resolve_coupon_family(resolved, manifest)
    return GeometrySpec(
        design_hash=design_hash_value,
        coupon_family=coupon_family,
        board=board,
        stackup=stackup,
        layers=layers,
        transmission_line=transmission_line,
        discontinuity=discontinuity,
        parameters_nm=dict(resolved.parameters_nm),
        derived_features=dict(resolved.derived_features),
        dimensionless_groups=dict(resolved.dimensionless_groups),
    )


def layer_positions_nm(stackup: StackupSpec) -> dict[str, int]:
    entries = _layer_thickness_entries(stackup)
    positions: dict[str, int] = {"L1": 0}
    current = 0
    for start, end, thickness in entries:
        current += thickness
        positions[f"L{end}"] = current
    return positions


def geometry_canonical_json(geometry: GeometrySpec) -> str:
    payload = geometry.model_dump(mode="json")
    return canonical_json_dumps(payload)


def write_geometry_spec(path: Path, geometry: GeometrySpec) -> None:
    text = geometry_canonical_json(geometry)
    path.write_text(f"{text}\n", encoding="utf-8")


def _require_param(params: Mapping[str, int], key: str) -> int:
    if key not in params:
        raise KeyError(f"Resolved design missing required parameter: {key}")
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Resolved design parameter {key} must be an integer nanometer value.")
    return value


def _load_stackup(manifest: Mapping[str, Any]) -> StackupSpec:
    stackup = manifest.get("stackup")
    if not isinstance(stackup, Mapping):
        raise ValueError("Manifest stackup must be a mapping.")
    return StackupSpec.model_validate(stackup)


def _resolve_design_hash(resolved: ResolvedDesign, manifest: Mapping[str, Any]) -> str:
    design_hash_value = manifest.get("design_hash")
    if isinstance(design_hash_value, str) and design_hash_value:
        return design_hash_value
    return compute_design_hash(resolved)


def _resolve_coupon_family(resolved: ResolvedDesign, manifest: Mapping[str, Any]) -> str:
    coupon_family = manifest.get("coupon_family")
    if isinstance(coupon_family, str) and coupon_family:
        return coupon_family
    return resolved.coupon_family


def _extract_discontinuity_params(params: Mapping[str, int]) -> dict[str, int]:
    extracted: dict[str, int] = {}
    prefix = "discontinuity."
    for key, value in params.items():
        if key.startswith(prefix):
            extracted[key[len(prefix) :]] = value
    return extracted


def _layer_thickness_entries(stackup: StackupSpec) -> list[tuple[int, int, int]]:
    entries: list[tuple[int, int, int]] = []
    for key, value in stackup.thicknesses_nm.items():
        match = _LAYER_THICKNESS_RE.match(key)
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2))
        if end != start + 1:
            raise ValueError("Stackup thickness keys must reference consecutive layers.")
        entries.append((start, end, int(value)))
    if not entries:
        raise ValueError("Stackup thicknesses_nm must include L1_to_L2 style entries.")
    entries.sort(key=lambda item: item[0])
    _validate_layer_entries(entries, stackup.copper_layers)
    return entries


def _validate_layer_entries(entries: list[tuple[int, int, int]], copper_layers: int) -> None:
    if len(entries) != max(copper_layers - 1, 0):
        raise ValueError("Stackup thickness count must match copper layer count.")
    expected = 1
    for start, end, _ in entries:
        if start != expected or end != start + 1:
            raise ValueError("Stackup thickness keys must be contiguous from L1.")
        expected += 1
    if entries and entries[-1][1] != copper_layers:
        raise ValueError("Stackup thickness keys must end at the last copper layer.")


def _build_layers(stackup: StackupSpec, *, signal_layer_id: str) -> list[LayerSpec]:
    positions = layer_positions_nm(stackup)
    if signal_layer_id not in positions:
        raise ValueError(f"Signal layer {signal_layer_id} is not defined in stackup.")
    layers: list[LayerSpec] = []
    for index in range(1, stackup.copper_layers + 1):
        layer_id = f"L{index}"
        role = "signal" if layer_id == signal_layer_id else "ground"
        layers.append(LayerSpec(id=layer_id, z_nm=positions[layer_id], role=role))
    return layers
