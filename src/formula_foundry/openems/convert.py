from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

from formula_foundry.coupongen.hashing import coupon_id_from_design_hash
from formula_foundry.coupongen.resolve import ResolvedDesign
from formula_foundry.substrate import canonical_json_dumps

from .geometry import GeometrySpec, StackupSpec, build_geometry_spec
from .spec import (
    ConductorMaterialSpec,
    DielectricMaterialSpec,
    ExcitationSpec,
    FrequencySpec,
    GeometryRefSpec,
    MaterialsSpec,
    OpenEMSToolchainSpec,
    PortSpec,
    SimulationSpec,
    ToolchainSpec,
)
from .toolchain import OpenEMSToolchain, load_openems_toolchain

_COPPER_CONDUCTIVITY = 5.8e7


def build_simulation_spec(
    resolved: ResolvedDesign,
    manifest: Mapping[str, Any],
    *,
    toolchain: OpenEMSToolchain | None = None,
    excitation: ExcitationSpec | None = None,
    frequency: FrequencySpec | None = None,
    signal_layer_id: str = "L1",
    transmission_line_type: str = "CPWG",
    transmission_line_layer: str = "F.Cu",
    discontinuity_type: str = "VIA_TRANSITION",
) -> SimulationSpec:
    geometry = build_geometry_spec(
        resolved,
        manifest,
        signal_layer_id=signal_layer_id,
        transmission_line_type=transmission_line_type,
        transmission_line_layer=transmission_line_layer,
        discontinuity_type=discontinuity_type,
    )
    toolchain = toolchain or load_openems_toolchain()
    return SimulationSpec(
        toolchain=_build_toolchain(toolchain),
        geometry_ref=_build_geometry_ref(manifest, geometry),
        excitation=excitation or _default_excitation(),
        frequency=frequency or _default_frequency(),
        ports=_build_ports(resolved, geometry, signal_layer_id=signal_layer_id),
        materials=_build_materials(geometry.stackup),
    )


def simulation_canonical_json(spec: SimulationSpec) -> str:
    payload = spec.model_dump(mode="json")
    return canonical_json_dumps(payload)


def write_simulation_spec(path: Path, spec: SimulationSpec) -> None:
    text = simulation_canonical_json(spec)
    path.write_text(f"{text}\n", encoding="utf-8")


def _build_toolchain(toolchain: OpenEMSToolchain) -> ToolchainSpec:
    return ToolchainSpec(
        openems=OpenEMSToolchainSpec(
            version=toolchain.version,
            docker_image=toolchain.docker_image,
        )
    )


def _default_excitation() -> ExcitationSpec:
    return ExcitationSpec(f0_hz=5_000_000_000, fc_hz=10_000_000_000)


def _default_frequency() -> FrequencySpec:
    return FrequencySpec(f_start_hz=1_000_000_000, f_stop_hz=20_000_000_000, n_points=401)


def _build_geometry_ref(
    manifest: Mapping[str, Any],
    geometry: GeometrySpec,
) -> GeometryRefSpec:
    design_hash_value = geometry.design_hash
    coupon_id = _resolve_coupon_id(manifest, design_hash_value)
    return GeometryRefSpec(design_hash=design_hash_value, coupon_id=coupon_id)


def _resolve_coupon_id(manifest: Mapping[str, Any], design_hash_value: str) -> str:
    coupon_id = manifest.get("coupon_id")
    if isinstance(coupon_id, str) and coupon_id:
        return coupon_id
    return coupon_id_from_design_hash(design_hash_value)


def _build_ports(
    resolved: ResolvedDesign,
    geometry: GeometrySpec,
    *,
    signal_layer_id: str,
) -> list[PortSpec]:
    params = resolved.parameters_nm
    left = _extract_connector_position(params, "left")
    right = _extract_connector_position(params, "right")
    if left is None or right is None:
        left, right = _fallback_port_positions(params)
    z_nm = _signal_layer_z(geometry.layers, signal_layer_id)
    ports = [
        PortSpec(
            id="P1",
            position_nm=(left[0], left[1], z_nm),
            direction="x",
            excite=True,
        ),
        PortSpec(
            id="P2",
            position_nm=(right[0], right[1], z_nm),
            direction="-x",
            excite=False,
        ),
    ]
    return ports


def _extract_connector_position(params: Mapping[str, int], side: str) -> tuple[int, int] | None:
    x_key = f"connectors.{side}.position_nm[0]"
    y_key = f"connectors.{side}.position_nm[1]"
    x_value = params.get(x_key)
    y_value = params.get(y_key)
    if x_value is None or y_value is None:
        return None
    return int(x_value), int(y_value)


def _fallback_port_positions(params: Mapping[str, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    left_length = params.get("transmission_line.length_left_nm")
    right_length = params.get("transmission_line.length_right_nm")
    if left_length is None or right_length is None:
        raise KeyError("Transmission line lengths are required to infer port positions.")
    return (-int(left_length), 0), (int(right_length), 0)


def _signal_layer_z(layers: Iterable[Any], signal_layer_id: str) -> int:
    for layer in layers:
        if getattr(layer, "id", None) == signal_layer_id:
            return int(layer.z_nm)
    raise ValueError(f"Signal layer {signal_layer_id} not found in geometry layers.")


def _build_materials(stackup: StackupSpec) -> MaterialsSpec:
    return MaterialsSpec(
        dielectrics=[
            DielectricMaterialSpec(
                id="substrate",
                epsilon_r=stackup.materials.er,
                loss_tangent=stackup.materials.loss_tangent,
            )
        ],
        conductors=[
            ConductorMaterialSpec(
                id="copper",
                conductivity=_COPPER_CONDUCTIVITY,
            )
        ],
    )
