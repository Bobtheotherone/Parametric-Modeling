from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Literal

from formula_foundry.coupongen.hashing import coupon_id_from_design_hash
from formula_foundry.coupongen.resolve import ResolvedDesign
from formula_foundry.substrate import canonical_json_dumps

from .geometry import GeometrySpec, StackupSpec, build_geometry_spec
from .ports import (
    DeembedType,
    PortBuilder,
    PortType,
    WaveguidePortSpec,
    build_ports_from_resolved,
    waveguide_port_to_basic_port_spec,
)
from .spec import (
    ConductorMaterialSpec,
    DeembedConfigSpec,
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

# Port type mapping from string to enum
_PORT_TYPE_MAP: dict[str, PortType] = {
    "lumped": PortType.LUMPED,
    "msl": PortType.MSL,
    "waveguide": PortType.WAVEGUIDE,
    "cpwg": PortType.CPWG,
}


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
    port_type: Literal["lumped", "msl", "waveguide", "cpwg"] = "waveguide",
    include_deembedding: bool = True,
) -> SimulationSpec:
    """Build a SimulationSpec from resolved design and manifest.

    Args:
        resolved: ResolvedDesign from M1 coupongen.
        manifest: Manifest dictionary with stackup and metadata.
        toolchain: OpenEMS toolchain specification. Defaults to auto-detect.
        excitation: Excitation specification. Defaults to Gaussian 5-10GHz.
        frequency: Frequency sweep specification. Defaults to 1-20GHz.
        signal_layer_id: Signal layer identifier (default "L1").
        transmission_line_type: Transmission line type (default "CPWG").
        transmission_line_layer: KiCad layer name for signal (default "F.Cu").
        discontinuity_type: Discontinuity type (default "VIA_TRANSITION").
        port_type: Port type for S-parameter extraction (default "waveguide").
        include_deembedding: Whether to include de-embedding configuration.

    Returns:
        Complete SimulationSpec ready for openEMS execution.
    """
    geometry = build_geometry_spec(
        resolved,
        manifest,
        signal_layer_id=signal_layer_id,
        transmission_line_type=transmission_line_type,
        transmission_line_layer=transmission_line_layer,
        discontinuity_type=discontinuity_type,
    )
    toolchain = toolchain or load_openems_toolchain()

    # Build ports using the new port builder with waveguide support
    ports = _build_ports_enhanced(
        resolved,
        geometry,
        signal_layer_id=signal_layer_id,
        port_type=port_type,
        include_deembedding=include_deembedding,
    )

    return SimulationSpec(
        toolchain=_build_toolchain(toolchain),
        geometry_ref=_build_geometry_ref(manifest, geometry),
        excitation=excitation or _default_excitation(),
        frequency=frequency or _default_frequency(),
        ports=ports,
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


def _build_ports_enhanced(
    resolved: ResolvedDesign,
    geometry: GeometrySpec,
    *,
    signal_layer_id: str,
    port_type: Literal["lumped", "msl", "waveguide", "cpwg"] = "waveguide",
    include_deembedding: bool = True,
) -> list[PortSpec]:
    """Build ports using the enhanced port builder with waveguide support.

    This function uses the new ports module to create properly configured
    ports with impedance matching and de-embedding support.

    Args:
        resolved: ResolvedDesign from M1 coupongen.
        geometry: GeometrySpec for the coupon.
        signal_layer_id: Signal layer identifier.
        port_type: Type of ports to create.
        include_deembedding: Whether to include de-embedding configuration.

    Returns:
        List of PortSpec instances.
    """
    params = resolved.parameters_nm

    # Use the PortBuilder for enhanced port configuration
    builder = PortBuilder(geometry=geometry, signal_layer_id=signal_layer_id)

    # Extract connector positions
    left = _extract_connector_position(params, "left")
    right = _extract_connector_position(params, "right")
    if left is None or right is None:
        left, right = _fallback_port_positions(params)

    # Get de-embedding distance from launch length if available
    deembed_distance: int | None = None
    if include_deembedding:
        launch_length = params.get("launch.length_nm")
        if launch_length is not None:
            deembed_distance = int(launch_length)

    # Map string port type to enum
    port_type_enum = _PORT_TYPE_MAP.get(port_type, PortType.WAVEGUIDE)

    # Build waveguide ports
    waveguide_ports = builder.build_transmission_line_ports(
        left,
        right,
        port_type=port_type_enum,
        deembed_distance_nm=deembed_distance,
    )

    # Convert to PortSpec instances for SimulationSpec
    return [_waveguide_to_port_spec(wp) for wp in waveguide_ports]


def _waveguide_to_port_spec(wp: WaveguidePortSpec) -> PortSpec:
    """Convert WaveguidePortSpec to PortSpec.

    Maps the enhanced waveguide port configuration to the SimulationSpec
    PortSpec model.

    Args:
        wp: WaveguidePortSpec instance.

    Returns:
        PortSpec instance.
    """
    # Build de-embedding config
    deembed_config = DeembedConfigSpec(enabled=False)
    if wp.deembed.method != DeembedType.NONE:
        deembed_config = DeembedConfigSpec(
            enabled=True,
            distance_nm=wp.deembed.distance_nm,
            epsilon_r_eff=wp.deembed.epsilon_r_eff,
        )

    # Get geometry dimensions
    width_nm: int | None = None
    height_nm: int | None = None
    signal_width_nm: int | None = None
    gap_nm: int | None = None
    if wp.geometry is not None:
        width_nm = wp.geometry.width_nm
        height_nm = wp.geometry.height_nm
        signal_width_nm = wp.geometry.signal_width_nm
        gap_nm = wp.geometry.gap_nm

    return PortSpec(
        id=wp.id,
        type=wp.port_type.value,
        impedance_ohm=wp.impedance.z0_ohm,
        excite=wp.excite,
        position_nm=wp.position_nm,
        direction=wp.direction,
        width_nm=width_nm,
        height_nm=height_nm,
        signal_width_nm=signal_width_nm,
        gap_nm=gap_nm,
        match_to_line=wp.impedance.match_to_line,
        calculated_z0_ohm=wp.impedance.calculated_z0_ohm,
        deembed=deembed_config,
        excite_weight=wp.excite_weight,
        polarization=wp.polarization,
    )


def _build_ports(
    resolved: ResolvedDesign,
    geometry: GeometrySpec,
    *,
    signal_layer_id: str,
) -> list[PortSpec]:
    """Legacy port builder for backward compatibility.

    This function creates simple lumped ports without waveguide features.
    Use _build_ports_enhanced for full waveguide support.

    Args:
        resolved: ResolvedDesign from M1 coupongen.
        geometry: GeometrySpec for the coupon.
        signal_layer_id: Signal layer identifier.

    Returns:
        List of PortSpec instances.
    """
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
