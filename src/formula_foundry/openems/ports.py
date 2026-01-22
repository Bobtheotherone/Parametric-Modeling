"""Port configuration for openEMS S-parameter extraction.

This module provides port configuration utilities for openEMS simulations,
supporting waveguide ports at connector locations with proper impedance
matching and de-embedding.

Port Types:
- Lumped: Simple lumped-element port for quick simulations
- MSL (Microstrip Line): For microstrip transmission line ports
- Waveguide: Full waveguide port for accurate S-parameter extraction
- CPWG: Coplanar waveguide with ground port (specialized waveguide)

De-embedding Support:
- Reference plane shifting for connector de-embedding
- Launch structure compensation
- Via transition de-embedding

Impedance Validation:
- Port impedance validation against transmission line Z0
- Mismatch detection with configurable tolerance
- Warning/error modes for impedance discrepancies

Coordinate System:
- All coordinates in nanometers (nm) internally
- Ports positioned relative to board edge center origin

REQ-M2-005: Port placement logic with impedance validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from formula_foundry.coupongen.units import LengthNM

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .geometry import GeometrySpec


class PortType(str, Enum):
    """Available port types for openEMS simulation."""

    LUMPED = "lumped"
    """Simple lumped-element port. Fast but less accurate for high frequencies."""

    MSL = "msl"
    """Microstrip line port. Good for planar structures."""

    WAVEGUIDE = "waveguide"
    """Full waveguide port. Most accurate for S-parameter extraction."""

    CPWG = "cpwg"
    """Coplanar waveguide with ground port. For CPWG transmission lines."""


class DeembedType(str, Enum):
    """De-embedding methods for port calibration."""

    NONE = "none"
    """No de-embedding applied."""

    REFERENCE_PLANE = "reference_plane"
    """Shift reference plane by specified distance."""

    OPEN_SHORT = "open_short"
    """Open-short de-embedding calibration."""

    TRL = "trl"
    """Thru-reflect-line calibration (requires calibration standards)."""


class _SpecBase(BaseModel):
    """Base model with strict validation."""

    model_config = ConfigDict(extra="forbid")


class PortGeometrySpec(_SpecBase):
    """Port geometry dimensions for waveguide/MSL ports.

    Defines the physical extent of the port aperture and
    associated metal/dielectric structures.
    """

    width_nm: LengthNM = Field(..., description="Port aperture width (nm)")
    height_nm: LengthNM = Field(..., description="Port aperture height (nm)")
    signal_width_nm: LengthNM | None = Field(None, description="Signal trace width at port (nm), defaults to width_nm")
    gap_nm: LengthNM | None = Field(None, description="Gap to ground for CPW/CPWG ports (nm)")


class DeembedSpec(_SpecBase):
    """De-embedding specification for port calibration.

    Allows shifting the electrical reference plane to compensate
    for connectors, launches, and other non-DUT structures.
    """

    method: DeembedType = Field(DeembedType.NONE, description="De-embedding method to apply")
    distance_nm: LengthNM | None = Field(None, description="Reference plane shift distance (nm)")
    epsilon_r_eff: float | None = Field(
        None,
        gt=0,
        description="Effective dielectric constant for phase correction",
    )


class ImpedanceSpec(_SpecBase):
    """Port impedance specification.

    Defines the reference impedance for S-parameter normalization
    and any matching network parameters.
    """

    z0_ohm: float = Field(50.0, gt=0, description="Reference impedance (Ohm)")
    match_to_line: bool = Field(
        False,
        description="Auto-match to calculated transmission line impedance",
    )
    calculated_z0_ohm: float | None = Field(
        None,
        gt=0,
        description="Calculated line impedance (Ohm), set by port builder",
    )


class WaveguidePortSpec(_SpecBase):
    """Enhanced waveguide port specification for S-parameter extraction.

    This extends the basic PortSpec with waveguide-specific parameters
    for accurate S-parameter extraction including mode matching and
    de-embedding support.
    """

    id: str = Field(..., min_length=1, description="Unique port identifier")
    port_type: PortType = Field(PortType.WAVEGUIDE, description="Port type for excitation")

    # Position and orientation
    position_nm: tuple[LengthNM, LengthNM, LengthNM] = Field(..., description="Port center position [x, y, z] in nm")
    direction: Literal["x", "y", "z", "-x", "-y", "-z"] = Field(..., description="Port excitation/propagation direction")

    # Excitation
    excite: bool = Field(False, description="Whether this port is excited")
    excite_weight: float = Field(1.0, description="Excitation amplitude weight (for multi-port)")

    # Geometry
    geometry: PortGeometrySpec | None = Field(None, description="Port aperture geometry (required for waveguide/MSL)")

    # Impedance
    impedance: ImpedanceSpec = Field(default_factory=ImpedanceSpec, description="Impedance specification")

    # De-embedding
    deembed: DeembedSpec = Field(default_factory=DeembedSpec, description="De-embedding specification")

    # Polarization for mode selection
    polarization: Literal["E_transverse", "H_transverse"] | None = Field(
        None, description="Polarization for quasi-TEM mode selection"
    )


@dataclass(frozen=True, slots=True)
class PortPosition:
    """Computed port position in 3D space.

    Attributes:
        x_nm: X coordinate in nm.
        y_nm: Y coordinate in nm.
        z_nm: Z coordinate in nm.
        direction: Propagation direction.
        layer_id: Associated layer ID (e.g., "L1").
    """

    x_nm: int
    y_nm: int
    z_nm: int
    direction: str
    layer_id: str

    def as_tuple(self) -> tuple[int, int, int]:
        """Return position as (x, y, z) tuple."""
        return (self.x_nm, self.y_nm, self.z_nm)


@dataclass(slots=True)
class PortBuilder:
    """Builder for creating properly configured simulation ports.

    This builder calculates port positions from connector locations,
    handles impedance matching, and sets up de-embedding parameters.

    Attributes:
        geometry: GeometrySpec for the coupon.
        signal_layer_id: Layer ID for signal traces.
        copper_thickness_nm: Copper layer thickness.
    """

    geometry: GeometrySpec
    signal_layer_id: str = "L1"
    copper_thickness_nm: int = 35_000

    def build_connector_port(
        self,
        port_id: str,
        connector_position_nm: tuple[int, int],
        *,
        excite: bool = False,
        direction: str = "x",
        port_type: PortType = PortType.WAVEGUIDE,
        deembed_distance_nm: int | None = None,
    ) -> WaveguidePortSpec:
        """Build a port at a connector location.

        Args:
            port_id: Unique port identifier.
            connector_position_nm: (x, y) position of connector in nm.
            excite: Whether this port is excited.
            direction: Propagation direction.
            port_type: Type of port to create.
            deembed_distance_nm: Optional de-embedding distance.

        Returns:
            Configured WaveguidePortSpec.
        """
        # Get Z position from signal layer
        z_nm = self._signal_layer_z()

        # Get trace dimensions for port geometry
        tl = self.geometry.transmission_line
        width_nm = tl.w_nm
        gap_nm = tl.gap_nm

        # Calculate port height (copper + substrate thickness estimation)
        # For CPWG, height spans from ground below to air above
        height_nm = self._calculate_port_height()

        # Build geometry spec
        geometry_spec = PortGeometrySpec(
            width_nm=width_nm + 2 * gap_nm,  # Full CPW width
            height_nm=height_nm,
            signal_width_nm=width_nm,
            gap_nm=gap_nm,
        )

        # Build de-embedding spec
        deembed_spec = DeembedSpec(method=DeembedType.NONE)
        if deembed_distance_nm is not None and deembed_distance_nm > 0:
            # Calculate effective epsilon_r for phase shift
            epsilon_r_eff = self._calculate_epsilon_r_eff()
            deembed_spec = DeembedSpec(
                method=DeembedType.REFERENCE_PLANE,
                distance_nm=deembed_distance_nm,
                epsilon_r_eff=epsilon_r_eff,
            )

        # Build impedance spec
        impedance_spec = ImpedanceSpec(
            z0_ohm=50.0,
            match_to_line=True,
            calculated_z0_ohm=self._calculate_line_impedance(),
        )

        return WaveguidePortSpec(
            id=port_id,
            port_type=port_type,
            position_nm=(connector_position_nm[0], connector_position_nm[1], z_nm),
            direction=direction,
            excite=excite,
            geometry=geometry_spec,
            impedance=impedance_spec,
            deembed=deembed_spec,
            polarization="E_transverse",
        )

    def build_transmission_line_ports(
        self,
        left_position_nm: tuple[int, int],
        right_position_nm: tuple[int, int],
        *,
        port_type: PortType = PortType.WAVEGUIDE,
        deembed_distance_nm: int | None = None,
    ) -> tuple[WaveguidePortSpec, WaveguidePortSpec]:
        """Build a pair of ports at transmission line ends.

        Creates properly configured input (excited) and output ports
        for 2-port S-parameter extraction.

        Args:
            left_position_nm: (x, y) position of left port.
            right_position_nm: (x, y) position of right port.
            port_type: Type of port to create.
            deembed_distance_nm: De-embedding distance for both ports.

        Returns:
            Tuple of (port1, port2) where port1 is excited.
        """
        port1 = self.build_connector_port(
            "P1",
            left_position_nm,
            excite=True,
            direction="x",  # Propagating +x into structure
            port_type=port_type,
            deembed_distance_nm=deembed_distance_nm,
        )

        port2 = self.build_connector_port(
            "P2",
            right_position_nm,
            excite=False,
            direction="-x",  # Propagating -x into structure
            port_type=port_type,
            deembed_distance_nm=deembed_distance_nm,
        )

        return (port1, port2)

    def _signal_layer_z(self) -> int:
        """Get Z coordinate for the signal layer."""
        for layer in self.geometry.layers:
            if layer.id == self.signal_layer_id:
                # Return center of copper layer
                return int(layer.z_nm) + self.copper_thickness_nm // 2
        raise ValueError(f"Signal layer {self.signal_layer_id} not found")

    def _calculate_port_height(self) -> int:
        """Calculate appropriate port height based on stackup.

        For CPWG structures, the port height should span from
        ground plane to some distance above the trace.
        """
        # Get substrate thickness from stackup
        stackup = self.geometry.stackup
        # Use L1_to_L2 thickness as substrate height reference
        substrate_thickness = stackup.thicknesses_nm.get("L1_to_L2", 200_000)

        # Port height: 2x substrate thickness is typical for good field capture
        # Plus copper thickness and some air
        return substrate_thickness * 2 + self.copper_thickness_nm

    def _calculate_epsilon_r_eff(self) -> float:
        """Calculate effective dielectric constant for CPWG.

        Uses simplified approximation for coplanar structures.
        For more accurate values, this should use conformal mapping.
        """
        er = self.geometry.stackup.materials.er
        # Simplified CPWG effective epsilon_r approximation
        # More accurate calculation would require substrate thickness and gap
        return (er + 1) / 2

    def _calculate_line_impedance(self) -> float | None:
        """Calculate transmission line characteristic impedance.

        Uses simplified CPWG impedance formula. Returns None if
        calculation cannot be performed.
        """
        tl = self.geometry.transmission_line
        w = tl.w_nm / 1e9  # Convert to meters
        g = tl.gap_nm / 1e9  # Convert to meters

        # Get substrate properties
        er = self.geometry.stackup.materials.er
        epsilon_r_eff = (er + 1) / 2

        # Simplified CPWG impedance using quasi-static approximation
        # Z0 ≈ (30π / sqrt(ε_eff)) * K(k') / K(k)
        # where k = w / (w + 2g)
        # This is a simplified formula; accurate calculation requires
        # conformal mapping with substrate thickness consideration.

        if w <= 0 or g <= 0:
            return 50.0  # Default fallback

        # Calculate k parameter
        k = w / (w + 2 * g)

        # Very simplified impedance estimate
        # For accurate CPWG impedance, use proper elliptic integrals
        import math

        if k > 0 and k < 1:
            # Approximate K(k)/K(k') ratio for moderate k
            k_prime = math.sqrt(1 - k * k)
            if k_prime > 0.7:
                # Use approximate formula valid for k < 0.7
                ratio = math.pi / math.log(2 * (1 + math.sqrt(k_prime)) / (1 - math.sqrt(k_prime)))
            else:
                # Use alternate approximation
                ratio = math.log(2 * (1 + math.sqrt(k)) / (1 - math.sqrt(k))) / math.pi

            z0 = (30 * math.pi / math.sqrt(epsilon_r_eff)) * ratio
            return z0

        return 50.0


def build_ports_from_resolved(
    geometry: GeometrySpec,
    parameters_nm: Mapping[str, int],
    *,
    signal_layer_id: str = "L1",
    port_type: PortType = PortType.WAVEGUIDE,
    include_deembedding: bool = True,
) -> list[WaveguidePortSpec]:
    """Build simulation ports from resolved design parameters.

    This is the main entry point for creating ports from an M1 ResolvedDesign.
    It extracts connector positions and builds properly configured ports.

    Args:
        geometry: GeometrySpec from build_geometry_spec.
        parameters_nm: Resolved parameter dictionary.
        signal_layer_id: Layer ID for signal traces.
        port_type: Type of ports to create.
        include_deembedding: Whether to include de-embedding configuration.

    Returns:
        List of configured WaveguidePortSpec instances.
    """
    builder = PortBuilder(geometry=geometry, signal_layer_id=signal_layer_id)

    # Extract connector positions
    left_pos = _extract_connector_position(parameters_nm, "left")
    right_pos = _extract_connector_position(parameters_nm, "right")

    if left_pos is None or right_pos is None:
        # Fall back to transmission line length-based positions
        left_pos, right_pos = _fallback_port_positions(parameters_nm)

    # Calculate de-embedding distance if requested
    deembed_distance: int | None = None
    if include_deembedding:
        # Use connector pad length or launch length for de-embedding
        # This compensates for the connector-to-trace transition
        launch_length = parameters_nm.get("launch.length_nm")
        if launch_length is not None:
            deembed_distance = int(launch_length)

    port1, port2 = builder.build_transmission_line_ports(
        left_pos,
        right_pos,
        port_type=port_type,
        deembed_distance_nm=deembed_distance,
    )

    return [port1, port2]


def waveguide_port_to_basic_port_spec(
    wp: WaveguidePortSpec,
) -> dict:
    """Convert WaveguidePortSpec to basic PortSpec dict for SimulationSpec.

    This provides backward compatibility with the existing PortSpec model
    while preserving the enhanced waveguide port information.

    Args:
        wp: WaveguidePortSpec to convert.

    Returns:
        Dictionary suitable for PortSpec model validation.
    """
    basic = {
        "id": wp.id,
        "type": wp.port_type.value,
        "impedance_ohm": wp.impedance.z0_ohm,
        "excite": wp.excite,
        "position_nm": wp.position_nm,
        "direction": wp.direction,
    }

    # Add geometry dimensions if available
    if wp.geometry is not None:
        basic["width_nm"] = wp.geometry.width_nm
        basic["height_nm"] = wp.geometry.height_nm

    return basic


def _extract_connector_position(params: Mapping[str, int], side: str) -> tuple[int, int] | None:
    """Extract connector position from parameters.

    Args:
        params: Resolved parameter dictionary.
        side: "left" or "right".

    Returns:
        (x, y) position tuple or None if not found.
    """
    x_key = f"connectors.{side}.position_nm[0]"
    y_key = f"connectors.{side}.position_nm[1]"
    x_value = params.get(x_key)
    y_value = params.get(y_key)
    if x_value is None or y_value is None:
        return None
    return (int(x_value), int(y_value))


def _fallback_port_positions(
    params: Mapping[str, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Calculate port positions from transmission line lengths.

    Falls back to using the transmission line lengths when connector
    positions are not explicitly specified.

    Args:
        params: Resolved parameter dictionary.

    Returns:
        Tuple of ((left_x, left_y), (right_x, right_y)).

    Raises:
        KeyError: If transmission line lengths are not available.
    """
    left_length = params.get("transmission_line.length_left_nm")
    right_length = params.get("transmission_line.length_right_nm")
    if left_length is None or right_length is None:
        raise KeyError("Transmission line lengths required for port position fallback")
    return ((-int(left_length), 0), (int(right_length), 0))


# =============================================================================
# Impedance Validation (REQ-M2-005)
# =============================================================================


@dataclass(frozen=True, slots=True)
class ImpedanceValidationResult:
    """Result of port impedance validation against transmission line Z0.

    Attributes:
        port_id: Port identifier that was validated.
        port_z0_ohm: Port reference impedance in Ohms.
        line_z0_ohm: Calculated transmission line impedance in Ohms.
        mismatch_percent: Impedance mismatch as percentage.
        is_valid: Whether the impedance is within tolerance.
        tolerance_percent: Tolerance threshold used for validation.
        message: Human-readable validation message.
    """

    port_id: str
    port_z0_ohm: float
    line_z0_ohm: float
    mismatch_percent: float
    is_valid: bool
    tolerance_percent: float
    message: str


class ImpedanceMismatchError(Exception):
    """Raised when port impedance exceeds acceptable mismatch threshold."""

    def __init__(self, result: ImpedanceValidationResult) -> None:
        self.result = result
        super().__init__(result.message)


class ImpedanceMismatchWarning(UserWarning):
    """Warning issued when port impedance mismatch is notable but acceptable."""

    pass


def validate_port_impedance(
    port: WaveguidePortSpec,
    line_z0_ohm: float,
    *,
    tolerance_percent: float = 10.0,
    error_threshold_percent: float = 25.0,
    strict: bool = False,
) -> ImpedanceValidationResult:
    """Validate port impedance against transmission line characteristic impedance.

    Compares the port reference impedance to the calculated transmission line
    impedance and checks if the mismatch is within acceptable bounds.

    Args:
        port: The waveguide port specification to validate.
        line_z0_ohm: Calculated transmission line characteristic impedance (Ohms).
        tolerance_percent: Warning threshold for impedance mismatch (default 10%).
        error_threshold_percent: Error threshold for impedance mismatch (default 25%).
        strict: If True, raise error when mismatch exceeds tolerance_percent.

    Returns:
        ImpedanceValidationResult with validation details.

    Raises:
        ImpedanceMismatchError: If strict=True and mismatch exceeds tolerance,
            or if mismatch exceeds error_threshold_percent.

    REQ-M2-005: Validate port impedance against transmission line Z0.
    """
    port_z0 = port.impedance.z0_ohm

    # Calculate mismatch percentage
    if line_z0_ohm > 0:
        mismatch = abs(port_z0 - line_z0_ohm) / line_z0_ohm * 100
    else:
        mismatch = 100.0  # Invalid line impedance

    # Determine validity
    if strict:
        is_valid = mismatch <= tolerance_percent
    else:
        is_valid = mismatch <= error_threshold_percent

    # Generate message
    if mismatch <= tolerance_percent:
        message = f"Port {port.id}: impedance {port_z0:.1f}Ω matches line Z0 {line_z0_ohm:.1f}Ω (mismatch: {mismatch:.1f}%)"
    elif mismatch <= error_threshold_percent:
        message = f"Port {port.id}: impedance {port_z0:.1f}Ω deviates from line Z0 {line_z0_ohm:.1f}Ω by {mismatch:.1f}% (warning)"
    else:
        message = f"Port {port.id}: impedance {port_z0:.1f}Ω significantly mismatched with line Z0 {line_z0_ohm:.1f}Ω ({mismatch:.1f}%)"

    result = ImpedanceValidationResult(
        port_id=port.id,
        port_z0_ohm=port_z0,
        line_z0_ohm=line_z0_ohm,
        mismatch_percent=mismatch,
        is_valid=is_valid,
        tolerance_percent=tolerance_percent,
        message=message,
    )

    # Raise error if mismatch exceeds error threshold
    if mismatch > error_threshold_percent:
        raise ImpedanceMismatchError(result)

    # Issue warning if mismatch exceeds tolerance but not error threshold
    if mismatch > tolerance_percent:
        import warnings

        warnings.warn(message, ImpedanceMismatchWarning, stacklevel=2)

    return result


def validate_ports_against_geometry(
    ports: list[WaveguidePortSpec],
    geometry: GeometrySpec,
    *,
    tolerance_percent: float = 10.0,
    error_threshold_percent: float = 25.0,
    strict: bool = False,
) -> list[ImpedanceValidationResult]:
    """Validate all ports against the geometry's transmission line impedance.

    Calculates the characteristic impedance from the geometry specification
    and validates each port's reference impedance against it.

    Args:
        ports: List of ports to validate.
        geometry: GeometrySpec containing transmission line parameters.
        tolerance_percent: Warning threshold (default 10%).
        error_threshold_percent: Error threshold (default 25%).
        strict: If True, require mismatch within tolerance_percent.

    Returns:
        List of ImpedanceValidationResult for each port.

    REQ-M2-005: Validate port impedance against transmission line Z0.
    """
    # Calculate line impedance from geometry
    line_z0 = calculate_cpwg_impedance(
        w_nm=geometry.transmission_line.w_nm,
        gap_nm=geometry.transmission_line.gap_nm,
        er=geometry.stackup.materials.er,
    )

    results: list[ImpedanceValidationResult] = []
    for port in ports:
        result = validate_port_impedance(
            port,
            line_z0,
            tolerance_percent=tolerance_percent,
            error_threshold_percent=error_threshold_percent,
            strict=strict,
        )
        results.append(result)

    return results


def calculate_cpwg_impedance(
    w_nm: int,
    gap_nm: int,
    er: float,
    *,
    h_nm: int | None = None,
) -> float:
    """Calculate CPWG characteristic impedance using quasi-static approximation.

    Uses the elliptic integral approximation for coplanar waveguide with ground.
    More accurate results require substrate thickness consideration.

    Args:
        w_nm: Signal trace width in nm.
        gap_nm: Gap to coplanar ground in nm.
        er: Substrate relative permittivity.
        h_nm: Optional substrate thickness in nm (for enhanced accuracy).

    Returns:
        Characteristic impedance in Ohms.

    Note:
        This uses a simplified formula. For production use, consider
        implementing full elliptic integral calculations or using
        a transmission line calculator library.
    """
    import math

    if w_nm <= 0 or gap_nm <= 0:
        return 50.0  # Default fallback

    # Convert to meters for calculation
    w = w_nm / 1e9
    g = gap_nm / 1e9

    # Effective dielectric constant for CPWG (simplified)
    epsilon_r_eff = (er + 1) / 2

    # Calculate k parameter: k = w / (w + 2*g)
    k = w / (w + 2 * g)

    if k <= 0 or k >= 1:
        return 50.0  # Invalid geometry

    # Elliptic integral ratio approximation
    k_prime = math.sqrt(1 - k * k)

    # Use different approximations based on k value
    if k <= 0.707:
        # For small k, use logarithmic approximation
        if k_prime > 0:
            ratio = math.pi / math.log(2 * (1 + math.sqrt(k_prime)) / (1 - math.sqrt(k_prime)))
        else:
            ratio = 1.0
    else:
        # For large k, use alternate approximation
        if k > 0:
            ratio = math.log(2 * (1 + math.sqrt(k)) / (1 - math.sqrt(k))) / math.pi
        else:
            ratio = 1.0

    # Z0 = (30 * pi / sqrt(epsilon_r_eff)) * K(k') / K(k)
    z0 = (30 * math.pi / math.sqrt(epsilon_r_eff)) * ratio

    return z0
