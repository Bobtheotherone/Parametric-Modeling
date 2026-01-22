"""openEMS simulation config specification.

This module defines the Pydantic models for openEMS simulation configuration,
following the same patterns established in coupongen/spec.py:
- Strict validation via ConfigDict(extra="forbid")
- Custom annotated types for frequency/time parsing
- Schema versioning
- JSON schema export

The SimulationSpec defines everything needed to run a deterministic openEMS
FDTD simulation on a coupon geometry.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from formula_foundry.coupongen.units import LengthNM

from .units import FrequencyHz, TimePS


class _SpecBase(BaseModel):
    """Base model with strict validation - no extra fields allowed."""

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Toolchain
# =============================================================================


class OpenEMSToolchainSpec(_SpecBase):
    """openEMS toolchain version specification."""

    version: str = Field(..., min_length=1, description="openEMS version (e.g., '0.0.35')")
    docker_image: str = Field(
        ...,
        min_length=1,
        description="Docker image with optional digest (e.g., 'ghcr.io/thliebig/openems:0.0.35@sha256:...')",
    )


class ToolchainSpec(_SpecBase):
    """Combined toolchain specification."""

    openems: OpenEMSToolchainSpec


# =============================================================================
# Frequency/Excitation
# =============================================================================


class ExcitationSpec(_SpecBase):
    """Excitation signal specification for FDTD simulation.

    The excitation defines the source waveform injected into the simulation.
    A Gaussian pulse is standard for broadband S-parameter extraction.
    """

    type: Literal["gaussian", "sinusoidal", "custom"] = Field("gaussian", description="Excitation waveform type")
    f0_hz: FrequencyHz = Field(..., description="Center frequency (Hz)")
    fc_hz: FrequencyHz = Field(..., description="Cutoff frequency for Gaussian (20dB bandwidth)")


class FrequencySpec(_SpecBase):
    """Frequency sweep specification for S-parameter extraction."""

    f_start_hz: FrequencyHz = Field(..., description="Start frequency (Hz)")
    f_stop_hz: FrequencyHz = Field(..., description="Stop frequency (Hz)")
    n_points: int = Field(201, ge=2, description="Number of frequency points")


# =============================================================================
# Boundary Conditions
# =============================================================================


BoundaryType = Literal["PEC", "PMC", "MUR", "PML_8", "PML_16", "PML_32"]


class BoundarySpec(_SpecBase):
    """Boundary conditions for the simulation domain.

    Standard configuration for S-parameter extraction uses PML on port faces
    and PEC/PMC on transverse faces depending on symmetry.
    """

    x_min: BoundaryType = Field("PML_8", description="Boundary at x_min")
    x_max: BoundaryType = Field("PML_8", description="Boundary at x_max")
    y_min: BoundaryType = Field("PEC", description="Boundary at y_min")
    y_max: BoundaryType = Field("PEC", description="Boundary at y_max")
    z_min: BoundaryType = Field("PEC", description="Boundary at z_min")
    z_max: BoundaryType = Field("PML_8", description="Boundary at z_max")


# =============================================================================
# Mesh Specification
# =============================================================================


class MeshResolutionSpec(_SpecBase):
    """Mesh resolution controls.

    The mesh resolution is specified relative to wavelength and material properties.
    Finer meshing near discontinuities (via transitions, antipads) is critical for accuracy.
    """

    lambda_resolution: int = Field(20, ge=10, le=100, description="Cells per wavelength at max frequency")
    metal_edge_resolution_nm: LengthNM = Field(50_000, description="Maximum cell size near metal edges (nm)")
    via_resolution_nm: LengthNM = Field(25_000, description="Maximum cell size near via barrels (nm)")
    substrate_resolution_nm: LengthNM | None = Field(None, description="Override for substrate vertical resolution (nm)")


class MeshSmoothingSpec(_SpecBase):
    """Mesh smoothing/grading controls."""

    max_ratio: float = Field(1.5, ge=1.0, le=3.0, description="Maximum adjacent cell size ratio")
    smooth_mesh_lines: bool = Field(True, description="Enable automatic mesh smoothing")


class MeshSpec(_SpecBase):
    """Complete mesh specification."""

    resolution: MeshResolutionSpec = Field(default_factory=lambda: MeshResolutionSpec())
    smoothing: MeshSmoothingSpec = Field(default_factory=lambda: MeshSmoothingSpec())
    fixed_lines_x_nm: list[LengthNM] = Field(default_factory=list, description="Fixed mesh lines in x (nm)")
    fixed_lines_y_nm: list[LengthNM] = Field(default_factory=list, description="Fixed mesh lines in y (nm)")
    fixed_lines_z_nm: list[LengthNM] = Field(default_factory=list, description="Fixed mesh lines in z (nm)")


# =============================================================================
# Port Specification
# =============================================================================


PortType = Literal["lumped", "waveguide", "msl", "cpwg"]


class DeembedConfigSpec(_SpecBase):
    """De-embedding configuration for port calibration.

    Allows reference plane shifting to compensate for connectors,
    launches, and other non-DUT structures.
    """

    enabled: bool = Field(False, description="Whether de-embedding is enabled")
    distance_nm: LengthNM | None = Field(None, description="Reference plane shift distance (nm)")
    epsilon_r_eff: float | None = Field(
        None,
        gt=0,
        description="Effective dielectric constant for phase correction",
    )


class PortSpec(_SpecBase):
    """Simulation port definition.

    Ports define where excitation is applied and S-parameters are extracted.
    For via transition coupons, typically two ports at the transmission line ends.

    Enhanced to support waveguide ports with proper impedance matching
    and de-embedding for accurate S-parameter extraction.
    """

    id: str = Field(..., min_length=1, description="Unique port identifier")
    type: PortType = Field("lumped", description="Port type")
    impedance_ohm: float = Field(50.0, gt=0, description="Reference impedance (Ohm)")
    excite: bool = Field(False, description="Whether this port is excited")

    position_nm: tuple[LengthNM, LengthNM, LengthNM] = Field(..., description="Port center position [x, y, z] in nm")
    direction: Literal["x", "y", "z", "-x", "-y", "-z"] = Field(..., description="Port excitation direction")

    # Geometry for waveguide/MSL ports
    width_nm: LengthNM | None = Field(None, description="Port width for MSL/waveguide ports (nm)")
    height_nm: LengthNM | None = Field(None, description="Port height for MSL/waveguide ports (nm)")
    signal_width_nm: LengthNM | None = Field(None, description="Signal trace width at port (nm), for CPW/CPWG")
    gap_nm: LengthNM | None = Field(None, description="Gap to ground for CPW/CPWG ports (nm)")

    # Impedance matching
    match_to_line: bool = Field(False, description="Auto-match reference impedance to calculated line impedance")
    calculated_z0_ohm: float | None = Field(None, gt=0, description="Calculated line characteristic impedance (Ohm)")

    # De-embedding
    deembed: DeembedConfigSpec = Field(default_factory=DeembedConfigSpec, description="De-embedding configuration")

    # Excitation control
    excite_weight: float = Field(1.0, gt=0, description="Excitation amplitude weight for multi-port simulations")

    # Mode selection
    polarization: Literal["E_transverse", "H_transverse"] | None = Field(
        None, description="Polarization for quasi-TEM mode selection"
    )


# =============================================================================
# Material Specification
# =============================================================================


class DielectricMaterialSpec(_SpecBase):
    """Dielectric material properties."""

    id: str = Field(..., min_length=1, description="Material identifier")
    epsilon_r: float = Field(..., gt=0, description="Relative permittivity")
    loss_tangent: float = Field(0.0, ge=0, description="Loss tangent (tan delta)")
    kappa: float = Field(0.0, ge=0, description="Conductivity (S/m)")


class ConductorMaterialSpec(_SpecBase):
    """Conductor/metal material properties."""

    id: str = Field(..., min_length=1, description="Material identifier")
    conductivity: float = Field(..., gt=0, description="Conductivity (S/m)")


class MaterialsSpec(_SpecBase):
    """Material library for the simulation."""

    dielectrics: list[DielectricMaterialSpec] = Field(default_factory=list)
    conductors: list[ConductorMaterialSpec] = Field(default_factory=list)


# =============================================================================
# Simulation Control
# =============================================================================


class TerminationSpec(_SpecBase):
    """Simulation termination criteria."""

    end_criteria_db: float = Field(-50.0, le=0, description="Energy decay threshold in dB for auto-termination")
    max_timesteps: int = Field(1_000_000, gt=0, description="Maximum number of timesteps")
    max_time_ps: TimePS | None = Field(None, description="Maximum simulation time (ps), optional")


class EngineSpec(_SpecBase):
    """FDTD engine configuration.

    Supports both CPU-based engines (basic, sse, multithreaded) and GPU
    acceleration when available. GPU mode requires NVIDIA Container Toolkit
    and a compatible GPU.
    """

    type: Literal["basic", "sse", "sse-compressed", "multithreaded"] = Field("multithreaded", description="Engine type")
    num_threads: int | None = Field(None, ge=1, description="Number of threads (None = auto)")
    use_gpu: bool = Field(False, description="Enable GPU acceleration (requires CUDA-capable GPU)")
    gpu_device_id: int | None = Field(None, ge=0, description="CUDA device ID (None = auto-select first available)")
    gpu_memory_fraction: float | None = Field(
        None, ge=0.1, le=1.0, description="Fraction of GPU memory to use (0.1-1.0, None = auto)"
    )


class SimulationControlSpec(_SpecBase):
    """Simulation execution control."""

    termination: TerminationSpec = Field(default_factory=lambda: TerminationSpec())
    engine: EngineSpec = Field(default_factory=lambda: EngineSpec())
    verbose: int = Field(0, ge=0, le=3, description="Verbosity level 0-3")
    dump_fields: bool = Field(False, description="Dump E/H fields for visualization")


# =============================================================================
# Output Specification
# =============================================================================


class OutputSpec(_SpecBase):
    """Simulation output configuration."""

    outputs_dir: str = Field("sim_outputs/", min_length=1, description="Output directory")
    s_params: bool = Field(True, description="Compute and export S-parameters")
    s_params_format: Literal["touchstone", "csv", "both"] = Field("touchstone", description="S-parameter output format")
    port_signals: bool = Field(True, description="Save port voltage/current time signals")
    energy_decay: bool = Field(True, description="Save energy decay curve")
    nf2ff: bool = Field(False, description="Near-to-far-field transformation (optional)")


# =============================================================================
# Geometry Reference (links to CouponSpec/ResolvedDesign)
# =============================================================================


class GeometryRefSpec(_SpecBase):
    """Reference to coupon geometry.

    The simulation config references a coupon design by its design_hash.
    The resolver will load the corresponding ResolvedDesign to build the
    FDTD geometry model.
    """

    design_hash: str = Field(
        ...,
        min_length=1,
        description="SHA256 design hash of the coupon's ResolvedDesign",
    )
    coupon_id: str | None = Field(None, description="Human-readable coupon ID (optional, derived from hash)")


# =============================================================================
# Top-level SimulationSpec
# =============================================================================


class SimulationSpec(_SpecBase):
    """Complete openEMS simulation specification.

    This is the top-level configuration for running an FDTD simulation.
    It references a coupon geometry (by design_hash), defines ports,
    materials, mesh, boundaries, and simulation control parameters.

    The schema is versioned to support forward-compatible evolution.
    """

    schema_version: int = Field(1, ge=1, description="Schema version for compatibility")
    simulation_id: str | None = Field(None, description="Optional unique simulation identifier")
    units: Literal["nm", "Hz", "ps"] = Field("nm", description="Primary units (lengths in nm, frequencies in Hz, times in ps)")

    toolchain: ToolchainSpec
    geometry_ref: GeometryRefSpec
    excitation: ExcitationSpec
    frequency: FrequencySpec
    boundaries: BoundarySpec = Field(default_factory=lambda: BoundarySpec())
    mesh: MeshSpec = Field(default_factory=lambda: MeshSpec())
    ports: list[PortSpec] = Field(..., min_length=1, description="At least one port required")
    materials: MaterialsSpec = Field(default_factory=lambda: MaterialsSpec())
    control: SimulationControlSpec = Field(default_factory=lambda: SimulationControlSpec())
    output: OutputSpec = Field(default_factory=lambda: OutputSpec())


# =============================================================================
# JSON Schema Export + Loader
# =============================================================================


SIMULATIONSPEC_SCHEMA = SimulationSpec.model_json_schema()
"""JSON Schema for SimulationSpec, auto-generated from Pydantic model."""


def load_simulationspec(data: dict[str, Any]) -> SimulationSpec:
    """Load and validate a SimulationSpec from a dictionary.

    Args:
        data: Dictionary representation of SimulationSpec (e.g., from JSON/YAML).

    Returns:
        Validated SimulationSpec instance.

    Raises:
        pydantic.ValidationError: If data fails validation.
    """
    return SimulationSpec.model_validate(data)
