"""Mesh configuration for EM simulations.

This module provides a solver-agnostic mesh configuration dataclass and
adaptive mesh density calculation based on coupon geometry. The mesh
parameters are designed for FDTD-style simulations (e.g., openEMS) but
the abstractions can support other EM solvers.

Key concepts:
- MeshConfig: Complete mesh configuration including smoothing, PML, resolution
- FrequencyRange: Frequency bounds for wavelength-based mesh sizing
- AdaptiveMeshDensity: Computed mesh densities for different geometry regions
- Adaptive mesh calculation: Automatically determines mesh density based on
  coupon geometry features (vias, traces, antipads, etc.)

All lengths are in nanometers (integer) for consistency with the coupongen
coordinate system and to ensure deterministic, reproducible results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from formula_foundry.openems.geometry import GeometrySpec

# Speed of light in vacuum (m/s)
C0_M_S: int = 299_792_458

# Speed of light in nm/s for internal calculations
C0_NM_S: int = C0_M_S * 1_000_000_000


@dataclass(frozen=True, slots=True)
class FrequencyRange:
    """Frequency range for simulation and mesh sizing.

    The frequency range determines the wavelength-based mesh resolution.
    The maximum frequency (f_max_hz) is used to compute the minimum
    wavelength, which drives the mesh cell size requirements.

    Attributes:
        f_min_hz: Minimum frequency in Hz (start of sweep).
        f_max_hz: Maximum frequency in Hz (end of sweep, determines mesh).
    """

    f_min_hz: int
    f_max_hz: int

    def __post_init__(self) -> None:
        if self.f_min_hz <= 0:
            raise ValueError("f_min_hz must be positive")
        if self.f_max_hz <= 0:
            raise ValueError("f_max_hz must be positive")
        if self.f_min_hz > self.f_max_hz:
            raise ValueError("f_min_hz must be <= f_max_hz")


@dataclass(frozen=True, slots=True)
class MeshConfig:
    """Complete mesh configuration for EM simulations.

    This dataclass contains all parameters needed to generate an FDTD mesh
    for coupon simulations. The parameters are designed to balance accuracy
    (fine mesh in critical regions) with computational efficiency.

    Attributes:
        smoothmesh_ratio: Maximum ratio between adjacent cell sizes.
            Controls mesh grading smoothness. Typical range: 1.2-1.5.
            Lower values = smoother transitions = more cells.
        edge_refinement_nm: Maximum cell size near metal edges in nm.
            Critical for accurate field calculation at conductor boundaries.
        pml_cells: Number of PML (Perfectly Matched Layer) cells for
            absorbing boundary conditions. Typical: 8-16 cells.
        frequency_range: Frequency range for the simulation.
        min_wavelength_fraction: Fraction of minimum wavelength for base
            cell size. E.g., 1/20 means ~20 cells per wavelength.
            Typical range: 1/10 to 1/30.
        via_refinement_nm: Maximum cell size near via barrels in nm.
            Vias require fine mesh due to rapid field variations.
        substrate_refinement_nm: Cell size for substrate layers in nm.
            Can be coarser than metal edges if materials are homogeneous.
        min_cell_size_nm: Absolute minimum cell size in nm.
            Prevents excessively fine mesh that would slow simulation.
        max_cell_size_nm: Absolute maximum cell size in nm.
            Prevents excessively coarse mesh that would reduce accuracy.
    """

    smoothmesh_ratio: float
    edge_refinement_nm: int
    pml_cells: int
    frequency_range: FrequencyRange
    min_wavelength_fraction: float
    via_refinement_nm: int = field(default=25_000)
    substrate_refinement_nm: int = field(default=100_000)
    min_cell_size_nm: int = field(default=1_000)
    max_cell_size_nm: int = field(default=1_000_000)

    def __post_init__(self) -> None:
        if self.smoothmesh_ratio < 1.0:
            raise ValueError("smoothmesh_ratio must be >= 1.0")
        if self.smoothmesh_ratio > 3.0:
            raise ValueError("smoothmesh_ratio must be <= 3.0 for stability")
        if self.edge_refinement_nm <= 0:
            raise ValueError("edge_refinement_nm must be positive")
        if self.pml_cells < 1:
            raise ValueError("pml_cells must be >= 1")
        if self.pml_cells > 64:
            raise ValueError("pml_cells must be <= 64 (excessive PML is wasteful)")
        if self.min_wavelength_fraction <= 0:
            raise ValueError("min_wavelength_fraction must be positive")
        if self.min_wavelength_fraction > 1.0:
            raise ValueError("min_wavelength_fraction must be <= 1.0")
        if self.via_refinement_nm <= 0:
            raise ValueError("via_refinement_nm must be positive")
        if self.substrate_refinement_nm <= 0:
            raise ValueError("substrate_refinement_nm must be positive")
        if self.min_cell_size_nm <= 0:
            raise ValueError("min_cell_size_nm must be positive")
        if self.max_cell_size_nm <= 0:
            raise ValueError("max_cell_size_nm must be positive")
        if self.min_cell_size_nm > self.max_cell_size_nm:
            raise ValueError("min_cell_size_nm must be <= max_cell_size_nm")

    @property
    def min_wavelength_nm(self) -> int:
        """Compute minimum wavelength in vacuum at max frequency.

        Returns:
            Minimum wavelength in nanometers.
        """
        return compute_min_wavelength_nm(self.frequency_range.f_max_hz, epsilon_r=1.0)

    @property
    def base_cell_size_nm(self) -> int:
        """Compute base cell size from wavelength fraction.

        This is the target cell size in regions without special refinement.

        Returns:
            Base cell size in nanometers, clamped to min/max bounds.
        """
        raw_size = int(self.min_wavelength_nm * self.min_wavelength_fraction)
        return max(self.min_cell_size_nm, min(raw_size, self.max_cell_size_nm))


@dataclass(frozen=True, slots=True)
class AdaptiveMeshDensity:
    """Computed adaptive mesh densities for different geometry regions.

    This dataclass holds the computed mesh cell sizes for various regions
    of the coupon geometry, based on the MeshConfig and geometry features.

    All cell sizes are in nanometers and represent the maximum cell size
    allowed in that region. The mesh generator should use these as upper
    bounds when placing mesh lines.

    Attributes:
        base_cell_nm: Default cell size in "free" regions.
        trace_cell_nm: Cell size along transmission line traces.
        via_cell_nm: Cell size near via transitions.
        antipad_cell_nm: Cell size in antipad/plane cutout regions.
        substrate_cell_nm: Cell size in substrate (z-direction).
        pml_cell_nm: Cell size in PML boundary regions.
        min_feature_size_nm: Smallest geometry feature detected.
    """

    base_cell_nm: int
    trace_cell_nm: int
    via_cell_nm: int
    antipad_cell_nm: int
    substrate_cell_nm: int
    pml_cell_nm: int
    min_feature_size_nm: int


def compute_min_wavelength_nm(f_max_hz: int, epsilon_r: float = 1.0) -> int:
    """Compute minimum wavelength at maximum frequency.

    The wavelength in a dielectric is reduced by sqrt(epsilon_r).
    This function returns the wavelength in nanometers.

    Args:
        f_max_hz: Maximum frequency in Hz.
        epsilon_r: Relative permittivity of the medium (default 1.0 for vacuum).

    Returns:
        Minimum wavelength in nanometers.

    Raises:
        ValueError: If f_max_hz <= 0 or epsilon_r <= 0.
    """
    if f_max_hz <= 0:
        raise ValueError("f_max_hz must be positive")
    if epsilon_r <= 0:
        raise ValueError("epsilon_r must be positive")

    # wavelength = c / (f * sqrt(epsilon_r))
    # Using integer math where possible for determinism
    wavelength_nm = int(C0_NM_S / (f_max_hz * math.sqrt(epsilon_r)))
    return wavelength_nm


def compute_adaptive_mesh_density(
    mesh_config: MeshConfig,
    geometry: GeometrySpec,
) -> AdaptiveMeshDensity:
    """Compute adaptive mesh densities based on coupon geometry.

    This function analyzes the coupon geometry and determines appropriate
    mesh cell sizes for different regions. The algorithm considers:

    1. Wavelength-based sizing: Base cell size from min_wavelength_fraction
    2. Feature-based refinement: Finer mesh near edges, vias, discontinuities
    3. Material properties: Wavelength reduction in dielectrics
    4. Geometry constraints: Mesh must resolve smallest features

    The returned densities are conservative (smaller = finer) to ensure
    accuracy. The mesh generator can optionally use coarser cells in
    regions far from critical features.

    Args:
        mesh_config: Mesh configuration parameters.
        geometry: Coupon geometry specification.

    Returns:
        AdaptiveMeshDensity with computed cell sizes for each region.
    """
    # Get material properties for wavelength calculation
    epsilon_r = geometry.stackup.materials.er

    # Compute wavelength in the dielectric
    min_wavelength_in_dielectric_nm = compute_min_wavelength_nm(
        mesh_config.frequency_range.f_max_hz,
        epsilon_r=epsilon_r,
    )

    # Base cell size from wavelength fraction (in dielectric)
    base_from_wavelength = int(
        min_wavelength_in_dielectric_nm * mesh_config.min_wavelength_fraction
    )
    base_cell_nm = max(
        mesh_config.min_cell_size_nm,
        min(base_from_wavelength, mesh_config.max_cell_size_nm),
    )

    # Trace cell size: based on trace width for accuracy
    trace_width_nm = geometry.transmission_line.w_nm
    trace_gap_nm = geometry.transmission_line.gap_nm

    # Use fraction of trace width, but not finer than edge refinement
    trace_cell_nm = max(
        mesh_config.min_cell_size_nm,
        min(
            mesh_config.edge_refinement_nm,
            trace_width_nm // 4,  # At least 4 cells across trace
            trace_gap_nm // 3,  # At least 3 cells in gap
            base_cell_nm,
        ),
    )

    # Via cell size: use configured refinement
    via_cell_nm = max(
        mesh_config.min_cell_size_nm,
        min(mesh_config.via_refinement_nm, base_cell_nm),
    )

    # If geometry has discontinuity with via parameters, refine further
    if geometry.discontinuity is not None:
        disc_params = geometry.discontinuity.parameters_nm
        # Check for via diameter/drill in discontinuity params
        via_drill = disc_params.get("signal_via.drill_nm", 0)
        via_diameter = disc_params.get("signal_via.diameter_nm", 0)
        if via_drill > 0:
            # At least 4 cells across via drill
            via_cell_nm = min(via_cell_nm, via_drill // 4)
        if via_diameter > 0:
            # At least 6 cells across via pad
            via_cell_nm = min(via_cell_nm, via_diameter // 6)
        via_cell_nm = max(mesh_config.min_cell_size_nm, via_cell_nm)

    # Antipad cell size: similar to via, may need to resolve antipad edges
    antipad_cell_nm = via_cell_nm  # Default to same as via
    if geometry.discontinuity is not None:
        disc_params = geometry.discontinuity.parameters_nm
        # Look for antipad radius parameters
        for key, value in disc_params.items():
            if "antipad" in key.lower() and "nm" in key and value > 0:
                # Resolve antipad edges
                antipad_cell_nm = min(antipad_cell_nm, value // 6)
        antipad_cell_nm = max(mesh_config.min_cell_size_nm, antipad_cell_nm)

    # Substrate cell size: can be coarser, mainly for z-direction
    substrate_cell_nm = max(
        mesh_config.min_cell_size_nm,
        min(mesh_config.substrate_refinement_nm, base_cell_nm),
    )

    # Consider layer thicknesses for substrate refinement
    for thickness_nm in geometry.stackup.thicknesses_nm.values():
        if thickness_nm > 0:
            # At least 3 cells per layer
            substrate_cell_nm = min(substrate_cell_nm, thickness_nm // 3)
    substrate_cell_nm = max(mesh_config.min_cell_size_nm, substrate_cell_nm)

    # PML cell size: can match base cell, PML handles absorption
    pml_cell_nm = base_cell_nm

    # Determine minimum feature size from geometry
    min_feature_size_nm = _detect_min_feature_size(geometry, mesh_config)

    return AdaptiveMeshDensity(
        base_cell_nm=base_cell_nm,
        trace_cell_nm=trace_cell_nm,
        via_cell_nm=via_cell_nm,
        antipad_cell_nm=antipad_cell_nm,
        substrate_cell_nm=substrate_cell_nm,
        pml_cell_nm=pml_cell_nm,
        min_feature_size_nm=min_feature_size_nm,
    )


def _detect_min_feature_size(geometry: GeometrySpec, mesh_config: MeshConfig) -> int:
    """Detect the minimum feature size in the geometry.

    Scans geometry parameters to find the smallest dimension that needs
    to be resolved by the mesh.

    Args:
        geometry: Coupon geometry specification.
        mesh_config: Mesh configuration for bounds.

    Returns:
        Minimum feature size in nanometers.
    """
    candidates: list[int] = []

    # Transmission line dimensions
    candidates.append(geometry.transmission_line.w_nm)
    candidates.append(geometry.transmission_line.gap_nm)

    # Discontinuity parameters (if present)
    if geometry.discontinuity is not None:
        for value in geometry.discontinuity.parameters_nm.values():
            if value > 0:
                candidates.append(value)

    # Layer thicknesses
    for thickness in geometry.stackup.thicknesses_nm.values():
        if thickness > 0:
            candidates.append(thickness)

    # Filter to positive values and find minimum
    positive = [c for c in candidates if c > 0]
    if not positive:
        return mesh_config.min_cell_size_nm

    return min(positive)


def create_default_mesh_config(
    f_min_hz: int = 100_000_000,  # 100 MHz
    f_max_hz: int = 20_000_000_000,  # 20 GHz
) -> MeshConfig:
    """Create a default MeshConfig suitable for high-speed interconnect coupons.

    This provides reasonable defaults for via transition coupon simulations
    in the typical frequency range used for signal integrity analysis.

    Args:
        f_min_hz: Minimum frequency in Hz (default 100 MHz).
        f_max_hz: Maximum frequency in Hz (default 20 GHz).

    Returns:
        MeshConfig with sensible defaults.
    """
    return MeshConfig(
        smoothmesh_ratio=1.4,
        edge_refinement_nm=50_000,  # 50 um
        pml_cells=8,
        frequency_range=FrequencyRange(f_min_hz=f_min_hz, f_max_hz=f_max_hz),
        min_wavelength_fraction=0.05,  # 1/20 wavelength = ~20 cells/wavelength
        via_refinement_nm=25_000,  # 25 um
        substrate_refinement_nm=100_000,  # 100 um
        min_cell_size_nm=5_000,  # 5 um minimum
        max_cell_size_nm=500_000,  # 500 um maximum
    )
