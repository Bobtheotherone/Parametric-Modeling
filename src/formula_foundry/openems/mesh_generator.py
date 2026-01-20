"""Adaptive mesh generator for openEMS FDTD simulations.

This module generates fixed mesh lines with adaptive refinement near geometry
features such as vias, antipads, metal edges, and transmission line transitions.

The mesh generator produces mesh line coordinates that:
- Are finer near discontinuities (vias, antipads, plane cutouts)
- Are finer near metal edges (trace edges, via pads)
- Transition smoothly between fine and coarse regions
- Respect wavelength-based limits from frequency requirements
- Align with layer boundaries in the Z direction

The output is a populated MeshSpec with fixed_lines_x_nm, fixed_lines_y_nm,
and fixed_lines_z_nm arrays suitable for openEMS RectilinearGrid.

Key concepts:
- Refinement zones: Regions requiring finer mesh (detected from geometry)
- Mesh grading: Smooth transitions between cell sizes (respecting max_ratio)
- Feature alignment: Mesh lines placed at critical geometry boundaries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from formula_foundry.em.mesh import (
    AdaptiveMeshDensity,
    MeshConfig,
    compute_adaptive_mesh_density,
)
from formula_foundry.openems.geometry import GeometrySpec, layer_positions_nm
from formula_foundry.openems.spec import MeshResolutionSpec, MeshSmoothingSpec, MeshSpec

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class RefinementZone:
    """A region requiring finer mesh.

    Attributes:
        center_nm: Center position of the zone in nm.
        radius_nm: Radius of the refinement zone in nm.
        cell_size_nm: Target cell size within this zone.
        axis: Which axis this zone applies to ('x', 'y', or 'z').
    """

    center_nm: int
    radius_nm: int
    cell_size_nm: int
    axis: str


@dataclass(slots=True)
class MeshLineGenerator:
    """Generates mesh lines for a single axis with adaptive refinement.

    This class handles mesh line placement for one axis (X, Y, or Z),
    placing finer lines near refinement zones and smoothly grading
    outward to coarser regions.

    Attributes:
        domain_min_nm: Minimum coordinate of the simulation domain.
        domain_max_nm: Maximum coordinate of the simulation domain.
        base_cell_nm: Default cell size in regions without refinement.
        min_cell_nm: Absolute minimum cell size.
        max_ratio: Maximum ratio between adjacent cell sizes.
        refinement_zones: List of zones requiring finer mesh.
    """

    domain_min_nm: int
    domain_max_nm: int
    base_cell_nm: int
    min_cell_nm: int
    max_ratio: float
    refinement_zones: list[RefinementZone] = field(default_factory=list)

    def generate_lines(self) -> list[int]:
        """Generate mesh lines for this axis.

        Returns:
            Sorted list of mesh line positions in nm.
        """
        lines: set[int] = set()

        # Always include domain boundaries
        lines.add(self.domain_min_nm)
        lines.add(self.domain_max_nm)

        # Add lines at refinement zone centers and edges
        for zone in self.refinement_zones:
            lines.add(zone.center_nm)
            lines.add(zone.center_nm - zone.radius_nm)
            lines.add(zone.center_nm + zone.radius_nm)

        # Generate lines within each refinement zone
        for zone in self.refinement_zones:
            zone_lines = self._generate_zone_lines(zone)
            lines.update(zone_lines)

        # Fill remaining regions with graded mesh
        sorted_lines = sorted(lines)
        filled_lines = self._fill_gaps_with_grading(sorted_lines)

        # Ensure all lines are within domain bounds
        result = [ln for ln in filled_lines if self.domain_min_nm <= ln <= self.domain_max_nm]

        return sorted(set(result))

    def _generate_zone_lines(self, zone: RefinementZone) -> list[int]:
        """Generate lines within a refinement zone.

        Args:
            zone: The refinement zone to mesh.

        Returns:
            List of mesh line positions within the zone.
        """
        lines: list[int] = []

        # Generate lines from center outward in both directions
        # Using the zone's cell size
        cell = max(zone.cell_size_nm, self.min_cell_nm)

        # From center to positive edge
        pos = zone.center_nm
        while pos < zone.center_nm + zone.radius_nm:
            lines.append(pos)
            pos += cell

        # From center to negative edge
        pos = zone.center_nm - cell
        while pos > zone.center_nm - zone.radius_nm:
            lines.append(pos)
            pos -= cell

        return lines

    def _fill_gaps_with_grading(self, sorted_lines: list[int]) -> list[int]:
        """Fill gaps between existing lines with graded mesh.

        Uses geometric grading to smoothly transition from fine to coarse
        regions, respecting the max_ratio constraint.

        Args:
            sorted_lines: Existing mesh lines, sorted.

        Returns:
            Complete list of mesh lines with gaps filled.
        """
        if len(sorted_lines) < 2:
            return sorted_lines

        result: list[int] = []

        for i in range(len(sorted_lines) - 1):
            start = sorted_lines[i]
            end = sorted_lines[i + 1]
            result.append(start)

            gap = end - start
            if gap <= self.base_cell_nm:
                # Gap is already smaller than base cell, don't add more lines
                continue

            # Fill gap with lines, respecting grading
            fill_lines = self._fill_single_gap(start, end)
            result.extend(fill_lines)

        result.append(sorted_lines[-1])

        return result

    def _fill_single_gap(self, start: int, end: int) -> list[int]:
        """Fill a single gap with graded mesh lines.

        Args:
            start: Start position of the gap.
            end: End position of the gap.

        Returns:
            List of intermediate mesh line positions.
        """
        end - start
        lines: list[int] = []

        # Determine target cell sizes at start and end based on nearby refinement zones
        start_cell = self._cell_size_at(start)
        end_cell = self._cell_size_at(end)

        # Use geometric grading from both ends to meet in the middle
        # Place lines from start moving toward end
        forward_lines = self._grade_from_point(start, end, start_cell)

        # Place lines from end moving toward start
        backward_lines = self._grade_from_point(end, start, end_cell)

        # Merge the two sets
        lines.extend(forward_lines)
        lines.extend(backward_lines)

        return lines

    def _grade_from_point(self, start: int, end: int, initial_cell: int) -> list[int]:
        """Generate graded mesh lines from a starting point.

        Args:
            start: Starting position.
            end: Target position (lines stop before this).
            initial_cell: Cell size at the starting point.

        Returns:
            List of mesh line positions.
        """
        lines: list[int] = []
        direction = 1 if end > start else -1
        gap = abs(end - start)

        pos = start
        cell = initial_cell
        half_gap = gap // 2

        accumulated = 0
        while accumulated < half_gap and cell <= self.base_cell_nm:
            pos += direction * cell
            accumulated += cell

            if direction > 0 and pos < end or direction < 0 and pos > end:
                lines.append(pos)

            # Grade cell size up (coarser) as we move away from refinement
            cell = min(int(cell * self.max_ratio), self.base_cell_nm)

        return lines

    def _cell_size_at(self, position: int) -> int:
        """Determine the target cell size at a given position.

        Checks if the position is within any refinement zone and returns
        the finest cell size required.

        Args:
            position: Position to check.

        Returns:
            Target cell size in nm.
        """
        cell_size = self.base_cell_nm

        for zone in self.refinement_zones:
            if abs(position - zone.center_nm) <= zone.radius_nm:
                cell_size = min(cell_size, zone.cell_size_nm)

        return max(cell_size, self.min_cell_nm)


def detect_via_refinement_zones(
    geometry: GeometrySpec,
    adaptive_density: AdaptiveMeshDensity,
) -> list[RefinementZone]:
    """Detect refinement zones for via transitions.

    Analyzes the discontinuity parameters to find via locations and creates
    refinement zones for the via barrel and surrounding region.

    Args:
        geometry: Coupon geometry specification.
        adaptive_density: Computed mesh densities.

    Returns:
        List of refinement zones for vias.
    """
    zones: list[RefinementZone] = []

    if geometry.discontinuity is None:
        return zones

    params = geometry.discontinuity.parameters_nm

    # Get via center position
    # Via is typically at the center of the board (between left and right transmission lines)
    # or at a specified position in discontinuity parameters
    via_x = params.get("signal_via.x_nm")
    via_y = params.get("signal_via.y_nm")

    # If position not specified, calculate from transmission line geometry
    if via_x is None:
        # Via is at the junction of left and right transmission lines
        # Which is at: left_length from the left edge (origin is EDGE_L_CENTER)
        via_x = geometry.transmission_line.length_left_nm

    if via_y is None:
        # Via is centered on the transmission line (y=0 for EDGE_L_CENTER origin)
        via_y = 0

    # Get via dimensions
    via_drill = params.get("signal_via.drill_nm", 300_000)
    via_diameter = params.get("signal_via.diameter_nm", 650_000)
    via_pad_diameter = params.get("signal_via.pad_diameter_nm", 900_000)

    # Create refinement zone for via
    # Zone radius should encompass the via pad plus some margin
    via_radius = max(via_pad_diameter // 2, via_diameter // 2) + via_drill

    zones.append(
        RefinementZone(
            center_nm=via_x,
            radius_nm=via_radius,
            cell_size_nm=adaptive_density.via_cell_nm,
            axis="x",
        )
    )
    zones.append(
        RefinementZone(
            center_nm=via_y,
            radius_nm=via_radius,
            cell_size_nm=adaptive_density.via_cell_nm,
            axis="y",
        )
    )

    return zones


def detect_antipad_refinement_zones(
    geometry: GeometrySpec,
    adaptive_density: AdaptiveMeshDensity,
) -> list[RefinementZone]:
    """Detect refinement zones for antipad cutouts.

    Analyzes discontinuity parameters to find antipad regions on internal
    layers and creates refinement zones for accurate field resolution.

    Args:
        geometry: Coupon geometry specification.
        adaptive_density: Computed mesh densities.

    Returns:
        List of refinement zones for antipads.
    """
    zones: list[RefinementZone] = []

    if geometry.discontinuity is None:
        return zones

    params = geometry.discontinuity.parameters_nm

    # Get via center position (antipads are centered on the via)
    via_x = params.get("signal_via.x_nm")
    if via_x is None:
        via_x = geometry.transmission_line.length_left_nm

    via_y = params.get("signal_via.y_nm", 0)

    # Find antipad parameters for each layer
    for key, value in params.items():
        if "antipad" in key.lower() and value > 0:
            # Extract antipad radius/dimension
            # Keys like "antipad.L2.r_nm", "antipad.L2.rx_nm", etc.
            antipad_radius = value

            # Create XY refinement zones for this antipad
            zones.append(
                RefinementZone(
                    center_nm=via_x,
                    radius_nm=antipad_radius,
                    cell_size_nm=adaptive_density.antipad_cell_nm,
                    axis="x",
                )
            )
            zones.append(
                RefinementZone(
                    center_nm=via_y,
                    radius_nm=antipad_radius,
                    cell_size_nm=adaptive_density.antipad_cell_nm,
                    axis="y",
                )
            )

    return zones


def detect_trace_refinement_zones(
    geometry: GeometrySpec,
    adaptive_density: AdaptiveMeshDensity,
) -> list[RefinementZone]:
    """Detect refinement zones for transmission line edges.

    Creates refinement zones along the trace edges where field gradients
    are highest (edge of trace, edge of gap).

    Args:
        geometry: Coupon geometry specification.
        adaptive_density: Computed mesh densities.

    Returns:
        List of refinement zones for trace edges.
    """
    zones: list[RefinementZone] = []

    tl = geometry.transmission_line
    trace_width = tl.w_nm
    trace_gap = tl.gap_nm

    # For CPWG, the trace center is at y=0 (EDGE_L_CENTER origin)
    # Trace edges are at y = ±(trace_width/2)
    # Gap outer edges are at y = ±(trace_width/2 + gap_nm)

    half_width = trace_width // 2
    gap_outer = half_width + trace_gap

    # Refinement at positive trace edge
    zones.append(
        RefinementZone(
            center_nm=half_width,
            radius_nm=trace_gap // 2,
            cell_size_nm=adaptive_density.trace_cell_nm,
            axis="y",
        )
    )

    # Refinement at negative trace edge
    zones.append(
        RefinementZone(
            center_nm=-half_width,
            radius_nm=trace_gap // 2,
            cell_size_nm=adaptive_density.trace_cell_nm,
            axis="y",
        )
    )

    # Refinement at gap outer edges (ground plane edge)
    zones.append(
        RefinementZone(
            center_nm=gap_outer,
            radius_nm=adaptive_density.trace_cell_nm * 2,
            cell_size_nm=adaptive_density.trace_cell_nm,
            axis="y",
        )
    )
    zones.append(
        RefinementZone(
            center_nm=-gap_outer,
            radius_nm=adaptive_density.trace_cell_nm * 2,
            cell_size_nm=adaptive_density.trace_cell_nm,
            axis="y",
        )
    )

    return zones


def generate_z_mesh_lines(
    geometry: GeometrySpec,
    adaptive_density: AdaptiveMeshDensity,
    copper_thickness_nm: int = 35_000,
) -> list[int]:
    """Generate Z-axis mesh lines aligned with layer boundaries.

    Places mesh lines at:
    - Layer boundaries (top and bottom of each copper layer)
    - Within substrate regions with appropriate refinement
    - With smooth grading between layers

    Args:
        geometry: Coupon geometry specification.
        adaptive_density: Computed mesh densities.
        copper_thickness_nm: Copper layer thickness in nm.

    Returns:
        Sorted list of Z mesh line positions.
    """
    lines: set[int] = set()

    # Get layer positions from stackup
    positions = layer_positions_nm(geometry.stackup)
    num_layers = geometry.stackup.copper_layers

    # Calculate total thickness
    max_z = positions.get(f"L{num_layers}", 0)
    total_thickness = max_z + copper_thickness_nm

    # Add lines at each layer boundary
    for i in range(1, num_layers + 1):
        layer_id = f"L{i}"
        pos = positions.get(layer_id, 0)

        # Convert to Z coordinate (L1 = top, higher Z)
        z_bottom = total_thickness - pos - copper_thickness_nm
        z_top = z_bottom + copper_thickness_nm

        lines.add(z_bottom)
        lines.add(z_top)

    # Add lines at domain boundaries (with some air above and below)
    air_padding = adaptive_density.base_cell_nm * 5
    lines.add(-air_padding)
    lines.add(total_thickness + air_padding)

    # Fill substrate regions with finer mesh
    sorted_lines = sorted(lines)
    result: list[int] = []

    for i in range(len(sorted_lines) - 1):
        z_start = sorted_lines[i]
        z_end = sorted_lines[i + 1]
        result.append(z_start)

        gap = z_end - z_start
        cell_size = adaptive_density.substrate_cell_nm

        # Fill gap with intermediate lines if needed
        if gap > cell_size:
            num_cells = max(1, gap // cell_size)
            actual_cell = gap // num_cells

            for j in range(1, num_cells):
                result.append(z_start + j * actual_cell)

    result.append(sorted_lines[-1])

    return sorted(set(result))


def generate_adaptive_mesh_lines(
    mesh_config: MeshConfig,
    geometry: GeometrySpec,
    adaptive_density: AdaptiveMeshDensity | None = None,
    *,
    copper_thickness_nm: int = 35_000,
) -> MeshSpec:
    """Generate fixed mesh lines with adaptive refinement.

    Places mesh lines with:
    - Coarser spacing in free regions
    - Finer spacing near vias, antipads, traces
    - Smooth transitions (respecting max_ratio from config)

    Args:
        mesh_config: Mesh configuration parameters.
        geometry: Coupon geometry with discontinuities.
        adaptive_density: Precomputed density (or compute if None).
        copper_thickness_nm: Copper layer thickness for Z calculations.

    Returns:
        MeshSpec with populated fixed_lines_x/y/z_nm.
    """
    # Compute adaptive density if not provided
    if adaptive_density is None:
        adaptive_density = compute_adaptive_mesh_density(mesh_config, geometry)

    # Collect all refinement zones
    all_zones: list[RefinementZone] = []
    all_zones.extend(detect_via_refinement_zones(geometry, adaptive_density))
    all_zones.extend(detect_antipad_refinement_zones(geometry, adaptive_density))
    all_zones.extend(detect_trace_refinement_zones(geometry, adaptive_density))

    # Separate zones by axis
    x_zones = [z for z in all_zones if z.axis == "x"]
    y_zones = [z for z in all_zones if z.axis == "y"]

    # Determine simulation domain bounds
    # X: from 0 to board length (plus PML padding)
    board_length = geometry.board.length_nm
    pml_padding = adaptive_density.base_cell_nm * mesh_config.pml_cells
    x_min = -pml_padding
    x_max = board_length + pml_padding

    # Y: symmetric around 0, half of board width plus padding
    board_half_width = geometry.board.width_nm // 2
    y_min = -board_half_width - pml_padding
    y_max = board_half_width + pml_padding

    # Generate X mesh lines
    x_generator = MeshLineGenerator(
        domain_min_nm=x_min,
        domain_max_nm=x_max,
        base_cell_nm=adaptive_density.base_cell_nm,
        min_cell_nm=mesh_config.min_cell_size_nm,
        max_ratio=mesh_config.smoothmesh_ratio,
        refinement_zones=x_zones,
    )
    x_lines = x_generator.generate_lines()

    # Generate Y mesh lines
    y_generator = MeshLineGenerator(
        domain_min_nm=y_min,
        domain_max_nm=y_max,
        base_cell_nm=adaptive_density.base_cell_nm,
        min_cell_nm=mesh_config.min_cell_size_nm,
        max_ratio=mesh_config.smoothmesh_ratio,
        refinement_zones=y_zones,
    )
    y_lines = y_generator.generate_lines()

    # Generate Z mesh lines
    z_lines = generate_z_mesh_lines(geometry, adaptive_density, copper_thickness_nm)

    # Build MeshSpec
    return MeshSpec(
        resolution=MeshResolutionSpec(
            lambda_resolution=int(1 / mesh_config.min_wavelength_fraction),
            metal_edge_resolution_nm=mesh_config.edge_refinement_nm,
            via_resolution_nm=mesh_config.via_refinement_nm,
            substrate_resolution_nm=mesh_config.substrate_refinement_nm,
        ),
        smoothing=MeshSmoothingSpec(
            max_ratio=mesh_config.smoothmesh_ratio,
            smooth_mesh_lines=True,
        ),
        fixed_lines_x_nm=x_lines,
        fixed_lines_y_nm=y_lines,
        fixed_lines_z_nm=z_lines,
    )


def mesh_line_summary(mesh_spec: MeshSpec) -> dict[str, int | float]:
    """Generate summary statistics for a mesh specification.

    Useful for debugging and verifying mesh quality.

    Args:
        mesh_spec: The mesh specification to summarize.

    Returns:
        Dictionary with mesh statistics.
    """
    x_lines = mesh_spec.fixed_lines_x_nm
    y_lines = mesh_spec.fixed_lines_y_nm
    z_lines = mesh_spec.fixed_lines_z_nm

    def cell_stats(lines: list[int]) -> tuple[int, int, float]:
        """Compute min, max, mean cell size for a set of lines."""
        if len(lines) < 2:
            return 0, 0, 0.0
        cells = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
        return min(cells), max(cells), sum(cells) / len(cells)

    x_min, x_max, x_mean = cell_stats(x_lines)
    y_min, y_max, y_mean = cell_stats(y_lines)
    z_min, z_max, z_mean = cell_stats(z_lines)

    total_cells = (len(x_lines) - 1) * (len(y_lines) - 1) * (len(z_lines) - 1)

    return {
        "total_cells": total_cells,
        "n_lines_x": len(x_lines),
        "n_lines_y": len(y_lines),
        "n_lines_z": len(z_lines),
        "x_cell_min_nm": x_min,
        "x_cell_max_nm": x_max,
        "x_cell_mean_nm": x_mean,
        "y_cell_min_nm": y_min,
        "y_cell_max_nm": y_max,
        "y_cell_mean_nm": y_mean,
        "z_cell_min_nm": z_min,
        "z_cell_max_nm": z_max,
        "z_cell_mean_nm": z_mean,
    }
