"""Oracle adapter base class and OpenEMS implementation.

This module provides the abstract OracleAdapter base class and the concrete
OpenEMSAdapter implementation for running FDTD simulations.

The adapter pattern allows:
- Uniform interface for different EM solvers
- Reconstruction of geometry from M1 manifest
- Generation of solver-specific mesh configurations
- Deterministic simulation setup

REQ-M2-001: OracleAdapter base class defines the interface for EM simulation adapters.
REQ-M2-002: OpenEMSAdapter implements the interface for openEMS FDTD simulations.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from formula_foundry.coupongen.geom.primitives import (
    Polygon,
    PolygonType,
    PositionNM,
    TrackSegment,
    Via,
)
from formula_foundry.coupongen.resolve import ResolvedDesign
from formula_foundry.em.mesh import (
    AdaptiveMeshDensity,
    FrequencyRange,
    MeshConfig,
    compute_adaptive_mesh_density,
    create_default_mesh_config,
)

from .csx_primitives import CSXGeometry
from .geometry import GeometrySpec, build_geometry_spec
from .geometry_adapter import (
    GeometryAdapter,
    StackupZMap,
    build_csx_geometry,
)
from .mesh_generator import (
    MeshLineGenerator,
    RefinementZone,
    detect_antipad_refinement_zones,
    detect_trace_refinement_zones,
    detect_via_refinement_zones,
    generate_adaptive_mesh_lines,
    generate_z_mesh_lines,
    mesh_line_summary,
)
from .spec import MeshSpec


@dataclass(frozen=True, slots=True)
class SimulationSetup:
    """Complete simulation setup ready for execution.

    Attributes:
        geometry_spec: Reconstructed geometry from manifest.
        csx_geometry: CSX primitives for the simulation.
        mesh_spec: Generated mesh specification.
        mesh_summary: Statistics about the generated mesh.
        design_hash: Hash of the original design for traceability.
        coupon_family: Coupon family identifier.
    """

    geometry_spec: GeometrySpec
    csx_geometry: CSXGeometry
    mesh_spec: MeshSpec
    mesh_summary: dict[str, int | float]
    design_hash: str
    coupon_family: str


@dataclass(frozen=True, slots=True)
class ThirdsRuleConfig:
    """Configuration for thirds-rule mesh grading near discontinuities.

    The thirds rule places mesh lines at 1/3 and 2/3 of the distance
    from a feature edge, ensuring smooth transitions and accurate
    field resolution at discontinuities.

    Attributes:
        enabled: Whether to apply thirds-rule grading.
        divisions: Number of divisions (3 = standard thirds rule).
        min_cell_nm: Minimum cell size to enforce.
        max_expansion_ratio: Maximum ratio between adjacent cells.
    """

    enabled: bool = True
    divisions: int = 3
    min_cell_nm: int = 5_000
    max_expansion_ratio: float = 1.5


class OracleAdapter(ABC):
    """Abstract base class for EM simulation adapters.

    OracleAdapter defines the interface that all simulation adapters must
    implement. The "oracle" represents the source of truth for S-parameter
    data - either from simulation or measurement.

    Subclasses must implement:
    - load_manifest: Load and validate an M1 manifest
    - reconstruct_geometry: Build geometry from manifest data
    - generate_mesh: Create mesh specification for the solver
    - setup_simulation: Complete simulation setup from manifest

    REQ-M2-001: This base class defines the adapter interface.
    """

    @abstractmethod
    def load_manifest(self, manifest_path: Path) -> dict[str, Any]:
        """Load and validate an M1 coupon manifest.

        Args:
            manifest_path: Path to the manifest.json file.

        Returns:
            Validated manifest dictionary.

        Raises:
            FileNotFoundError: If manifest file does not exist.
            ValueError: If manifest is invalid or missing required fields.
        """
        ...

    @abstractmethod
    def reconstruct_geometry(
        self,
        manifest: dict[str, Any],
    ) -> GeometrySpec:
        """Reconstruct geometry specification from manifest.

        Args:
            manifest: Validated manifest dictionary.

        Returns:
            GeometrySpec with all coupon geometry parameters.

        Raises:
            ValueError: If manifest data is insufficient.
        """
        ...

    @abstractmethod
    def generate_mesh(
        self,
        geometry: GeometrySpec,
        frequency_range: FrequencyRange | None = None,
        mesh_config: MeshConfig | None = None,
    ) -> MeshSpec:
        """Generate mesh specification for the geometry.

        Args:
            geometry: Coupon geometry specification.
            frequency_range: Optional frequency range override.
            mesh_config: Optional mesh configuration override.

        Returns:
            MeshSpec with mesh lines for all axes.
        """
        ...

    @abstractmethod
    def setup_simulation(
        self,
        manifest_path: Path,
        frequency_range: FrequencyRange | None = None,
    ) -> SimulationSetup:
        """Complete simulation setup from manifest.

        This method combines manifest loading, geometry reconstruction,
        and mesh generation into a single operation.

        Args:
            manifest_path: Path to the M1 manifest.
            frequency_range: Optional frequency range for simulation.

        Returns:
            Complete SimulationSetup ready for execution.
        """
        ...


@dataclass(slots=True)
class OpenEMSAdapter(OracleAdapter):
    """OpenEMS FDTD simulation adapter.

    Implements the OracleAdapter interface for openEMS FDTD simulations.
    Handles:
    - M1 manifest loading and validation
    - Geometry reconstruction from resolved design
    - FDTD mesh generation with thirds-rule grading
    - CSX primitive generation

    REQ-M2-002: Implements OpenEMS adapter with mesh generation.

    Attributes:
        copper_thickness_nm: Copper layer thickness in nm.
        thirds_rule: Configuration for thirds-rule mesh grading.
        default_frequency_range: Default frequency range for mesh sizing.
    """

    copper_thickness_nm: int = 35_000
    thirds_rule: ThirdsRuleConfig = field(default_factory=ThirdsRuleConfig)
    default_frequency_range: FrequencyRange = field(
        default_factory=lambda: FrequencyRange(f_min_hz=100_000_000, f_max_hz=20_000_000_000)
    )

    def load_manifest(self, manifest_path: Path) -> dict[str, Any]:
        """Load and validate an M1 coupon manifest.

        Validates that the manifest contains required fields:
        - schema_version
        - coupon_family
        - design_hash
        - resolved_design (with parameters_nm)
        - stackup
        - toolchain_hash

        Args:
            manifest_path: Path to the manifest.json file.

        Returns:
            Validated manifest dictionary.

        Raises:
            FileNotFoundError: If manifest file does not exist.
            ValueError: If manifest is invalid or missing required fields.
        """
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in manifest: {e}") from e

        # Validate required fields
        required_fields = [
            "schema_version",
            "coupon_family",
            "design_hash",
            "resolved_design",
            "stackup",
        ]
        missing = [f for f in required_fields if f not in manifest]
        if missing:
            raise ValueError(f"Manifest missing required fields: {missing}")

        # Validate resolved_design structure
        resolved = manifest["resolved_design"]
        if not isinstance(resolved, dict):
            raise ValueError("resolved_design must be a dictionary")
        if "parameters_nm" not in resolved:
            raise ValueError("resolved_design must contain parameters_nm")

        return manifest

    def reconstruct_geometry(
        self,
        manifest: dict[str, Any],
    ) -> GeometrySpec:
        """Reconstruct geometry specification from manifest.

        Rebuilds the GeometrySpec from the manifest's resolved_design
        and stackup information.

        Args:
            manifest: Validated manifest dictionary.

        Returns:
            GeometrySpec with all coupon geometry parameters.
        """
        # Reconstruct ResolvedDesign from manifest
        resolved_data = manifest["resolved_design"]
        resolved = ResolvedDesign(
            schema_version=resolved_data.get("schema_version", 1),
            coupon_family=manifest["coupon_family"],
            parameters_nm=resolved_data.get("parameters_nm", {}),
            derived_features=manifest.get("derived_features", {}),
            dimensionless_groups=manifest.get("dimensionless_groups", {}),
        )

        # Build geometry spec
        return build_geometry_spec(resolved, manifest)

    def generate_mesh(
        self,
        geometry: GeometrySpec,
        frequency_range: FrequencyRange | None = None,
        mesh_config: MeshConfig | None = None,
    ) -> MeshSpec:
        """Generate FDTD mesh with thirds-rule grading.

        Generates mesh lines with adaptive refinement near discontinuities.
        When thirds_rule is enabled, applies thirds-rule grading at:
        - Via transitions
        - Antipad edges
        - Trace edges
        - Plane boundaries

        Args:
            geometry: Coupon geometry specification.
            frequency_range: Optional frequency range override.
            mesh_config: Optional mesh configuration override.

        Returns:
            MeshSpec with fixed mesh lines for all axes.
        """
        freq_range = frequency_range or self.default_frequency_range

        if mesh_config is None:
            mesh_config = create_default_mesh_config(
                f_min_hz=freq_range.f_min_hz,
                f_max_hz=freq_range.f_max_hz,
            )

        # Compute adaptive density
        adaptive_density = compute_adaptive_mesh_density(mesh_config, geometry)

        # Apply thirds-rule grading if enabled
        if self.thirds_rule.enabled:
            mesh_spec = self._generate_mesh_with_thirds_rule(
                geometry, mesh_config, adaptive_density
            )
        else:
            mesh_spec = generate_adaptive_mesh_lines(
                mesh_config,
                geometry,
                adaptive_density,
                copper_thickness_nm=self.copper_thickness_nm,
            )

        return mesh_spec

    def _generate_mesh_with_thirds_rule(
        self,
        geometry: GeometrySpec,
        mesh_config: MeshConfig,
        adaptive_density: AdaptiveMeshDensity,
    ) -> MeshSpec:
        """Generate mesh with thirds-rule grading at discontinuities.

        The thirds rule places mesh lines at 1/3 and 2/3 of the distance
        from feature edges, ensuring:
        - Smooth field interpolation across boundaries
        - Accurate resolution of rapid field variations
        - Controlled cell size transitions

        Args:
            geometry: Coupon geometry.
            mesh_config: Mesh configuration.
            adaptive_density: Computed adaptive densities.

        Returns:
            MeshSpec with thirds-rule grading applied.
        """
        # Collect all refinement zones
        all_zones: list[RefinementZone] = []
        all_zones.extend(detect_via_refinement_zones(geometry, adaptive_density))
        all_zones.extend(detect_antipad_refinement_zones(geometry, adaptive_density))
        all_zones.extend(detect_trace_refinement_zones(geometry, adaptive_density))

        # Apply thirds-rule to each zone
        thirds_zones = self._apply_thirds_rule_to_zones(all_zones, adaptive_density)

        # Separate zones by axis
        x_zones = [z for z in thirds_zones if z.axis == "x"]
        y_zones = [z for z in thirds_zones if z.axis == "y"]

        # Determine simulation domain bounds
        board_length = geometry.board.length_nm
        pml_padding = adaptive_density.base_cell_nm * mesh_config.pml_cells
        x_min = -pml_padding
        x_max = board_length + pml_padding

        board_half_width = geometry.board.width_nm // 2
        y_min = -board_half_width - pml_padding
        y_max = board_half_width + pml_padding

        # Generate X mesh lines with thirds-rule zones
        x_generator = MeshLineGenerator(
            domain_min_nm=x_min,
            domain_max_nm=x_max,
            base_cell_nm=adaptive_density.base_cell_nm,
            min_cell_nm=max(mesh_config.min_cell_size_nm, self.thirds_rule.min_cell_nm),
            max_ratio=min(mesh_config.smoothmesh_ratio, self.thirds_rule.max_expansion_ratio),
            refinement_zones=x_zones,
        )
        x_lines = x_generator.generate_lines()

        # Add thirds-rule fixed lines at key X positions
        x_lines = self._add_thirds_rule_lines_x(x_lines, geometry, adaptive_density)

        # Generate Y mesh lines with thirds-rule zones
        y_generator = MeshLineGenerator(
            domain_min_nm=y_min,
            domain_max_nm=y_max,
            base_cell_nm=adaptive_density.base_cell_nm,
            min_cell_nm=max(mesh_config.min_cell_size_nm, self.thirds_rule.min_cell_nm),
            max_ratio=min(mesh_config.smoothmesh_ratio, self.thirds_rule.max_expansion_ratio),
            refinement_zones=y_zones,
        )
        y_lines = y_generator.generate_lines()

        # Add thirds-rule fixed lines at key Y positions
        y_lines = self._add_thirds_rule_lines_y(y_lines, geometry, adaptive_density)

        # Generate Z mesh lines
        z_lines = generate_z_mesh_lines(
            geometry, adaptive_density, self.copper_thickness_nm
        )

        # Build MeshSpec
        from .spec import MeshResolutionSpec, MeshSmoothingSpec

        return MeshSpec(
            resolution=MeshResolutionSpec(
                lambda_resolution=int(1 / mesh_config.min_wavelength_fraction),
                metal_edge_resolution_nm=mesh_config.edge_refinement_nm,
                via_resolution_nm=mesh_config.via_refinement_nm,
                substrate_resolution_nm=mesh_config.substrate_refinement_nm,
            ),
            smoothing=MeshSmoothingSpec(
                max_ratio=self.thirds_rule.max_expansion_ratio,
                smooth_mesh_lines=True,
            ),
            fixed_lines_x_nm=sorted(set(x_lines)),
            fixed_lines_y_nm=sorted(set(y_lines)),
            fixed_lines_z_nm=z_lines,
        )

    def _apply_thirds_rule_to_zones(
        self,
        zones: list[RefinementZone],
        adaptive_density: AdaptiveMeshDensity,
    ) -> list[RefinementZone]:
        """Apply thirds-rule subdivision to refinement zones.

        Creates additional zones at 1/3 and 2/3 positions within
        each original zone for smoother mesh transitions.

        Args:
            zones: Original refinement zones.
            adaptive_density: Adaptive density configuration.

        Returns:
            Extended list of zones with thirds-rule subdivisions.
        """
        result: list[RefinementZone] = []

        for zone in zones:
            # Add original zone
            result.append(zone)

            # Add thirds-rule subdivisions
            third = zone.radius_nm // self.thirds_rule.divisions

            for i in range(1, self.thirds_rule.divisions):
                # Subdivision at each third position
                offset = i * third

                # Positive side
                result.append(
                    RefinementZone(
                        center_nm=zone.center_nm + offset,
                        radius_nm=third // 2,
                        cell_size_nm=zone.cell_size_nm,
                        axis=zone.axis,
                    )
                )

                # Negative side
                result.append(
                    RefinementZone(
                        center_nm=zone.center_nm - offset,
                        radius_nm=third // 2,
                        cell_size_nm=zone.cell_size_nm,
                        axis=zone.axis,
                    )
                )

        return result

    def _add_thirds_rule_lines_x(
        self,
        lines: list[int],
        geometry: GeometrySpec,
        adaptive_density: AdaptiveMeshDensity,
    ) -> list[int]:
        """Add thirds-rule mesh lines at key X positions.

        Places fixed mesh lines at thirds positions around:
        - Via transition (discontinuity center)
        - Transmission line terminations

        Args:
            lines: Existing mesh lines.
            geometry: Coupon geometry.
            adaptive_density: Adaptive density config.

        Returns:
            Extended list of mesh lines.
        """
        result = set(lines)

        if geometry.discontinuity is not None:
            # Via center is at transmission line junction
            via_x = geometry.transmission_line.length_left_nm

            # Get via radius from discontinuity parameters
            params = geometry.discontinuity.parameters_nm
            via_pad_diameter = params.get("signal_via.pad_diameter_nm", 900_000)
            via_radius = via_pad_diameter // 2

            # Add thirds-rule lines around via
            for i in range(1, self.thirds_rule.divisions + 1):
                offset = i * via_radius // self.thirds_rule.divisions
                result.add(via_x - offset)
                result.add(via_x + offset)

        return sorted(result)

    def _add_thirds_rule_lines_y(
        self,
        lines: list[int],
        geometry: GeometrySpec,
        adaptive_density: AdaptiveMeshDensity,
    ) -> list[int]:
        """Add thirds-rule mesh lines at key Y positions.

        Places fixed mesh lines at thirds positions around:
        - Trace edges
        - Gap boundaries
        - Via center (y=0)

        Args:
            lines: Existing mesh lines.
            geometry: Coupon geometry.
            adaptive_density: Adaptive density config.

        Returns:
            Extended list of mesh lines.
        """
        result = set(lines)

        tl = geometry.transmission_line
        trace_half_width = tl.w_nm // 2
        gap_outer = trace_half_width + tl.gap_nm

        # Add thirds-rule lines at trace edges
        for i in range(1, self.thirds_rule.divisions + 1):
            # Around trace edge
            offset = i * tl.gap_nm // self.thirds_rule.divisions
            result.add(trace_half_width + offset)
            result.add(trace_half_width - offset)
            result.add(-trace_half_width + offset)
            result.add(-trace_half_width - offset)

        # Add thirds-rule lines at gap outer edges
        for i in range(1, self.thirds_rule.divisions + 1):
            offset = i * adaptive_density.trace_cell_nm
            result.add(gap_outer + offset)
            result.add(gap_outer - offset)
            result.add(-gap_outer + offset)
            result.add(-gap_outer - offset)

        return sorted(result)

    def build_csx_geometry(
        self,
        geometry: GeometrySpec,
        tracks: list[TrackSegment] | None = None,
        vias: list[Via] | None = None,
        polygons: list[Polygon] | None = None,
    ) -> CSXGeometry:
        """Build CSX geometry from specification.

        Converts geometry spec to CSX primitives. If tracks, vias, and
        polygons are not provided, generates them from the geometry spec.

        Args:
            geometry: Coupon geometry specification.
            tracks: Optional list of track segments.
            vias: Optional list of vias.
            polygons: Optional list of polygons (cutouts, pours).

        Returns:
            CSXGeometry with all primitives.
        """
        # Generate primitives from geometry if not provided
        if tracks is None:
            tracks = self._generate_tracks_from_geometry(geometry)
        if vias is None:
            vias = self._generate_vias_from_geometry(geometry)
        if polygons is None:
            polygons = self._generate_polygons_from_geometry(geometry)

        return build_csx_geometry(
            geometry,
            tracks=tracks,
            vias=vias,
            polygons=polygons,
            copper_thickness_nm=self.copper_thickness_nm,
            include_substrate=True,
            include_via_pads=True,
        )

    def _generate_tracks_from_geometry(self, geometry: GeometrySpec) -> list[TrackSegment]:
        """Generate track segments from geometry specification."""
        tracks: list[TrackSegment] = []
        tl = geometry.transmission_line

        # Left transmission line (from origin to via)
        if tl.length_left_nm > 0:
            tracks.append(
                TrackSegment(
                    start=PositionNM(0, 0),
                    end=PositionNM(tl.length_left_nm, 0),
                    width_nm=tl.w_nm,
                    layer=tl.layer,
                    net_id=1,
                )
            )

        # Right transmission line (from via to end)
        if tl.length_right_nm > 0:
            via_x = tl.length_left_nm
            tracks.append(
                TrackSegment(
                    start=PositionNM(via_x, 0),
                    end=PositionNM(via_x + tl.length_right_nm, 0),
                    width_nm=tl.w_nm,
                    layer=tl.layer,
                    net_id=1,
                )
            )

        return tracks

    def _generate_vias_from_geometry(self, geometry: GeometrySpec) -> list[Via]:
        """Generate vias from geometry specification."""
        vias: list[Via] = []

        if geometry.discontinuity is None:
            return vias

        params = geometry.discontinuity.parameters_nm
        via_x = geometry.transmission_line.length_left_nm

        # Signal via
        drill = params.get("signal_via.drill_nm", 300_000)
        diameter = params.get("signal_via.diameter_nm", 650_000)

        vias.append(
            Via(
                position=PositionNM(via_x, 0),
                diameter_nm=diameter,
                drill_nm=drill,
            )
        )

        # Return vias (if specified)
        return_count = params.get("return_vias.count", 0)
        return_radius = params.get("return_vias.radius_nm", 0)
        return_drill = params.get("return_vias.via.drill_nm", drill)
        return_diameter = params.get("return_vias.via.diameter_nm", diameter)

        if return_count > 0 and return_radius > 0:
            import math

            for i in range(return_count):
                angle = 2 * math.pi * i / return_count
                x = via_x + int(return_radius * math.cos(angle))
                y = int(return_radius * math.sin(angle))
                vias.append(
                    Via(
                        position=PositionNM(x, y),
                        diameter_nm=return_diameter,
                        drill_nm=return_drill,
                    )
                )

        return vias

    def _generate_polygons_from_geometry(self, geometry: GeometrySpec) -> list[Polygon]:
        """Generate polygons (antipads) from geometry specification."""
        polygons: list[Polygon] = []

        if geometry.discontinuity is None:
            return polygons

        params = geometry.discontinuity.parameters_nm
        via_x = geometry.transmission_line.length_left_nm

        # Generate antipad polygons for each layer
        for key, value in params.items():
            if "antipad" in key.lower() and value > 0:
                # Extract layer from key (e.g., "antipad.L2.r_nm" -> "L2")
                parts = key.split(".")
                if len(parts) >= 2:
                    layer_id = parts[1]
                    # Convert layer ID to KiCad layer name
                    layer_name = self._layer_id_to_kicad(layer_id, geometry)

                    # Create circular antipad approximation
                    antipad_vertices = self._create_circular_polygon(
                        center_x=via_x,
                        center_y=0,
                        radius=value,
                        num_segments=16,
                    )

                    polygons.append(
                        Polygon(
                            vertices=antipad_vertices,
                            layer=layer_name,
                            polygon_type=PolygonType.CUTOUT,
                        )
                    )

        return polygons

    def _layer_id_to_kicad(self, layer_id: str, geometry: GeometrySpec) -> str:
        """Convert layer ID to KiCad layer name."""
        if layer_id == "L1":
            return "F.Cu"
        num_layers = geometry.stackup.copper_layers
        if layer_id == f"L{num_layers}":
            return "B.Cu"
        # Internal layers: L2 -> In1.Cu, L3 -> In2.Cu, etc.
        try:
            n = int(layer_id[1:])
            return f"In{n - 1}.Cu"
        except (ValueError, IndexError):
            return layer_id

    def _create_circular_polygon(
        self,
        center_x: int,
        center_y: int,
        radius: int,
        num_segments: int = 16,
    ) -> tuple[PositionNM, ...]:
        """Create a circular polygon approximation."""
        import math

        vertices: list[PositionNM] = []
        for i in range(num_segments):
            angle = 2 * math.pi * i / num_segments
            x = center_x + int(radius * math.cos(angle))
            y = center_y + int(radius * math.sin(angle))
            vertices.append(PositionNM(x, y))

        return tuple(vertices)

    def setup_simulation(
        self,
        manifest_path: Path,
        frequency_range: FrequencyRange | None = None,
    ) -> SimulationSetup:
        """Complete simulation setup from M1 manifest.

        Loads the manifest, reconstructs geometry, generates mesh,
        and builds CSX primitives.

        Args:
            manifest_path: Path to the M1 manifest.json.
            frequency_range: Optional frequency range for simulation.

        Returns:
            Complete SimulationSetup ready for execution.
        """
        # Load and validate manifest
        manifest = self.load_manifest(manifest_path)

        # Reconstruct geometry
        geometry_spec = self.reconstruct_geometry(manifest)

        # Generate mesh with thirds-rule grading
        mesh_spec = self.generate_mesh(
            geometry_spec,
            frequency_range=frequency_range,
        )

        # Build CSX geometry
        csx_geometry = self.build_csx_geometry(geometry_spec)

        # Compute mesh summary
        summary = mesh_line_summary(mesh_spec)

        return SimulationSetup(
            geometry_spec=geometry_spec,
            csx_geometry=csx_geometry,
            mesh_spec=mesh_spec,
            mesh_summary=summary,
            design_hash=manifest["design_hash"],
            coupon_family=manifest["coupon_family"],
        )
