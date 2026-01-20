"""Geometry adapter for transforming M1 ResolvedDesign into openEMS CSX primitives.

This module provides the adapter layer that maps M1 coupon geometry (CPWG traces,
vias, antipads, planes) to CSXCAD 3D primitives for FDTD simulation.

The adapter handles:
- CPWG signal traces -> 3D copper boxes
- Via barrels -> cylinders
- Via pads -> short cylinders
- Ground planes with antipad cutouts -> boxes with polygon voids
- Substrate dielectric -> dielectric boxes

Coordinate Mapping:
- M1 uses 2D coordinates in the XY plane with layer-based Z positioning
- CSX uses full 3D coordinates with explicit Z ranges
- The adapter maps M1 layers to Z coordinates based on stackup thicknesses

Material Mapping:
- Copper traces/vias -> PEC or lossy metal
- Substrate -> dielectric with epsilon_r and loss_tangent from stackup
- Air gaps (antipads) -> air dielectric (epsilon_r = 1.0)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from formula_foundry.coupongen.geom.primitives import (
    Polygon,
    PolygonType,
    TrackSegment,
    Via,
)
from formula_foundry.coupongen.resolve import ResolvedDesign

from .csx_primitives import (
    BoundingBox3D,
    CSXBox,
    CSXCylinder,
    CSXGeometry,
    CSXMaterial,
    CSXPolygon,
    CSXViaPad,
    Point3D,
    air_material,
    copper_material,
    substrate_material,
)
from .geometry import GeometrySpec, layer_positions_nm

if TYPE_CHECKING:
    pass


# Default copper thickness in nm (typically 35um = 1oz copper)
DEFAULT_COPPER_THICKNESS_NM = 35_000

# Priority levels for material stacking
PRIORITY_AIR = 0
PRIORITY_SUBSTRATE = 10
PRIORITY_COPPER = 100
PRIORITY_CUTOUT = 150  # Cutouts override copper


@dataclass(frozen=True, slots=True)
class LayerZInfo:
    """Z-coordinate information for a PCB layer.

    Attributes:
        layer_id: Layer identifier (e.g., "L1", "L2").
        z_bottom_nm: Bottom Z coordinate of the copper in nm.
        z_top_nm: Top Z coordinate of the copper in nm.
        is_top: Whether this is the top copper layer.
        is_bottom: Whether this is the bottom copper layer.
    """

    layer_id: str
    z_bottom_nm: int
    z_top_nm: int
    is_top: bool = False
    is_bottom: bool = False


@dataclass(frozen=True, slots=True)
class StackupZMap:
    """Z-coordinate mapping for a PCB stackup.

    Provides methods to convert M1 layer references to 3D Z coordinates.

    Attributes:
        layers: Mapping of layer ID to Z info.
        total_thickness_nm: Total stackup thickness in nm.
        dielectric_regions: List of dielectric region Z ranges.
    """

    layers: dict[str, LayerZInfo]
    total_thickness_nm: int
    dielectric_regions: tuple[tuple[int, int, float, float], ...]  # (z_bot, z_top, er, tan_d)

    @classmethod
    def from_geometry_spec(
        cls,
        geometry: GeometrySpec,
        copper_thickness_nm: int = DEFAULT_COPPER_THICKNESS_NM,
    ) -> StackupZMap:
        """Build Z-coordinate map from a GeometrySpec.

        Args:
            geometry: GeometrySpec with stackup information.
            copper_thickness_nm: Copper layer thickness in nm.

        Returns:
            StackupZMap for coordinate conversion.
        """
        # Get layer positions from stackup
        positions = layer_positions_nm(geometry.stackup)
        num_layers = geometry.stackup.copper_layers

        layers: dict[str, LayerZInfo] = {}
        dielectric_regions: list[tuple[int, int, float, float]] = []

        er = geometry.stackup.materials.er
        tan_d = geometry.stackup.materials.loss_tangent

        # Build layer Z info
        # Positions are measured from top surface (L1 = 0)
        # We convert to bottom-up Z coordinates for CSX
        max_z = positions.get(f"L{num_layers}", 0)
        total_thickness = max_z + copper_thickness_nm

        for i in range(1, num_layers + 1):
            layer_id = f"L{i}"
            # Position is distance from L1 top
            pos = positions.get(layer_id, 0)

            # Convert to Z coordinate (bottom = 0)
            # L1 is at top, so z_bottom = total_thickness - pos - copper_thickness
            z_bottom = total_thickness - pos - copper_thickness_nm
            z_top = z_bottom + copper_thickness_nm

            layers[layer_id] = LayerZInfo(
                layer_id=layer_id,
                z_bottom_nm=z_bottom,
                z_top_nm=z_top,
                is_top=(i == 1),
                is_bottom=(i == num_layers),
            )

            # Add dielectric region below this layer (except for bottom layer)
            if i < num_layers:
                next_layer_id = f"L{i + 1}"
                next_pos = positions.get(next_layer_id, 0)
                dielectric_z_top = z_bottom  # Top of dielectric is bottom of current copper
                dielectric_z_bottom = total_thickness - next_pos - copper_thickness_nm + copper_thickness_nm
                if dielectric_z_bottom < dielectric_z_top:
                    dielectric_regions.append((dielectric_z_bottom, dielectric_z_top, er, tan_d))

        return cls(
            layers=layers,
            total_thickness_nm=total_thickness,
            dielectric_regions=tuple(dielectric_regions),
        )

    def get_layer_z(self, layer_id: str) -> LayerZInfo:
        """Get Z info for a layer."""
        if layer_id not in self.layers:
            raise KeyError(f"Unknown layer: {layer_id}")
        return self.layers[layer_id]

    def layer_to_kicad_name(self, layer_id: str) -> str:
        """Convert layer ID to KiCad layer name."""
        if layer_id == "L1":
            return "F.Cu"
        if layer_id in self.layers:
            n = int(layer_id[1:])
            if self.layers[layer_id].is_bottom:
                return "B.Cu"
            return f"In{n - 1}.Cu"
        return layer_id


@dataclass(slots=True)
class GeometryAdapter:
    """Adapter for converting M1 geometry to CSX primitives.

    This class provides methods to convert individual M1 primitives and
    complete coupon geometries to CSX representations.

    Attributes:
        stackup_z: Z-coordinate mapping for the stackup.
        copper_material: Material for copper traces.
        substrate_material: Material for dielectric substrate.
        air_material: Material for air/voids.
        copper_thickness_nm: Copper layer thickness.
    """

    stackup_z: StackupZMap
    copper_mat: CSXMaterial = field(default_factory=lambda: copper_material())
    substrate_mat: CSXMaterial | None = None
    air_mat: CSXMaterial = field(default_factory=lambda: air_material())
    copper_thickness_nm: int = DEFAULT_COPPER_THICKNESS_NM

    @classmethod
    def from_geometry_spec(
        cls,
        geometry: GeometrySpec,
        copper_thickness_nm: int = DEFAULT_COPPER_THICKNESS_NM,
    ) -> GeometryAdapter:
        """Create adapter from a GeometrySpec.

        Args:
            geometry: GeometrySpec with stackup and material info.
            copper_thickness_nm: Copper layer thickness in nm.

        Returns:
            Configured GeometryAdapter.
        """
        stackup_z = StackupZMap.from_geometry_spec(geometry, copper_thickness_nm)
        substrate_mat = substrate_material(
            epsilon_r=geometry.stackup.materials.er,
            loss_tangent=geometry.stackup.materials.loss_tangent,
        )
        return cls(
            stackup_z=stackup_z,
            substrate_mat=substrate_mat,
            copper_thickness_nm=copper_thickness_nm,
        )

    def track_to_csx_box(
        self,
        track: TrackSegment,
        layer_id: str = "L1",
        name: str = "",
    ) -> CSXBox:
        """Convert a TrackSegment to a CSX box.

        Args:
            track: M1 TrackSegment primitive.
            layer_id: Layer ID for Z positioning (default "L1").
            name: Optional name for the box.

        Returns:
            CSXBox representing the track in 3D.
        """
        layer_z = self.stackup_z.get_layer_z(layer_id)

        return CSXBox.from_trace(
            x_start=track.start.x,
            y_start=track.start.y,
            x_end=track.end.x,
            y_end=track.end.y,
            z_bottom=layer_z.z_bottom_nm,
            z_top=layer_z.z_top_nm,
            width_nm=track.width_nm,
            material=self.copper_mat,
            name=name or f"track_{layer_id}",
        )

    def via_to_csx_cylinder(
        self,
        via: Via,
        layer_start: str = "L1",
        layer_end: str | None = None,
        name: str = "",
    ) -> CSXCylinder:
        """Convert a Via to a CSX cylinder (via barrel).

        Args:
            via: M1 Via primitive.
            layer_start: Start layer ID.
            layer_end: End layer ID (default: bottom layer).
            name: Optional name for the cylinder.

        Returns:
            CSXCylinder representing the via barrel.
        """
        start_z = self.stackup_z.get_layer_z(layer_start)

        if layer_end is None:
            # Find bottom layer
            for lid, info in self.stackup_z.layers.items():
                if info.is_bottom:
                    layer_end = lid
                    break
            if layer_end is None:
                layer_end = layer_start

        end_z = self.stackup_z.get_layer_z(layer_end)

        # Via barrel spans from top of start layer to bottom of end layer
        z_top = start_z.z_top_nm
        z_bottom = end_z.z_bottom_nm

        return CSXCylinder.from_via(
            x=via.position.x,
            y=via.position.y,
            z_bottom=z_bottom,
            z_top=z_top,
            drill_nm=via.drill_nm,
            material=self.copper_mat,
            name=name or "via_barrel",
        )

    def via_to_csx_pads(
        self,
        via: Via,
        layers: tuple[str, ...] | None = None,
        name: str = "",
    ) -> list[CSXViaPad]:
        """Convert a Via to CSX pads on specified layers.

        Args:
            via: M1 Via primitive.
            layers: Tuple of layer IDs for pads (default: all layers).
            name: Optional base name for the pads.

        Returns:
            List of CSXViaPad primitives.
        """
        if layers is None:
            layers = tuple(self.stackup_z.layers.keys())

        pads: list[CSXViaPad] = []
        for layer_id in layers:
            layer_z = self.stackup_z.get_layer_z(layer_id)
            pad = CSXViaPad(
                center=Point3D(via.position.x, via.position.y, layer_z.z_bottom_nm),
                radius_nm=via.diameter_nm // 2,
                thickness_nm=self.copper_thickness_nm,
                material=self.copper_mat,
                name=f"{name or 'via_pad'}_{layer_id}",
            )
            pads.append(pad)

        return pads

    def polygon_to_csx(
        self,
        polygon: Polygon,
        layer_id: str = "L1",
        name: str = "",
    ) -> CSXPolygon:
        """Convert a M1 Polygon to a CSX polygon.

        Args:
            polygon: M1 Polygon primitive.
            layer_id: Layer ID for Z positioning.
            name: Optional name for the polygon.

        Returns:
            CSXPolygon representing the shape in 3D.
        """
        layer_z = self.stackup_z.get_layer_z(layer_id)

        # Determine material based on polygon type
        if polygon.polygon_type == PolygonType.CUTOUT:
            material = self.air_mat
        elif polygon.polygon_type == PolygonType.COPPER_POUR:
            material = self.copper_mat
        else:
            material = self.air_mat

        vertices_xy = tuple((v.x, v.y) for v in polygon.vertices)

        return CSXPolygon(
            vertices_xy=vertices_xy,
            z_bottom=layer_z.z_bottom_nm,
            z_top=layer_z.z_top_nm,
            material=material,
            name=name or f"polygon_{layer_id}",
        )

    def create_ground_plane(
        self,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        layer_id: str,
        name: str = "",
    ) -> CSXBox:
        """Create a ground plane as a CSX box.

        Args:
            x_min: Minimum X coordinate in nm.
            y_min: Minimum Y coordinate in nm.
            x_max: Maximum X coordinate in nm.
            y_max: Maximum Y coordinate in nm.
            layer_id: Layer ID for Z positioning.
            name: Optional name for the plane.

        Returns:
            CSXBox representing the ground plane.
        """
        layer_z = self.stackup_z.get_layer_z(layer_id)

        bbox = BoundingBox3D.from_corners(
            x_min,
            y_min,
            layer_z.z_bottom_nm,
            x_max,
            y_max,
            layer_z.z_top_nm,
        )

        return CSXBox(
            bbox=bbox,
            material=self.copper_mat,
            name=name or f"ground_{layer_id}",
        )

    def create_substrate_region(
        self,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        z_bottom: int,
        z_top: int,
        name: str = "",
    ) -> CSXBox | None:
        """Create a substrate dielectric region.

        Args:
            x_min: Minimum X coordinate in nm.
            y_min: Minimum Y coordinate in nm.
            x_max: Maximum X coordinate in nm.
            y_max: Maximum Y coordinate in nm.
            z_bottom: Bottom Z coordinate in nm.
            z_top: Top Z coordinate in nm.
            name: Optional name for the region.

        Returns:
            CSXBox for the substrate, or None if no substrate material defined.
        """
        if self.substrate_mat is None:
            return None

        bbox = BoundingBox3D.from_corners(x_min, y_min, z_bottom, x_max, y_max, z_top)

        return CSXBox(
            bbox=bbox,
            material=self.substrate_mat,
            name=name or "substrate",
        )


def build_csx_geometry(
    geometry_spec: GeometrySpec,
    tracks: list[TrackSegment],
    vias: list[Via],
    polygons: list[Polygon],
    *,
    copper_thickness_nm: int = DEFAULT_COPPER_THICKNESS_NM,
    include_substrate: bool = True,
    include_via_pads: bool = True,
) -> CSXGeometry:
    """Build complete CSX geometry from M1 primitives.

    This function converts all M1 primitives to CSX format and assembles
    them into a complete geometry suitable for openEMS simulation.

    Args:
        geometry_spec: GeometrySpec with stackup and board info.
        tracks: List of TrackSegment primitives.
        vias: List of Via primitives.
        polygons: List of Polygon primitives.
        copper_thickness_nm: Copper layer thickness in nm.
        include_substrate: Whether to include substrate dielectric regions.
        include_via_pads: Whether to include via pads on each layer.

    Returns:
        CSXGeometry containing all converted primitives.
    """
    adapter = GeometryAdapter.from_geometry_spec(geometry_spec, copper_thickness_nm)
    csx_geom = CSXGeometry()

    # Add materials
    csx_geom.add_material(adapter.copper_mat)
    csx_geom.add_material(adapter.air_mat)
    if adapter.substrate_mat:
        csx_geom.add_material(adapter.substrate_mat)

    # Convert tracks
    for i, track in enumerate(tracks):
        # Determine layer from track.layer (KiCad name to layer ID)
        layer_id = _kicad_layer_to_id(track.layer)
        csx_box = adapter.track_to_csx_box(track, layer_id, name=f"track_{i}")
        csx_geom.add_primitive(csx_box)

    # Convert vias
    for i, via in enumerate(vias):
        # Via barrel
        cylinder = adapter.via_to_csx_cylinder(via, name=f"via_{i}")
        csx_geom.add_primitive(cylinder)

        # Via pads on each layer
        if include_via_pads:
            pads = adapter.via_to_csx_pads(via, name=f"via_{i}")
            for pad in pads:
                csx_geom.add_primitive(pad)

    # Convert polygons (cutouts, copper pours)
    for i, poly in enumerate(polygons):
        layer_id = _kicad_layer_to_id(poly.layer)
        csx_poly = adapter.polygon_to_csx(poly, layer_id, name=f"polygon_{i}")
        csx_geom.add_primitive(csx_poly)

    # Add substrate regions
    if include_substrate and csx_geom.bbox is not None:
        for z_bot, z_top, _er, _tan_d in adapter.stackup_z.dielectric_regions:
            substrate_box = adapter.create_substrate_region(
                csx_geom.bbox.min_pt.x,
                csx_geom.bbox.min_pt.y,
                csx_geom.bbox.max_pt.x,
                csx_geom.bbox.max_pt.y,
                z_bot,
                z_top,
                name="substrate",
            )
            if substrate_box:
                csx_geom.add_primitive(substrate_box)

    return csx_geom


def _kicad_layer_to_id(kicad_layer: str) -> str:
    """Convert KiCad layer name to layer ID.

    Args:
        kicad_layer: KiCad layer name (e.g., "F.Cu", "B.Cu", "In1.Cu").

    Returns:
        Layer ID (e.g., "L1", "L2", "L4").
    """
    if kicad_layer == "F.Cu":
        return "L1"
    if kicad_layer == "B.Cu":
        # Assume 4-layer default; could be parameterized
        return "L4"
    if kicad_layer.startswith("In"):
        # In1.Cu -> L2, In2.Cu -> L3
        try:
            num = int(kicad_layer[2:].split(".")[0])
            return f"L{num + 1}"
        except (ValueError, IndexError):
            pass
    return "L1"  # Default fallback


def build_csx_geometry_from_resolved(
    resolved: ResolvedDesign,
    manifest: Mapping[str, Any],
    tracks: list[TrackSegment],
    vias: list[Via],
    polygons: list[Polygon],
    *,
    copper_thickness_nm: int = DEFAULT_COPPER_THICKNESS_NM,
    include_substrate: bool = True,
    include_via_pads: bool = True,
) -> CSXGeometry:
    """Build CSX geometry from ResolvedDesign and M1 primitives.

    This is a convenience function that builds the GeometrySpec first
    and then constructs the CSX geometry.

    Args:
        resolved: M1 ResolvedDesign.
        manifest: Manifest dictionary with stackup info.
        tracks: List of TrackSegment primitives.
        vias: List of Via primitives.
        polygons: List of Polygon primitives.
        copper_thickness_nm: Copper layer thickness in nm.
        include_substrate: Whether to include substrate dielectric regions.
        include_via_pads: Whether to include via pads on each layer.

    Returns:
        CSXGeometry containing all converted primitives.
    """
    from .geometry import build_geometry_spec

    geometry_spec = build_geometry_spec(resolved, manifest)

    return build_csx_geometry(
        geometry_spec,
        tracks,
        vias,
        polygons,
        copper_thickness_nm=copper_thickness_nm,
        include_substrate=include_substrate,
        include_via_pads=include_via_pads,
    )
