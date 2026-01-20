"""CSX primitive dataclasses for openEMS CSXCAD geometry.

This module defines the internal representation for CSXCAD 3D geometry primitives.
These primitives are used to build the FDTD simulation geometry from M1 ResolvedDesign
coupon geometry.

All coordinates use integer nanometers (nm) internally for consistency with M1,
but conversion to meters is provided for CSXCAD output.

CSXCAD supports several primitive types:
- Box: Axis-aligned 3D box (most efficient)
- Cylinder: Cylindrical primitive for round vias
- Polygon: 2.5D extruded polygon for complex shapes
- LinPoly: Linear polygon for arbitrary shapes

For coupon geometries, we primarily use:
- Box for copper traces (CPWG signal, ground planes)
- Cylinder for via barrels
- Polygon for antipad cutouts and non-rectangular shapes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Conversion factor: nanometers to meters
NM_TO_M = 1e-9


class CSXMaterialType(Enum):
    """CSXCAD material types."""

    METAL = "metal"
    """Perfect electric conductor (PEC) or lossy metal."""

    DIELECTRIC = "dielectric"
    """Dielectric material with epsilon_r and loss tangent."""

    EXCITATION = "excitation"
    """Excitation region (port)."""

    PROBE = "probe"
    """Field probe region."""

    DUMP = "dump"
    """Field dump region for visualization."""


class CSXPrimitiveType(Enum):
    """CSXCAD primitive geometry types."""

    BOX = "box"
    """Axis-aligned 3D box."""

    CYLINDER = "cylinder"
    """Cylinder with circular cross-section."""

    POLYGON = "polygon"
    """2.5D extruded polygon."""

    LINPOLY = "linpoly"
    """Linear polygon (arbitrary 3D shape)."""


@dataclass(frozen=True, slots=True)
class Point3D:
    """3D point in nanometers.

    Attributes:
        x: X coordinate in nm.
        y: Y coordinate in nm.
        z: Z coordinate in nm.
    """

    x: int
    y: int
    z: int

    def to_meters(self) -> tuple[float, float, float]:
        """Convert to meters for CSXCAD output."""
        return (self.x * NM_TO_M, self.y * NM_TO_M, self.z * NM_TO_M)

    def __add__(self, other: Point3D) -> Point3D:
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Point3D) -> Point3D:
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)


@dataclass(frozen=True, slots=True)
class BoundingBox3D:
    """3D axis-aligned bounding box in nanometers.

    Attributes:
        min_pt: Minimum corner point.
        max_pt: Maximum corner point.
    """

    min_pt: Point3D
    max_pt: Point3D

    @classmethod
    def from_corners(
        cls,
        x1: int,
        y1: int,
        z1: int,
        x2: int,
        y2: int,
        z2: int,
    ) -> BoundingBox3D:
        """Create bounding box from two corner coordinates."""
        return cls(
            min_pt=Point3D(min(x1, x2), min(y1, y2), min(z1, z2)),
            max_pt=Point3D(max(x1, x2), max(y1, y2), max(z1, z2)),
        )

    def to_meters(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Convert to meters for CSXCAD output."""
        return (self.min_pt.to_meters(), self.max_pt.to_meters())

    @property
    def size_nm(self) -> tuple[int, int, int]:
        """Return size in each dimension (nm)."""
        return (
            self.max_pt.x - self.min_pt.x,
            self.max_pt.y - self.min_pt.y,
            self.max_pt.z - self.min_pt.z,
        )


@dataclass(frozen=True, slots=True)
class CSXMaterial:
    """CSXCAD material definition.

    Attributes:
        name: Unique material name.
        material_type: Material type (metal, dielectric, etc.).
        priority: Priority for overlapping regions (higher wins).
        conductivity: Conductivity in S/m (for lossy metal).
        epsilon_r: Relative permittivity (for dielectric).
        loss_tangent: Dielectric loss tangent.
        kappa: Dielectric conductivity in S/m.
    """

    name: str
    material_type: CSXMaterialType
    priority: int = 0
    conductivity: float | None = None
    epsilon_r: float | None = None
    loss_tangent: float | None = None
    kappa: float | None = None


@dataclass(frozen=True, slots=True)
class CSXBox:
    """3D axis-aligned box primitive.

    Used for copper traces, ground planes, and rectangular shapes.

    Attributes:
        bbox: Bounding box defining the box geometry.
        material: Material for this box.
        name: Optional name for identification.
    """

    bbox: BoundingBox3D
    material: CSXMaterial
    name: str = ""

    @classmethod
    def from_trace(
        cls,
        x_start: int,
        y_start: int,
        x_end: int,
        y_end: int,
        z_bottom: int,
        z_top: int,
        width_nm: int,
        material: CSXMaterial,
        name: str = "",
    ) -> CSXBox:
        """Create a box from a trace segment.

        For horizontal/vertical traces, expands the trace width symmetrically.

        Args:
            x_start: Start X coordinate in nm.
            y_start: Start Y coordinate in nm.
            x_end: End X coordinate in nm.
            y_end: End Y coordinate in nm.
            z_bottom: Bottom Z coordinate in nm.
            z_top: Top Z coordinate in nm.
            width_nm: Trace width in nm.
            material: Material for the trace.
            name: Optional name.

        Returns:
            CSXBox representing the trace.
        """
        half_width = width_nm // 2

        # Determine trace orientation and expand perpendicular to direction
        if abs(x_end - x_start) >= abs(y_end - y_start):
            # Primarily horizontal trace - expand in Y
            y_min = min(y_start, y_end) - half_width
            y_max = max(y_start, y_end) + half_width
            x_min = min(x_start, x_end)
            x_max = max(x_start, x_end)
        else:
            # Primarily vertical trace - expand in X
            x_min = min(x_start, x_end) - half_width
            x_max = max(x_start, x_end) + half_width
            y_min = min(y_start, y_end)
            y_max = max(y_start, y_end)

        bbox = BoundingBox3D.from_corners(x_min, y_min, z_bottom, x_max, y_max, z_top)
        return cls(bbox=bbox, material=material, name=name)


@dataclass(frozen=True, slots=True)
class CSXCylinder:
    """Cylindrical primitive.

    Used for via barrels and round structures.

    Attributes:
        center_bottom: Center point at bottom of cylinder.
        center_top: Center point at top of cylinder.
        radius_nm: Cylinder radius in nm.
        material: Material for this cylinder.
        name: Optional name for identification.
    """

    center_bottom: Point3D
    center_top: Point3D
    radius_nm: int
    material: CSXMaterial
    name: str = ""

    @classmethod
    def from_via(
        cls,
        x: int,
        y: int,
        z_bottom: int,
        z_top: int,
        drill_nm: int,
        material: CSXMaterial,
        name: str = "",
    ) -> CSXCylinder:
        """Create a cylinder from a via specification.

        Args:
            x: Via center X coordinate in nm.
            y: Via center Y coordinate in nm.
            z_bottom: Bottom Z coordinate in nm.
            z_top: Top Z coordinate in nm.
            drill_nm: Via drill diameter in nm.
            material: Material for the via barrel.
            name: Optional name.

        Returns:
            CSXCylinder representing the via barrel.
        """
        return cls(
            center_bottom=Point3D(x, y, z_bottom),
            center_top=Point3D(x, y, z_top),
            radius_nm=drill_nm // 2,
            material=material,
            name=name,
        )


@dataclass(frozen=True, slots=True)
class CSXPolygon:
    """2.5D extruded polygon primitive.

    Used for antipads, cutouts, and non-rectangular copper shapes.

    Attributes:
        vertices_xy: Sequence of (x, y) vertex coordinates in nm.
        z_bottom: Bottom Z coordinate in nm.
        z_top: Top Z coordinate in nm.
        material: Material for this polygon.
        name: Optional name for identification.
    """

    vertices_xy: tuple[tuple[int, int], ...]
    z_bottom: int
    z_top: int
    material: CSXMaterial
    name: str = ""

    def vertices_to_meters(self) -> list[tuple[float, float]]:
        """Convert vertex coordinates to meters."""
        return [(x * NM_TO_M, y * NM_TO_M) for x, y in self.vertices_xy]


@dataclass(frozen=True, slots=True)
class CSXViaPad:
    """Via pad primitive (circular pad on a layer).

    Represented as a short cylinder for the pad metal.

    Attributes:
        center: Center point of the pad.
        radius_nm: Pad radius in nm.
        thickness_nm: Metal thickness in nm.
        material: Material for the pad.
        name: Optional name.
    """

    center: Point3D
    radius_nm: int
    thickness_nm: int
    material: CSXMaterial
    name: str = ""


# Type alias for any CSX primitive
CSXPrimitive = CSXBox | CSXCylinder | CSXPolygon | CSXViaPad


@dataclass(slots=True)
class CSXGeometry:
    """Collection of CSX primitives forming a complete geometry.

    Attributes:
        materials: Dictionary of material definitions by name.
        primitives: List of all primitives in the geometry.
        bbox: Overall bounding box of the geometry.
    """

    materials: dict[str, CSXMaterial] = field(default_factory=dict)
    primitives: list[CSXPrimitive] = field(default_factory=list)
    bbox: BoundingBox3D | None = None

    def add_material(self, material: CSXMaterial) -> None:
        """Add a material definition."""
        self.materials[material.name] = material

    def add_primitive(self, primitive: CSXPrimitive) -> None:
        """Add a primitive and update the bounding box."""
        self.primitives.append(primitive)
        self._update_bbox(primitive)

    def _update_bbox(self, primitive: CSXPrimitive) -> None:
        """Update overall bounding box to include the new primitive."""
        if isinstance(primitive, CSXBox):
            prim_bbox = primitive.bbox
        elif isinstance(primitive, CSXCylinder):
            r = primitive.radius_nm
            prim_bbox = BoundingBox3D(
                min_pt=Point3D(
                    primitive.center_bottom.x - r,
                    primitive.center_bottom.y - r,
                    primitive.center_bottom.z,
                ),
                max_pt=Point3D(
                    primitive.center_top.x + r,
                    primitive.center_top.y + r,
                    primitive.center_top.z,
                ),
            )
        elif isinstance(primitive, CSXPolygon):
            xs = [v[0] for v in primitive.vertices_xy]
            ys = [v[1] for v in primitive.vertices_xy]
            prim_bbox = BoundingBox3D(
                min_pt=Point3D(min(xs), min(ys), primitive.z_bottom),
                max_pt=Point3D(max(xs), max(ys), primitive.z_top),
            )
        elif isinstance(primitive, CSXViaPad):
            r = primitive.radius_nm
            prim_bbox = BoundingBox3D(
                min_pt=Point3D(
                    primitive.center.x - r,
                    primitive.center.y - r,
                    primitive.center.z,
                ),
                max_pt=Point3D(
                    primitive.center.x + r,
                    primitive.center.y + r,
                    primitive.center.z + primitive.thickness_nm,
                ),
            )
        else:
            return

        if self.bbox is None:
            self.bbox = prim_bbox
        else:
            self.bbox = BoundingBox3D(
                min_pt=Point3D(
                    min(self.bbox.min_pt.x, prim_bbox.min_pt.x),
                    min(self.bbox.min_pt.y, prim_bbox.min_pt.y),
                    min(self.bbox.min_pt.z, prim_bbox.min_pt.z),
                ),
                max_pt=Point3D(
                    max(self.bbox.max_pt.x, prim_bbox.max_pt.x),
                    max(self.bbox.max_pt.y, prim_bbox.max_pt.y),
                    max(self.bbox.max_pt.z, prim_bbox.max_pt.z),
                ),
            )


# Default material definitions for PCB structures
def copper_material(priority: int = 100, name: str = "copper") -> CSXMaterial:
    """Create copper material definition for PEC approximation."""
    return CSXMaterial(
        name=name,
        material_type=CSXMaterialType.METAL,
        priority=priority,
        conductivity=5.8e7,  # Copper conductivity S/m
    )


def substrate_material(
    epsilon_r: float,
    loss_tangent: float = 0.0,
    priority: int = 10,
    name: str = "substrate",
) -> CSXMaterial:
    """Create substrate dielectric material definition."""
    return CSXMaterial(
        name=name,
        material_type=CSXMaterialType.DIELECTRIC,
        priority=priority,
        epsilon_r=epsilon_r,
        loss_tangent=loss_tangent,
    )


def air_material(priority: int = 0, name: str = "air") -> CSXMaterial:
    """Create air material definition (for cutouts/voids)."""
    return CSXMaterial(
        name=name,
        material_type=CSXMaterialType.DIELECTRIC,
        priority=priority,
        epsilon_r=1.0,
        loss_tangent=0.0,
    )
