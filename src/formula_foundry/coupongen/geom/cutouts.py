"""Antipad and plane cutout shape generators.

This module provides functions to generate cutout shapes for antipads and plane
clearances around vias and other structures. Supported shapes include:
- Circle: Simple circular antipad
- RoundRect: Rounded rectangle for oval-shaped vias or larger clearances
- Slot: Elongated slot for via transitions or thermal relief
- Polygon: Custom polygon shapes

All coordinates use integer nanometers (LengthNM) to ensure determinism and
avoid cross-platform rounding drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .primitives import Polygon, PolygonType, PositionNM

if TYPE_CHECKING:
    pass


class CutoutShape(Enum):
    """Supported antipad and cutout shapes."""

    CIRCLE = "CIRCLE"
    """Circular cutout (approximated as polygon)."""

    ROUNDRECT = "ROUNDRECT"
    """Rounded rectangle cutout."""

    SLOT = "SLOT"
    """Slot-shaped cutout (stadium/obround)."""

    RECTANGLE = "RECTANGLE"
    """Simple rectangular cutout."""

    OBROUND = "OBROUND"
    """Obround (stadium) shape - same as SLOT but different naming convention."""


@dataclass(frozen=True, slots=True)
class CircleAntipadSpec:
    """Specification for a circular antipad.

    Attributes:
        shape: Shape type (should be CIRCLE).
        r_nm: Radius in nanometers.
        layer: Layer for the cutout.
        segments: Number of polygon segments to approximate the circle.
    """

    shape: CutoutShape
    r_nm: int
    layer: str
    segments: int = 32


@dataclass(frozen=True, slots=True)
class RoundRectAntipadSpec:
    """Specification for a rounded rectangle antipad.

    Attributes:
        shape: Shape type (should be ROUNDRECT).
        rx_nm: Half-width in x direction in nanometers.
        ry_nm: Half-height in y direction in nanometers.
        corner_nm: Corner radius in nanometers.
        layer: Layer for the cutout.
        corner_segments: Number of segments per corner arc.
    """

    shape: CutoutShape
    rx_nm: int
    ry_nm: int
    corner_nm: int
    layer: str
    corner_segments: int = 8


@dataclass(frozen=True, slots=True)
class SlotAntipadSpec:
    """Specification for a slot-shaped antipad.

    Attributes:
        shape: Shape type (should be SLOT).
        length_nm: Total length of the slot in nanometers.
        width_nm: Width of the slot in nanometers.
        rotation_mdeg: Rotation in millidegrees (0 = horizontal).
        layer: Layer for the cutout.
        end_segments: Number of segments for each semicircular end.
    """

    shape: CutoutShape
    length_nm: int
    width_nm: int
    rotation_mdeg: int
    layer: str
    end_segments: int = 16


@dataclass(frozen=True, slots=True)
class RectangleAntipadSpec:
    """Specification for a simple rectangular antipad.

    Attributes:
        shape: Shape type (should be RECTANGLE).
        width_nm: Width in x direction in nanometers.
        height_nm: Height in y direction in nanometers.
        layer: Layer for the cutout.
    """

    shape: CutoutShape
    width_nm: int
    height_nm: int
    layer: str


# Type alias for any antipad spec
AntipadSpec = CircleAntipadSpec | RoundRectAntipadSpec | SlotAntipadSpec | RectangleAntipadSpec


def generate_circle_antipad(
    center: PositionNM,
    spec: CircleAntipadSpec,
) -> Polygon:
    """Generate a circular antipad polygon.

    The circle is approximated as a regular polygon with the specified number
    of segments.

    Args:
        center: Center position in nm.
        spec: Circle antipad specification.

    Returns:
        Polygon primitive for the cutout.
    """
    if spec.r_nm <= 0:
        raise ValueError("r_nm must be positive")

    if spec.segments < 3:
        raise ValueError("segments must be at least 3")

    vertices: list[PositionNM] = []
    angle_step = 2 * math.pi / spec.segments

    for i in range(spec.segments):
        angle = i * angle_step
        x = center.x + int(spec.r_nm * math.cos(angle))
        y = center.y + int(spec.r_nm * math.sin(angle))
        vertices.append(PositionNM(x, y))

    return Polygon(
        vertices=tuple(vertices),
        layer=spec.layer,
        polygon_type=PolygonType.CUTOUT,
    )


def generate_roundrect_antipad(
    center: PositionNM,
    spec: RoundRectAntipadSpec,
) -> Polygon:
    """Generate a rounded rectangle antipad polygon.

    Creates a rectangle with rounded corners. The corner radius is clamped
    to the minimum of rx_nm and ry_nm.

    Args:
        center: Center position in nm.
        spec: Rounded rectangle antipad specification.

    Returns:
        Polygon primitive for the cutout.
    """
    if spec.rx_nm <= 0 or spec.ry_nm <= 0:
        raise ValueError("rx_nm and ry_nm must be positive")

    if spec.corner_segments < 1:
        raise ValueError("corner_segments must be at least 1")

    # Clamp corner radius
    corner_r = min(spec.corner_nm, spec.rx_nm, spec.ry_nm)
    if corner_r < 0:
        corner_r = 0

    vertices: list[PositionNM] = []

    # Inner rectangle dimensions (from center to where arcs start)
    inner_rx = spec.rx_nm - corner_r
    inner_ry = spec.ry_nm - corner_r

    # Generate corners with arcs
    # Start from top-right corner and go counterclockwise

    def add_corner_arc(
        corner_center_x: int,
        corner_center_y: int,
        start_angle: float,
        end_angle: float,
    ) -> None:
        """Add vertices for a corner arc."""
        if corner_r <= 0:
            # Sharp corner
            vertices.append(PositionNM(corner_center_x, corner_center_y))
            return

        angle_step = (end_angle - start_angle) / spec.corner_segments
        for i in range(spec.corner_segments + 1):
            angle = start_angle + i * angle_step
            x = corner_center_x + int(corner_r * math.cos(angle))
            y = corner_center_y + int(corner_r * math.sin(angle))
            vertices.append(PositionNM(x, y))

    # Top-right corner (0 to 90 degrees = π/2)
    add_corner_arc(
        center.x + inner_rx,
        center.y + inner_ry,
        0,
        math.pi / 2,
    )

    # Top-left corner (90 to 180 degrees)
    add_corner_arc(
        center.x - inner_rx,
        center.y + inner_ry,
        math.pi / 2,
        math.pi,
    )

    # Bottom-left corner (180 to 270 degrees)
    add_corner_arc(
        center.x - inner_rx,
        center.y - inner_ry,
        math.pi,
        3 * math.pi / 2,
    )

    # Bottom-right corner (270 to 360 degrees)
    add_corner_arc(
        center.x + inner_rx,
        center.y - inner_ry,
        3 * math.pi / 2,
        2 * math.pi,
    )

    return Polygon(
        vertices=tuple(vertices),
        layer=spec.layer,
        polygon_type=PolygonType.CUTOUT,
    )


def generate_slot_antipad(
    center: PositionNM,
    spec: SlotAntipadSpec,
) -> Polygon:
    """Generate a slot-shaped (stadium/obround) antipad polygon.

    Creates an elongated shape with semicircular ends. The slot can be rotated
    around its center.

    Args:
        center: Center position in nm.
        spec: Slot antipad specification.

    Returns:
        Polygon primitive for the cutout.
    """
    if spec.length_nm <= 0 or spec.width_nm <= 0:
        raise ValueError("length_nm and width_nm must be positive")

    if spec.end_segments < 2:
        raise ValueError("end_segments must be at least 2")

    # Slot dimensions
    half_length = spec.length_nm // 2
    radius = spec.width_nm // 2

    # If length <= width, it's essentially a circle
    if half_length <= radius:
        # Generate a circle instead
        circle_spec = CircleAntipadSpec(
            shape=CutoutShape.CIRCLE,
            r_nm=radius,
            layer=spec.layer,
            segments=spec.end_segments * 2,
        )
        return generate_circle_antipad(center, circle_spec)

    # Straight section half-length
    straight_half = half_length - radius

    vertices: list[PositionNM] = []
    rotation_rad = spec.rotation_mdeg / 1000.0 * math.pi / 180

    def rotate_point(x: int, y: int) -> PositionNM:
        """Rotate a point around the center."""
        cos_a = math.cos(rotation_rad)
        sin_a = math.sin(rotation_rad)
        rx = int(x * cos_a - y * sin_a)
        ry = int(x * sin_a + y * cos_a)
        return PositionNM(center.x + rx, center.y + ry)

    # Generate right semicircle (centered at +straight_half, 0)
    # From -90 to +90 degrees
    angle_step = math.pi / spec.end_segments
    for i in range(spec.end_segments + 1):
        angle = -math.pi / 2 + i * angle_step
        local_x = straight_half + int(radius * math.cos(angle))
        local_y = int(radius * math.sin(angle))
        vertices.append(rotate_point(local_x, local_y))

    # Generate left semicircle (centered at -straight_half, 0)
    # From +90 to +270 degrees
    for i in range(spec.end_segments + 1):
        angle = math.pi / 2 + i * angle_step
        local_x = -straight_half + int(radius * math.cos(angle))
        local_y = int(radius * math.sin(angle))
        vertices.append(rotate_point(local_x, local_y))

    return Polygon(
        vertices=tuple(vertices),
        layer=spec.layer,
        polygon_type=PolygonType.CUTOUT,
    )


def generate_rectangle_antipad(
    center: PositionNM,
    spec: RectangleAntipadSpec,
) -> Polygon:
    """Generate a simple rectangular antipad polygon.

    Args:
        center: Center position in nm.
        spec: Rectangle antipad specification.

    Returns:
        Polygon primitive for the cutout.
    """
    if spec.width_nm <= 0 or spec.height_nm <= 0:
        raise ValueError("width_nm and height_nm must be positive")

    half_w = spec.width_nm // 2
    half_h = spec.height_nm // 2

    vertices = (
        PositionNM(center.x + half_w, center.y + half_h),
        PositionNM(center.x - half_w, center.y + half_h),
        PositionNM(center.x - half_w, center.y - half_h),
        PositionNM(center.x + half_w, center.y - half_h),
    )

    return Polygon(
        vertices=vertices,
        layer=spec.layer,
        polygon_type=PolygonType.CUTOUT,
    )


def generate_antipad(
    center: PositionNM,
    spec: AntipadSpec,
) -> Polygon:
    """Generate an antipad polygon based on the specification type.

    This is a convenience function that dispatches to the appropriate
    generator based on the spec's shape field.

    Args:
        center: Center position in nm.
        spec: Antipad specification (any supported type).

    Returns:
        Polygon primitive for the cutout.
    """
    if isinstance(spec, CircleAntipadSpec):
        return generate_circle_antipad(center, spec)
    elif isinstance(spec, RoundRectAntipadSpec):
        return generate_roundrect_antipad(center, spec)
    elif isinstance(spec, SlotAntipadSpec):
        return generate_slot_antipad(center, spec)
    elif isinstance(spec, RectangleAntipadSpec):
        return generate_rectangle_antipad(center, spec)
    else:
        raise ValueError(f"Unsupported antipad spec type: {type(spec)}")


def generate_plane_cutout_for_via(
    via_center: PositionNM,
    via_diameter_nm: int,
    clearance_nm: int,
    layer: str,
    shape: CutoutShape = CutoutShape.CIRCLE,
    segments: int = 32,
) -> Polygon:
    """Generate a plane cutout (antipad) centered on a via.

    This is a convenience function for the common case of creating a circular
    antipad around a via with a specified clearance.

    Args:
        via_center: Via center position in nm.
        via_diameter_nm: Via pad diameter in nm.
        clearance_nm: Additional clearance beyond the via edge in nm.
        layer: Layer for the cutout.
        shape: Cutout shape (default CIRCLE).
        segments: Number of segments for circular approximation.

    Returns:
        Polygon primitive for the cutout.
    """
    if clearance_nm < 0:
        raise ValueError("clearance_nm must be non-negative")

    # Total radius = via radius + clearance
    total_radius = via_diameter_nm // 2 + clearance_nm

    if shape == CutoutShape.CIRCLE:
        spec = CircleAntipadSpec(
            shape=CutoutShape.CIRCLE,
            r_nm=total_radius,
            layer=layer,
            segments=segments,
        )
        return generate_circle_antipad(via_center, spec)
    elif shape == CutoutShape.RECTANGLE:
        spec = RectangleAntipadSpec(
            shape=CutoutShape.RECTANGLE,
            width_nm=total_radius * 2,
            height_nm=total_radius * 2,
            layer=layer,
        )
        return generate_rectangle_antipad(via_center, spec)
    else:
        raise ValueError(f"Unsupported shape for via cutout: {shape}")


class OutlineElement:
    """Base class for board outline elements."""

    pass


@dataclass(frozen=True, slots=True)
class OutlineLine:
    """Straight line segment for board outline.

    Attributes:
        start: Start position in nanometers.
        end: End position in nanometers.
    """

    start: PositionNM
    end: PositionNM


@dataclass(frozen=True, slots=True)
class OutlineArc:
    """Arc segment for board outline.

    KiCad uses start/mid/end format where 'mid' is a point on the arc.

    Attributes:
        start: Start position in nanometers.
        mid: Midpoint on the arc (not center) in nanometers.
        end: End position in nanometers.
    """

    start: PositionNM
    mid: PositionNM
    end: PositionNM


@dataclass(frozen=True, slots=True)
class RoundedOutline:
    """Rounded rectangle board outline.

    Contains a sequence of line and arc elements that together form a
    closed rounded rectangle on the Edge.Cuts layer.

    Attributes:
        elements: Tuple of OutlineLine and OutlineArc elements forming the outline.
        corner_radius_nm: Corner radius used in nanometers.
        width_nm: Total width of the rectangle in nanometers.
        height_nm: Total height of the rectangle in nanometers.
    """

    elements: tuple[OutlineLine | OutlineArc, ...]
    corner_radius_nm: int
    width_nm: int
    height_nm: int


class OutlineFeasibilityError(ValueError):
    """Error raised when outline parameters are not feasible."""

    pass


def validate_rounded_outline_feasibility(
    width_nm: int,
    height_nm: int,
    corner_radius_nm: int,
) -> None:
    """Validate that rounded outline parameters are feasible.

    Feasibility checks:
    1. Width and height must be positive.
    2. Corner radius must be non-negative.
    3. Corner radius must be at most half of the smaller dimension.

    Args:
        width_nm: Rectangle width in nanometers.
        height_nm: Rectangle height in nanometers.
        corner_radius_nm: Corner radius in nanometers.

    Raises:
        OutlineFeasibilityError: If parameters are not feasible.
    """
    if width_nm <= 0:
        raise OutlineFeasibilityError(f"Width must be positive, got {width_nm} nm")
    if height_nm <= 0:
        raise OutlineFeasibilityError(f"Height must be positive, got {height_nm} nm")
    if corner_radius_nm < 0:
        raise OutlineFeasibilityError(f"Corner radius must be non-negative, got {corner_radius_nm} nm")
    max_radius = min(width_nm, height_nm) // 2
    if corner_radius_nm > max_radius:
        raise OutlineFeasibilityError(
            f"Corner radius ({corner_radius_nm} nm) exceeds maximum ({max_radius} nm = min({width_nm}, {height_nm}) / 2)"
        )


def generate_rounded_outline(
    x_left_nm: int,
    y_bottom_nm: int,
    width_nm: int,
    height_nm: int,
    corner_radius_nm: int,
) -> RoundedOutline:
    """Generate a rounded rectangle outline for Edge.Cuts layer.

    Creates line and arc elements that form a closed rounded rectangle.
    The outline starts at the top-left corner (after the corner arc) and
    proceeds clockwise.

    The arc midpoints are computed to be exactly on the arc at 45 degrees
    from each endpoint, ensuring deterministic integer-nm coordinates.

    Args:
        x_left_nm: X coordinate of left edge in nanometers.
        y_bottom_nm: Y coordinate of bottom edge in nanometers.
        width_nm: Total width in nanometers.
        height_nm: Total height in nanometers.
        corner_radius_nm: Corner radius in nanometers.

    Returns:
        RoundedOutline with line and arc elements.

    Raises:
        OutlineFeasibilityError: If parameters are not feasible.
    """
    # Validate feasibility
    validate_rounded_outline_feasibility(width_nm, height_nm, corner_radius_nm)

    # If no corner radius, return simple rectangle with 4 lines
    if corner_radius_nm == 0:
        x_right_nm = x_left_nm + width_nm
        y_top_nm = y_bottom_nm + height_nm

        return RoundedOutline(
            elements=(
                OutlineLine(
                    start=PositionNM(x_left_nm, y_top_nm),
                    end=PositionNM(x_right_nm, y_top_nm),
                ),
                OutlineLine(
                    start=PositionNM(x_right_nm, y_top_nm),
                    end=PositionNM(x_right_nm, y_bottom_nm),
                ),
                OutlineLine(
                    start=PositionNM(x_right_nm, y_bottom_nm),
                    end=PositionNM(x_left_nm, y_bottom_nm),
                ),
                OutlineLine(
                    start=PositionNM(x_left_nm, y_bottom_nm),
                    end=PositionNM(x_left_nm, y_top_nm),
                ),
            ),
            corner_radius_nm=0,
            width_nm=width_nm,
            height_nm=height_nm,
        )

    # Compute corner positions
    r = corner_radius_nm
    x_right_nm = x_left_nm + width_nm
    y_top_nm = y_bottom_nm + height_nm

    # Corner centers
    tl_center = PositionNM(x_left_nm + r, y_top_nm - r)
    tr_center = PositionNM(x_right_nm - r, y_top_nm - r)
    br_center = PositionNM(x_right_nm - r, y_bottom_nm + r)
    bl_center = PositionNM(x_left_nm + r, y_bottom_nm + r)

    # Compute arc midpoint offset (45 degrees from center)
    # The midpoint should be on the arc at the midpoint angle
    # Using integer math: cos(45°) ≈ sin(45°) ≈ 0.7071
    # We use integer approximation: r * 707 // 1000
    mid_offset = r * 707 // 1000

    elements: list[OutlineLine | OutlineArc] = []

    # Top edge (from top-left corner end to top-right corner start)
    elements.append(
        OutlineLine(
            start=PositionNM(x_left_nm + r, y_top_nm),
            end=PositionNM(x_right_nm - r, y_top_nm),
        )
    )

    # Top-right corner arc (from top to right)
    elements.append(
        OutlineArc(
            start=PositionNM(x_right_nm - r, y_top_nm),
            mid=PositionNM(tr_center.x + mid_offset, tr_center.y + mid_offset),
            end=PositionNM(x_right_nm, y_top_nm - r),
        )
    )

    # Right edge
    elements.append(
        OutlineLine(
            start=PositionNM(x_right_nm, y_top_nm - r),
            end=PositionNM(x_right_nm, y_bottom_nm + r),
        )
    )

    # Bottom-right corner arc (from right to bottom)
    elements.append(
        OutlineArc(
            start=PositionNM(x_right_nm, y_bottom_nm + r),
            mid=PositionNM(br_center.x + mid_offset, br_center.y - mid_offset),
            end=PositionNM(x_right_nm - r, y_bottom_nm),
        )
    )

    # Bottom edge
    elements.append(
        OutlineLine(
            start=PositionNM(x_right_nm - r, y_bottom_nm),
            end=PositionNM(x_left_nm + r, y_bottom_nm),
        )
    )

    # Bottom-left corner arc (from bottom to left)
    elements.append(
        OutlineArc(
            start=PositionNM(x_left_nm + r, y_bottom_nm),
            mid=PositionNM(bl_center.x - mid_offset, bl_center.y - mid_offset),
            end=PositionNM(x_left_nm, y_bottom_nm + r),
        )
    )

    # Left edge
    elements.append(
        OutlineLine(
            start=PositionNM(x_left_nm, y_bottom_nm + r),
            end=PositionNM(x_left_nm, y_top_nm - r),
        )
    )

    # Top-left corner arc (from left to top)
    elements.append(
        OutlineArc(
            start=PositionNM(x_left_nm, y_top_nm - r),
            mid=PositionNM(tl_center.x - mid_offset, tl_center.y + mid_offset),
            end=PositionNM(x_left_nm + r, y_top_nm),
        )
    )

    return RoundedOutline(
        elements=tuple(elements),
        corner_radius_nm=corner_radius_nm,
        width_nm=width_nm,
        height_nm=height_nm,
    )


def generate_multivia_antipad(
    via_centers: tuple[PositionNM, ...],
    via_diameter_nm: int,
    clearance_nm: int,
    layer: str,
    convex_hull: bool = False,
) -> Polygon:
    """Generate a single antipad encompassing multiple vias.

    Creates either a union of circular cutouts or a convex hull around all
    vias with the specified clearance.

    Note: For simplicity, this currently only supports convex_hull=False
    which generates a bounding box with rounded corners.

    Args:
        via_centers: Tuple of via center positions in nm.
        via_diameter_nm: Via pad diameter in nm (assumed same for all).
        clearance_nm: Additional clearance beyond via edges in nm.
        layer: Layer for the cutout.
        convex_hull: If True, generate convex hull; otherwise bounding box.

    Returns:
        Polygon primitive for the cutout.
    """
    if not via_centers:
        raise ValueError("via_centers must not be empty")

    if len(via_centers) == 1:
        return generate_plane_cutout_for_via(
            via_centers[0],
            via_diameter_nm,
            clearance_nm,
            layer,
        )

    # Calculate bounding box of all via centers
    min_x = min(p.x for p in via_centers)
    max_x = max(p.x for p in via_centers)
    min_y = min(p.y for p in via_centers)
    max_y = max(p.y for p in via_centers)

    # Total offset from bounding box = via radius + clearance
    offset = via_diameter_nm // 2 + clearance_nm

    # Expand bounding box
    min_x -= offset
    max_x += offset
    min_y -= offset
    max_y += offset

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    width = max_x - min_x
    height = max_y - min_y

    # Use rounded rectangle with corner radius equal to offset
    spec = RoundRectAntipadSpec(
        shape=CutoutShape.ROUNDRECT,
        rx_nm=width // 2,
        ry_nm=height // 2,
        corner_nm=offset,
        layer=layer,
    )

    return generate_roundrect_antipad(PositionNM(center_x, center_y), spec)
