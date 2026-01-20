"""Via pattern geometry generators for signal and return vias.

This module provides functions to generate various via patterns commonly used
in PCB design:
- Signal vias (single via for layer transitions)
- Return via rings (circular patterns around signal vias)
- Return via grids (rectangular patterns for broader grounding)

All coordinates use integer nanometers (LengthNM) to ensure determinism and
avoid cross-platform rounding drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .primitives import PositionNM, Via

if TYPE_CHECKING:
    pass


class ReturnViaPattern(Enum):
    """Supported patterns for return via placement."""

    RING = "RING"
    """Circular ring pattern around a signal via."""

    GRID = "GRID"
    """Rectangular grid pattern around a signal via."""

    QUADRANT = "QUADRANT"
    """Vias placed in quadrant positions (4 vias at 45/135/225/315 degrees)."""


@dataclass(frozen=True, slots=True)
class SignalViaSpec:
    """Specification for a signal via.

    Attributes:
        drill_nm: Drill hole diameter in nanometers.
        diameter_nm: Via pad diameter in nanometers.
        pad_diameter_nm: Optional larger pad diameter for the landing pad.
            If None, uses diameter_nm.
        layers: Tuple of layer names the via connects.
        net_id: Net ID for the signal via.
    """

    drill_nm: int
    diameter_nm: int
    pad_diameter_nm: int | None = None
    layers: tuple[str, str] = ("F.Cu", "B.Cu")
    net_id: int = 1

    @property
    def effective_pad_diameter_nm(self) -> int:
        """Return the effective pad diameter (pad_diameter_nm or diameter_nm)."""
        return self.pad_diameter_nm if self.pad_diameter_nm is not None else self.diameter_nm


@dataclass(frozen=True, slots=True)
class ReturnViaSpec:
    """Specification for return vias.

    Attributes:
        drill_nm: Drill hole diameter in nanometers.
        diameter_nm: Via pad diameter in nanometers.
        layers: Tuple of layer names the vias connect.
        net_id: Net ID for the return vias (typically ground).
    """

    drill_nm: int
    diameter_nm: int
    layers: tuple[str, str] = ("F.Cu", "B.Cu")
    net_id: int = 0


@dataclass(frozen=True, slots=True)
class ReturnViaRingSpec:
    """Specification for a ring pattern of return vias.

    Attributes:
        pattern: Pattern type (should be RING).
        count: Number of vias in the ring.
        radius_nm: Distance from center to via centers in nanometers.
        via: Return via specification.
        start_angle_mdeg: Starting angle in millidegrees (0 = +x axis).
    """

    pattern: ReturnViaPattern
    count: int
    radius_nm: int
    via: ReturnViaSpec
    start_angle_mdeg: int = 0


@dataclass(frozen=True, slots=True)
class ReturnViaGridSpec:
    """Specification for a grid pattern of return vias.

    Attributes:
        pattern: Pattern type (should be GRID).
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        row_pitch_nm: Pitch between rows in nanometers.
        col_pitch_nm: Pitch between columns in nanometers.
        via: Return via specification.
        exclude_center: If True, exclude via at grid center position.
    """

    pattern: ReturnViaPattern
    rows: int
    cols: int
    row_pitch_nm: int
    col_pitch_nm: int
    via: ReturnViaSpec
    exclude_center: bool = True


@dataclass(frozen=True, slots=True)
class ViaTransitionResult:
    """Result of via transition geometry generation.

    Attributes:
        signal_via: The signal via.
        return_vias: Tuple of return vias.
    """

    signal_via: Via
    return_vias: tuple[Via, ...]


def generate_signal_via(
    center: PositionNM,
    spec: SignalViaSpec,
) -> Via:
    """Generate a signal via at the specified position.

    Args:
        center: Center position of the via in nm.
        spec: Signal via specification.

    Returns:
        Via primitive for the signal via.
    """
    return Via(
        position=center,
        diameter_nm=spec.diameter_nm,
        drill_nm=spec.drill_nm,
        layers=spec.layers,
        net_id=spec.net_id,
    )


def generate_return_via_ring(
    center: PositionNM,
    spec: ReturnViaRingSpec,
) -> tuple[Via, ...]:
    """Generate a ring pattern of return vias around a center point.

    The vias are placed evenly distributed on a circle of the specified radius.
    The starting angle can be adjusted to rotate the entire pattern.

    Args:
        center: Center position of the ring in nm.
        spec: Return via ring specification.

    Returns:
        Tuple of Via primitives.
    """
    if spec.count <= 0:
        return ()

    if spec.radius_nm <= 0:
        raise ValueError("radius_nm must be positive")

    vias: list[Via] = []
    angle_step_mdeg = 360_000 // spec.count  # Full circle in millidegrees

    for i in range(spec.count):
        angle_mdeg = spec.start_angle_mdeg + i * angle_step_mdeg
        angle_rad = math.radians(angle_mdeg / 1000.0)

        x = center.x + int(spec.radius_nm * math.cos(angle_rad))
        y = center.y + int(spec.radius_nm * math.sin(angle_rad))

        via = Via(
            position=PositionNM(x, y),
            diameter_nm=spec.via.diameter_nm,
            drill_nm=spec.via.drill_nm,
            layers=spec.via.layers,
            net_id=spec.via.net_id,
        )
        vias.append(via)

    return tuple(vias)


def generate_return_via_grid(
    center: PositionNM,
    spec: ReturnViaGridSpec,
) -> tuple[Via, ...]:
    """Generate a grid pattern of return vias around a center point.

    The grid is centered on the given position. If exclude_center is True,
    the via closest to the center position is excluded.

    Args:
        center: Center position of the grid in nm.
        spec: Return via grid specification.

    Returns:
        Tuple of Via primitives.
    """
    if spec.rows <= 0 or spec.cols <= 0:
        return ()

    if spec.row_pitch_nm <= 0 or spec.col_pitch_nm <= 0:
        raise ValueError("pitch values must be positive")

    vias: list[Via] = []

    # Calculate grid origin (top-left corner relative to center)
    # For odd counts, center aligns exactly; for even counts, center is between cells
    origin_x = center.x - (spec.cols - 1) * spec.col_pitch_nm // 2
    origin_y = center.y - (spec.rows - 1) * spec.row_pitch_nm // 2

    # Center grid position (for exclusion check)
    center_row = (spec.rows - 1) // 2
    center_col = (spec.cols - 1) // 2

    for row in range(spec.rows):
        for col in range(spec.cols):
            # Check if this is the center position
            if spec.exclude_center:
                # For odd grids, exclude exact center; for even grids, exclude closest to center
                is_center = row == center_row and col == center_col
                if is_center:
                    continue

            x = origin_x + col * spec.col_pitch_nm
            y = origin_y + row * spec.row_pitch_nm

            via = Via(
                position=PositionNM(x, y),
                diameter_nm=spec.via.diameter_nm,
                drill_nm=spec.via.drill_nm,
                layers=spec.via.layers,
                net_id=spec.via.net_id,
            )
            vias.append(via)

    return tuple(vias)


def generate_return_via_quadrant(
    center: PositionNM,
    radius_nm: int,
    via_spec: ReturnViaSpec,
) -> tuple[Via, ...]:
    """Generate vias at the four quadrant positions (45°, 135°, 225°, 315°).

    This creates a symmetric pattern with one via in each quadrant, commonly
    used for balanced return current distribution.

    Args:
        center: Center position in nm.
        radius_nm: Distance from center to via centers in nm.
        via_spec: Return via specification.

    Returns:
        Tuple of 4 Via primitives.
    """
    if radius_nm <= 0:
        raise ValueError("radius_nm must be positive")

    # Offset for 45-degree diagonal (radius / sqrt(2))
    offset = int(radius_nm / math.sqrt(2))

    positions = [
        PositionNM(center.x + offset, center.y + offset),  # Q1: 45°
        PositionNM(center.x - offset, center.y + offset),  # Q2: 135°
        PositionNM(center.x - offset, center.y - offset),  # Q3: 225°
        PositionNM(center.x + offset, center.y - offset),  # Q4: 315°
    ]

    vias = [
        Via(
            position=pos,
            diameter_nm=via_spec.diameter_nm,
            drill_nm=via_spec.drill_nm,
            layers=via_spec.layers,
            net_id=via_spec.net_id,
        )
        for pos in positions
    ]

    return tuple(vias)


def generate_via_transition(
    center: PositionNM,
    signal_spec: SignalViaSpec,
    return_ring_spec: ReturnViaRingSpec | None = None,
) -> ViaTransitionResult:
    """Generate a complete via transition with signal via and return vias.

    Creates the signal via at the center position and, if a return ring spec
    is provided, generates the surrounding return vias.

    Args:
        center: Center position of the via transition in nm.
        signal_spec: Signal via specification.
        return_ring_spec: Optional return via ring specification.

    Returns:
        ViaTransitionResult containing signal via and return vias.
    """
    signal_via = generate_signal_via(center, signal_spec)

    if return_ring_spec is None:
        return ViaTransitionResult(
            signal_via=signal_via,
            return_vias=(),
        )

    return_vias = generate_return_via_ring(center, return_ring_spec)

    return ViaTransitionResult(
        signal_via=signal_via,
        return_vias=return_vias,
    )


def calculate_minimum_return_via_radius(
    signal_via_spec: SignalViaSpec,
    return_via_spec: ReturnViaSpec,
    clearance_nm: int,
) -> int:
    """Calculate minimum radius for return vias to avoid clearance violations.

    Returns the minimum center-to-center distance between the signal via and
    return vias to maintain the specified clearance.

    Args:
        signal_via_spec: Signal via specification.
        return_via_spec: Return via specification.
        clearance_nm: Required clearance between via edges in nm.

    Returns:
        Minimum radius in nm for return via placement.
    """
    signal_radius = signal_via_spec.effective_pad_diameter_nm // 2
    return_radius = return_via_spec.diameter_nm // 2

    return signal_radius + return_radius + clearance_nm


def calculate_via_ring_circumference_clearance(
    ring_spec: ReturnViaRingSpec,
) -> int:
    """Calculate the edge-to-edge clearance between adjacent vias on a ring.

    This helps verify that the via count is appropriate for the ring radius.

    Args:
        ring_spec: Return via ring specification.

    Returns:
        Edge-to-edge clearance between adjacent vias in nm.
    """
    if ring_spec.count <= 1:
        return ring_spec.radius_nm * 2  # Only one via, "clearance" is the full diameter

    # Arc length between via centers
    circumference = 2 * math.pi * ring_spec.radius_nm
    arc_length = circumference / ring_spec.count

    # Clearance is arc length minus one via diameter
    clearance = int(arc_length) - ring_spec.via.diameter_nm

    return max(0, clearance)
