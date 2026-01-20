"""Geometry primitives and internal representation for coupon generation.

This module provides the internal representation (IR) for coupon geometry,
consisting of primitive types like footprints, pads, tracks, vias, and polygons.
All coordinates use integer nanometers (LengthNM) to ensure determinism and
avoid cross-platform rounding drift.

Submodules:
- primitives: Core geometry primitives (TrackSegment, Via, Polygon, etc.)
- cpwg: CPWG transmission line generators with ground via fencing
- via_patterns: Signal via and return via pattern generators
- cutouts: Antipad and plane cutout shape generators
"""
from __future__ import annotations

from .cpwg import (
    CPWGResult,
    CPWGSpec,
    GroundViaFenceSpec,
    generate_cpwg_horizontal,
    generate_cpwg_segment,
    generate_cpwg_with_fence,
    generate_ground_via_fence,
    generate_symmetric_cpwg_pair,
)
from .cutouts import (
    AntipadSpec,
    CircleAntipadSpec,
    CutoutShape,
    RectangleAntipadSpec,
    RoundRectAntipadSpec,
    SlotAntipadSpec,
    generate_antipad,
    generate_circle_antipad,
    generate_multivia_antipad,
    generate_plane_cutout_for_via,
    generate_rectangle_antipad,
    generate_roundrect_antipad,
    generate_slot_antipad,
)
from .primitives import (
    ArcTrack,
    BoardOutline,
    CoordinateFrame,
    FootprintInstance,
    Net,
    NetClass,
    OriginMode,
    Pad,
    PadShape,
    Polygon,
    PolygonType,
    PositionNM,
    RuleArea,
    RuleAreaType,
    Text,
    TextJustify,
    TextLayer,
    TrackSegment,
    Via,
)
from .via_patterns import (
    ReturnViaGridSpec,
    ReturnViaPattern,
    ReturnViaRingSpec,
    ReturnViaSpec,
    SignalViaSpec,
    ViaTransitionResult,
    calculate_minimum_return_via_radius,
    calculate_via_ring_circumference_clearance,
    generate_return_via_grid,
    generate_return_via_quadrant,
    generate_return_via_ring,
    generate_signal_via,
    generate_via_transition,
)

__all__ = [
    # Primitives
    "ArcTrack",
    "BoardOutline",
    "CoordinateFrame",
    "FootprintInstance",
    "Net",
    "NetClass",
    "OriginMode",
    "Pad",
    "PadShape",
    "Polygon",
    "PolygonType",
    "PositionNM",
    "RuleArea",
    "RuleAreaType",
    "Text",
    "TextJustify",
    "TextLayer",
    "TrackSegment",
    "Via",
    # CPWG generators
    "CPWGResult",
    "CPWGSpec",
    "GroundViaFenceSpec",
    "generate_cpwg_horizontal",
    "generate_cpwg_segment",
    "generate_cpwg_with_fence",
    "generate_ground_via_fence",
    "generate_symmetric_cpwg_pair",
    # Via pattern generators
    "ReturnViaGridSpec",
    "ReturnViaPattern",
    "ReturnViaRingSpec",
    "ReturnViaSpec",
    "SignalViaSpec",
    "ViaTransitionResult",
    "calculate_minimum_return_via_radius",
    "calculate_via_ring_circumference_clearance",
    "generate_return_via_grid",
    "generate_return_via_quadrant",
    "generate_return_via_ring",
    "generate_signal_via",
    "generate_via_transition",
    # Cutout generators
    "AntipadSpec",
    "CircleAntipadSpec",
    "CutoutShape",
    "RectangleAntipadSpec",
    "RoundRectAntipadSpec",
    "SlotAntipadSpec",
    "generate_antipad",
    "generate_circle_antipad",
    "generate_multivia_antipad",
    "generate_plane_cutout_for_via",
    "generate_rectangle_antipad",
    "generate_roundrect_antipad",
    "generate_slot_antipad",
]
