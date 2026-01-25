"""Geometry primitives and internal representation for coupon generation.

This module provides the internal representation (IR) for coupon geometry,
consisting of primitive types like footprints, pads, tracks, vias, and polygons.
All coordinates use integer nanometers (LengthNM) to ensure determinism and
avoid cross-platform rounding drift.

Submodules:
- primitives: Core geometry primitives (TrackSegment, Via, Polygon, etc.)
- layout: Layout plan dataclasses (LayoutPlan, SegmentPlan, PortPlan)
- footprint_meta: Footprint metadata loader (anchor, signal_pad, ground_pads, launch_reference)
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
    OutlineArc,
    OutlineFeasibilityError,
    OutlineLine,
    RectangleAntipadSpec,
    RoundedOutline,
    RoundRectAntipadSpec,
    SlotAntipadSpec,
    generate_antipad,
    generate_circle_antipad,
    generate_multivia_antipad,
    generate_plane_cutout_for_via,
    generate_rectangle_antipad,
    generate_rounded_outline,
    generate_roundrect_antipad,
    generate_slot_antipad,
    validate_rounded_outline_feasibility,
)
from .footprint_meta import (
    CourtyardMeta,
    FootprintMeta,
    LaunchRefMeta,
    PadMeta,
    PointMeta,
    get_footprint_meta_path,
    list_available_footprint_meta,
    load_footprint_meta,
)
from .layout import (
    LayoutPlan,
    PortPlan,
    SegmentPlan,
    compute_layout_plan,
    create_f0_layout_plan,
    create_f1_layout_plan,
    derive_right_length_nm,
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
    # Layout plan (single source of truth for geometry)
    "LayoutPlan",
    "PortPlan",
    "SegmentPlan",
    "compute_layout_plan",
    "create_f0_layout_plan",
    "create_f1_layout_plan",
    "derive_right_length_nm",
    # Footprint metadata
    "FootprintMeta",
    "PadMeta",
    "PointMeta",
    "LaunchRefMeta",
    "CourtyardMeta",
    "load_footprint_meta",
    "get_footprint_meta_path",
    "list_available_footprint_meta",
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
    # Rounded outline generators (REQ-M1-009)
    "OutlineArc",
    "OutlineFeasibilityError",
    "OutlineLine",
    "RoundedOutline",
    "generate_rounded_outline",
    "validate_rounded_outline_feasibility",
]
