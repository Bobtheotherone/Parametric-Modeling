"""Geometry primitives and internal representation for coupon generation.

This module provides the internal representation (IR) for coupon geometry,
consisting of primitive types like footprints, pads, tracks, vias, and polygons.
All coordinates use integer nanometers (LengthNM) to ensure determinism and
avoid cross-platform rounding drift.
"""
from __future__ import annotations

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

__all__ = [
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
]
