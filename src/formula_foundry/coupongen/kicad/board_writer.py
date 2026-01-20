"""KiCad board writer with deterministic S-expression generation.

This module implements headless .kicad_pcb file generation using the S-expression
format. It satisfies the IKiCadBackend interface and provides deterministic output
through UUIDv5-based tstamp generation.

Key features:
- Deterministic UUIDv5 generation for all tstamp/uuid fields
- S-expression output using the sexpr module
- Support for F0 (calibration) and F1 (via transition) coupon families
- Integer nanometer coordinate system with mm conversion for output

Satisfies REQ-M1-012 and REQ-M1-013.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from ..geom.primitives import (
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
    Text,
    TextJustify,
    TextLayer,
    TrackSegment,
    Via,
    create_coordinate_frame,
)
from ..resolve import ResolvedDesign
from ..spec import CouponSpec
from . import sexpr
from .sexpr import SExprList, nm_to_mm

if TYPE_CHECKING:
    from collections.abc import Iterable

# UUIDv5 namespace for coupongen deterministic IDs
_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "coupongen")

# KiCad version constants
KICAD_FILE_VERSION = 20240101
KICAD_GENERATOR = "coupongen"

# Default layer definitions for 4-layer board
DEFAULT_LAYER_DEFS: tuple[tuple[int, str, str], ...] = (
    (0, "F.Cu", "signal"),
    (1, "In1.Cu", "signal"),
    (2, "In2.Cu", "signal"),
    (31, "B.Cu", "signal"),
    (32, "B.Adhes", "user"),
    (33, "F.Adhes", "user"),
    (34, "B.Paste", "user"),
    (35, "F.Paste", "user"),
    (36, "B.SilkS", "user"),
    (37, "F.SilkS", "user"),
    (38, "B.Mask", "user"),
    (39, "F.Mask", "user"),
    (40, "Dwgs.User", "user"),
    (41, "Cmts.User", "user"),
    (42, "Eco1.User", "user"),
    (43, "Eco2.User", "user"),
    (44, "Edge.Cuts", "user"),
    (45, "Margin", "user"),
    (46, "B.CrtYd", "user"),
    (47, "F.CrtYd", "user"),
    (48, "B.Fab", "user"),
    (49, "F.Fab", "user"),
)


def deterministic_uuid(schema_version: int, path: str) -> str:
    """Generate a deterministic UUIDv5 for a given path.

    The UUID is generated using a two-level namespace:
    1. Base namespace derived from "coupongen" URL
    2. Version-specific namespace derived from schema version

    This ensures that:
    - Same spec version + same path = same UUID
    - Different spec versions produce different UUIDs

    Args:
        schema_version: CouponSpec schema version.
        path: Hierarchical path identifying the element (e.g., "board.outline").

    Returns:
        UUID string in standard format.
    """
    version_namespace = uuid.uuid5(_UUID_NAMESPACE, f"schema:{schema_version}")
    return str(uuid.uuid5(version_namespace, path))


def deterministic_uuid_indexed(schema_version: int, path: str, index: int) -> str:
    """Generate a deterministic UUIDv5 for an indexed element.

    Args:
        schema_version: CouponSpec schema version.
        path: Base path for the element type.
        index: Element index.

    Returns:
        UUID string in standard format.
    """
    return deterministic_uuid(schema_version, f"{path}[{index}]")


class BoardWriter:
    """KiCad board file writer with deterministic output.

    This class generates .kicad_pcb files from CouponSpec and ResolvedDesign
    using the S-expression format. All UUIDs/timestamps are deterministically
    generated from the spec content.
    """

    def __init__(self, spec: CouponSpec, resolved: ResolvedDesign) -> None:
        """Initialize the board writer.

        Args:
            spec: Coupon specification.
            resolved: Resolved design parameters.
        """
        self.spec = spec
        self.resolved = resolved
        self._uuid_counter = 0

        # Set up coordinate frame
        self.frame = create_coordinate_frame(
            origin_mode=OriginMode(spec.board.origin.mode),
            board_width_nm=int(spec.board.outline.length_nm),
            board_height_nm=int(spec.board.outline.width_nm),
        )

    def _next_uuid(self, path: str) -> str:
        """Generate the next deterministic UUID for a path."""
        return deterministic_uuid(self.spec.schema_version, path)

    def _indexed_uuid(self, base_path: str, index: int) -> str:
        """Generate a deterministic UUID for an indexed element."""
        return deterministic_uuid_indexed(self.spec.schema_version, base_path, index)

    def build_board(self) -> SExprList:
        """Build the complete board S-expression.

        Returns:
            S-expression list representing the full .kicad_pcb file.
        """
        elements: SExprList = [
            "kicad_pcb",
            ["version", KICAD_FILE_VERSION],
            ["generator", KICAD_GENERATOR],
        ]

        # Add header sections
        elements.append(self._build_general())
        elements.append(self._build_paper())
        elements.append(self._build_layers())

        # Add net declarations
        elements.extend(self._build_nets())

        # Add board outline
        elements.extend(self._build_outline())

        # Add footprints (connectors)
        elements.extend(self._build_footprints())

        # Add tracks (transmission lines)
        elements.extend(self._build_tracks())

        # Add vias if discontinuity present
        if self.spec.discontinuity is not None:
            elements.extend(self._build_vias())

        return elements

    def _build_general(self) -> SExprList:
        """Build the general section."""
        return ["general", ["thickness", 1.6]]

    def _build_paper(self) -> SExprList:
        """Build the paper size declaration."""
        return ["paper", "A4"]

    def _build_layers(self) -> SExprList:
        """Build the layers section based on stackup."""
        result: SExprList = ["layers"]
        for layer_id, name, layer_type in DEFAULT_LAYER_DEFS:
            result.append([layer_id, name, layer_type])
        return result

    def _build_nets(self) -> list[SExprList]:
        """Build net declarations."""
        return [
            ["net", 0, ""],
            ["net", 1, "SIG"],
            ["net", 2, "GND"],
        ]

    def _build_outline(self) -> list[SExprList]:
        """Build board outline as gr_rect on Edge.Cuts layer."""
        width_nm = int(self.spec.board.outline.width_nm)
        length_nm = int(self.spec.board.outline.length_nm)
        half_width = width_nm // 2

        outline_uuid = self._next_uuid("board.outline")

        return [
            [
                "gr_rect",
                ["start", nm_to_mm(0), nm_to_mm(-half_width)],
                ["end", nm_to_mm(length_nm), nm_to_mm(half_width)],
                ["layer", "Edge.Cuts"],
                ["width", 0.1],
                ["tstamp", outline_uuid],
            ]
        ]

    def _build_footprints(self) -> list[SExprList]:
        """Build footprint instances for connectors."""
        footprints: list[SExprList] = []

        for side in ("left", "right"):
            connector = getattr(self.spec.connectors, side)
            uuid_value = self._next_uuid(f"connector.{side}")
            x_nm = int(connector.position_nm[0])
            y_nm = int(connector.position_nm[1])

            fp: SExprList = [
                "footprint",
                connector.footprint,
                ["layer", "F.Cu"],
                ["at", nm_to_mm(x_nm), nm_to_mm(y_nm), connector.rotation_deg],
                ["tstamp", uuid_value],
            ]
            footprints.append(fp)

        return footprints

    def _build_tracks(self) -> list[SExprList]:
        """Build track segments for transmission lines."""
        tracks: list[SExprList] = []
        tl = self.spec.transmission_line

        # Left side track: from left connector to center (or discontinuity)
        left_conn = self.spec.connectors.left
        left_start_x = int(left_conn.position_nm[0])
        left_end_x = left_start_x + int(tl.length_left_nm)

        left_track_uuid = self._next_uuid("track.left")
        tracks.append(
            [
                "segment",
                ["start", nm_to_mm(left_start_x), nm_to_mm(0)],
                ["end", nm_to_mm(left_end_x), nm_to_mm(0)],
                ["width", nm_to_mm(int(tl.w_nm))],
                ["layer", tl.layer],
                ["net", 1],
                ["tstamp", left_track_uuid],
            ]
        )

        # Right side track: from center (or discontinuity) to right connector
        right_conn = self.spec.connectors.right
        right_end_x = int(right_conn.position_nm[0])
        right_start_x = right_end_x - int(tl.length_right_nm)

        right_track_uuid = self._next_uuid("track.right")
        tracks.append(
            [
                "segment",
                ["start", nm_to_mm(right_start_x), nm_to_mm(0)],
                ["end", nm_to_mm(right_end_x), nm_to_mm(0)],
                ["width", nm_to_mm(int(tl.w_nm))],
                ["layer", tl.layer],
                ["net", 1],
                ["tstamp", right_track_uuid],
            ]
        )

        return tracks

    def _build_vias(self) -> list[SExprList]:
        """Build via elements for discontinuity."""
        vias: list[SExprList] = []
        disc = self.spec.discontinuity
        if disc is None:
            return vias

        # Signal via at the center
        center_x = int(self.spec.board.outline.length_nm) // 2
        signal_via_uuid = self._next_uuid("via.signal")

        vias.append(
            [
                "via",
                ["at", nm_to_mm(center_x), nm_to_mm(0)],
                ["size", nm_to_mm(int(disc.signal_via.diameter_nm))],
                ["drill", nm_to_mm(int(disc.signal_via.drill_nm))],
                ["layers", "F.Cu", "B.Cu"],
                ["net", 1],
                ["tstamp", signal_via_uuid],
            ]
        )

        # Return vias if present
        if disc.return_vias is not None:
            import math

            rv = disc.return_vias
            radius = int(rv.radius_nm)
            count = rv.count

            for i in range(count):
                angle = 2 * math.pi * i / count
                vx = center_x + int(radius * math.cos(angle))
                vy = int(radius * math.sin(angle))
                via_uuid = self._indexed_uuid("via.return", i)

                vias.append(
                    [
                        "via",
                        ["at", nm_to_mm(vx), nm_to_mm(vy)],
                        ["size", nm_to_mm(int(rv.via.diameter_nm))],
                        ["drill", nm_to_mm(int(rv.via.drill_nm))],
                        ["layers", "F.Cu", "B.Cu"],
                        ["net", 2],
                        ["tstamp", via_uuid],
                    ]
                )

        return vias

    def write(self, out_path: Path) -> None:
        """Write the board file to disk.

        Args:
            out_path: Path for the output .kicad_pcb file.
        """
        board = self.build_board()
        content = sexpr.dump(board)
        out_path.write_text(content, encoding="utf-8")


def build_board_sexpr(spec: CouponSpec, resolved: ResolvedDesign) -> SExprList:
    """Build board S-expression from spec and resolved design.

    Args:
        spec: Coupon specification.
        resolved: Resolved design parameters.

    Returns:
        S-expression list representing the board.
    """
    writer = BoardWriter(spec, resolved)
    return writer.build_board()


def build_board_text(spec: CouponSpec, resolved: ResolvedDesign) -> str:
    """Build board file content from spec and resolved design.

    Args:
        spec: Coupon specification.
        resolved: Resolved design parameters.

    Returns:
        Formatted S-expression string for .kicad_pcb file.
    """
    board = build_board_sexpr(spec, resolved)
    return sexpr.dump(board)


def write_board(
    spec: CouponSpec, resolved: ResolvedDesign, out_dir: Path
) -> Path:
    """Write board file to output directory.

    Args:
        spec: Coupon specification.
        resolved: Resolved design parameters.
        out_dir: Output directory path.

    Returns:
        Path to the generated .kicad_pcb file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    board_path = out_dir / "coupon.kicad_pcb"
    writer = BoardWriter(spec, resolved)
    writer.write(board_path)
    return board_path
