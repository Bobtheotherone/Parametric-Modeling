"""KiCad board writer with deterministic S-expression generation.

This module implements headless .kicad_pcb file generation using the S-expression
format. It satisfies the IKiCadBackend interface and provides deterministic output
through UUIDv5-based tstamp generation.

Key features:
- Deterministic UUIDv5 generation for all tstamp/uuid fields
- S-expression output using the sexpr module
- Support for F0 (calibration) and F1 (via transition) coupon families
- Integer nanometer coordinate system with mm conversion for output
- F1 antipads/cutouts and return vias with configurable patterns
- All geometry consumed from LayoutPlan (single source of truth per CP-2.6)

Satisfies REQ-M1-007, REQ-M1-010, REQ-M1-012, REQ-M1-013 and CP-2.6 (ECO-M1-ALIGN-0001).
"""

from __future__ import annotations

import base64
import copy
import math
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from ..builders.f1_builder import F1CouponComposition, build_f1_coupon
from ..constraints.core import resolve_fab_limits
from ..families import FAMILY_F1
from ..geom.cpwg import CPWGSpec, GroundViaFenceSpec, generate_cpwg_ground_tracks, generate_ground_via_fence
from ..geom.footprint_meta import load_footprint_meta
from ..geom.layout import LayoutPlan
from ..geom.primitives import PositionNM, Via
from ..resolve import ResolvedDesign
from ..spec import CouponSpec
from . import sexpr
from .annotations import build_annotations_from_spec
from .sexpr import SExprList, nm_to_mm


def _coupon_id_from_design_hash(design_hash: str) -> str:
    """Derive a human-readable coupon ID from a design hash.

    Local implementation to avoid circular imports with hashing module.

    Args:
        design_hash: SHA256 hex digest of the resolved design.

    Returns:
        12-character lowercase base32-encoded identifier.
    """
    digest = bytes.fromhex(design_hash)
    encoded = base64.b32encode(digest).decode("ascii").lower().rstrip("=")
    return encoded[:12]


if TYPE_CHECKING:
    from ..geom.footprint_meta import FootprintMeta

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

# Mapping from spec layer names to KiCad layer names (4-layer board)
# The spec uses logical names (L1, L2, L3, L4) while KiCad uses physical names
SPEC_TO_KICAD_LAYER: dict[str, str] = {
    "L1": "F.Cu",
    "L2": "In1.Cu",
    "L3": "In2.Cu",
    "L4": "B.Cu",
    # Pass through KiCad layer names unchanged
    "F.Cu": "F.Cu",
    "In1.Cu": "In1.Cu",
    "In2.Cu": "In2.Cu",
    "B.Cu": "B.Cu",
}

_LAYER_FLIP_MAP: dict[str, str] = {
    "F.Cu": "B.Cu",
    "B.Cu": "F.Cu",
    "F.Paste": "B.Paste",
    "B.Paste": "F.Paste",
    "F.Mask": "B.Mask",
    "B.Mask": "F.Mask",
    "F.SilkS": "B.SilkS",
    "B.SilkS": "F.SilkS",
    "F.CrtYd": "B.CrtYd",
    "B.CrtYd": "F.CrtYd",
    "F.Fab": "B.Fab",
    "B.Fab": "F.Fab",
    "F.Adhes": "B.Adhes",
    "B.Adhes": "F.Adhes",
}


def map_layer_to_kicad(spec_layer: str) -> str:
    """Map a spec layer name to a KiCad layer name.

    Args:
        spec_layer: Layer name from spec (e.g., "L2") or KiCad name (e.g., "In1.Cu").

    Returns:
        KiCad layer name (e.g., "In1.Cu").

    Raises:
        ValueError: If the layer name is not recognized.
    """
    if spec_layer in SPEC_TO_KICAD_LAYER:
        return SPEC_TO_KICAD_LAYER[spec_layer]
    raise ValueError(f"Unknown layer name: {spec_layer!r}. Expected one of: {sorted(SPEC_TO_KICAD_LAYER.keys())}")


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

    All geometry (track endpoints, via positions, footprint placements) is
    consumed from the LayoutPlan in ResolvedDesign, which is the single source
    of truth for geometry (CP-2.6). No duplicate geometry math is performed.

    For F1 coupons, this class uses the F1 builder composition to correctly
    generate antipads/cutouts with configurable return via patterns.
    """

    def __init__(
        self,
        spec: CouponSpec,
        resolved: ResolvedDesign,
        design_hash: str | None = None,
    ) -> None:
        """Initialize the board writer.

        Args:
            spec: Coupon specification.
            resolved: Resolved design parameters with computed LayoutPlan.
            design_hash: SHA256 hash of the resolved design for silkscreen annotations.
                        If None, annotations requiring hash will use a placeholder.

        Raises:
            ValueError: If resolved.layout_plan is None.
        """
        self.spec = spec
        self.resolved = resolved
        self._design_hash = design_hash
        self._uuid_counter = 0

        # Get LayoutPlan from ResolvedDesign - this is the single source of
        # truth for all geometry (CP-2.6)
        layout_plan = resolved.layout_plan
        if layout_plan is None:
            raise ValueError(
                "ResolvedDesign.layout_plan is None. The resolver must compute a LayoutPlan before passing to BoardWriter."
            )
        self._layout_plan: LayoutPlan = layout_plan

        # Build F1 composition if applicable (still needed for antipads/cutouts)
        self._f1_composition: F1CouponComposition | None = None
        if spec.coupon_family == FAMILY_F1:
            self._f1_composition = build_f1_coupon(spec, resolved)

    def _next_uuid(self, path: str) -> str:
        """Generate the next deterministic UUID for a path."""
        return deterministic_uuid(self.spec.schema_version, path)

    def _indexed_uuid(self, base_path: str, index: int) -> str:
        """Generate a deterministic UUID for an indexed element."""
        return deterministic_uuid_indexed(self.spec.schema_version, base_path, index)

    @staticmethod
    def _build_track_segment(
        start: PositionNM,
        end: PositionNM,
        width_nm: int,
        layer: str,
        net_id: int,
        track_uuid: str,
    ) -> SExprList:
        return [
            "segment",
            ["start", nm_to_mm(start.x), nm_to_mm(start.y)],
            ["end", nm_to_mm(end.x), nm_to_mm(end.y)],
            ["width", nm_to_mm(width_nm)],
            ["layer", layer],
            ["net", net_id],
            ["tstamp", track_uuid],
        ]

    @staticmethod
    def _build_via_element(via: Via, via_uuid: str, net_id: int | None = None) -> SExprList:
        resolved_net = net_id if net_id is not None else (via.net_id if via.net_id > 0 else 0)
        return [
            "via",
            ["at", nm_to_mm(via.position.x), nm_to_mm(via.position.y)],
            ["size", nm_to_mm(via.diameter_nm)],
            ["drill", nm_to_mm(via.drill_nm)],
            ["layers", via.layers[0], via.layers[1]],
            ["net", resolved_net],
            ["tstamp", via_uuid],
        ]

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
        elements.append(self._build_setup())

        # Add net declarations and net classes (design rules)
        elements.extend(self._build_nets())
        elements.extend(self._build_net_classes())

        # Add board outline
        elements.extend(self._build_outline())

        # Add footprints (connectors)
        elements.extend(self._build_footprints())

        # Add tracks (transmission lines)
        elements.extend(self._build_tracks())

        # Add vias if discontinuity present
        if self.spec.discontinuity is not None:
            elements.extend(self._build_vias())

        # Add antipads and cutouts for F1 coupons
        if self._f1_composition is not None:
            elements.extend(self._build_antipads())
            elements.extend(self._build_cutouts())

        # Add ground plane fills for return via connectivity (F1 coupons)
        if self.spec.discontinuity is not None and self.spec.discontinuity.return_vias is not None:
            elements.extend(self._build_ground_planes())

        # Add silkscreen annotations with coupon_id and hash (REQ-M1-010)
        elements.extend(self._build_silkscreen_annotations())

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

    def _build_setup(self) -> SExprList:
        """Build the setup section with design rules from fab profile.

        Configures KiCad design rules based on the fab profile limits:
        - min_via_diameter: minimum via pad diameter
        - min_via_drill: minimum via hole size
        - min_track_width: minimum trace width
        - min_clearance: minimum copper-to-copper clearance

        These are specified in the default netclass to set board-wide design rules.

        Raises:
            RuntimeError: If fab limits cannot be resolved. The oracle MUST fail
                loudly when constraints are unknown - silent defaults would hide
                real DRC violations.
        """
        # Get resolved fab limits using the standard constraint resolution
        # M1 Oracle requirement: NO silent fallback - if we can't resolve limits,
        # the board rules will be wrong and DRC results meaningless.
        try:
            limits = resolve_fab_limits(self.spec)
        except Exception as e:
            raise RuntimeError(
                f"Cannot resolve fab limits for spec (fab_profile={self.spec.fab_profile.id!r}). "
                f"Board rules require known fab constraints for DRC validity. "
                f"Original error: {e}"
            ) from e

        # Convert from nm to mm for KiCad
        nm_to_mm(limits.get("min_via_diameter_nm", 200000))
        nm_to_mm(limits.get("min_drill_nm", 200000))
        nm_to_mm(limits.get("min_trace_width_nm", 100000))
        nm_to_mm(limits.get("min_gap_nm", 100000))

        return [
            "setup",
            ["pad_to_mask_clearance", 0],
            ["allow_soldermask_bridges_in_footprints", "no"],
            [
                "pcbplotparams",
                ["layerselection", "0x00010fc_ffffffff"],
                ["plot_on_all_layers_selection", "0x0000000_00000000"],
                ["disableapertmacros", "false"],
                ["usegerberextensions", "false"],
                ["usegerberattributes", "true"],
                ["usegerberadvancedattributes", "true"],
                ["creategerberjobfile", "true"],
                ["svguseinch", "false"],
                ["svgprecision", 4],
                ["excludeedgelayer", "true"],
                ["plotframeref", "false"],
                ["viasonmask", "false"],
                ["mode", 1],
                ["useauxorigin", "false"],
                ["hpglpennumber", 1],
                ["hpglpenspeed", 20],
                ["hpglpendiameter", 15.000000],
                ["pdf_front_fp_property_popups", "true"],
                ["pdf_back_fp_property_popups", "true"],
                ["dxfpolygonmode", "true"],
                ["dxfimperialunits", "true"],
                ["dxfusepcbnewfont", "true"],
                ["psnegative", "false"],
                ["psa4output", "false"],
                ["plotreference", "true"],
                ["plotvalue", "true"],
                ["plotfptext", "true"],
                ["plotinvisibletext", "false"],
                ["sketchpadsonfab", "false"],
                ["subtractmaskfromsilk", "false"],
                ["outputformat", 1],
                ["mirror", "false"],
                ["drillshape", 1],
                ["scaleselection", 1],
                ["outputdirectory", ""],
            ],
        ]

    def _build_net_classes(self) -> list[SExprList]:
        """Build net class definitions with design rules from fab profile.

        The Default net class sets the board-wide minimum design rules.

        Raises:
            RuntimeError: If fab limits cannot be resolved. The oracle MUST fail
                loudly when constraints are unknown - silent defaults would hide
                real DRC violations.
        """
        # Get resolved fab limits
        # M1 Oracle requirement: NO silent fallback - if we can't resolve limits,
        # the board rules will be wrong and DRC results meaningless.
        try:
            limits = resolve_fab_limits(self.spec)
        except Exception as e:
            raise RuntimeError(
                f"Cannot resolve fab limits for spec (fab_profile={self.spec.fab_profile.id!r}). "
                f"Net class rules require known fab constraints for DRC validity. "
                f"Original error: {e}"
            ) from e

        min_clearance = nm_to_mm(limits.get("min_gap_nm", 100000))
        min_track = nm_to_mm(limits.get("min_trace_width_nm", 100000))
        min_via_dia = nm_to_mm(limits.get("min_via_diameter_nm", 200000))
        min_via_drill = nm_to_mm(limits.get("min_drill_nm", 200000))

        return [
            [
                "net_class",
                "Default",
                "",
                ["clearance", min_clearance],
                ["trace_width", min_track],
                ["via_dia", min_via_dia],
                ["via_drill", min_via_drill],
                ["uvia_dia", min_via_dia],
                ["uvia_drill", min_via_drill],
            ]
        ]

    def _build_nets(self) -> list[SExprList]:
        """Build net declarations."""
        return [
            ["net", 0, ""],
            ["net", 1, "SIG"],
            ["net", 2, "GND"],
        ]

    def _build_outline(self) -> list[SExprList]:
        """Build board outline on Edge.Cuts layer.

        Uses LayoutPlan as the single source of truth for board dimensions
        (CP-2.6). The board dimensions and edge coordinates are read directly
        from the LayoutPlan.

        When corner_radius_nm > 0, generates a rounded rectangle using
        gr_line and gr_arc elements. Otherwise generates a simple gr_rect.

        Satisfies REQ-M1-009: Rounded board outline with deterministic
        integer-nm arcs/segments and feasibility validation.
        """
        from ..geom.cutouts import (
            OutlineArc,
            OutlineLine,
            generate_rounded_outline,
        )

        # Get board dimensions from LayoutPlan (single source of truth)
        lp = self._layout_plan
        y_top = lp.y_board_top_edge_nm
        y_bottom = lp.y_board_bottom_edge_nm
        x_left = lp.x_board_left_edge_nm
        x_right = lp.x_board_right_edge_nm
        corner_radius = lp.board_corner_radius_nm

        # If no corner radius, use simple gr_rect
        if corner_radius == 0:
            outline_uuid = self._next_uuid("board.outline")
            return [
                [
                    "gr_rect",
                    ["start", nm_to_mm(x_left), nm_to_mm(y_bottom)],
                    ["end", nm_to_mm(x_right), nm_to_mm(y_top)],
                    ["layer", "Edge.Cuts"],
                    ["width", 0.1],
                    ["tstamp", outline_uuid],
                ]
            ]

        # Generate rounded outline with lines and arcs
        width_nm = x_right - x_left
        height_nm = y_top - y_bottom

        # generate_rounded_outline validates feasibility and raises
        # OutlineFeasibilityError if invalid
        rounded = generate_rounded_outline(
            x_left_nm=x_left,
            y_bottom_nm=y_bottom,
            width_nm=width_nm,
            height_nm=height_nm,
            corner_radius_nm=corner_radius,
        )

        # Convert to KiCad S-expressions
        elements: list[SExprList] = []
        for i, elem in enumerate(rounded.elements):
            elem_uuid = self._indexed_uuid("board.outline", i)

            if isinstance(elem, OutlineLine):
                elements.append(
                    [
                        "gr_line",
                        ["start", nm_to_mm(elem.start.x), nm_to_mm(elem.start.y)],
                        ["end", nm_to_mm(elem.end.x), nm_to_mm(elem.end.y)],
                        ["layer", "Edge.Cuts"],
                        ["width", 0.1],
                        ["tstamp", elem_uuid],
                    ]
                )
            elif isinstance(elem, OutlineArc):
                elements.append(
                    [
                        "gr_arc",
                        ["start", nm_to_mm(elem.start.x), nm_to_mm(elem.start.y)],
                        ["mid", nm_to_mm(elem.mid.x), nm_to_mm(elem.mid.y)],
                        ["end", nm_to_mm(elem.end.x), nm_to_mm(elem.end.y)],
                        ["layer", "Edge.Cuts"],
                        ["width", 0.1],
                        ["tstamp", elem_uuid],
                    ]
                )

        return elements

    def _build_footprints(self) -> list[SExprList]:
        """Build footprint instances for connectors using vendored .kicad_mod files."""
        footprints: list[SExprList] = []
        lp = self._layout_plan

        # Determine the layer for each side based on connecting segments
        segment_layers_by_side: dict[str, str] = {}
        for seg in lp.segments:
            if seg.label == "left":
                segment_layers_by_side["left"] = seg.layer
            elif seg.label == "right":
                segment_layers_by_side["right"] = seg.layer
            elif seg.label == "through":
                segment_layers_by_side["left"] = seg.layer
                segment_layers_by_side["right"] = seg.layer

        port_plans = {
            "left": lp.left_port,
            "right": lp.right_port,
        }

        footprint_cache: dict[str, SExprList] = {}

        for side, port in port_plans.items():
            meta = load_footprint_meta(port.footprint)
            footprint = self._load_footprint_template(meta, footprint_cache)
            self._set_footprint_name(footprint, meta.footprint_path)
            self._set_footprint_reference(footprint, f"J_{side.upper()}")

            rotation_deg = port.rotation_mdeg // 1000
            self._set_footprint_at(footprint, port.x_ref_nm, port.y_ref_nm, rotation_deg)

            target_layer = segment_layers_by_side.get(side, "F.Cu")
            if target_layer == "B.Cu":
                self._flip_footprint_layers(footprint)
                self._set_top_level_entry(footprint, ["layer", "B.Cu"], key="layer")
            else:
                self._set_top_level_entry(footprint, ["layer", "F.Cu"], key="layer")

            self._apply_pad_nets(footprint, meta)
            self._add_footprint_provenance(footprint, meta)
            self._remap_footprint_uuids(footprint, f"connector.{side}.footprint")
            self._set_top_level_entry(
                footprint,
                ["tstamp", self._next_uuid(f"connector.{side}")],
                key="tstamp",
            )
            footprints.append(footprint)

        return footprints

    def _load_footprint_template(
        self,
        meta: FootprintMeta,
        cache: dict[str, SExprList],
    ) -> SExprList:
        cache_key = str(meta.footprint_file)
        if cache_key not in cache:
            cache[cache_key] = self._load_footprint_module(meta.footprint_file)
        return copy.deepcopy(cache[cache_key])

    @staticmethod
    def _load_footprint_module(path: Path) -> SExprList:
        text = path.read_text(encoding="utf-8", errors="replace")
        parsed = sexpr.parse(text)
        if not isinstance(parsed, list) or not parsed or parsed[0] != "footprint":
            raise ValueError(f"Invalid footprint module in {path}")
        return parsed

    @staticmethod
    def _set_footprint_name(footprint: SExprList, footprint_path: str) -> None:
        if len(footprint) < 2:
            footprint.insert(1, footprint_path)
        else:
            footprint[1] = footprint_path

    @staticmethod
    def _set_top_level_entry(footprint: SExprList, entry: SExprList, *, key: str) -> None:
        for idx in range(2, len(footprint)):
            item = footprint[idx]
            if isinstance(item, list) and item and item[0] == key:
                footprint[idx] = entry
                return
        insert_idx = 2
        for idx in range(2, len(footprint)):
            item = footprint[idx]
            if isinstance(item, list) and item and item[0] == "layer":
                insert_idx = idx + 1
                break
        footprint.insert(insert_idx, entry)

    def _set_footprint_at(self, footprint: SExprList, x_nm: int, y_nm: int, rotation_deg: int) -> None:
        self._set_top_level_entry(
            footprint,
            ["at", nm_to_mm(x_nm), nm_to_mm(y_nm), rotation_deg],
            key="at",
        )

    @staticmethod
    def _set_footprint_reference(footprint: SExprList, reference: str) -> None:
        for item in footprint:
            if isinstance(item, list) and item and item[0] == "fp_text" and len(item) >= 3:
                if item[1] == "reference":
                    item[2] = reference
                    return

    @staticmethod
    def _set_pad_net(pad: SExprList, net_id: int, net_name: str) -> None:
        net_entry: SExprList = ["net", net_id, net_name]
        for idx, entry in enumerate(pad):
            if isinstance(entry, list) and entry and entry[0] == "net":
                pad[idx] = net_entry
                return
        pad.append(net_entry)

    def _apply_pad_nets(self, footprint: SExprList, meta: FootprintMeta) -> None:
        pad_map: dict[str, tuple[int, str]] = {str(meta.signal_pad.pad_number): (1, "SIG")}
        for ground in meta.ground_pads:
            pad_map[str(ground.pad_number)] = (2, "GND")
        for item in footprint:
            if not (isinstance(item, list) and item and item[0] == "pad"):
                continue
            if len(item) < 2:
                continue
            pad_number = str(item[1])
            if pad_number not in pad_map:
                continue
            net_id, net_name = pad_map[pad_number]
            self._set_pad_net(item, net_id, net_name)

    def _add_footprint_provenance(self, footprint: SExprList, meta: FootprintMeta) -> None:
        entries = (
            ("coupongen_footprint_hash", meta.footprint_hash),
            ("coupongen_meta_hash", meta.metadata_hash),
        )
        for key, value in entries:
            footprint.append(self._build_hidden_user_text(f"{key}={value}"))

    @staticmethod
    def _build_hidden_user_text(text: str) -> SExprList:
        return [
            "fp_text",
            "user",
            text,
            ["at", nm_to_mm(0), nm_to_mm(0), 0],
            ["layer", "F.Fab"],
            "hide",
            ["effects", ["font", ["size", 1, 1], ["thickness", 0.15]]],
            ["uuid", "00000000-0000-0000-0000-000000000000"],
        ]

    def _remap_footprint_uuids(self, footprint: SExprList, base_path: str) -> None:
        index = 0

        def walk(node: sexpr.SExprNode) -> None:
            nonlocal index
            if isinstance(node, list):
                if node and node[0] in ("uuid", "tstamp") and len(node) >= 2:
                    node[1] = self._indexed_uuid(base_path, index)
                    index += 1
                    return
                for child in node:
                    walk(child)

        for item in footprint[2:]:
            walk(item)

    def _flip_footprint_layers(self, footprint: SExprList) -> None:
        def walk(node: sexpr.SExprNode) -> None:
            if isinstance(node, list) and node:
                if node[0] == "layer" and len(node) >= 2 and isinstance(node[1], str):
                    node[1] = _LAYER_FLIP_MAP.get(node[1], node[1])
                elif node[0] == "layers":
                    for idx in range(1, len(node)):
                        if isinstance(node[idx], str):
                            node[idx] = _LAYER_FLIP_MAP.get(node[idx], node[idx])
                for child in node[1:]:
                    walk(child)

        walk(footprint)

    def _build_tracks(self) -> list[SExprList]:
        """Build track segments for transmission lines.

        Uses LayoutPlan segments as the single source of truth for track
        endpoints (CP-2.6). All segment positions, widths, and layers are
        read from the LayoutPlan's segment definitions.

        This eliminates duplicate geometry math - the LayoutPlan already
        has computed the correct segment endpoints ensuring topological
        continuity (e.g., left segment end == discontinuity == right segment start).
        """
        tracks: list[SExprList] = []
        lp = self._layout_plan
        tl_spec = self.spec.transmission_line
        use_cpwg = tl_spec.type.upper() == "CPWG"
        gap_nm = int(tl_spec.gap_nm)
        net_map = {"SIG": 1, "GND": 2}
        launch_plan_by_side = {
            "left": self._layout_plan.get_launch_plan("left"),
            "right": self._layout_plan.get_launch_plan("right"),
        }

        def append_launch_tracks(side: str) -> None:
            plan = launch_plan_by_side.get(side)
            if plan is None:
                return
            for idx, segment in enumerate(plan.segments):
                track_uuid = self._indexed_uuid(f"track.launch.{side}", idx)
                net_id = net_map.get(segment.net_name, 0)
                tracks.append(
                    self._build_track_segment(
                        segment.start,
                        segment.end,
                        segment.width_nm,
                        segment.layer,
                        net_id,
                        track_uuid,
                    )
                )

        append_launch_tracks("left")

        # Iterate over all segments in the LayoutPlan
        for segment in lp.segments:
            track_uuid = self._next_uuid(f"track.{segment.label}")
            net_id = net_map.get(segment.net_name, 0)

            tracks.append(
                self._build_track_segment(
                    PositionNM(segment.x_start_nm, segment.y_nm),
                    PositionNM(segment.x_end_nm, segment.y_nm),
                    segment.width_nm,
                    segment.layer,
                    net_id,
                    track_uuid,
                )
            )

            if use_cpwg and segment.net_name == "SIG":
                cpwg_spec = CPWGSpec(
                    w_nm=segment.width_nm,
                    gap_nm=gap_nm,
                    length_nm=segment.length_nm,
                    layer=segment.layer,
                    net_id=net_map["SIG"],
                    ground_net_id=net_map["GND"],
                )
                start = PositionNM(segment.x_start_nm, segment.y_nm)
                end = PositionNM(segment.x_end_nm, segment.y_nm)
                ground_tracks = generate_cpwg_ground_tracks(start, end, cpwg_spec)

                for idx, ground_track in enumerate(ground_tracks):
                    suffix = "gnd_pos" if idx == 0 else "gnd_neg"
                    ground_uuid = self._next_uuid(f"track.{segment.label}.{suffix}")
                    tracks.append(
                        self._build_track_segment(
                            ground_track.start,
                            ground_track.end,
                            ground_track.width_nm,
                            ground_track.layer,
                            ground_track.net_id,
                            ground_uuid,
                        )
                    )

        append_launch_tracks("right")

        return tracks

    def _build_vias(self) -> list[SExprList]:
        """Build via elements for discontinuity.

        Uses LayoutPlan's x_disc_nm as the single source of truth for the
        discontinuity center X position (CP-2.6). The Y position is taken
        from the LayoutPlan's y_centerline_nm.

        For F1 coupons, the F1 builder composition is still used for return
        via positions and antipad/cutout geometry, but the signal via position
        comes from the LayoutPlan.
        """
        vias: list[SExprList] = []
        lp = self._layout_plan
        disc = self.spec.discontinuity

        if disc is None or not lp.has_discontinuity:
            vias.extend(self._build_launch_stitch_vias())
            vias.extend(self._build_cpwg_via_fence())
            return vias

        # Get discontinuity position from LayoutPlan (single source of truth)
        center_x = lp.x_disc_nm
        assert center_x is not None  # Checked by has_discontinuity
        center_y = lp.y_centerline_nm

        signal_via_uuid = self._next_uuid("via.signal")

        vias.append(
            [
                "via",
                ["at", nm_to_mm(center_x), nm_to_mm(center_y)],
                ["size", nm_to_mm(int(disc.signal_via.diameter_nm))],
                ["drill", nm_to_mm(int(disc.signal_via.drill_nm))],
                ["layers", "F.Cu", "B.Cu"],
                ["net", 1],
                ["tstamp", signal_via_uuid],
            ]
        )

        # Return vias if present - use F1 composition for correct positioning
        # (the F1 builder uses the same discontinuity position internally)
        # Return vias are on GND net (net 2) and connect to ground plane fills
        # on F.Cu and B.Cu generated by _build_ground_planes().
        if disc.return_vias is not None:
            if self._f1_composition is not None and self._f1_composition.return_vias:
                # Use pre-computed return via positions from the F1 builder
                for i, return_via in enumerate(self._f1_composition.return_vias):
                    via_uuid = self._indexed_uuid("via.return", i)
                    vias.append(
                        [
                            "via",
                            ["at", nm_to_mm(return_via.position.x), nm_to_mm(return_via.position.y)],
                            ["size", nm_to_mm(return_via.diameter_nm)],
                            ["drill", nm_to_mm(return_via.drill_nm)],
                            ["layers", return_via.layers[0], return_via.layers[1]],
                            ["net", 2],  # GND net
                            ["tstamp", via_uuid],
                        ]
                    )
            else:
                # Fallback calculation using LayoutPlan discontinuity position
                rv = disc.return_vias
                radius = int(rv.radius_nm)
                count = rv.count

                for i in range(count):
                    angle = 2 * math.pi * i / count
                    vx = center_x + int(radius * math.cos(angle))
                    vy = center_y + int(radius * math.sin(angle))
                    via_uuid = self._indexed_uuid("via.return", i)

                    vias.append(
                        [
                            "via",
                            ["at", nm_to_mm(vx), nm_to_mm(vy)],
                            ["size", nm_to_mm(int(rv.via.diameter_nm))],
                            ["drill", nm_to_mm(int(rv.via.drill_nm))],
                            ["layers", "F.Cu", "B.Cu"],
                            ["net", 2],  # GND net
                            ["tstamp", via_uuid],
                        ]
                    )

        vias.extend(self._build_launch_stitch_vias())
        vias.extend(self._build_cpwg_via_fence())

        return vias

    def _build_launch_stitch_vias(self) -> list[SExprList]:
        vias: list[SExprList] = []
        for side in ("left", "right"):
            plan = self._layout_plan.get_launch_plan(side)
            if plan is None:
                continue
            for idx, via in enumerate(plan.stitch_vias):
                via_uuid = self._indexed_uuid(f"via.launch.{side}", idx)
                vias.append(self._build_via_element(via, via_uuid, net_id=2))
        return vias

    def _build_cpwg_via_fence(self) -> list[SExprList]:
        tl_spec = self.spec.transmission_line
        fence = tl_spec.ground_via_fence
        if fence is None or not fence.enabled:
            return []
        if tl_spec.type.upper() != "CPWG":
            return []

        fence_spec = GroundViaFenceSpec(
            pitch_nm=int(fence.pitch_nm),
            offset_from_gap_nm=int(fence.offset_from_gap_nm),
            drill_nm=int(fence.via.drill_nm),
            diameter_nm=int(fence.via.diameter_nm),
            layers=("F.Cu", "B.Cu"),
            net_id=2,
        )
        gap_nm = int(tl_spec.gap_nm)
        elements: list[SExprList] = []

        for segment in self._layout_plan.segments:
            cpwg_spec = CPWGSpec(
                w_nm=segment.width_nm,
                gap_nm=gap_nm,
                length_nm=segment.length_nm,
                layer=segment.layer,
                net_id=1,
                ground_net_id=2,
            )
            start = PositionNM(segment.x_start_nm, segment.y_nm)
            end = PositionNM(segment.x_end_nm, segment.y_nm)
            pos_vias, neg_vias = generate_ground_via_fence(start, end, cpwg_spec, fence_spec)

            for idx, via in enumerate(pos_vias):
                via_uuid = self._indexed_uuid(f"via.fence.{segment.label}.pos", idx)
                elements.append(self._build_via_element(via, via_uuid, net_id=2))
            for idx, via in enumerate(neg_vias):
                via_uuid = self._indexed_uuid(f"via.fence.{segment.label}.neg", idx)
                elements.append(self._build_via_element(via, via_uuid, net_id=2))

        return elements

    def _build_antipads(self) -> list[SExprList]:
        """Build antipad polygons for F1 coupons.

        Antipads are cutout regions on internal copper layers that provide
        clearance around the signal via. They are generated as zones with
        keepout restrictions for copper fill, but allowing vias to pass through.

        Note: The keepout allows vias because the signal via must pass through
        the antipad area. The antipad's purpose is to clear copper fill from
        ground planes, not to block the via structure itself.
        """
        antipads: list[SExprList] = []

        if self._f1_composition is None:
            return antipads

        all_antipads = self._f1_composition.all_antipads
        for i, antipad in enumerate(all_antipads):
            antipad_uuid = self._indexed_uuid("antipad", i)

            # Build the polygon points
            pts: SExprList = ["pts"]
            for vertex in antipad.vertices:
                pts.append(["xy", nm_to_mm(vertex.x), nm_to_mm(vertex.y)])

            # Map spec layer name to KiCad layer name
            kicad_layer = map_layer_to_kicad(antipad.layer)

            # Create a zone cutout on the appropriate layer
            # Note: vias are "allowed" because the signal via must pass through
            zone: SExprList = [
                "zone",
                ["net", 0],
                ["net_name", ""],
                ["layer", kicad_layer],
                ["tstamp", antipad_uuid],
                ["hatch", "edge", 0.5],
                ["priority", 0],
                ["connect_pads", ["clearance", 0]],
                ["min_thickness", 0.1],
                ["filled_areas_thickness", "no"],
                [
                    "keepout",
                    ["tracks", "allowed"],
                    ["vias", "allowed"],
                    ["pads", "allowed"],
                    ["copperpour", "not_allowed"],
                    ["footprints", "allowed"],
                ],
                ["fill", ["thermal_gap", 0.5], ["thermal_bridge_width", 0.5]],
                ["polygon", pts],
            ]
            antipads.append(zone)

        return antipads

    def _build_cutouts(self) -> list[SExprList]:
        """Build plane cutout polygons for F1 coupons.

        Plane cutouts are typically slot-shaped or rectangular regions
        that provide impedance tuning or thermal relief around the via
        transition. Like antipads, they block copper fill but allow vias.
        """
        cutouts: list[SExprList] = []

        if self._f1_composition is None:
            return cutouts

        all_cutouts = self._f1_composition.all_cutouts
        for i, cutout in enumerate(all_cutouts):
            cutout_uuid = self._indexed_uuid("cutout", i)

            # Build the polygon points
            pts: SExprList = ["pts"]
            for vertex in cutout.vertices:
                pts.append(["xy", nm_to_mm(vertex.x), nm_to_mm(vertex.y)])

            # Map spec layer name to KiCad layer name
            kicad_layer = map_layer_to_kicad(cutout.layer)

            # Create a zone cutout on the appropriate layer
            # Note: vias are "allowed" - cutouts clear copper fill, not block vias
            zone: SExprList = [
                "zone",
                ["net", 0],
                ["net_name", ""],
                ["layer", kicad_layer],
                ["tstamp", cutout_uuid],
                ["hatch", "edge", 0.5],
                ["priority", 0],
                ["connect_pads", ["clearance", 0]],
                ["min_thickness", 0.1],
                ["filled_areas_thickness", "no"],
                [
                    "keepout",
                    ["tracks", "allowed"],
                    ["vias", "allowed"],
                    ["pads", "allowed"],
                    ["copperpour", "not_allowed"],
                    ["footprints", "allowed"],
                ],
                ["fill", ["thermal_gap", 0.5], ["thermal_bridge_width", 0.5]],
                ["polygon", pts],
            ]
            cutouts.append(zone)

        return cutouts

    def _build_ground_planes(self) -> list[SExprList]:
        """Build ground ring traces for return via connectivity.

        Creates copper track segments connecting return vias in a ring on
        F.Cu and B.Cu, connected to GND net. This ring provides the copper
        connectivity that return vias need on each layer, satisfying KiCad
        DRC via connectivity requirements.

        The ring is drawn with trace width equal to the return via diameter
        to ensure solid overlap with the via pads.
        """
        elements: list[SExprList] = []
        disc = self.spec.discontinuity

        if disc is None or disc.return_vias is None:
            return elements

        lp = self._layout_plan
        center_x = lp.x_disc_nm
        center_y = lp.y_centerline_nm

        if center_x is None:
            return elements

        # Get return via positions and diameter
        if self._f1_composition is not None and self._f1_composition.return_vias:
            return_via_positions = [(v.position.x, v.position.y) for v in self._f1_composition.return_vias]
            via_diameter = self._f1_composition.return_vias[0].diameter_nm if self._f1_composition.return_vias else 500000
        else:
            # Fallback: calculate positions
            rv = disc.return_vias
            radius = int(rv.radius_nm)
            count = rv.count
            via_diameter = int(rv.via.diameter_nm)
            return_via_positions = []
            for i in range(count):
                angle = 2 * math.pi * i / count
                vx = center_x + int(radius * math.cos(angle))
                vy = center_y + int(radius * math.sin(angle))
                return_via_positions.append((vx, vy))

        if len(return_via_positions) < 2:
            return elements

        # Create ground ring on both F.Cu and B.Cu
        # Connect vias in sequence, then close the ring
        for layer in ["F.Cu", "B.Cu"]:
            for i in range(len(return_via_positions)):
                start_x, start_y = return_via_positions[i]
                end_x, end_y = return_via_positions[(i + 1) % len(return_via_positions)]

                track_uuid = self._indexed_uuid(f"gnd_ring.{layer}", i)

                track: SExprList = [
                    "segment",
                    ["start", nm_to_mm(start_x), nm_to_mm(start_y)],
                    ["end", nm_to_mm(end_x), nm_to_mm(end_y)],
                    ["width", nm_to_mm(via_diameter)],  # Wide trace for solid overlap
                    ["layer", layer],
                    ["net", 2],  # GND net
                    ["tstamp", track_uuid],
                ]
                elements.append(track)

        return elements

    def _build_silkscreen_annotations(self) -> list[SExprList]:
        """Build silkscreen text annotations with coupon_id and short hash.

        Satisfies REQ-M1-010: Deterministic silkscreen text includes coupon_id
        and short hash marker for provenance and visibility.

        Returns:
            List of gr_text S-expression elements for silkscreen layers.
        """
        # If no design_hash provided, use a placeholder for deterministic output
        design_hash = self._design_hash or ("0" * 64)
        coupon_id = _coupon_id_from_design_hash(design_hash)

        # Get board.text config from spec
        board_text = self.spec.board.text

        return build_annotations_from_spec(
            coupon_id_template=board_text.coupon_id,
            include_manifest_hash=board_text.include_manifest_hash,
            actual_coupon_id=coupon_id,
            design_hash=design_hash,
            layout_plan=self._layout_plan,
            uuid_generator=self._next_uuid,
        )

    def write(self, out_path: Path) -> None:
        """Write the board file to disk.

        Args:
            out_path: Path for the output .kicad_pcb file.
        """
        board = self.build_board()
        content = sexpr.dump(board)
        out_path.write_text(content, encoding="utf-8")


def build_board_sexpr(
    spec: CouponSpec,
    resolved: ResolvedDesign,
    design_hash: str | None = None,
) -> SExprList:
    """Build board S-expression from spec and resolved design.

    Args:
        spec: Coupon specification.
        resolved: Resolved design parameters.
        design_hash: Optional SHA256 hash for silkscreen annotations.

    Returns:
        S-expression list representing the board.
    """
    writer = BoardWriter(spec, resolved, design_hash)
    return writer.build_board()


def build_board_text(
    spec: CouponSpec,
    resolved: ResolvedDesign,
    design_hash: str | None = None,
) -> str:
    """Build board file content from spec and resolved design.

    Args:
        spec: Coupon specification.
        resolved: Resolved design parameters.
        design_hash: Optional SHA256 hash for silkscreen annotations.

    Returns:
        Formatted S-expression string for .kicad_pcb file.
    """
    board = build_board_sexpr(spec, resolved, design_hash)
    return sexpr.dump(board)


def write_board(
    spec: CouponSpec,
    resolved: ResolvedDesign,
    out_dir: Path,
    design_hash: str | None = None,
) -> Path:
    """Write board file to output directory.

    Args:
        spec: Coupon specification.
        resolved: Resolved design parameters.
        out_dir: Output directory path.
        design_hash: Optional SHA256 hash for silkscreen annotations.

    Returns:
        Path to the generated .kicad_pcb file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    board_path = out_dir / "coupon.kicad_pcb"
    writer = BoardWriter(spec, resolved, design_hash)
    writer.write(board_path)
    return board_path
