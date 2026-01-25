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
import math
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from ..builders.f1_builder import F1CouponComposition, build_f1_coupon
from ..constraints.core import resolve_fab_limits
from ..families import FAMILY_F1
from ..geom.layout import LayoutPlan
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
    pass

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
    raise ValueError(
        f"Unknown layer name: {spec_layer!r}. "
        f"Expected one of: {sorted(SPEC_TO_KICAD_LAYER.keys())}"
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
                "ResolvedDesign.layout_plan is None. The resolver must compute "
                "a LayoutPlan before passing to BoardWriter."
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
        min_via_dia = nm_to_mm(limits.get("min_via_diameter_nm", 200000))
        min_via_drill = nm_to_mm(limits.get("min_drill_nm", 200000))
        min_track = nm_to_mm(limits.get("min_trace_width_nm", 100000))
        min_clearance = nm_to_mm(limits.get("min_gap_nm", 100000))

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
        """Build board outline as gr_rect on Edge.Cuts layer.

        Uses LayoutPlan as the single source of truth for board dimensions
        (CP-2.6). The board dimensions and edge coordinates are read directly
        from the LayoutPlan.
        """
        # Get board dimensions from LayoutPlan (single source of truth)
        lp = self._layout_plan
        y_top = lp.y_board_top_edge_nm
        y_bottom = lp.y_board_bottom_edge_nm
        x_left = lp.x_board_left_edge_nm
        x_right = lp.x_board_right_edge_nm

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

    def _build_footprints(self) -> list[SExprList]:
        """Build footprint instances for connectors with inline pad definitions.

        Uses LayoutPlan port positions as the single source of truth for
        connector placement (CP-2.6). The footprint reference position and
        rotation are read from the LayoutPlan's left_port and right_port.

        Footprints include inline pad definitions so that:
        1. No external footprint library is required
        2. Tracks can properly connect to the signal pad
        3. DRC does not report dangling tracks or missing library errors

        The signal pad is positioned at the signal_pad_x_nm, signal_pad_y_nm
        coordinates from the LayoutPlan, converted to footprint-local coordinates.

        The pad layer is determined from the connecting segment's layer, which
        ensures proper connectivity for via transition coupons where left and
        right traces may be on different layers.
        """
        footprints: list[SExprList] = []
        lp = self._layout_plan

        # Get trace width from spec for pad sizing
        trace_width_nm = int(self.spec.transmission_line.w_nm)

        # Determine the layer for each side based on connecting segments
        # For F0: both segments on same layer
        # For F1: left segment on entry layer, right segment on exit layer
        segment_layers_by_side = {}
        for seg in lp.segments:
            if seg.label == "left":
                segment_layers_by_side["left"] = seg.layer
            elif seg.label == "right":
                segment_layers_by_side["right"] = seg.layer

        # Map side to port plan
        port_plans = {
            "left": lp.left_port,
            "right": lp.right_port,
        }

        for side, port in port_plans.items():
            uuid_value = self._next_uuid(f"connector.{side}")
            pad_uuid = self._next_uuid(f"connector.{side}.pad1")

            # Use port reference position from LayoutPlan
            x_nm = port.x_ref_nm
            y_nm = port.y_ref_nm
            # Convert rotation from millidegrees to degrees for KiCad
            rotation_deg = port.rotation_mdeg // 1000

            # Determine layer from connecting segment
            pad_layer = segment_layers_by_side.get(side, "F.Cu")

            # Calculate local pad position relative to footprint reference
            # For rotated footprints, we need to transform the signal pad position
            local_pad_x_nm = port.signal_pad_x_nm - x_nm
            local_pad_y_nm = port.signal_pad_y_nm - y_nm

            # If rotated 180 degrees, negate the local offset
            if rotation_deg == 180:
                local_pad_x_nm = -local_pad_x_nm
                local_pad_y_nm = -local_pad_y_nm

            # Pad size: use trace width for both dimensions (square pad)
            # This ensures the track connects cleanly
            pad_size_nm = trace_width_nm

            # Build signal pad element
            # Use short library-less footprint name to avoid library lookup issues
            fp_name = port.footprint.split(":")[-1] if ":" in port.footprint else port.footprint

            pad: SExprList = [
                "pad",
                "1",
                "smd",
                "rect",
                ["at", nm_to_mm(local_pad_x_nm), nm_to_mm(local_pad_y_nm)],
                ["size", nm_to_mm(pad_size_nm), nm_to_mm(pad_size_nm)],
                ["layers", pad_layer],
                ["net", 1, "SIG"],
                ["uuid", pad_uuid],
            ]

            fp: SExprList = [
                "footprint",
                fp_name,
                ["layer", pad_layer],
                ["at", nm_to_mm(x_nm), nm_to_mm(y_nm), rotation_deg],
                ["tstamp", uuid_value],
                pad,
            ]
            footprints.append(fp)

        return footprints

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

        # Iterate over all segments in the LayoutPlan
        for segment in lp.segments:
            track_uuid = self._next_uuid(f"track.{segment.label}")

            # Net ID: SIG net is 1
            net_id = 1 if segment.net_name == "SIG" else 0

            tracks.append(
                [
                    "segment",
                    ["start", nm_to_mm(segment.x_start_nm), nm_to_mm(segment.y_nm)],
                    ["end", nm_to_mm(segment.x_end_nm), nm_to_mm(segment.y_nm)],
                    ["width", nm_to_mm(segment.width_nm)],
                    ["layer", segment.layer],
                    ["net", net_id],
                    ["tstamp", track_uuid],
                ]
            )

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

        return vias

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
            return_via_positions = [
                (v.position.x, v.position.y)
                for v in self._f1_composition.return_vias
            ]
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
