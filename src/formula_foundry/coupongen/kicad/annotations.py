"""Silkscreen annotation generation for KiCad boards.

This module provides deterministic silkscreen text annotations for coupon boards,
including coupon_id and short hash markers for provenance and traceability.

Satisfies REQ-M1-010: Improves provenance and visibility in outputs through
deterministic silkscreen text on F.SilkS (front) and B.SilkS (back) layers.

The annotations include:
- Coupon ID: derived from the design hash for unique identification
- Short hash: first 8 characters of the design hash for quick verification

Text elements are generated as gr_text S-expression elements positioned
on silkscreen layers with deterministic UUIDs for reproducible output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .sexpr import SExprList, nm_to_mm

if TYPE_CHECKING:
    from ..geom.layout import LayoutPlan


# Default text properties for silkscreen annotations
DEFAULT_TEXT_HEIGHT_NM = 1_000_000  # 1mm text height
DEFAULT_TEXT_WIDTH_NM = 1_000_000   # 1mm text width
DEFAULT_TEXT_THICKNESS_NM = 150_000  # 0.15mm stroke thickness
TEXT_Y_OFFSET_FROM_EDGE_NM = 1_500_000  # 1.5mm offset from board edge


def build_silkscreen_text(
    *,
    text: str,
    x_nm: int,
    y_nm: int,
    layer: str,
    uuid_value: str,
    height_nm: int = DEFAULT_TEXT_HEIGHT_NM,
    width_nm: int = DEFAULT_TEXT_WIDTH_NM,
    thickness_nm: int = DEFAULT_TEXT_THICKNESS_NM,
    justify: str | None = None,
) -> SExprList:
    """Build a gr_text S-expression element for silkscreen annotation.

    Args:
        text: The text string to display.
        x_nm: X position in nanometers.
        y_nm: Y position in nanometers.
        layer: Silkscreen layer ("F.SilkS" or "B.SilkS").
        uuid_value: Deterministic UUID for this text element.
        height_nm: Text height in nanometers.
        width_nm: Text width in nanometers.
        thickness_nm: Text stroke thickness in nanometers.
        justify: Optional text justification ("left", "center", "right").

    Returns:
        S-expression list representing the gr_text element.
    """
    effects: SExprList = [
        "effects",
        [
            "font",
            ["size", nm_to_mm(height_nm), nm_to_mm(width_nm)],
            ["thickness", nm_to_mm(thickness_nm)],
        ],
    ]

    if justify:
        effects.append(["justify", justify])

    result: SExprList = [
        "gr_text",
        text,
        ["at", nm_to_mm(x_nm), nm_to_mm(y_nm)],
        ["layer", layer],
        ["tstamp", uuid_value],
        effects,
    ]

    return result


def build_coupon_annotation(
    *,
    coupon_id: str,
    design_hash: str,
    layout_plan: "LayoutPlan",
    uuid_generator: callable,
) -> list[SExprList]:
    """Build silkscreen annotation elements with coupon_id and short hash.

    Creates two gr_text elements:
    1. Front silkscreen (F.SilkS): "coupon_id" at top of board
    2. Front silkscreen (F.SilkS): Short hash marker at bottom of board

    Args:
        coupon_id: Human-readable coupon identifier (12 chars from design hash).
        design_hash: Full SHA256 design hash (hex string).
        layout_plan: The layout plan with board dimensions.
        uuid_generator: Function to generate deterministic UUIDs.

    Returns:
        List of gr_text S-expression elements for silkscreen annotations.
    """
    annotations: list[SExprList] = []

    # Get board dimensions from layout plan
    board_width_nm = layout_plan.x_board_right_edge_nm - layout_plan.x_board_left_edge_nm
    x_center_nm = layout_plan.x_board_left_edge_nm + board_width_nm // 2

    # Positions for text elements
    y_top_text = layout_plan.y_board_top_edge_nm - TEXT_Y_OFFSET_FROM_EDGE_NM
    y_bottom_text = layout_plan.y_board_bottom_edge_nm + TEXT_Y_OFFSET_FROM_EDGE_NM

    # Short hash: first 8 characters of design hash
    short_hash = design_hash[:8] if len(design_hash) >= 8 else design_hash

    # Annotation text: "coupon_id:short_hash"
    annotation_text = f"{coupon_id}:{short_hash}"

    # Front silkscreen annotation - single line with both identifiers
    front_uuid = uuid_generator("silkscreen.front.annotation")
    annotations.append(
        build_silkscreen_text(
            text=annotation_text,
            x_nm=x_center_nm,
            y_nm=y_top_text,
            layer="F.SilkS",
            uuid_value=front_uuid,
            justify="center",
        )
    )

    # Back silkscreen annotation - mirror of front for identification from either side
    back_uuid = uuid_generator("silkscreen.back.annotation")
    annotations.append(
        build_silkscreen_text(
            text=annotation_text,
            x_nm=x_center_nm,
            y_nm=y_bottom_text,
            layer="B.SilkS",
            uuid_value=back_uuid,
            justify="center",
        )
    )

    return annotations


def build_annotations_from_spec(
    *,
    coupon_id_template: str,
    include_manifest_hash: bool,
    actual_coupon_id: str,
    design_hash: str,
    layout_plan: "LayoutPlan",
    uuid_generator: callable,
) -> list[SExprList]:
    """Build silkscreen annotations based on spec configuration.

    This is the primary entry point for annotation generation, using the
    board.text configuration from the CouponSpec.

    Args:
        coupon_id_template: Template from spec (e.g., "${COUPON_ID}" or static text).
        include_manifest_hash: Whether to include the design hash marker.
        actual_coupon_id: The computed coupon_id from the design hash.
        design_hash: Full SHA256 design hash (hex string).
        layout_plan: The layout plan with board dimensions.
        uuid_generator: Function to generate deterministic UUIDs.

    Returns:
        List of gr_text S-expression elements for silkscreen annotations.
    """
    # Resolve the coupon_id template
    resolved_id = coupon_id_template
    if "${COUPON_ID}" in coupon_id_template:
        resolved_id = coupon_id_template.replace("${COUPON_ID}", actual_coupon_id)

    if include_manifest_hash:
        # Full annotation with both coupon_id and hash
        return build_coupon_annotation(
            coupon_id=resolved_id,
            design_hash=design_hash,
            layout_plan=layout_plan,
            uuid_generator=uuid_generator,
        )
    else:
        # Just the coupon_id, no hash
        annotations: list[SExprList] = []

        board_width_nm = layout_plan.x_board_right_edge_nm - layout_plan.x_board_left_edge_nm
        x_center_nm = layout_plan.x_board_left_edge_nm + board_width_nm // 2
        y_top_text = layout_plan.y_board_top_edge_nm - TEXT_Y_OFFSET_FROM_EDGE_NM

        front_uuid = uuid_generator("silkscreen.front.id")
        annotations.append(
            build_silkscreen_text(
                text=resolved_id,
                x_nm=x_center_nm,
                y_nm=y_top_text,
                layer="F.SilkS",
                uuid_value=front_uuid,
                justify="center",
            )
        )

        return annotations
