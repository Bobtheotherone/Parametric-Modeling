# SPDX-License-Identifier: MIT
"""Unit tests for KiCad silkscreen annotation generation.

Tests the annotations module that generates deterministic silkscreen text
for coupon boards, including coupon_id and short hash markers for provenance.

Satisfies REQ-M1-010: Board annotations include coupon_id and hash marker
and appear in silkscreen exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class MockLayoutPlan:
    """Mock LayoutPlan for testing annotations positioning."""

    x_board_left_edge_nm: int
    x_board_right_edge_nm: int
    y_board_top_edge_nm: int
    y_board_bottom_edge_nm: int


def make_uuid_generator():
    """Create a deterministic UUID generator for testing."""
    counter = [0]

    def generate(key: str) -> str:
        counter[0] += 1
        return f"uuid-{key}-{counter[0]:04d}"

    return generate


class TestBuildSilkscreenText:
    """Tests for build_silkscreen_text function."""

    def test_returns_gr_text_sexpr(self) -> None:
        """Should return a gr_text S-expression list."""
        from formula_foundry.coupongen.kicad.annotations import build_silkscreen_text

        result = build_silkscreen_text(
            text="TEST",
            x_nm=1_000_000,
            y_nm=2_000_000,
            layer="F.SilkS",
            uuid_value="test-uuid-001",
        )

        assert result[0] == "gr_text"
        assert result[1] == "TEST"

    def test_position_converted_to_mm(self) -> None:
        """Position should be converted from nm to mm."""
        from formula_foundry.coupongen.kicad.annotations import build_silkscreen_text

        result = build_silkscreen_text(
            text="TEST",
            x_nm=10_000_000,  # 10mm
            y_nm=5_000_000,   # 5mm
            layer="F.SilkS",
            uuid_value="test-uuid-001",
        )

        # Find the "at" element
        at_element = None
        for elem in result:
            if isinstance(elem, list) and elem[0] == "at":
                at_element = elem
                break

        assert at_element is not None
        assert at_element[1] == "10"  # 10mm (string from nm_to_mm)
        assert at_element[2] == "5"   # 5mm (string from nm_to_mm)

    def test_layer_included(self) -> None:
        """Layer should be included in the S-expression."""
        from formula_foundry.coupongen.kicad.annotations import build_silkscreen_text

        result = build_silkscreen_text(
            text="TEST",
            x_nm=1_000_000,
            y_nm=2_000_000,
            layer="B.SilkS",
            uuid_value="test-uuid-001",
        )

        # Find the "layer" element
        layer_element = None
        for elem in result:
            if isinstance(elem, list) and elem[0] == "layer":
                layer_element = elem
                break

        assert layer_element is not None
        assert layer_element[1] == "B.SilkS"

    def test_uuid_included(self) -> None:
        """UUID (tstamp) should be included in the S-expression."""
        from formula_foundry.coupongen.kicad.annotations import build_silkscreen_text

        result = build_silkscreen_text(
            text="TEST",
            x_nm=1_000_000,
            y_nm=2_000_000,
            layer="F.SilkS",
            uuid_value="my-custom-uuid",
        )

        # Find the "tstamp" element
        tstamp_element = None
        for elem in result:
            if isinstance(elem, list) and elem[0] == "tstamp":
                tstamp_element = elem
                break

        assert tstamp_element is not None
        assert tstamp_element[1] == "my-custom-uuid"

    def test_effects_with_font(self) -> None:
        """Effects should include font settings."""
        from formula_foundry.coupongen.kicad.annotations import build_silkscreen_text

        result = build_silkscreen_text(
            text="TEST",
            x_nm=1_000_000,
            y_nm=2_000_000,
            layer="F.SilkS",
            uuid_value="test-uuid-001",
            height_nm=2_000_000,
            width_nm=2_000_000,
            thickness_nm=200_000,
        )

        # Find the "effects" element
        effects_element = None
        for elem in result:
            if isinstance(elem, list) and elem[0] == "effects":
                effects_element = elem
                break

        assert effects_element is not None
        # Find font in effects
        font_element = None
        for elem in effects_element:
            if isinstance(elem, list) and elem[0] == "font":
                font_element = elem
                break

        assert font_element is not None

    def test_justify_included_when_provided(self) -> None:
        """Justify should be included when specified."""
        from formula_foundry.coupongen.kicad.annotations import build_silkscreen_text

        result = build_silkscreen_text(
            text="TEST",
            x_nm=1_000_000,
            y_nm=2_000_000,
            layer="F.SilkS",
            uuid_value="test-uuid-001",
            justify="center",
        )

        # Find the "effects" element
        effects_element = None
        for elem in result:
            if isinstance(elem, list) and elem[0] == "effects":
                effects_element = elem
                break

        assert effects_element is not None
        # Check for justify in effects
        has_justify = any(
            isinstance(e, list) and e[0] == "justify" for e in effects_element
        )
        assert has_justify

    def test_justify_not_included_when_none(self) -> None:
        """Justify should not be included when not specified."""
        from formula_foundry.coupongen.kicad.annotations import build_silkscreen_text

        result = build_silkscreen_text(
            text="TEST",
            x_nm=1_000_000,
            y_nm=2_000_000,
            layer="F.SilkS",
            uuid_value="test-uuid-001",
            justify=None,
        )

        # Find the "effects" element
        effects_element = None
        for elem in result:
            if isinstance(elem, list) and elem[0] == "effects":
                effects_element = elem
                break

        assert effects_element is not None
        # Check that justify is NOT in effects
        has_justify = any(
            isinstance(e, list) and e[0] == "justify" for e in effects_element
        )
        assert not has_justify


class TestBuildCouponAnnotation:
    """Tests for build_coupon_annotation function."""

    def test_returns_two_annotations(self) -> None:
        """Should return front and back silkscreen annotations."""
        from formula_foundry.coupongen.kicad.annotations import build_coupon_annotation

        layout = MockLayoutPlan(
            x_board_left_edge_nm=-10_000_000,
            x_board_right_edge_nm=10_000_000,
            y_board_top_edge_nm=5_000_000,
            y_board_bottom_edge_nm=-5_000_000,
        )

        uuid_gen = make_uuid_generator()

        annotations = build_coupon_annotation(
            coupon_id="TEST123",
            design_hash="abc12345" + "0" * 56,
            layout_plan=layout,
            uuid_generator=uuid_gen,
        )

        assert len(annotations) == 2

    def test_front_annotation_on_f_silks(self) -> None:
        """Front annotation should be on F.SilkS layer."""
        from formula_foundry.coupongen.kicad.annotations import build_coupon_annotation

        layout = MockLayoutPlan(
            x_board_left_edge_nm=-10_000_000,
            x_board_right_edge_nm=10_000_000,
            y_board_top_edge_nm=5_000_000,
            y_board_bottom_edge_nm=-5_000_000,
        )

        uuid_gen = make_uuid_generator()

        annotations = build_coupon_annotation(
            coupon_id="TEST123",
            design_hash="abc12345" + "0" * 56,
            layout_plan=layout,
            uuid_generator=uuid_gen,
        )

        # First annotation should be on F.SilkS
        front_annotation = annotations[0]
        layer_element = None
        for elem in front_annotation:
            if isinstance(elem, list) and elem[0] == "layer":
                layer_element = elem
                break
        assert layer_element[1] == "F.SilkS"

    def test_back_annotation_on_b_silks(self) -> None:
        """Back annotation should be on B.SilkS layer."""
        from formula_foundry.coupongen.kicad.annotations import build_coupon_annotation

        layout = MockLayoutPlan(
            x_board_left_edge_nm=-10_000_000,
            x_board_right_edge_nm=10_000_000,
            y_board_top_edge_nm=5_000_000,
            y_board_bottom_edge_nm=-5_000_000,
        )

        uuid_gen = make_uuid_generator()

        annotations = build_coupon_annotation(
            coupon_id="TEST123",
            design_hash="abc12345" + "0" * 56,
            layout_plan=layout,
            uuid_generator=uuid_gen,
        )

        # Second annotation should be on B.SilkS
        back_annotation = annotations[1]
        layer_element = None
        for elem in back_annotation:
            if isinstance(elem, list) and elem[0] == "layer":
                layer_element = elem
                break
        assert layer_element[1] == "B.SilkS"

    def test_annotation_text_contains_coupon_id(self) -> None:
        """Annotation text should contain the coupon_id."""
        from formula_foundry.coupongen.kicad.annotations import build_coupon_annotation

        layout = MockLayoutPlan(
            x_board_left_edge_nm=-10_000_000,
            x_board_right_edge_nm=10_000_000,
            y_board_top_edge_nm=5_000_000,
            y_board_bottom_edge_nm=-5_000_000,
        )

        uuid_gen = make_uuid_generator()

        annotations = build_coupon_annotation(
            coupon_id="MYCOUPON",
            design_hash="abc12345" + "0" * 56,
            layout_plan=layout,
            uuid_generator=uuid_gen,
        )

        # Check front annotation text
        front_text = annotations[0][1]
        assert "MYCOUPON" in front_text

    def test_annotation_text_contains_short_hash(self) -> None:
        """Annotation text should contain the short hash (first 8 chars)."""
        from formula_foundry.coupongen.kicad.annotations import build_coupon_annotation

        layout = MockLayoutPlan(
            x_board_left_edge_nm=-10_000_000,
            x_board_right_edge_nm=10_000_000,
            y_board_top_edge_nm=5_000_000,
            y_board_bottom_edge_nm=-5_000_000,
        )

        uuid_gen = make_uuid_generator()
        design_hash = "abc12345def67890" + "0" * 48

        annotations = build_coupon_annotation(
            coupon_id="TEST",
            design_hash=design_hash,
            layout_plan=layout,
            uuid_generator=uuid_gen,
        )

        # Check that short hash (first 8 chars) is in the text
        front_text = annotations[0][1]
        assert "abc12345" in front_text

    def test_deterministic_uuid_generation(self) -> None:
        """UUIDs should be generated deterministically."""
        from formula_foundry.coupongen.kicad.annotations import build_coupon_annotation

        layout = MockLayoutPlan(
            x_board_left_edge_nm=-10_000_000,
            x_board_right_edge_nm=10_000_000,
            y_board_top_edge_nm=5_000_000,
            y_board_bottom_edge_nm=-5_000_000,
        )

        uuid_gen1 = make_uuid_generator()
        uuid_gen2 = make_uuid_generator()

        annotations1 = build_coupon_annotation(
            coupon_id="TEST",
            design_hash="abc12345" + "0" * 56,
            layout_plan=layout,
            uuid_generator=uuid_gen1,
        )

        annotations2 = build_coupon_annotation(
            coupon_id="TEST",
            design_hash="abc12345" + "0" * 56,
            layout_plan=layout,
            uuid_generator=uuid_gen2,
        )

        # Extract tstamp values
        def get_tstamp(annotation: list) -> str:
            for elem in annotation:
                if isinstance(elem, list) and elem[0] == "tstamp":
                    return elem[1]
            return ""

        assert get_tstamp(annotations1[0]) == get_tstamp(annotations2[0])
        assert get_tstamp(annotations1[1]) == get_tstamp(annotations2[1])


class TestBuildAnnotationsFromSpec:
    """Tests for build_annotations_from_spec function."""

    def test_template_resolution(self) -> None:
        """${COUPON_ID} template should be resolved."""
        from formula_foundry.coupongen.kicad.annotations import (
            build_annotations_from_spec,
        )

        layout = MockLayoutPlan(
            x_board_left_edge_nm=-10_000_000,
            x_board_right_edge_nm=10_000_000,
            y_board_top_edge_nm=5_000_000,
            y_board_bottom_edge_nm=-5_000_000,
        )

        uuid_gen = make_uuid_generator()

        annotations = build_annotations_from_spec(
            coupon_id_template="COUPON-${COUPON_ID}-END",
            include_manifest_hash=True,
            actual_coupon_id="RESOLVED123",
            design_hash="abc12345" + "0" * 56,
            layout_plan=layout,
            uuid_generator=uuid_gen,
        )

        # Check that template was resolved
        front_text = annotations[0][1]
        assert "RESOLVED123" in front_text
        assert "${COUPON_ID}" not in front_text

    def test_static_coupon_id(self) -> None:
        """Static coupon_id (no template) should be used as-is."""
        from formula_foundry.coupongen.kicad.annotations import (
            build_annotations_from_spec,
        )

        layout = MockLayoutPlan(
            x_board_left_edge_nm=-10_000_000,
            x_board_right_edge_nm=10_000_000,
            y_board_top_edge_nm=5_000_000,
            y_board_bottom_edge_nm=-5_000_000,
        )

        uuid_gen = make_uuid_generator()

        annotations = build_annotations_from_spec(
            coupon_id_template="STATIC_NAME",
            include_manifest_hash=True,
            actual_coupon_id="IGNORED",
            design_hash="abc12345" + "0" * 56,
            layout_plan=layout,
            uuid_generator=uuid_gen,
        )

        front_text = annotations[0][1]
        assert "STATIC_NAME" in front_text

    def test_include_manifest_hash_true(self) -> None:
        """When include_manifest_hash=True, should include full annotation."""
        from formula_foundry.coupongen.kicad.annotations import (
            build_annotations_from_spec,
        )

        layout = MockLayoutPlan(
            x_board_left_edge_nm=-10_000_000,
            x_board_right_edge_nm=10_000_000,
            y_board_top_edge_nm=5_000_000,
            y_board_bottom_edge_nm=-5_000_000,
        )

        uuid_gen = make_uuid_generator()

        annotations = build_annotations_from_spec(
            coupon_id_template="TEST",
            include_manifest_hash=True,
            actual_coupon_id="TEST",
            design_hash="abc12345" + "0" * 56,
            layout_plan=layout,
            uuid_generator=uuid_gen,
        )

        # Should have both front and back annotations
        assert len(annotations) == 2
        # Both should include short hash
        assert "abc12345" in annotations[0][1]

    def test_include_manifest_hash_false(self) -> None:
        """When include_manifest_hash=False, should not include hash."""
        from formula_foundry.coupongen.kicad.annotations import (
            build_annotations_from_spec,
        )

        layout = MockLayoutPlan(
            x_board_left_edge_nm=-10_000_000,
            x_board_right_edge_nm=10_000_000,
            y_board_top_edge_nm=5_000_000,
            y_board_bottom_edge_nm=-5_000_000,
        )

        uuid_gen = make_uuid_generator()

        annotations = build_annotations_from_spec(
            coupon_id_template="TEST",
            include_manifest_hash=False,
            actual_coupon_id="TEST",
            design_hash="abc12345" + "0" * 56,
            layout_plan=layout,
            uuid_generator=uuid_gen,
        )

        # Should have only one annotation (no back annotation when no hash)
        assert len(annotations) == 1
        # Should not include hash
        assert "abc12345" not in annotations[0][1]


class TestAnnotationConstants:
    """Tests for annotation module constants."""

    def test_default_text_height(self) -> None:
        """Default text height should be 1mm (1_000_000 nm)."""
        from formula_foundry.coupongen.kicad.annotations import DEFAULT_TEXT_HEIGHT_NM

        assert DEFAULT_TEXT_HEIGHT_NM == 1_000_000

    def test_default_text_width(self) -> None:
        """Default text width should be 1mm (1_000_000 nm)."""
        from formula_foundry.coupongen.kicad.annotations import DEFAULT_TEXT_WIDTH_NM

        assert DEFAULT_TEXT_WIDTH_NM == 1_000_000

    def test_default_text_thickness(self) -> None:
        """Default text thickness should be 0.15mm (150_000 nm)."""
        from formula_foundry.coupongen.kicad.annotations import DEFAULT_TEXT_THICKNESS_NM

        assert DEFAULT_TEXT_THICKNESS_NM == 150_000

    def test_text_y_offset_from_edge(self) -> None:
        """Text Y offset from edge should be 1.5mm (1_500_000 nm)."""
        from formula_foundry.coupongen.kicad.annotations import (
            TEXT_Y_OFFSET_FROM_EDGE_NM,
        )

        assert TEXT_Y_OFFSET_FROM_EDGE_NM == 1_500_000
