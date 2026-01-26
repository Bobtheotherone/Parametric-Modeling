# SPDX-License-Identifier: MIT
"""Additional edge case tests for S-expression module.

Tests edge cases and additional functionality for the sexpr module that
are not covered by the main test_m1_sexpr.py tests.

Includes:
- Arc element building (kicad_gr_arc)
- Footprint element building
- Writer configuration options
- Error position reporting
- Special character handling
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

# Ensure src is in path for direct subpackage import
_src_path = Path(__file__).parent.parent / "src"
_kicad_path = _src_path / "formula_foundry" / "coupongen" / "kicad"
if str(_kicad_path) not in sys.path:
    sys.path.insert(0, str(_kicad_path))

# Now we can import sexpr directly as a top-level module
import sexpr


class TestKicadGrArc:
    """Tests for kicad_gr_arc element builder."""

    def test_gr_arc_basic(self) -> None:
        """Should build a gr_arc element with all required fields."""
        result = sexpr.kicad_gr_arc(
            start_x_nm=0,
            start_y_nm=0,
            mid_x_nm=1_000_000,  # midpoint on arc
            mid_y_nm=1_000_000,
            end_x_nm=2_000_000,
            end_y_nm=0,
            layer="Edge.Cuts",
            width=0.1,
            tstamp="arc-uuid-001",
        )

        assert result[0] == "gr_arc"

    def test_gr_arc_has_start(self) -> None:
        """Arc should have start point."""
        result = sexpr.kicad_gr_arc(
            start_x_nm=5_000_000,
            start_y_nm=3_000_000,
            mid_x_nm=6_000_000,
            mid_y_nm=4_000_000,
            end_x_nm=7_000_000,
            end_y_nm=3_000_000,
            layer="Edge.Cuts",
            width=0.1,
            tstamp="arc-uuid",
        )

        start_elem = None
        for elem in result:
            if isinstance(elem, list) and elem[0] == "start":
                start_elem = elem
                break

        assert start_elem is not None
        assert start_elem[1] == "5"  # 5mm
        assert start_elem[2] == "3"  # 3mm

    def test_gr_arc_has_mid(self) -> None:
        """Arc should have mid point (point on arc, not center)."""
        result = sexpr.kicad_gr_arc(
            start_x_nm=0,
            start_y_nm=0,
            mid_x_nm=1_500_000,
            mid_y_nm=500_000,
            end_x_nm=2_000_000,
            end_y_nm=0,
            layer="Edge.Cuts",
            width=0.1,
            tstamp="arc-uuid",
        )

        mid_elem = None
        for elem in result:
            if isinstance(elem, list) and elem[0] == "mid":
                mid_elem = elem
                break

        assert mid_elem is not None
        assert mid_elem[1] == "1.5"  # 1.5mm
        assert mid_elem[2] == "0.5"  # 0.5mm

    def test_gr_arc_has_end(self) -> None:
        """Arc should have end point."""
        result = sexpr.kicad_gr_arc(
            start_x_nm=0,
            start_y_nm=0,
            mid_x_nm=1_000_000,
            mid_y_nm=1_000_000,
            end_x_nm=3_000_000,
            end_y_nm=2_000_000,
            layer="Edge.Cuts",
            width=0.1,
            tstamp="arc-uuid",
        )

        end_elem = None
        for elem in result:
            if isinstance(elem, list) and elem[0] == "end":
                end_elem = elem
                break

        assert end_elem is not None
        assert end_elem[1] == "3"  # 3mm
        assert end_elem[2] == "2"  # 2mm

    def test_gr_arc_has_layer(self) -> None:
        """Arc should have layer specification."""
        result = sexpr.kicad_gr_arc(
            start_x_nm=0,
            start_y_nm=0,
            mid_x_nm=1_000_000,
            mid_y_nm=1_000_000,
            end_x_nm=2_000_000,
            end_y_nm=0,
            layer="F.SilkS",
            width=0.1,
            tstamp="arc-uuid",
        )

        assert ["layer", "F.SilkS"] in result

    def test_gr_arc_has_width(self) -> None:
        """Arc should have width specification."""
        result = sexpr.kicad_gr_arc(
            start_x_nm=0,
            start_y_nm=0,
            mid_x_nm=1_000_000,
            mid_y_nm=1_000_000,
            end_x_nm=2_000_000,
            end_y_nm=0,
            layer="Edge.Cuts",
            width=0.25,
            tstamp="arc-uuid",
        )

        assert ["width", 0.25] in result

    def test_gr_arc_has_tstamp(self) -> None:
        """Arc should have tstamp (UUID)."""
        result = sexpr.kicad_gr_arc(
            start_x_nm=0,
            start_y_nm=0,
            mid_x_nm=1_000_000,
            mid_y_nm=1_000_000,
            end_x_nm=2_000_000,
            end_y_nm=0,
            layer="Edge.Cuts",
            width=0.1,
            tstamp="my-custom-arc-uuid",
        )

        assert ["tstamp", "my-custom-arc-uuid"] in result


class TestKicadFootprint:
    """Tests for kicad_footprint element builder."""

    def test_footprint_basic(self) -> None:
        """Should build a footprint element."""
        result = sexpr.kicad_footprint(
            name="Connector:SMA",
            layer="F.Cu",
            at_x_nm=10_000_000,
            at_y_nm=5_000_000,
            rotation_deg=90,
            tstamp="fp-uuid-001",
        )

        assert result[0] == "footprint"
        assert result[1] == "Connector:SMA"

    def test_footprint_has_layer(self) -> None:
        """Footprint should have layer specification."""
        result = sexpr.kicad_footprint(
            name="Test:FP",
            layer="B.Cu",
            at_x_nm=0,
            at_y_nm=0,
            rotation_deg=0,
            tstamp="fp-uuid",
        )

        assert ["layer", "B.Cu"] in result

    def test_footprint_has_at_with_rotation(self) -> None:
        """Footprint should have at element with rotation."""
        result = sexpr.kicad_footprint(
            name="Test:FP",
            layer="F.Cu",
            at_x_nm=5_000_000,
            at_y_nm=10_000_000,
            rotation_deg=180,
            tstamp="fp-uuid",
        )

        at_elem = None
        for elem in result:
            if isinstance(elem, list) and elem[0] == "at":
                at_elem = elem
                break

        assert at_elem is not None
        assert at_elem[1] == "5"  # 5mm
        assert at_elem[2] == "10"  # 10mm
        assert at_elem[3] == 180  # rotation

    def test_footprint_with_children(self) -> None:
        """Footprint should include children elements."""
        child1 = ["pad", 1, "thru_hole", "circle"]
        child2 = ["pad", 2, "thru_hole", "circle"]

        result = sexpr.kicad_footprint(
            "Test:FP",  # name (positional)
            "F.Cu",  # layer (positional)
            0,  # at_x_nm (positional)
            0,  # at_y_nm (positional)
            0,  # rotation_deg (positional)
            "fp-uuid",  # tstamp (positional)
            child1,  # children
            child2,
        )

        assert child1 in result
        assert child2 in result


class TestKicadGrLine:
    """Tests for kicad_gr_line element builder."""

    def test_gr_line_basic(self) -> None:
        """Should build a gr_line element."""
        result = sexpr.kicad_gr_line(
            start_x_nm=0,
            start_y_nm=0,
            end_x_nm=10_000_000,
            end_y_nm=0,
            layer="Edge.Cuts",
            width=0.1,
            tstamp="line-uuid",
        )

        assert result[0] == "gr_line"

    def test_gr_line_coordinates(self) -> None:
        """Line should have correct coordinates converted to mm."""
        result = sexpr.kicad_gr_line(
            start_x_nm=1_500_000,
            start_y_nm=2_500_000,
            end_x_nm=3_500_000,
            end_y_nm=4_500_000,
            layer="Edge.Cuts",
            width=0.1,
            tstamp="line-uuid",
        )

        start_elem = None
        end_elem = None
        for elem in result:
            if isinstance(elem, list):
                if elem[0] == "start":
                    start_elem = elem
                elif elem[0] == "end":
                    end_elem = elem

        assert start_elem is not None
        assert end_elem is not None
        assert start_elem[1] == "1.5"
        assert start_elem[2] == "2.5"
        assert end_elem[1] == "3.5"
        assert end_elem[2] == "4.5"


class TestSExprWriter:
    """Tests for SExprWriter configuration."""

    def test_writer_default_indent(self) -> None:
        """Default indent should be 2 spaces."""
        writer = sexpr.SExprWriter()
        assert writer.indent == 2

    def test_writer_default_inline_threshold(self) -> None:
        """Default inline threshold should be 3."""
        writer = sexpr.SExprWriter()
        assert writer.inline_threshold == 3

    def test_writer_custom_indent(self) -> None:
        """Custom indent should be respected."""
        writer = sexpr.SExprWriter(indent=4)
        assert writer.indent == 4

    def test_writer_newline_after_first_includes_kicad_elements(self) -> None:
        """newline_after_first should include common KiCad elements."""
        writer = sexpr.SExprWriter()

        expected_elements = {"kicad_pcb", "footprint", "pad", "zone", "segment", "via"}
        for elem in expected_elements:
            assert elem in writer.newline_after_first

    def test_writer_always_inline_includes_coordinates(self) -> None:
        """always_inline should include coordinate elements."""
        writer = sexpr.SExprWriter()

        expected_inline = {"at", "start", "mid", "end", "size", "drill"}
        for elem in expected_inline:
            assert elem in writer.always_inline

    def test_writer_inline_elements_stay_inline(self) -> None:
        """Elements in always_inline should stay on one line."""
        writer = sexpr.SExprWriter()
        node = ["at", 10.5, 20.5, 90]
        result = writer.write(node)

        # Should be single line
        assert "\n" not in result
        assert result == "(at 10.5 20.5 90)"


class TestSExprParseErrorPosition:
    """Tests for error position reporting in SExprParseError."""

    def test_error_has_position(self) -> None:
        """Parse error should include position information."""
        with pytest.raises(sexpr.SExprParseError) as exc_info:
            sexpr.parse("(a b c")  # Unclosed list

        error = exc_info.value
        assert hasattr(error, "position")
        assert hasattr(error, "line")
        assert hasattr(error, "column")

    def test_error_str_includes_position(self) -> None:
        """Error string should include position info."""
        with pytest.raises(sexpr.SExprParseError) as exc_info:
            sexpr.parse(")")

        error_str = str(exc_info.value)
        assert "line" in error_str.lower()
        assert "column" in error_str.lower()


class TestSpecialCharacterHandling:
    """Tests for special character handling in strings."""

    def test_parse_newline_escape(self) -> None:
        """Should parse escaped newline correctly."""
        result = sexpr.parse(r'("line1\nline2")')
        assert result == ["line1\nline2"]

    def test_parse_tab_escape(self) -> None:
        """Should parse escaped tab correctly."""
        result = sexpr.parse(r'("col1\tcol2")')
        assert result == ["col1\tcol2"]

    def test_parse_quote_escape(self) -> None:
        """Should parse escaped quote correctly."""
        result = sexpr.parse(r'("say \"hello\"")')
        assert result == ['say "hello"']

    def test_parse_backslash_escape(self) -> None:
        """Should parse escaped backslash correctly."""
        result = sexpr.parse(r'("path\\to\\file")')
        assert result == [r"path\to\file"]

    def test_format_string_with_newline(self) -> None:
        """Should escape newline when formatting."""
        result = sexpr.format_string("line1\nline2")
        assert result == '"line1\\nline2"'

    def test_format_string_with_quote(self) -> None:
        """Should escape quote when formatting."""
        result = sexpr.format_string('say "hello"')
        assert result == '"say \\"hello\\""'

    def test_format_string_with_parentheses(self) -> None:
        """Should quote string with parentheses."""
        result = sexpr.format_string("(value)")
        assert result == '"(value)"'


class TestBuildList:
    """Tests for build_list helper function."""

    def test_build_list_empty(self) -> None:
        """Should return empty list for no arguments."""
        result = sexpr.build_list()
        assert result == []

    def test_build_list_single(self) -> None:
        """Should return list with single element."""
        result = sexpr.build_list("test")
        assert result == ["test"]

    def test_build_list_multiple(self) -> None:
        """Should return list with multiple elements."""
        result = sexpr.build_list("a", 1, "b")
        assert result == ["a", 1, "b"]

    def test_build_list_with_nested(self) -> None:
        """Should handle nested lists."""
        inner = ["inner", 1]
        result = sexpr.build_list("outer", inner, "end")
        assert result == ["outer", ["inner", 1], "end"]


class TestKicadHelperBuilders:
    """Tests for KiCad helper element builders."""

    def test_kicad_general(self) -> None:
        """Should build general section."""
        result = sexpr.kicad_general(thickness=1.6)
        assert result == ["general", ["thickness", 1.6]]

    def test_kicad_general_custom_thickness(self) -> None:
        """Should accept custom thickness."""
        result = sexpr.kicad_general(thickness=2.0)
        assert ["thickness", 2.0] in result

    def test_kicad_paper(self) -> None:
        """Should build paper element."""
        result = sexpr.kicad_paper(size="A3")
        assert result == ["paper", "A3"]

    def test_kicad_paper_default(self) -> None:
        """Default paper should be A4."""
        result = sexpr.kicad_paper()
        assert result == ["paper", "A4"]


class TestFormatAtomBooleans:
    """Tests for boolean formatting in format_atom."""

    def test_format_atom_true(self) -> None:
        """True should format as 'true'."""
        result = sexpr.format_atom(True)
        assert result == "true"

    def test_format_atom_false(self) -> None:
        """False should format as 'false'."""
        result = sexpr.format_atom(False)
        assert result == "false"


class TestDecimalFormatting:
    """Tests for decimal value formatting."""

    def test_format_decimal_integer(self) -> None:
        """Integer-valued decimal should format without decimal point."""
        result = sexpr.format_decimal(Decimal("100"))
        assert result == "100"

    def test_format_decimal_trailing_zeros(self) -> None:
        """Trailing zeros should be stripped."""
        result = sexpr.format_decimal(Decimal("1.50000"))
        assert result == "1.5"

    def test_format_decimal_small_value(self) -> None:
        """Small decimal values should be preserved."""
        result = sexpr.format_decimal(Decimal("0.001"))
        assert result == "0.001"

    def test_format_decimal_negative(self) -> None:
        """Negative decimals should format correctly."""
        result = sexpr.format_decimal(Decimal("-2.5"))
        assert result == "-2.5"

    def test_format_decimal_zero(self) -> None:
        """Zero should format as '0'."""
        result = sexpr.format_decimal(Decimal("0"))
        assert result == "0"


class TestNmToMmConversion:
    """Tests for nanometer to millimeter conversion."""

    def test_nm_to_mm_zero(self) -> None:
        """Zero nanometers should be '0'."""
        result = sexpr.nm_to_mm(0)
        assert result == "0"

    def test_nm_to_mm_exact_mm(self) -> None:
        """Exact millimeter values should format without decimals."""
        result = sexpr.nm_to_mm(5_000_000)
        assert result == "5"

    def test_nm_to_mm_sub_mm(self) -> None:
        """Sub-millimeter values should include decimals."""
        result = sexpr.nm_to_mm(750_000)
        assert result == "0.75"

    def test_nm_to_mm_negative(self) -> None:
        """Negative values should be handled."""
        result = sexpr.nm_to_mm(-1_000_000)
        assert result == "-1"

    def test_nm_to_mm_small_value(self) -> None:
        """Small nanometer values should preserve precision."""
        result = sexpr.nm_to_mm(1_000)  # 0.001mm = 1um
        assert result == "0.001"
