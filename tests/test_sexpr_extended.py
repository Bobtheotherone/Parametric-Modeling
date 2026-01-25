# SPDX-License-Identifier: MIT
"""Extended unit tests for S-expression parsing and generation.

Complements test_m1_sexpr.py with additional coverage for:
- SExprWriter customization
- Additional KiCad element builders (gr_line, gr_arc, footprint)
- Edge cases in tokenization and parsing
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from formula_foundry.coupongen.kicad import sexpr


class TestSExprWriterConfiguration:
    """Tests for SExprWriter with custom configuration."""

    def test_writer_custom_indent(self) -> None:
        """Writer with custom indent value."""
        writer = sexpr.SExprWriter(indent=4)
        # Use an element that triggers multiline formatting
        tree = ["kicad_pcb", ["version", 20240101], ["generator", "test"]]

        result = writer.write(tree)

        # Should have 4-space indentation for child elements
        assert "    (version 20240101)" in result

    def test_writer_inline_threshold(self) -> None:
        """Writer respects inline threshold for short lists."""
        # High threshold - keeps more things inline
        writer_high = sexpr.SExprWriter(inline_threshold=10)
        # Low threshold - forces multiline earlier
        writer_low = sexpr.SExprWriter(inline_threshold=1)

        tree = ["a", "b", "c", "d"]

        result_high = writer_high.write(tree)
        result_low = writer_low.write(tree)

        # High threshold keeps it inline
        assert result_high == "(a b c d)"
        # Low threshold forces multiline
        assert "\n" in result_low

    def test_writer_always_inline_elements(self) -> None:
        """Elements in always_inline set stay on one line."""
        writer = sexpr.SExprWriter()

        # 'at' should be always inline
        at_elem = ["at", "1.5", "2.5", 90]
        result = writer.write(at_elem)

        assert result == "(at 1.5 2.5 90)"

    def test_writer_newline_after_first_elements(self) -> None:
        """Elements in newline_after_first set use multiline format."""
        writer = sexpr.SExprWriter()

        # 'kicad_pcb' should use multiline format
        tree = ["kicad_pcb", ["version", 20240101]]
        result = writer.write(tree)

        assert "\n" in result
        assert "(kicad_pcb" in result


class TestSExprTokenizerEdgeCases:
    """Tests for edge cases in tokenization."""

    def test_tokenize_escaped_quote_in_string(self) -> None:
        """Escaped quote inside quoted string."""
        result = sexpr.parse(r'("hello \"world\"")')
        assert result == ['hello "world"']

    def test_tokenize_escaped_backslash(self) -> None:
        """Escaped backslash in string."""
        result = sexpr.parse(r'("path\\to\\file")')
        assert result == ["path\\to\\file"]

    def test_tokenize_escaped_tab(self) -> None:
        """Escaped tab character."""
        result = sexpr.parse(r'("col1\tcol2")')
        assert result == ["col1\tcol2"]

    def test_tokenize_escaped_carriage_return(self) -> None:
        """Escaped carriage return."""
        result = sexpr.parse(r'("line1\rline2")')
        assert result == ["line1\rline2"]

    def test_tokenize_whitespace_handling(self) -> None:
        """Various whitespace characters are handled."""
        result = sexpr.parse("(\ta\n\tb\r\n\tc\t)")
        assert result == ["a", "b", "c"]

    def test_tokenize_deeply_nested(self) -> None:
        """Deeply nested expressions parse correctly."""
        result = sexpr.parse("(a (b (c (d (e 1)))))")
        assert result == ["a", ["b", ["c", ["d", ["e", 1]]]]]

    def test_parse_negative_float(self) -> None:
        """Negative floating point numbers."""
        result = sexpr.parse("(-1.5 -0.001)")
        assert result == [-1.5, -0.001]

    def test_parse_scientific_notation(self) -> None:
        """Scientific notation is parsed as float."""
        result = sexpr.parse("(1e-6 2.5e10)")
        assert result == [1e-6, 2.5e10]


class TestSExprParsingErrors:
    """Tests for parsing error handling."""

    def test_parse_error_has_position_info(self) -> None:
        """Parse errors include position information."""
        try:
            sexpr.parse("(a b")  # Unclosed
        except sexpr.SExprParseError as e:
            assert e.position >= 0
            assert e.line >= 1
            assert e.column >= 1

    def test_parse_error_message_format(self) -> None:
        """Parse error __str__ includes all info."""
        error = sexpr.SExprParseError("test error", position=10, line=2, column=5)
        msg = str(error)

        assert "test error" in msg
        assert "line 2" in msg
        assert "column 5" in msg
        assert "position 10" in msg

    def test_parse_trailing_content_raises(self) -> None:
        """Extra content after complete expression raises."""
        with pytest.raises(sexpr.SExprParseError, match="Unexpected token"):
            sexpr.parse("(a b) extra")

    def test_parse_empty_string_raises(self) -> None:
        """Empty string raises SExprParseError."""
        with pytest.raises(sexpr.SExprParseError, match="Empty input"):
            sexpr.parse("")

    def test_parse_whitespace_only_raises(self) -> None:
        """Whitespace-only input raises SExprParseError."""
        with pytest.raises(sexpr.SExprParseError, match="Empty input"):
            sexpr.parse("   \n\t  ")


class TestKicadGrLineBuilder:
    """Tests for kicad_gr_line builder."""

    def test_gr_line_basic(self) -> None:
        """Basic gr_line element generation."""
        result = sexpr.kicad_gr_line(
            start_x_nm=0,
            start_y_nm=0,
            end_x_nm=5_000_000,
            end_y_nm=0,
            layer="Edge.Cuts",
            width=0.15,
            tstamp="line-uuid",
        )

        assert result[0] == "gr_line"
        assert ["start", "0", "0"] in result
        assert ["end", "5", "0"] in result
        assert ["layer", "Edge.Cuts"] in result
        assert ["width", 0.15] in result
        assert ["tstamp", "line-uuid"] in result

    def test_gr_line_diagonal(self) -> None:
        """gr_line with diagonal coordinates."""
        result = sexpr.kicad_gr_line(
            start_x_nm=1_000_000,
            start_y_nm=1_000_000,
            end_x_nm=5_000_000,
            end_y_nm=3_000_000,
            layer="F.SilkS",
            width=0.12,
            tstamp="diag-uuid",
        )

        assert ["start", "1", "1"] in result
        assert ["end", "5", "3"] in result


class TestKicadGrArcBuilder:
    """Tests for kicad_gr_arc builder."""

    def test_gr_arc_basic(self) -> None:
        """Basic gr_arc element generation."""
        result = sexpr.kicad_gr_arc(
            start_x_nm=0,
            start_y_nm=0,
            mid_x_nm=1_000_000,
            mid_y_nm=1_000_000,
            end_x_nm=2_000_000,
            end_y_nm=0,
            layer="Edge.Cuts",
            width=0.1,
            tstamp="arc-uuid",
        )

        assert result[0] == "gr_arc"
        assert ["start", "0", "0"] in result
        assert ["mid", "1", "1"] in result
        assert ["end", "2", "0"] in result
        assert ["layer", "Edge.Cuts"] in result
        assert ["width", 0.1] in result
        assert ["tstamp", "arc-uuid"] in result

    def test_gr_arc_quarter_circle(self) -> None:
        """gr_arc for quarter circle (corner radius)."""
        # Quarter circle: start at (10, 0), mid at (7.07, 7.07), end at (0, 10)
        result = sexpr.kicad_gr_arc(
            start_x_nm=10_000_000,
            start_y_nm=0,
            mid_x_nm=7_071_000,
            mid_y_nm=7_071_000,
            end_x_nm=0,
            end_y_nm=10_000_000,
            layer="Edge.Cuts",
            width=0.15,
            tstamp="corner-uuid",
        )

        assert result[0] == "gr_arc"
        assert ["start", "10", "0"] in result
        assert ["mid", "7.071", "7.071"] in result
        assert ["end", "0", "10"] in result


class TestKicadFootprintBuilder:
    """Tests for kicad_footprint builder."""

    def test_footprint_basic(self) -> None:
        """Basic footprint element generation."""
        result = sexpr.kicad_footprint(
            name="Connector:Test",
            layer="F.Cu",
            at_x_nm=10_000_000,
            at_y_nm=5_000_000,
            rotation_deg=0,
            tstamp="fp-uuid",
        )

        assert result[0] == "footprint"
        assert result[1] == "Connector:Test"
        assert ["layer", "F.Cu"] in result
        assert ["at", "10", "5", 0] in result
        assert ["tstamp", "fp-uuid"] in result

    def test_footprint_with_rotation(self) -> None:
        """Footprint with rotation."""
        result = sexpr.kicad_footprint(
            name="Package:SMD",
            layer="B.Cu",
            at_x_nm=20_000_000,
            at_y_nm=10_000_000,
            rotation_deg=180,
            tstamp="rotated-uuid",
        )

        assert ["at", "20", "10", 180] in result
        assert ["layer", "B.Cu"] in result

    def test_footprint_with_children(self) -> None:
        """Footprint with child elements."""
        pad_element = ["pad", 1, "smd", "rect"]
        line_element = ["fp_line", ["start", 0, 0], ["end", 1, 1]]

        result = sexpr.kicad_footprint(
            "Custom:Part",
            "F.Cu",
            0,
            0,
            0,
            "parent-uuid",
            pad_element,
            line_element,
        )

        assert pad_element in result
        assert line_element in result


class TestKicadGeneralAndPaper:
    """Tests for kicad_general and kicad_paper builders."""

    def test_kicad_general_default(self) -> None:
        """kicad_general with default thickness."""
        result = sexpr.kicad_general()

        assert result[0] == "general"
        assert ["thickness", 1.6] in result

    def test_kicad_general_custom_thickness(self) -> None:
        """kicad_general with custom thickness."""
        result = sexpr.kicad_general(thickness=0.8)

        assert ["thickness", 0.8] in result

    def test_kicad_paper_default(self) -> None:
        """kicad_paper with default size."""
        result = sexpr.kicad_paper()

        assert result == ["paper", "A4"]

    def test_kicad_paper_custom_size(self) -> None:
        """kicad_paper with custom size."""
        result = sexpr.kicad_paper(size="A3")

        assert result == ["paper", "A3"]


class TestBuildListHelper:
    """Tests for build_list helper function."""

    def test_build_list_empty(self) -> None:
        """Build empty list."""
        result = sexpr.build_list()
        assert result == []

    def test_build_list_with_atoms(self) -> None:
        """Build list with atoms."""
        result = sexpr.build_list("a", 1, 2.5)
        assert result == ["a", 1, 2.5]

    def test_build_list_with_nested(self) -> None:
        """Build list with nested lists."""
        result = sexpr.build_list("outer", ["inner", 1])
        assert result == ["outer", ["inner", 1]]


class TestFormatAtomEdgeCases:
    """Tests for format_atom edge cases."""

    def test_format_atom_bool_true(self) -> None:
        """Boolean true formats as 'true'."""
        result = sexpr.format_atom(True)
        assert result == "true"

    def test_format_atom_bool_false(self) -> None:
        """Boolean false formats as 'false'."""
        result = sexpr.format_atom(False)
        assert result == "false"

    def test_format_atom_empty_string(self) -> None:
        """Empty string formats as quoted empty."""
        result = sexpr.format_atom("")
        assert result == '""'

    def test_format_atom_string_with_parens(self) -> None:
        """String with parentheses needs quoting."""
        result = sexpr.format_atom("func(x)")
        assert result == '"func(x)"'

    def test_format_atom_zero_decimal(self) -> None:
        """Zero Decimal formats as '0'."""
        result = sexpr.format_decimal(Decimal("0.000"))
        assert result == "0"


class TestNmToMmConversion:
    """Tests for nm_to_mm conversion."""

    def test_nm_to_mm_zero(self) -> None:
        """Zero nanometers."""
        assert sexpr.nm_to_mm(0) == "0"

    def test_nm_to_mm_negative(self) -> None:
        """Negative nanometers."""
        result = sexpr.nm_to_mm(-500_000)
        assert result == "-0.5"

    def test_nm_to_mm_large_value(self) -> None:
        """Large value (100mm)."""
        result = sexpr.nm_to_mm(100_000_000)
        assert result == "100"

    def test_nm_to_mm_fractional(self) -> None:
        """Fractional millimeters."""
        result = sexpr.nm_to_mm(127_000)  # 0.127mm (5 mil)
        assert result == "0.127"


class TestParseAllEmpty:
    """Tests for parse_all with various inputs."""

    def test_parse_all_empty_string(self) -> None:
        """parse_all with empty string returns empty list."""
        result = sexpr.parse_all("")
        assert result == []

    def test_parse_all_whitespace_only(self) -> None:
        """parse_all with whitespace only returns empty list."""
        result = sexpr.parse_all("   \n\t  ")
        assert result == []

    def test_parse_all_single_expression(self) -> None:
        """parse_all with single expression."""
        result = sexpr.parse_all("(a 1)")
        assert result == [["a", 1]]
