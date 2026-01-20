"""Tests for S-expression parsing and generation (REQ-M1-012, REQ-M1-013)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from formula_foundry.coupongen.kicad import sexpr


class TestSExprParsing:
    """Tests for S-expression parsing."""

    def test_parse_empty_list(self) -> None:
        result = sexpr.parse("()")
        assert result == []

    def test_parse_simple_list(self) -> None:
        result = sexpr.parse("(a b c)")
        assert result == ["a", "b", "c"]

    def test_parse_nested_list(self) -> None:
        result = sexpr.parse("(a (b c) d)")
        assert result == ["a", ["b", "c"], "d"]

    def test_parse_integers(self) -> None:
        result = sexpr.parse("(1 2 -3)")
        assert result == [1, 2, -3]

    def test_parse_floats(self) -> None:
        result = sexpr.parse("(1.5 -2.25 3.0)")
        assert result == [1.5, -2.25, 3.0]

    def test_parse_quoted_string(self) -> None:
        result = sexpr.parse('("hello world")')
        assert result == ["hello world"]

    def test_parse_quoted_string_with_escapes(self) -> None:
        result = sexpr.parse(r'("line1\nline2")')
        assert result == ["line1\nline2"]

    def test_parse_mixed_content(self) -> None:
        result = sexpr.parse('(name "value" 123)')
        assert result == ["name", "value", 123]

    def test_parse_kicad_style(self) -> None:
        # parse() expects a single expression; use parse_all() for multiple
        result = sexpr.parse_all('(version 20240101) (generator "coupongen")')
        assert len(result) == 2
        assert result[0] == ["version", 20240101]
        assert result[1] == ["generator", "coupongen"]

    def test_parse_all_multiple(self) -> None:
        result = sexpr.parse_all("(a 1) (b 2) (c 3)")
        assert result == [["a", 1], ["b", 2], ["c", 3]]

    def test_parse_empty_input_raises(self) -> None:
        with pytest.raises(sexpr.SExprParseError):
            sexpr.parse("")

    def test_parse_unclosed_list_raises(self) -> None:
        with pytest.raises(sexpr.SExprParseError):
            sexpr.parse("(a b c")

    def test_parse_unexpected_close_raises(self) -> None:
        with pytest.raises(sexpr.SExprParseError):
            sexpr.parse(")")

    def test_parse_unterminated_string_raises(self) -> None:
        with pytest.raises(sexpr.SExprParseError):
            sexpr.parse('("unclosed')


class TestSExprFormatting:
    """Tests for S-expression formatting."""

    def test_format_atom_string(self) -> None:
        assert sexpr.format_atom("hello") == "hello"

    def test_format_atom_string_needs_quotes(self) -> None:
        assert sexpr.format_atom("hello world") == '"hello world"'

    def test_format_atom_int(self) -> None:
        assert sexpr.format_atom(42) == "42"

    def test_format_atom_float(self) -> None:
        # Trailing zeros are stripped
        assert sexpr.format_atom(1.5) == "1.5"

    def test_format_atom_decimal(self) -> None:
        assert sexpr.format_atom(Decimal("3.14159")) == "3.14159"

    def test_format_decimal_strips_trailing_zeros(self) -> None:
        assert sexpr.format_decimal(Decimal("1.500")) == "1.5"
        assert sexpr.format_decimal(Decimal("2.000")) == "2"

    def test_format_string_empty(self) -> None:
        assert sexpr.format_string("") == '""'

    def test_format_string_simple(self) -> None:
        assert sexpr.format_string("hello") == "hello"

    def test_format_string_with_spaces(self) -> None:
        assert sexpr.format_string("hello world") == '"hello world"'

    def test_nm_to_mm(self) -> None:
        assert sexpr.nm_to_mm(1_000_000) == "1"
        assert sexpr.nm_to_mm(500_000) == "0.5"
        assert sexpr.nm_to_mm(250_000) == "0.25"

    def test_mm_point(self) -> None:
        result = sexpr.mm_point(1_000_000, 2_000_000)
        assert result == "1 2"


class TestSExprDump:
    """Tests for S-expression dumping."""

    def test_dump_empty_list(self) -> None:
        result = sexpr.dump([])
        assert result == "()"

    def test_dump_simple_list(self) -> None:
        result = sexpr.dump(["a", "b", "c"])
        assert result == "(a b c)"

    def test_dump_compact(self) -> None:
        result = sexpr.dump_compact(["a", ["b", "c"], "d"])
        assert result == "(a (b c) d)"

    def test_dump_nested_multiline(self) -> None:
        tree = [
            "kicad_pcb",
            ["version", 20240101],
            ["generator", "coupongen"],
        ]
        result = sexpr.dump(tree)
        assert "(kicad_pcb" in result
        assert "(version 20240101)" in result

    def test_dump_roundtrip(self) -> None:
        original = ["test", ["nested", 1, 2], "end"]
        dumped = sexpr.dump_compact(original)
        parsed = sexpr.parse(dumped)
        assert parsed == original


class TestKicadElementBuilders:
    """Tests for KiCad-specific S-expression builders."""

    def test_kicad_version(self) -> None:
        result = sexpr.kicad_version()
        assert result[0] == "kicad_pcb"
        assert ["version", 20240101] in result
        assert ["generator", "coupongen"] in result

    def test_kicad_net(self) -> None:
        result = sexpr.kicad_net(1, "SIG")
        assert result == ["net", 1, "SIG"]

    def test_kicad_layers(self) -> None:
        result = sexpr.kicad_layers(
            (0, "F.Cu", "signal"),
            (31, "B.Cu", "signal"),
        )
        assert result[0] == "layers"
        assert [0, "F.Cu", "signal"] in result
        assert [31, "B.Cu", "signal"] in result

    def test_kicad_gr_rect(self) -> None:
        result = sexpr.kicad_gr_rect(
            start_x_nm=0,
            start_y_nm=0,
            end_x_nm=10_000_000,
            end_y_nm=5_000_000,
            layer="Edge.Cuts",
            width=0.1,
            tstamp="test-uuid",
        )
        assert result[0] == "gr_rect"
        assert ["start", "0", "0"] in result
        assert ["end", "10", "5"] in result
        assert ["layer", "Edge.Cuts"] in result
        assert ["tstamp", "test-uuid"] in result

    def test_kicad_segment(self) -> None:
        result = sexpr.kicad_segment(
            start_x_nm=0,
            start_y_nm=0,
            end_x_nm=5_000_000,
            end_y_nm=0,
            width_nm=300_000,
            layer="F.Cu",
            net_id=1,
            tstamp="track-uuid",
        )
        assert result[0] == "segment"
        assert ["width", "0.3"] in result
        assert ["net", 1] in result

    def test_kicad_via(self) -> None:
        result = sexpr.kicad_via(
            x_nm=5_000_000,
            y_nm=0,
            diameter_nm=650_000,
            drill_nm=300_000,
            layers=("F.Cu", "B.Cu"),
            net_id=1,
            tstamp="via-uuid",
        )
        assert result[0] == "via"
        assert ["size", "0.65"] in result
        assert ["drill", "0.3"] in result
        assert ["layers", "F.Cu", "B.Cu"] in result
