"""S-expression parsing and generation for KiCad board files.

This module provides utilities for parsing and generating S-expression (sexpr)
format used by KiCad for .kicad_pcb and other files. It supports:

- Parsing S-expression strings into nested Python structures
- Generating S-expression strings from Python structures
- Pretty-printing with configurable indentation
- Proper handling of quoted strings and special characters

The S-expression format used by KiCad is a subset of Lisp-like syntax:
- Atoms: unquoted tokens (identifiers, numbers)
- Quoted strings: "text with spaces or special chars"
- Lists: (element1 element2 ...)

Satisfies REQ-M1-012 and REQ-M1-013.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# S-expression atom types
SExprAtom = Union[str, int, float, Decimal]
SExprNode = Union[SExprAtom, "SExprList"]
SExprList = list["SExprNode"]

# Pattern for tokens requiring quoting
_NEEDS_QUOTE_RE = re.compile(r'[\s"()\\]')

# Escape sequences for quoted strings
_ESCAPE_MAP = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}

_UNESCAPE_MAP = {v: k for k, v in _ESCAPE_MAP.items()}


@dataclass
class SExprParseError(Exception):
    """Error during S-expression parsing."""

    message: str
    position: int = 0
    line: int = 1
    column: int = 1

    def __str__(self) -> str:
        return f"{self.message} at line {self.line}, column {self.column} (position {self.position})"


@dataclass
class SExprToken:
    """Token from S-expression tokenizer."""

    type: str  # 'LPAREN', 'RPAREN', 'STRING', 'ATOM'
    value: str
    position: int
    line: int
    column: int


@dataclass
class SExprWriter:
    """Configurable S-expression writer with pretty-printing.

    Attributes:
        indent: Number of spaces per indentation level.
        inline_threshold: Maximum list length to keep on a single line.
        newline_after_first: Element names that should have newline after first child.
    """

    indent: int = 2
    inline_threshold: int = 3
    newline_after_first: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "kicad_pcb",
                "footprint",
                "module",
                "pad",
                "zone",
                "segment",
                "via",
                "gr_line",
                "gr_rect",
                "gr_circle",
                "gr_arc",
                "gr_poly",
                "fp_line",
                "fp_rect",
                "fp_circle",
                "fp_arc",
                "fp_poly",
                "net",
                "layer",
                "layers",
                "general",
                "setup",
            }
        )
    )

    def write(self, node: SExprNode) -> str:
        """Write S-expression node to string.

        Args:
            node: S-expression node (atom or list).

        Returns:
            Formatted S-expression string.
        """
        lines: list[str] = []
        self._write_node(node, 0, lines)
        return "\n".join(lines)

    def _write_node(self, node: SExprNode, depth: int, lines: list[str]) -> None:
        """Write a node with proper formatting."""
        if isinstance(node, list):
            self._write_list(node, depth, lines)
        else:
            lines.append(" " * (depth * self.indent) + format_atom(node))

    def _write_list(self, items: SExprList, depth: int, lines: list[str]) -> None:
        """Write a list with proper formatting."""
        if not items:
            lines.append(" " * (depth * self.indent) + "()")
            return

        prefix = " " * (depth * self.indent)
        first = items[0]
        first_str = format_atom(first) if not isinstance(first, list) else ""

        # Decide if we should use multiline format
        use_multiline = (
            len(items) > self.inline_threshold
            or first_str in self.newline_after_first
            or any(isinstance(item, list) and len(item) > 2 for item in items[1:])
        )

        if not use_multiline:
            # Single line format
            inline = "(" + " ".join(self._format_inline(item) for item in items) + ")"
            lines.append(prefix + inline)
        else:
            # Multiline format
            lines.append(prefix + "(" + first_str)
            for item in items[1:]:
                if isinstance(item, list):
                    self._write_node(item, depth + 1, lines)
                else:
                    lines.append(" " * ((depth + 1) * self.indent) + format_atom(item))
            lines[-1] += ")"

    def _format_inline(self, node: SExprNode) -> str:
        """Format a node for inline (single-line) output."""
        if isinstance(node, list):
            return "(" + " ".join(self._format_inline(item) for item in node) + ")"
        return format_atom(node)


def format_atom(value: SExprAtom) -> str:
    """Format an atom value for S-expression output.

    Args:
        value: Atom value (string, int, float, or Decimal).

    Returns:
        Formatted string representation.
    """
    if isinstance(value, str):
        return format_string(value)
    if isinstance(value, Decimal):
        return format_decimal(value)
    if isinstance(value, float):
        # Use Decimal for consistent formatting
        return format_decimal(Decimal(str(value)))
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return str(value)


def format_string(value: str) -> str:
    """Format a string value, quoting if necessary.

    Args:
        value: String value.

    Returns:
        Quoted string if needed, otherwise raw string.
    """
    if not value:
        return '""'
    if _NEEDS_QUOTE_RE.search(value):
        escaped = "".join(_ESCAPE_MAP.get(c, c) for c in value)
        return f'"{escaped}"'
    return value


def format_decimal(value: Decimal) -> str:
    """Format a Decimal value, removing trailing zeros.

    Args:
        value: Decimal value.

    Returns:
        Formatted string without unnecessary trailing zeros.
    """
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def nm_to_mm(value_nm: int) -> str:
    """Convert nanometers to millimeters as string.

    Args:
        value_nm: Value in nanometers.

    Returns:
        Value in millimeters as formatted string.
    """
    mm = Decimal(value_nm) / Decimal(1_000_000)
    return format_decimal(mm)


def mm_point(x_nm: int, y_nm: int) -> str:
    """Format a point in mm for S-expression output.

    Args:
        x_nm: X coordinate in nanometers.
        y_nm: Y coordinate in nanometers.

    Returns:
        Space-separated "x y" string in millimeters.
    """
    return f"{nm_to_mm(x_nm)} {nm_to_mm(y_nm)}"


class SExprTokenizer:
    """Tokenizer for S-expression parsing."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1

    def tokenize(self) -> Iterator[SExprToken]:
        """Generate tokens from input text."""
        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.pos >= len(self.text):
                break

            char = self.text[self.pos]
            start_pos = self.pos
            start_line = self.line
            start_column = self.column

            if char == "(":
                self._advance()
                yield SExprToken("LPAREN", "(", start_pos, start_line, start_column)
            elif char == ")":
                self._advance()
                yield SExprToken("RPAREN", ")", start_pos, start_line, start_column)
            elif char == '"':
                value = self._read_quoted_string()
                yield SExprToken("STRING", value, start_pos, start_line, start_column)
            else:
                value = self._read_atom()
                yield SExprToken("ATOM", value, start_pos, start_line, start_column)

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.pos < len(self.text) and self.text[self.pos] in " \t\n\r":
            if self.text[self.pos] == "\n":
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.pos += 1

    def _advance(self) -> str:
        """Advance position and return current character."""
        char = self.text[self.pos]
        self.pos += 1
        if char == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char

    def _read_quoted_string(self) -> str:
        """Read a quoted string value."""
        self._advance()  # Skip opening quote
        chars: list[str] = []

        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char == '"':
                self._advance()
                return "".join(chars)
            if char == "\\":
                self._advance()
                if self.pos >= len(self.text):
                    raise SExprParseError(
                        "Unexpected end of input in escape sequence",
                        self.pos,
                        self.line,
                        self.column,
                    )
                escape_char = self._advance()
                escaped = "\\" + escape_char
                chars.append(_UNESCAPE_MAP.get(escaped, escape_char))
            else:
                chars.append(self._advance())

        raise SExprParseError("Unterminated string", self.pos, self.line, self.column)

    def _read_atom(self) -> str:
        """Read an unquoted atom value."""
        chars: list[str] = []
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char in ' \t\n\r()"':
                break
            chars.append(self._advance())
        return "".join(chars)


def parse(text: str) -> SExprNode:
    """Parse S-expression text into nested Python structure.

    Args:
        text: S-expression text to parse.

    Returns:
        Parsed S-expression (atom or nested lists).

    Raises:
        SExprParseError: If parsing fails.
    """
    tokenizer = SExprTokenizer(text)
    tokens = list(tokenizer.tokenize())
    if not tokens:
        raise SExprParseError("Empty input", 0, 1, 1)

    result, pos = _parse_node(tokens, 0)
    if pos < len(tokens):
        tok = tokens[pos]
        raise SExprParseError(
            f"Unexpected token after expression: {tok.value}",
            tok.position,
            tok.line,
            tok.column,
        )
    return result


def parse_all(text: str) -> list[SExprNode]:
    """Parse multiple S-expressions from text.

    Args:
        text: S-expression text containing one or more expressions.

    Returns:
        List of parsed S-expressions.

    Raises:
        SExprParseError: If parsing fails.
    """
    tokenizer = SExprTokenizer(text)
    tokens = list(tokenizer.tokenize())
    if not tokens:
        return []

    results: list[SExprNode] = []
    pos = 0
    while pos < len(tokens):
        node, pos = _parse_node(tokens, pos)
        results.append(node)
    return results


def _parse_node(tokens: Sequence[SExprToken], pos: int) -> tuple[SExprNode, int]:
    """Parse a single node from token stream."""
    if pos >= len(tokens):
        raise SExprParseError("Unexpected end of input", 0, 1, 1)

    token = tokens[pos]

    if token.type == "LPAREN":
        return _parse_list(tokens, pos)
    if token.type == "RPAREN":
        raise SExprParseError(
            "Unexpected closing parenthesis",
            token.position,
            token.line,
            token.column,
        )
    if token.type == "STRING":
        return token.value, pos + 1
    # ATOM - try to parse as number
    return _parse_atom(token.value), pos + 1


def _parse_list(tokens: Sequence[SExprToken], pos: int) -> tuple[SExprList, int]:
    """Parse a list from token stream."""
    pos += 1  # Skip LPAREN
    items: SExprList = []

    while pos < len(tokens):
        if tokens[pos].type == "RPAREN":
            return items, pos + 1
        node, pos = _parse_node(tokens, pos)
        items.append(node)

    raise SExprParseError("Unclosed list", 0, 1, 1)


def _parse_atom(value: str) -> SExprAtom:
    """Parse an atom value, converting to number if possible."""
    # Try integer
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string
    return value


def build_list(*items: SExprNode) -> SExprList:
    """Build an S-expression list from items.

    Args:
        items: Elements to include in the list.

    Returns:
        S-expression list.
    """
    return list(items)


def dump(node: SExprNode, *, indent: int = 2, inline_threshold: int = 3) -> str:
    """Dump S-expression node to formatted string.

    Args:
        node: S-expression node to dump.
        indent: Spaces per indentation level.
        inline_threshold: Max items before using multiline format.

    Returns:
        Formatted S-expression string.
    """
    writer = SExprWriter(indent=indent, inline_threshold=inline_threshold)
    return writer.write(node)


def dump_compact(node: SExprNode) -> str:
    """Dump S-expression node to compact (single-line) string.

    Args:
        node: S-expression node to dump.

    Returns:
        Compact S-expression string.
    """
    if isinstance(node, list):
        inner = " ".join(dump_compact(item) for item in node)
        return f"({inner})"
    return format_atom(node)


# KiCad-specific element builders


def kicad_version(version: int = 20240101, generator: str = "coupongen") -> SExprList:
    """Build kicad_pcb header element."""
    return ["kicad_pcb", ["version", version], ["generator", generator]]


def kicad_general(thickness: float = 1.6) -> SExprList:
    """Build general section element."""
    return ["general", ["thickness", thickness]]


def kicad_paper(size: str = "A4") -> SExprList:
    """Build paper size element."""
    return ["paper", size]


def kicad_layers(*layer_defs: tuple[int, str, str]) -> SExprList:
    """Build layers section.

    Args:
        layer_defs: Tuples of (id, name, type) for each layer.

    Returns:
        S-expression list for layers section.
    """
    result: SExprList = ["layers"]
    for layer_id, name, layer_type in layer_defs:
        result.append([layer_id, name, layer_type])
    return result


def kicad_net(net_id: int, name: str) -> SExprList:
    """Build net declaration element."""
    return ["net", net_id, name]


def kicad_gr_rect(
    start_x_nm: int,
    start_y_nm: int,
    end_x_nm: int,
    end_y_nm: int,
    layer: str,
    width: float,
    tstamp: str,
) -> SExprList:
    """Build graphic rectangle element."""
    return [
        "gr_rect",
        ["start", nm_to_mm(start_x_nm), nm_to_mm(start_y_nm)],
        ["end", nm_to_mm(end_x_nm), nm_to_mm(end_y_nm)],
        ["layer", layer],
        ["width", width],
        ["tstamp", tstamp],
    ]


def kicad_gr_line(
    start_x_nm: int,
    start_y_nm: int,
    end_x_nm: int,
    end_y_nm: int,
    layer: str,
    width: float,
    tstamp: str,
) -> SExprList:
    """Build graphic line element."""
    return [
        "gr_line",
        ["start", nm_to_mm(start_x_nm), nm_to_mm(start_y_nm)],
        ["end", nm_to_mm(end_x_nm), nm_to_mm(end_y_nm)],
        ["layer", layer],
        ["width", width],
        ["tstamp", tstamp],
    ]


def kicad_footprint(
    name: str,
    layer: str,
    at_x_nm: int,
    at_y_nm: int,
    rotation_deg: int,
    tstamp: str,
    *children: SExprList,
) -> SExprList:
    """Build footprint element."""
    result: SExprList = [
        "footprint",
        name,
        ["layer", layer],
        ["at", nm_to_mm(at_x_nm), nm_to_mm(at_y_nm), rotation_deg],
        ["tstamp", tstamp],
    ]
    result.extend(children)
    return result


def kicad_segment(
    start_x_nm: int,
    start_y_nm: int,
    end_x_nm: int,
    end_y_nm: int,
    width_nm: int,
    layer: str,
    net_id: int,
    tstamp: str,
) -> SExprList:
    """Build track segment element."""
    return [
        "segment",
        ["start", nm_to_mm(start_x_nm), nm_to_mm(start_y_nm)],
        ["end", nm_to_mm(end_x_nm), nm_to_mm(end_y_nm)],
        ["width", nm_to_mm(width_nm)],
        ["layer", layer],
        ["net", net_id],
        ["tstamp", tstamp],
    ]


def kicad_via(
    x_nm: int,
    y_nm: int,
    diameter_nm: int,
    drill_nm: int,
    layers: tuple[str, str],
    net_id: int,
    tstamp: str,
) -> SExprList:
    """Build via element."""
    return [
        "via",
        ["at", nm_to_mm(x_nm), nm_to_mm(y_nm)],
        ["size", nm_to_mm(diameter_nm)],
        ["drill", nm_to_mm(drill_nm)],
        ["layers", layers[0], layers[1]],
        ["net", net_id],
        ["tstamp", tstamp],
    ]
