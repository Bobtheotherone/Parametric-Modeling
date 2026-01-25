# SPDX-License-Identifier: MIT
"""Additional unit tests for canonicalization utility functions.

Tests additional edge cases for the canonicalization module:
- normalize_line_endings function
- Gerber/drill file edge cases
- DRC JSON edge cases with complex nested structures
- File-based canonicalization utilities

These tests complement test_m1_canonicalize.py by covering additional
utility functions and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from formula_foundry.coupongen.kicad import (
    canonical_hash_file,
    canonical_hash_files,
    canonicalize_drc_json,
    canonicalize_drill,
    canonicalize_export,
    canonicalize_gerber,
    normalize_line_endings,
)


class TestNormalizeLineEndings:
    """Tests for normalize_line_endings utility function."""

    def test_lf_unchanged(self) -> None:
        """Unix line endings (LF) should remain unchanged."""
        text = "line1\nline2\nline3\n"
        result = normalize_line_endings(text)
        assert result == text

    def test_crlf_to_lf(self) -> None:
        """Windows line endings (CRLF) should convert to LF."""
        text = "line1\r\nline2\r\nline3\r\n"
        result = normalize_line_endings(text)
        assert result == "line1\nline2\nline3\n"
        assert "\r" not in result

    def test_cr_to_lf(self) -> None:
        """Old Mac line endings (CR) should convert to LF."""
        text = "line1\rline2\rline3\r"
        result = normalize_line_endings(text)
        assert result == "line1\nline2\nline3\n"
        assert "\r" not in result

    def test_mixed_line_endings(self) -> None:
        """Mixed line endings should all convert to LF."""
        text = "line1\r\nline2\nline3\rline4\n"
        result = normalize_line_endings(text)
        assert result == "line1\nline2\nline3\nline4\n"
        assert "\r" not in result

    def test_empty_string(self) -> None:
        """Empty string should remain empty."""
        assert normalize_line_endings("") == ""

    def test_no_line_endings(self) -> None:
        """Text without line endings should remain unchanged."""
        text = "single line without newline"
        result = normalize_line_endings(text)
        assert result == text

    def test_only_crlf(self) -> None:
        """String with only CRLF should become single LF."""
        text = "\r\n"
        result = normalize_line_endings(text)
        assert result == "\n"

    def test_only_cr(self) -> None:
        """String with only CR should become single LF."""
        text = "\r"
        result = normalize_line_endings(text)
        assert result == "\n"

    def test_consecutive_crlf(self) -> None:
        """Multiple consecutive CRLFs should become multiple LFs."""
        text = "\r\n\r\n\r\n"
        result = normalize_line_endings(text)
        assert result == "\n\n\n"

    def test_crlf_in_middle(self) -> None:
        """CRLF in the middle of content."""
        text = "before\r\nafter"
        result = normalize_line_endings(text)
        assert result == "before\nafter"


class TestCanonicalizeExport:
    """Tests for canonicalize_export (combined Gerber/drill handler)."""

    def test_strips_gerber_comments(self) -> None:
        """Should strip G04 Gerber comments."""
        text = "G04 This is a comment*\nX0Y0D02*\n"
        result = canonicalize_export(text)
        assert "G04" not in result
        assert "X0Y0D02*" in result

    def test_strips_drill_comments(self) -> None:
        """Should strip semicolon drill comments."""
        text = "; This is a drill comment\nM48\nT1C0.3\n"
        result = canonicalize_export(text)
        assert ";" not in result
        assert "M48" in result
        assert "T1C0.3" in result

    def test_strips_both_comment_types(self) -> None:
        """Should strip both G04 and semicolon comments."""
        text = "G04 Gerber comment*\n; Drill comment\nX0Y0D02*\nM48\n"
        result = canonicalize_export(text)
        assert "G04" not in result
        assert ";" not in result
        assert "X0Y0D02*" in result
        assert "M48" in result

    def test_strips_creation_date(self) -> None:
        """Should strip TF.CreationDate attributes."""
        text = "%TF.CreationDate,2026-01-20*%\nX0Y0D02*\n"
        result = canonicalize_export(text)
        assert "CreationDate" not in result
        assert "X0Y0D02*" in result

    def test_normalizes_line_endings(self) -> None:
        """Should normalize CRLF to LF."""
        text = "X0Y0D02*\r\nX1Y1D01*\r\n"
        result = canonicalize_export(text)
        assert "\r" not in result

    def test_ensures_trailing_newline(self) -> None:
        """Result should end with newline."""
        text = "X0Y0D02*"
        result = canonicalize_export(text)
        assert result.endswith("\n")

    def test_removes_empty_lines(self) -> None:
        """Empty lines should be removed."""
        text = "X0Y0D02*\n\n\nX1Y1D01*\n"
        result = canonicalize_export(text)
        lines = [l for l in result.split("\n") if l]
        assert len(lines) == 2


class TestCanonicalizeGerberEdgeCases:
    """Additional edge case tests for Gerber canonicalization."""

    def test_aperture_macro(self) -> None:
        """Aperture macros should be preserved."""
        text = "%AMROUNDRECT*21,1,$1,$2,0,0,0*1,1,$3,$4,$5*%\nX0Y0D10*\n"
        result = canonicalize_gerber(text)
        assert "%AMROUNDRECT*21,1,$1,$2,0,0,0*1,1,$3,$4,$5*%" in result

    def test_format_specification(self) -> None:
        """Format specification should be preserved."""
        text = "%FSLAX35Y35*%\nG04 Comment*\nD10*\n"
        result = canonicalize_gerber(text)
        assert "%FSLAX35Y35*%" in result
        assert "G04" not in result

    def test_units_command(self) -> None:
        """Units command should be preserved."""
        text = "%MOMM*%\n%LPD*%\nX0Y0D02*\n"
        result = canonicalize_gerber(text)
        assert "%MOMM*%" in result
        assert "%LPD*%" in result

    def test_step_and_repeat(self) -> None:
        """Step and repeat commands should be preserved."""
        text = "%SRX3Y2I5.0J4.0*%\nD10*\nX0Y0D03*\n%SR*%\n"
        result = canonicalize_gerber(text)
        assert "%SRX3Y2I5.0J4.0*%" in result
        assert "%SR*%" in result

    def test_negative_coordinates(self) -> None:
        """Negative coordinates should be preserved."""
        text = "X-100000Y-200000D02*\nX100000Y200000D01*\n"
        result = canonicalize_gerber(text)
        assert "X-100000Y-200000D02*" in result


class TestCanonicalizeDrillEdgeCases:
    """Additional edge case tests for drill file canonicalization."""

    def test_excellon_routing_commands(self) -> None:
        """Routing commands should be preserved."""
        text = "M48\nT1C0.3\n%\nG00X1000Y1000\nG01X2000Y1000\nM30\n"
        result = canonicalize_drill(text)
        assert "G00X1000Y1000" in result
        assert "G01X2000Y1000" in result

    def test_tool_change_sequence(self) -> None:
        """Tool change commands should be preserved."""
        text = "M48\nT1C0.3\nT2C0.5\n%\nT1\nX1000Y1000\nT2\nX2000Y2000\nM30\n"
        result = canonicalize_drill(text)
        assert "T1\n" in result
        assert "T2\n" in result
        assert "X1000Y1000" in result
        assert "X2000Y2000" in result

    def test_repeat_command(self) -> None:
        """Repeat commands should be preserved."""
        text = "M48\nT1C0.3\n%\nR5X1000Y0\nM30\n"
        result = canonicalize_drill(text)
        assert "R5X1000Y0" in result

    def test_canned_cycle_commands(self) -> None:
        """Canned cycle commands should be preserved."""
        text = "M48\nT1C0.3\n%\nG81\nX1000Y1000\nG80\nM30\n"
        result = canonicalize_drill(text)
        assert "G81" in result
        assert "G80" in result


class TestCanonicalizeDrcJsonEdgeCases:
    """Additional edge case tests for DRC JSON canonicalization."""

    def test_preserves_numeric_types(self) -> None:
        """Numeric types should be preserved."""
        drc = {"value_int": 42, "value_float": 3.14, "violations": []}
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert parsed["value_int"] == 42
        assert parsed["value_float"] == 3.14

    def test_preserves_boolean_types(self) -> None:
        """Boolean types should be preserved."""
        drc = {"is_valid": True, "has_errors": False, "violations": []}
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert parsed["is_valid"] is True
        assert parsed["has_errors"] is False

    def test_handles_unicode_strings(self) -> None:
        """Unicode strings should be preserved."""
        drc = {"message": "Violation: µm tolerance exceeded", "violations": []}
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert parsed["message"] == "Violation: µm tolerance exceeded"

    def test_handles_empty_list_values(self) -> None:
        """Empty lists should be preserved."""
        drc = {"violations": [], "unconnected_items": [], "schematic_parity": []}
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert parsed["violations"] == []
        assert parsed["unconnected_items"] == []
        assert parsed["schematic_parity"] == []

    def test_handles_nested_arrays(self) -> None:
        """Nested arrays should be preserved."""
        drc = {"coordinates": [[1, 2], [3, 4], [5, 6]], "violations": []}
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert parsed["coordinates"] == [[1, 2], [3, 4], [5, 6]]

    def test_sorts_complex_violation_structures(self) -> None:
        """Complex violation structures should be sorted deterministically."""
        drc = {
            "violations": [
                {"type": "clearance", "layer": "B.Cu", "pos": {"x": 100, "y": 200}},
                {"type": "clearance", "layer": "F.Cu", "pos": {"x": 100, "y": 200}},
            ]
        }
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        # Violations should be sorted, B.Cu before F.Cu
        assert parsed["violations"][0]["layer"] == "B.Cu"
        assert parsed["violations"][1]["layer"] == "F.Cu"

    def test_handles_mixed_types_in_list(self) -> None:
        """Lists with mixed types should be preserved."""
        drc = {"mixed": [1, "two", 3.0, True, None], "violations": []}
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert parsed["mixed"] == [1, "two", 3.0, True, None]

    def test_removes_path_variations(self) -> None:
        """Multiple path key variations should be normalized."""
        drc = {
            "file_path": "/absolute/path/file.txt",
            "source_file": "/other/source.kicad_pcb",
            "board_file": "C:\\Windows\\board.kicad_pcb",
            "violations": [],
        }
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        # All paths should be normalized to just filenames
        assert parsed["file_path"] == "file.txt"
        assert parsed["source_file"] == "source.kicad_pcb"
        assert parsed["board_file"] == "board.kicad_pcb"


class TestCanonicalHashFile:
    """Tests for file-based canonical hash functions."""

    def test_hash_kicad_pcb_file(self, tmp_path: Path) -> None:
        """Should use PCB canonicalization for .kicad_pcb files."""
        pcb_file = tmp_path / "test.kicad_pcb"
        pcb_file.write_text(
            "(kicad_pcb (tstamp 12345) (uuid abcde) (net 1))", encoding="utf-8"
        )

        hash_result = canonical_hash_file(pcb_file)

        assert len(hash_result) == 64
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_hash_drill_file(self, tmp_path: Path) -> None:
        """Should use drill canonicalization for .drl files."""
        drl_file = tmp_path / "test.drl"
        drl_file.write_text(
            "M48\n; Comment to be stripped\nT1C0.3\n%\nM30\n", encoding="utf-8"
        )

        hash_result = canonical_hash_file(drl_file)

        assert len(hash_result) == 64

    def test_hash_excellon_file(self, tmp_path: Path) -> None:
        """Should use drill canonicalization for .xln files."""
        xln_file = tmp_path / "test.xln"
        xln_file.write_text("M48\nT1C0.3\n%\nM30\n", encoding="utf-8")

        hash_result = canonical_hash_file(xln_file)

        assert len(hash_result) == 64

    def test_hash_gerber_file(self, tmp_path: Path) -> None:
        """Should use Gerber canonicalization for layer files."""
        gbr_file = tmp_path / "test-F_Cu.gbr"
        gbr_file.write_text(
            "G04 Comment*\n%FSLAX35Y35*%\nX0Y0D02*\nM02*\n", encoding="utf-8"
        )

        hash_result = canonical_hash_file(gbr_file)

        assert len(hash_result) == 64


class TestCanonicalHashFiles:
    """Tests for batch file hashing."""

    def test_hash_multiple_files(self, tmp_path: Path) -> None:
        """Should hash multiple files and return dict."""
        # Create test files
        (tmp_path / "layer1.gbr").write_text("X0Y0D02*\n", encoding="utf-8")
        (tmp_path / "layer2.gbr").write_text("X1Y1D02*\n", encoding="utf-8")
        (tmp_path / "drill.drl").write_text("M48\nT1C0.3\n%\nM30\n", encoding="utf-8")

        files = list(tmp_path.glob("*"))
        result = canonical_hash_files(files)

        assert len(result) == 3
        assert "layer1.gbr" in result
        assert "layer2.gbr" in result
        assert "drill.drl" in result
        # Different content should produce different hashes
        assert result["layer1.gbr"] != result["layer2.gbr"]

    def test_hash_empty_file_list(self) -> None:
        """Should return empty dict for empty file list."""
        result = canonical_hash_files([])
        assert result == {}

    def test_hash_files_deterministic(self, tmp_path: Path) -> None:
        """Hashing same files should produce same results."""
        (tmp_path / "test.gbr").write_text("X0Y0D02*\n", encoding="utf-8")

        files = [tmp_path / "test.gbr"]
        result1 = canonical_hash_files(files)
        result2 = canonical_hash_files(files)

        assert result1 == result2
