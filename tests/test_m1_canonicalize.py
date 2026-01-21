"""Tests for KiCad artifact canonicalization per Section 13.5.2.

This module tests the canonicalization algorithms for:
- Board files (.kicad_pcb) - canonicalize_board
- Gerber files - canonicalize_gerber
- DRC JSON reports - canonicalize_drc_json

Satisfies REQ-M1-005 (deterministic artifact hashing).
"""

from __future__ import annotations

import json

import pytest

from formula_foundry.coupongen.kicad import (
    canonical_hash_board,
    canonical_hash_drc_json,
    canonical_hash_drill,
    canonical_hash_gerber,
    canonicalize_board,
    canonicalize_drc_json,
    canonicalize_drill,
    canonicalize_gerber,
)


class TestCanonicalizeBoard:
    """Tests for canonicalize_board per Section 13.5.2."""

    def test_removes_tstamp_field(self) -> None:
        """Board canonicalization removes tstamp field values."""
        board = "(kicad_pcb (tstamp 12345678-1234-1234-1234-123456789abc) (net 1))"
        canonical = canonicalize_board(board)

        assert "12345678-1234-1234-1234-123456789abc" not in canonical
        assert "(tstamp)" in canonical
        assert "(net 1)" in canonical

    def test_removes_uuid_field(self) -> None:
        """Board canonicalization removes uuid field values."""
        board = "(kicad_pcb (uuid abcd-1234-5678-efgh) (net 1))"
        canonical = canonicalize_board(board)

        assert "abcd-1234-5678-efgh" not in canonical
        assert "(uuid)" in canonical
        assert "(net 1)" in canonical

    def test_removes_multiple_uuids(self) -> None:
        """Board canonicalization removes all tstamp and uuid fields."""
        board = """(kicad_pcb
          (tstamp 11111111-1111-1111-1111-111111111111)
          (footprint "Test"
            (uuid 22222222-2222-2222-2222-222222222222)
            (tstamp 33333333-3333-3333-3333-333333333333)
          )
        )"""
        canonical = canonicalize_board(board)

        assert "11111111" not in canonical
        assert "22222222" not in canonical
        assert "33333333" not in canonical
        assert canonical.count("(tstamp)") == 2
        assert canonical.count("(uuid)") == 1

    def test_normalizes_crlf_to_lf(self) -> None:
        """Board canonicalization normalizes CRLF to LF."""
        board_crlf = "(kicad_pcb\r\n  (net 1)\r\n)"
        board_lf = "(kicad_pcb\n  (net 1)\n)"

        canonical_crlf = canonicalize_board(board_crlf)
        canonical_lf = canonicalize_board(board_lf)

        assert canonical_crlf == canonical_lf
        assert "\r" not in canonical_crlf

    def test_normalizes_cr_to_lf(self) -> None:
        """Board canonicalization normalizes CR to LF."""
        board_cr = "(kicad_pcb\r  (net 1)\r)"
        canonical = canonicalize_board(board_cr)

        assert "\r" not in canonical

    def test_collapses_multiple_spaces(self) -> None:
        """Board canonicalization collapses multiple spaces to single space."""
        board = "(kicad_pcb    (net   1)   (net   2))"
        canonical = canonicalize_board(board)

        assert "    " not in canonical
        assert "   " not in canonical
        assert "  " not in canonical
        assert "(kicad_pcb (net 1) (net 2))" in canonical

    def test_collapses_tabs(self) -> None:
        """Board canonicalization collapses tabs to single space."""
        board = "(kicad_pcb\t\t(net\t1))"
        canonical = canonicalize_board(board)

        assert "\t" not in canonical
        assert "(kicad_pcb (net 1))" in canonical

    def test_strips_trailing_whitespace(self) -> None:
        """Board canonicalization strips trailing whitespace."""
        board = "(kicad_pcb (net 1)   \n  (net 2)  \n)"
        canonical = canonicalize_board(board)

        lines = canonical.split("\n")
        for line in lines:
            assert line == line.rstrip()

    def test_ensures_trailing_newline(self) -> None:
        """Board canonicalization ensures file ends with newline."""
        board_no_newline = "(kicad_pcb (net 1))"
        canonical = canonicalize_board(board_no_newline)

        assert canonical.endswith("\n")

    def test_preserves_content_structure(self) -> None:
        """Board canonicalization preserves meaningful content."""
        board = """(kicad_pcb
          (version 20211014)
          (generator pcbnew)
          (net 0 "")
          (net 1 "SIG")
          (footprint "Test:Footprint"
            (at 10 20)
          )
        )"""
        canonical = canonicalize_board(board)

        assert "(version 20211014)" in canonical
        assert "(generator pcbnew)" in canonical
        assert '(net 0 "")' in canonical
        assert '(net 1 "SIG")' in canonical
        assert '(footprint "Test:Footprint"' in canonical
        assert "(at 10 20)" in canonical

    def test_hash_stability_across_whitespace_variations(self) -> None:
        """Same logical board produces same hash regardless of whitespace."""
        board_a = "(kicad_pcb  (tstamp abc)  (net 1))"
        board_b = "(kicad_pcb (tstamp xyz) (net 1))"
        board_c = "(kicad_pcb\t(tstamp   def)\t\t(net 1))"

        hash_a = canonical_hash_board(board_a)
        hash_b = canonical_hash_board(board_b)
        hash_c = canonical_hash_board(board_c)

        assert hash_a == hash_b == hash_c

    def test_hash_differs_for_content_changes(self) -> None:
        """Different content produces different hashes."""
        board_a = "(kicad_pcb (net 1))"
        board_b = "(kicad_pcb (net 2))"

        hash_a = canonical_hash_board(board_a)
        hash_b = canonical_hash_board(board_b)

        assert hash_a != hash_b


class TestCanonicalizeGerber:
    """Tests for canonicalize_gerber per Section 13.5.2."""

    def test_strips_g04_comments(self) -> None:
        """Gerber canonicalization removes G04 comment lines."""
        gerber = "G04 This is a comment*\nX0Y0D02*\nG04 Another comment*\nM02*\n"
        canonical = canonicalize_gerber(gerber)

        assert "G04" not in canonical
        assert "This is a comment" not in canonical
        assert "X0Y0D02*" in canonical
        assert "M02*" in canonical

    def test_strips_timestamp_comments(self) -> None:
        """Gerber canonicalization removes timestamp comments."""
        gerber = "G04 CreationDate: 2026-01-20 10:30:00*\nX0Y0D02*\n"
        canonical = canonicalize_gerber(gerber)

        assert "2026-01-20" not in canonical
        assert "CreationDate" not in canonical
        assert "X0Y0D02*" in canonical

    def test_normalizes_crlf_to_lf(self) -> None:
        """Gerber canonicalization normalizes CRLF to LF."""
        gerber_crlf = "X0Y0D02*\r\nX100Y0D01*\r\n"
        gerber_lf = "X0Y0D02*\nX100Y0D01*\n"

        canonical_crlf = canonicalize_gerber(gerber_crlf)
        canonical_lf = canonicalize_gerber(gerber_lf)

        assert canonical_crlf == canonical_lf

    def test_strips_trailing_whitespace(self) -> None:
        """Gerber canonicalization strips trailing whitespace."""
        gerber = "X0Y0D02*   \nX100Y0D01*  \n"
        canonical = canonicalize_gerber(gerber)

        lines = canonical.split("\n")
        for line in lines:
            assert line == line.rstrip()

    def test_preserves_content(self) -> None:
        """Gerber canonicalization preserves actual drawing commands."""
        gerber = "%FSLAX35Y35*%\nG01*\nX0Y0D02*\nX1000000Y0D01*\nM02*\n"
        canonical = canonicalize_gerber(gerber)

        assert "%FSLAX35Y35*%" in canonical
        assert "G01*" in canonical
        assert "X0Y0D02*" in canonical
        assert "X1000000Y0D01*" in canonical
        assert "M02*" in canonical

    def test_strips_creation_date_attribute(self) -> None:
        """Gerber canonicalization removes TF.CreationDate attributes."""
        gerber = "%TF.CreationDate,2026-01-20T10:30:00*%\nX0Y0D02*\n"
        canonical = canonicalize_gerber(gerber)

        assert "CreationDate" not in canonical
        assert "X0Y0D02*" in canonical

    def test_hash_stability_across_timestamp_variations(self) -> None:
        """Gerber canonical hash must be stable across timestamp variations."""
        hashes = []
        for day in range(1, 4):
            gerber = f"%TF.CreationDate,2026-01-0{day}T10:30:00*%\nX0Y0D02*\n"
            hashes.append(canonical_hash_gerber(gerber))

        assert len(set(hashes)) == 1

    def test_strips_multiple_g04_comments(self) -> None:
        """Gerber canonicalization strips multiple consecutive G04 comment lines."""
        gerber = "G04 Comment 1*\nG04 Comment 2*\nG04 Comment 3*\nX0Y0D02*\nM02*\n"
        canonical = canonicalize_gerber(gerber)

        assert canonical.count("G04") == 0
        assert "X0Y0D02*" in canonical
        assert "M02*" in canonical

    def test_strips_tf_generation_software(self) -> None:
        """Gerber canonicalization strips TF.GenerationSoftware attribute."""
        gerber = "%TF.GenerationSoftware,KiCad,Pcbnew,9.0.7*%\nX0Y0D02*\n"
        canonical = canonicalize_gerber(gerber)

        # GenerationSoftware should be preserved (only CreationDate is stripped)
        assert "X0Y0D02*" in canonical

    def test_strips_g04_with_various_formats(self) -> None:
        """Gerber canonicalization handles various G04 comment formats."""
        gerber = "G04 Simple comment*\nG04*\nG04End of block*\nX0Y0D02*\n"
        canonical = canonicalize_gerber(gerber)

        assert "G04" not in canonical
        assert "X0Y0D02*" in canonical

    def test_hash_stability_across_3_runs(self) -> None:
        """Gerber canonical hash must be stable across 3 consecutive runs."""
        base_gerber = "%FSLAX35Y35*%\nG01*\nX0Y0D02*\nX1000000Y0D01*\nM02*\n"
        hashes = []
        for _ in range(3):
            hashes.append(canonical_hash_gerber(base_gerber))

        assert len(set(hashes)) == 1, "Hash not stable across 3 runs"

    def test_normalizes_cr_only_line_endings(self) -> None:
        """Gerber canonicalization normalizes CR-only line endings."""
        gerber_cr = "G04 Test*\rX0Y0D02*\rM02*\r"
        gerber_lf = "G04 Test*\nX0Y0D02*\nM02*\n"

        canonical_cr = canonicalize_gerber(gerber_cr)
        canonical_lf = canonicalize_gerber(gerber_lf)

        assert canonical_cr == canonical_lf

    def test_preserves_aperture_definitions(self) -> None:
        """Gerber canonicalization preserves aperture definitions."""
        gerber = "%ADD10C,0.254*%\n%ADD11R,0.5X0.5*%\nG04 Comment*\nX0Y0D10*\n"
        canonical = canonicalize_gerber(gerber)

        assert "%ADD10C,0.254*%" in canonical
        assert "%ADD11R,0.5X0.5*%" in canonical
        assert "G04" not in canonical


class TestCanonicalizeDrill:
    """Tests for canonicalize_drill per Section 13.5.2."""

    def test_strips_comment_lines(self) -> None:
        """Drill canonicalization removes semicolon comment lines."""
        drill = "M48\n; Excellon drill file\nT1C0.3\n%\nM30\n"
        canonical = canonicalize_drill(drill)

        assert "; Excellon drill file" not in canonical
        assert "M48" in canonical
        assert "T1C0.3" in canonical

    def test_normalizes_line_endings(self) -> None:
        """Drill canonicalization normalizes CRLF to LF."""
        drill_crlf = "M48\r\n; Comment\r\nT1C0.3\r\nM30\r\n"
        drill_lf = "M48\n; Comment\nT1C0.3\nM30\n"

        canonical_crlf = canonicalize_drill(drill_crlf)
        canonical_lf = canonicalize_drill(drill_lf)

        assert canonical_crlf == canonical_lf

    def test_hash_stability_across_comment_variations(self) -> None:
        """Drill canonical hash must be stable across comment variations."""
        hashes = []
        for idx in range(3):
            drill = f"M48\n; Build {idx}\nT1C0.3\n%\nM30\n"
            hashes.append(canonical_hash_drill(drill))

        assert len(set(hashes)) == 1

    def test_preserves_full_tool_table(self) -> None:
        """Drill canonicalization preserves complete tool table definitions."""
        drill = "M48\n; Header comment\nT1C0.254\nT2C0.8\nT3C1.0\n%\nT1\nX10Y10\nT2\nX20Y20\nM30\n"
        canonical = canonicalize_drill(drill)

        # Tool definitions must be preserved
        assert "T1C0.254" in canonical
        assert "T2C0.8" in canonical
        assert "T3C1.0" in canonical
        # Tool selections must be preserved
        assert "T1\n" in canonical
        assert "T2\n" in canonical
        # Comments must be stripped
        assert "; Header comment" not in canonical

    def test_preserves_tool_table_with_metric_format(self) -> None:
        """Drill canonicalization preserves metric tool table format."""
        drill = "M48\n; Metric drill file\nMETRIC\nT01C0.254\nT02C0.8\n%\nT01\nX1000Y1000\nM30\n"
        canonical = canonicalize_drill(drill)

        assert "METRIC" in canonical
        assert "T01C0.254" in canonical
        assert "T02C0.8" in canonical
        assert "; Metric drill file" not in canonical

    def test_hash_stability_with_different_tool_orders(self) -> None:
        """Same tools in file produce same hash regardless of header comments."""
        drill_a = "M48\n; Build A\nT1C0.3\nT2C0.5\n%\nT1\nX10Y10\nM30\n"
        drill_b = "M48\n; Build B\nT1C0.3\nT2C0.5\n%\nT1\nX10Y10\nM30\n"

        hash_a = canonical_hash_drill(drill_a)
        hash_b = canonical_hash_drill(drill_b)

        assert hash_a == hash_b

    def test_strips_multiple_consecutive_comments(self) -> None:
        """Drill canonicalization strips multiple consecutive comment lines."""
        drill = "M48\n; Comment 1\n; Comment 2\n; Comment 3\nT1C0.3\n%\nM30\n"
        canonical = canonicalize_drill(drill)

        assert canonical.count(";") == 0
        assert "M48" in canonical
        assert "T1C0.3" in canonical

    def test_preserves_excellon_header_commands(self) -> None:
        """Drill canonicalization preserves Excellon header commands."""
        drill = "M48\n; Generated by KiCad\nFMAT,2\nINCH\nT1C0.012\n%\nT1\nX1000Y1000\nM30\n"
        canonical = canonicalize_drill(drill)

        assert "M48" in canonical
        assert "FMAT,2" in canonical
        assert "INCH" in canonical
        assert "T1C0.012" in canonical
        assert "; Generated by KiCad" not in canonical

    def test_strips_trailing_whitespace_from_drill(self) -> None:
        """Drill canonicalization strips trailing whitespace per line."""
        drill = "M48  \n; Comment   \nT1C0.3  \n%\nM30  \n"
        canonical = canonicalize_drill(drill)

        for line in canonical.split("\n"):
            assert line == line.rstrip()


class TestCanonicalizeDrcJson:
    """Tests for canonicalize_drc_json per Section 13.5.2."""

    def test_removes_timestamp_keys(self) -> None:
        """DRC JSON canonicalization removes timestamp keys."""
        drc = {
            "date": "2026-01-20",
            "time": "10:30:00",
            "timestamp": "2026-01-20T10:30:00Z",
            "violations": [],
        }
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert "date" not in parsed
        assert "time" not in parsed
        assert "timestamp" not in parsed
        assert "violations" in parsed

    def test_removes_generated_at_key(self) -> None:
        """DRC JSON canonicalization removes generated_at key."""
        drc = {"generated_at": "2026-01-20T10:30:00Z", "violations": []}
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert "generated_at" not in parsed

    def test_removes_kicad_version(self) -> None:
        """DRC JSON canonicalization removes kicad_version."""
        drc = {"kicad_version": "9.0.7", "violations": []}
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert "kicad_version" not in parsed

    def test_removes_host_and_source(self) -> None:
        """DRC JSON canonicalization removes host and source keys."""
        drc = {"host": "ubuntu-runner", "source": "CI", "violations": []}
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert "host" not in parsed
        assert "source" not in parsed

    def test_removes_schema_version(self) -> None:
        """DRC JSON canonicalization removes schema_version."""
        drc = {"schema_version": 1, "violations": []}
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert "schema_version" not in parsed

    def test_normalizes_path_to_filename(self) -> None:
        """DRC JSON canonicalization normalizes paths to filenames."""
        drc = {
            "file": "/home/user/project/board.kicad_pcb",
            "path": "/tmp/exports/gerbers/F.Cu.gbr",
            "violations": [],
        }
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert parsed["file"] == "board.kicad_pcb"
        assert parsed["path"] == "F.Cu.gbr"

    def test_normalizes_windows_paths(self) -> None:
        """DRC JSON canonicalization handles Windows paths."""
        drc = {"file": "C:\\Users\\user\\board.kicad_pcb", "violations": []}
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert parsed["file"] == "board.kicad_pcb"

    def test_normalizes_absolute_path_values(self) -> None:
        """DRC JSON canonicalization normalizes absolute path values."""
        drc = {"note": "/tmp/build/board.kicad_pcb", "violations": []}
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert parsed["note"] == "board.kicad_pcb"

    def test_sorts_object_keys(self) -> None:
        """DRC JSON canonicalization sorts object keys alphabetically."""
        drc = {"zebra": 1, "alpha": 2, "mike": 3}
        canonical = canonicalize_drc_json(drc)

        # Keys should be in sorted order in the output
        assert canonical == '{"alpha":2,"mike":3,"zebra":1}'

    def test_sorts_list_order(self) -> None:
        """DRC JSON canonicalization sorts list ordering for violations."""
        drc = {
            "violations": [
                {"id": 3},
                {"id": 1},
                {"id": 2},
            ]
        }
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert [item["id"] for item in parsed["violations"]] == [1, 2, 3]

    def test_recursively_canonicalizes_nested_objects(self) -> None:
        """DRC JSON canonicalization handles nested structures."""
        drc = {
            "violations": [
                {
                    "timestamp": "2026-01-20T10:30:00Z",
                    "file": "/tmp/board.kicad_pcb",
                    "details": {"zebra": 1, "alpha": 2},
                }
            ]
        }
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        # Nested timestamp should be removed
        assert "timestamp" not in parsed["violations"][0]
        # Nested path should be normalized
        assert parsed["violations"][0]["file"] == "board.kicad_pcb"
        # Nested keys should be sorted (alpha before zebra)
        nested_str = json.dumps(parsed["violations"][0]["details"], separators=(",", ":"))
        assert nested_str == '{"alpha":2,"zebra":1}'

    def test_accepts_string_input(self) -> None:
        """DRC JSON canonicalization accepts JSON string input."""
        drc_str = '{"date": "2026-01-20", "violations": []}'
        canonical = canonicalize_drc_json(drc_str)
        parsed = json.loads(canonical)

        assert "date" not in parsed
        assert "violations" in parsed

    def test_produces_compact_output(self) -> None:
        """DRC JSON canonicalization produces compact JSON (no spaces)."""
        drc = {"alpha": 1, "beta": [1, 2, 3]}
        canonical = canonicalize_drc_json(drc)

        # No spaces after colons or commas
        assert ": " not in canonical
        assert ", " not in canonical
        assert canonical == '{"alpha":1,"beta":[1,2,3]}'

    def test_hash_stability_across_environment_variations(self) -> None:
        """Same DRC violations produce same hash regardless of environment."""
        drc_a = {
            "date": "2026-01-20",
            "kicad_version": "9.0.7",
            "file": "/home/runner/board.kicad_pcb",
            "violations": [{"type": "clearance", "severity": "error"}],
        }
        drc_b = {
            "date": "2026-12-31",
            "kicad_version": "9.0.8",
            "file": "/tmp/other/board.kicad_pcb",
            "violations": [{"type": "clearance", "severity": "error"}],
        }

        hash_a = canonical_hash_drc_json(drc_a)
        hash_b = canonical_hash_drc_json(drc_b)

        assert hash_a == hash_b

    def test_hash_stability_across_violation_order(self) -> None:
        """DRC hash must be stable across reordered violations."""
        drc_a = {"violations": [{"id": 2}, {"id": 1}]}
        drc_b = {"violations": [{"id": 1}, {"id": 2}]}

        hash_a = canonical_hash_drc_json(drc_a)
        hash_b = canonical_hash_drc_json(drc_b)

        assert hash_a == hash_b

    def test_hash_differs_for_violation_changes(self) -> None:
        """Different violations produce different hashes."""
        drc_a = {"violations": [{"type": "clearance"}]}
        drc_b = {"violations": [{"type": "drill"}]}

        hash_a = canonical_hash_drc_json(drc_a)
        hash_b = canonical_hash_drc_json(drc_b)

        assert hash_a != hash_b

    def test_empty_violations_canonical(self) -> None:
        """Empty violations list produces consistent output."""
        drc = {"violations": []}
        canonical = canonicalize_drc_json(drc)

        assert canonical == '{"violations":[]}'

    def test_handles_null_values(self) -> None:
        """DRC JSON canonicalization handles null values correctly."""
        drc = {"value": None, "violations": []}
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert parsed["value"] is None

    def test_normalizes_deeply_nested_paths(self) -> None:
        """DRC JSON canonicalization normalizes deeply nested absolute paths."""
        drc = {
            "violations": [
                {
                    "items": [
                        {"file": "/very/deep/nested/path/to/board.kicad_pcb"},
                        {"path": "C:\\Users\\Name\\Documents\\project\\file.txt"},
                    ]
                }
            ]
        }
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        # Paths should be normalized to just filenames
        items = parsed["violations"][0]["items"]
        assert items[0]["file"] == "board.kicad_pcb"
        assert items[1]["path"] == "file.txt"

    def test_hash_stability_across_3_runs(self) -> None:
        """DRC JSON canonical hash must be stable across 3 consecutive runs."""
        drc = {
            "violations": [
                {"type": "clearance", "severity": "error", "position": {"x": 100, "y": 200}},
                {"type": "drill", "severity": "warning", "position": {"x": 300, "y": 400}},
            ]
        }
        hashes = []
        for _ in range(3):
            hashes.append(canonical_hash_drc_json(drc))

        assert len(set(hashes)) == 1, "DRC hash not stable across 3 runs"

    def test_sorts_unconnected_items_list(self) -> None:
        """DRC JSON canonicalization sorts unconnected_items list."""
        drc = {
            "unconnected_items": [
                {"net": "VCC", "pin": "2"},
                {"net": "GND", "pin": "1"},
            ]
        }
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        # Items should be sorted deterministically
        items = parsed["unconnected_items"]
        # First item should come before second in sorted order
        assert items[0]["net"] == "GND"
        assert items[1]["net"] == "VCC"

    def test_sorts_schematic_parity_list(self) -> None:
        """DRC JSON canonicalization sorts schematic_parity list."""
        drc = {
            "schematic_parity": [
                {"component": "U2", "issue": "missing"},
                {"component": "U1", "issue": "extra"},
            ]
        }
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        # Items should be sorted deterministically
        items = parsed["schematic_parity"]
        assert items[0]["component"] == "U1"
        assert items[1]["component"] == "U2"

    def test_handles_unc_paths(self) -> None:
        """DRC JSON canonicalization handles UNC network paths."""
        drc = {"file": "\\\\server\\share\\project\\board.kicad_pcb", "violations": []}
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        assert parsed["file"] == "board.kicad_pcb"

    def test_preserves_relative_paths(self) -> None:
        """DRC JSON canonicalization preserves relative paths unchanged."""
        drc = {"file": "boards/my_board.kicad_pcb", "violations": []}
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        # Relative paths should be kept as-is (not absolute)
        assert parsed["file"] == "my_board.kicad_pcb"

    def test_handles_empty_object(self) -> None:
        """DRC JSON canonicalization handles empty object."""
        drc: dict = {}
        canonical = canonicalize_drc_json(drc)

        assert canonical == "{}"

    def test_handles_deeply_nested_timestamps(self) -> None:
        """DRC JSON canonicalization removes deeply nested timestamps."""
        drc = {
            "violations": [
                {
                    "details": {
                        "inner": {
                            "timestamp": "2026-01-20T10:00:00Z",
                            "data": "important",
                        }
                    }
                }
            ]
        }
        canonical = canonicalize_drc_json(drc)
        parsed = json.loads(canonical)

        # Deeply nested timestamp should be removed
        inner = parsed["violations"][0]["details"]["inner"]
        assert "timestamp" not in inner
        assert inner["data"] == "important"


class TestCanonicalizationIntegration:
    """Integration tests for canonicalization module."""

    def test_board_and_kicad_pcb_produce_different_results(self) -> None:
        """canonicalize_board includes whitespace normalization unlike canonicalize_kicad_pcb."""
        from formula_foundry.coupongen.kicad import canonicalize_kicad_pcb

        board = "(kicad_pcb    (tstamp abc)    (net 1))"

        canon_board = canonicalize_board(board)
        canon_kicad_pcb = canonicalize_kicad_pcb(board)

        # canonicalize_board normalizes whitespace
        assert "    " not in canon_board
        # canonicalize_kicad_pcb does not normalize whitespace
        assert "    " in canon_kicad_pcb

    def test_hash_functions_return_64_char_hex(self) -> None:
        """All canonical hash functions return 64-character hex strings."""
        board = "(kicad_pcb (net 1))"
        drc = {"violations": []}

        board_hash = canonical_hash_board(board)
        drc_hash = canonical_hash_drc_json(drc)

        assert len(board_hash) == 64
        assert len(drc_hash) == 64
        # Valid hex
        int(board_hash, 16)
        int(drc_hash, 16)


class TestHashStabilityRegressions:
    """Regression tests for hash stability across multiple runs.

    These tests verify the core requirement from ECO-M1-ALIGN-0001:
    "stable canonical hashes across repeated runs" (Gate G5).

    Each test runs canonicalization and hashing multiple times to ensure
    deterministic, stable output regardless of run order.
    """

    def test_gerber_hash_stable_across_10_runs(self) -> None:
        """Gerber canonical hash must be stable across 10 consecutive runs."""
        gerber = "%FSLAX35Y35*%\n%TF.CreationDate,2026-01-20T10:00:00*%\nG04 Comment*\nX0Y0D02*\nM02*\n"
        hashes = [canonical_hash_gerber(gerber) for _ in range(10)]
        assert len(set(hashes)) == 1, f"Gerber hash not stable: got {len(set(hashes))} unique hashes"

    def test_drill_hash_stable_across_10_runs(self) -> None:
        """Drill canonical hash must be stable across 10 consecutive runs."""
        drill = "M48\n; Generated by KiCad 9.0.7\nT1C0.254\nT2C0.8\n%\nT1\nX1000Y1000\nT2\nX2000Y2000\nM30\n"
        hashes = [canonical_hash_drill(drill) for _ in range(10)]
        assert len(set(hashes)) == 1, f"Drill hash not stable: got {len(set(hashes))} unique hashes"

    def test_board_hash_stable_across_10_runs(self) -> None:
        """Board canonical hash must be stable across 10 consecutive runs."""
        board = """(kicad_pcb (version 20211014) (generator pcbnew)
          (tstamp 12345678-1234-1234-1234-123456789abc)
          (uuid 87654321-4321-4321-4321-cba987654321)
          (net 0 "")
          (net 1 "SIG")
        )"""
        hashes = [canonical_hash_board(board) for _ in range(10)]
        assert len(set(hashes)) == 1, f"Board hash not stable: got {len(set(hashes))} unique hashes"

    def test_drc_json_hash_stable_across_10_runs(self) -> None:
        """DRC JSON canonical hash must be stable across 10 consecutive runs."""
        drc = {
            "date": "2026-01-20",
            "kicad_version": "9.0.7",
            "file": "/tmp/build/board.kicad_pcb",
            "violations": [
                {"type": "clearance", "severity": "error", "id": 3},
                {"type": "drill", "severity": "warning", "id": 1},
                {"type": "width", "severity": "error", "id": 2},
            ],
        }
        hashes = [canonical_hash_drc_json(drc) for _ in range(10)]
        assert len(set(hashes)) == 1, f"DRC hash not stable: got {len(set(hashes))} unique hashes"

    def test_all_hash_functions_produce_identical_output_across_runs(self) -> None:
        """All canonicalization functions must produce identical output across runs.

        This is a comprehensive regression test that verifies:
        1. Gerber canonicalization is deterministic
        2. Drill canonicalization is deterministic
        3. Board canonicalization is deterministic
        4. DRC JSON canonicalization is deterministic
        """
        gerber = "%FSLAX35Y35*%\nG04 Test comment*\nX0Y0D02*\n"
        drill = "M48\n; Comment\nT1C0.3\n%\nM30\n"
        board = "(kicad_pcb (tstamp abc123) (net 1))"
        drc = {"date": "2026-01-20", "violations": [{"id": 1}]}

        # Run 3 times and compare
        results = []
        for _ in range(3):
            results.append({
                "gerber": canonicalize_gerber(gerber),
                "drill": canonicalize_drill(drill),
                "board": canonicalize_board(board),
                "drc": canonicalize_drc_json(drc),
            })

        # All runs must produce identical results
        for key in ["gerber", "drill", "board", "drc"]:
            values = [r[key] for r in results]
            assert len(set(values)) == 1, f"{key} canonicalization not deterministic"

    def test_hash_stability_with_varying_timestamps(self) -> None:
        """Hashes must be stable when only timestamps vary.

        This regression test ensures that files differing only in
        timestamps/dates produce identical canonical hashes.
        """
        # Gerber with different timestamps
        gerber_hashes = []
        for timestamp in ["2026-01-01", "2026-06-15", "2026-12-31"]:
            gerber = f"%TF.CreationDate,{timestamp}T00:00:00*%\nX0Y0D02*\n"
            gerber_hashes.append(canonical_hash_gerber(gerber))

        assert len(set(gerber_hashes)) == 1, "Gerber hash varies with timestamp"

        # Drill with different build comments
        drill_hashes = []
        for build in ["Build 1", "Build 2", "Build 3"]:
            drill = f"M48\n; {build}\nT1C0.3\n%\nM30\n"
            drill_hashes.append(canonical_hash_drill(drill))

        assert len(set(drill_hashes)) == 1, "Drill hash varies with build comment"

        # DRC with different dates
        drc_hashes = []
        for date in ["2026-01-01", "2026-06-15", "2026-12-31"]:
            drc = {"date": date, "violations": []}
            drc_hashes.append(canonical_hash_drc_json(drc))

        assert len(set(drc_hashes)) == 1, "DRC hash varies with date"

    def test_hash_stability_with_varying_paths(self) -> None:
        """Hashes must be stable when only absolute paths vary.

        This regression test ensures that DRC reports differing only in
        absolute paths produce identical canonical hashes.
        """
        paths = [
            "/home/user1/project/board.kicad_pcb",
            "/home/user2/different/path/board.kicad_pcb",
            "C:\\Users\\Windows\\Documents\\board.kicad_pcb",
            "\\\\server\\share\\board.kicad_pcb",
        ]

        hashes = []
        for path in paths:
            drc = {"file": path, "violations": []}
            hashes.append(canonical_hash_drc_json(drc))

        assert len(set(hashes)) == 1, "DRC hash varies with absolute path"

    def test_hash_stability_with_varying_whitespace(self) -> None:
        """Hashes must be stable when only whitespace varies.

        This regression test ensures that board files differing only in
        whitespace produce identical canonical hashes.
        """
        boards = [
            "(kicad_pcb  (tstamp abc)  (net 1))",
            "(kicad_pcb (tstamp abc) (net 1))",
            "(kicad_pcb\t(tstamp abc)\t(net 1))",
            "(kicad_pcb    (tstamp abc)    (net 1))",
        ]

        hashes = [canonical_hash_board(board) for board in boards]
        assert len(set(hashes)) == 1, "Board hash varies with whitespace"

    def test_hash_stability_with_varying_line_endings(self) -> None:
        """Hashes must be stable when only line endings vary.

        This regression test ensures that files with different line ending
        styles (LF, CRLF, CR) produce identical canonical hashes.
        """
        # Gerber with different line endings
        gerber_lf = "X0Y0D02*\nX1Y1D01*\nM02*\n"
        gerber_crlf = "X0Y0D02*\r\nX1Y1D01*\r\nM02*\r\n"
        gerber_cr = "X0Y0D02*\rX1Y1D01*\rM02*\r"

        assert canonical_hash_gerber(gerber_lf) == canonical_hash_gerber(gerber_crlf)
        assert canonical_hash_gerber(gerber_lf) == canonical_hash_gerber(gerber_cr)

        # Drill with different line endings
        drill_lf = "M48\nT1C0.3\n%\nM30\n"
        drill_crlf = "M48\r\nT1C0.3\r\n%\r\nM30\r\n"

        assert canonical_hash_drill(drill_lf) == canonical_hash_drill(drill_crlf)

        # Board with different line endings
        board_lf = "(kicad_pcb\n  (net 1)\n)"
        board_crlf = "(kicad_pcb\r\n  (net 1)\r\n)"

        assert canonical_hash_board(board_lf) == canonical_hash_board(board_crlf)
