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
    canonical_hash_drill,
    canonical_hash_gerber,
    canonical_hash_drc_json,
    canonicalize_board,
    canonicalize_drill,
    canonicalize_drc_json,
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
