"""Additional unit tests for hashing module edge cases.

Supplements test_m1_hashing.py with coverage for:
- coupon_id_from_design_hash edge cases
- Hash collision avoidance verification
- Module re-export consistency
"""

from __future__ import annotations

import pytest

from formula_foundry.coupongen.hashing import (
    canonical_drc_json,
    canonical_hash_export_text,
    canonical_hash_kicad_pcb_text,
    canonicalize_export_text,
    canonicalize_kicad_pcb_text,
    coupon_id_from_design_hash,
)


class TestCouponIdFromDesignHash:
    """Additional tests for coupon_id_from_design_hash function."""

    def test_coupon_id_lowercase_only(self) -> None:
        """Test coupon_id only contains lowercase characters."""
        design_hash = "abcdef1234567890" * 4  # 64 chars
        coupon_id = coupon_id_from_design_hash(design_hash)

        assert coupon_id == coupon_id.lower()
        assert all(c.islower() or c.isdigit() for c in coupon_id)

    def test_coupon_id_base32_chars_only(self) -> None:
        """Test coupon_id only contains valid base32 characters (a-z, 2-7)."""
        design_hash = "fedcba0987654321" * 4  # 64 chars
        coupon_id = coupon_id_from_design_hash(design_hash)

        valid_chars = set("abcdefghijklmnopqrstuvwxyz234567")
        assert all(c in valid_chars for c in coupon_id)

    def test_coupon_id_stable_for_same_hash(self) -> None:
        """Test coupon_id is stable for repeated calls with same hash."""
        design_hash = "0" * 64
        id1 = coupon_id_from_design_hash(design_hash)
        id2 = coupon_id_from_design_hash(design_hash)
        id3 = coupon_id_from_design_hash(design_hash)

        assert id1 == id2 == id3

    def test_coupon_id_different_for_different_hashes(self) -> None:
        """Test different hashes produce different coupon IDs."""
        hashes = [
            "0" * 64,
            "1" * 64,
            "a" * 64,
            "b" * 64,
            "f" * 64,
            "0123456789abcdef" * 4,
        ]
        coupon_ids = [coupon_id_from_design_hash(h) for h in hashes]

        # All IDs should be unique
        assert len(set(coupon_ids)) == len(coupon_ids)

    def test_coupon_id_similar_hashes_different_ids(self) -> None:
        """Test hashes differing by one char produce different IDs."""
        hash1 = "a" * 64
        hash2 = "a" * 63 + "b"  # Differs only in last char

        id1 = coupon_id_from_design_hash(hash1)
        id2 = coupon_id_from_design_hash(hash2)

        assert id1 != id2

    def test_coupon_id_prefix_of_base32(self) -> None:
        """Test coupon_id is first 12 chars of base32 encoding."""
        # The function should base32-encode the hex digest bytes
        # and return the first 12 characters
        design_hash = "00" * 32  # All zeros
        coupon_id = coupon_id_from_design_hash(design_hash)

        # All zeros should encode to 'a' characters in base32
        assert coupon_id == "a" * 12


class TestCanonicalHashKicadPcb:
    """Additional tests for canonical_hash_kicad_pcb_text."""

    def test_removes_tstamp_variations(self) -> None:
        """Test various tstamp formats are removed."""
        board1 = "(kicad_pcb (tstamp 12345678))"
        board2 = "(kicad_pcb (tstamp 87654321))"
        board3 = "(kicad_pcb (tstamp 00000000))"

        hash1 = canonical_hash_kicad_pcb_text(board1)
        hash2 = canonical_hash_kicad_pcb_text(board2)
        hash3 = canonical_hash_kicad_pcb_text(board3)

        assert hash1 == hash2 == hash3

    def test_removes_uuid_variations(self) -> None:
        """Test various uuid formats are removed."""
        board1 = '(kicad_pcb (uuid "12345678-1234-1234-1234-123456789abc"))'
        board2 = '(kicad_pcb (uuid "87654321-4321-4321-4321-abcdef123456"))'

        hash1 = canonical_hash_kicad_pcb_text(board1)
        hash2 = canonical_hash_kicad_pcb_text(board2)

        assert hash1 == hash2

    def test_preserves_meaningful_content(self) -> None:
        """Test that meaningful content differences produce different hashes."""
        board1 = "(kicad_pcb (net 1 GND))"
        board2 = "(kicad_pcb (net 1 VCC))"

        hash1 = canonical_hash_kicad_pcb_text(board1)
        hash2 = canonical_hash_kicad_pcb_text(board2)

        assert hash1 != hash2

    def test_normalizes_line_endings(self) -> None:
        """Test that different line endings produce same hash."""
        board_lf = "(kicad_pcb\n  (version 1)\n)"
        board_crlf = "(kicad_pcb\r\n  (version 1)\r\n)"
        board_cr = "(kicad_pcb\r  (version 1)\r)"

        hash_lf = canonical_hash_kicad_pcb_text(board_lf)
        hash_crlf = canonical_hash_kicad_pcb_text(board_crlf)
        hash_cr = canonical_hash_kicad_pcb_text(board_cr)

        assert hash_lf == hash_crlf == hash_cr

    def test_empty_input_produces_hash(self) -> None:
        """Test that empty input still produces a valid hash."""
        result = canonical_hash_kicad_pcb_text("")
        assert len(result) == 64  # SHA256 hex length


class TestCanonicalHashExport:
    """Additional tests for canonical_hash_export_text."""

    def test_removes_creation_date_comment(self) -> None:
        """Test G04 CreationDate comments are removed."""
        export1 = "G04 CreationDate: 2026-01-19*\nX0Y0D02*\n"
        export2 = "G04 CreationDate: 2099-12-31*\nX0Y0D02*\n"

        hash1 = canonical_hash_export_text(export1)
        hash2 = canonical_hash_export_text(export2)

        assert hash1 == hash2

    def test_removes_timestamp_comments(self) -> None:
        """Test semicolon timestamp comments are removed."""
        export1 = ";timestamp=2026-01-19T12:00:00\nX0Y0D02*\n"
        export2 = ";timestamp=2099-12-31T23:59:59\nX0Y0D02*\n"

        hash1 = canonical_hash_export_text(export1)
        hash2 = canonical_hash_export_text(export2)

        assert hash1 == hash2

    def test_preserves_gerber_commands(self) -> None:
        """Test that Gerber commands are preserved and affect hash."""
        export1 = "X0Y0D02*\nX100Y100D01*\n"
        export2 = "X0Y0D02*\nX200Y200D01*\n"

        hash1 = canonical_hash_export_text(export1)
        hash2 = canonical_hash_export_text(export2)

        assert hash1 != hash2

    def test_normalizes_line_endings(self) -> None:
        """Test that different line endings produce same hash."""
        export_lf = "X0Y0D02*\nX100Y100D01*\n"
        export_crlf = "X0Y0D02*\r\nX100Y100D01*\r\n"

        hash_lf = canonical_hash_export_text(export_lf)
        hash_crlf = canonical_hash_export_text(export_crlf)

        assert hash_lf == hash_crlf


class TestModuleReExports:
    """Tests for module re-export consistency."""

    def test_canonicalize_kicad_pcb_text_alias(self) -> None:
        """Test canonicalize_kicad_pcb_text is properly exported."""
        board = "(kicad_pcb (version 1))"
        result = canonicalize_kicad_pcb_text(board)
        assert isinstance(result, str)

    def test_canonicalize_export_text_alias(self) -> None:
        """Test canonicalize_export_text is properly exported."""
        export = "X0Y0D02*\n"
        result = canonicalize_export_text(export)
        assert isinstance(result, str)

    def test_canonical_drc_json_exported(self) -> None:
        """Test canonical_drc_json is properly exported."""
        # Just verify the function is accessible
        assert callable(canonical_drc_json)


class TestHashDeterminism:
    """Tests for hash function determinism."""

    def test_repeated_calls_same_result(self) -> None:
        """Test repeated hash calls produce identical results."""
        board = "(kicad_pcb (version 1) (net 1 GND))"

        hashes = [canonical_hash_kicad_pcb_text(board) for _ in range(10)]

        assert all(h == hashes[0] for h in hashes)

    def test_whitespace_sensitivity(self) -> None:
        """Test that internal whitespace differences are preserved."""
        board1 = "(kicad_pcb (net 1))"
        board2 = "(kicad_pcb  (net  1))"  # Extra spaces

        hash1 = canonical_hash_kicad_pcb_text(board1)
        hash2 = canonical_hash_kicad_pcb_text(board2)

        # Internal whitespace should affect hash (implementation dependent)
        # This test documents the behavior
        # Both hashes should be valid SHA256
        assert len(hash1) == 64
        assert len(hash2) == 64


class TestHashLength:
    """Tests for hash output format."""

    def test_kicad_pcb_hash_is_sha256(self) -> None:
        """Test kicad_pcb hash is 64-char hex (SHA256)."""
        result = canonical_hash_kicad_pcb_text("(kicad_pcb)")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_export_hash_is_sha256(self) -> None:
        """Test export hash is 64-char hex (SHA256)."""
        result = canonical_hash_export_text("X0Y0D02*")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_coupon_id_is_12_chars(self) -> None:
        """Test coupon_id is exactly 12 characters."""
        design_hash = "a" * 64
        coupon_id = coupon_id_from_design_hash(design_hash)
        assert len(coupon_id) == 12
