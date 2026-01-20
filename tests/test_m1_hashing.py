from __future__ import annotations

from formula_foundry.coupongen.hashing import (
    canonical_hash_export_text,
    canonical_hash_kicad_pcb_text,
    coupon_id_from_design_hash,
)


def test_canonical_hashing_removes_nondeterminism() -> None:
    board_a = "(kicad_pcb\n  (tstamp 12345678)\n  (uuid 11111111)\n  (net 1)\n)"
    board_b = "(kicad_pcb\r\n  (tstamp 87654321)\r\n  (uuid 22222222)\r\n  (net 1)\r\n)"
    board_c = "(kicad_pcb\n  (tstamp 12345678)\n  (uuid 11111111)\n  (net 2)\n)"

    assert canonical_hash_kicad_pcb_text(board_a) == canonical_hash_kicad_pcb_text(board_b)
    assert canonical_hash_kicad_pcb_text(board_a) != canonical_hash_kicad_pcb_text(board_c)

    export_a = "G04 CreationDate: 2026-01-19*\nX0Y0D02*\n;timestamp=2026-01-19\n"
    export_b = "G04 CreationDate: 2026-02-01*\r\nX0Y0D02*\r\n;timestamp=2026-02-01\r\n"
    export_c = "X1Y1D02*\n"

    assert canonical_hash_export_text(export_a) == canonical_hash_export_text(export_b)
    assert canonical_hash_export_text(export_a) != canonical_hash_export_text(export_c)


def test_coupon_id_from_design_hash_produces_12_char_id() -> None:
    """coupon_id_from_design_hash must produce a 12-character base32-encoded ID."""
    # A known SHA256 hex digest (64 hex chars = 32 bytes)
    design_hash = "a" * 64  # 0xaa... repeated
    coupon_id = coupon_id_from_design_hash(design_hash)

    assert len(coupon_id) == 12
    assert coupon_id.islower()
    # Base32 uses only a-z and 2-7
    assert all(c in "abcdefghijklmnopqrstuvwxyz234567" for c in coupon_id)


def test_coupon_id_from_design_hash_is_deterministic() -> None:
    """coupon_id_from_design_hash must be deterministic for the same input."""
    design_hash = "deadbeef" * 8  # 64 hex chars
    id_a = coupon_id_from_design_hash(design_hash)
    id_b = coupon_id_from_design_hash(design_hash)

    assert id_a == id_b


def test_coupon_id_from_design_hash_differs_for_different_hashes() -> None:
    """Different design hashes must produce different coupon IDs."""
    hash_a = "a" * 64
    hash_b = "b" * 64

    id_a = coupon_id_from_design_hash(hash_a)
    id_b = coupon_id_from_design_hash(hash_b)

    assert id_a != id_b
