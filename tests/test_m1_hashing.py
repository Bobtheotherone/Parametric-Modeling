from __future__ import annotations

from formula_foundry.coupongen.hashing import (
    canonical_hash_export_text,
    canonical_hash_kicad_pcb_text,
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
