"""Hashing utilities for coupon generation.

This module provides hashing functions that delegate to the authoritative
canonicalization implementations in kicad/canonicalize.py per Section 13.5.2.

For direct access to canonicalization algorithms, use:
    from formula_foundry.coupongen.kicad.canonicalize import (
        canonicalize_board,
        canonicalize_gerber,
        canonicalize_drc_json,
        # etc.
    )

This module maintains backward compatibility with existing imports while
ensuring all canonicalization uses the single authoritative source.
"""

from __future__ import annotations

import base64

from formula_foundry.substrate import sha256_bytes

# Import authoritative canonicalization functions from kicad/canonicalize.py
from .kicad.canonicalize import (
    canonicalize_export,
    canonicalize_kicad_pcb,
    canonical_hash_drc_json,
    canonical_hash_export,
    canonical_hash_kicad_pcb,
)

# Re-export for backward compatibility with existing module-level names
canonicalize_kicad_pcb_text = canonicalize_kicad_pcb
canonicalize_export_text = canonicalize_export
canonical_hash_kicad_pcb_text = canonical_hash_kicad_pcb
canonical_hash_export_text = canonical_hash_export

# Also export the DRC JSON functions for completeness
canonical_drc_json = canonical_hash_drc_json


def coupon_id_from_design_hash(design_hash: str) -> str:
    """Derive a human-readable coupon ID from a design hash.

    Args:
        design_hash: SHA256 hex digest of the resolved design.

    Returns:
        12-character lowercase base32-encoded identifier.
    """
    digest = bytes.fromhex(design_hash)
    encoded = base64.b32encode(digest).decode("ascii").lower().rstrip("=")
    return encoded[:12]
