from __future__ import annotations

import base64
import re

from formula_foundry.substrate import sha256_bytes

_KICAD_UUID_RE = re.compile(r"\((tstamp|uuid)\s+[^)]+\)")

_EXPORT_COMMENT_PREFIXES = ("G04", ";")


def canonicalize_kicad_pcb_text(text: str) -> str:
    normalized = _normalize_line_endings(text)
    return _KICAD_UUID_RE.sub(r"(\1)", normalized)


def canonicalize_export_text(text: str) -> str:
    normalized = _normalize_line_endings(text)
    lines: list[str] = []
    for line in normalized.split("\n"):
        stripped = line.rstrip()
        if stripped.startswith(_EXPORT_COMMENT_PREFIXES):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def canonical_hash_kicad_pcb_text(text: str) -> str:
    canonical = canonicalize_kicad_pcb_text(text)
    return sha256_bytes(canonical.encode("utf-8"))


def canonical_hash_export_text(text: str) -> str:
    canonical = canonicalize_export_text(text)
    return sha256_bytes(canonical.encode("utf-8"))


def coupon_id_from_design_hash(design_hash: str) -> str:
    digest = bytes.fromhex(design_hash)
    encoded = base64.b32encode(digest).decode("ascii").lower().rstrip("=")
    return encoded[:12]


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")
