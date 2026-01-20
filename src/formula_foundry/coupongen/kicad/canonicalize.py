"""Canonical hash computation for KiCad artifacts.

This module provides functions to canonicalize Gerber, drill (Excellon), and
KiCad PCB files by removing nondeterministic noise (timestamps, comments,
UUIDs) before hashing. This ensures stable, reproducible hashes across
multiple exports of the same design.

Satisfies REQ-M1-005.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from formula_foundry.substrate import sha256_bytes

# KiCad PCB S-expression UUID/timestamp pattern
_KICAD_UUID_RE = re.compile(r"\((tstamp|uuid)\s+[^)]+\)")

# Gerber RS-274X comment prefixes (G04 command is a comment)
_GERBER_COMMENT_PREFIX = "G04"

# Excellon drill file comment prefix
_EXCELLON_COMMENT_PREFIX = ";"


def normalize_line_endings(text: str) -> str:
    """Normalize all line endings to Unix-style (LF).

    Args:
        text: Input text with potentially mixed line endings.

    Returns:
        Text with all CRLF and CR converted to LF.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def canonicalize_gerber(text: str) -> str:
    """Canonicalize Gerber RS-274X file content.

    Removes nondeterministic elements:
    - G04 comment lines (contain timestamps, tool info, etc.)
    - Trailing whitespace on each line
    - CRLF/CR line endings (normalized to LF)

    Args:
        text: Raw Gerber file content.

    Returns:
        Canonicalized Gerber content suitable for hashing.
    """
    normalized = normalize_line_endings(text)
    lines: list[str] = []
    for line in normalized.split("\n"):
        stripped = line.rstrip()
        # Skip G04 comment lines (they contain timestamps/tool info)
        if stripped.startswith(_GERBER_COMMENT_PREFIX):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def canonicalize_drill(text: str) -> str:
    """Canonicalize Excellon drill file content.

    Removes nondeterministic elements:
    - Semicolon comment lines (contain timestamps, tool info)
    - Trailing whitespace on each line
    - CRLF/CR line endings (normalized to LF)

    Args:
        text: Raw Excellon drill file content.

    Returns:
        Canonicalized drill content suitable for hashing.
    """
    normalized = normalize_line_endings(text)
    lines: list[str] = []
    for line in normalized.split("\n"):
        stripped = line.rstrip()
        # Skip semicolon comment lines
        if stripped.startswith(_EXCELLON_COMMENT_PREFIX):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def canonicalize_kicad_pcb(text: str) -> str:
    """Canonicalize KiCad PCB S-expression file content.

    Removes nondeterministic elements:
    - tstamp fields (timestamps/UUIDs)
    - uuid fields
    - CRLF/CR line endings (normalized to LF)

    The tstamp and uuid values are stripped but the field markers remain,
    e.g., "(tstamp abc123)" becomes "(tstamp)".

    Args:
        text: Raw KiCad PCB file content.

    Returns:
        Canonicalized PCB content suitable for hashing.
    """
    normalized = normalize_line_endings(text)
    return _KICAD_UUID_RE.sub(r"(\1)", normalized)


def canonicalize_export(text: str) -> str:
    """Canonicalize generic KiCad export file content (Gerber or drill).

    This is a convenience function that handles both Gerber (G04) and
    Excellon (;) comment formats.

    Args:
        text: Raw export file content.

    Returns:
        Canonicalized content suitable for hashing.
    """
    normalized = normalize_line_endings(text)
    lines: list[str] = []
    for line in normalized.split("\n"):
        stripped = line.rstrip()
        # Skip both Gerber G04 comments and Excellon semicolon comments
        if stripped.startswith(_GERBER_COMMENT_PREFIX) or stripped.startswith(_EXCELLON_COMMENT_PREFIX):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def canonical_hash_gerber(text: str) -> str:
    """Compute canonical SHA-256 hash of Gerber file content.

    Args:
        text: Raw Gerber file content.

    Returns:
        Lowercase hex digest of the canonical content hash.
    """
    canonical = canonicalize_gerber(text)
    return sha256_bytes(canonical.encode("utf-8"))


def canonical_hash_drill(text: str) -> str:
    """Compute canonical SHA-256 hash of Excellon drill file content.

    Args:
        text: Raw Excellon drill file content.

    Returns:
        Lowercase hex digest of the canonical content hash.
    """
    canonical = canonicalize_drill(text)
    return sha256_bytes(canonical.encode("utf-8"))


def canonical_hash_kicad_pcb(text: str) -> str:
    """Compute canonical SHA-256 hash of KiCad PCB file content.

    Args:
        text: Raw KiCad PCB file content.

    Returns:
        Lowercase hex digest of the canonical content hash.
    """
    canonical = canonicalize_kicad_pcb(text)
    return sha256_bytes(canonical.encode("utf-8"))


def canonical_hash_export(text: str) -> str:
    """Compute canonical SHA-256 hash of generic export file content.

    Args:
        text: Raw export file content (Gerber or drill).

    Returns:
        Lowercase hex digest of the canonical content hash.
    """
    canonical = canonicalize_export(text)
    return sha256_bytes(canonical.encode("utf-8"))


def canonical_hash_file(path: Path) -> str:
    """Compute canonical hash for a file based on its extension.

    Automatically selects the appropriate canonicalization:
    - .kicad_pcb -> KiCad PCB canonicalization
    - .drl, .xln -> Excellon drill canonicalization
    - All others (Gerber layers) -> Gerber canonicalization

    Args:
        path: Path to the file.

    Returns:
        Lowercase hex digest of the canonical content hash.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()
    if suffix == ".kicad_pcb":
        return canonical_hash_kicad_pcb(text)
    if suffix in (".drl", ".xln"):
        return canonical_hash_drill(text)
    # Default to Gerber canonicalization for layer files
    return canonical_hash_gerber(text)


def canonical_hash_files(paths: Iterable[Path]) -> dict[str, str]:
    """Compute canonical hashes for multiple files.

    Args:
        paths: Iterable of file paths.

    Returns:
        Dictionary mapping file names to their canonical hashes.
    """
    return {path.name: canonical_hash_file(path) for path in paths}
