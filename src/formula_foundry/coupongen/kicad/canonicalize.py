"""Canonical hash computation for KiCad artifacts.

This module provides functions to canonicalize Gerber, drill (Excellon),
KiCad PCB files, and DRC JSON reports by removing nondeterministic noise
(timestamps, comments, UUIDs, absolute paths) before hashing. This ensures
stable, reproducible hashes across multiple exports of the same design.

Canonicalization Algorithms (per Design Doc Section 13.5.2):

Board canonicalization (canonicalize_board):
    - Remove: (tstamp ...), (uuid ...)
    - Normalize: whitespace (collapse multiple spaces, strip trailing)
    - Ensure: newline \\n endings (CRLF -> LF)

Gerber canonicalization (canonicalize_gerber):
    - Strip: comment lines starting with G04
    - Strip: timestamp attributes (e.g., TF.CreationDate)
    - Normalize: CRLF -> LF, trim leading/trailing whitespace, drop empty lines

DRC JSON canonicalization (canonicalize_drc_json):
    - Remove/normalize: timestamps, absolute paths, tool invocation environment
    - Sort: object keys alphabetically
    - Stable: sort list entries for order-insensitive keys (violations, unconnected_items)

Satisfies REQ-M1-005.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from formula_foundry.substrate import sha256_bytes

# KiCad PCB S-expression UUID/timestamp pattern
_KICAD_UUID_RE = re.compile(r"\((tstamp|uuid)\s+[^)]+\)")

# Pattern to normalize multiple whitespaces (excluding newlines)
_MULTI_SPACE_RE = re.compile(r"[ \t]+")

# Gerber RS-274X comment prefixes (G04 command is a comment)
_GERBER_COMMENT_PREFIX = "G04"

# Excellon drill file comment prefix
_EXCELLON_COMMENT_PREFIX = ";"

# Gerber attributes that include timestamps (strip for deterministic hashing)
_GERBER_TIMESTAMP_RE = re.compile(r"creation\s*date|creation\s*time", re.IGNORECASE)

# DRC JSON keys that contain nondeterministic values (to be removed or normalized)
_DRC_NONDETERMINISTIC_KEYS = frozenset({
    "date",
    "time",
    "timestamp",
    "generated_at",
    "kicad_version",
    "host",
    "source",
    "schema_version",
})

# Keys that contain paths (to be normalized by removing directory prefixes)
_DRC_PATH_KEYS = frozenset({
    "path",
    "file",
    "filename",
    "source_file",
    "board_file",
})

# DRC list keys whose ordering is not semantically meaningful
_DRC_SORTED_LIST_KEYS = frozenset({
    "violations",
    "unconnected_items",
    "schematic_parity",
})

# Absolute path detection (POSIX, Windows drive, UNC)
_ABS_PATH_RE = re.compile(r"^(?:/|[a-zA-Z]:\\\\|\\\\\\\\)")

# Sentinel value to indicate a key should be removed (distinct from None)
_REMOVE_KEY = object()


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
    - TF.CreationDate attribute lines (timestamp metadata)
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
        trimmed = stripped.lstrip()
        if not trimmed:
            continue
        # Skip G04 comment lines (they contain timestamps/tool info)
        if trimmed.startswith(_GERBER_COMMENT_PREFIX):
            continue
        # Strip timestamp attributes like TF.CreationDate
        if _GERBER_TIMESTAMP_RE.search(trimmed):
            continue
        lines.append(trimmed)
    result = "\n".join(lines)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


def canonicalize_drill(text: str) -> str:
    """Canonicalize Excellon drill file content.

    Removes nondeterministic elements:
    - Semicolon comment lines (contain timestamps, tool info)
    - Trailing whitespace on each line
    - CRLF/CR line endings (normalized to LF)
    - Empty lines (normalization)

    Args:
        text: Raw Excellon drill file content.

    Returns:
        Canonicalized drill content suitable for hashing.
    """
    normalized = normalize_line_endings(text)
    lines: list[str] = []
    for line in normalized.split("\n"):
        stripped = line.rstrip()
        trimmed = stripped.lstrip()
        # Skip empty or semicolon comment lines
        if not trimmed or trimmed.startswith(_EXCELLON_COMMENT_PREFIX):
            continue
        lines.append(trimmed)
    result = "\n".join(lines)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


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


def canonicalize_board(text: str) -> str:
    """Canonicalize KiCad board (.kicad_pcb) file content per Section 13.5.2.

    This is the authoritative board canonicalization function implementing
    the exact algorithms specified in the design document:

    1. Remove nondeterministic S-expression fields:
       - (tstamp ...) - timestamp/UUID fields
       - (uuid ...) - UUID fields

    2. Normalize whitespace:
       - Collapse multiple consecutive spaces/tabs to single space
       - Strip trailing whitespace on each line
       - Preserve structure-significant newlines

    3. Normalize line endings:
       - CRLF -> LF
       - CR -> LF
       - Ensure file ends with newline

    Note: The current implementation is conservative and avoids reordering
    S-expression subtrees, as the board writer is expected to produce
    deterministic ordering. Future versions may add stable reordering if
    writer determinism is not guaranteed.

    Args:
        text: Raw KiCad board file content (.kicad_pcb).

    Returns:
        Canonicalized board content suitable for deterministic hashing.

    Example:
        >>> board = "(kicad_pcb  (tstamp abc-123)  (net 1))"
        >>> canonicalize_board(board)
        '(kicad_pcb (tstamp) (net 1))\\n'
    """
    # Step 1: Normalize line endings (CRLF/CR -> LF)
    normalized = normalize_line_endings(text)

    # Step 2: Remove tstamp and uuid field values
    # Transforms "(tstamp abc123)" -> "(tstamp)" and "(uuid xyz789)" -> "(uuid)"
    no_uuids = _KICAD_UUID_RE.sub(r"(\1)", normalized)

    # Step 3: Normalize whitespace per line
    lines: list[str] = []
    for line in no_uuids.split("\n"):
        # Collapse multiple spaces/tabs to single space
        collapsed = _MULTI_SPACE_RE.sub(" ", line)
        # Strip trailing whitespace
        stripped = collapsed.rstrip()
        lines.append(stripped)

    # Step 4: Join and ensure trailing newline
    result = "\n".join(lines)
    if result and not result.endswith("\n"):
        result += "\n"

    return result


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
        trimmed = stripped.lstrip()
        if not trimmed:
            continue
        # Skip both Gerber G04 comments and Excellon semicolon comments
        if trimmed.startswith(_GERBER_COMMENT_PREFIX) or trimmed.startswith(_EXCELLON_COMMENT_PREFIX):
            continue
        # Strip timestamp attributes like TF.CreationDate (Gerber)
        if _GERBER_TIMESTAMP_RE.search(trimmed):
            continue
        lines.append(trimmed)
    result = "\n".join(lines)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


def _normalize_drc_value(key: str, value: Any) -> Any:
    """Normalize a single DRC JSON value based on its key.

    Args:
        key: The key name in the DRC JSON structure.
        value: The value to normalize.

    Returns:
        Normalized value, or _REMOVE_KEY sentinel if the key should be removed.
        Note: Returns actual None for null JSON values (not the same as removal).
    """
    key_lower = key.lower()

    # Remove entirely nondeterministic keys
    if key_lower in _DRC_NONDETERMINISTIC_KEYS:
        return _REMOVE_KEY

    # Normalize path values to just filename (remove directory prefixes)
    if isinstance(value, str):
        key_is_path = key_lower in _DRC_PATH_KEYS or key_lower.endswith("_path") or key_lower.endswith("_file")
        key_is_path = key_is_path or key_lower.startswith("path_") or key_lower.startswith("file_")
        if key_is_path or _ABS_PATH_RE.match(value):
            if "/" in value:
                return value.rsplit("/", 1)[-1]
            if "\\" in value:
                return value.rsplit("\\", 1)[-1]
            return value

    return value


def _canonicalize_drc_object(obj: Any, *, parent_key: str | None = None) -> Any:
    """Recursively canonicalize a DRC JSON object.

    Args:
        obj: JSON-like object (dict, list, or primitive).

    Returns:
        Canonicalized object with sorted keys and removed nondeterministic fields.
    """
    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        # Sort keys alphabetically for deterministic output
        for key in sorted(obj.keys()):
            normalized = _normalize_drc_value(key, obj[key])
            if normalized is not _REMOVE_KEY:
                # Recursively canonicalize nested structures
                result[key] = _canonicalize_drc_object(normalized, parent_key=key)
        return result
    elif isinstance(obj, list):
        # Recursively canonicalize list items
        canonical_items = [_canonicalize_drc_object(item, parent_key=parent_key) for item in obj]
        if parent_key in _DRC_SORTED_LIST_KEYS:
            return sorted(
                canonical_items,
                key=lambda item: json.dumps(item, separators=(",", ":"), sort_keys=True),
            )
        return canonical_items
    else:
        # Primitives (including None) pass through unchanged
        return obj


def canonicalize_drc_json(data: str | dict[str, Any]) -> str:
    """Canonicalize KiCad DRC JSON report per Section 13.5.2.

    This function removes nondeterministic elements from DRC JSON reports
    to enable stable, reproducible hashing:

    1. Remove nondeterministic keys:
       - date, time, timestamp, generated_at
       - kicad_version, host, source, schema_version
       - Any key containing environment-specific information

    2. Normalize path values:
       - Remove directory prefixes from path/file keys
       - Keep only the filename portion

    3. Sort all object keys alphabetically:
       - Ensures deterministic JSON output
       - Nested objects are also sorted

    4. Stabilize list ordering where semantics allow:
       - Violations, unconnected items, and schematic parity lists are sorted
         deterministically after recursive canonicalization.
       - Other lists preserve ordering.

    The output is compact JSON (no indentation) with sorted keys.

    Args:
        data: Either a JSON string or a parsed dict representing DRC output.

    Returns:
        Canonicalized JSON string suitable for deterministic hashing.

    Example:
        >>> drc = '{"date": "2026-01-20", "violations": [], "file": "/tmp/board.kicad_pcb"}'
        >>> canonicalize_drc_json(drc)
        '{"file":"board.kicad_pcb","violations":[]}'
    """
    # Parse if string input
    if isinstance(data, str):
        parsed = json.loads(data)
    else:
        parsed = data

    # Canonicalize the structure
    canonical = _canonicalize_drc_object(parsed)

    # Output as compact JSON with sorted keys (already sorted by canonicalization)
    return json.dumps(canonical, separators=(",", ":"), sort_keys=True)


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


def canonical_hash_board(text: str) -> str:
    """Compute canonical SHA-256 hash of KiCad board file content.

    Uses the full board canonicalization algorithm (Section 13.5.2):
    - Strips tstamp/uuid fields
    - Normalizes whitespace
    - Ensures consistent line endings

    Args:
        text: Raw KiCad board file content (.kicad_pcb).

    Returns:
        Lowercase hex digest of the canonical content hash.
    """
    canonical = canonicalize_board(text)
    return sha256_bytes(canonical.encode("utf-8"))


def canonical_hash_drc_json(data: str | dict[str, Any]) -> str:
    """Compute canonical SHA-256 hash of DRC JSON report.

    Uses the DRC JSON canonicalization algorithm (Section 13.5.2):
    - Removes timestamps, paths, and other nondeterministic keys
    - Sorts all object keys
    - Outputs compact JSON

    Args:
        data: Either a JSON string or parsed dict of DRC report.

    Returns:
        Lowercase hex digest of the canonical content hash.
    """
    canonical = canonicalize_drc_json(data)
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
