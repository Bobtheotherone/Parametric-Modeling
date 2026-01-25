"""
Atomic I/O helpers for orchestrator robustness.

This module is intentionally stdlib-only and safe for use in CI / headless runs.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(
    path: str | Path,
    data: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
) -> Path:
    """
    Atomically write JSON to `path` by writing to a temp file in the same directory
    and then `os.replace()` into place.

    Returns the resolved Path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(
        data,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
    )
    if not payload.endswith("\n"):
        payload += "\n"

    # Temp file in same dir => atomic replace on same filesystem
    fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(p))
    finally:
        # If something went wrong before replace, remove temp file
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return p.resolve()


def validate_json_file(path: str | Path) -> bool:
    """
    Return True iff `path` exists and contains valid JSON.
    Does not enforce a schema (that’s done elsewhere in the repo).
    """
    p = Path(path)
    if not p.exists():
        return False
    try:
        _ = json.loads(p.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False
