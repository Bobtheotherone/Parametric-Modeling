#!/usr/bin/env python3
"""Atomic I/O utilities for robust file operations.

This module provides atomic write operations that ensure files are never
left in a corrupted or empty state, even if the process is interrupted.

Key guarantees:
- Files are written completely or not at all
- 0-byte files are never produced on interrupted writes
- JSON files are validated before being considered written
- All writes use fsync to ensure durability
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any


class AtomicWriteError(Exception):
    """Raised when atomic write fails."""

    pass


class JSONValidationError(Exception):
    """Raised when JSON validation fails."""

    pass


def atomic_write_text(
    path: Path | str,
    content: str,
    encoding: str = "utf-8",
) -> None:
    """Write text to a file atomically.

    Uses write-to-temp + fsync + rename pattern to ensure:
    - The file is never left empty or partial
    - Interrupted writes don't corrupt the file
    - The operation is atomic on POSIX systems

    Args:
        path: Target file path
        content: Text content to write
        encoding: Text encoding (default: utf-8)

    Raises:
        AtomicWriteError: If write fails
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file with fsync
        with open(tmp_path, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        tmp_path.rename(path)

    except Exception as e:
        # Clean up temp file if it exists
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise AtomicWriteError(f"Failed to write {path}: {e}") from e


def atomic_write_json(
    path: Path | str,
    data: dict[str, Any] | list[Any],
    indent: int = 2,
    validate_schema: Callable[[dict], bool] | None = None,
) -> None:
    """Write JSON data to a file atomically with optional validation.

    Args:
        path: Target file path
        data: JSON-serializable data
        indent: JSON indentation level
        validate_schema: Optional validation function that returns True if valid

    Raises:
        AtomicWriteError: If write fails
        JSONValidationError: If validation function returns False
    """
    path = Path(path)

    # Serialize to string first to catch serialization errors early
    try:
        content = json.dumps(data, indent=indent, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        raise JSONValidationError(f"Failed to serialize JSON: {e}") from e

    # Validate content if validator provided
    if validate_schema is not None:
        try:
            parsed = json.loads(content)
            if not validate_schema(parsed):
                raise JSONValidationError("Schema validation failed")
        except json.JSONDecodeError as e:
            raise JSONValidationError(f"Invalid JSON after serialization: {e}") from e

    # Write atomically
    atomic_write_text(path, content)


def atomic_copy_file(
    src: Path | str,
    dst: Path | str,
) -> None:
    """Copy a file atomically.

    Args:
        src: Source file path
        dst: Destination file path

    Raises:
        AtomicWriteError: If copy fails
    """
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise AtomicWriteError(f"Source file does not exist: {src}")

    content = src.read_bytes()
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)

        with open(tmp_path, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        tmp_path.rename(dst)

    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise AtomicWriteError(f"Failed to copy {src} to {dst}: {e}") from e


def validate_json_file(
    path: Path | str,
    validator: Callable[[dict], tuple[bool, str | None]] | None = None,
) -> tuple[bool, dict | None, str | None]:
    """Validate a JSON file exists, is non-empty, and parses correctly.

    Args:
        path: Path to JSON file
        validator: Optional function that returns (is_valid, error_message)

    Returns:
        Tuple of (is_valid, parsed_data, error_message)
    """
    path = Path(path)

    # Check file exists
    if not path.exists():
        return False, None, f"File does not exist: {path}"

    # Check file is not empty
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return False, None, f"Failed to read file: {e}"

    if not content.strip():
        return False, None, "File is empty (0 bytes)"

    # Parse JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {e}"

    # Run custom validator if provided
    if validator is not None:
        try:
            is_valid, error_msg = validator(data)
            if not is_valid:
                return False, data, error_msg
        except Exception as e:
            return False, data, f"Validation error: {e}"

    return True, data, None


def safe_read_json(
    path: Path | str,
    default: dict | list | None = None,
) -> tuple[dict | list | None, str | None]:
    """Safely read a JSON file, returning default if file is missing/invalid.

    Args:
        path: Path to JSON file
        default: Default value if file cannot be read

    Returns:
        Tuple of (data, error_message or None)
    """
    is_valid, data, error = validate_json_file(path)
    if is_valid:
        return data, None
    else:
        return default, error


def recover_or_create_json(
    path: Path | str,
    creator: Callable[[], dict],
    validator: Callable[[dict], tuple[bool, str | None]] | None = None,
    max_attempts: int = 2,
) -> tuple[bool, dict | None, str | None]:
    """Try to read a JSON file, create it if missing/invalid.

    This function implements a recovery pattern:
    1. Try to read and validate the existing file
    2. If invalid, call the creator function to regenerate
    3. Write the new content atomically
    4. Validate the result

    Args:
        path: Path to JSON file
        creator: Function to create new content if needed
        validator: Optional validation function
        max_attempts: Maximum creation attempts

    Returns:
        Tuple of (success, data, error_message)
    """
    path = Path(path)

    for attempt in range(max_attempts):
        # Try to read existing file
        is_valid, data, error = validate_json_file(path, validator)
        if is_valid:
            return True, data, None

        # File is missing or invalid, try to create
        try:
            new_data = creator()
            atomic_write_json(path, new_data)
        except Exception as e:
            if attempt == max_attempts - 1:
                return False, None, f"Failed to create file after {max_attempts} attempts: {e}"
            continue

    # Final validation
    return validate_json_file(path, validator)


class AtomicJSONWriter:
    """Context manager for atomic JSON file writing with rollback support."""

    def __init__(self, path: Path | str, indent: int = 2):
        self.path = Path(path)
        self.indent = indent
        self.data: dict | list | None = None
        self._backup_path: Path | None = None

    def __enter__(self) -> AtomicJSONWriter:
        # Create backup if file exists
        if self.path.exists():
            self._backup_path = self.path.with_suffix(self.path.suffix + ".backup")
            try:
                atomic_copy_file(self.path, self._backup_path)
            except Exception:
                self._backup_path = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Exception occurred, restore backup if available
            if self._backup_path and self._backup_path.exists():
                try:
                    self._backup_path.rename(self.path)
                except Exception:
                    pass
        else:
            # Success, remove backup
            if self._backup_path and self._backup_path.exists():
                try:
                    self._backup_path.unlink()
                except Exception:
                    pass
        return False

    def write(self, data: dict | list) -> None:
        """Write data to file atomically."""
        self.data = data
        atomic_write_json(self.path, data, indent=self.indent)
