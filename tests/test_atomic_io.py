# SPDX-License-Identifier: MIT
"""Unit tests for bridge/atomic_io.py.

Tests the atomic I/O utilities for robust file operations. Key functionality:
- Atomic text file writing (write-to-temp + fsync + rename)
- Atomic JSON writing with optional schema validation
- Atomic file copying
- JSON file validation
- Safe JSON reading with defaults
- Recovery/create patterns for JSON files
- AtomicJSONWriter context manager with rollback support
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.atomic_io import (
    AtomicJSONWriter,
    AtomicWriteError,
    JSONValidationError,
    atomic_copy_file,
    atomic_write_json,
    atomic_write_text,
    recover_or_create_json,
    safe_read_json,
    validate_json_file,
)


# -----------------------------------------------------------------------------
# atomic_write_text tests
# -----------------------------------------------------------------------------


class TestAtomicWriteText:
    """Tests for atomic_write_text function."""

    def test_basic_write(self, tmp_path: Path) -> None:
        """Basic text write succeeds."""
        target = tmp_path / "test.txt"
        atomic_write_text(target, "Hello, World!")
        assert target.exists()
        assert target.read_text() == "Hello, World!"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        target = tmp_path / "deep" / "nested" / "dir" / "test.txt"
        atomic_write_text(target, "Content")
        assert target.exists()
        assert target.read_text() == "Content"

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """Existing file is overwritten."""
        target = tmp_path / "test.txt"
        target.write_text("Old content")
        atomic_write_text(target, "New content")
        assert target.read_text() == "New content"

    def test_empty_string_write(self, tmp_path: Path) -> None:
        """Empty string can be written."""
        target = tmp_path / "empty.txt"
        atomic_write_text(target, "")
        assert target.exists()
        assert target.read_text() == ""

    def test_unicode_content(self, tmp_path: Path) -> None:
        """Unicode content is handled correctly."""
        target = tmp_path / "unicode.txt"
        content = "Hello, 世界! 🌍 Привет!"
        atomic_write_text(target, content)
        assert target.read_text() == content

    def test_multiline_content(self, tmp_path: Path) -> None:
        """Multiline content is preserved."""
        target = tmp_path / "multiline.txt"
        content = "Line 1\nLine 2\nLine 3\n"
        atomic_write_text(target, content)
        assert target.read_text() == content

    def test_no_temp_file_left_on_success(self, tmp_path: Path) -> None:
        """Temp file is removed after successful write."""
        target = tmp_path / "test.txt"
        atomic_write_text(target, "Content")
        temp_path = target.with_suffix(".txt.tmp")
        assert not temp_path.exists()

    def test_path_as_string(self, tmp_path: Path) -> None:
        """Path can be passed as string."""
        target = str(tmp_path / "test.txt")
        atomic_write_text(target, "Content")
        assert Path(target).read_text() == "Content"


# -----------------------------------------------------------------------------
# atomic_write_json tests
# -----------------------------------------------------------------------------


class TestAtomicWriteJson:
    """Tests for atomic_write_json function."""

    def test_basic_dict_write(self, tmp_path: Path) -> None:
        """Basic dict JSON write succeeds."""
        target = tmp_path / "test.json"
        data = {"key": "value", "number": 42}
        atomic_write_json(target, data)
        assert target.exists()
        loaded = json.loads(target.read_text())
        assert loaded == data

    def test_basic_list_write(self, tmp_path: Path) -> None:
        """Basic list JSON write succeeds."""
        target = tmp_path / "test.json"
        data = [1, 2, 3, "four"]
        atomic_write_json(target, data)
        loaded = json.loads(target.read_text())
        assert loaded == data

    def test_nested_structure(self, tmp_path: Path) -> None:
        """Nested JSON structure is preserved."""
        target = tmp_path / "test.json"
        data = {
            "nested": {"deep": {"value": 123}},
            "list": [{"a": 1}, {"b": 2}],
        }
        atomic_write_json(target, data)
        loaded = json.loads(target.read_text())
        assert loaded == data

    def test_custom_indent(self, tmp_path: Path) -> None:
        """Custom indent is respected."""
        target = tmp_path / "test.json"
        data = {"key": "value"}
        atomic_write_json(target, data, indent=4)
        content = target.read_text()
        assert "    " in content  # 4-space indent

    def test_unicode_in_json(self, tmp_path: Path) -> None:
        """Unicode in JSON is preserved."""
        target = tmp_path / "test.json"
        data = {"greeting": "Hello, 世界! 🌍"}
        atomic_write_json(target, data)
        loaded = json.loads(target.read_text())
        assert loaded["greeting"] == "Hello, 世界! 🌍"

    def test_with_passing_validator(self, tmp_path: Path) -> None:
        """Write succeeds with passing validator."""
        target = tmp_path / "test.json"
        data = {"required_field": "present"}

        def validator(obj: dict) -> bool:
            return "required_field" in obj

        atomic_write_json(target, data, validate_schema=validator)
        assert target.exists()

    def test_with_failing_validator(self, tmp_path: Path) -> None:
        """Write fails with failing validator."""
        target = tmp_path / "test.json"
        data = {"other_field": "present"}

        def validator(obj: dict) -> bool:
            return "required_field" in obj

        with pytest.raises(JSONValidationError, match="validation failed"):
            atomic_write_json(target, data, validate_schema=validator)

        # File should not be created
        assert not target.exists()

    def test_non_serializable_raises(self, tmp_path: Path) -> None:
        """Non-serializable data raises JSONValidationError."""
        target = tmp_path / "test.json"

        class NotSerializable:
            pass

        data = {"obj": NotSerializable()}
        with pytest.raises(JSONValidationError, match="serialize"):
            atomic_write_json(target, data)


# -----------------------------------------------------------------------------
# atomic_copy_file tests
# -----------------------------------------------------------------------------


class TestAtomicCopyFile:
    """Tests for atomic_copy_file function."""

    def test_basic_copy(self, tmp_path: Path) -> None:
        """Basic file copy succeeds."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "dest.txt"
        src.write_text("Source content")
        atomic_copy_file(src, dst)
        assert dst.exists()
        assert dst.read_text() == "Source content"

    def test_copy_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Copy creates parent directories for destination."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "deep" / "nested" / "dest.txt"
        src.write_text("Content")
        atomic_copy_file(src, dst)
        assert dst.exists()
        assert dst.read_text() == "Content"

    def test_copy_binary_file(self, tmp_path: Path) -> None:
        """Binary files are copied correctly."""
        src = tmp_path / "source.bin"
        dst = tmp_path / "dest.bin"
        binary_content = bytes(range(256))
        src.write_bytes(binary_content)
        atomic_copy_file(src, dst)
        assert dst.read_bytes() == binary_content

    def test_copy_overwrites_existing(self, tmp_path: Path) -> None:
        """Existing destination file is overwritten."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "dest.txt"
        src.write_text("New content")
        dst.write_text("Old content")
        atomic_copy_file(src, dst)
        assert dst.read_text() == "New content"

    def test_copy_nonexistent_source_raises(self, tmp_path: Path) -> None:
        """Copying non-existent source raises error."""
        src = tmp_path / "nonexistent.txt"
        dst = tmp_path / "dest.txt"
        with pytest.raises(AtomicWriteError, match="does not exist"):
            atomic_copy_file(src, dst)

    def test_copy_path_as_string(self, tmp_path: Path) -> None:
        """Paths can be passed as strings."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "dest.txt"
        src.write_text("Content")
        atomic_copy_file(str(src), str(dst))
        assert dst.exists()


# -----------------------------------------------------------------------------
# validate_json_file tests
# -----------------------------------------------------------------------------


class TestValidateJsonFile:
    """Tests for validate_json_file function."""

    def test_valid_file(self, tmp_path: Path) -> None:
        """Valid JSON file passes validation."""
        target = tmp_path / "valid.json"
        target.write_text('{"key": "value"}')
        is_valid, data, error = validate_json_file(target)
        assert is_valid
        assert data == {"key": "value"}
        assert error is None

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Non-existent file fails validation."""
        target = tmp_path / "nonexistent.json"
        is_valid, data, error = validate_json_file(target)
        assert not is_valid
        assert data is None
        assert "does not exist" in (error or "")

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file fails validation."""
        target = tmp_path / "empty.json"
        target.write_text("")
        is_valid, data, error = validate_json_file(target)
        assert not is_valid
        assert "empty" in (error or "").lower()

    def test_whitespace_only_file(self, tmp_path: Path) -> None:
        """Whitespace-only file fails validation."""
        target = tmp_path / "whitespace.json"
        target.write_text("   \n\t  ")
        is_valid, data, error = validate_json_file(target)
        assert not is_valid
        assert "empty" in (error or "").lower()

    def test_invalid_json(self, tmp_path: Path) -> None:
        """Invalid JSON fails validation."""
        target = tmp_path / "invalid.json"
        target.write_text("{not valid json}")
        is_valid, data, error = validate_json_file(target)
        assert not is_valid
        assert "Invalid JSON" in (error or "")

    def test_with_passing_validator(self, tmp_path: Path) -> None:
        """Custom validator that passes."""
        target = tmp_path / "valid.json"
        target.write_text('{"required": true}')

        def validator(obj: dict) -> tuple[bool, str | None]:
            if "required" in obj:
                return True, None
            return False, "Missing required field"

        is_valid, data, error = validate_json_file(target, validator)
        assert is_valid
        assert data == {"required": True}

    def test_with_failing_validator(self, tmp_path: Path) -> None:
        """Custom validator that fails."""
        target = tmp_path / "valid.json"
        target.write_text('{"other": true}')

        def validator(obj: dict) -> tuple[bool, str | None]:
            if "required" in obj:
                return True, None
            return False, "Missing required field"

        is_valid, data, error = validate_json_file(target, validator)
        assert not is_valid
        assert data == {"other": True}  # Data is still returned
        assert "Missing required field" in (error or "")


# -----------------------------------------------------------------------------
# safe_read_json tests
# -----------------------------------------------------------------------------


class TestSafeReadJson:
    """Tests for safe_read_json function."""

    def test_reads_valid_file(self, tmp_path: Path) -> None:
        """Valid file is read successfully."""
        target = tmp_path / "valid.json"
        target.write_text('{"key": "value"}')
        data, error = safe_read_json(target)
        assert data == {"key": "value"}
        assert error is None

    def test_returns_default_for_missing_file(self, tmp_path: Path) -> None:
        """Default is returned for missing file."""
        target = tmp_path / "nonexistent.json"
        default = {"default": True}
        data, error = safe_read_json(target, default=default)
        assert data == default
        assert "does not exist" in (error or "")

    def test_returns_default_for_invalid_file(self, tmp_path: Path) -> None:
        """Default is returned for invalid file."""
        target = tmp_path / "invalid.json"
        target.write_text("{invalid}")
        default = {"default": True}
        data, error = safe_read_json(target, default=default)
        assert data == default
        assert "Invalid JSON" in (error or "")

    def test_returns_none_default(self, tmp_path: Path) -> None:
        """None is returned as default when not specified."""
        target = tmp_path / "nonexistent.json"
        data, error = safe_read_json(target)
        assert data is None
        assert error is not None


# -----------------------------------------------------------------------------
# recover_or_create_json tests
# -----------------------------------------------------------------------------


class TestRecoverOrCreateJson:
    """Tests for recover_or_create_json function."""

    def test_reads_existing_valid_file(self, tmp_path: Path) -> None:
        """Existing valid file is read without calling creator."""
        target = tmp_path / "existing.json"
        target.write_text('{"existing": true}')

        creator_called = []

        def creator() -> dict:
            creator_called.append(True)
            return {"created": True}

        success, data, error = recover_or_create_json(target, creator)
        assert success
        assert data == {"existing": True}
        assert len(creator_called) == 0  # Creator not called

    def test_creates_missing_file(self, tmp_path: Path) -> None:
        """Missing file is created using creator."""
        target = tmp_path / "missing.json"

        def creator() -> dict:
            return {"created": True}

        success, data, error = recover_or_create_json(target, creator)
        assert success
        assert data == {"created": True}
        assert target.exists()

    def test_recreates_invalid_file(self, tmp_path: Path) -> None:
        """Invalid file is recreated using creator."""
        target = tmp_path / "invalid.json"
        target.write_text("{invalid}")

        def creator() -> dict:
            return {"created": True}

        success, data, error = recover_or_create_json(target, creator)
        assert success
        assert data == {"created": True}

    def test_with_validator(self, tmp_path: Path) -> None:
        """Validator is used to check file."""
        target = tmp_path / "test.json"
        target.write_text('{"version": 1}')

        def validator(obj: dict) -> tuple[bool, str | None]:
            if obj.get("version") == 2:
                return True, None
            return False, "Wrong version"

        def creator() -> dict:
            return {"version": 2}

        success, data, error = recover_or_create_json(
            target, creator, validator=validator
        )
        assert success
        assert data == {"version": 2}

    def test_creator_failure(self, tmp_path: Path) -> None:
        """Creator failure is handled."""
        target = tmp_path / "test.json"

        def creator() -> dict:
            raise RuntimeError("Creator failed")

        success, data, error = recover_or_create_json(
            target, creator, max_attempts=1
        )
        assert not success
        assert "Creator failed" in (error or "")


# -----------------------------------------------------------------------------
# AtomicJSONWriter context manager tests
# -----------------------------------------------------------------------------


class TestAtomicJSONWriter:
    """Tests for AtomicJSONWriter context manager."""

    def test_successful_write(self, tmp_path: Path) -> None:
        """Successful write completes normally."""
        target = tmp_path / "test.json"
        with AtomicJSONWriter(target) as writer:
            writer.write({"success": True})

        assert target.exists()
        data = json.loads(target.read_text())
        assert data == {"success": True}

    def test_rollback_on_exception(self, tmp_path: Path) -> None:
        """Backup is restored on exception."""
        target = tmp_path / "test.json"
        target.write_text('{"original": true}')

        try:
            with AtomicJSONWriter(target) as writer:
                writer.write({"new": True})
                raise RuntimeError("Simulated error")
        except RuntimeError:
            pass

        # Original should be restored
        data = json.loads(target.read_text())
        assert data == {"original": True}

    def test_backup_removed_on_success(self, tmp_path: Path) -> None:
        """Backup file is removed after successful write."""
        target = tmp_path / "test.json"
        target.write_text('{"original": true}')
        backup = target.with_suffix(".json.backup")

        with AtomicJSONWriter(target) as writer:
            writer.write({"new": True})

        assert not backup.exists()

    def test_creates_new_file(self, tmp_path: Path) -> None:
        """Creates new file if it doesn't exist."""
        target = tmp_path / "new.json"

        with AtomicJSONWriter(target) as writer:
            writer.write({"new": True})

        assert target.exists()
        data = json.loads(target.read_text())
        assert data == {"new": True}

    def test_custom_indent(self, tmp_path: Path) -> None:
        """Custom indent is respected."""
        target = tmp_path / "test.json"

        with AtomicJSONWriter(target, indent=4) as writer:
            writer.write({"key": "value"})

        content = target.read_text()
        assert "    " in content  # 4-space indent

    def test_data_attribute(self, tmp_path: Path) -> None:
        """Written data is stored in data attribute."""
        target = tmp_path / "test.json"

        with AtomicJSONWriter(target) as writer:
            writer.write({"test": 123})
            assert writer.data == {"test": 123}

    def test_path_as_string(self, tmp_path: Path) -> None:
        """Path can be passed as string."""
        target = str(tmp_path / "test.json")

        with AtomicJSONWriter(target) as writer:
            writer.write({"test": True})

        assert Path(target).exists()


# -----------------------------------------------------------------------------
# Exception types tests
# -----------------------------------------------------------------------------


class TestExceptionTypes:
    """Tests for exception types."""

    def test_atomic_write_error_is_exception(self) -> None:
        """AtomicWriteError is an Exception."""
        error = AtomicWriteError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_json_validation_error_is_exception(self) -> None:
        """JSONValidationError is an Exception."""
        error = JSONValidationError("Validation failed")
        assert isinstance(error, Exception)
        assert str(error) == "Validation failed"
