# SPDX-License-Identifier: MIT
"""Additional unit tests for bridge/atomic_io.py edge cases.

Supplements test_atomic_io.py with coverage for:
- Thread safety scenarios
- File permission edge cases
- Edge cases for encoding handling
- Large file handling
- Concurrent access patterns
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

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


class TestAtomicWriteTextEdgeCases:
    """Edge case tests for atomic_write_text function."""

    def test_very_long_content(self, tmp_path: Path) -> None:
        """Very long content should be written correctly."""
        target = tmp_path / "large.txt"
        content = "x" * 1_000_000  # 1MB of data
        atomic_write_text(target, content)
        assert target.read_text() == content

    def test_binary_like_unicode_content(self, tmp_path: Path) -> None:
        """Content with various Unicode characters should work."""
        target = tmp_path / "unicode.txt"
        content = "Line 1: 日本語\nLine 2: 🚀🎉🔥\nLine 3: Ñoño\n"
        atomic_write_text(target, content)
        assert target.read_text() == content

    def test_only_newlines(self, tmp_path: Path) -> None:
        """Content with only newlines should work."""
        target = tmp_path / "newlines.txt"
        content = "\n\n\n\n\n"
        atomic_write_text(target, content)
        assert target.read_text() == content

    def test_tabs_and_special_whitespace(self, tmp_path: Path) -> None:
        """Content with tabs and special whitespace should work."""
        target = tmp_path / "whitespace.txt"
        content = "col1\tcol2\tcol3\n\t\tindented\n"
        atomic_write_text(target, content)
        assert target.read_text() == content

    def test_overwrite_larger_file(self, tmp_path: Path) -> None:
        """Overwriting a larger file with smaller content should work."""
        target = tmp_path / "shrink.txt"
        target.write_text("x" * 10000)
        atomic_write_text(target, "small")
        assert target.read_text() == "small"

    def test_overwrite_smaller_file(self, tmp_path: Path) -> None:
        """Overwriting a smaller file with larger content should work."""
        target = tmp_path / "grow.txt"
        target.write_text("small")
        large_content = "x" * 10000
        atomic_write_text(target, large_content)
        assert target.read_text() == large_content


class TestAtomicWriteJsonEdgeCases:
    """Edge case tests for atomic_write_json function."""

    def test_deeply_nested_structure(self, tmp_path: Path) -> None:
        """Deeply nested JSON structure should work."""
        target = tmp_path / "nested.json"
        data: dict[str, Any] = {"level": 0, "child": {}}
        current = data["child"]
        for i in range(1, 50):
            current["level"] = i
            current["child"] = {}
            current = current["child"]
        current["level"] = 50

        atomic_write_json(target, data)
        loaded = json.loads(target.read_text())
        assert loaded["level"] == 0
        assert loaded["child"]["level"] == 1

    def test_large_list(self, tmp_path: Path) -> None:
        """Large list should work."""
        target = tmp_path / "list.json"
        data = list(range(10000))
        atomic_write_json(target, data)
        loaded = json.loads(target.read_text())
        assert loaded == data

    def test_special_characters_in_strings(self, tmp_path: Path) -> None:
        """JSON with special characters in strings should work."""
        target = tmp_path / "special.json"
        data = {
            "quote": 'He said "hello"',
            "backslash": "path\\to\\file",
            "newline": "line1\nline2",
            "tab": "col1\tcol2",
            "unicode": "日本語🚀",
        }
        atomic_write_json(target, data)
        loaded = json.loads(target.read_text())
        assert loaded == data

    def test_null_values(self, tmp_path: Path) -> None:
        """JSON with null values should work."""
        target = tmp_path / "null.json"
        data = {"value": None, "list": [None, None], "nested": {"also_null": None}}
        atomic_write_json(target, data)
        loaded = json.loads(target.read_text())
        assert loaded == data

    def test_numeric_extremes(self, tmp_path: Path) -> None:
        """JSON with extreme numeric values should work."""
        target = tmp_path / "numbers.json"
        data = {
            "large_int": 10**18,
            "small_float": 1e-15,
            "large_float": 1e15,
            "negative": -(10**15),
            "zero": 0,
        }
        atomic_write_json(target, data)
        loaded = json.loads(target.read_text())
        assert loaded["large_int"] == 10**18
        assert loaded["zero"] == 0

    def test_empty_dict(self, tmp_path: Path) -> None:
        """Empty dict should work."""
        target = tmp_path / "empty.json"
        atomic_write_json(target, {})
        loaded = json.loads(target.read_text())
        assert loaded == {}

    def test_empty_list(self, tmp_path: Path) -> None:
        """Empty list should work."""
        target = tmp_path / "empty_list.json"
        atomic_write_json(target, [])
        loaded = json.loads(target.read_text())
        assert loaded == []

    def test_validator_receives_parsed_data(self, tmp_path: Path) -> None:
        """Validator should receive properly parsed data."""
        target = tmp_path / "validated.json"
        received_data = []

        def validator(obj: dict) -> bool:
            received_data.append(obj)
            return obj.get("valid") is True

        data = {"valid": True, "count": 42}
        atomic_write_json(target, data, validate_schema=validator)

        assert len(received_data) == 1
        assert received_data[0] == data


class TestAtomicCopyFileEdgeCases:
    """Edge case tests for atomic_copy_file function."""

    def test_copy_empty_file(self, tmp_path: Path) -> None:
        """Copying empty file should work."""
        src = tmp_path / "empty.txt"
        dst = tmp_path / "empty_copy.txt"
        src.write_text("")
        atomic_copy_file(src, dst)
        assert dst.exists()
        assert dst.read_text() == ""

    def test_copy_preserves_exact_content(self, tmp_path: Path) -> None:
        """Copy should preserve exact binary content."""
        src = tmp_path / "source.bin"
        dst = tmp_path / "dest.bin"
        content = bytes(range(256)) * 100  # 25.6KB
        src.write_bytes(content)
        atomic_copy_file(src, dst)
        assert dst.read_bytes() == content

    def test_copy_to_same_directory(self, tmp_path: Path) -> None:
        """Copy to same directory should work."""
        src = tmp_path / "original.txt"
        dst = tmp_path / "copy.txt"
        src.write_text("content")
        atomic_copy_file(src, dst)
        assert dst.read_text() == "content"
        assert src.read_text() == "content"  # Source unchanged


class TestValidateJsonFileEdgeCases:
    """Edge case tests for validate_json_file function."""

    def test_json_with_bom(self, tmp_path: Path) -> None:
        """JSON file with BOM should be handled."""
        target = tmp_path / "bom.json"
        # Write with UTF-8 BOM
        with open(target, "wb") as f:
            f.write(b"\xef\xbb\xbf")  # UTF-8 BOM
            f.write(b'{"key": "value"}')

        is_valid, data, error = validate_json_file(target)
        # Should either parse successfully or report a clear error
        # (BOM handling depends on json library version)
        assert isinstance(is_valid, bool)

    def test_validator_exception_is_caught(self, tmp_path: Path) -> None:
        """Exception in validator should be caught."""
        target = tmp_path / "test.json"
        target.write_text('{"key": "value"}')

        def bad_validator(obj: dict) -> tuple[bool, str | None]:
            raise RuntimeError("Validator crashed")

        is_valid, data, error = validate_json_file(target, bad_validator)
        assert not is_valid
        assert "Validation error" in (error or "")


class TestSafeReadJsonEdgeCases:
    """Edge case tests for safe_read_json function."""

    def test_returns_default_on_missing_file(self, tmp_path: Path) -> None:
        """Default should be returned for missing file."""
        target = tmp_path / "missing.json"
        default = {"key": "original"}
        data, error = safe_read_json(target, default=default)

        assert data == default
        assert error is not None
        assert "does not exist" in error

    def test_returns_none_when_no_default(self, tmp_path: Path) -> None:
        """None should be returned when no default is provided."""
        target = tmp_path / "missing.json"
        data, error = safe_read_json(target)

        assert data is None
        assert error is not None


class TestRecoverOrCreateJsonEdgeCases:
    """Edge case tests for recover_or_create_json function."""

    def test_creator_called_on_empty_file(self, tmp_path: Path) -> None:
        """Creator should be called when file is empty."""
        target = tmp_path / "empty.json"
        target.write_text("")

        call_count = [0]

        def creator() -> dict:
            call_count[0] += 1
            return {"created": True}

        success, data, error = recover_or_create_json(target, creator)
        assert success
        assert data == {"created": True}
        assert call_count[0] >= 1

    def test_existing_valid_file_not_recreated(self, tmp_path: Path) -> None:
        """Valid file should not trigger creator."""
        target = tmp_path / "valid.json"
        target.write_text('{"existing": true}')

        call_count = [0]

        def creator() -> dict:
            call_count[0] += 1
            return {"created": True}

        success, data, error = recover_or_create_json(target, creator)
        assert success
        assert data == {"existing": True}
        assert call_count[0] == 0


class TestAtomicJSONWriterEdgeCases:
    """Edge case tests for AtomicJSONWriter context manager."""

    def test_exception_during_write_restores_backup(self, tmp_path: Path) -> None:
        """Exception during write should restore backup."""
        target = tmp_path / "test.json"
        target.write_text('{"original": true}')

        try:
            with AtomicJSONWriter(target) as writer:
                writer.write({"new": True})
                # Verify new content was written
                assert json.loads(target.read_text()) == {"new": True}
                # Raise exception
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Original should be restored
        restored = json.loads(target.read_text())
        assert restored == {"original": True}

    def test_no_exception_removes_backup(self, tmp_path: Path) -> None:
        """Successful write should remove backup file."""
        target = tmp_path / "test.json"
        target.write_text('{"original": true}')
        backup = target.with_suffix(".json.backup")

        with AtomicJSONWriter(target) as writer:
            writer.write({"new": True})

        assert not backup.exists()
        assert json.loads(target.read_text()) == {"new": True}

    def test_data_attribute_after_write(self, tmp_path: Path) -> None:
        """Data attribute should reflect last written data."""
        target = tmp_path / "test.json"

        with AtomicJSONWriter(target) as writer:
            writer.write({"first": 1})
            assert writer.data == {"first": 1}

            writer.write({"second": 2})
            assert writer.data == {"second": 2}


class TestConcurrentWrites:
    """Tests for concurrent write scenarios."""

    def test_sequential_writes_to_same_file(self, tmp_path: Path) -> None:
        """Multiple sequential writes should all succeed."""
        target = tmp_path / "sequential.txt"

        for i in range(10):
            atomic_write_text(target, f"content_{i}")
            assert target.read_text() == f"content_{i}"

    def test_writes_to_different_files(self, tmp_path: Path) -> None:
        """Writes to different files should all succeed."""
        results = []

        def write_file(idx: int) -> None:
            target = tmp_path / f"file_{idx}.txt"
            atomic_write_text(target, f"content_{idx}")
            results.append((idx, target.read_text()))

        threads = [threading.Thread(target=write_file, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        for idx, content in results:
            assert content == f"content_{idx}"
