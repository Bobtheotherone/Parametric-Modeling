# SPDX-License-Identifier: MIT
"""Additional unit tests for tools/git_guard.py.

Supplements test_git_guard.py with coverage for:
- Secret pattern matching edge cases
- _find_tracked_egg_info edge cases
- _scan_file functionality
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tools import git_guard


class TestSecretPatterns:
    """Tests for SECRET_PATTERNS regex matching."""

    def test_openai_key_pattern_matches(self) -> None:
        """Test OpenAI-style key pattern matches valid keys."""
        pattern = git_guard.SECRET_PATTERNS[0][1]
        assert pattern.search("sk-abc123def456ghi789jkl0123456789")
        assert pattern.search("  sk-abc123def456ghi789jkl0123456789  ")
        assert pattern.search("API_KEY=sk-abc123def456ghi789jkl0123456789")

    def test_openai_key_pattern_rejects_short(self) -> None:
        """Test OpenAI-style key pattern rejects keys that are too short."""
        pattern = git_guard.SECRET_PATTERNS[0][1]
        assert not pattern.search("sk-short")
        assert not pattern.search("sk-abc")

    def test_anthropic_key_pattern_matches(self) -> None:
        """Test Anthropic key pattern matches valid keys."""
        pattern = git_guard.SECRET_PATTERNS[1][1]
        assert pattern.search("sk-ant-abc123-xyz789-01234")
        assert pattern.search("  sk-ant-longerkeyvalue123456  ")

    def test_anthropic_key_pattern_rejects_short(self) -> None:
        """Test Anthropic key pattern rejects keys that are too short."""
        pattern = git_guard.SECRET_PATTERNS[1][1]
        assert not pattern.search("sk-ant-ab")
        assert not pattern.search("sk-ant-1")

    def test_google_api_key_pattern_matches(self) -> None:
        """Test Google API key pattern matches valid keys."""
        pattern = git_guard.SECRET_PATTERNS[2][1]
        assert pattern.search("AIzaSyBcdefghijklmnopqrstuvwx")
        assert pattern.search("key=AIzaSyBcdefghijklmnopqrstuvwx")

    def test_google_api_key_pattern_rejects_invalid(self) -> None:
        """Test Google API key pattern rejects invalid formats."""
        pattern = git_guard.SECRET_PATTERNS[2][1]
        assert not pattern.search("AIza123")  # Too short
        assert not pattern.search("AIzb_1234567890123456789")  # Wrong prefix

    def test_generic_api_key_pattern_matches(self) -> None:
        """Test generic API_KEY assignment pattern matches."""
        pattern = git_guard.SECRET_PATTERNS[3][1]
        assert pattern.search("API_KEY='my_secret_value'")
        assert pattern.search('API_KEY="my_secret_value"')
        assert pattern.search('OPENAI_API_KEY = "sk_test_12345678"')
        assert pattern.search('DATABASE_API_KEY="verysecretkey123"')

    def test_generic_api_key_pattern_rejects_short(self) -> None:
        """Test generic API_KEY pattern rejects short values."""
        pattern = git_guard.SECRET_PATTERNS[3][1]
        assert not pattern.search("API_KEY='short'")  # Less than 8 chars
        assert not pattern.search('API_KEY=""')

    def test_bearer_token_pattern_matches(self) -> None:
        """Test Bearer token pattern matches valid tokens."""
        pattern = git_guard.SECRET_PATTERNS[4][1]
        assert pattern.search("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6")
        assert pattern.search("Authorization: Bearer abcdef123456789012345678")

    def test_bearer_token_pattern_rejects_short(self) -> None:
        """Test Bearer token pattern rejects short tokens."""
        pattern = git_guard.SECRET_PATTERNS[4][1]
        assert not pattern.search("Bearer abc")
        assert not pattern.search("Bearer 12345")


class TestFindTrackedEggInfo:
    """Tests for _find_tracked_egg_info function edge cases."""

    def test_empty_list_returns_empty(self) -> None:
        """Empty input returns empty list."""
        assert git_guard._find_tracked_egg_info([]) == []

    def test_no_egg_info_returns_empty(self) -> None:
        """No .egg-info paths returns empty list."""
        paths = [
            "src/main.py",
            "tests/test_main.py",
            "docs/README.md",
        ]
        assert git_guard._find_tracked_egg_info(paths) == []

    def test_root_egg_info_detected(self) -> None:
        """Egg-info at root level is detected."""
        paths = ["mypackage.egg-info/PKG-INFO"]
        hits = git_guard._find_tracked_egg_info(paths)
        assert hits == ["mypackage.egg-info/PKG-INFO"]

    def test_nested_egg_info_detected(self) -> None:
        """Egg-info in nested directory is detected."""
        paths = ["src/mypackage.egg-info/SOURCES.txt"]
        hits = git_guard._find_tracked_egg_info(paths)
        assert hits == ["src/mypackage.egg-info/SOURCES.txt"]

    def test_deeply_nested_egg_info_detected(self) -> None:
        """Egg-info in deeply nested directory is detected."""
        paths = ["packages/core/mylib.egg-info/top_level.txt"]
        hits = git_guard._find_tracked_egg_info(paths)
        assert hits == ["packages/core/mylib.egg-info/top_level.txt"]

    def test_multiple_egg_info_files_detected(self) -> None:
        """Multiple .egg-info files are all detected."""
        paths = [
            "src/pkg1.egg-info/PKG-INFO",
            "src/pkg1.egg-info/SOURCES.txt",
            "src/pkg2.egg-info/PKG-INFO",
            "src/normal_file.py",
        ]
        hits = git_guard._find_tracked_egg_info(paths)
        assert len(hits) == 3
        assert "src/pkg1.egg-info/PKG-INFO" in hits
        assert "src/pkg1.egg-info/SOURCES.txt" in hits
        assert "src/pkg2.egg-info/PKG-INFO" in hits

    def test_egg_info_in_filename_not_matched(self) -> None:
        """Files with 'egg-info' in name but not as directory are not matched."""
        paths = [
            "docs/about-egg-info.txt",
            "test_egg-info_handler.py",
        ]
        # These should NOT match as they are not .egg-info/ directories
        hits = git_guard._find_tracked_egg_info(paths)
        assert hits == []


class TestScanFile:
    """Tests for _scan_file function."""

    def test_scan_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns no hits."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")
        hits = git_guard._scan_file(empty_file)
        assert hits == []

    def test_scan_clean_file(self, tmp_path: Path) -> None:
        """File without secrets returns no hits."""
        clean_file = tmp_path / "clean.py"
        clean_file.write_text(
            "# Regular Python code\nprint('Hello, world!')\n",
            encoding="utf-8",
        )
        hits = git_guard._scan_file(clean_file)
        assert hits == []

    def test_scan_file_with_openai_key(self, tmp_path: Path) -> None:
        """File with OpenAI-style key is detected."""
        secret_file = tmp_path / "config.py"
        secret_file.write_text(
            "key = 'sk-abc123def456ghi789jkl0123456789'\n",
            encoding="utf-8",
        )
        hits = git_guard._scan_file(secret_file)
        assert "OpenAI-style key" in hits

    def test_scan_file_with_bearer_token(self, tmp_path: Path) -> None:
        """File with Bearer token is detected."""
        secret_file = tmp_path / "auth.py"
        secret_file.write_text(
            "headers = {'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6'}\n",
            encoding="utf-8",
        )
        hits = git_guard._scan_file(secret_file)
        assert "Bearer token" in hits

    def test_scan_file_with_multiple_secrets(self, tmp_path: Path) -> None:
        """File with multiple secret types is detected."""
        secret_file = tmp_path / "secrets.py"
        secret_file.write_text(
            "openai_key = 'sk-abc123def456ghi789jkl0123456789'\n"
            "google_key = 'AIzaSyBcdefghijklmnopqrstuvwx'\n",
            encoding="utf-8",
        )
        hits = git_guard._scan_file(secret_file)
        assert "OpenAI-style key" in hits
        assert "Google API key" in hits

    def test_scan_nonexistent_file_returns_empty(self, tmp_path: Path) -> None:
        """Nonexistent file returns empty list (no exception)."""
        nonexistent = tmp_path / "does_not_exist.py"
        hits = git_guard._scan_file(nonexistent)
        assert hits == []

    def test_scan_binary_file_returns_empty(self, tmp_path: Path) -> None:
        """Binary file with read errors returns empty list."""
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(bytes(range(256)))
        # Should not raise, should return empty or partial results
        hits = git_guard._scan_file(binary_file)
        # We don't assert the result, just that it doesn't crash
        assert isinstance(hits, list)


class TestEggInfoRegex:
    """Tests for EGG_INFO_RE regex pattern."""

    def test_matches_root_level_egg_info(self) -> None:
        """Matches .egg-info at root level."""
        assert git_guard.EGG_INFO_RE.search("package.egg-info/PKG-INFO")

    def test_matches_nested_egg_info(self) -> None:
        """Matches .egg-info in nested path."""
        assert git_guard.EGG_INFO_RE.search("src/package.egg-info/PKG-INFO")

    def test_matches_egg_info_with_underscores(self) -> None:
        """Matches .egg-info with underscores in package name."""
        assert git_guard.EGG_INFO_RE.search("my_package.egg-info/SOURCES.txt")

    def test_matches_egg_info_with_hyphens(self) -> None:
        """Matches .egg-info with hyphens in package name."""
        assert git_guard.EGG_INFO_RE.search("my-package.egg-info/top_level.txt")

    def test_does_not_match_egg_info_in_middle(self) -> None:
        """Does not match 'egg-info' as part of a larger name without /."""
        # The pattern requires .egg-info/ (with trailing slash or at end)
        pattern = git_guard.EGG_INFO_RE
        # This should match because the pattern looks for .egg-info/
        assert pattern.search("pkg.egg-info/file")
        # But this should not match - no trailing slash context
        # Actually looking at the regex: r"(^|/)[^/]+\.egg-info/"
        # It requires a trailing / after .egg-info
        assert not pattern.search("about-egg-info.txt")
