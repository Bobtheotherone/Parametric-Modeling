#!/usr/bin/env python3
"""Unit tests for scope validator path handling.

Tests verify that the scope guard:
1. Correctly validates paths against allowlist/denylist
2. Does NOT truncate paths (e.g., "tests/..." should NOT become "ests/...")
3. Handles edge cases like git diff prefixes (a/, b/)
4. Reports full, correct paths in error messages

Run with: pytest tests/test_scope_validator_paths.py -v
"""

from __future__ import annotations

import pytest

from bridge.patch_integration import (
    BACKFILL_ALLOWLIST,
    ScopeGuard,
    create_backfill_scope_guard,
    normalize_diff_path,
)


class TestNormalizeDiffPath:
    """Test path normalization for git diff output."""

    def test_removes_a_prefix(self):
        """Verify 'a/' prefix is removed correctly."""
        assert normalize_diff_path("a/tests/test_foo.py") == "tests/test_foo.py"
        assert normalize_diff_path("a/docs/readme.md") == "docs/readme.md"
        assert normalize_diff_path("a/bridge/loop.py") == "bridge/loop.py"

    def test_removes_b_prefix(self):
        """Verify 'b/' prefix is removed correctly."""
        assert normalize_diff_path("b/tests/test_foo.py") == "tests/test_foo.py"
        assert normalize_diff_path("b/docs/readme.md") == "docs/readme.md"
        assert normalize_diff_path("b/bridge/loop.py") == "bridge/loop.py"

    def test_no_truncation_of_tests_directory(self):
        """CRITICAL: Verify 'tests/' does NOT become 'ests/'."""
        # This was the actual bug - paths starting with 't' were losing that character
        assert normalize_diff_path("tests/test_foo.py") == "tests/test_foo.py"
        assert normalize_diff_path("a/tests/test_foo.py") == "tests/test_foo.py"
        assert normalize_diff_path("b/tests/test_foo.py") == "tests/test_foo.py"

        # Verify full path integrity
        path = "tests/test_m3_concurrency.py"
        normalized = normalize_diff_path(path)
        assert normalized == path
        assert normalized.startswith("tests/")
        assert not normalized.startswith("ests/")

    def test_no_truncation_of_docs_directory(self):
        """CRITICAL: Verify 'docs/' does NOT become 'ocs/'."""
        assert normalize_diff_path("docs/readme.md") == "docs/readme.md"
        assert normalize_diff_path("a/docs/readme.md") == "docs/readme.md"
        assert normalize_diff_path("b/docs/readme.md") == "docs/readme.md"

        # Verify full path integrity
        path = "docs/experiment-tracking.md"
        normalized = normalize_diff_path(path)
        assert normalized == path
        assert normalized.startswith("docs/")
        assert not normalized.startswith("ocs/")

    def test_no_truncation_of_bridge_directory(self):
        """CRITICAL: Verify 'bridge/' does NOT become 'ridge/'."""
        assert normalize_diff_path("bridge/loop.py") == "bridge/loop.py"
        assert normalize_diff_path("a/bridge/loop.py") == "bridge/loop.py"
        assert normalize_diff_path("b/bridge/loop.py") == "bridge/loop.py"

    def test_removes_dot_slash_prefix(self):
        """Verify './' prefix is removed correctly."""
        assert normalize_diff_path("./tests/test_foo.py") == "tests/test_foo.py"
        assert normalize_diff_path("./docs/readme.md") == "docs/readme.md"

    def test_handles_leading_slashes(self):
        """Verify leading slashes are stripped."""
        assert normalize_diff_path("/tests/test_foo.py") == "tests/test_foo.py"
        assert normalize_diff_path("//tests/test_foo.py") == "tests/test_foo.py"

    def test_plain_paths_unchanged(self):
        """Verify paths without prefixes are unchanged."""
        assert normalize_diff_path("tests/test_foo.py") == "tests/test_foo.py"
        assert normalize_diff_path("src/module.py") == "src/module.py"
        assert normalize_diff_path("pyproject.toml") == "pyproject.toml"


class TestScopeGuardPathValidation:
    """Test ScopeGuard validates paths correctly without truncation."""

    def test_allowed_paths_pass_validation(self):
        """Verify paths in allowlist are accepted."""
        guard = create_backfill_scope_guard()

        # Test with paths that should be allowed
        result = guard.check_paths(["tests/test_foo.py", "docs/readme.md"])
        assert result.allowed, f"Paths should be allowed: {result.violations}"

    def test_disallowed_paths_fail_validation(self):
        """Verify paths outside allowlist are rejected."""
        guard = create_backfill_scope_guard()

        # Test with paths that should be denied
        result = guard.check_paths(["src/module.py"])
        assert not result.allowed
        assert len(result.violations) == 1
        assert result.violations[0].path == "src/module.py"

    def test_truncated_paths_are_detected(self):
        """Verify truncated paths (like 'ests/') are NOT accepted as 'tests/'."""
        guard = create_backfill_scope_guard()

        # A truncated path should fail validation
        result = guard.check_paths(["ests/test_foo.py"])
        assert not result.allowed, "Truncated path 'ests/' should NOT match 'tests/**'"

        # The violation should show the full (truncated) path, not further truncate it
        assert result.violations[0].path == "ests/test_foo.py"

    def test_checked_paths_unchanged_in_result(self):
        """Verify checked_paths in result contains the original paths."""
        guard = create_backfill_scope_guard()

        original_paths = ["tests/test_foo.py", "docs/readme.md"]
        result = guard.check_paths(original_paths)

        assert result.checked_paths == original_paths

    def test_violation_paths_are_full_paths(self):
        """Verify violation paths are complete, not truncated."""
        guard = create_backfill_scope_guard()

        # Test a path that will be rejected
        result = guard.check_paths(["src/forbidden_module.py"])
        assert not result.allowed
        assert result.violations[0].path == "src/forbidden_module.py"
        assert not result.violations[0].path.startswith("rc/")  # Not truncated

    def test_mixed_allowed_and_denied_paths(self):
        """Verify mixed paths correctly identify only violations."""
        guard = create_backfill_scope_guard()

        paths = [
            "tests/test_foo.py",  # allowed
            "src/module.py",  # denied
            "docs/readme.md",  # allowed
        ]
        result = guard.check_paths(paths)

        assert not result.allowed
        assert len(result.violations) == 1
        assert result.violations[0].path == "src/module.py"


class TestBackfillScopeGuardConfig:
    """Test backfill-specific scope guard configuration."""

    def test_backfill_allowlist_includes_tests(self):
        """Verify backfill allowlist includes tests/**."""
        assert any("tests" in pattern for pattern in BACKFILL_ALLOWLIST)

    def test_backfill_allowlist_includes_docs(self):
        """Verify backfill allowlist includes docs/**."""
        assert any("docs" in pattern for pattern in BACKFILL_ALLOWLIST)

    def test_backfill_guard_accepts_test_files(self):
        """Verify backfill guard accepts test files."""
        guard = create_backfill_scope_guard()
        result = guard.check_paths(["tests/test_example.py"])
        assert result.allowed

    def test_backfill_guard_rejects_src_files(self):
        """Verify backfill guard rejects src/ files."""
        guard = create_backfill_scope_guard()
        result = guard.check_paths(["src/module.py"])
        assert not result.allowed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
