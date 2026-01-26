# SPDX-License-Identifier: MIT
"""Unit tests for helper functions in tools/verify.py.

This module tests internal helper functions not covered by test_verify_artifacts.py:
- _gate_slug: Gate name sanitization
- _format_cmd: Command formatting
- _safe_relpath: Safe relative path computation
- _path_mtime: Safe modification time retrieval
- _resolve_m0_timeout: Timeout resolution from args and environment
- _detect_milestone_id: Milestone detection from DESIGN_DOCUMENT.md
- GateResult dataclass
- VerifyArtifacts dataclass
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest
from tools import verify


class TestGateSlug:
    """Tests for _gate_slug function."""

    def test_simple_name(self) -> None:
        """Simple alphanumeric name passes through."""
        assert verify._gate_slug("pytest") == "pytest"

    def test_name_with_underscore(self) -> None:
        """Underscores are preserved."""
        assert verify._gate_slug("spec_lint") == "spec_lint"

    def test_name_with_hyphen(self) -> None:
        """Hyphens are preserved."""
        assert verify._gate_slug("git-guard") == "git-guard"

    def test_name_with_dot(self) -> None:
        """Dots are preserved."""
        assert verify._gate_slug("m1.smoke") == "m1.smoke"

    def test_name_with_spaces(self) -> None:
        """Spaces are converted to underscores."""
        assert verify._gate_slug("my gate name") == "my_gate_name"

    def test_name_with_special_chars(self) -> None:
        """Special characters are converted to underscores."""
        assert verify._gate_slug("test@#$%gate") == "test____gate"

    def test_leading_trailing_spaces(self) -> None:
        """Leading/trailing spaces are stripped."""
        assert verify._gate_slug("  pytest  ") == "pytest"

    def test_empty_string(self) -> None:
        """Empty string returns 'gate'."""
        assert verify._gate_slug("") == "gate"

    def test_all_special_chars(self) -> None:
        """All special chars converted to underscores."""
        # The function converts special chars to underscores, not to 'gate'
        assert verify._gate_slug("@#$%^") == "_____"


class TestFormatCmd:
    """Tests for _format_cmd function."""

    def test_simple_command(self) -> None:
        """Simple command list formats correctly."""
        result = verify._format_cmd(["echo", "hello"])
        assert result == "echo hello"

    def test_command_with_quotes(self) -> None:
        """Command with arguments needing quotes is properly escaped."""
        result = verify._format_cmd(["echo", "hello world"])
        assert "echo" in result
        assert "hello world" in result or "'hello world'" in result

    def test_empty_command(self) -> None:
        """Empty command list returns empty string."""
        assert verify._format_cmd([]) == ""

    def test_none_command(self) -> None:
        """None returns empty string."""
        assert verify._format_cmd(None) == ""


class TestSafeRelpath:
    """Tests for _safe_relpath function."""

    def test_relative_path_inside_root(self, tmp_path: Path) -> None:
        """Path inside root returns relative path."""
        subdir = tmp_path / "subdir" / "file.txt"
        subdir.parent.mkdir(parents=True, exist_ok=True)
        subdir.touch()
        result = verify._safe_relpath(subdir, tmp_path)
        assert result == Path("subdir") / "file.txt"

    def test_path_outside_root(self, tmp_path: Path) -> None:
        """Path outside root returns just the file name."""
        other_path = Path("/some/other/path/file.txt")
        result = verify._safe_relpath(other_path, tmp_path)
        assert result == Path("file.txt")

    def test_same_path_and_root(self, tmp_path: Path) -> None:
        """Path equal to root returns '.'."""
        result = verify._safe_relpath(tmp_path, tmp_path)
        assert result == Path(".")


class TestPathMtime:
    """Tests for _path_mtime function."""

    def test_existing_file(self, tmp_path: Path) -> None:
        """Existing file returns non-zero mtime."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content", encoding="utf-8")
        mtime = verify._path_mtime(test_file)
        assert mtime > 0

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Nonexistent file returns 0.0."""
        nonexistent = tmp_path / "does_not_exist.txt"
        mtime = verify._path_mtime(nonexistent)
        assert mtime == 0.0


class TestResolveM0Timeout:
    """Tests for _resolve_m0_timeout function."""

    def test_uses_arg_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Timeout from args takes precedence."""
        monkeypatch.delenv("FF_M0_GATE_TIMEOUT_S", raising=False)
        args = argparse.Namespace(m0_timeout_s=120)
        result = verify._resolve_m0_timeout(args)
        assert result == 120

    def test_uses_env_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variable used when arg is None."""
        monkeypatch.setenv("FF_M0_GATE_TIMEOUT_S", "180")
        args = argparse.Namespace(m0_timeout_s=None)
        result = verify._resolve_m0_timeout(args)
        assert result == 180

    def test_invalid_env_value_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid environment variable falls back to default."""
        monkeypatch.setenv("FF_M0_GATE_TIMEOUT_S", "invalid")
        args = argparse.Namespace(m0_timeout_s=None)
        result = verify._resolve_m0_timeout(args)
        assert result == verify.DEFAULT_M0_GATE_TIMEOUT_S

    def test_no_arg_no_env_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No arg and no env falls back to default."""
        monkeypatch.delenv("FF_M0_GATE_TIMEOUT_S", raising=False)
        args = argparse.Namespace(m0_timeout_s=None)
        result = verify._resolve_m0_timeout(args)
        assert result == verify.DEFAULT_M0_GATE_TIMEOUT_S

    def test_missing_attr_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing attribute in args falls back to env or default."""
        monkeypatch.delenv("FF_M0_GATE_TIMEOUT_S", raising=False)
        args = argparse.Namespace()
        result = verify._resolve_m0_timeout(args)
        assert result == verify.DEFAULT_M0_GATE_TIMEOUT_S


class TestDetectMilestoneId:
    """Tests for _detect_milestone_id function."""

    def test_milestone_m0(self, tmp_path: Path) -> None:
        """Detects M0 milestone."""
        doc = tmp_path / "DESIGN_DOCUMENT.md"
        doc.write_text("**Milestone:** M0 — test\n", encoding="utf-8")
        result = verify._detect_milestone_id(tmp_path)
        assert result == "M0"

    def test_milestone_m1(self, tmp_path: Path) -> None:
        """Detects M1 milestone."""
        doc = tmp_path / "DESIGN_DOCUMENT.md"
        doc.write_text("**Milestone:** M1 — fabrication\n", encoding="utf-8")
        result = verify._detect_milestone_id(tmp_path)
        assert result == "M1"

    def test_milestone_m2(self, tmp_path: Path) -> None:
        """Detects M2 milestone."""
        doc = tmp_path / "DESIGN_DOCUMENT.md"
        doc.write_text("**Milestone:** M2 — simulation\n", encoding="utf-8")
        result = verify._detect_milestone_id(tmp_path)
        assert result == "M2"

    def test_no_design_document(self, tmp_path: Path) -> None:
        """Returns None when DESIGN_DOCUMENT.md doesn't exist."""
        result = verify._detect_milestone_id(tmp_path)
        assert result is None

    def test_no_milestone_in_document(self, tmp_path: Path) -> None:
        """Returns None when no milestone pattern found."""
        doc = tmp_path / "DESIGN_DOCUMENT.md"
        doc.write_text("# Design Document\nNo milestone here.\n", encoding="utf-8")
        result = verify._detect_milestone_id(tmp_path)
        assert result is None


class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_default_values(self) -> None:
        """GateResult has sensible defaults."""
        result = verify.GateResult(name="test", passed=True)
        assert result.name == "test"
        assert result.passed is True
        assert result.cmd is None
        assert result.returncode is None
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.note == ""

    def test_with_all_fields(self) -> None:
        """GateResult can be created with all fields."""
        result = verify.GateResult(
            name="pytest",
            passed=False,
            cmd=["python", "-m", "pytest"],
            returncode=1,
            stdout="test output",
            stderr="test error",
            note="rc=1",
        )
        assert result.name == "pytest"
        assert result.passed is False
        assert result.cmd == ["python", "-m", "pytest"]
        assert result.returncode == 1
        assert result.stdout == "test output"
        assert result.stderr == "test error"
        assert result.note == "rc=1"


class TestVerifyArtifacts:
    """Tests for VerifyArtifacts dataclass."""

    def test_frozen_dataclass(self, tmp_path: Path) -> None:
        """VerifyArtifacts is frozen (immutable)."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        logs_dir = run_dir / "logs"
        logs_dir.mkdir()
        failures_dir = run_dir / "failures"
        failures_dir.mkdir()
        tmp_dir = run_dir / "tmp"
        tmp_dir.mkdir()

        artifacts = verify.VerifyArtifacts(
            run_id="test123",
            run_dir=run_dir,
            logs_dir=logs_dir,
            failures_dir=failures_dir,
            tmp_dir=tmp_dir,
        )

        assert artifacts.run_id == "test123"
        assert artifacts.run_dir == run_dir

        # Frozen dataclass should raise on mutation
        with pytest.raises(AttributeError):
            artifacts.run_id = "changed"  # type: ignore[misc]


class TestInitVerifyArtifacts:
    """Tests for _init_verify_artifacts function."""

    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        """Creates expected directory structure."""
        artifacts = verify._init_verify_artifacts(tmp_path)

        assert artifacts.run_dir.exists()
        assert artifacts.logs_dir.exists()
        assert artifacts.failures_dir.exists()
        assert artifacts.tmp_dir.exists()

        assert artifacts.logs_dir == artifacts.run_dir / "logs"
        assert artifacts.failures_dir == artifacts.run_dir / "failures"
        assert artifacts.tmp_dir == artifacts.run_dir / "tmp"

    def test_run_id_format(self, tmp_path: Path) -> None:
        """Run ID is in expected timestamp format."""
        artifacts = verify._init_verify_artifacts(tmp_path)
        # Format: YYYYMMDDTHHMMSSZ
        assert len(artifacts.run_id) == 16
        assert artifacts.run_id.endswith("Z")
        assert "T" in artifacts.run_id


class TestBuildVerifyEnv:
    """Tests for _build_verify_env function."""

    def test_includes_deterministic_env(self, tmp_path: Path) -> None:
        """Includes all deterministic environment variables."""
        env = verify._build_verify_env(tmp_path)
        for key, value in verify.DETERMINISTIC_ENV.items():
            assert env[key] == value

    def test_sets_temp_dirs(self, tmp_path: Path) -> None:
        """Sets TMPDIR, TEMP, and TMP to provided path."""
        env = verify._build_verify_env(tmp_path)
        expected = str(tmp_path)
        assert env["TMPDIR"] == expected
        assert env["TEMP"] == expected
        assert env["TMP"] == expected


class TestTimestampUtc:
    """Tests for _timestamp_utc function."""

    def test_format(self) -> None:
        """Returns timestamp in expected format."""
        ts = verify._timestamp_utc()
        # Format: YYYYMMDDTHHMMSSZ
        assert len(ts) == 16
        assert ts[8] == "T"
        assert ts.endswith("Z")
        # All other characters should be digits
        digits = ts[:8] + ts[9:15]
        assert digits.isdigit()
