"""Tests for the no_skipped_tests gate in verify.py.

This gate ensures that any tests mapped to the current milestone's requirements
in the DESIGN_DOCUMENT.md test matrix cannot be skipped, xfail, or placeholder.
This prevents "green-but-incomplete" milestones.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from tools import verify


def _write_design_doc(path: Path, milestone_id: str, req_tests: dict[str, list[str]]) -> None:
    """Write a DESIGN_DOCUMENT.md with a test matrix.

    Args:
        path: Path to write the file.
        milestone_id: The milestone identifier (e.g., "M0").
        req_tests: Mapping of requirement IDs to test node IDs.
    """
    lines = [
        f"**Milestone:** {milestone_id} — test milestone",
        "",
        "## Normative Requirements (must)",
    ]
    for req_id in req_tests:
        lines.append(f"- [{req_id}] Some requirement description")
    lines.extend(
        [
            "",
            "## Definition of Done",
            "- Tests pass",
            "",
            "## Test Matrix",
            "| Requirement | Pytest(s) |",
            "|---|---|",
        ]
    )
    for req_id, tests in req_tests.items():
        lines.append(f"| {req_id} | {', '.join(tests)} |")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_test_file(path: Path, content: str) -> None:
    """Write a test file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestGateNoSkippedTests:
    """Tests for _gate_no_skipped_tests."""

    def test_passes_when_no_milestone(self, tmp_path: Path) -> None:
        """Gate passes when no milestone is detected."""
        result = verify._gate_no_skipped_tests(tmp_path, None)
        assert result.passed
        assert "no milestone" in result.note

    def test_passes_when_no_test_matrix(self, tmp_path: Path) -> None:
        """Gate passes when there's no test matrix."""
        doc_path = tmp_path / "DESIGN_DOCUMENT.md"
        doc_path.write_text("**Milestone:** M0 — test\n", encoding="utf-8")
        result = verify._gate_no_skipped_tests(tmp_path, "M0")
        assert result.passed
        assert "no test matrix" in result.note

    def test_passes_when_all_tests_active(self, tmp_path: Path) -> None:
        """Gate passes when all milestone tests are active (not skipped)."""
        doc_path = tmp_path / "DESIGN_DOCUMENT.md"
        _write_design_doc(
            doc_path,
            "M0",
            {"REQ-M0-001": ["tests/test_example.py::test_active"]},
        )

        # Write an active test
        test_content = dedent(
            """\
            def test_active():
                assert True
            """
        )
        _write_test_file(tmp_path / "tests" / "test_example.py", test_content)

        result = verify._gate_no_skipped_tests(tmp_path, "M0")
        assert result.passed
        assert "active" in result.note

    def test_fails_when_test_is_placeholder_pass(self, tmp_path: Path) -> None:
        """Gate fails when a milestone test is just 'pass' (placeholder)."""
        doc_path = tmp_path / "DESIGN_DOCUMENT.md"
        _write_design_doc(
            doc_path,
            "M0",
            {"REQ-M0-001": ["tests/test_example.py::test_placeholder"]},
        )

        # Write a placeholder test
        test_content = dedent(
            """\
            def test_placeholder():
                pass
            """
        )
        _write_test_file(tmp_path / "tests" / "test_example.py", test_content)

        result = verify._gate_no_skipped_tests(tmp_path, "M0")
        assert not result.passed
        assert "placeholder" in result.note.lower() or "skipped" in result.note.lower()

    def test_fails_when_test_is_placeholder_notimplemented(self, tmp_path: Path) -> None:
        """Gate fails when a milestone test raises NotImplementedError."""
        doc_path = tmp_path / "DESIGN_DOCUMENT.md"
        _write_design_doc(
            doc_path,
            "M0",
            {"REQ-M0-001": ["tests/test_example.py::test_not_impl"]},
        )

        # Write a NotImplementedError test
        test_content = dedent(
            """\
            def test_not_impl():
                raise NotImplementedError
            """
        )
        _write_test_file(tmp_path / "tests" / "test_example.py", test_content)

        result = verify._gate_no_skipped_tests(tmp_path, "M0")
        assert not result.passed
        assert "placeholder" in result.note.lower() or "1" in result.note

    def test_fails_when_test_is_placeholder_ellipsis(self, tmp_path: Path) -> None:
        """Gate fails when a milestone test body is just '...'."""
        doc_path = tmp_path / "DESIGN_DOCUMENT.md"
        _write_design_doc(
            doc_path,
            "M0",
            {"REQ-M0-001": ["tests/test_example.py::test_ellipsis"]},
        )

        # Write an ellipsis placeholder test
        test_content = dedent(
            """\
            def test_ellipsis():
                ...
            """
        )
        _write_test_file(tmp_path / "tests" / "test_example.py", test_content)

        result = verify._gate_no_skipped_tests(tmp_path, "M0")
        assert not result.passed
        assert "placeholder" in result.note.lower() or "1" in result.note

    def test_only_checks_current_milestone_tests(self, tmp_path: Path) -> None:
        """Gate only checks tests for the current milestone, not other milestones."""
        doc_path = tmp_path / "DESIGN_DOCUMENT.md"
        _write_design_doc(
            doc_path,
            "M0",
            {
                "REQ-M0-001": ["tests/test_m0.py::test_active"],
                "REQ-M1-001": ["tests/test_m1.py::test_placeholder"],  # Different milestone
            },
        )

        # Write an active M0 test
        test_content_m0 = dedent(
            """\
            def test_active():
                assert True
            """
        )
        _write_test_file(tmp_path / "tests" / "test_m0.py", test_content_m0)

        # Write a placeholder M1 test (should be ignored when checking M0)
        test_content_m1 = dedent(
            """\
            def test_placeholder():
                pass
            """
        )
        _write_test_file(tmp_path / "tests" / "test_m1.py", test_content_m1)

        # Checking M0 should pass (the M1 placeholder is ignored)
        result = verify._gate_no_skipped_tests(tmp_path, "M0")
        assert result.passed

    def test_passes_when_no_tests_for_milestone(self, tmp_path: Path) -> None:
        """Gate passes when there are no tests mapped to the current milestone."""
        doc_path = tmp_path / "DESIGN_DOCUMENT.md"
        _write_design_doc(
            doc_path,
            "M0",
            {"REQ-M1-001": ["tests/test_m1.py::test_something"]},  # Only M1 tests
        )

        result = verify._gate_no_skipped_tests(tmp_path, "M0")
        assert result.passed
        assert "no tests mapped" in result.note


class TestIsPlaceholderTest:
    """Tests for _is_placeholder_test helper."""

    def test_detects_pass_placeholder(self, tmp_path: Path) -> None:
        """Detects a test that only contains 'pass'."""
        test_content = dedent(
            """\
            def test_placeholder():
                pass
            """
        )
        _write_test_file(tmp_path / "tests" / "test_example.py", test_content)

        result = verify._is_placeholder_test(tmp_path, "tests/test_example.py::test_placeholder")
        assert result is True

    def test_detects_notimplemented_placeholder(self, tmp_path: Path) -> None:
        """Detects a test that raises NotImplementedError."""
        test_content = dedent(
            """\
            def test_not_impl():
                raise NotImplementedError
            """
        )
        _write_test_file(tmp_path / "tests" / "test_example.py", test_content)

        result = verify._is_placeholder_test(tmp_path, "tests/test_example.py::test_not_impl")
        assert result is True

    def test_detects_ellipsis_placeholder(self, tmp_path: Path) -> None:
        """Detects a test that only contains '...'."""
        test_content = dedent(
            """\
            def test_ellipsis():
                ...
            """
        )
        _write_test_file(tmp_path / "tests" / "test_example.py", test_content)

        result = verify._is_placeholder_test(tmp_path, "tests/test_example.py::test_ellipsis")
        assert result is True

    def test_does_not_flag_real_test(self, tmp_path: Path) -> None:
        """Does not flag a test with real assertions."""
        test_content = dedent(
            """\
            def test_real():
                x = 1 + 1
                assert x == 2
            """
        )
        _write_test_file(tmp_path / "tests" / "test_example.py", test_content)

        result = verify._is_placeholder_test(tmp_path, "tests/test_example.py::test_real")
        assert result is False

    def test_handles_missing_file(self, tmp_path: Path) -> None:
        """Returns False for missing test files."""
        result = verify._is_placeholder_test(tmp_path, "tests/nonexistent.py::test_foo")
        assert result is False

    def test_handles_invalid_node_id(self, tmp_path: Path) -> None:
        """Returns False for invalid node IDs."""
        result = verify._is_placeholder_test(tmp_path, "invalid_node_id")
        assert result is False


class TestVerifyIntegration:
    """Integration tests for verify.main with the no_skipped_tests gate."""

    def test_verify_fails_with_placeholder_m0_test(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """verify.main fails when M0 has placeholder tests."""
        doc_path = tmp_path / "DESIGN_DOCUMENT.md"
        _write_design_doc(
            doc_path,
            "M0",
            {"REQ-M0-001": ["tests/test_example.py::test_placeholder"]},
        )

        # Write a placeholder test
        test_content = dedent(
            """\
            def test_placeholder():
                pass
            """
        )
        _write_test_file(tmp_path / "tests" / "test_example.py", test_content)

        # Mock _run to avoid running actual pytest/m0 commands
        def fake_run(cmd: list[str], cwd: Path, *, timeout_s: int | None = None) -> verify.GateResult:
            return verify.GateResult(name="fake", passed=True, cmd=cmd)

        monkeypatch.setattr(verify, "_run", fake_run)

        rc = verify.main(
            [
                "--project-root",
                str(tmp_path),
                "--skip-pytest",
                "--skip-quality",
                "--skip-git",
            ]
        )
        # Should fail due to placeholder test
        assert rc == 2

    def test_verify_passes_with_active_m0_test(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """verify.main passes when M0 tests are all active."""
        doc_path = tmp_path / "DESIGN_DOCUMENT.md"
        _write_design_doc(
            doc_path,
            "M0",
            {"REQ-M0-001": ["tests/test_example.py::test_active"]},
        )

        # Write an active test
        test_content = dedent(
            """\
            def test_active():
                assert True
            """
        )
        _write_test_file(tmp_path / "tests" / "test_example.py", test_content)

        # Mock _run to avoid running actual pytest/m0 commands
        def fake_run(cmd: list[str], cwd: Path, *, timeout_s: int | None = None) -> verify.GateResult:
            return verify.GateResult(name="fake", passed=True, cmd=cmd)

        monkeypatch.setattr(verify, "_run", fake_run)

        rc = verify.main(
            [
                "--project-root",
                str(tmp_path),
                "--skip-pytest",
                "--skip-quality",
                "--skip-git",
            ]
        )
        assert rc == 0

    def test_skip_no_skipped_tests_flag(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """--skip-no-skipped-tests skips the gate."""
        doc_path = tmp_path / "DESIGN_DOCUMENT.md"
        _write_design_doc(
            doc_path,
            "M0",
            {"REQ-M0-001": ["tests/test_example.py::test_placeholder"]},
        )

        # Write a placeholder test
        test_content = dedent(
            """\
            def test_placeholder():
                pass
            """
        )
        _write_test_file(tmp_path / "tests" / "test_example.py", test_content)

        # Mock _run to avoid running actual pytest/m0 commands
        def fake_run(cmd: list[str], cwd: Path, *, timeout_s: int | None = None) -> verify.GateResult:
            return verify.GateResult(name="fake", passed=True, cmd=cmd)

        monkeypatch.setattr(verify, "_run", fake_run)

        rc = verify.main(
            [
                "--project-root",
                str(tmp_path),
                "--skip-pytest",
                "--skip-quality",
                "--skip-git",
                "--skip-no-skipped-tests",
            ]
        )
        # Should pass because the gate is skipped
        assert rc == 0
