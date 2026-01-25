# SPDX-License-Identifier: MIT
"""Unit tests for tools/git_guard.py main() and helper functions.

This module tests functionality not covered by test_git_guard.py and
test_git_guard_extras.py:
- main() function with various scenarios
- _run helper function
- _git_root function
- _git_ls_files function
- _tracked_files function
- Integration scenarios
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tools import git_guard


class TestRunHelper:
    """Tests for _run helper function."""

    def test_successful_command(self, tmp_path: Path) -> None:
        """Successful command returns rc=0 and output."""
        rc, out, err = git_guard._run(["echo", "hello"], tmp_path)
        assert rc == 0
        assert "hello" in out
        assert err == ""

    def test_failed_command(self, tmp_path: Path) -> None:
        """Failed command returns non-zero rc."""
        rc, out, err = git_guard._run(["false"], tmp_path)
        assert rc != 0

    def test_command_with_stderr(self, tmp_path: Path) -> None:
        """Command with stderr captures error output."""
        # Use bash to write to stderr for cross-platform compatibility
        rc, out, err = git_guard._run(
            ["bash", "-c", "echo error >&2"],
            tmp_path,
        )
        assert rc == 0
        assert "error" in err


class TestGitRoot:
    """Tests for _git_root function."""

    def test_in_git_repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns git root when in a git repository."""
        # Initialize a git repo in tmp_path
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)

        monkeypatch.chdir(tmp_path)
        root = git_guard._git_root()
        assert root.exists()
        assert (root / ".git").exists()

    def test_not_in_git_repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises RuntimeError when not in a git repository."""
        # Create a non-git directory
        non_git_dir = tmp_path / "non_git"
        non_git_dir.mkdir()
        monkeypatch.chdir(non_git_dir)

        with pytest.raises(RuntimeError, match="Not a git repo"):
            git_guard._git_root()


class TestGitLsFiles:
    """Tests for _git_ls_files function."""

    def test_empty_repo(self, tmp_path: Path) -> None:
        """Empty repo returns empty list."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
        files = git_guard._git_ls_files(tmp_path)
        assert files == []

    def test_repo_with_files(self, tmp_path: Path) -> None:
        """Repo with tracked files returns file list."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)

        # Create and track files
        (tmp_path / "file1.py").write_text("# file1", encoding="utf-8")
        (tmp_path / "file2.py").write_text("# file2", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)

        files = git_guard._git_ls_files(tmp_path)
        assert "file1.py" in files
        assert "file2.py" in files


class TestTrackedFiles:
    """Tests for _tracked_files function."""

    def test_returns_path_objects(self, tmp_path: Path) -> None:
        """Returns list of Path objects."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)

        (tmp_path / "test.py").write_text("# test", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)

        files = git_guard._tracked_files(tmp_path)
        assert len(files) == 1
        assert isinstance(files[0], Path)
        assert files[0].name == "test.py"

    def test_excludes_deleted_files(self, tmp_path: Path) -> None:
        """Excludes files that were deleted but still in index."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)

        # Create, track, then delete a file
        test_file = tmp_path / "deleted.py"
        test_file.write_text("# deleted", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
        test_file.unlink()

        files = git_guard._tracked_files(tmp_path)
        # Should not include the deleted file (only existing files)
        assert not any(f.name == "deleted.py" for f in files)


class TestMainFunction:
    """Tests for main() function."""

    def test_clean_repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clean repo without issues returns 0."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)

        # Create a clean file
        (tmp_path / "clean.py").write_text("# No secrets here\nprint('hello')\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=str(tmp_path),
            check=True,
            capture_output=True,
            env={
                "GIT_AUTHOR_NAME": "Test",
                "GIT_AUTHOR_EMAIL": "test@test.com",
                "GIT_COMMITTER_NAME": "Test",
                "GIT_COMMITTER_EMAIL": "test@test.com",
            },
        )

        monkeypatch.chdir(tmp_path)
        rc = git_guard.main()
        assert rc == 0

    def test_repo_with_secrets(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        """Repo with secrets returns 2."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)

        # Create a file with a secret
        secret_file = tmp_path / "secret.py"
        secret_file.write_text("key = 'sk-abc123def456ghi789jkl0123456789'\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)

        monkeypatch.chdir(tmp_path)
        rc = git_guard.main()
        assert rc == 2

        captured = capsys.readouterr()
        assert "secret" in captured.out.lower() or "OpenAI-style key" in captured.out

    def test_repo_with_unignored_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Repo with unignored .env returns 2."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)

        # Create .env file (not ignored)
        (tmp_path / ".env").write_text("SECRET=value\n", encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        rc = git_guard.main()
        assert rc == 2

        captured = capsys.readouterr()
        assert ".env" in captured.out

    def test_repo_with_ignored_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Repo with properly ignored .env returns 0."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)

        # Create .gitignore and .env
        (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
        (tmp_path / ".env").write_text("SECRET=value\n", encoding="utf-8")
        subprocess.run(["git", "add", ".gitignore"], cwd=str(tmp_path), check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add gitignore"],
            cwd=str(tmp_path),
            check=True,
            capture_output=True,
            env={
                "GIT_AUTHOR_NAME": "Test",
                "GIT_AUTHOR_EMAIL": "test@test.com",
                "GIT_COMMITTER_NAME": "Test",
                "GIT_COMMITTER_EMAIL": "test@test.com",
            },
        )

        monkeypatch.chdir(tmp_path)
        rc = git_guard.main()
        assert rc == 0

    def test_repo_with_egg_info(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Repo with tracked .egg-info returns 2."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)

        # Create tracked .egg-info directory
        egg_dir = tmp_path / "mypackage.egg-info"
        egg_dir.mkdir()
        (egg_dir / "PKG-INFO").write_text("Name: mypackage\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)

        monkeypatch.chdir(tmp_path)
        rc = git_guard.main()
        assert rc == 2

        captured = capsys.readouterr()
        assert "egg-info" in captured.out.lower()

    def test_not_in_git_repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        """Not in git repo returns 3."""
        # Create a non-git directory
        non_git_dir = tmp_path / "non_git"
        non_git_dir.mkdir()
        monkeypatch.chdir(non_git_dir)

        rc = git_guard.main()
        assert rc == 3

        captured = capsys.readouterr()
        assert "error" in captured.err.lower() or "git" in captured.err.lower()


class TestSecretPatternMatching:
    """Additional tests for secret pattern edge cases."""

    def test_openai_key_in_different_contexts(self, tmp_path: Path) -> None:
        """OpenAI keys detected in various code contexts."""
        test_file = tmp_path / "test.py"

        # Test as string assignment
        test_file.write_text(
            'OPENAI_KEY = "sk-abc123def456ghi789jkl0123456789"',
            encoding="utf-8",
        )
        hits = git_guard._scan_file(test_file)
        assert "OpenAI-style key" in hits

        # Test in dict
        test_file.write_text(
            "config = {'key': 'sk-abc123def456ghi789jkl0123456789'}",
            encoding="utf-8",
        )
        hits = git_guard._scan_file(test_file)
        assert "OpenAI-style key" in hits

    def test_anthropic_key_variations(self, tmp_path: Path) -> None:
        """Anthropic keys with different formats are detected."""
        test_file = tmp_path / "test.py"

        # Standard format
        test_file.write_text(
            "key = 'sk-ant-api-key-12345678901234567890'",
            encoding="utf-8",
        )
        hits = git_guard._scan_file(test_file)
        assert "Anthropic key" in hits

    def test_google_api_key_variations(self, tmp_path: Path) -> None:
        """Google API keys are detected."""
        test_file = tmp_path / "test.py"

        test_file.write_text(
            "GOOGLE_KEY = 'AIzaSyB1234567890abcdefghij'",
            encoding="utf-8",
        )
        hits = git_guard._scan_file(test_file)
        assert "Google API key" in hits

    def test_generic_api_key_assignment(self, tmp_path: Path) -> None:
        """Generic API_KEY assignments are detected."""
        test_file = tmp_path / "test.py"

        test_file.write_text(
            'MY_API_KEY = "super_secret_value_12345"',
            encoding="utf-8",
        )
        hits = git_guard._scan_file(test_file)
        assert "Generic API_KEY assignment" in hits

    def test_no_false_positives_for_comments(self, tmp_path: Path) -> None:
        """Example keys in comments may still be detected (intentional)."""
        test_file = tmp_path / "test.py"

        # Even in comments, we want to flag potential secrets
        test_file.write_text(
            "# Example: sk-abc123def456ghi789jkl0123456789",
            encoding="utf-8",
        )
        hits = git_guard._scan_file(test_file)
        # This is intentionally detected - better safe than sorry
        assert "OpenAI-style key" in hits
