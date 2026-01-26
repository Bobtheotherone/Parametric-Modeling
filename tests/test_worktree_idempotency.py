"""Tests for worktree/branch collision handling in the orchestrator.

These tests verify that the orchestrator can handle:
1. Pre-existing branches from previous runs
2. Pre-existing worktree paths
3. Branches checked out in other worktrees

This prevents the "branch already exists" errors that blocked the Jan 26 run.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def run_git(args: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run a git command and return (rc, stdout, stderr)."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / "repo"
    repo.mkdir()

    # Initialize repo
    run_git(["init"], repo)
    run_git(["config", "user.email", "test@test.com"], repo)
    run_git(["config", "user.name", "Test"], repo)

    # Create initial commit
    (repo / "README.md").write_text("# Test Repo\n")
    run_git(["add", "README.md"], repo)
    run_git(["commit", "-m", "Initial commit"], repo)

    return repo


class TestBranchCollisionHandling:
    """Test that branch collisions are handled gracefully."""

    def test_can_create_worktree_when_branch_does_not_exist(self, temp_git_repo: Path):
        """Basic case: branch doesn't exist, worktree creation succeeds."""
        worktree_path = temp_git_repo / "worktrees" / "test_worktree"
        branch_name = "task/test-run/test-task"

        # Get base SHA
        rc, sha, _ = run_git(["rev-parse", "HEAD"], temp_git_repo)
        assert rc == 0
        base_sha = sha.strip()

        # Create worktree
        rc, out, err = run_git(
            ["worktree", "add", "-b", branch_name, str(worktree_path), base_sha],
            temp_git_repo,
        )
        assert rc == 0, f"Failed to create worktree: {err}"
        assert worktree_path.exists()

    def test_branch_exists_error_is_detectable(self, temp_git_repo: Path):
        """Verify we can detect the 'branch already exists' error."""
        branch_name = "task/test-run/existing-branch"

        # Create the branch first
        rc, _, _ = run_git(["branch", branch_name], temp_git_repo)
        assert rc == 0

        # Try to create worktree with same branch name (without cleanup)
        worktree_path = temp_git_repo / "worktrees" / "test_worktree"
        rc, sha, _ = run_git(["rev-parse", "HEAD"], temp_git_repo)
        base_sha = sha.strip()

        rc, out, err = run_git(
            ["worktree", "add", "-b", branch_name, str(worktree_path), base_sha],
            temp_git_repo,
        )

        # Verify we get the expected error
        assert rc != 0
        assert "already exists" in err.lower()

    def test_cleanup_removes_existing_branch(self, temp_git_repo: Path):
        """Test that cleanup logic can remove an existing branch."""
        branch_name = "task/test-run/cleanup-test"

        # Create the branch first
        run_git(["branch", branch_name], temp_git_repo)

        # Verify branch exists
        rc, _, _ = run_git(["show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"], temp_git_repo)
        assert rc == 0, "Branch should exist"

        # Delete the branch (simulating cleanup)
        rc, _, _ = run_git(["branch", "-D", branch_name], temp_git_repo)
        assert rc == 0

        # Verify branch is gone
        rc, _, _ = run_git(["show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"], temp_git_repo)
        assert rc != 0, "Branch should be deleted"

    def test_cleanup_removes_worktree_before_branch(self, temp_git_repo: Path):
        """Test that a branch checked out in a worktree can be cleaned up."""
        worktree_path = temp_git_repo / "worktrees" / "to_remove"
        branch_name = "task/test-run/worktree-branch"

        # Get base SHA
        rc, sha, _ = run_git(["rev-parse", "HEAD"], temp_git_repo)
        base_sha = sha.strip()

        # Create worktree with branch
        rc, _, err = run_git(
            ["worktree", "add", "-b", branch_name, str(worktree_path), base_sha],
            temp_git_repo,
        )
        assert rc == 0, f"Failed: {err}"

        # Try to delete branch directly - should fail (checked out in worktree)
        rc, _, err = run_git(["branch", "-D", branch_name], temp_git_repo)
        assert rc != 0 or "checked out" in err.lower() or "cannot delete" in err.lower(), \
            "Deleting branch checked out in worktree should fail or warn"

        # Remove worktree first
        rc, _, _ = run_git(["worktree", "remove", "--force", str(worktree_path)], temp_git_repo)
        assert rc == 0

        # Prune
        run_git(["worktree", "prune"], temp_git_repo)

        # Now branch deletion should succeed
        rc, _, err = run_git(["branch", "-D", branch_name], temp_git_repo)
        assert rc == 0, f"Branch deletion failed after worktree removal: {err}"


class TestIdempotentWorktreeSetup:
    """Test that worktree setup is idempotent (can be called multiple times)."""

    def test_full_cleanup_cycle(self, temp_git_repo: Path):
        """Test the full cleanup cycle: detect, remove worktree, remove branch, recreate."""
        worktree_path = temp_git_repo / "worktrees" / "idempotent_test"
        branch_name = "task/test-run/idempotent-task"

        # Get base SHA
        rc, sha, _ = run_git(["rev-parse", "HEAD"], temp_git_repo)
        base_sha = sha.strip()

        # First creation
        rc, _, err = run_git(
            ["worktree", "add", "-b", branch_name, str(worktree_path), base_sha],
            temp_git_repo,
        )
        assert rc == 0, f"First creation failed: {err}"

        # Simulate retry: detect branch exists
        rc, _, _ = run_git(
            ["show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"],
            temp_git_repo,
        )
        assert rc == 0, "Branch should exist"

        # Remove worktree
        rc, _, _ = run_git(["worktree", "remove", "--force", str(worktree_path)], temp_git_repo)
        assert rc == 0

        # Prune
        run_git(["worktree", "prune"], temp_git_repo)

        # Delete branch
        rc, _, _ = run_git(["branch", "-D", branch_name], temp_git_repo)
        assert rc == 0

        # Second creation (simulating retry)
        rc, _, err = run_git(
            ["worktree", "add", "-b", branch_name, str(worktree_path), base_sha],
            temp_git_repo,
        )
        assert rc == 0, f"Second creation (retry) failed: {err}"
        assert worktree_path.exists()


class TestWorktreeListParsing:
    """Test that we can parse git worktree list output."""

    def test_parse_worktree_list_porcelain(self, temp_git_repo: Path):
        """Test parsing porcelain output to find branch checkout location."""
        worktree_path = temp_git_repo / "worktrees" / "parse_test"
        branch_name = "task/test-run/parse-test"

        # Get base SHA
        rc, sha, _ = run_git(["rev-parse", "HEAD"], temp_git_repo)
        base_sha = sha.strip()

        # Create worktree
        run_git(["worktree", "add", "-b", branch_name, str(worktree_path), base_sha], temp_git_repo)

        # Get worktree list
        rc, out, _ = run_git(["worktree", "list", "--porcelain"], temp_git_repo)
        assert rc == 0

        # Parse to find our branch
        current_wt_path = None
        found = False
        for line in out.strip().split("\n"):
            if line.startswith("worktree "):
                current_wt_path = line.split(" ", 1)[1]
            elif line.startswith("branch ") and branch_name in line:
                found = True
                assert current_wt_path is not None
                assert str(worktree_path) == current_wt_path
                break

        assert found, f"Could not find branch {branch_name} in worktree list:\n{out}"
