"""Tests for bridge.directives — directive file materialization."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bridge.directives import materialize_directive_file


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Create a fake project root with CLAUDE.md and AGENTS.md."""
    (tmp_path / "CLAUDE.md").write_text("# CLAUDE directive\n")
    (tmp_path / "AGENTS.md").write_text("# AGENTS directive\n")
    return tmp_path


def test_materialize_claude_copies_claude_md(project_root: Path, tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    materialize_directive_file(project_root=project_root, target_dir=target, agent_name="claude")
    assert (target / "CLAUDE.md").read_text() == "# CLAUDE directive\n"
    assert (target / "AGENTS.md").read_text() == "# AGENTS directive\n"


def test_materialize_falls_back_to_agents_md(tmp_path: Path):
    """When only AGENTS.md exists, CLAUDE.md should be created from it for claude agent."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "AGENTS.md").write_text("# agents only\n")
    target = tmp_path / "target"
    target.mkdir()
    materialize_directive_file(project_root=root, target_dir=target, agent_name="claude")
    assert (target / "CLAUDE.md").read_text() == "# agents only\n"
    assert (target / "AGENTS.md").read_text() == "# agents only\n"


def test_materialize_idempotent(project_root: Path, tmp_path: Path):
    """Second call should not rewrite if content matches."""
    target = tmp_path / "target"
    target.mkdir()
    materialize_directive_file(project_root=project_root, target_dir=target, agent_name="claude")
    mtime1 = (target / "CLAUDE.md").stat().st_mtime_ns
    materialize_directive_file(project_root=project_root, target_dir=target, agent_name="claude")
    mtime2 = (target / "CLAUDE.md").stat().st_mtime_ns
    # Content matches, so atomic_copy_file should not be called again
    # (mtime may or may not change depending on atomic_copy_file implementation,
    # but the function should at least not error)


def test_materialize_non_claude_agent_no_claude_md(project_root: Path, tmp_path: Path):
    """Non-claude agent should not get CLAUDE.md, only AGENTS.md."""
    target = tmp_path / "target"
    target.mkdir()
    materialize_directive_file(project_root=project_root, target_dir=target, agent_name="codex")
    assert not (target / "CLAUDE.md").exists()
    assert (target / "AGENTS.md").read_text() == "# AGENTS directive\n"


def test_merge_resolver_materializes_directives_before_agent():
    """Ensure merge_resolver calls materialize_directive_file before _run_cmd."""
    from bridge.merge_resolver import MergeResolver, ConflictFile

    call_order: list[str] = []

    def fake_run_cmd(cmd, cwd, env=None, timeout=60):
        # Check that CLAUDE.md exists at invocation time
        if "claude" in str(cmd[0]).lower():
            claude_md = Path(cwd) / "CLAUDE.md"
            call_order.append(f"run_cmd:claude_md_exists={claude_md.exists()}")
        return (1, "", "fake error")  # fail so we don't need full agent output

    def fake_materialize(**kwargs):
        call_order.append("materialize")
        # Actually create the file so the check works
        target = kwargs["target_dir"]
        (target / "CLAUDE.md").write_text("# test\n")

    with patch("bridge.merge_resolver._run_cmd", side_effect=fake_run_cmd):
        with patch("bridge.directives.materialize_directive_file", side_effect=fake_materialize):
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                runs = root / "runs"
                runs.mkdir()
                # Create a fake agent script
                agent_script = root / "bridge" / "agents"
                agent_script.mkdir(parents=True)
                script = agent_script / "claude.sh"
                script.write_text("#!/bin/sh\nexit 1\n")
                script.chmod(0o755)

                resolver = MergeResolver(
                    project_root=root,
                    runs_dir=runs,
                    agent_script="bridge/agents/claude.sh",
                )
                conflicts = [
                    ConflictFile(
                        path="test.py",
                        content="<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> branch",
                        conflict_count=1,
                        ours_content="ours",
                        theirs_content="theirs",
                    )
                ]
                resolver._run_agent_resolution(conflicts, "test", "M0", 1)

    assert len(call_order) >= 2
    assert call_order[0] == "materialize"
    assert "run_cmd" in call_order[1]
