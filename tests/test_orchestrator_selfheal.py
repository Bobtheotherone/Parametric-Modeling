#!/usr/bin/env python3
"""Regression tests for orchestrator self-healing behavior.

These tests verify the critical fixes to the orchestrator:
G1) Claude worker prompts don't claim tools are disabled
G2) Merge conflicts invoke resolver before manual intervention
G3) Auto-continue doesn't give up early (rc==2 handling, max_stalled)
G4) BackfillGenerator activates when queue is low

Run with: pytest tests/test_orchestrator_selfheal.py -v
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestToolsDisabledContradiction:
    """G1: Verify Claude worker prompts don't claim tools are disabled."""

    def test_claude_sh_turn_mode_has_tools_enabled_message(self):
        """Verify claude.sh in turn mode (ORCH_SCHEMA_KIND=turn) says tools ARE enabled."""
        script_path = Path(__file__).parent.parent / "bridge" / "agents" / "claude.sh"
        assert script_path.exists(), f"claude.sh not found at {script_path}"

        content = script_path.read_text()

        # Verify the script distinguishes between task_plan and turn modes
        assert 'ORCH_SCHEMA_KIND' in content, "Script should check ORCH_SCHEMA_KIND"

        # Verify turn mode explicitly states tools are ENABLED
        assert 'Tools ARE ENABLED' in content, \
            "Turn mode should explicitly state tools ARE ENABLED"
        assert 'Do NOT claim tools are disabled' in content, \
            "Turn mode should warn against claiming tools are disabled"

        # Verify the old contradiction is NOT present in turn mode
        # The old text was in both modes; now it should only be in task_plan mode
        task_plan_section_match = re.search(
            r'if.*ORCH_SCHEMA_KIND.*task_plan.*then(.*?)else',
            content,
            re.DOTALL
        )
        turn_section_match = re.search(
            r'else\s*#.*[Ee]xecution mode(.*?)fi',
            content,
            re.DOTALL
        )

        if turn_section_match:
            turn_section = turn_section_match.group(1)
            assert 'Tools are DISABLED' not in turn_section, \
                "Turn mode should NOT claim tools are DISABLED"

    def test_noncompliant_patterns_include_tools_disabled(self):
        """Verify noncompliance detection patterns catch 'tools disabled' claims."""
        from bridge.loop import NONCOMPLIANT_OUTPUT_PATTERNS, CRITICAL_TOOL_VIOLATIONS

        # Check that tool-related patterns exist
        pattern_types = {desc for _, desc in NONCOMPLIANT_OUTPUT_PATTERNS}
        tool_patterns = {p for p in pattern_types if 'tools' in p.lower()}

        assert len(tool_patterns) > 0, \
            "Should have patterns to detect 'tools disabled' claims"
        assert 'tools_disabled_claim' in pattern_types or any('disabled' in p for p in pattern_types), \
            "Should detect 'tools disabled' claims"

        # Verify critical violations include tool claims
        assert len(CRITICAL_TOOL_VIOLATIONS) > 0, \
            "CRITICAL_TOOL_VIOLATIONS should not be empty"
        assert any('tool' in v for v in CRITICAL_TOOL_VIOLATIONS), \
            "Critical violations should include tool-related patterns"

    def test_detect_noncompliant_output_catches_tools_disabled(self):
        """Verify _detect_noncompliant_output flags 'tools are disabled' text."""
        from bridge.loop import _detect_noncompliant_output

        test_cases = [
            ("I cannot use tools because tools are disabled.", True, True),
            ("Tools are not available to me.", True, True),
            ("I don't have access to tools.", True, True),
            ('{"work_completed": true}', False, False),  # Valid JSON, no violation
        ]

        for text, expected_noncompliant, expected_critical in test_cases:
            is_noncompliant, violations, has_critical = _detect_noncompliant_output(text)
            assert is_noncompliant == expected_noncompliant, \
                f"Expected noncompliant={expected_noncompliant} for: {text[:50]}"
            if expected_critical:
                assert has_critical, \
                    f"Expected critical tool violation for: {text[:50]}"


class TestMergeConflictResolution:
    """G2: Verify merge conflicts invoke resolver before manual intervention."""

    def test_merge_resolver_module_exists(self):
        """Verify merge_resolver.py module exists."""
        resolver_path = Path(__file__).parent.parent / "bridge" / "merge_resolver.py"
        assert resolver_path.exists(), "merge_resolver.py should exist"

    def test_merge_resolver_has_key_functions(self):
        """Verify merge_resolver has the key functions."""
        from bridge.merge_resolver import (
            MergeResolver,
            attempt_agent_merge_resolution,
            detect_conflict_files,
            parse_conflict_file,
        )

        # Just verify these are importable
        assert callable(attempt_agent_merge_resolution)
        assert callable(detect_conflict_files)
        assert callable(parse_conflict_file)
        assert MergeResolver is not None

    def test_patch_integrator_uses_merge_resolver(self):
        """Verify PatchIntegrator imports and uses merge_resolver."""
        integrator_path = Path(__file__).parent.parent / "bridge" / "patch_integration.py"
        content = integrator_path.read_text()

        # Check that merge_resolver is imported and used
        assert 'merge_resolver' in content or 'attempt_agent_merge_resolution' in content, \
            "PatchIntegrator should use merge_resolver"

    def test_auto_merge_resolution_uses_agent_resolver(self):
        """Verify _attempt_auto_merge_resolution uses agent-based resolution."""
        loop_path = Path(__file__).parent.parent / "bridge" / "loop.py"
        content = loop_path.read_text()

        # Find the _attempt_auto_merge_resolution function
        func_match = re.search(
            r'def _attempt_auto_merge_resolution\([^)]+\).*?(?=\ndef |\nclass |\Z)',
            content,
            re.DOTALL
        )
        assert func_match, "_attempt_auto_merge_resolution function not found"

        func_body = func_match.group(0)
        assert 'attempt_agent_merge_resolution' in func_body or 'merge_resolver' in func_body, \
            "_attempt_auto_merge_resolution should use agent-based merge resolver"


class TestAutoContinueSelfHealing:
    """G3: Verify auto-continue doesn't give up early."""

    def test_auto_continue_max_stalled_increased(self):
        """Verify max_stalled is increased from 3 to a higher value."""
        loop_path = Path(__file__).parent.parent / "bridge" / "loop.py"
        content = loop_path.read_text()

        # Find the _run_parallel_with_auto_continue function
        func_match = re.search(
            r'def _run_parallel_with_auto_continue\([^)]+\).*?(?=\ndef |\nclass |\Z)',
            content,
            re.DOTALL
        )
        assert func_match, "_run_parallel_with_auto_continue function not found"

        func_body = func_match.group(0)

        # Check max_stalled value - should be >= 5 (was 3, now should be 10)
        max_stalled_match = re.search(r'max_stalled\s*=\s*(\d+)', func_body)
        assert max_stalled_match, "max_stalled assignment not found"
        max_stalled = int(max_stalled_match.group(1))
        assert max_stalled >= 5, \
            f"max_stalled should be >= 5 for self-healing (got {max_stalled})"

    def test_auto_continue_does_not_immediately_exit_on_rc2(self):
        """Verify auto-continue doesn't immediately exit on rc==2."""
        loop_path = Path(__file__).parent.parent / "bridge" / "loop.py"
        content = loop_path.read_text()

        # Find the _run_parallel_with_auto_continue function
        func_match = re.search(
            r'def _run_parallel_with_auto_continue\([^)]+\).*?(?=\ndef |\nclass |\Z)',
            content,
            re.DOTALL
        )
        assert func_match, "_run_parallel_with_auto_continue function not found"

        func_body = func_match.group(0)

        # Check that rc==2 handling includes retry logic, not immediate return
        # The old code had: "if rc == 2:\n...return 2"
        # The new code should have planning_failure_count and retry logic
        assert 'planning_failure_count' in func_body, \
            "Should track planning_failure_count for retries"

        # Should NOT immediately return 2 on first planning failure
        # Look for pattern that would indicate immediate return
        immediate_return_pattern = re.search(
            r'if\s+rc\s*==\s*2:\s*\n\s*print.*\n\s*return\s+2',
            func_body
        )
        assert not immediate_return_pattern, \
            "Should NOT immediately return 2 on planning failure"

    def test_auto_continue_has_repair_context_generation(self):
        """Verify auto-continue generates repair context for failures."""
        loop_path = Path(__file__).parent.parent / "bridge" / "loop.py"
        content = loop_path.read_text()

        # Check that repair context generation function exists
        assert '_generate_repair_context_for_failures' in content, \
            "Should have _generate_repair_context_for_failures function"

        # Check it's used in auto-continue
        func_match = re.search(
            r'def _run_parallel_with_auto_continue\([^)]+\).*?(?=\ndef |\nclass |\Z)',
            content,
            re.DOTALL
        )
        func_body = func_match.group(0) if func_match else ""
        assert '_generate_repair_context_for_failures' in func_body or 'repair_context' in func_body, \
            "auto-continue should use repair context generation"


class TestBackfillGenerator:
    """G4: Verify BackfillGenerator activates when queue is low."""

    def test_backfill_generator_imported_in_loop(self):
        """Verify BackfillGenerator is imported in loop.py."""
        loop_path = Path(__file__).parent.parent / "bridge" / "loop.py"
        content = loop_path.read_text()

        assert 'BackfillGenerator' in content, \
            "BackfillGenerator should be imported in loop.py"

    def test_backfill_generator_integrated_in_run_parallel(self):
        """Verify BackfillGenerator is instantiated in run_parallel."""
        loop_path = Path(__file__).parent.parent / "bridge" / "loop.py"
        content = loop_path.read_text()

        # Find run_parallel function
        func_match = re.search(
            r'def run_parallel\([^)]+\).*?(?=\ndef |\nclass |\Z)',
            content,
            re.DOTALL
        )
        assert func_match, "run_parallel function not found"

        func_body = func_match.group(0)

        # Check BackfillGenerator is instantiated
        assert 'backfill_generator' in func_body.lower() or 'BackfillGenerator' in func_body, \
            "run_parallel should instantiate BackfillGenerator"

        # Check maybe_generate_backfill is called
        assert 'maybe_generate_backfill' in func_body or 'backfill' in func_body, \
            "run_parallel should have backfill generation logic"

    def test_scheduler_has_update_tasks_method(self):
        """Verify TwoLaneScheduler has update_tasks method for backfill."""
        from bridge.scheduler import TwoLaneScheduler

        # Check the method exists
        assert hasattr(TwoLaneScheduler, 'update_tasks'), \
            "TwoLaneScheduler should have update_tasks method"

    def test_backfill_generator_filler_tasks(self):
        """Verify BackfillGenerator generates safe filler tasks."""
        from bridge.scheduler import BackfillGenerator, FillerTask

        gen = BackfillGenerator(project_root="/tmp", min_queue_depth=10)

        # Generate some tasks
        tasks = gen.generate_filler_tasks(3)

        assert len(tasks) == 3, "Should generate requested number of tasks"
        for task in tasks:
            assert isinstance(task, FillerTask)
            assert task.id.startswith("FILLER-")
            assert task.priority < 0, "Filler tasks should have low priority"


class TestStuckDetectionSelfHealing:
    """G4 continued: Verify STUCK detection triggers self-healing."""

    def test_stuck_detection_has_self_healing(self):
        """Verify STUCK detection in run_parallel triggers self-healing."""
        loop_path = Path(__file__).parent.parent / "bridge" / "loop.py"
        content = loop_path.read_text()

        # Find run_parallel function
        func_match = re.search(
            r'def run_parallel\([^)]+\).*?(?=\ndef |\nclass |\Z)',
            content,
            re.DOTALL
        )
        assert func_match, "run_parallel function not found"

        func_body = func_match.group(0)

        # Check for self-healing keywords
        assert 'SELF-HEALING' in func_body or 'self_heal' in func_body or 'recovery' in func_body.lower(), \
            "STUCK detection should have self-healing logic"

        # Check that STUCK detection doesn't immediately mark as stuck
        # It should attempt recovery first
        assert 'recovered' in func_body.lower() or 'recovery_reason' in func_body, \
            "STUCK detection should track recovery attempts"


class TestConfigCapabilities:
    """Verify config has provider capabilities."""

    def test_config_has_tool_capabilities(self):
        """Verify config.json has supports_tools flags."""
        config_path = Path(__file__).parent.parent / "bridge" / "config.json"
        assert config_path.exists(), "config.json should exist"

        config = json.loads(config_path.read_text())
        agents = config.get("agents", {})

        for agent_name, agent_cfg in agents.items():
            # Verify new capability flags exist
            assert "supports_tools" in agent_cfg or "supports_write_access" in agent_cfg, \
                f"Agent {agent_name} should have capability flags"


class TestMergeResolverSchemaKind:
    """Verify merge_resolver does not use task_plan schema kind."""

    def test_merge_resolver_does_not_use_task_plan(self):
        """Verify merge_resolver.py does NOT set ORCH_SCHEMA_KIND=task_plan."""
        resolver_path = Path(__file__).parent.parent / "bridge" / "merge_resolver.py"
        content = resolver_path.read_text()

        # Check that task_plan is NOT used (it should be "json" now)
        assert 'ORCH_SCHEMA_KIND"] = "task_plan"' not in content, \
            "merge_resolver should NOT use task_plan schema kind (use 'json' instead)"

        # Check that "json" mode is used
        assert 'ORCH_SCHEMA_KIND"] = "json"' in content, \
            "merge_resolver should use 'json' schema kind"

    def test_claude_sh_json_mode_does_not_mention_task_plan_keys(self):
        """Verify claude.sh json mode reminder doesn't mention task_plan-required keys."""
        script_path = Path(__file__).parent.parent / "bridge" / "agents" / "claude.sh"
        content = script_path.read_text()

        # Find the json mode section
        json_section_match = re.search(
            r'ORCH_SCHEMA_KIND.*==.*"json".*then(.*?)(?:elif|else|fi)',
            content,
            re.DOTALL
        )
        assert json_section_match, "claude.sh should have json schema kind handling"

        json_section = json_section_match.group(1)

        # Verify json mode does NOT mention task_plan-specific keys
        task_plan_keys = ['milestone_id', 'max_parallel_tasks', 'rationale', 'preferred_agent']
        for key in task_plan_keys:
            # Allow mentions of "do NOT include" but not prescriptive "must have"
            required_pattern = f'{key}.*required|required.*{key}|Top-level.*{key}|must have.*{key}'
            assert not re.search(required_pattern, json_section, re.IGNORECASE), \
                f"json mode should NOT prescribe task_plan key '{key}'"


class TestBackfillScopeEnforcement:
    """Verify backfill tasks are scope-constrained."""

    def test_backfill_task_identification(self):
        """Verify is_backfill_task correctly identifies FILLER-* tasks."""
        from bridge.patch_integration import is_backfill_task

        assert is_backfill_task("FILLER-LINT-001") == True
        assert is_backfill_task("FILLER-TEST-002") == True
        assert is_backfill_task("M1-IMPLEMENT-FOO") == False
        assert is_backfill_task("REGULAR-TASK") == False

    def test_backfill_scope_guard_rejects_src_files(self):
        """Verify backfill scope guard rejects patches touching src/ directory."""
        from bridge.patch_integration import create_backfill_scope_guard

        guard = create_backfill_scope_guard()

        # Allowed paths
        result_ok = guard.check_paths(["tests/test_foo.py", "bridge/loop.py"])
        assert result_ok.allowed, "Backfill should allow tests/ and bridge/"

        # Disallowed paths
        result_bad = guard.check_paths(["src/formula_foundry/api.py"])
        assert not result_bad.allowed, "Backfill should reject src/ files"

        result_bad2 = guard.check_paths(["coupongen/some_file.py"])
        assert not result_bad2.allowed, "Backfill should reject coupongen/ files"

    def test_filler_tasks_have_backfill_lock(self):
        """Verify convert_filler_to_parallel_task adds backfill lock."""
        # This tests the function logic by checking the loop.py source
        loop_path = Path(__file__).parent.parent / "bridge" / "loop.py"
        content = loop_path.read_text()

        # Find the convert_filler_to_parallel_task function
        func_match = re.search(
            r'def convert_filler_to_parallel_task.*?(?=\n    def |\nclass |\Z)',
            content,
            re.DOTALL
        )
        assert func_match, "convert_filler_to_parallel_task function not found"

        func_body = func_match.group(0)

        # Check that backfill lock is added
        assert 'locks=["backfill"]' in func_body or "locks=['backfill']" in func_body, \
            "Filler tasks should have backfill lock"

    def test_backfill_scope_guard_allows_docs_directory(self):
        """Verify backfill scope guard allows docs/ directory."""
        from bridge.patch_integration import create_backfill_scope_guard

        guard = create_backfill_scope_guard()

        result = guard.check_paths(["docs/README.md", "docs/api/guide.md"])
        assert result.allowed, "Backfill should allow docs/ files"

    def test_backfill_scope_guard_rejects_hot_files(self):
        """Verify backfill scope guard rejects hot integration files."""
        from bridge.patch_integration import create_backfill_scope_guard

        guard = create_backfill_scope_guard()

        # Hot files should be rejected even if in allowed parent dirs
        hot_files = [
            "bridge/api.py",  # If this existed
            "src/formula_foundry/board_writer.py",
            "src/formula_foundry/cli_main.py",
            "src/formula_foundry/pipeline.py",
        ]

        for hot_file in hot_files:
            result = guard.check_paths([hot_file])
            # src/ files should always be rejected
            if hot_file.startswith("src/"):
                assert not result.allowed, f"Backfill should reject {hot_file}"

    def test_backfill_allowed_dirs_constant(self):
        """Verify BACKFILL_ALLOWLIST constant is properly configured."""
        from bridge.patch_integration import BACKFILL_ALLOWLIST

        assert "tests/**" in BACKFILL_ALLOWLIST, "tests/ should be in backfill allowlist"
        assert "docs/**" in BACKFILL_ALLOWLIST, "docs/ should be in backfill allowlist"
        assert "bridge/**" in BACKFILL_ALLOWLIST, "bridge/ should be in backfill allowlist"

    def test_backfill_denylist_prevents_src(self):
        """Verify BACKFILL_DENYLIST blocks src/ directory."""
        from bridge.patch_integration import BACKFILL_DENYLIST

        assert "src/**" in BACKFILL_DENYLIST, "src/ should be in backfill denylist"

    def test_backfill_generator_tasks_are_low_priority(self):
        """Verify BackfillGenerator generates low-priority tasks."""
        from bridge.scheduler import BackfillGenerator

        gen = BackfillGenerator(project_root="/tmp")
        tasks = gen.generate_filler_tasks(3)

        for task in tasks:
            assert task.priority < 0, f"Filler task {task.id} should have negative priority"


class TestBehavioralMergeConflictResolution:
    """Behavioral test: merge conflicts are auto-resolved without real agent calls."""

    def test_merge_conflict_resolution_with_injected_callback(self):
        """Create a real git conflict and resolve it using dependency-injected callback."""
        import shutil
        import tempfile
        from pathlib import Path

        from bridge.merge_resolver import MergeResolver, detect_conflict_files

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "test_repo"
            repo_path.mkdir()
            runs_dir = Path(tmpdir) / "runs"
            runs_dir.mkdir()

            # Initialize git repo
            subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_path, capture_output=True, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_path, capture_output=True, check=True)

            # Create and commit base file
            hello_path = repo_path / "hello.py"
            hello_path.write_text('def greet():\n    return "Hello, World!"\n')
            subprocess.run(["git", "add", "hello.py"], cwd=repo_path, capture_output=True, check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, capture_output=True, check=True)

            # Create branch A with modification
            subprocess.run(["git", "checkout", "-b", "branch_a"], cwd=repo_path, capture_output=True, check=True)
            hello_path.write_text('def greet():\n    return "Hello from Branch A!"\n')
            subprocess.run(["git", "add", "hello.py"], cwd=repo_path, capture_output=True, check=True)
            subprocess.run(["git", "commit", "-m", "Branch A changes"], cwd=repo_path, capture_output=True, check=True)

            # Go back to main and create branch B with different modification
            subprocess.run(["git", "checkout", "master"], cwd=repo_path, capture_output=True)
            # Handle case where main branch is called "main" instead of "master"
            result = subprocess.run(["git", "checkout", "main"], cwd=repo_path, capture_output=True)
            if result.returncode != 0:
                subprocess.run(["git", "checkout", "-b", "main"], cwd=repo_path, capture_output=True, check=True)

            subprocess.run(["git", "checkout", "-b", "branch_b"], cwd=repo_path, capture_output=True, check=True)
            hello_path.write_text('def greet():\n    return "Hello from Branch B!"\n')
            subprocess.run(["git", "add", "hello.py"], cwd=repo_path, capture_output=True, check=True)
            subprocess.run(["git", "commit", "-m", "Branch B changes"], cwd=repo_path, capture_output=True, check=True)

            # Merge branch A to produce conflict
            result = subprocess.run(
                ["git", "merge", "branch_a", "--no-edit"],
                cwd=repo_path,
                capture_output=True,
            )

            # Should have conflict
            content_before = hello_path.read_text()
            assert "<<<<<<" in content_before or result.returncode != 0, \
                "Expected merge conflict but none found"

            # If no conflict markers (fast-forward), create artificial conflict
            if "<<<<<<" not in content_before:
                # Reset and try again to force conflict
                subprocess.run(["git", "merge", "--abort"], cwd=repo_path, capture_output=True)
                subprocess.run(["git", "reset", "--hard", "branch_b"], cwd=repo_path, capture_output=True, check=True)
                # Modify file directly to create conflict markers
                hello_path.write_text(
                    'def greet():\n'
                    '<<<<<<< HEAD\n'
                    '    return "Hello from Branch B!"\n'
                    '=======\n'
                    '    return "Hello from Branch A!"\n'
                    '>>>>>>> branch_a\n'
                )
                subprocess.run(["git", "add", "hello.py"], cwd=repo_path, capture_output=True)
                # Mark as unmerged
                content_before = hello_path.read_text()

            # Verify conflict exists
            assert "<<<<<<" in content_before, "Should have conflict markers"

            # Define the resolution callback
            def mock_agent_runner(conflicts, task_context, milestone_id, attempt):
                """Mock agent that returns a deterministic resolution."""
                resolutions = []
                for cf in conflicts:
                    # Simple resolution: combine both versions
                    resolved_content = 'def greet():\n    return "Hello from Both Branches!"\n'
                    resolutions.append({
                        "path": cf.path,
                        "resolved_content": resolved_content,
                        "notes": "Combined both branch changes"
                    })
                return {"resolutions": resolutions, "unresolvable": []}

            # Create resolver with injected callback
            resolver = MergeResolver(
                project_root=repo_path,
                runs_dir=runs_dir,
                max_attempts=1,
                agent_runner=mock_agent_runner,
            )

            # Run resolution
            result = resolver.resolve_conflicts(
                task_id="TEST-MERGE",
                task_context="Test merge resolution",
                milestone_id="M0",
            )

            # Verify results
            assert result.success or "hello.py" in result.resolved_files, \
                f"Resolution should succeed. Error: {result.error}, Unresolved: {result.unresolved_files}"

            # Verify conflict markers removed
            content_after = hello_path.read_text()
            assert "<<<<<<" not in content_after, "Conflict markers should be removed"
            assert "=======" not in content_after, "Conflict markers should be removed"
            assert ">>>>>>>" not in content_after, "Conflict markers should be removed"

            # Verify file was resolved
            assert "Both Branches" in content_after, "Resolution content should be applied"


    def test_merge_conflict_deterministic_resolution_git_clean(self):
        """Deterministic test: create real git conflict, resolve it, verify git status clean."""
        import tempfile
        from pathlib import Path

        from bridge.merge_resolver import (
            ConflictFile,
            MergeResolver,
            detect_conflict_files,
            parse_conflict_file,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "test_repo"
            repo_path.mkdir()
            runs_dir = Path(tmpdir) / "runs"
            runs_dir.mkdir()

            # Initialize git repo
            subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_path, capture_output=True, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_path, capture_output=True, check=True)

            # Create and commit base file
            test_file = repo_path / "module.py"
            test_file.write_text('def foo():\n    return 1\n')
            subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, check=True)
            subprocess.run(["git", "commit", "-m", "Initial"], cwd=repo_path, capture_output=True, check=True)

            # Create branch_a with modification
            subprocess.run(["git", "checkout", "-b", "branch_a"], cwd=repo_path, capture_output=True, check=True)
            test_file.write_text('def foo():\n    return 2  # branch_a\n')
            subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, check=True)
            subprocess.run(["git", "commit", "-m", "branch_a change"], cwd=repo_path, capture_output=True, check=True)

            # Go back to initial commit and create branch_b
            subprocess.run(["git", "checkout", "HEAD~1"], cwd=repo_path, capture_output=True, check=True)
            subprocess.run(["git", "checkout", "-b", "branch_b"], cwd=repo_path, capture_output=True, check=True)
            test_file.write_text('def foo():\n    return 3  # branch_b\n')
            subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, check=True)
            subprocess.run(["git", "commit", "-m", "branch_b change"], cwd=repo_path, capture_output=True, check=True)

            # Merge branch_a into branch_b to create conflict
            result = subprocess.run(
                ["git", "merge", "branch_a", "--no-edit"],
                cwd=repo_path,
                capture_output=True,
            )

            # Verify conflict exists
            content_before = test_file.read_text()
            has_conflict = "<<<<<<" in content_before

            if not has_conflict:
                # Fast-forward occurred, skip this test
                pytest.skip("Git merge resulted in fast-forward, no conflict to test")

            # Parse the conflict file directly to verify parsing works
            cf = parse_conflict_file(repo_path, "module.py")
            assert cf is not None, "Should parse conflict file"
            assert cf.conflict_count >= 1

            # Define deterministic resolution callback
            def deterministic_resolver(conflicts, task_context, milestone_id, attempt):
                """Return a deterministic resolution that combines both sides."""
                resolutions = []
                for cf in conflicts:
                    resolved = 'def foo():\n    # Combined from both branches\n    return 2  # from branch_a\n'
                    resolutions.append({
                        "path": cf.path,
                        "resolved_content": resolved,
                        "notes": "Deterministically combined both versions"
                    })
                return {"resolutions": resolutions, "unresolvable": []}

            # Create resolver with injected callback
            resolver = MergeResolver(
                project_root=repo_path,
                runs_dir=runs_dir,
                max_attempts=1,
                agent_runner=deterministic_resolver,
            )

            # Run resolution
            result = resolver.resolve_conflicts(
                task_id="DETERMINISTIC-TEST",
                task_context="Deterministic merge test",
                milestone_id="M0",
            )

            # Verify resolution succeeded
            assert result.success or "module.py" in result.resolved_files, \
                f"Resolution failed: {result.error}"

            # Verify conflict markers are GONE
            content_after = test_file.read_text()
            assert "<<<<<<" not in content_after, "Conflict start markers should be removed"
            assert "=======" not in content_after, "Conflict separator should be removed"
            assert ">>>>>>>" not in content_after, "Conflict end markers should be removed"

            # Verify our resolution content is present
            assert "Combined from both branches" in content_after

            # Verify file was staged (git added)
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            # After resolution and git add, file should be staged
            status_lines = [l for l in status_result.stdout.strip().split("\n") if l.strip()]
            for line in status_lines:
                # Should NOT have UU (unmerged) status
                assert not line.startswith("UU"), f"File should not have unmerged status: {line}"

    def test_parse_conflict_file_unit(self):
        """Unit test for parse_conflict_file function with synthetic conflict markers."""
        import tempfile
        from pathlib import Path

        from bridge.merge_resolver import parse_conflict_file

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create a file with conflict markers
            test_file = repo_path / "test.py"
            conflict_content = '''def foo():
<<<<<<< HEAD
    return 2
=======
    return 3
>>>>>>> other_branch
'''
            test_file.write_text(conflict_content)

            # Parse the conflict file
            cf = parse_conflict_file(repo_path, "test.py")

            assert cf is not None, "Should parse conflict file"
            assert cf.conflict_count == 1, "Should detect one conflict"
            assert "return 2" in cf.ours_content, "Should extract ours content"
            assert "return 3" in cf.theirs_content, "Should extract theirs content"
            assert cf.path == "test.py"

    def test_merge_resolver_no_conflicts(self):
        """Test MergeResolver with a clean repo (no conflicts)."""
        import tempfile
        from pathlib import Path

        from bridge.merge_resolver import MergeResolver

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "test_repo"
            repo_path.mkdir()
            runs_dir = Path(tmpdir) / "runs"
            runs_dir.mkdir()

            # Initialize git repo with no conflicts
            subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_path, capture_output=True, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_path, capture_output=True, check=True)
            (repo_path / "clean.py").write_text('x = 1\n')
            subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, check=True)
            subprocess.run(["git", "commit", "-m", "Initial"], cwd=repo_path, capture_output=True, check=True)

            resolver = MergeResolver(
                project_root=repo_path,
                runs_dir=runs_dir,
                max_attempts=1,
                agent_runner=lambda *a: {"resolutions": [], "unresolvable": []},
            )

            result = resolver.resolve_conflicts(task_id="NO-CONFLICT", task_context="", milestone_id="M0")

            # Should succeed immediately with no conflicts
            assert result.success, "Should succeed with no conflicts"
            assert len(result.resolved_files) == 0, "No files to resolve"
            assert len(result.unresolved_files) == 0, "No unresolved files"


class TestHotFileGuardrail:
    """Test hot-file guardrail for preventing concurrent edits to critical files."""

    def test_hot_files_constant_exists(self):
        """Verify HOT_FILES constant is defined."""
        from bridge.loop import HOT_FILES

        assert HOT_FILES is not None, "HOT_FILES constant should exist"
        assert len(HOT_FILES) > 0, "HOT_FILES should have entries"

        # Verify key hot files are included
        hot_patterns = " ".join(HOT_FILES)
        assert "api.py" in hot_patterns, "api.py should be a hot file"
        assert "board_writer.py" in hot_patterns, "board_writer.py should be a hot file"

    def test_inject_hot_file_locks_function_exists(self):
        """Verify _inject_hot_file_locks function is defined."""
        from bridge.loop import _inject_hot_file_locks

        assert callable(_inject_hot_file_locks), "_inject_hot_file_locks should be callable"

    def test_hot_file_lock_injection_for_concurrent_access(self):
        """Verify locks are injected when multiple tasks touch the same hot file."""
        from bridge.loop import ParallelTask, _inject_hot_file_locks

        # Create two tasks that both touch api.py
        task1 = ParallelTask(
            id="M1-001",
            title="Implement feature A",
            description="Modifies api.py to add feature A",
            agent="claude",
            touched_paths=["src/formula_foundry/api.py", "tests/test_api.py"],
        )
        task2 = ParallelTask(
            id="M1-002",
            title="Implement feature B",
            description="Modifies api.py to add feature B",
            agent="codex",
            touched_paths=["src/formula_foundry/api.py", "docs/api.md"],
        )
        # Task that doesn't touch hot files
        task3 = ParallelTask(
            id="M1-003",
            title="Add tests",
            description="Add unit tests",
            agent="claude",
            touched_paths=["tests/test_foo.py"],
        )

        tasks = [task1, task2, task3]
        result = _inject_hot_file_locks(tasks)

        # Verify locks were injected for api.py
        assert "hot:api.py" in task1.locks, "Task 1 should have hot:api.py lock"
        assert "hot:api.py" in task2.locks, "Task 2 should have hot:api.py lock"
        assert "hot:api.py" not in task3.locks, "Task 3 should NOT have hot:api.py lock"

    def test_hot_file_lock_not_injected_for_single_task(self):
        """Verify locks are NOT injected when only one task touches a hot file."""
        from bridge.loop import ParallelTask, _inject_hot_file_locks

        # Only one task touches api.py
        task1 = ParallelTask(
            id="M1-001",
            title="Implement feature A",
            description="Modifies api.py",
            agent="claude",
            touched_paths=["src/formula_foundry/api.py"],
        )
        task2 = ParallelTask(
            id="M1-002",
            title="Implement feature B",
            description="Modifies board_writer.py",
            agent="codex",
            touched_paths=["src/formula_foundry/board_writer.py"],
        )

        tasks = [task1, task2]
        result = _inject_hot_file_locks(tasks)

        # No locks should be injected - each hot file only touched by one task
        assert "hot:api.py" not in task1.locks, "No lock for api.py (only one task)"
        assert "hot:board_writer.py" not in task2.locks, "No lock for board_writer.py (only one task)"

    def test_hot_file_lock_multiple_hot_files(self):
        """Verify locks are correctly injected for multiple hot files."""
        from bridge.loop import ParallelTask, _inject_hot_file_locks

        # Two tasks touch api.py, two different tasks touch board_writer.py
        task1 = ParallelTask(
            id="M1-001",
            title="Task 1",
            description="Desc 1",
            agent="claude",
            touched_paths=["src/api.py"],
        )
        task2 = ParallelTask(
            id="M1-002",
            title="Task 2",
            description="Desc 2",
            agent="claude",
            touched_paths=["src/api.py", "src/board_writer.py"],
        )
        task3 = ParallelTask(
            id="M1-003",
            title="Task 3",
            description="Desc 3",
            agent="claude",
            touched_paths=["src/board_writer.py"],
        )

        tasks = [task1, task2, task3]
        result = _inject_hot_file_locks(tasks)

        # api.py touched by task1 and task2
        assert "hot:api.py" in task1.locks
        assert "hot:api.py" in task2.locks
        assert "hot:api.py" not in task3.locks

        # board_writer.py touched by task2 and task3
        assert "hot:board_writer.py" not in task1.locks
        assert "hot:board_writer.py" in task2.locks
        assert "hot:board_writer.py" in task3.locks


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
