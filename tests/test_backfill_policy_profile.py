#!/usr/bin/env python3
"""Unit tests for profile-based backfill policy.

Tests verify that:
1. Throughput profile disables low-ROI task types (docs, type_hints) by default
2. Default profile allows all task types
3. Profile policy is correctly applied during task type selection

Run with: pytest tests/test_backfill_policy_profile.py -v
"""

from __future__ import annotations

import pytest
from bridge.scheduler import BackfillGenerator


class TestThroughputProfilePolicy:
    """Test throughput profile backfill restrictions."""

    def test_throughput_profile_disables_docs(self, tmp_path):
        """Verify throughput profile disables docs task type."""
        generator = BackfillGenerator(
            project_root=str(tmp_path),
            min_queue_depth=100,
            planner_profile="throughput",
        )

        # docs should be on cooldown due to profile policy
        assert generator.is_type_on_cooldown("docs")

    def test_throughput_profile_disables_type_hints(self, tmp_path):
        """Verify throughput profile disables type_hints task type."""
        generator = BackfillGenerator(
            project_root=str(tmp_path),
            min_queue_depth=100,
            planner_profile="throughput",
        )

        # type_hints should be on cooldown due to profile policy
        assert generator.is_type_on_cooldown("type_hints")

    def test_throughput_profile_allows_test(self, tmp_path):
        """Verify throughput profile allows test task type (high ROI)."""
        generator = BackfillGenerator(
            project_root=str(tmp_path),
            min_queue_depth=100,
            planner_profile="throughput",
        )

        # test should NOT be on cooldown
        assert not generator.is_type_on_cooldown("test")

    def test_throughput_profile_allows_lint(self, tmp_path):
        """Verify throughput profile allows lint task type (high ROI)."""
        generator = BackfillGenerator(
            project_root=str(tmp_path),
            min_queue_depth=100,
            planner_profile="throughput",
        )

        # lint should NOT be on cooldown
        assert not generator.is_type_on_cooldown("lint")

    def test_throughput_generated_tasks_exclude_docs(self, tmp_path):
        """Verify throughput profile does not generate docs tasks."""
        generator = BackfillGenerator(
            project_root=str(tmp_path),
            min_queue_depth=100,
            planner_profile="throughput",
        )

        # Generate many tasks
        tasks = generator.generate_filler_tasks(20)

        # None should be docs type
        docs_tasks = [t for t in tasks if t.task_type == "docs"]
        assert len(docs_tasks) == 0, f"Expected no docs tasks, got {docs_tasks}"

    def test_throughput_generated_tasks_exclude_type_hints(self, tmp_path):
        """Verify throughput profile does not generate type_hints tasks."""
        generator = BackfillGenerator(
            project_root=str(tmp_path),
            min_queue_depth=100,
            planner_profile="throughput",
        )

        # Generate many tasks
        tasks = generator.generate_filler_tasks(20)

        # None should be type_hints type
        type_hints_tasks = [t for t in tasks if t.task_type == "type_hints"]
        assert len(type_hints_tasks) == 0, f"Expected no type_hints tasks, got {type_hints_tasks}"


class TestDefaultProfilePolicy:
    """Test default/balanced profile allows all task types."""

    def test_default_profile_allows_docs(self, tmp_path):
        """Verify default profile allows docs task type."""
        generator = BackfillGenerator(
            project_root=str(tmp_path),
            min_queue_depth=100,
            planner_profile="default",
        )

        # docs should NOT be on cooldown with default profile
        assert not generator.is_type_on_cooldown("docs")

    def test_default_profile_allows_type_hints(self, tmp_path):
        """Verify default profile allows type_hints task type."""
        generator = BackfillGenerator(
            project_root=str(tmp_path),
            min_queue_depth=100,
            planner_profile="default",
        )

        # type_hints should NOT be on cooldown with default profile
        assert not generator.is_type_on_cooldown("type_hints")

    def test_balanced_profile_allows_all(self, tmp_path):
        """Verify balanced profile allows all task types."""
        generator = BackfillGenerator(
            project_root=str(tmp_path),
            min_queue_depth=100,
            planner_profile="balanced",
        )

        # All types should be allowed
        for task_type in ["lint", "test", "docs", "type_hints", "schema_lint"]:
            assert not generator.is_type_on_cooldown(task_type), f"{task_type} should not be on cooldown"


class TestEngineeringProfilePolicy:
    """Test engineering profile disables all backfill generation."""

    def test_engineering_profile_disables_generation(self, tmp_path):
        """Verify engineering profile disables backfill generation."""
        generator = BackfillGenerator(
            project_root=str(tmp_path),
            min_queue_depth=100,
            planner_profile="engineering",
        )

        assert not generator.should_generate(current_queue_depth=0, worker_count=4)
        assert generator.generate_filler_tasks(5) == []

    def test_engineering_profile_disables_all_types(self, tmp_path):
        """Verify engineering profile treats all task types as disabled."""
        generator = BackfillGenerator(
            project_root=str(tmp_path),
            min_queue_depth=100,
            planner_profile="engineering",
        )

        for task_type in ["lint", "test", "docs", "type_hints", "schema_lint"]:
            assert generator.is_type_on_cooldown(task_type), f"{task_type} should be disabled in engineering profile"


class TestProfileDisabledTypesConfig:
    """Test profile disabled types configuration."""

    def test_throughput_disabled_types_set(self):
        """Verify THROUGHPUT_PROFILE_DISABLED_TYPES contains expected types."""
        disabled = BackfillGenerator.THROUGHPUT_PROFILE_DISABLED_TYPES
        assert "docs" in disabled
        assert "type_hints" in disabled
        # High-ROI types should NOT be disabled
        assert "lint" not in disabled
        assert "test" not in disabled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
