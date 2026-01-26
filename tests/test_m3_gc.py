"""Tests for the M3 garbage collection system with retention policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from formula_foundry.m3.artifact_store import ArtifactStore
from formula_foundry.m3.gc import (
    BUILTIN_POLICIES,
    GarbageCollector,
    GCCandidate,
    GCResult,
    PinnedArtifact,
    PolicyNotFoundError,
    RetentionPolicy,
    format_bytes,
    load_pins_from_file,
    save_pins_to_file,
)
from formula_foundry.m3.lineage_graph import LineageGraph
from formula_foundry.m3.registry import ArtifactRegistry

if TYPE_CHECKING:
    pass


class TestRetentionPolicy:
    """Tests for RetentionPolicy dataclass."""

    def test_to_dict(self) -> None:
        policy = RetentionPolicy(
            name="test_policy",
            description="A test policy",
            keep_min_age_days=7,
            keep_min_count=3,
            keep_pinned=True,
            keep_with_descendants=True,
            keep_artifact_types=["touchstone"],
            keep_roles=["final_output"],
            space_budget_bytes=1000000000,
        )
        result = policy.to_dict()

        assert result["name"] == "test_policy"
        assert result["description"] == "A test policy"
        assert result["keep_min_age_days"] == 7
        assert result["keep_min_count"] == 3
        assert result["keep_pinned"] is True
        assert result["keep_with_descendants"] is True
        assert result["keep_artifact_types"] == ["touchstone"]
        assert result["keep_roles"] == ["final_output"]
        assert result["space_budget_bytes"] == 1000000000

    def test_from_dict(self) -> None:
        data = {
            "name": "from_dict_policy",
            "keep_min_age_days": 14,
            "keep_min_count": 5,
        }
        policy = RetentionPolicy.from_dict(data)

        assert policy.name == "from_dict_policy"
        assert policy.keep_min_age_days == 14
        assert policy.keep_min_count == 5
        # Defaults
        assert policy.keep_pinned is True
        assert policy.keep_with_descendants is True

    def test_builtin_policies_exist(self) -> None:
        """Verify all built-in policies are defined."""
        assert "laptop_default" in BUILTIN_POLICIES
        assert "ci_aggressive" in BUILTIN_POLICIES
        assert "archive" in BUILTIN_POLICIES
        assert "dev_minimal" in BUILTIN_POLICIES


class TestPinnedArtifact:
    """Tests for PinnedArtifact dataclass."""

    def test_to_dict_artifact_id(self) -> None:
        pin = PinnedArtifact(artifact_id="art-001", reason="important")
        result = pin.to_dict()

        assert result["artifact_id"] == "art-001"
        assert result["reason"] == "important"
        assert "run_id" not in result  # Not set

    def test_to_dict_run_id(self) -> None:
        pin = PinnedArtifact(run_id="run-001", reason="good run")
        result = pin.to_dict()

        assert result["run_id"] == "run-001"
        assert "artifact_id" not in result

    def test_from_dict(self) -> None:
        data = {"dataset_id": "ds-001", "reason": "production dataset"}
        pin = PinnedArtifact.from_dict(data)

        assert pin.dataset_id == "ds-001"
        assert pin.reason == "production dataset"


class TestPinsFileIO:
    """Tests for pins file read/write functions."""

    def test_save_and_load_pins(self, tmp_path: Path) -> None:
        pins_file = tmp_path / "pins.json"
        pins = [
            PinnedArtifact(artifact_id="art-001", reason="test"),
            PinnedArtifact(run_id="run-001", reason="test run"),
        ]

        save_pins_to_file(pins_file, pins)
        loaded = load_pins_from_file(pins_file)

        assert len(loaded) == 2
        assert loaded[0].artifact_id == "art-001"
        assert loaded[1].run_id == "run-001"

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        pins_file = tmp_path / "nonexistent.json"
        loaded = load_pins_from_file(pins_file)
        assert loaded == []


class TestGarbageCollector:
    """Tests for GarbageCollector class."""

    @pytest.fixture
    def gc_setup(self, tmp_path: Path):
        """Set up a GC environment with store, registry, and test artifacts."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "objects").mkdir()
        (data_dir / "manifests").mkdir()

        store = ArtifactStore(
            root=data_dir,
            generator="test",
            generator_version="1.0.0",
        )
        store._ensure_dirs()

        registry_db = data_dir / "registry.db"
        registry = ArtifactRegistry(registry_db)
        registry.initialize()

        lineage_db = data_dir / "lineage.sqlite"
        lineage = LineageGraph(lineage_db)
        lineage.initialize()

        gc = GarbageCollector(
            data_dir=data_dir,
            store=store,
            registry=registry,
            lineage=lineage,
        )

        return {
            "data_dir": data_dir,
            "store": store,
            "registry": registry,
            "lineage": lineage,
            "gc": gc,
        }

    def test_get_policy_builtin(self, gc_setup) -> None:
        gc = gc_setup["gc"]
        policy = gc.get_policy("laptop_default")
        assert policy.name == "laptop_default"

    def test_get_policy_not_found(self, gc_setup) -> None:
        gc = gc_setup["gc"]
        with pytest.raises(PolicyNotFoundError):
            gc.get_policy("nonexistent_policy")

    def test_pin_and_unpin_artifact(self, gc_setup) -> None:
        gc = gc_setup["gc"]

        # Pin
        pin = gc.pin_artifact(artifact_id="art-001", reason="test")
        assert pin.artifact_id == "art-001"
        assert gc.is_pinned("art-001")

        # Unpin
        removed = gc.unpin_artifact(artifact_id="art-001")
        assert removed is True
        assert not gc.is_pinned("art-001")

    def test_pin_by_run_id(self, gc_setup) -> None:
        gc = gc_setup["gc"]

        gc.pin_artifact(run_id="run-001", reason="test run")
        assert gc.is_pinned("any-artifact", run_id="run-001")
        assert not gc.is_pinned("any-artifact", run_id="run-002")

    def test_compute_candidates_empty_store(self, gc_setup) -> None:
        gc = gc_setup["gc"]

        to_delete, to_keep = gc.compute_candidates("laptop_default")
        assert len(to_delete) == 0
        assert len(to_keep) == 0

    def test_compute_candidates_with_artifacts(self, gc_setup) -> None:
        store = gc_setup["store"]
        registry = gc_setup["registry"]
        gc = gc_setup["gc"]

        # Add a recent artifact (should be kept)
        manifest1 = store.put(
            content=b"recent content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        registry.index_artifact(manifest1)

        to_delete, to_keep = gc.compute_candidates("laptop_default")

        # Recent artifact should be kept
        assert len(to_keep) == 1
        assert to_keep[0].artifact_id == manifest1.artifact_id

    def test_pinned_artifact_protected(self, gc_setup) -> None:
        store = gc_setup["store"]
        registry = gc_setup["registry"]
        gc = gc_setup["gc"]

        # Add an artifact
        manifest = store.put(
            content=b"pinned content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        registry.index_artifact(manifest)

        # Pin it
        gc.pin_artifact(artifact_id=manifest.artifact_id, reason="important")

        # Use a policy with very short retention
        policy = RetentionPolicy(
            name="aggressive",
            keep_min_age_days=0,  # Would delete everything by age
            keep_min_count=0,  # Would delete everything by count
            keep_pinned=True,
        )

        to_delete, to_keep = gc.compute_candidates(policy)

        # Should be kept because it's pinned
        assert len(to_keep) == 1
        assert "pinned" in to_keep[0].reasons_to_keep

    def test_protected_artifact_type(self, gc_setup) -> None:
        store = gc_setup["store"]
        registry = gc_setup["registry"]
        gc = gc_setup["gc"]

        # Add a dataset_snapshot artifact
        manifest = store.put(
            content=b"dataset content",
            artifact_type="dataset_snapshot",
            roles=["final_output"],
            run_id="run-001",
        )
        registry.index_artifact(manifest)

        # laptop_default protects dataset_snapshot
        to_delete, to_keep = gc.compute_candidates("laptop_default")

        assert len(to_keep) == 1
        assert any("protected_type" in r for r in to_keep[0].reasons_to_keep)

    def test_run_dry_run(self, gc_setup) -> None:
        store = gc_setup["store"]
        registry = gc_setup["registry"]
        gc = gc_setup["gc"]

        # Add an artifact
        manifest = store.put(
            content=b"test content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        registry.index_artifact(manifest)

        # Run GC in dry-run mode
        result = gc.run(policy="laptop_default", dry_run=True, run_dvc_gc=False)

        assert result.dry_run is True
        assert result.artifacts_scanned >= 1
        # Artifact should still exist (dry run)
        assert store.exists_by_id(manifest.artifact_id)

    def test_estimate_savings(self, gc_setup) -> None:
        store = gc_setup["store"]
        registry = gc_setup["registry"]
        gc = gc_setup["gc"]

        # Add an artifact
        manifest = store.put(
            content=b"test content for estimate",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        registry.index_artifact(manifest)

        estimate = gc.estimate_savings("laptop_default")

        assert estimate["total_artifacts"] >= 1
        assert estimate["policy"] == "laptop_default"
        assert "bytes_to_delete" in estimate
        assert "bytes_to_keep" in estimate


class TestGCResult:
    """Tests for GCResult dataclass."""

    def test_to_dict(self) -> None:
        result = GCResult(
            policy_name="test",
            started_utc="2025-01-01T00:00:00Z",
            finished_utc="2025-01-01T00:01:00Z",
            dry_run=True,
            artifacts_scanned=100,
            artifacts_deleted=10,
            bytes_freed=1000000,
            bytes_total_before=10000000,
            bytes_total_after=9000000,
            pinned_protected=5,
            descendant_protected=3,
            ancestor_protected=2,
            dvc_gc_ran=False,
            dvc_gc_output=None,
        )

        d = result.to_dict()
        assert d["policy_name"] == "test"
        assert d["artifacts_deleted"] == 10
        assert d["bytes_freed"] == 1000000
        assert d["ancestor_protected"] == 2

    def test_to_json(self) -> None:
        result = GCResult(
            policy_name="test",
            started_utc="2025-01-01T00:00:00Z",
            finished_utc="2025-01-01T00:01:00Z",
            dry_run=True,
            artifacts_scanned=100,
            artifacts_deleted=10,
            bytes_freed=1000000,
            bytes_total_before=10000000,
            bytes_total_after=9000000,
            pinned_protected=5,
            descendant_protected=3,
            ancestor_protected=2,
            dvc_gc_ran=False,
            dvc_gc_output=None,
        )

        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["policy_name"] == "test"
        assert parsed["ancestor_protected"] == 2


class TestFormatBytes:
    """Tests for the format_bytes utility function."""

    def test_format_bytes(self) -> None:
        assert "B" in format_bytes(100)
        assert "KB" in format_bytes(2000)
        assert "MB" in format_bytes(2000000)
        assert "GB" in format_bytes(2000000000)
        assert "TB" in format_bytes(2000000000000)


class TestGCCandidateShouldDelete:
    """Tests for GCCandidate.should_delete property."""

    def test_should_delete_true(self) -> None:
        candidate = GCCandidate(
            artifact_id="test",
            content_hash_digest="abc",
            byte_size=100,
            created_utc="2025-01-01T00:00:00Z",
            artifact_type="other",
            run_id="run-001",
            storage_path=None,
            reasons_to_delete=["age:100d >= 30d"],
            reasons_to_keep=[],
        )
        assert candidate.should_delete is True

    def test_should_delete_false_has_keep_reason(self) -> None:
        candidate = GCCandidate(
            artifact_id="test",
            content_hash_digest="abc",
            byte_size=100,
            created_utc="2025-01-01T00:00:00Z",
            artifact_type="other",
            run_id="run-001",
            storage_path=None,
            reasons_to_delete=["age:100d >= 30d"],
            reasons_to_keep=["pinned"],
        )
        assert candidate.should_delete is False

    def test_should_delete_false_no_delete_reasons(self) -> None:
        candidate = GCCandidate(
            artifact_id="test",
            content_hash_digest="abc",
            byte_size=100,
            created_utc="2025-01-01T00:00:00Z",
            artifact_type="other",
            run_id="run-001",
            storage_path=None,
            reasons_to_delete=[],
            reasons_to_keep=[],
        )
        assert candidate.should_delete is False


class TestAncestorProtection:
    """Tests for ancestor-of-pinned protection in garbage collection."""

    @pytest.fixture
    def gc_with_lineage(self, tmp_path: Path):
        """Set up a GC environment with store, registry, lineage, and test artifacts."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "objects").mkdir()
        (data_dir / "manifests").mkdir()

        store = ArtifactStore(
            root=data_dir,
            generator="test",
            generator_version="1.0.0",
        )
        store._ensure_dirs()

        registry_db = data_dir / "registry.db"
        registry = ArtifactRegistry(registry_db)
        registry.initialize()

        lineage_db = data_dir / "lineage.sqlite"
        lineage = LineageGraph(lineage_db)
        lineage.initialize()

        gc = GarbageCollector(
            data_dir=data_dir,
            store=store,
            registry=registry,
            lineage=lineage,
        )

        return {
            "data_dir": data_dir,
            "store": store,
            "registry": registry,
            "lineage": lineage,
            "gc": gc,
        }

    def test_ancestor_protected_when_descendant_is_pinned(self, gc_with_lineage) -> None:
        """Ancestors of pinned artifacts should be protected from deletion."""
        store = gc_with_lineage["store"]
        registry = gc_with_lineage["registry"]
        lineage = gc_with_lineage["lineage"]
        gc = gc_with_lineage["gc"]

        # Create an ancestor artifact (root input)
        ancestor_manifest = store.put(
            content=b"ancestor content",
            artifact_type="other",
            roles=["root_input"],
            run_id="run-001",
        )
        registry.index_artifact(ancestor_manifest)

        # Add the ancestor node to lineage graph
        lineage.add_node(
            artifact_id=ancestor_manifest.artifact_id,
            artifact_type="other",
            content_hash_digest=ancestor_manifest.content_hash.digest,
        )

        # Create a descendant artifact that depends on the ancestor
        descendant_manifest = store.put(
            content=b"descendant content",
            artifact_type="other",
            roles=["final_output"],
            run_id="run-001",
        )
        registry.index_artifact(descendant_manifest)

        # Add descendant node to lineage
        lineage.add_node(
            artifact_id=descendant_manifest.artifact_id,
            artifact_type="other",
            content_hash_digest=descendant_manifest.content_hash.digest,
        )

        # Add edge from ancestor to descendant (ancestor -> descendant)
        lineage.add_edge(
            source_id=ancestor_manifest.artifact_id,
            target_id=descendant_manifest.artifact_id,
            relation="derived_from",
        )

        # Pin the descendant
        gc.pin_artifact(artifact_id=descendant_manifest.artifact_id, reason="test pinned")

        # Create a policy with very short retention but keep_ancestors_of_pinned=True
        policy = RetentionPolicy(
            name="test_policy",
            keep_min_age_days=0,  # Would delete everything by age
            keep_min_count=0,  # Would delete everything by count
            keep_pinned=True,
            keep_ancestors_of_pinned=True,
        )

        to_delete, to_keep = gc.compute_candidates(policy)

        # Both should be kept: descendant is pinned, ancestor is ancestor_of_pinned
        assert len(to_keep) == 2
        assert len(to_delete) == 0

        # Verify the ancestor is kept because it's an ancestor of the pinned descendant
        ancestor_candidate = next((c for c in to_keep if c.artifact_id == ancestor_manifest.artifact_id), None)
        assert ancestor_candidate is not None
        assert "ancestor_of_pinned" in ancestor_candidate.reasons_to_keep

    def test_ancestor_not_protected_when_disabled(self, gc_with_lineage) -> None:
        """When keep_ancestors_of_pinned=False, ancestors can be deleted."""
        store = gc_with_lineage["store"]
        registry = gc_with_lineage["registry"]
        lineage = gc_with_lineage["lineage"]
        gc = gc_with_lineage["gc"]

        # Create an ancestor artifact
        ancestor_manifest = store.put(
            content=b"ancestor content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        registry.index_artifact(ancestor_manifest)
        lineage.add_node(
            artifact_id=ancestor_manifest.artifact_id,
            artifact_type="other",
            content_hash_digest=ancestor_manifest.content_hash.digest,
        )

        # Create a descendant
        descendant_manifest = store.put(
            content=b"descendant content",
            artifact_type="other",
            roles=["final_output"],
            run_id="run-001",
        )
        registry.index_artifact(descendant_manifest)
        lineage.add_node(
            artifact_id=descendant_manifest.artifact_id,
            artifact_type="other",
            content_hash_digest=descendant_manifest.content_hash.digest,
        )
        lineage.add_edge(
            source_id=ancestor_manifest.artifact_id,
            target_id=descendant_manifest.artifact_id,
            relation="derived_from",
        )

        # Pin the descendant
        gc.pin_artifact(artifact_id=descendant_manifest.artifact_id, reason="test pinned")

        # Create a policy with keep_ancestors_of_pinned=False
        policy = RetentionPolicy(
            name="test_policy",
            keep_min_age_days=0,
            keep_min_count=0,
            keep_pinned=True,
            keep_with_descendants=False,  # Disable this too
            keep_ancestors_of_pinned=False,  # Disable ancestor protection
        )

        to_delete, to_keep = gc.compute_candidates(policy)

        # Only the pinned descendant should be kept
        assert len(to_keep) == 1
        assert to_keep[0].artifact_id == descendant_manifest.artifact_id

        # The ancestor should be a deletion candidate
        assert len(to_delete) == 1
        assert to_delete[0].artifact_id == ancestor_manifest.artifact_id

    def test_get_ancestors_of_pinned(self, gc_with_lineage) -> None:
        """Test that get_ancestors_of_pinned returns correct set of ancestors."""
        store = gc_with_lineage["store"]
        registry = gc_with_lineage["registry"]
        lineage = gc_with_lineage["lineage"]
        gc = gc_with_lineage["gc"]

        # Create a chain: root -> middle -> pinned
        root_manifest = store.put(
            content=b"root content",
            artifact_type="other",
            roles=["root_input"],
            run_id="run-001",
        )
        registry.index_artifact(root_manifest)
        lineage.add_node(
            artifact_id=root_manifest.artifact_id,
            artifact_type="other",
            content_hash_digest=root_manifest.content_hash.digest,
        )

        middle_manifest = store.put(
            content=b"middle content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        registry.index_artifact(middle_manifest)
        lineage.add_node(
            artifact_id=middle_manifest.artifact_id,
            artifact_type="other",
            content_hash_digest=middle_manifest.content_hash.digest,
        )

        pinned_manifest = store.put(
            content=b"pinned content",
            artifact_type="other",
            roles=["final_output"],
            run_id="run-001",
        )
        registry.index_artifact(pinned_manifest)
        lineage.add_node(
            artifact_id=pinned_manifest.artifact_id,
            artifact_type="other",
            content_hash_digest=pinned_manifest.content_hash.digest,
        )

        # Create edges: root -> middle -> pinned
        lineage.add_edge(
            source_id=root_manifest.artifact_id,
            target_id=middle_manifest.artifact_id,
            relation="derived_from",
        )
        lineage.add_edge(
            source_id=middle_manifest.artifact_id,
            target_id=pinned_manifest.artifact_id,
            relation="derived_from",
        )

        # Pin the final artifact
        gc.pin_artifact(artifact_id=pinned_manifest.artifact_id, reason="test pinned")

        # Get ancestors of pinned
        ancestors = gc.get_ancestors_of_pinned()

        # Should contain both root and middle, but not the pinned artifact itself
        assert len(ancestors) == 2
        assert root_manifest.artifact_id in ancestors
        assert middle_manifest.artifact_id in ancestors
        assert pinned_manifest.artifact_id not in ancestors

    def test_gc_result_tracks_ancestor_protected(self, gc_with_lineage) -> None:
        """GCResult should track how many artifacts were protected as ancestors."""
        store = gc_with_lineage["store"]
        registry = gc_with_lineage["registry"]
        lineage = gc_with_lineage["lineage"]
        gc = gc_with_lineage["gc"]

        # Create ancestor -> descendant chain
        ancestor_manifest = store.put(
            content=b"ancestor",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        registry.index_artifact(ancestor_manifest)
        lineage.add_node(
            artifact_id=ancestor_manifest.artifact_id,
            artifact_type="other",
            content_hash_digest=ancestor_manifest.content_hash.digest,
        )

        descendant_manifest = store.put(
            content=b"descendant",
            artifact_type="other",
            roles=["final_output"],
            run_id="run-001",
        )
        registry.index_artifact(descendant_manifest)
        lineage.add_node(
            artifact_id=descendant_manifest.artifact_id,
            artifact_type="other",
            content_hash_digest=descendant_manifest.content_hash.digest,
        )
        lineage.add_edge(
            source_id=ancestor_manifest.artifact_id,
            target_id=descendant_manifest.artifact_id,
            relation="derived_from",
        )

        gc.pin_artifact(artifact_id=descendant_manifest.artifact_id, reason="test")

        # Run GC with policy that would delete by age
        policy = RetentionPolicy(
            name="test",
            keep_min_age_days=0,
            keep_min_count=0,
            keep_pinned=True,
            keep_ancestors_of_pinned=True,
        )

        result = gc.run(policy=policy, dry_run=True, run_dvc_gc=False)

        # Check that ancestor_protected is tracked
        assert result.ancestor_protected == 1
        assert result.pinned_protected == 1
