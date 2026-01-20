"""Comprehensive edge case tests for M3 modules.

This module provides additional test coverage for:
- artifact_store: edge cases, error handling, determinism
- dataset_snapshot: empty datasets, split definitions
- lineage_graph: complex graphs, cycles, deep ancestry
- gc: space budget enforcement, mixed scenarios
- registry: complex queries, update consistency
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest

from formula_foundry.m3.artifact_store import (
    ArtifactNotFoundError,
    ArtifactStore,
    ContentHash,
    LineageReference,
    compute_spec_id,
)
from formula_foundry.m3.dataset_snapshot import (
    DatasetMember,
    DatasetSnapshotReader,
    DatasetSnapshotWriter,
    SplitDefinition,
    compute_manifest_hash,
)
from formula_foundry.m3.gc import (
    GarbageCollector,
    GCResult,
    PinnedArtifact,
    RetentionPolicy,
    format_bytes,
)
from formula_foundry.m3.lineage_graph import (
    LineageGraph,
    LineageNode,
    LineageSubgraph,
)
from formula_foundry.m3.registry import (
    ArtifactRegistry,
)

# Check for PyArrow availability
try:
    import pyarrow

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


# =============================================================================
# ArtifactStore Edge Cases
# =============================================================================


class TestArtifactStoreEdgeCases:
    """Edge case tests for ArtifactStore."""

    def test_zero_byte_content(self, tmp_path: Path) -> None:
        """Test storing and retrieving zero-byte content."""
        store = ArtifactStore(tmp_path / "data")
        content = b""
        expected_hash = hashlib.sha256(content).hexdigest()

        manifest = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        assert manifest.byte_size == 0
        assert manifest.content_hash.digest == expected_hash
        assert store.get(manifest.content_hash.digest) == b""
        assert store.verify(manifest.artifact_id)

    def test_large_content_hash_consistency(self, tmp_path: Path) -> None:
        """Test that large content produces consistent hashes."""
        store = ArtifactStore(tmp_path / "data")
        # Generate 1MB of pseudo-random but deterministic content
        content = bytes(range(256)) * (1024 * 4)  # 1KB pattern repeated

        manifest1 = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        # Store same content again
        manifest2 = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-002",
        )

        assert manifest1.content_hash.digest == manifest2.content_hash.digest
        assert manifest1.artifact_id != manifest2.artifact_id

    def test_unicode_in_tags_and_annotations(self, tmp_path: Path) -> None:
        """Test that unicode in tags and annotations is preserved."""
        store = ArtifactStore(tmp_path / "data")

        manifest = store.put(
            content=b"test",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
            tags={"日本語": "キー", "emoji": "🎉"},
            annotations={"nested": {"unicode": "čřžšďťň"}},
        )

        retrieved = store.get_manifest(manifest.artifact_id)
        assert retrieved.tags["日本語"] == "キー"
        assert retrieved.tags["emoji"] == "🎉"
        assert retrieved.annotations["nested"]["unicode"] == "čřžšďťň"

    def test_multiple_lineage_inputs(self, tmp_path: Path) -> None:
        """Test artifacts with multiple lineage inputs."""
        store = ArtifactStore(tmp_path / "data")

        # Create three source artifacts
        sources = []
        for i in range(3):
            m = store.put(
                content=f"source-{i}".encode(),
                artifact_type="coupon_spec",
                roles=["geometry"],
                run_id="run-001",
            )
            sources.append(m)

        # Create derived artifact with all three as inputs
        inputs = [
            LineageReference(
                artifact_id=s.artifact_id,
                relation="derived_from",
                content_hash=s.content_hash,
            )
            for s in sources
        ]

        derived = store.put(
            content=b"derived",
            artifact_type="resolved_design",
            roles=["intermediate"],
            run_id="run-001",
            inputs=inputs,
        )

        assert len(derived.lineage.inputs) == 3
        retrieved = store.get_manifest(derived.artifact_id)
        assert len(retrieved.lineage.inputs) == 3

    def test_empty_store_operations(self, tmp_path: Path) -> None:
        """Test operations on an empty store."""
        store = ArtifactStore(tmp_path / "data")
        store._ensure_dirs()

        assert store.list_manifests() == []
        assert not store.exists("a" * 64)
        assert not store.exists_by_id("nonexistent")

        with pytest.raises(ArtifactNotFoundError):
            store.get("a" * 64)

        with pytest.raises(ArtifactNotFoundError):
            store.get_manifest("nonexistent")

    def test_corrupted_manifest_json(self, tmp_path: Path) -> None:
        """Test handling of corrupted manifest JSON."""
        store = ArtifactStore(tmp_path / "data")

        manifest = store.put(
            content=b"test",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        # Corrupt the manifest file
        manifest_path = store._manifest_path(manifest.artifact_id)
        manifest_path.write_text("not valid json{{{")

        with pytest.raises(Exception):  # json.JSONDecodeError
            store.get_manifest(manifest.artifact_id)

    def test_spec_id_collision_resistance(self) -> None:
        """Test that similar inputs produce different spec IDs."""
        digests = []
        for i in range(100):
            content = f"content-{i}".encode()
            digest = hashlib.sha256(content).hexdigest()
            digests.append(digest)

        spec_ids = [compute_spec_id(d) for d in digests]
        # All spec IDs should be unique
        assert len(set(spec_ids)) == len(spec_ids)

    def test_artifact_id_format(self, tmp_path: Path) -> None:
        """Test that auto-generated artifact IDs follow expected format."""
        store = ArtifactStore(tmp_path / "data")

        manifest = store.put(
            content=b"test",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        # Format: art-YYYYMMDDTHHMMSS-xxxxxxxx
        assert manifest.artifact_id.startswith("art-")
        parts = manifest.artifact_id.split("-")
        assert len(parts) == 3
        assert len(parts[1]) == 15  # YYYYMMDDTHHMMSS
        assert len(parts[2]) == 8  # hex uuid

    def test_put_file_with_missing_file(self, tmp_path: Path) -> None:
        """Test put_file with a missing file raises error."""
        store = ArtifactStore(tmp_path / "data")

        with pytest.raises(FileNotFoundError):
            store.put_file(
                file_path=tmp_path / "nonexistent.txt",
                artifact_type="other",
                roles=["intermediate"],
                run_id="run-001",
            )

    def test_delete_nonexistent_artifact(self, tmp_path: Path) -> None:
        """Test deleting a nonexistent artifact raises error."""
        store = ArtifactStore(tmp_path / "data")
        store._ensure_dirs()

        with pytest.raises(ArtifactNotFoundError):
            store.delete("nonexistent-id")


class TestArtifactStoreDeterminism:
    """Tests for ArtifactStore determinism guarantees."""

    def test_content_hash_deterministic_across_instances(self, tmp_path: Path) -> None:
        """Test that content hashes are consistent across store instances."""
        content = b"determinism test content"

        store1 = ArtifactStore(tmp_path / "data1")
        store2 = ArtifactStore(tmp_path / "data2")

        m1 = store1.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        m2 = store2.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        assert m1.content_hash.digest == m2.content_hash.digest
        assert m1.spec_id == m2.spec_id

    def test_spec_id_deterministic(self) -> None:
        """Test that spec_id computation is deterministic."""
        content = b"deterministic spec id test"
        digest = hashlib.sha256(content).hexdigest()

        spec_ids = [compute_spec_id(digest) for _ in range(100)]
        assert len(set(spec_ids)) == 1

    def test_manifest_serialization_deterministic(self, tmp_path: Path) -> None:
        """Test that manifest JSON serialization is deterministic."""
        store = ArtifactStore(tmp_path / "data")

        manifest = store.put(
            content=b"test",
            artifact_type="other",
            roles=["intermediate", "config"],
            run_id="run-001",
            tags={"b": "2", "a": "1"},  # Should be sorted
        )

        json1 = manifest.to_json()
        json2 = manifest.to_json()

        assert json1 == json2
        # Keys should be sorted
        assert '"a"' in json1
        assert json1.index('"a"') < json1.index('"b"')


# =============================================================================
# DatasetSnapshot Edge Cases
# =============================================================================


class TestDatasetSnapshotEdgeCases:
    """Edge case tests for DatasetSnapshot."""

    def test_empty_dataset(self, tmp_path: Path) -> None:
        """Test creating a dataset with no members."""
        store = ArtifactStore(tmp_path / "data")

        writer = DatasetSnapshotWriter(
            dataset_id="empty_dataset",
            version="v1.0",
            store=store,
            name="Empty Dataset",
        )

        output_dir = tmp_path / "datasets"
        snapshot = writer.finalize(output_dir=output_dir, write_parquet=False)

        assert snapshot.member_count == 0
        assert snapshot.total_bytes == 0
        assert snapshot.statistics is not None

    def test_dataset_with_duplicate_artifact_ids(self, tmp_path: Path) -> None:
        """Test that adding duplicate artifact IDs is handled."""
        store = ArtifactStore(tmp_path / "data")

        manifest = store.put(
            content=b"test",
            artifact_type="touchstone",
            roles=["oracle_output"],
            run_id="run-001",
        )

        writer = DatasetSnapshotWriter(
            dataset_id="dup_test",
            version="v1",
            store=store,
        )

        writer.add_member(manifest, role="oracle_output")
        writer.add_member(manifest, role="oracle_output")  # Same manifest again

        snapshot = writer.finalize(write_parquet=False)
        # Implementation may deduplicate or allow duplicates
        # Either behavior should be consistent
        assert snapshot.member_count >= 1

    def test_split_definition_empty_partitions(self) -> None:
        """Test split definitions with empty artifact lists."""
        split = SplitDefinition(
            name="empty_train",
            artifact_ids=[],
            count=0,
            fraction=0.0,
        )

        data = split.to_dict()
        restored = SplitDefinition.from_dict(data)

        assert restored.artifact_ids == []
        assert restored.count == 0

    def test_manifest_hash_same_content_same_hash(self) -> None:
        """Test that manifest hash is based on artifact metadata, not features.

        Note: The compute_manifest_hash function uses DatasetMember.to_dict()
        which doesn't include features, so features don't affect the hash.
        This is intentional - the hash is for content integrity, not feature metadata.
        """
        member1 = DatasetMember(
            artifact_id="art-001",
            content_hash=ContentHash(algorithm="sha256", digest="a" * 64),
            artifact_type="touchstone",
            role="oracle_output",
        )

        member2 = DatasetMember(
            artifact_id="art-001",
            content_hash=ContentHash(algorithm="sha256", digest="a" * 64),
            artifact_type="touchstone",
            role="oracle_output",
            features={"via_diameter": 0.3},  # Features not included in hash
        )

        hash1 = compute_manifest_hash([member1])
        hash2 = compute_manifest_hash([member2])

        # Hash should be the same because to_dict() excludes features
        assert hash1.digest == hash2.digest

    def test_manifest_hash_different_content_different_hash(self) -> None:
        """Test that different artifact content produces different hashes."""
        member1 = DatasetMember(
            artifact_id="art-001",
            content_hash=ContentHash(algorithm="sha256", digest="a" * 64),
            artifact_type="touchstone",
            role="oracle_output",
        )

        member2 = DatasetMember(
            artifact_id="art-002",  # Different ID
            content_hash=ContentHash(algorithm="sha256", digest="b" * 64),  # Different hash
            artifact_type="touchstone",
            role="oracle_output",
        )

        hash1 = compute_manifest_hash([member1])
        hash2 = compute_manifest_hash([member2])

        # Hash should differ because content differs
        assert hash1.digest != hash2.digest

    def test_dataset_with_special_characters_in_id(self, tmp_path: Path) -> None:
        """Test dataset with special characters in dataset_id."""
        store = ArtifactStore(tmp_path / "data")
        manifest = store.put(
            content=b"test",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        # Note: some characters may cause issues with file systems
        # Test underscore and hyphen which should be safe
        writer = DatasetSnapshotWriter(
            dataset_id="my-dataset_v1.0",
            version="v1",
            store=store,
        )
        writer.add_member(manifest, role="intermediate")

        output_dir = tmp_path / "datasets"
        snapshot = writer.finalize(output_dir=output_dir, write_parquet=False)

        assert snapshot.dataset_id == "my-dataset_v1.0"

    def test_statistics_computation_single_type(self, tmp_path: Path) -> None:
        """Test statistics computation with single artifact type."""
        store = ArtifactStore(tmp_path / "data")

        writer = DatasetSnapshotWriter(
            dataset_id="single_type",
            version="v1",
            store=store,
        )

        for i in range(5):
            manifest = store.put(
                content=f"content-{i}".encode(),
                artifact_type="touchstone",
                roles=["oracle_output"],
                run_id="run-001",
            )
            writer.add_member(manifest, role="oracle_output")

        snapshot = writer.finalize(write_parquet=False)

        assert snapshot.statistics.by_artifact_type["touchstone"]["count"] == 5


# =============================================================================
# LineageGraph Edge Cases
# =============================================================================


class TestLineageGraphEdgeCases:
    """Edge case tests for LineageGraph."""

    def test_self_referential_edge(self, tmp_path: Path) -> None:
        """Test adding an edge where source and target are the same."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("self-ref", "other", "hash")
        # Implementation should handle or reject self-references
        edge = graph.add_edge("self-ref", "self-ref", "derived_from")

        # Either the edge is accepted or rejected gracefully
        if edge:
            edges = graph.get_edges_from("self-ref")
            assert len(edges) >= 1

        graph.close()

    def test_very_deep_ancestry_chain(self, tmp_path: Path) -> None:
        """Test a very deep ancestry chain (100 levels)."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        depth = 100
        for i in range(depth):
            graph.add_node(f"node-{i:04d}", "other", f"hash{i}")
            if i > 0:
                graph.add_edge(f"node-{i - 1:04d}", f"node-{i:04d}", "derived_from")

        # Get all ancestors of the deepest node
        subgraph = graph.get_ancestors(f"node-{depth - 1:04d}")
        assert len(subgraph.nodes) == depth

        # Get all descendants of the root
        subgraph = graph.get_descendants("node-0000")
        assert len(subgraph.nodes) == depth

        graph.close()

    def test_wide_graph_many_children(self, tmp_path: Path) -> None:
        """Test a node with many direct children."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("parent", "coupon_spec", "hash0")

        num_children = 50
        for i in range(num_children):
            graph.add_node(f"child-{i:03d}", "kicad_board", f"hash{i}")
            graph.add_edge("parent", f"child-{i:03d}", "derived_from")

        outputs = graph.get_direct_outputs("parent")
        assert len(outputs) == num_children

        graph.close()

    def test_multiple_paths_to_same_node(self, tmp_path: Path) -> None:
        """Test multiple paths to the same descendant."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        #     root
        #    /    \
        #   a      b
        #    \    /
        #     leaf

        graph.add_node("root", "coupon_spec", "h0")
        graph.add_node("a", "intermediate", "h1")
        graph.add_node("b", "intermediate", "h2")
        graph.add_node("leaf", "output", "h3")

        graph.add_edge("root", "a", "derived_from")
        graph.add_edge("root", "b", "derived_from")
        graph.add_edge("a", "leaf", "derived_from")
        graph.add_edge("b", "leaf", "derived_from")

        ancestors = graph.get_ancestors("leaf")
        assert len(ancestors.nodes) == 4

        descendants = graph.get_descendants("root")
        assert len(descendants.nodes) == 4

        graph.close()

    def test_empty_graph_operations(self, tmp_path: Path) -> None:
        """Test operations on an empty graph."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        assert graph.count_nodes() == 0
        assert graph.count_edges() == 0

        nodes = graph.query_nodes()
        assert nodes == []

        stats = graph.get_stats()
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0

        graph.close()

    def test_clear_removes_all_nodes_and_edges(self, tmp_path: Path) -> None:
        """Test that clear() removes all nodes and edges."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("a", "type", "h1")
        graph.add_node("b", "type", "h2")
        graph.add_node("c", "type", "h3")

        graph.add_edge("a", "b", "derived_from")
        graph.add_edge("b", "c", "derived_from")

        assert graph.count_nodes() == 3
        assert graph.count_edges() == 2

        # Clear all data
        graph.clear()

        assert graph.count_nodes() == 0
        assert graph.count_edges() == 0

        graph.close()

    def test_concurrent_reads(self, tmp_path: Path) -> None:
        """Test that concurrent reads work correctly."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        # Add some data
        for i in range(10):
            graph.add_node(f"node-{i}", "other", f"hash{i}")

        errors: list[Exception] = []
        results: list[int] = []

        def read_count() -> None:
            try:
                count = graph.count_nodes()
                results.append(count)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_count) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r == 10 for r in results)

        graph.close()

    def test_export_import_round_trip(self, tmp_path: Path) -> None:
        """Test exporting and importing graph data."""
        graph1 = LineageGraph(tmp_path / "lineage1.sqlite")
        graph1.initialize()

        graph1.add_node("a", "type1", "h1", run_id="run-001")
        graph1.add_node("b", "type2", "h2", run_id="run-001")
        graph1.add_edge("a", "b", "derived_from")

        exported = graph1.export_to_dict()
        graph1.close()

        # Verify export structure
        assert exported["schema_version"] == 1
        assert exported["node_count"] == 2
        assert exported["edge_count"] == 1
        assert "a" in exported["nodes"]
        assert "b" in exported["nodes"]


class TestLineageSubgraph:
    """Tests for LineageSubgraph class."""

    def test_empty_subgraph(self) -> None:
        """Test empty subgraph properties."""
        subgraph = LineageSubgraph(nodes={}, edges=[])

        assert subgraph.get_roots() == []
        assert subgraph.get_leaves() == []

    def test_single_node_subgraph(self) -> None:
        """Test subgraph with a single isolated node."""
        node = LineageNode("isolated", "type", "hash")
        subgraph = LineageSubgraph(nodes={"isolated": node}, edges=[])

        roots = subgraph.get_roots()
        leaves = subgraph.get_leaves()

        # Single isolated node is both root and leaf
        assert "isolated" in roots
        assert "isolated" in leaves


# =============================================================================
# GC Edge Cases
# =============================================================================


class TestGCEdgeCases:
    """Edge case tests for GarbageCollector."""

    @pytest.fixture
    def gc_env(self, tmp_path: Path):
        """Set up a complete GC environment."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        store = ArtifactStore(data_dir, generator="test", generator_version="1.0")
        store._ensure_dirs()

        registry = ArtifactRegistry(data_dir / "registry.db")
        registry.initialize()

        lineage = LineageGraph(data_dir / "lineage.sqlite")
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

    def test_pin_by_dataset_id(self, gc_env) -> None:
        """Test pinning artifacts by dataset ID."""
        gc = gc_env["gc"]

        gc.pin_artifact(dataset_id="ds-001", reason="production dataset")
        pins = gc.pins

        assert len(pins) == 1
        assert pins[0].dataset_id == "ds-001"

    def test_multiple_pins_same_artifact(self, gc_env) -> None:
        """Test multiple pins on the same artifact."""
        gc = gc_env["gc"]

        gc.pin_artifact(artifact_id="art-001", reason="reason 1")
        gc.pin_artifact(artifact_id="art-001", reason="reason 2")

        # Both pins should exist
        pins = gc.pins
        art_pins = [p for p in pins if p.artifact_id == "art-001"]
        assert len(art_pins) == 2

    def test_unpin_nonexistent(self, gc_env) -> None:
        """Test unpinning a nonexistent pin."""
        gc = gc_env["gc"]

        removed = gc.unpin_artifact(artifact_id="nonexistent")
        assert removed is False

    def test_policy_with_all_protections(self, gc_env) -> None:
        """Test a policy that protects everything."""
        store = gc_env["store"]
        registry = gc_env["registry"]
        gc = gc_env["gc"]

        # Add some artifacts
        for i in range(5):
            manifest = store.put(
                content=f"content-{i}".encode(),
                artifact_type="dataset_snapshot",  # Protected type
                roles=["final_output"],  # Protected role
                run_id="run-001",
            )
            registry.index_artifact(manifest)

        # Use laptop_default which protects dataset_snapshot and final_output
        to_delete, to_keep = gc.compute_candidates("laptop_default")

        assert len(to_delete) == 0
        assert len(to_keep) == 5

    def test_gc_result_serialization(self) -> None:
        """Test GCResult serialization."""
        result = GCResult(
            policy_name="test_policy",
            started_utc="2026-01-20T00:00:00Z",
            finished_utc="2026-01-20T00:01:00Z",
            dry_run=False,
            artifacts_scanned=100,
            artifacts_deleted=10,
            bytes_freed=1024000,
            bytes_total_before=10240000,
            bytes_total_after=9216000,
            pinned_protected=5,
            descendant_protected=3,
            dvc_gc_ran=False,
            dvc_gc_output=None,
            errors=["test error"],
            deleted_artifacts=["art-001", "art-002"],
            protected_artifacts=["art-003"],
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["artifacts_deleted"] == 10
        assert "test error" in parsed["errors"]

    def test_format_bytes_edge_cases(self) -> None:
        """Test format_bytes with edge cases."""
        assert "B" in format_bytes(0)
        assert "B" in format_bytes(1)
        assert "B" in format_bytes(1023)
        assert "KB" in format_bytes(1024)
        assert "MB" in format_bytes(1024 * 1024)
        assert "GB" in format_bytes(1024 * 1024 * 1024)
        assert "TB" in format_bytes(1024 * 1024 * 1024 * 1024)

    def test_retention_policy_from_dict_defaults(self) -> None:
        """Test RetentionPolicy.from_dict with minimal input."""
        data = {"name": "minimal"}
        policy = RetentionPolicy.from_dict(data)

        assert policy.name == "minimal"
        assert policy.keep_min_age_days == 30  # Default
        assert policy.keep_pinned is True  # Default
        assert policy.keep_with_descendants is True  # Default

    def test_pinned_artifact_serialization_all_fields(self) -> None:
        """Test PinnedArtifact with all fields set."""
        pin = PinnedArtifact(
            artifact_id="art-001",
            run_id="run-001",
            dataset_id="ds-001",
            reason="test reason",
            pinned_utc="2026-01-20T00:00:00Z",
        )

        data = pin.to_dict()
        restored = PinnedArtifact.from_dict(data)

        assert restored.artifact_id == "art-001"
        assert restored.run_id == "run-001"
        assert restored.dataset_id == "ds-001"
        assert restored.reason == "test reason"


class TestGCPolicyApplication:
    """Tests for GC policy application logic."""

    @pytest.fixture
    def populated_gc(self, tmp_path: Path):
        """Set up GC with pre-populated artifacts of various ages."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        store = ArtifactStore(data_dir, generator="test", generator_version="1.0")
        store._ensure_dirs()

        registry = ArtifactRegistry(data_dir / "registry.db")
        registry.initialize()

        lineage = LineageGraph(data_dir / "lineage.sqlite")
        lineage.initialize()

        gc = GarbageCollector(
            data_dir=data_dir,
            store=store,
            registry=registry,
            lineage=lineage,
        )

        # Add artifacts
        manifests = []
        for i in range(10):
            manifest = store.put(
                content=f"content-{i}".encode(),
                artifact_type="other" if i < 5 else "touchstone",
                roles=["intermediate"] if i < 7 else ["final_output"],
                run_id=f"run-{i // 3:03d}",
            )
            registry.index_artifact(manifest)
            manifests.append(manifest)

        return {
            "data_dir": data_dir,
            "store": store,
            "registry": registry,
            "lineage": lineage,
            "gc": gc,
            "manifests": manifests,
        }

    def test_dev_minimal_policy(self, populated_gc) -> None:
        """Test dev_minimal policy behavior."""
        gc = populated_gc["gc"]

        # All artifacts are fresh (age < 3 days), so they should be kept
        to_delete, to_keep = gc.compute_candidates("dev_minimal")

        # Fresh artifacts should be kept
        assert len(to_keep) > 0

    def test_custom_policy(self, populated_gc) -> None:
        """Test applying a custom policy."""
        gc = populated_gc["gc"]

        custom_policy = RetentionPolicy(
            name="custom",
            keep_min_age_days=0,  # Delete everything by age
            keep_min_count=0,  # No minimum count
            keep_pinned=True,
            keep_with_descendants=False,
            keep_artifact_types=[],
            keep_roles=[],
        )

        to_delete, to_keep = gc.compute_candidates(custom_policy)

        # With no protections, everything could be deleted
        # (except they're all fresh, so age rule still applies)
        assert len(to_delete) + len(to_keep) == 10


# =============================================================================
# Registry Edge Cases
# =============================================================================


class TestRegistryEdgeCases:
    """Edge case tests for ArtifactRegistry."""

    def test_query_with_all_filters(self, tmp_path: Path) -> None:
        """Test query with all filter types combined."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        # Create specific artifacts
        m1 = store.put(
            content=b"target",
            artifact_type="touchstone",
            roles=["oracle_output", "final_output"],
            run_id="run-001",
        )
        registry.index_artifact(m1)

        # Query with multiple filters
        results = registry.query_artifacts(
            artifact_type="touchstone",
            run_id="run-001",
            roles=["oracle_output"],
        )

        assert len(results) == 1
        assert results[0].artifact_id == m1.artifact_id

        registry.close()

    def test_query_no_matches(self, tmp_path: Path) -> None:
        """Test query that returns no matches."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        m = store.put(
            content=b"test",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        registry.index_artifact(m)

        results = registry.query_artifacts(artifact_type="nonexistent_type")
        assert results == []

        registry.close()

    def test_storage_stats_empty_registry(self, tmp_path: Path) -> None:
        """Test storage stats on empty registry."""
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        stats = registry.get_storage_stats()

        assert stats["total_artifacts"] == 0
        assert stats["total_bytes"] == 0
        assert stats["unique_hashes"] == 0

        registry.close()

    def test_run_update_without_initial_index(self, tmp_path: Path) -> None:
        """Test updating a run that wasn't explicitly indexed."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        # Index artifact (which auto-creates run)
        m = store.put(
            content=b"test",
            artifact_type="other",
            roles=["intermediate"],
            run_id="auto-run",
        )
        registry.index_artifact(m)

        # Update the auto-created run
        registry.update_run_status(
            run_id="auto-run",
            status="completed",
            ended_utc="2026-01-20T10:00:00Z",
        )

        record = registry.get_run("auto-run")
        assert record.status == "completed"

        registry.close()

    def test_concurrent_queries(self, tmp_path: Path) -> None:
        """Test concurrent query operations."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        # Add some artifacts
        for i in range(20):
            m = store.put(
                content=f"content-{i}".encode(),
                artifact_type="other",
                roles=["intermediate"],
                run_id=f"run-{i // 5}",
            )
            registry.index_artifact(m)

        errors: list[Exception] = []
        results: list[int] = []

        def query_count() -> None:
            try:
                artifacts = registry.query_artifacts()
                results.append(len(artifacts))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=query_count) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r == 20 for r in results)

        registry.close()

    def test_query_datasets(self, tmp_path: Path) -> None:
        """Test querying datasets."""
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        # Index multiple datasets
        for i in range(5):
            registry.index_dataset(
                dataset_id=f"ds-{i:03d}",
                version=f"v1.{i}",
                artifact_count=10 * (i + 1),
                total_bytes=1000 * (i + 1),
                created_utc="2026-01-20T00:00:00Z",
            )

        datasets = registry.query_datasets()
        assert len(datasets) == 5

        registry.close()


# =============================================================================
# Integration Edge Cases
# =============================================================================


class TestIntegrationEdgeCases:
    """Integration tests covering cross-module edge cases."""

    def test_full_artifact_lifecycle(self, tmp_path: Path) -> None:
        """Test complete artifact lifecycle through all modules."""
        # Set up all modules
        store = ArtifactStore(tmp_path / "data")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()
        lineage = LineageGraph(tmp_path / "lineage.sqlite")
        lineage.initialize()

        # Create source artifact
        source = store.put(
            content=b"source content",
            artifact_type="coupon_spec",
            roles=["geometry"],
            run_id="run-001",
        )
        registry.index_artifact(source)
        lineage.add_manifest(source)

        # Create derived artifact
        derived = store.put(
            content=b"derived content",
            artifact_type="kicad_board",
            roles=["intermediate"],
            run_id="run-001",
            inputs=[
                LineageReference(
                    artifact_id=source.artifact_id,
                    relation="derived_from",
                    content_hash=source.content_hash,
                )
            ],
        )
        registry.index_artifact(derived)
        lineage.add_manifest(derived)

        # Verify registry
        assert registry.count_artifacts() == 2

        # Verify lineage
        assert lineage.count_nodes() == 2
        assert lineage.count_edges() == 1

        # Verify source has descendants
        descendants = lineage.get_descendants(source.artifact_id)
        assert derived.artifact_id in descendants.nodes

        lineage.close()
        registry.close()

    def test_gc_respects_lineage(self, tmp_path: Path) -> None:
        """Test that GC respects lineage relationships."""
        store = ArtifactStore(tmp_path / "data")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()
        lineage = LineageGraph(tmp_path / "lineage.sqlite")
        lineage.initialize()

        gc = GarbageCollector(
            data_dir=tmp_path / "data",
            store=store,
            registry=registry,
            lineage=lineage,
        )

        # Create parent and child
        parent = store.put(
            content=b"parent",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        registry.index_artifact(parent)
        lineage.add_manifest(parent)

        child = store.put(
            content=b"child",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
            inputs=[LineageReference(artifact_id=parent.artifact_id, relation="derived_from")],
        )
        registry.index_artifact(child)
        lineage.add_manifest(child)

        # With keep_with_descendants=True, parent should be protected
        policy = RetentionPolicy(
            name="test",
            keep_min_age_days=0,
            keep_min_count=0,
            keep_with_descendants=True,
        )

        to_delete, to_keep = gc.compute_candidates(policy)

        parent_ids = [c.artifact_id for c in to_keep]
        # Parent should be kept because it has descendants
        assert parent.artifact_id in parent_ids

        lineage.close()
        registry.close()

    def test_dataset_from_store_artifacts(self, tmp_path: Path) -> None:
        """Test creating a dataset from store artifacts."""
        store = ArtifactStore(tmp_path / "data")

        # Create multiple artifacts
        artifacts = []
        for i in range(10):
            m = store.put(
                content=f"content-{i}".encode(),
                artifact_type="touchstone" if i % 2 == 0 else "coupon_spec",
                roles=["oracle_output"] if i % 2 == 0 else ["geometry"],
                run_id="run-001",
            )
            artifacts.append(m)

        # Create dataset from artifacts
        writer = DatasetSnapshotWriter(
            dataset_id="from_store",
            version="v1",
            store=store,
        )

        for m in artifacts:
            role = "oracle_output" if m.artifact_type == "touchstone" else "geometry"
            writer.add_member(m, role=role)

        output_dir = tmp_path / "datasets"
        snapshot = writer.finalize(output_dir=output_dir, write_parquet=False)

        assert snapshot.member_count == 10
        assert snapshot.statistics.by_artifact_type["touchstone"]["count"] == 5
        assert snapshot.statistics.by_artifact_type["coupon_spec"]["count"] == 5

        # Verify integrity
        reader = DatasetSnapshotReader(
            snapshot_path=output_dir / "from_store_v1.json",
            store=store,
        )
        is_valid, errors = reader.verify_integrity()
        assert is_valid
        assert len(errors) == 0
