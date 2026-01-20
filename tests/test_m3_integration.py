"""Integration tests for the M3 subsystem.

These integration tests verify the full M3 workflow including:
1. Full workflow: init -> run -> artifact tracking -> dataset snapshot -> gc -> audit -> verify
2. Determinism across multiple runs
3. Component interaction between ArtifactStore, Registry, LineageGraph, and CLI
4. Data consistency through the entire pipeline
5. Recovery and error scenarios

The tests ensure that:
- Artifacts are correctly tracked through the entire lifecycle
- Content hashes remain consistent (determinism)
- Lineage relationships are preserved through operations
- GC respects pinning and retention policies
- Audit can trace full provenance chains
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from formula_foundry.m3.artifact_store import (
    ArtifactStore,
    ArtifactManifest,
    LineageReference,
)
from formula_foundry.m3.cli_main import build_parser, cmd_audit, cmd_init, cmd_run, main
from formula_foundry.m3.dataset_snapshot import (
    DatasetSnapshot,
    DatasetSnapshotWriter,
    DatasetSnapshotReader,
    compute_manifest_hash,
)
from formula_foundry.m3.gc import (
    GarbageCollector,
    RetentionPolicy,
)
from formula_foundry.m3.lineage_graph import LineageGraph
from formula_foundry.m3.registry import ArtifactRegistry

if TYPE_CHECKING:
    pass


class TestFullWorkflowIntegration:
    """Tests for the complete M3 workflow from init to verification."""

    @pytest.fixture
    def full_project(self, tmp_path: Path) -> Path:
        """Create a fully initialized M3 project with git."""
        # Initialize git
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path, check=True, capture_output=True
        )
        (tmp_path / "test.txt").write_text("test")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            cwd=tmp_path, check=True, capture_output=True
        )

        # Initialize M3
        cmd_init(root=tmp_path, force=False, quiet=True)

        # Create DVC config for run command
        (tmp_path / "dvc.yaml").write_text(
            "stages:\n"
            "  generate_coupon:\n"
            "    cmd: echo 'generating coupon'\n"
            "  run_simulation:\n"
            "    cmd: echo 'running simulation'\n"
            "  create_dataset:\n"
            "    cmd: echo 'creating dataset'\n"
        )

        return tmp_path

    def test_init_creates_complete_structure(self, full_project: Path) -> None:
        """Init should create all required directories and databases."""
        data_dir = full_project / "data"

        # Directory structure
        assert (data_dir / "objects").is_dir()
        assert (data_dir / "manifests").is_dir()
        assert (data_dir / "mlflow").is_dir()
        assert (data_dir / "mlflow" / "artifacts").is_dir()
        assert (data_dir / "datasets").is_dir()

        # Databases
        assert (data_dir / "registry.db").exists()

        # Registry should be queryable
        registry = ArtifactRegistry(data_dir / "registry.db")
        runs = registry.query_runs()
        registry.close()
        assert len(runs) >= 1  # At least the init run

    def test_full_workflow_init_to_verify(self, full_project: Path) -> None:
        """Test the complete workflow from init through verification."""
        data_dir = full_project / "data"

        # Step 1: Create artifacts simulating coupon generation
        store = ArtifactStore(root=data_dir, generator="integration_test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")
        lineage = LineageGraph(data_dir / "lineage.sqlite")
        lineage.initialize()

        # Create a coupon spec (root input)
        spec_content = b'{"coupon_family": "F1", "version": "1.0"}'
        spec_manifest = store.put(
            content=spec_content,
            artifact_type="coupon_spec",
            roles=["config", "root_input"],
            run_id="run-001-coupon-gen",
            artifact_id="art-coupon-spec-001",
            stage_name="coupon_generation",
        )
        registry.index_artifact(spec_manifest)
        lineage.add_manifest(spec_manifest)

        # Create resolved design (derived from spec)
        design_content = b'{"w_nm": 300000, "gap_nm": 180000}'
        design_manifest = store.put(
            content=design_content,
            artifact_type="resolved_design",
            roles=["geometry"],
            run_id="run-001-coupon-gen",
            artifact_id="art-design-001",
            stage_name="coupon_generation",
            inputs=[LineageReference(artifact_id="art-coupon-spec-001", relation="config_from")],
        )
        registry.index_artifact(design_manifest)
        lineage.add_manifest(design_manifest)

        # Create KiCad board (derived from design)
        board_content = b"(kicad_pcb (version 9.0.7) ...)"
        board_manifest = store.put(
            content=board_content,
            artifact_type="kicad_board",
            roles=["final_output"],
            run_id="run-001-coupon-gen",
            artifact_id="art-board-001",
            stage_name="coupon_generation",
            inputs=[LineageReference(artifact_id="art-design-001", relation="derived_from")],
        )
        registry.index_artifact(board_manifest)
        lineage.add_manifest(board_manifest)

        # Step 2: Create dataset snapshot
        dataset_dir = data_dir / "datasets"
        writer = DatasetSnapshotWriter(
            dataset_id="coupon_dataset_v1",
            version="v1.0",
            store=store,
            generator="integration_test",
            generator_version="1.0.0",
            name="Coupon Test Dataset",
            description="Integration test dataset",
        )
        writer.add_member(spec_manifest, role="config", features={"coupon_family": "F1"})
        writer.add_member(design_manifest, role="geometry", features={"w_nm": 300000})
        writer.add_member(board_manifest, role="output", features={})
        snapshot = writer.finalize(output_dir=dataset_dir, write_parquet=False)

        registry.index_dataset_snapshot(snapshot)

        # Step 3: Verify the dataset was created correctly
        assert snapshot.member_count == 3
        assert snapshot.content_hash.algorithm == "sha256"
        assert len(snapshot.content_hash.digest) == 64

        # Step 4: Test GC with pinning
        gc = GarbageCollector(
            data_dir=data_dir,
            store=store,
            registry=registry,
            lineage=lineage,
        )

        # Pin the dataset
        gc.pin_artifact(artifact_id=board_manifest.artifact_id, reason="important output")

        # Use aggressive policy that would delete everything by age
        policy = RetentionPolicy(
            name="test_aggressive",
            keep_min_age_days=0,
            keep_min_count=0,
            keep_pinned=True,
            keep_with_descendants=False,
        )

        # Dry run should show pinned artifact is protected
        to_delete, to_keep = gc.compute_candidates(policy)
        pinned_kept = [c for c in to_keep if c.artifact_id == board_manifest.artifact_id]
        assert len(pinned_kept) == 1
        assert "pinned" in pinned_kept[0].reasons_to_keep

        # Step 5: Audit the output artifact
        result = cmd_audit(
            artifact_id="art-board-001",
            root=full_project,
            output_format="text",
            trace_roots=True,
            verify_hashes=True,
            max_depth=None,
            required_roles="config",  # Should trace back to spec which has config role
            quiet=True,
        )
        assert result == 0

        # Step 6: Verify lineage integrity
        ancestors = lineage.get_ancestors("art-board-001")
        assert "art-design-001" in ancestors.nodes
        assert "art-coupon-spec-001" in ancestors.nodes

        descendants = lineage.get_descendants("art-coupon-spec-001")
        assert "art-design-001" in descendants.nodes
        assert "art-board-001" in descendants.nodes

        # Clean up
        registry.close()
        lineage.close()


class TestDeterminismAcrossRuns:
    """Tests ensuring deterministic behavior across multiple runs."""

    @pytest.fixture
    def initialized_store(self, tmp_path: Path) -> tuple[Path, ArtifactStore, ArtifactRegistry]:
        """Create an initialized store for determinism tests."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        data_dir = tmp_path / "data"
        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")
        return tmp_path, store, registry

    def test_same_content_same_hash(self, initialized_store: tuple[Path, ArtifactStore, ArtifactRegistry]) -> None:
        """Same content should always produce the same content hash."""
        _, store, registry = initialized_store

        content = b"deterministic test content"

        # Create first artifact
        manifest1 = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        registry.index_artifact(manifest1)

        # Create second artifact with same content
        manifest2 = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-002",
        )
        registry.index_artifact(manifest2)

        # Content hashes should be identical
        assert manifest1.content_hash.digest == manifest2.content_hash.digest
        assert manifest1.content_hash.algorithm == manifest2.content_hash.algorithm

        # spec_id should be identical
        assert manifest1.spec_id == manifest2.spec_id

        registry.close()

    def test_spec_id_consistency(self, initialized_store: tuple[Path, ArtifactStore, ArtifactRegistry]) -> None:
        """Spec ID computation should be consistent across sessions."""
        _, store, registry = initialized_store

        content = b'{"test": "spec_id_consistency"}'

        manifest = store.put(
            content=content,
            artifact_type="coupon_spec",
            roles=["config"],
            run_id="run-001",
        )
        registry.index_artifact(manifest)

        # Compute spec ID directly
        computed_spec_id = store.compute_spec_id(content)

        # Should match the manifest's spec_id
        assert manifest.spec_id == computed_spec_id

        # Re-retrieve and check
        retrieved_manifest = store.get_manifest(manifest.artifact_id)
        assert retrieved_manifest.spec_id == computed_spec_id

        registry.close()

    def test_dataset_hash_determinism(self, initialized_store: tuple[Path, ArtifactStore, ArtifactRegistry]) -> None:
        """Dataset manifest hash should be deterministic regardless of member order."""
        _, store, registry = initialized_store

        # Create three artifacts
        manifests = []
        for i in range(3):
            manifest = store.put(
                content=f"content {i}".encode(),
                artifact_type="other",
                roles=["intermediate"],
                run_id="run-001",
            )
            registry.index_artifact(manifest)
            manifests.append(manifest)

        # Create dataset with members in order A, B, C
        from formula_foundry.m3.dataset_snapshot import DatasetMember

        members_abc = [
            DatasetMember.from_manifest(manifests[0], role="test"),
            DatasetMember.from_manifest(manifests[1], role="test"),
            DatasetMember.from_manifest(manifests[2], role="test"),
        ]
        hash_abc = compute_manifest_hash(members_abc)

        # Create dataset with members in order C, A, B
        members_cab = [
            DatasetMember.from_manifest(manifests[2], role="test"),
            DatasetMember.from_manifest(manifests[0], role="test"),
            DatasetMember.from_manifest(manifests[1], role="test"),
        ]
        hash_cab = compute_manifest_hash(members_cab)

        # Hashes should be identical (order independent)
        assert hash_abc.digest == hash_cab.digest

        registry.close()

    def test_lineage_graph_deterministic_traversal(
        self, initialized_store: tuple[Path, ArtifactStore, ArtifactRegistry]
    ) -> None:
        """Lineage graph traversal should be deterministic."""
        root, store, registry = initialized_store
        data_dir = root / "data"

        lineage = LineageGraph(data_dir / "lineage.sqlite")
        lineage.initialize()

        # Create a chain of artifacts: root -> middle -> leaf
        root_manifest = store.put(
            content=b"root",
            artifact_type="coupon_spec",
            roles=["root_input"],
            run_id="run-001",
            artifact_id="art-root",
        )
        registry.index_artifact(root_manifest)
        lineage.add_manifest(root_manifest)

        middle_manifest = store.put(
            content=b"middle",
            artifact_type="resolved_design",
            roles=["geometry"],
            run_id="run-001",
            artifact_id="art-middle",
            inputs=[LineageReference(artifact_id="art-root", relation="derived_from")],
        )
        registry.index_artifact(middle_manifest)
        lineage.add_manifest(middle_manifest)

        leaf_manifest = store.put(
            content=b"leaf",
            artifact_type="kicad_board",
            roles=["final_output"],
            run_id="run-001",
            artifact_id="art-leaf",
            inputs=[LineageReference(artifact_id="art-middle", relation="derived_from")],
        )
        registry.index_artifact(leaf_manifest)
        lineage.add_manifest(leaf_manifest)

        # Multiple traversals should return the same result
        ancestors1 = lineage.get_ancestors("art-leaf")
        ancestors2 = lineage.get_ancestors("art-leaf")

        assert set(ancestors1.nodes.keys()) == set(ancestors2.nodes.keys())
        assert ancestors1.edge_count == ancestors2.edge_count

        # Trace to roots should be deterministic
        roots1 = lineage.trace_to_roots("art-leaf")
        roots2 = lineage.trace_to_roots("art-leaf")

        assert roots1.get_roots() == roots2.get_roots()

        registry.close()
        lineage.close()


class TestComponentInteraction:
    """Tests for interaction between M3 components."""

    @pytest.fixture
    def full_setup(self, tmp_path: Path) -> dict[str, Any]:
        """Create a full M3 setup with all components."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        data_dir = tmp_path / "data"

        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")
        lineage = LineageGraph(data_dir / "lineage.sqlite")
        lineage.initialize()

        return {
            "root": tmp_path,
            "data_dir": data_dir,
            "store": store,
            "registry": registry,
            "lineage": lineage,
        }

    def test_store_to_registry_sync(self, full_setup: dict[str, Any]) -> None:
        """Registry should accurately reflect store contents."""
        store = full_setup["store"]
        registry = full_setup["registry"]

        # Add artifacts to store and registry
        manifests = []
        for i in range(5):
            manifest = store.put(
                content=f"content {i}".encode(),
                artifact_type="other" if i % 2 == 0 else "touchstone",
                roles=["intermediate"],
                run_id="run-001",
            )
            registry.index_artifact(manifest)
            manifests.append(manifest)

        # Registry counts should match
        assert registry.count_artifacts() == 5
        assert registry.count_artifacts(artifact_type="other") == 3
        assert registry.count_artifacts(artifact_type="touchstone") == 2

        # Each artifact should be queryable
        for manifest in manifests:
            record = registry.get_artifact(manifest.artifact_id)
            assert record.content_hash_digest == manifest.content_hash.digest
            assert record.byte_size == manifest.byte_size

        registry.close()

    def test_registry_rebuild_from_store(self, full_setup: dict[str, Any]) -> None:
        """Registry should be rebuildable from store manifests."""
        store = full_setup["store"]
        registry = full_setup["registry"]

        # Add some artifacts
        for i in range(3):
            manifest = store.put(
                content=f"rebuild test {i}".encode(),
                artifact_type="other",
                roles=["intermediate"],
                run_id="run-001",
            )
            registry.index_artifact(manifest)

        original_count = registry.count_artifacts()

        # Clear and rebuild
        rebuilt_count = registry.rebuild_from_store(store, clear_first=True)

        # Should have same number
        assert rebuilt_count == original_count

        registry.close()

    def test_lineage_from_store(self, full_setup: dict[str, Any]) -> None:
        """Lineage graph should be buildable from store."""
        store = full_setup["store"]
        lineage = full_setup["lineage"]

        # Create artifacts with lineage
        root = store.put(
            content=b"root artifact",
            artifact_type="coupon_spec",
            roles=["root_input"],
            run_id="run-001",
            artifact_id="art-lineage-root",
        )
        lineage.add_manifest(root)

        child = store.put(
            content=b"child artifact",
            artifact_type="resolved_design",
            roles=["geometry"],
            run_id="run-001",
            artifact_id="art-lineage-child",
            inputs=[LineageReference(artifact_id="art-lineage-root", relation="derived_from")],
        )
        lineage.add_manifest(child)

        # Build from store
        rebuilt_count = lineage.build_from_store(store, clear_first=True)

        # Should have rebuilt the graph
        assert rebuilt_count == 2
        assert lineage.has_node("art-lineage-root")
        assert lineage.has_node("art-lineage-child")

        # Edges should be preserved
        edges = lineage.get_edges_to("art-lineage-child")
        assert len(edges) == 1
        assert edges[0].source_id == "art-lineage-root"

        lineage.close()

    def test_gc_respects_lineage(self, full_setup: dict[str, Any]) -> None:
        """GC should protect artifacts with descendants when configured."""
        store = full_setup["store"]
        registry = full_setup["registry"]
        lineage = full_setup["lineage"]
        data_dir = full_setup["data_dir"]

        # Create parent and child artifacts
        parent = store.put(
            content=b"parent",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
            artifact_id="art-parent",
        )
        registry.index_artifact(parent)
        lineage.add_manifest(parent)

        child = store.put(
            content=b"child",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
            artifact_id="art-child",
            inputs=[LineageReference(artifact_id="art-parent", relation="derived_from")],
        )
        registry.index_artifact(child)
        lineage.add_manifest(child)

        # Create GC with policy protecting descendants
        gc = GarbageCollector(
            data_dir=data_dir,
            store=store,
            registry=registry,
            lineage=lineage,
        )

        policy = RetentionPolicy(
            name="test_protect_descendants",
            keep_min_age_days=0,
            keep_min_count=0,
            keep_pinned=True,
            keep_with_descendants=True,  # Protect artifacts with children
        )

        to_delete, to_keep = gc.compute_candidates(policy)

        # Parent should be kept because it has descendants
        parent_kept = [c for c in to_keep if c.artifact_id == "art-parent"]
        assert len(parent_kept) == 1
        assert any("has_descendants" in r for r in parent_kept[0].reasons_to_keep)

        registry.close()
        lineage.close()


class TestDatasetLifecycle:
    """Tests for dataset snapshot lifecycle."""

    @pytest.fixture
    def project_with_store(self, tmp_path: Path) -> tuple[Path, ArtifactStore, ArtifactRegistry]:
        """Create project with store for dataset tests."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        data_dir = tmp_path / "data"
        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")
        return tmp_path, store, registry

    def test_dataset_create_read_verify(
        self, project_with_store: tuple[Path, ArtifactStore, ArtifactRegistry]
    ) -> None:
        """Test complete dataset lifecycle: create, read, verify."""
        root, store, registry = project_with_store
        data_dir = root / "data"
        dataset_dir = data_dir / "datasets"

        # Create artifacts
        manifests = []
        for i in range(5):
            manifest = store.put(
                content=f"dataset member {i}".encode(),
                artifact_type="touchstone",
                roles=["oracle_output"],
                run_id="run-001",
                tags={"index": str(i)},
            )
            registry.index_artifact(manifest)
            manifests.append(manifest)

        # Create dataset
        writer = DatasetSnapshotWriter(
            dataset_id="test_dataset",
            version="v1.0",
            store=store,
            generator="test",
            generator_version="1.0.0",
            name="Test Dataset",
            description="A test dataset for integration testing",
        )

        for i, manifest in enumerate(manifests):
            writer.add_member(manifest, role="data", features={"index": i, "value": i * 10})

        snapshot = writer.finalize(output_dir=dataset_dir, write_parquet=False)

        # Register dataset
        registry.index_dataset_snapshot(snapshot)

        # Verify dataset was created correctly
        assert snapshot.member_count == 5
        assert snapshot.dataset_id == "test_dataset"
        assert snapshot.version == "v1.0"

        # Read dataset back
        manifest_path = dataset_dir / "test_dataset_v1.0.json"
        assert manifest_path.exists()

        reader = DatasetSnapshotReader(snapshot_path=manifest_path, store=store)
        loaded_snapshot = reader.load()

        assert loaded_snapshot.member_count == 5
        assert loaded_snapshot.content_hash.digest == snapshot.content_hash.digest

        # Verify integrity
        is_valid, errors = reader.verify_integrity()
        assert is_valid
        assert len(errors) == 0

        registry.close()

    def test_dataset_versioning(
        self, project_with_store: tuple[Path, ArtifactStore, ArtifactRegistry]
    ) -> None:
        """Test creating multiple versions of a dataset."""
        root, store, registry = project_with_store
        data_dir = root / "data"
        dataset_dir = data_dir / "datasets"

        # Create artifacts for v1
        v1_manifests = []
        for i in range(3):
            manifest = store.put(
                content=f"v1 content {i}".encode(),
                artifact_type="touchstone",
                roles=["oracle_output"],
                run_id="run-001",
            )
            registry.index_artifact(manifest)
            v1_manifests.append(manifest)

        # Create v1 dataset
        writer_v1 = DatasetSnapshotWriter(
            dataset_id="versioned_dataset",
            version="v1.0",
            store=store,
            generator="test",
            generator_version="1.0.0",
        )
        for m in v1_manifests:
            writer_v1.add_member(m, role="data")
        snapshot_v1 = writer_v1.finalize(output_dir=dataset_dir, write_parquet=False)

        # Create artifacts for v2 (superset of v1)
        v2_manifests = list(v1_manifests)
        for i in range(2):
            manifest = store.put(
                content=f"v2 content {i}".encode(),
                artifact_type="touchstone",
                roles=["oracle_output"],
                run_id="run-002",
            )
            registry.index_artifact(manifest)
            v2_manifests.append(manifest)

        # Create v2 dataset
        writer_v2 = DatasetSnapshotWriter(
            dataset_id="versioned_dataset",
            version="v2.0",
            store=store,
            generator="test",
            generator_version="1.0.0",
            parent_version="v1.0",
        )
        for m in v2_manifests:
            writer_v2.add_member(m, role="data")
        snapshot_v2 = writer_v2.finalize(output_dir=dataset_dir, write_parquet=False)

        # Verify versions
        assert snapshot_v1.member_count == 3
        assert snapshot_v2.member_count == 5
        assert snapshot_v2.parent_version == "v1.0"

        # Content hashes should be different
        assert snapshot_v1.content_hash.digest != snapshot_v2.content_hash.digest

        registry.close()


class TestGCIntegration:
    """Integration tests for garbage collection."""

    @pytest.fixture
    def gc_project(self, tmp_path: Path) -> dict[str, Any]:
        """Create project for GC integration tests."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        data_dir = tmp_path / "data"

        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")
        lineage = LineageGraph(data_dir / "lineage.sqlite")
        lineage.initialize()

        gc = GarbageCollector(
            data_dir=data_dir,
            store=store,
            registry=registry,
            lineage=lineage,
        )

        return {
            "root": tmp_path,
            "data_dir": data_dir,
            "store": store,
            "registry": registry,
            "lineage": lineage,
            "gc": gc,
        }

    def test_gc_actual_deletion(self, gc_project: dict[str, Any]) -> None:
        """Test that GC actually deletes artifacts when not in dry-run mode."""
        store = gc_project["store"]
        registry = gc_project["registry"]
        gc = gc_project["gc"]

        # Create an artifact
        manifest = store.put(
            content=b"deletable content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        registry.index_artifact(manifest)

        # Verify it exists
        assert store.exists_by_id(manifest.artifact_id)
        assert registry.count_artifacts() >= 1

        # Create a very aggressive policy
        policy = RetentionPolicy(
            name="delete_all",
            keep_min_age_days=0,
            keep_min_count=0,
            keep_pinned=False,
            keep_with_descendants=False,
        )

        # Run GC (not dry-run)
        result = gc.run(policy=policy, dry_run=False, run_dvc_gc=False)

        assert result.dry_run is False
        assert result.artifacts_deleted >= 1

        # Artifact should be gone
        assert not store.exists_by_id(manifest.artifact_id)

        registry.close()

    def test_gc_pin_persistence(self, gc_project: dict[str, Any]) -> None:
        """Test that pins are persisted across GC instances."""
        gc = gc_project["gc"]
        data_dir = gc_project["data_dir"]
        store = gc_project["store"]
        registry = gc_project["registry"]
        lineage = gc_project["lineage"]

        # Create and pin an artifact
        manifest = store.put(
            content=b"pinned content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        registry.index_artifact(manifest)

        gc.pin_artifact(artifact_id=manifest.artifact_id, reason="test persistence")

        # Create a new GC instance (simulating new session)
        gc2 = GarbageCollector(
            data_dir=data_dir,
            store=store,
            registry=registry,
            lineage=lineage,
        )

        # Pin should still be there
        assert gc2.is_pinned(manifest.artifact_id)

        # Cleanup
        gc2.unpin_artifact(artifact_id=manifest.artifact_id)
        assert not gc2.is_pinned(manifest.artifact_id)

        registry.close()
        lineage.close()

    def test_gc_estimate_accuracy(self, gc_project: dict[str, Any]) -> None:
        """Test that GC estimates match actual results."""
        store = gc_project["store"]
        registry = gc_project["registry"]
        gc = gc_project["gc"]

        # Create artifacts
        for i in range(5):
            manifest = store.put(
                content=f"estimable content {i}".encode(),
                artifact_type="other",
                roles=["intermediate"],
                run_id="run-001",
            )
            registry.index_artifact(manifest)

        policy = RetentionPolicy(
            name="test_estimate",
            keep_min_age_days=0,
            keep_min_count=2,  # Keep 2, delete 3
            keep_pinned=False,
            keep_with_descendants=False,
        )

        # Get estimate
        estimate = gc.estimate_savings(policy)

        # Dry run
        result = gc.run(policy=policy, dry_run=True, run_dvc_gc=False)

        # Estimate should match dry run
        assert estimate["artifacts_to_delete"] == result.artifacts_deleted
        assert estimate["bytes_to_delete"] == result.bytes_freed

        registry.close()


class TestAuditIntegration:
    """Integration tests for audit functionality."""

    @pytest.fixture
    def auditable_project(self, tmp_path: Path) -> tuple[Path, list[str]]:
        """Create a project with a lineage chain for auditing."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        data_dir = tmp_path / "data"

        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")
        lineage = LineageGraph(data_dir / "lineage.sqlite")
        lineage.initialize()

        # Create a complex lineage chain:
        # spec1 ─┬─> design1 ─> board1
        # spec2 ─┘

        spec1 = store.put(
            content=b"spec 1 content",
            artifact_type="coupon_spec",
            roles=["config", "root_input"],
            run_id="run-001",
            artifact_id="art-spec1",
        )
        registry.index_artifact(spec1)
        lineage.add_manifest(spec1)

        spec2 = store.put(
            content=b"spec 2 content (stackup)",
            artifact_type="coupon_spec",
            roles=["config", "root_input"],
            run_id="run-001",
            artifact_id="art-spec2",
        )
        registry.index_artifact(spec2)
        lineage.add_manifest(spec2)

        design1 = store.put(
            content=b"design 1 content",
            artifact_type="resolved_design",
            roles=["geometry"],
            run_id="run-002",
            artifact_id="art-design1",
            inputs=[
                LineageReference(artifact_id="art-spec1", relation="config_from"),
                LineageReference(artifact_id="art-spec2", relation="config_from"),
            ],
        )
        registry.index_artifact(design1)
        lineage.add_manifest(design1)

        board1 = store.put(
            content=b"board 1 content",
            artifact_type="kicad_board",
            roles=["final_output"],
            run_id="run-003",
            artifact_id="art-board1",
            inputs=[
                LineageReference(artifact_id="art-design1", relation="derived_from"),
            ],
        )
        registry.index_artifact(board1)
        lineage.add_manifest(board1)

        registry.close()
        lineage.close()

        return tmp_path, ["art-spec1", "art-spec2", "art-design1", "art-board1"]

    def test_audit_full_lineage_trace(
        self, auditable_project: tuple[Path, list[str]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test audit can trace full lineage to roots."""
        project_root, artifact_ids = auditable_project

        result = cmd_audit(
            artifact_id="art-board1",
            root=project_root,
            output_format="json",
            trace_roots=True,
            verify_hashes=False,
            max_depth=None,
            required_roles=None,
            quiet=True,
        )
        assert result == 0

        captured = capsys.readouterr()
        report = json.loads(captured.out)

        # Should have found the artifact with ancestors
        assert report["total_artifacts"] >= 1
        artifact = report["artifacts"][0]
        assert artifact["artifact_id"] == "art-board1"

        # Should have traced back to both root specs
        assert artifact["ancestor_count"] == 3  # design1, spec1, spec2

    def test_audit_all_artifacts(
        self, auditable_project: tuple[Path, list[str]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test auditing all artifacts at once."""
        project_root, artifact_ids = auditable_project

        result = cmd_audit(
            artifact_id=None,  # Audit all
            root=project_root,
            output_format="json",
            trace_roots=False,
            verify_hashes=False,
            max_depth=None,
            required_roles=None,
            quiet=True,
        )
        assert result == 0

        captured = capsys.readouterr()
        report = json.loads(captured.out)

        assert report["total_artifacts"] == 4

    def test_audit_with_hash_verification_success(self, auditable_project: tuple[Path, list[str]]) -> None:
        """Test that audit passes hash verification for intact artifacts."""
        project_root, artifact_ids = auditable_project

        result = cmd_audit(
            artifact_id="art-board1",
            root=project_root,
            output_format="text",
            trace_roots=False,
            verify_hashes=True,
            max_depth=None,
            required_roles=None,
            quiet=True,
        )
        assert result == 0

    def test_audit_required_roles_success(self, auditable_project: tuple[Path, list[str]]) -> None:
        """Test audit passes when required roles exist in roots."""
        project_root, artifact_ids = auditable_project

        result = cmd_audit(
            artifact_id="art-board1",
            root=project_root,
            output_format="text",
            trace_roots=True,
            verify_hashes=False,
            max_depth=None,
            required_roles="config,root_input",  # Both specs have these
            quiet=True,
        )
        assert result == 0

    def test_audit_required_roles_failure(self, auditable_project: tuple[Path, list[str]]) -> None:
        """Test audit fails when required roles are missing."""
        project_root, artifact_ids = auditable_project

        result = cmd_audit(
            artifact_id="art-board1",
            root=project_root,
            output_format="text",
            trace_roots=True,
            verify_hashes=False,
            max_depth=None,
            required_roles="simulation_result",  # No artifact has this
            quiet=True,
        )
        assert result == 2


class TestErrorRecovery:
    """Tests for error handling and recovery scenarios."""

    def test_registry_recovery_from_corrupted_db(self, tmp_path: Path) -> None:
        """Registry should be rebuildable after removing corrupted DB."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        data_dir = tmp_path / "data"

        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")

        # Add some artifacts to store
        for i in range(3):
            store.put(
                content=f"recovery test {i}".encode(),
                artifact_type="other",
                roles=["intermediate"],
                run_id="run-001",
            )

        # Simulate recovery by removing the database and recreating
        db_path = data_dir / "registry.db"
        # Also remove WAL files if they exist
        for wal_file in data_dir.glob("registry.db*"):
            wal_file.unlink()

        # Create new registry and rebuild from store manifests
        registry = ArtifactRegistry(db_path)
        registry.initialize()
        rebuilt_count = registry.rebuild_from_store(store, clear_first=True)

        # Should have rebuilt all artifacts
        assert rebuilt_count == 3
        assert registry.count_artifacts() == 3

        registry.close()

    def test_store_handles_missing_object(self, tmp_path: Path) -> None:
        """Store should raise appropriate error for missing objects."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        data_dir = tmp_path / "data"

        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")

        from formula_foundry.m3.artifact_store import ArtifactNotFoundError

        # Try to get a non-existent object
        with pytest.raises(ArtifactNotFoundError):
            store.get("0" * 64)

    def test_lineage_handles_orphan_edges(self, tmp_path: Path) -> None:
        """Lineage graph should handle edges to non-existent nodes gracefully."""
        cmd_init(root=tmp_path, force=False, quiet=True)
        data_dir = tmp_path / "data"

        lineage = LineageGraph(data_dir / "lineage.sqlite")
        lineage.initialize()

        # Add node
        lineage.add_node(
            artifact_id="art-child",
            artifact_type="other",
            content_hash_digest="a" * 64,
        )

        # Add edge to non-existent parent
        lineage.add_edge(
            source_id="art-parent-missing",
            target_id="art-child",
            relation="derived_from",
        )

        # Ancestor query should still work (but not find the missing parent as a node)
        ancestors = lineage.get_ancestors("art-child")
        assert "art-child" in ancestors.nodes
        # The edge exists but parent node doesn't
        assert len(ancestors.edges) == 1

        lineage.close()


class TestCLIIntegration:
    """Integration tests for the M3 CLI commands."""

    @pytest.fixture
    def cli_project(self, tmp_path: Path) -> Path:
        """Create a project for CLI tests."""
        # Initialize git
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path, check=True, capture_output=True
        )
        (tmp_path / "test.txt").write_text("test")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            cwd=tmp_path, check=True, capture_output=True
        )

        # Create DVC config
        (tmp_path / "dvc.yaml").write_text(
            "stages:\n"
            "  test_stage:\n"
            "    cmd: echo 'test'\n"
        )

        return tmp_path

    def test_cli_workflow_init_audit(self, cli_project: Path) -> None:
        """Test CLI workflow: init -> audit."""
        # Init
        result = main(["init", "--root", str(cli_project), "-q"])
        assert result == 0

        # Audit (should succeed with empty store)
        result = main(["audit", "--root", str(cli_project), "-q"])
        assert result == 0

    def test_cli_gc_list_policies(self, cli_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Test gc --list-policies command."""
        main(["init", "--root", str(cli_project), "-q"])

        result = main(["gc", "--root", str(cli_project), "--list-policies"])
        assert result == 0

        captured = capsys.readouterr()
        assert "laptop_default" in captured.out
        assert "ci_aggressive" in captured.out
        assert "archive" in captured.out

    def test_cli_artifact_list_empty(self, cli_project: Path) -> None:
        """Test artifact list on empty store."""
        main(["init", "--root", str(cli_project), "-q"])

        result = main(["artifact", "list", "--root", str(cli_project), "-q"])
        assert result == 0

    def test_cli_dataset_show_not_found(self, cli_project: Path) -> None:
        """Test dataset show for non-existent dataset."""
        main(["init", "--root", str(cli_project), "-q"])

        result = main(["dataset", "show", "nonexistent", "--root", str(cli_project), "-q"])
        assert result == 2  # Should fail


class TestConcurrencyAndThreadSafety:
    """Tests for thread safety of M3 components."""

    def test_registry_thread_safety(self, tmp_path: Path) -> None:
        """Test that registry handles concurrent access correctly."""
        import threading
        import queue

        cmd_init(root=tmp_path, force=False, quiet=True)
        data_dir = tmp_path / "data"

        store = ArtifactStore(root=data_dir, generator="test", generator_version="1.0.0")
        registry = ArtifactRegistry(data_dir / "registry.db")

        errors: queue.Queue[Exception] = queue.Queue()

        def worker(thread_id: int) -> None:
            try:
                for i in range(10):
                    manifest = store.put(
                        content=f"thread {thread_id} content {i}".encode(),
                        artifact_type="other",
                        roles=["intermediate"],
                        run_id=f"run-thread-{thread_id}",
                    )
                    registry.index_artifact(manifest)
            except Exception as e:
                errors.put(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check for errors
        assert errors.empty(), f"Thread errors: {[errors.get() for _ in range(errors.qsize())]}"

        # Verify all artifacts were indexed
        assert registry.count_artifacts() == 50  # 5 threads * 10 artifacts each

        registry.close()

    def test_lineage_thread_safety(self, tmp_path: Path) -> None:
        """Test that lineage graph handles concurrent access correctly."""
        import threading
        import queue

        cmd_init(root=tmp_path, force=False, quiet=True)
        data_dir = tmp_path / "data"

        lineage = LineageGraph(data_dir / "lineage.sqlite")
        lineage.initialize()

        errors: queue.Queue[Exception] = queue.Queue()

        def worker(thread_id: int) -> None:
            try:
                for i in range(10):
                    lineage.add_node(
                        artifact_id=f"art-thread{thread_id}-{i}",
                        artifact_type="other",
                        content_hash_digest=f"{thread_id:032x}{i:032x}",
                    )
            except Exception as e:
                errors.put(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check for errors
        assert errors.empty(), f"Thread errors: {[errors.get() for _ in range(errors.qsize())]}"

        # Verify all nodes were added
        assert lineage.count_nodes() == 50

        lineage.close()
