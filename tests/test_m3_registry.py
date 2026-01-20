"""Tests for the M3 ArtifactRegistry with SQLite-backed indexing."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from formula_foundry.m3.artifact_store import ArtifactStore
from formula_foundry.m3.registry import (
    ArtifactNotIndexedError,
    ArtifactRecord,
    ArtifactRegistry,
    DatasetRecord,
    RunRecord,
)


class TestArtifactRegistryInit:
    """Tests for registry initialization."""

    def test_initialize_creates_database(self, tmp_path: Path) -> None:
        """Test that initialize creates the database file."""
        db_path = tmp_path / "registry.db"
        registry = ArtifactRegistry(db_path)
        registry.initialize()

        assert db_path.exists()
        registry.close()

    def test_initialize_creates_tables(self, tmp_path: Path) -> None:
        """Test that initialize creates all required tables."""
        db_path = tmp_path / "registry.db"
        registry = ArtifactRegistry(db_path)
        registry.initialize()

        # Verify tables exist by attempting queries
        conn = registry._get_connection()
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        cursor.close()

        assert "artifacts" in tables
        assert "datasets" in tables
        assert "runs" in tables
        assert "schema_version" in tables

        registry.close()

    def test_initialize_idempotent(self, tmp_path: Path) -> None:
        """Test that initialize can be called multiple times safely."""
        db_path = tmp_path / "registry.db"
        registry = ArtifactRegistry(db_path)

        registry.initialize()
        registry.initialize()
        registry.initialize()

        # Should not raise
        assert registry.count_artifacts() == 0
        registry.close()


class TestArtifactIndexing:
    """Tests for artifact indexing."""

    def test_index_artifact_from_manifest(self, tmp_path: Path) -> None:
        """Test indexing an artifact from a manifest."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        manifest = store.put(
            content=b"test content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        registry.index_artifact(manifest)

        record = registry.get_artifact(manifest.artifact_id)
        assert record.artifact_id == manifest.artifact_id
        assert record.artifact_type == "other"
        assert record.content_hash_digest == manifest.content_hash.digest
        assert record.byte_size == len(b"test content")
        assert "intermediate" in record.roles

        registry.close()

    def test_index_artifact_updates_existing(self, tmp_path: Path) -> None:
        """Test that indexing the same artifact updates the record."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        manifest = store.put(
            content=b"test content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        registry.index_artifact(manifest)
        registry.index_artifact(manifest)  # Second index should update

        assert registry.count_artifacts() == 1
        registry.close()

    def test_get_artifact_not_found(self, tmp_path: Path) -> None:
        """Test that get_artifact raises for missing artifact."""
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        with pytest.raises(ArtifactNotIndexedError):
            registry.get_artifact("nonexistent-id")

        registry.close()


class TestRunIndexing:
    """Tests for run indexing."""

    def test_index_run_basic(self, tmp_path: Path) -> None:
        """Test basic run indexing."""
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        registry.index_run(
            run_id="run-001",
            started_utc="2026-01-20T09:00:00Z",
            status="in_progress",
            hostname="testhost",
            generator="test_gen",
            generator_version="1.0.0",
        )

        record = registry.get_run("run-001")
        assert record.run_id == "run-001"
        assert record.status == "in_progress"
        assert record.hostname == "testhost"

        registry.close()

    def test_update_run_status(self, tmp_path: Path) -> None:
        """Test updating run status."""
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        registry.index_run(
            run_id="run-001",
            started_utc="2026-01-20T09:00:00Z",
            status="in_progress",
        )

        registry.update_run_status(
            run_id="run-001",
            status="completed",
            ended_utc="2026-01-20T09:30:00Z",
        )

        record = registry.get_run("run-001")
        assert record.status == "completed"
        assert record.ended_utc == "2026-01-20T09:30:00Z"

        registry.close()

    def test_run_created_from_artifact(self, tmp_path: Path) -> None:
        """Test that indexing an artifact creates a run record."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        manifest = store.put(
            content=b"test",
            artifact_type="other",
            roles=["intermediate"],
            run_id="auto-run-001",
        )

        registry.index_artifact(manifest)

        record = registry.get_run("auto-run-001")
        assert record.run_id == "auto-run-001"
        assert record.artifact_count == 1

        registry.close()


class TestDatasetIndexing:
    """Tests for dataset indexing."""

    def test_index_dataset(self, tmp_path: Path) -> None:
        """Test basic dataset indexing."""
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        registry.index_dataset(
            dataset_id="ds-001",
            version="v1.0.0",
            artifact_count=100,
            total_bytes=1024000,
            created_utc="2026-01-20T09:00:00Z",
            description="Test dataset",
        )

        record = registry.get_dataset("ds-001")
        assert record.dataset_id == "ds-001"
        assert record.version == "v1.0.0"
        assert record.artifact_count == 100
        assert record.total_bytes == 1024000
        assert record.description == "Test dataset"

        registry.close()

    def test_get_dataset_not_found(self, tmp_path: Path) -> None:
        """Test that get_dataset raises for missing dataset."""
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        with pytest.raises(ArtifactNotIndexedError):
            registry.get_dataset("nonexistent-ds")

        registry.close()


class TestQueryArtifacts:
    """Tests for artifact querying."""

    def test_query_by_type(self, tmp_path: Path) -> None:
        """Test querying artifacts by type."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        # Create artifacts of different types
        m1 = store.put(
            content=b"coupon",
            artifact_type="coupon_spec",
            roles=["geometry"],
            run_id="run-001",
        )
        m2 = store.put(
            content=b"touchstone",
            artifact_type="touchstone",
            roles=["oracle_output"],
            run_id="run-001",
        )
        m3 = store.put(
            content=b"another coupon",
            artifact_type="coupon_spec",
            roles=["geometry"],
            run_id="run-001",
        )

        for m in [m1, m2, m3]:
            registry.index_artifact(m)

        coupon_specs = registry.query_artifacts(artifact_type="coupon_spec")
        assert len(coupon_specs) == 2

        touchstones = registry.query_artifacts(artifact_type="touchstone")
        assert len(touchstones) == 1

        registry.close()

    def test_query_by_run(self, tmp_path: Path) -> None:
        """Test querying artifacts by run ID."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        m1 = store.put(
            content=b"run1-a",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        m2 = store.put(
            content=b"run1-b",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        m3 = store.put(
            content=b"run2-a",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-002",
        )

        for m in [m1, m2, m3]:
            registry.index_artifact(m)

        run1_artifacts = registry.query_artifacts(run_id="run-001")
        assert len(run1_artifacts) == 2

        run2_artifacts = registry.query_artifacts(run_id="run-002")
        assert len(run2_artifacts) == 1

        registry.close()

    def test_query_with_limit_offset(self, tmp_path: Path) -> None:
        """Test pagination with limit and offset."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        # Create 10 artifacts
        for i in range(10):
            m = store.put(
                content=f"content-{i}".encode(),
                artifact_type="other",
                roles=["intermediate"],
                run_id="run-001",
            )
            registry.index_artifact(m)

        # First page
        page1 = registry.query_artifacts(limit=3, offset=0)
        assert len(page1) == 3

        # Second page
        page2 = registry.query_artifacts(limit=3, offset=3)
        assert len(page2) == 3

        # Pages should have different artifacts
        page1_ids = {r.artifact_id for r in page1}
        page2_ids = {r.artifact_id for r in page2}
        assert page1_ids.isdisjoint(page2_ids)

        registry.close()

    def test_query_by_roles(self, tmp_path: Path) -> None:
        """Test querying artifacts by roles."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        m1 = store.put(
            content=b"geometry",
            artifact_type="coupon_spec",
            roles=["geometry", "config"],
            run_id="run-001",
        )
        m2 = store.put(
            content=b"output",
            artifact_type="touchstone",
            roles=["oracle_output", "final_output"],
            run_id="run-001",
        )

        registry.index_artifact(m1)
        registry.index_artifact(m2)

        geometry = registry.query_artifacts(roles=["geometry"])
        assert len(geometry) == 1
        assert geometry[0].artifact_id == m1.artifact_id

        outputs = registry.query_artifacts(roles=["oracle_output", "geometry"])
        assert len(outputs) == 2  # Both match at least one role

        registry.close()


class TestQueryRuns:
    """Tests for run querying."""

    def test_query_runs_by_status(self, tmp_path: Path) -> None:
        """Test querying runs by status."""
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        registry.index_run(
            run_id="run-001",
            started_utc="2026-01-20T09:00:00Z",
            status="completed",
        )
        registry.index_run(
            run_id="run-002",
            started_utc="2026-01-20T09:10:00Z",
            status="in_progress",
        )
        registry.index_run(
            run_id="run-003",
            started_utc="2026-01-20T09:20:00Z",
            status="completed",
        )

        completed = registry.query_runs(status="completed")
        assert len(completed) == 2

        in_progress = registry.query_runs(status="in_progress")
        assert len(in_progress) == 1

        registry.close()


class TestRebuildFromStore:
    """Tests for rebuilding the registry from a store."""

    def test_rebuild_indexes_all_manifests(self, tmp_path: Path) -> None:
        """Test that rebuild indexes all manifests from the store."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        # Create artifacts in the store
        manifests = []
        for i in range(5):
            m = store.put(
                content=f"content-{i}".encode(),
                artifact_type="other",
                roles=["intermediate"],
                run_id="run-001",
            )
            manifests.append(m)

        # Rebuild from store
        count = registry.rebuild_from_store(store)
        assert count == 5
        assert registry.count_artifacts() == 5

        # Verify all artifacts are indexed
        for m in manifests:
            record = registry.get_artifact(m.artifact_id)
            assert record.content_hash_digest == m.content_hash.digest

        registry.close()

    def test_rebuild_clears_existing(self, tmp_path: Path) -> None:
        """Test that rebuild clears existing data by default."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        # Add some data directly
        registry.index_dataset(
            dataset_id="ds-old",
            version="v0.1",
            artifact_count=10,
            total_bytes=1000,
            created_utc="2026-01-01T00:00:00Z",
        )

        # Create new artifacts
        store.put(
            content=b"new content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        # Rebuild should clear the dataset
        registry.rebuild_from_store(store)

        with pytest.raises(ArtifactNotIndexedError):
            registry.get_dataset("ds-old")

        registry.close()


class TestDeleteOperations:
    """Tests for delete operations."""

    def test_delete_artifact(self, tmp_path: Path) -> None:
        """Test deleting an artifact from the index."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        m = store.put(
            content=b"to delete",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        registry.index_artifact(m)

        assert registry.delete_artifact(m.artifact_id)

        with pytest.raises(ArtifactNotIndexedError):
            registry.get_artifact(m.artifact_id)

        registry.close()

    def test_delete_nonexistent_returns_false(self, tmp_path: Path) -> None:
        """Test that deleting nonexistent artifact returns False."""
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        assert not registry.delete_artifact("nonexistent-id")

        registry.close()


class TestStorageStats:
    """Tests for storage statistics."""

    def test_get_storage_stats(self, tmp_path: Path) -> None:
        """Test getting storage statistics."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        # Create artifacts with some duplicates
        m1 = store.put(
            content=b"unique content 1",
            artifact_type="coupon_spec",
            roles=["geometry"],
            run_id="run-001",
        )
        m2 = store.put(
            content=b"unique content 2",
            artifact_type="touchstone",
            roles=["oracle_output"],
            run_id="run-001",
        )
        m3 = store.put(
            content=b"unique content 1",  # Duplicate content
            artifact_type="coupon_spec",
            roles=["geometry"],
            run_id="run-002",
        )

        for m in [m1, m2, m3]:
            registry.index_artifact(m)

        stats = registry.get_storage_stats()

        assert stats["total_artifacts"] == 3
        assert stats["unique_hashes"] == 2  # One duplicate
        assert stats["artifacts_by_type"]["coupon_spec"] == 2
        assert stats["artifacts_by_type"]["touchstone"] == 1
        assert stats["deduplication_ratio"] == 1.5  # 3/2

        registry.close()


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_indexing(self, tmp_path: Path) -> None:
        """Test that concurrent indexing works correctly."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        num_threads = 10
        errors: list[Exception] = []

        def index_artifact(i: int) -> None:
            try:
                m = store.put(
                    content=f"content-{i}".encode(),
                    artifact_type="other",
                    roles=["intermediate"],
                    run_id=f"run-{i}",
                )
                registry.index_artifact(m)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=index_artifact, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert registry.count_artifacts() == num_threads

        registry.close()


class TestArtifactsByHash:
    """Tests for finding artifacts by content hash."""

    def test_get_artifacts_by_hash(self, tmp_path: Path) -> None:
        """Test finding all artifacts with a given content hash."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        content = b"shared content"
        m1 = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        m2 = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-002",
        )

        registry.index_artifact(m1)
        registry.index_artifact(m2)

        artifacts = registry.get_artifacts_by_hash(m1.content_hash.digest)
        assert len(artifacts) == 2
        artifact_ids = {a.artifact_id for a in artifacts}
        assert m1.artifact_id in artifact_ids
        assert m2.artifact_id in artifact_ids

        registry.close()


class TestRecordDataclasses:
    """Tests for record dataclass conversions."""

    def test_artifact_record_from_row(self, tmp_path: Path) -> None:
        """Test ArtifactRecord.from_row conversion."""
        store = ArtifactStore(tmp_path / "store")
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        m = store.put(
            content=b"test",
            artifact_type="coupon_spec",
            roles=["geometry", "config"],
            run_id="run-001",
            tags={"env": "test"},
        )
        registry.index_artifact(m)

        record = registry.get_artifact(m.artifact_id)

        assert isinstance(record, ArtifactRecord)
        assert record.roles == ["geometry", "config"]
        assert record.tags == {"env": "test"}

        registry.close()

    def test_dataset_record_from_row(self, tmp_path: Path) -> None:
        """Test DatasetRecord.from_row conversion."""
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        registry.index_dataset(
            dataset_id="ds-001",
            version="v1.0",
            artifact_count=50,
            total_bytes=5000,
            created_utc="2026-01-20T09:00:00Z",
            description="Test",
        )

        record = registry.get_dataset("ds-001")

        assert isinstance(record, DatasetRecord)
        assert record.version == "v1.0"
        assert record.artifact_count == 50

        registry.close()

    def test_run_record_from_row(self, tmp_path: Path) -> None:
        """Test RunRecord.from_row conversion."""
        registry = ArtifactRegistry(tmp_path / "registry.db")
        registry.initialize()

        registry.index_run(
            run_id="run-001",
            started_utc="2026-01-20T09:00:00Z",
            status="completed",
            config={"param": "value"},
        )

        record = registry.get_run("run-001")

        assert isinstance(record, RunRecord)
        assert record.config == {"param": "value"}

        registry.close()
