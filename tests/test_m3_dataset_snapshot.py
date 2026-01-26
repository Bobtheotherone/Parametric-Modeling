"""Tests for the M3 DatasetSnapshot with Parquet index and manifest hashing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from formula_foundry.m3.artifact_store import (
    ArtifactManifest,
    ArtifactStore,
    ContentHash,
    Lineage,
    Provenance,
)
from formula_foundry.m3.dataset_snapshot import (
    DatasetMember,
    DatasetNotFoundError,
    DatasetProvenance,
    DatasetSnapshot,
    DatasetSnapshotError,
    DatasetSnapshotReader,
    DatasetSnapshotWriter,
    DatasetStatistics,
    IncrementalSnapshotBuilder,
    SplitDefinition,
    compute_manifest_hash,
)

# Check for PyArrow availability
try:
    import pyarrow

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


class TestDatasetMember:
    """Tests for DatasetMember dataclass."""

    def test_to_dict(self) -> None:
        member = DatasetMember(
            artifact_id="art-001",
            content_hash=ContentHash(algorithm="sha256", digest="a" * 64),
            artifact_type="touchstone",
            role="oracle_output",
            byte_size=1024,
            storage_path="objects/aa/aaa...",
        )
        result = member.to_dict()

        assert result["artifact_id"] == "art-001"
        assert result["content_hash"]["algorithm"] == "sha256"
        assert result["artifact_type"] == "touchstone"
        assert result["role"] == "oracle_output"
        assert result["byte_size"] == 1024

    def test_from_dict(self) -> None:
        data = {
            "artifact_id": "art-002",
            "content_hash": {"algorithm": "sha256", "digest": "b" * 64},
            "artifact_type": "coupon_spec",
            "role": "geometry",
            "byte_size": 512,
        }
        member = DatasetMember.from_dict(data)

        assert member.artifact_id == "art-002"
        assert member.content_hash.digest == "b" * 64
        assert member.artifact_type == "coupon_spec"

    def test_from_manifest(self) -> None:
        manifest = ArtifactManifest(
            artifact_id="art-003",
            artifact_type="touchstone",
            content_hash=ContentHash(algorithm="sha256", digest="c" * 64),
            byte_size=2048,
            created_utc="2026-01-20T00:00:00Z",
            provenance=Provenance(
                generator="test",
                generator_version="1.0",
                hostname="localhost",
            ),
            roles=["oracle_output"],
            lineage=Lineage(run_id="run-001"),
            storage_path="objects/cc/ccc...",
        )
        member = DatasetMember.from_manifest(manifest, role="oracle_output", features={"freq": 1e9})

        assert member.artifact_id == "art-003"
        assert member.content_hash.digest == "c" * 64
        assert member.role == "oracle_output"
        assert member.features == {"freq": 1e9}


class TestSplitDefinition:
    """Tests for SplitDefinition dataclass."""

    def test_to_dict(self) -> None:
        split = SplitDefinition(
            name="train",
            artifact_ids=["art-001", "art-002"],
            count=2,
            fraction=0.8,
        )
        result = split.to_dict()

        assert result["name"] == "train"
        assert len(result["artifact_ids"]) == 2
        assert result["fraction"] == 0.8

    def test_from_dict(self) -> None:
        data = {
            "name": "validation",
            "artifact_ids": ["art-003"],
            "count": 1,
            "fraction": 0.1,
        }
        split = SplitDefinition.from_dict(data)

        assert split.name == "validation"
        assert split.count == 1


class TestComputeManifestHash:
    """Tests for manifest hash computation."""

    def test_deterministic_hash(self) -> None:
        """Test that manifest hash is deterministic."""
        members = [
            DatasetMember(
                artifact_id="art-001",
                content_hash=ContentHash(algorithm="sha256", digest="a" * 64),
                artifact_type="touchstone",
                role="oracle_output",
            ),
            DatasetMember(
                artifact_id="art-002",
                content_hash=ContentHash(algorithm="sha256", digest="b" * 64),
                artifact_type="coupon_spec",
                role="geometry",
            ),
        ]

        hash1 = compute_manifest_hash(members)
        hash2 = compute_manifest_hash(members)

        assert hash1.digest == hash2.digest
        assert hash1.algorithm == "sha256"

    def test_order_independent(self) -> None:
        """Test that hash is independent of member order."""
        member_a = DatasetMember(
            artifact_id="art-001",
            content_hash=ContentHash(algorithm="sha256", digest="a" * 64),
            artifact_type="touchstone",
            role="oracle_output",
        )
        member_b = DatasetMember(
            artifact_id="art-002",
            content_hash=ContentHash(algorithm="sha256", digest="b" * 64),
            artifact_type="coupon_spec",
            role="geometry",
        )

        hash_ab = compute_manifest_hash([member_a, member_b])
        hash_ba = compute_manifest_hash([member_b, member_a])

        assert hash_ab.digest == hash_ba.digest

    def test_different_members_different_hash(self) -> None:
        """Test that different members produce different hashes."""
        members_a = [
            DatasetMember(
                artifact_id="art-001",
                content_hash=ContentHash(algorithm="sha256", digest="a" * 64),
                artifact_type="touchstone",
                role="oracle_output",
            ),
        ]
        members_b = [
            DatasetMember(
                artifact_id="art-002",
                content_hash=ContentHash(algorithm="sha256", digest="b" * 64),
                artifact_type="touchstone",
                role="oracle_output",
            ),
        ]

        hash_a = compute_manifest_hash(members_a)
        hash_b = compute_manifest_hash(members_b)

        assert hash_a.digest != hash_b.digest


class TestDatasetSnapshot:
    """Tests for DatasetSnapshot class."""

    def test_to_dict_schema_compliance(self) -> None:
        """Test that to_dict produces schema-compliant output."""
        snapshot = DatasetSnapshot(
            dataset_id="test_dataset",
            version="v1.0",
            created_utc="2026-01-20T00:00:00Z",
            members=[
                DatasetMember(
                    artifact_id="art-001",
                    content_hash=ContentHash(algorithm="sha256", digest="a" * 64),
                    artifact_type="touchstone",
                    role="oracle_output",
                    byte_size=1024,
                ),
            ],
            content_hash=ContentHash(algorithm="sha256", digest="d" * 64),
            provenance=DatasetProvenance(
                generator="test",
                generator_version="1.0",
            ),
        )
        result = snapshot.to_dict()

        # Required fields per dataset.v1 schema
        assert result["schema_version"] == 1
        assert result["dataset_id"] == "test_dataset"
        assert result["version"] == "v1.0"
        assert result["created_utc"] == "2026-01-20T00:00:00Z"
        assert "members" in result
        assert result["members"]["count"] == 1
        assert result["members"]["total_bytes"] == 1024
        assert "content_hash" in result
        assert "provenance" in result

    def test_from_dict(self) -> None:
        """Test round-trip serialization."""
        original = DatasetSnapshot(
            dataset_id="roundtrip_test",
            version="v2.0",
            created_utc="2026-01-20T12:00:00Z",
            members=[
                DatasetMember(
                    artifact_id="art-001",
                    content_hash=ContentHash(algorithm="sha256", digest="a" * 64),
                    artifact_type="touchstone",
                    role="oracle_output",
                    byte_size=2048,
                ),
                DatasetMember(
                    artifact_id="art-002",
                    content_hash=ContentHash(algorithm="sha256", digest="b" * 64),
                    artifact_type="coupon_spec",
                    role="geometry",
                    byte_size=512,
                ),
            ],
            content_hash=ContentHash(algorithm="sha256", digest="e" * 64),
            provenance=DatasetProvenance(
                generator="formula_foundry",
                generator_version="0.1.0",
                source_runs=["run-001", "run-002"],
            ),
            name="Test Dataset",
            description="A test dataset for round-trip",
            tags={"env": "test"},
        )

        data = original.to_dict()
        restored = DatasetSnapshot.from_dict(data)

        assert restored.dataset_id == original.dataset_id
        assert restored.version == original.version
        assert restored.member_count == 2
        assert restored.total_bytes == 2560
        assert restored.name == "Test Dataset"

    def test_member_count_property(self) -> None:
        """Test member_count property."""
        snapshot = DatasetSnapshot(
            dataset_id="count_test",
            version="v1",
            created_utc="2026-01-20T00:00:00Z",
            members=[
                DatasetMember(
                    artifact_id=f"art-{i}",
                    content_hash=ContentHash(algorithm="sha256", digest=f"{i:064d}"),
                    artifact_type="other",
                    role="intermediate",
                )
                for i in range(5)
            ],
            content_hash=ContentHash(algorithm="sha256", digest="f" * 64),
            provenance=DatasetProvenance(generator="test", generator_version="1.0"),
        )

        assert snapshot.member_count == 5

    def test_get_members_by_type(self) -> None:
        """Test filtering members by type."""
        snapshot = DatasetSnapshot(
            dataset_id="filter_test",
            version="v1",
            created_utc="2026-01-20T00:00:00Z",
            members=[
                DatasetMember(
                    artifact_id="art-1",
                    content_hash=ContentHash(algorithm="sha256", digest="1" * 64),
                    artifact_type="touchstone",
                    role="oracle_output",
                ),
                DatasetMember(
                    artifact_id="art-2",
                    content_hash=ContentHash(algorithm="sha256", digest="2" * 64),
                    artifact_type="coupon_spec",
                    role="geometry",
                ),
                DatasetMember(
                    artifact_id="art-3",
                    content_hash=ContentHash(algorithm="sha256", digest="3" * 64),
                    artifact_type="touchstone",
                    role="oracle_output",
                ),
            ],
            content_hash=ContentHash(algorithm="sha256", digest="g" * 64),
            provenance=DatasetProvenance(generator="test", generator_version="1.0"),
        )

        touchstones = snapshot.get_members_by_type("touchstone")
        assert len(touchstones) == 2

        coupon_specs = snapshot.get_members_by_type("coupon_spec")
        assert len(coupon_specs) == 1


class TestDatasetSnapshotWriter:
    """Tests for DatasetSnapshotWriter class."""

    def test_basic_workflow(self, tmp_path: Path) -> None:
        """Test basic writer workflow."""
        store = ArtifactStore(tmp_path / "data")

        # Create some artifacts
        manifest1 = store.put(
            content=b"content 1",
            artifact_type="touchstone",
            roles=["oracle_output"],
            run_id="run-001",
        )
        manifest2 = store.put(
            content=b"content 2",
            artifact_type="coupon_spec",
            roles=["geometry"],
            run_id="run-001",
        )

        # Create snapshot
        writer = DatasetSnapshotWriter(
            dataset_id="test_dataset",
            version="v1.0",
            store=store,
            name="Test Dataset",
        )
        writer.add_member(manifest1, role="oracle_output")
        writer.add_member(manifest2, role="geometry")

        output_dir = tmp_path / "datasets"
        snapshot = writer.finalize(output_dir=output_dir, write_parquet=False)

        assert snapshot.dataset_id == "test_dataset"
        assert snapshot.version == "v1.0"
        assert snapshot.member_count == 2
        assert snapshot.name == "Test Dataset"

        # Verify manifest file created
        manifest_file = output_dir / "test_dataset_v1.0.json"
        assert manifest_file.exists()

    def test_add_member_by_id(self, tmp_path: Path) -> None:
        """Test adding members by artifact ID."""
        store = ArtifactStore(tmp_path / "data")

        manifest = store.put(
            content=b"test content",
            artifact_type="touchstone",
            roles=["oracle_output"],
            run_id="run-001",
        )

        writer = DatasetSnapshotWriter(
            dataset_id="by_id_test",
            version="v1",
            store=store,
        )
        member = writer.add_member_by_id(
            manifest.artifact_id,
            role="oracle_output",
            features={"freq_min": 1e6, "freq_max": 10e9},
        )

        assert member.artifact_id == manifest.artifact_id
        assert member.features["freq_min"] == 1e6

    def test_add_member_by_id_no_store_raises(self) -> None:
        """Test that add_member_by_id raises without store."""
        writer = DatasetSnapshotWriter(
            dataset_id="no_store_test",
            version="v1",
        )

        with pytest.raises(DatasetSnapshotError, match="No ArtifactStore"):
            writer.add_member_by_id("nonexistent", role="test")

    def test_statistics_computation(self, tmp_path: Path) -> None:
        """Test that statistics are computed correctly."""
        store = ArtifactStore(tmp_path / "data")

        # Create artifacts of different types
        for i in range(3):
            store.put(
                content=f"touchstone {i}".encode(),
                artifact_type="touchstone",
                roles=["oracle_output"],
                run_id="run-001",
            )
        for i in range(2):
            store.put(
                content=f"coupon {i}".encode(),
                artifact_type="coupon_spec",
                roles=["geometry"],
                run_id="run-001",
            )

        writer = DatasetSnapshotWriter(
            dataset_id="stats_test",
            version="v1",
            store=store,
        )

        for manifest_id in store.list_manifests():
            manifest = store.get_manifest(manifest_id)
            role = "oracle_output" if manifest.artifact_type == "touchstone" else "geometry"
            writer.add_member(manifest, role=role)

        snapshot = writer.finalize(write_parquet=False)

        assert snapshot.statistics is not None
        assert snapshot.statistics.by_artifact_type["touchstone"]["count"] == 3
        assert snapshot.statistics.by_artifact_type["coupon_spec"]["count"] == 2

    def test_tags_and_annotations(self, tmp_path: Path) -> None:
        """Test setting tags and annotations."""
        store = ArtifactStore(tmp_path / "data")
        manifest = store.put(
            content=b"test",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        writer = DatasetSnapshotWriter(
            dataset_id="tags_test",
            version="v1",
            store=store,
        )
        writer.add_member(manifest, role="intermediate")
        writer.set_tags({"environment": "test", "version": "1.0"})
        writer.set_annotations({"split_seed": 42, "custom": {"nested": True}})

        snapshot = writer.finalize(write_parquet=False)

        assert snapshot.tags["environment"] == "test"
        assert snapshot.annotations["split_seed"] == 42

    @pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow not installed")
    def test_parquet_index_creation(self, tmp_path: Path) -> None:
        """Test Parquet index file creation."""
        store = ArtifactStore(tmp_path / "data")

        writer = DatasetSnapshotWriter(
            dataset_id="parquet_test",
            version="v1",
            store=store,
        )

        # Create artifacts with features
        for i in range(5):
            manifest = store.put(
                content=f"content {i}".encode(),
                artifact_type="touchstone",
                roles=["oracle_output"],
                run_id="run-001",
            )
            writer.add_member(
                manifest,
                role="oracle_output",
                features={"via_diameter": 0.3 + i / 10.0},
            )

        output_dir = tmp_path / "datasets"
        snapshot = writer.finalize(output_dir=output_dir, write_parquet=True)

        assert snapshot.index_path is not None
        parquet_file = output_dir / snapshot.index_path
        assert parquet_file.exists()


class TestDatasetSnapshotReader:
    """Tests for DatasetSnapshotReader class."""

    def test_load_from_file(self, tmp_path: Path) -> None:
        """Test loading a snapshot from a JSON file."""
        store = ArtifactStore(tmp_path / "data")
        manifest = store.put(
            content=b"test content",
            artifact_type="touchstone",
            roles=["oracle_output"],
            run_id="run-001",
        )

        # Create and write snapshot
        writer = DatasetSnapshotWriter(
            dataset_id="load_test",
            version="v1",
            store=store,
        )
        writer.add_member(manifest, role="oracle_output")
        output_dir = tmp_path / "datasets"
        writer.finalize(output_dir=output_dir, write_parquet=False)

        # Load with reader
        snapshot_path = output_dir / "load_test_v1.json"
        reader = DatasetSnapshotReader(snapshot_path=snapshot_path)
        snapshot = reader.load()

        assert snapshot.dataset_id == "load_test"
        assert snapshot.member_count == 1

    def test_load_not_found_raises(self, tmp_path: Path) -> None:
        """Test that loading nonexistent file raises error."""
        reader = DatasetSnapshotReader(snapshot_path=tmp_path / "nonexistent.json")

        with pytest.raises(DatasetNotFoundError):
            reader.load()

    def test_iter_members(self, tmp_path: Path) -> None:
        """Test iterating over members with filtering."""
        store = ArtifactStore(tmp_path / "data")

        # Create mixed artifacts
        m1 = store.put(
            content=b"ts1",
            artifact_type="touchstone",
            roles=["oracle_output"],
            run_id="run-001",
        )
        m2 = store.put(
            content=b"spec1",
            artifact_type="coupon_spec",
            roles=["geometry"],
            run_id="run-001",
        )
        m3 = store.put(
            content=b"ts2",
            artifact_type="touchstone",
            roles=["oracle_output"],
            run_id="run-001",
        )

        writer = DatasetSnapshotWriter(
            dataset_id="iter_test",
            version="v1",
            store=store,
        )
        writer.add_member(m1, role="oracle_output")
        writer.add_member(m2, role="geometry")
        writer.add_member(m3, role="oracle_output")

        output_dir = tmp_path / "datasets"
        writer.finalize(output_dir=output_dir, write_parquet=False)

        reader = DatasetSnapshotReader(snapshot_path=output_dir / "iter_test_v1.json")

        # All members
        all_members = list(reader.iter_members())
        assert len(all_members) == 3

        # Filter by type
        touchstones = list(reader.iter_members(artifact_type="touchstone"))
        assert len(touchstones) == 2

        # Filter by role
        geometry = list(reader.iter_members(role="geometry"))
        assert len(geometry) == 1

    def test_verify_integrity(self, tmp_path: Path) -> None:
        """Test integrity verification."""
        store = ArtifactStore(tmp_path / "data")
        manifest = store.put(
            content=b"integrity test",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        writer = DatasetSnapshotWriter(
            dataset_id="integrity_test",
            version="v1",
            store=store,
        )
        writer.add_member(manifest, role="intermediate")

        output_dir = tmp_path / "datasets"
        writer.finalize(output_dir=output_dir, write_parquet=False)

        reader = DatasetSnapshotReader(snapshot_path=output_dir / "integrity_test_v1.json")
        is_valid, errors = reader.verify_integrity()

        assert is_valid
        assert len(errors) == 0

    def test_verify_integrity_detects_corruption(self, tmp_path: Path) -> None:
        """Test that integrity verification detects corruption."""
        store = ArtifactStore(tmp_path / "data")
        manifest = store.put(
            content=b"will be corrupted",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        writer = DatasetSnapshotWriter(
            dataset_id="corrupt_test",
            version="v1",
            store=store,
        )
        writer.add_member(manifest, role="intermediate")

        output_dir = tmp_path / "datasets"
        writer.finalize(output_dir=output_dir, write_parquet=False)

        # Corrupt the manifest file
        manifest_path = output_dir / "corrupt_test_v1.json"
        data = json.loads(manifest_path.read_text())
        data["content_hash"]["digest"] = "0" * 64  # Wrong hash
        manifest_path.write_text(json.dumps(data))

        reader = DatasetSnapshotReader(snapshot_path=manifest_path)
        is_valid, errors = reader.verify_integrity()

        assert not is_valid
        assert len(errors) > 0
        assert "hash mismatch" in errors[0].lower()

    def test_get_artifact_content(self, tmp_path: Path) -> None:
        """Test retrieving artifact content through reader."""
        store = ArtifactStore(tmp_path / "data")
        content = b"retrievable content"
        manifest = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        writer = DatasetSnapshotWriter(
            dataset_id="content_test",
            version="v1",
            store=store,
        )
        writer.add_member(manifest, role="intermediate")

        output_dir = tmp_path / "datasets"
        writer.finalize(output_dir=output_dir, write_parquet=False)

        reader = DatasetSnapshotReader(
            snapshot_path=output_dir / "content_test_v1.json",
            store=store,
        )
        retrieved = reader.get_artifact_content(manifest.artifact_id)

        assert retrieved == content

    def test_get_artifact_content_no_store_raises(self, tmp_path: Path) -> None:
        """Test that getting content without store raises."""
        store = ArtifactStore(tmp_path / "data")
        manifest = store.put(
            content=b"test",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        writer = DatasetSnapshotWriter(
            dataset_id="no_store_test",
            version="v1",
            store=store,
        )
        writer.add_member(manifest, role="intermediate")

        output_dir = tmp_path / "datasets"
        writer.finalize(output_dir=output_dir, write_parquet=False)

        reader = DatasetSnapshotReader(
            snapshot_path=output_dir / "no_store_test_v1.json",
            # No store provided
        )

        with pytest.raises(DatasetSnapshotError, match="No ArtifactStore"):
            reader.get_artifact_content(manifest.artifact_id)

    @pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow not installed")
    def test_query_parquet_features(self, tmp_path: Path) -> None:
        """Test querying features from Parquet index."""
        store = ArtifactStore(tmp_path / "data")

        writer = DatasetSnapshotWriter(
            dataset_id="query_test",
            version="v1",
            store=store,
        )

        # Add artifacts with features
        for i in range(10):
            atype = "touchstone" if i % 2 == 0 else "coupon_spec"
            role = "oracle_output" if i % 2 == 0 else "geometry"
            manifest = store.put(
                content=f"content {i}".encode(),
                artifact_type=atype,
                roles=[role],
                run_id="run-001",
            )
            writer.add_member(
                manifest,
                role=role,
                features={"via_diameter": 0.3 + i * 0.01, "index": i},
            )

        output_dir = tmp_path / "datasets"
        writer.finalize(output_dir=output_dir, write_parquet=True)

        reader = DatasetSnapshotReader(snapshot_path=output_dir / "query_test_v1.json")

        # Query all features
        table = reader.query_features()
        assert len(table) == 10

        # Query by type
        touchstones = reader.query_features(artifact_type="touchstone")
        assert len(touchstones) == 5

        # Query by role
        geometry = reader.query_features(role="geometry")
        assert len(geometry) == 5

    @pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow not installed")
    def test_to_pandas(self, tmp_path: Path) -> None:
        """Test converting to pandas DataFrame."""
        store = ArtifactStore(tmp_path / "data")

        writer = DatasetSnapshotWriter(
            dataset_id="pandas_test",
            version="v1",
            store=store,
        )

        for i in range(5):
            manifest = store.put(
                content=f"content {i}".encode(),
                artifact_type="touchstone",
                roles=["oracle_output"],
                run_id="run-001",
            )
            writer.add_member(manifest, role="oracle_output", features={"value": i})

        output_dir = tmp_path / "datasets"
        writer.finalize(output_dir=output_dir, write_parquet=True)

        reader = DatasetSnapshotReader(snapshot_path=output_dir / "pandas_test_v1.json")
        df = reader.to_pandas()

        assert len(df) == 5
        assert "artifact_id" in df.columns
        assert "feature_value" in df.columns


class TestParquetNotAvailable:
    """Tests for graceful degradation without PyArrow."""

    def test_writer_without_parquet(self, tmp_path: Path) -> None:
        """Test that writer works without PyArrow."""
        store = ArtifactStore(tmp_path / "data")
        manifest = store.put(
            content=b"test",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        writer = DatasetSnapshotWriter(
            dataset_id="no_parquet_test",
            version="v1",
            store=store,
        )
        writer.add_member(manifest, role="intermediate")

        output_dir = tmp_path / "datasets"
        # Should work even without PyArrow
        snapshot = writer.finalize(output_dir=output_dir, write_parquet=False)

        assert snapshot is not None
        assert snapshot.index_path is None  # No Parquet index


class TestDatasetProvenanceAndStatistics:
    """Tests for DatasetProvenance and DatasetStatistics."""

    def test_provenance_serialization(self) -> None:
        """Test DatasetProvenance serialization."""
        prov = DatasetProvenance(
            generator="formula_foundry",
            generator_version="0.1.0",
            source_runs=["run-001", "run-002"],
            pipeline_stage="em_simulation",
            git_commit="a" * 40,
        )
        data = prov.to_dict()

        assert data["generator"] == "formula_foundry"
        assert len(data["source_runs"]) == 2
        assert data["git_commit"] == "a" * 40

        restored = DatasetProvenance.from_dict(data)
        assert restored.generator == prov.generator

    def test_statistics_serialization(self) -> None:
        """Test DatasetStatistics serialization."""
        stats = DatasetStatistics(
            by_artifact_type={
                "touchstone": {"count": 10, "total_bytes": 10240},
                "coupon_spec": {"count": 5, "total_bytes": 2560},
            },
            by_role={
                "oracle_output": {"count": 10, "total_bytes": 10240},
            },
            unique_coupons=5,
            frequency_range_hz={"min": 1e6, "max": 10e9},
            parameter_ranges={
                "via_diameter": {"min": 0.2, "max": 0.5},
            },
        )
        data = stats.to_dict()

        assert data["by_artifact_type"]["touchstone"]["count"] == 10
        assert data["frequency_range_hz"]["max"] == 10e9

        restored = DatasetStatistics.from_dict(data)
        assert restored.unique_coupons == 5


class TestIncrementalSnapshotBuilder:
    """Tests for IncrementalSnapshotBuilder class."""

    def test_basic_incremental_workflow(self, tmp_path: Path) -> None:
        """Test creating an incremental snapshot from a parent."""
        store = ArtifactStore(tmp_path / "data")

        # Create parent snapshot
        parent_writer = DatasetSnapshotWriter(
            dataset_id="incremental_test",
            version="v1.0",
            store=store,
            name="Parent Dataset",
        )

        manifests = []
        for i in range(3):
            m = store.put(
                content=f"content {i}".encode(),
                artifact_type="touchstone",
                roles=["oracle_output"],
                run_id="run-001",
            )
            manifests.append(m)
            parent_writer.add_member(m, role="oracle_output")

        output_dir = tmp_path / "datasets"
        parent = parent_writer.finalize(output_dir=output_dir, write_parquet=False)

        # Create incremental snapshot
        builder = IncrementalSnapshotBuilder(
            parent=parent,
            new_version="v1.1",
            store=store,
        )

        # Add a new member
        new_manifest = store.put(
            content=b"new content",
            artifact_type="touchstone",
            roles=["oracle_output"],
            run_id="run-002",
        )
        builder.add_member(new_manifest, role="oracle_output")

        # Finalize
        child = builder.finalize(output_dir=output_dir, write_parquet=False)

        assert child.dataset_id == "incremental_test"
        assert child.version == "v1.1"
        assert child.parent_version == "v1.0"
        assert child.member_count == 4  # 3 from parent + 1 new

    def test_incremental_remove_member(self, tmp_path: Path) -> None:
        """Test removing members in incremental snapshot."""
        store = ArtifactStore(tmp_path / "data")

        # Create parent with 3 members
        parent_writer = DatasetSnapshotWriter(
            dataset_id="remove_test",
            version="v1",
            store=store,
        )

        manifests = []
        for i in range(3):
            m = store.put(
                content=f"content {i}".encode(),
                artifact_type="touchstone",
                roles=["oracle_output"],
                run_id="run-001",
            )
            manifests.append(m)
            parent_writer.add_member(m, role="oracle_output")

        parent = parent_writer.finalize(write_parquet=False)

        # Create incremental and remove one
        builder = IncrementalSnapshotBuilder(
            parent=parent,
            new_version="v2",
            store=store,
        )
        removed = builder.remove_member(manifests[0].artifact_id)

        assert removed is True
        child = builder.finalize(write_parquet=False)

        assert child.member_count == 2
        assert child.get_member(manifests[0].artifact_id) is None
        assert child.get_member(manifests[1].artifact_id) is not None

    def test_incremental_diff_summary(self, tmp_path: Path) -> None:
        """Test getting diff summary from incremental builder."""
        store = ArtifactStore(tmp_path / "data")

        # Create parent
        parent_writer = DatasetSnapshotWriter(
            dataset_id="diff_test",
            version="v1",
            store=store,
        )
        m1 = store.put(content=b"1", artifact_type="other", roles=["intermediate"], run_id="run-001")
        m2 = store.put(content=b"2", artifact_type="other", roles=["intermediate"], run_id="run-001")
        parent_writer.add_member(m1, role="intermediate")
        parent_writer.add_member(m2, role="intermediate")
        parent = parent_writer.finalize(write_parquet=False)

        # Create incremental
        builder = IncrementalSnapshotBuilder(parent=parent, new_version="v2", store=store)

        # Add one, remove one
        m3 = store.put(content=b"3", artifact_type="other", roles=["intermediate"], run_id="run-002")
        builder.add_member(m3, role="intermediate")
        builder.remove_member(m1.artifact_id)

        diff = builder.get_diff_summary()

        assert diff["parent_version"] == "v1"
        assert diff["new_version"] == "v2"
        assert diff["parent_count"] == 2
        assert diff["new_count"] == 2
        assert diff["added"] == 1
        assert diff["removed"] == 1

    def test_incremental_update_features(self, tmp_path: Path) -> None:
        """Test updating member features in incremental snapshot."""
        store = ArtifactStore(tmp_path / "data")

        # Create parent
        parent_writer = DatasetSnapshotWriter(
            dataset_id="feature_test",
            version="v1",
            store=store,
        )
        m = store.put(content=b"test", artifact_type="other", roles=["intermediate"], run_id="run-001")
        parent_writer.add_member(m, role="intermediate", features={"value": 1.0})
        parent = parent_writer.finalize(write_parquet=False)

        # Create incremental and update features
        builder = IncrementalSnapshotBuilder(parent=parent, new_version="v2", store=store)
        updated = builder.update_member_features(
            m.artifact_id,
            {"value": 2.0, "extra": "new"},
            merge=True,
        )

        assert updated is True
        child = builder.finalize(write_parquet=False)
        member = child.get_member(m.artifact_id)

        assert member is not None
        assert member.features["value"] == 2.0
        assert member.features["extra"] == "new"

    def test_incremental_annotations_preserved(self, tmp_path: Path) -> None:
        """Test that incremental diff info is recorded in annotations."""
        store = ArtifactStore(tmp_path / "data")

        parent_writer = DatasetSnapshotWriter(
            dataset_id="annotation_test",
            version="v1",
            store=store,
        )
        m = store.put(content=b"test", artifact_type="other", roles=["intermediate"], run_id="run-001")
        parent_writer.add_member(m, role="intermediate")
        parent = parent_writer.finalize(write_parquet=False)

        builder = IncrementalSnapshotBuilder(parent=parent, new_version="v2", store=store)
        child = builder.finalize(write_parquet=False)

        assert "incremental_from" in child.annotations
        assert child.annotations["incremental_from"] == "v1"
        assert "incremental_diff" in child.annotations


class TestDatasetSnapshotSlicing:
    """Tests for efficient slicing and filtering."""

    def test_slice_basic(self, tmp_path: Path) -> None:
        """Test basic slicing functionality."""
        store = ArtifactStore(tmp_path / "data")

        writer = DatasetSnapshotWriter(
            dataset_id="slice_test",
            version="v1",
            store=store,
        )

        # Create 10 members with features
        for i in range(10):
            m = store.put(
                content=f"content {i}".encode(),
                artifact_type="touchstone" if i % 2 == 0 else "coupon_spec",
                roles=["oracle_output" if i % 2 == 0 else "geometry"],
                run_id="run-001",
            )
            writer.add_member(
                m,
                role="oracle_output" if i % 2 == 0 else "geometry",
                features={"index": i, "value": i * 0.1},
            )

        output_dir = tmp_path / "datasets"
        writer.finalize(output_dir=output_dir, write_parquet=False)

        reader = DatasetSnapshotReader(snapshot_path=output_dir / "slice_test_v1.json")

        # Test range slicing
        sliced = reader.slice(start=2, end=5)
        assert len(sliced) == 3

        # Test type filtering
        touchstones = reader.slice(artifact_type="touchstone")
        assert len(touchstones) == 5

        # Test role filtering
        geometry = reader.slice(role="geometry")
        assert len(geometry) == 5

    def test_slice_with_feature_filter(self, tmp_path: Path) -> None:
        """Test slicing with feature range filters.

        Note: Feature filtering on in-memory members works when the snapshot
        is created and members have features populated. When loaded from JSON,
        features are only available via the Parquet index (if written).
        This test uses the snapshot directly from the writer.
        """
        store = ArtifactStore(tmp_path / "data")

        writer = DatasetSnapshotWriter(
            dataset_id="feature_filter_test",
            version="v1",
            store=store,
        )

        for i in range(20):
            m = store.put(
                content=f"content {i}".encode(),
                artifact_type="touchstone",
                roles=["oracle_output"],
                run_id="run-001",
            )
            writer.add_member(
                m,
                role="oracle_output",
                features={"value": i * 0.1},  # 0.0 to 1.9
            )

        # Finalize returns snapshot with features in memory
        snapshot = writer.finalize(write_parquet=False)

        # Create reader from the in-memory snapshot (not from file)
        reader = DatasetSnapshotReader(snapshot=snapshot)

        # Filter by feature range
        filtered = reader.slice(
            feature_filter={"value": (0.5, 1.0)}  # Values 0.5 to 1.0
        )

        # Should get indices 5-10 (values 0.5-1.0)
        assert len(filtered) == 6
        for m in filtered:
            assert 0.5 <= m.features["value"] <= 1.0

    def test_slice_combined_filters(self, tmp_path: Path) -> None:
        """Test slicing with multiple filters combined.

        Uses in-memory snapshot since features are not persisted in JSON.
        """
        store = ArtifactStore(tmp_path / "data")

        writer = DatasetSnapshotWriter(
            dataset_id="combined_filter_test",
            version="v1",
            store=store,
        )

        for i in range(20):
            m = store.put(
                content=f"content {i}".encode(),
                artifact_type="touchstone" if i % 2 == 0 else "coupon_spec",
                roles=["oracle_output" if i % 2 == 0 else "geometry"],
                run_id="run-001",
            )
            writer.add_member(
                m,
                role="oracle_output" if i % 2 == 0 else "geometry",
                features={"value": i * 0.1},
            )

        # Use in-memory snapshot for feature access
        snapshot = writer.finalize(write_parquet=False)
        reader = DatasetSnapshotReader(snapshot=snapshot)

        # Combine type filter + feature range + limit
        filtered = reader.slice(
            artifact_type="touchstone",
            feature_filter={"value": (0.0, 1.0)},
            start=0,
            end=3,
        )

        # touchstones are at indices 0, 2, 4, 6, 8... with values 0.0, 0.2, 0.4, 0.6, 0.8...
        # Filtered by value <= 1.0: 0, 2, 4, 6, 8, 10 (6 items)
        # Take first 3
        assert len(filtered) == 3
        for m in filtered:
            assert m.artifact_type == "touchstone"

    @pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow not installed")
    def test_slice_parquet_basic(self, tmp_path: Path) -> None:
        """Test Parquet-based slicing."""
        store = ArtifactStore(tmp_path / "data")

        writer = DatasetSnapshotWriter(
            dataset_id="parquet_slice_test",
            version="v1",
            store=store,
        )

        for i in range(10):
            m = store.put(
                content=f"content {i}".encode(),
                artifact_type="touchstone" if i % 2 == 0 else "coupon_spec",
                roles=["oracle_output"],
                run_id="run-001",
            )
            writer.add_member(m, role="oracle_output", features={"index": i})

        output_dir = tmp_path / "datasets"
        writer.finalize(output_dir=output_dir, write_parquet=True)

        reader = DatasetSnapshotReader(snapshot_path=output_dir / "parquet_slice_test_v1.json")

        # Basic slice
        table = reader.slice_parquet(artifact_type="touchstone")
        assert len(table) == 5

        # Column projection
        table = reader.slice_parquet(columns=["artifact_id", "artifact_type"])
        assert len(table.column_names) == 2

    @pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow not installed")
    def test_slice_parquet_with_filters(self, tmp_path: Path) -> None:
        """Test Parquet slicing with predicate pushdown filters."""
        store = ArtifactStore(tmp_path / "data")

        writer = DatasetSnapshotWriter(
            dataset_id="predicate_test",
            version="v1",
            store=store,
        )

        for i in range(20):
            m = store.put(
                content=f"content {i}".encode(),
                artifact_type="touchstone",
                roles=["oracle_output"],
                run_id="run-001",
            )
            writer.add_member(m, role="oracle_output", features={"value": float(i)})

        output_dir = tmp_path / "datasets"
        writer.finalize(output_dir=output_dir, write_parquet=True)

        reader = DatasetSnapshotReader(snapshot_path=output_dir / "predicate_test_v1.json")

        # Filter with predicates
        table = reader.slice_parquet(filters=[("feature_value", ">=", 5.0), ("feature_value", "<=", 10.0)])

        # Should get values 5-10 (6 rows)
        assert len(table) == 6

    @pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow not installed")
    def test_row_group_metadata(self, tmp_path: Path) -> None:
        """Test getting row group metadata."""
        store = ArtifactStore(tmp_path / "data")

        writer = DatasetSnapshotWriter(
            dataset_id="rowgroup_test",
            version="v1",
            store=store,
        )

        for i in range(10):
            m = store.put(
                content=f"content {i}".encode(),
                artifact_type="touchstone",
                roles=["oracle_output"],
                run_id="run-001",
            )
            writer.add_member(m, role="oracle_output")

        output_dir = tmp_path / "datasets"
        writer.finalize(output_dir=output_dir, write_parquet=True)

        reader = DatasetSnapshotReader(snapshot_path=output_dir / "rowgroup_test_v1.json")

        metadata = reader.get_row_group_metadata()

        assert len(metadata) >= 1
        assert "num_rows" in metadata[0]
        assert "total_byte_size" in metadata[0]
