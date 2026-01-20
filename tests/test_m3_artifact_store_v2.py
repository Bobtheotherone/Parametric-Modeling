"""Tests for the M3 ArtifactStore with atomic writes and content-addressed storage."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from formula_foundry.m3.artifact_store import (
    ArtifactExistsError,
    ArtifactManifest,
    ArtifactNotFoundError,
    ArtifactStore,
    ContentHash,
    Lineage,
    LineageReference,
    Provenance,
)

if TYPE_CHECKING:
    pass


class TestContentHash:
    """Tests for ContentHash dataclass."""

    def test_to_dict(self) -> None:
        ch = ContentHash(algorithm="sha256", digest="a" * 64)
        result = ch.to_dict()
        assert result == {"algorithm": "sha256", "digest": "a" * 64}

    def test_from_dict(self) -> None:
        data = {"algorithm": "sha256", "digest": "b" * 64}
        ch = ContentHash.from_dict(data)
        assert ch.algorithm == "sha256"
        assert ch.digest == "b" * 64


class TestArtifactStore:
    """Tests for ArtifactStore class."""

    def test_init_creates_directories(self, tmp_path: Path) -> None:
        """Test that initializing the store creates necessary directories."""
        store = ArtifactStore(tmp_path / "data")
        store._ensure_dirs()

        assert store.objects_dir.exists()
        assert store.manifests_dir.exists()

    def test_put_content_addressed(self, tmp_path: Path) -> None:
        """Test that put stores content by its SHA256 hash."""
        store = ArtifactStore(tmp_path / "data")
        content = b"hello world"
        expected_hash = hashlib.sha256(content).hexdigest()

        manifest = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="test-run-001",
        )

        assert manifest.content_hash.algorithm == "sha256"
        assert manifest.content_hash.digest == expected_hash
        assert manifest.byte_size == len(content)

        # Verify the object exists at the expected path
        object_path = store._object_path(expected_hash)
        assert object_path.exists()
        assert object_path.read_bytes() == content

    def test_put_creates_prefix_directories(self, tmp_path: Path) -> None:
        """Test that put creates prefix directories for sharding."""
        store = ArtifactStore(tmp_path / "data")
        content = b"test content"
        expected_hash = hashlib.sha256(content).hexdigest()
        prefix = expected_hash[:2]

        manifest = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        prefix_dir = store.objects_dir / prefix
        assert prefix_dir.exists()
        assert prefix_dir.is_dir()

    def test_put_deduplicates_content(self, tmp_path: Path) -> None:
        """Test that identical content is not stored twice."""
        store = ArtifactStore(tmp_path / "data")
        content = b"duplicate content"

        manifest1 = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        manifest2 = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-002",
        )

        # Both manifests point to the same content hash
        assert manifest1.content_hash.digest == manifest2.content_hash.digest

        # But have different artifact IDs
        assert manifest1.artifact_id != manifest2.artifact_id

    def test_put_writes_manifest(self, tmp_path: Path) -> None:
        """Test that put writes a valid manifest file."""
        store = ArtifactStore(tmp_path / "data")
        content = b"manifest test"

        manifest = store.put(
            content=content,
            artifact_type="touchstone",
            roles=["oracle_output", "final_output"],
            run_id="run-manifest-001",
            media_type="application/x-touchstone",
            tags={"project": "test"},
        )

        manifest_path = store._manifest_path(manifest.artifact_id)
        assert manifest_path.exists()

        # Verify manifest contents
        loaded_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert loaded_data["schema_version"] == 1
        assert loaded_data["artifact_type"] == "touchstone"
        assert "oracle_output" in loaded_data["roles"]
        assert loaded_data["media_type"] == "application/x-touchstone"

    def test_get_retrieves_content(self, tmp_path: Path) -> None:
        """Test that get retrieves the correct content."""
        store = ArtifactStore(tmp_path / "data")
        content = b"retrievable content"

        manifest = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        retrieved = store.get(manifest.content_hash.digest)
        assert retrieved == content

    def test_get_not_found_raises(self, tmp_path: Path) -> None:
        """Test that get raises ArtifactNotFoundError for missing content."""
        store = ArtifactStore(tmp_path / "data")
        store._ensure_dirs()

        with pytest.raises(ArtifactNotFoundError):
            store.get("a" * 64)

    def test_get_by_id(self, tmp_path: Path) -> None:
        """Test that get_by_id retrieves content by artifact ID."""
        store = ArtifactStore(tmp_path / "data")
        content = b"get by id test"

        manifest = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        retrieved = store.get_by_id(manifest.artifact_id)
        assert retrieved == content

    def test_get_manifest(self, tmp_path: Path) -> None:
        """Test that get_manifest retrieves and parses a manifest."""
        store = ArtifactStore(tmp_path / "data")
        content = b"manifest retrieval test"

        original = store.put(
            content=content,
            artifact_type="coupon_spec",
            roles=["geometry", "config"],
            run_id="run-001",
            stage_name="generation",
        )

        retrieved = store.get_manifest(original.artifact_id)

        assert retrieved.artifact_id == original.artifact_id
        assert retrieved.artifact_type == "coupon_spec"
        assert retrieved.content_hash.digest == original.content_hash.digest
        assert "geometry" in retrieved.roles
        assert retrieved.lineage.stage_name == "generation"

    def test_exists(self, tmp_path: Path) -> None:
        """Test exists method."""
        store = ArtifactStore(tmp_path / "data")
        content = b"existence test"

        manifest = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        assert store.exists(manifest.content_hash.digest)
        assert not store.exists("f" * 64)

    def test_exists_by_id(self, tmp_path: Path) -> None:
        """Test exists_by_id method."""
        store = ArtifactStore(tmp_path / "data")
        content = b"id existence test"

        manifest = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        assert store.exists_by_id(manifest.artifact_id)
        assert not store.exists_by_id("nonexistent-id")

    def test_verify_valid_artifact(self, tmp_path: Path) -> None:
        """Test verify returns True for valid artifact."""
        store = ArtifactStore(tmp_path / "data")
        content = b"verify test"

        manifest = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        assert store.verify(manifest.artifact_id)

    def test_verify_corrupted_artifact(self, tmp_path: Path) -> None:
        """Test verify returns False for corrupted artifact."""
        store = ArtifactStore(tmp_path / "data")
        content = b"will be corrupted"

        manifest = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        # Corrupt the object
        object_path = store._object_path(manifest.content_hash.digest)
        object_path.write_bytes(b"corrupted content")

        assert not store.verify(manifest.artifact_id)

    def test_list_manifests(self, tmp_path: Path) -> None:
        """Test list_manifests returns all artifact IDs."""
        store = ArtifactStore(tmp_path / "data")

        ids = []
        for i in range(3):
            manifest = store.put(
                content=f"content {i}".encode(),
                artifact_type="other",
                roles=["intermediate"],
                run_id="run-001",
            )
            ids.append(manifest.artifact_id)

        listed = store.list_manifests()
        assert set(listed) == set(ids)

    def test_delete_manifest(self, tmp_path: Path) -> None:
        """Test delete removes manifest but keeps content."""
        store = ArtifactStore(tmp_path / "data")
        content = b"delete test"

        manifest = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        digest = manifest.content_hash.digest
        store.delete(manifest.artifact_id, delete_content=False)

        assert not store.exists_by_id(manifest.artifact_id)
        assert store.exists(digest)  # Content still exists

    def test_delete_with_content(self, tmp_path: Path) -> None:
        """Test delete with delete_content=True removes content too."""
        store = ArtifactStore(tmp_path / "data")
        content = b"delete with content test"

        manifest = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        digest = manifest.content_hash.digest
        store.delete(manifest.artifact_id, delete_content=True)

        assert not store.exists_by_id(manifest.artifact_id)
        assert not store.exists(digest)

    def test_delete_keeps_shared_content(self, tmp_path: Path) -> None:
        """Test delete_content doesn't remove content referenced by other manifests."""
        store = ArtifactStore(tmp_path / "data")
        content = b"shared content"

        manifest1 = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )
        manifest2 = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-002",
        )

        digest = manifest1.content_hash.digest
        store.delete(manifest1.artifact_id, delete_content=True)

        assert not store.exists_by_id(manifest1.artifact_id)
        assert store.exists_by_id(manifest2.artifact_id)
        assert store.exists(digest)  # Content still exists (referenced by manifest2)

    def test_put_file(self, tmp_path: Path) -> None:
        """Test put_file stores a file."""
        store = ArtifactStore(tmp_path / "data")
        file_path = tmp_path / "test_file.txt"
        content = b"file content"
        file_path.write_bytes(content)

        manifest = store.put_file(
            file_path=file_path,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        assert manifest.byte_size == len(content)
        assert store.get(manifest.content_hash.digest) == content

    def test_explicit_artifact_id(self, tmp_path: Path) -> None:
        """Test put with explicit artifact_id."""
        store = ArtifactStore(tmp_path / "data")
        content = b"explicit id test"

        manifest = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
            artifact_id="my-custom-artifact-id",
        )

        assert manifest.artifact_id == "my-custom-artifact-id"

    def test_duplicate_artifact_id_raises(self, tmp_path: Path) -> None:
        """Test put with duplicate artifact_id raises error."""
        store = ArtifactStore(tmp_path / "data")

        store.put(
            content=b"first",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
            artifact_id="duplicate-id",
        )

        with pytest.raises(ArtifactExistsError):
            store.put(
                content=b"second",
                artifact_type="other",
                roles=["intermediate"],
                run_id="run-002",
                artifact_id="duplicate-id",
            )

    def test_allow_overwrite(self, tmp_path: Path) -> None:
        """Test put with allow_overwrite=True allows duplicate artifact_id."""
        store = ArtifactStore(tmp_path / "data")

        store.put(
            content=b"first",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
            artifact_id="overwrite-id",
        )

        manifest = store.put(
            content=b"second",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-002",
            artifact_id="overwrite-id",
            allow_overwrite=True,
        )

        assert store.get_by_id("overwrite-id") == b"second"


class TestAtomicWrites:
    """Tests for atomic write behavior."""

    def test_atomic_write_creates_temp_file(self, tmp_path: Path) -> None:
        """Test that atomic write uses temp file pattern."""
        store = ArtifactStore(tmp_path / "data")
        store._ensure_dirs()

        # Verify no .tmp files exist after successful write
        content = b"atomic test"
        manifest = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        # Check no temp files remain
        tmp_files = list(store.objects_dir.rglob("*.tmp"))
        assert len(tmp_files) == 0

        tmp_files = list(store.manifests_dir.rglob("*.tmp"))
        assert len(tmp_files) == 0

    def test_concurrent_writes(self, tmp_path: Path) -> None:
        """Test that concurrent writes don't corrupt data."""
        store = ArtifactStore(tmp_path / "data")
        num_threads = 10
        results: list[ArtifactManifest] = []
        errors: list[Exception] = []

        def write_artifact(i: int) -> None:
            try:
                manifest = store.put(
                    content=f"content-{i}".encode(),
                    artifact_type="other",
                    roles=["intermediate"],
                    run_id=f"run-{i}",
                )
                results.append(manifest)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_artifact, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == num_threads

        # Verify all artifacts are retrievable and valid
        for manifest in results:
            assert store.verify(manifest.artifact_id)


class TestLineageAndProvenance:
    """Tests for lineage and provenance tracking."""

    def test_provenance_captured(self, tmp_path: Path) -> None:
        """Test that provenance information is captured."""
        store = ArtifactStore(
            tmp_path / "data",
            generator="test_generator",
            generator_version="1.2.3",
        )

        manifest = store.put(
            content=b"provenance test",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        assert manifest.provenance.generator == "test_generator"
        assert manifest.provenance.generator_version == "1.2.3"
        assert manifest.provenance.hostname != ""

    def test_lineage_inputs(self, tmp_path: Path) -> None:
        """Test that lineage inputs are recorded."""
        store = ArtifactStore(tmp_path / "data")

        # Create a source artifact
        source_manifest = store.put(
            content=b"source content",
            artifact_type="coupon_spec",
            roles=["geometry"],
            run_id="run-001",
        )

        # Create a derived artifact with lineage reference
        input_ref = LineageReference(
            artifact_id=source_manifest.artifact_id,
            relation="derived_from",
            content_hash=source_manifest.content_hash,
        )

        derived_manifest = store.put(
            content=b"derived content",
            artifact_type="resolved_design",
            roles=["intermediate"],
            run_id="run-001",
            inputs=[input_ref],
        )

        assert len(derived_manifest.lineage.inputs) == 1
        assert derived_manifest.lineage.inputs[0].artifact_id == source_manifest.artifact_id
        assert derived_manifest.lineage.inputs[0].relation == "derived_from"


class TestManifestSchema:
    """Tests for manifest schema compliance."""

    def test_manifest_has_required_fields(self, tmp_path: Path) -> None:
        """Test that manifest contains all required fields."""
        store = ArtifactStore(tmp_path / "data")

        manifest = store.put(
            content=b"schema test",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        manifest_dict = manifest.to_dict()

        # Required fields per artifact.v1 schema
        required_fields = [
            "schema_version",
            "artifact_id",
            "artifact_type",
            "content_hash",
            "byte_size",
            "created_utc",
            "provenance",
            "roles",
            "lineage",
        ]

        for field in required_fields:
            assert field in manifest_dict, f"Missing required field: {field}"

    def test_manifest_json_serialization(self, tmp_path: Path) -> None:
        """Test that manifest can be serialized to valid JSON."""
        store = ArtifactStore(tmp_path / "data")

        manifest = store.put(
            content=b"json test",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        json_str = manifest.to_json()
        parsed = json.loads(json_str)

        assert parsed["schema_version"] == 1
        assert parsed["artifact_id"] == manifest.artifact_id
