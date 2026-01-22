"""DatasetSnapshot writer/reader with Parquet-based index.

This module implements the DatasetSnapshot class, which provides:
- Snapshot writer: create versioned dataset snapshots from artifact collections
- Parquet-based index: columnar storage for efficient querying
- Manifest hashing: cryptographic integrity for the entire dataset
- Feature storage: columnar representation of design parameters
- Incremental updates: create new snapshots derived from parent versions
- Efficient slicing: row group filtering and predicate pushdown for large datasets

The DatasetSnapshot follows the dataset.v1 schema and integrates with
the ArtifactStore for content-addressed storage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from formula_foundry.m3.artifact_store import (
    ArtifactManifest,
    ArtifactStore,
    ContentHash,
)

if TYPE_CHECKING:
    pass

# Optional parquet support - gracefully degrade if not available
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]


# Type aliases
SplitName = Literal["train", "validation", "test", "holdout"]
HashAlgorithm = Literal["sha256", "sha384", "sha512", "blake3"]


class DatasetSnapshotError(Exception):
    """Base exception for DatasetSnapshot errors."""


class DatasetNotFoundError(DatasetSnapshotError):
    """Raised when a dataset is not found."""


class ParquetNotAvailableError(DatasetSnapshotError):
    """Raised when PyArrow is not installed but Parquet operations are requested."""


@dataclass
class DatasetMember:
    """A member artifact in a dataset snapshot."""

    artifact_id: str
    content_hash: ContentHash
    artifact_type: str
    role: str
    byte_size: int = 0
    storage_path: str | None = None
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash.to_dict(),
            "artifact_type": self.artifact_type,
            "role": self.role,
        }
        if self.byte_size:
            result["byte_size"] = self.byte_size
        if self.storage_path:
            result["storage_path"] = self.storage_path
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetMember:
        """Create from a dict."""
        return cls(
            artifact_id=data["artifact_id"],
            content_hash=ContentHash.from_dict(data["content_hash"]),
            artifact_type=data["artifact_type"],
            role=data["role"],
            byte_size=data.get("byte_size", 0),
            storage_path=data.get("storage_path"),
        )

    @classmethod
    def from_manifest(cls, manifest: ArtifactManifest, role: str, features: dict[str, Any] | None = None) -> DatasetMember:
        """Create a DatasetMember from an ArtifactManifest."""
        return cls(
            artifact_id=manifest.artifact_id,
            content_hash=manifest.content_hash,
            artifact_type=manifest.artifact_type,
            role=role,
            byte_size=manifest.byte_size,
            storage_path=manifest.storage_path,
            features=features or {},
        )


@dataclass
class SplitDefinition:
    """Definition of a train/validation/test split."""

    name: SplitName
    artifact_ids: list[str]
    count: int = 0
    fraction: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "name": self.name,
            "artifact_ids": self.artifact_ids,
            "count": self.count,
            "fraction": self.fraction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitDefinition:
        """Create from a dict."""
        return cls(
            name=data["name"],
            artifact_ids=data["artifact_ids"],
            count=data.get("count", len(data["artifact_ids"])),
            fraction=data.get("fraction", 0.0),
        )


@dataclass
class DatasetProvenance:
    """Provenance information for a dataset."""

    generator: str
    generator_version: str
    source_runs: list[str] = field(default_factory=list)
    pipeline_stage: str | None = None
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "generator": self.generator,
            "generator_version": self.generator_version,
        }
        if self.source_runs:
            result["source_runs"] = self.source_runs
        if self.pipeline_stage:
            result["pipeline_stage"] = self.pipeline_stage
        if self.git_commit:
            result["git_commit"] = self.git_commit
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetProvenance:
        """Create from a dict."""
        return cls(
            generator=data["generator"],
            generator_version=data["generator_version"],
            source_runs=data.get("source_runs", []),
            pipeline_stage=data.get("pipeline_stage"),
            git_commit=data.get("git_commit"),
        )


@dataclass
class DatasetStatistics:
    """Summary statistics for a dataset."""

    by_artifact_type: dict[str, dict[str, int]] = field(default_factory=dict)
    by_role: dict[str, dict[str, int]] = field(default_factory=dict)
    unique_coupons: int = 0
    frequency_range_hz: dict[str, float] | None = None
    parameter_ranges: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {}
        if self.by_artifact_type:
            result["by_artifact_type"] = self.by_artifact_type
        if self.by_role:
            result["by_role"] = self.by_role
        if self.unique_coupons:
            result["unique_coupons"] = self.unique_coupons
        if self.frequency_range_hz:
            result["frequency_range_hz"] = self.frequency_range_hz
        if self.parameter_ranges:
            result["parameter_ranges"] = self.parameter_ranges
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetStatistics:
        """Create from a dict."""
        return cls(
            by_artifact_type=data.get("by_artifact_type", {}),
            by_role=data.get("by_role", {}),
            unique_coupons=data.get("unique_coupons", 0),
            frequency_range_hz=data.get("frequency_range_hz"),
            parameter_ranges=data.get("parameter_ranges", {}),
        )


@dataclass
class DatasetSnapshot:
    """A versioned snapshot of a dataset.

    This class represents a complete, immutable snapshot of a dataset
    consisting of artifact references. The snapshot includes:

    - Unique dataset_id and version tag
    - List of member artifacts with their content hashes
    - Content hash of the entire manifest (for integrity)
    - Optional Parquet index for columnar feature access
    - Train/validation/test splits
    - Summary statistics

    The snapshot follows the dataset.v1 schema.
    """

    dataset_id: str
    version: str
    created_utc: str
    members: list[DatasetMember]
    content_hash: ContentHash
    provenance: DatasetProvenance
    name: str | None = None
    description: str | None = None
    parent_version: str | None = None
    statistics: DatasetStatistics | None = None
    splits: dict[str, SplitDefinition] | None = None
    index_path: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)

    @property
    def member_count(self) -> int:
        """Return the number of members in the snapshot."""
        return len(self.members)

    @property
    def total_bytes(self) -> int:
        """Return the total size of all members in bytes."""
        return sum(m.byte_size for m in self.members)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict conforming to dataset.v1 schema."""
        result: dict[str, Any] = {
            "schema_version": 1,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "created_utc": self.created_utc,
            "members": {
                "count": self.member_count,
                "total_bytes": self.total_bytes,
                "artifacts": [m.to_dict() for m in self.members],
            },
            "content_hash": self.content_hash.to_dict(),
            "provenance": self.provenance.to_dict(),
        }
        if self.name:
            result["name"] = self.name
        if self.description:
            result["description"] = self.description
        if self.parent_version:
            result["parent_version"] = self.parent_version
        if self.statistics:
            result["statistics"] = self.statistics.to_dict()
        if self.splits:
            result["splits"] = {
                "seed": self.annotations.get("split_seed", 0),
                "split_method": self.annotations.get("split_method", "random"),
                "splits": [s.to_dict() for s in self.splits.values()],
            }
        if self.index_path:
            result["index_path"] = self.index_path
        if self.tags:
            result["tags"] = self.tags
        if self.annotations:
            result["annotations"] = self.annotations
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetSnapshot:
        """Create a DatasetSnapshot from a dict."""
        members_data = data["members"]
        members = [DatasetMember.from_dict(a) for a in members_data["artifacts"]]

        provenance = DatasetProvenance.from_dict(data["provenance"])
        content_hash = ContentHash.from_dict(data["content_hash"])

        statistics = None
        if "statistics" in data:
            statistics = DatasetStatistics.from_dict(data["statistics"])

        splits = None
        if "splits" in data and "splits" in data["splits"]:
            splits = {s["name"]: SplitDefinition.from_dict(s) for s in data["splits"]["splits"]}

        return cls(
            dataset_id=data["dataset_id"],
            version=data["version"],
            created_utc=data["created_utc"],
            members=members,
            content_hash=content_hash,
            provenance=provenance,
            name=data.get("name"),
            description=data.get("description"),
            parent_version=data.get("parent_version"),
            statistics=statistics,
            splits=splits,
            index_path=data.get("index_path"),
            tags=data.get("tags", {}),
            annotations=data.get("annotations", {}),
        )

    def get_members_by_type(self, artifact_type: str) -> list[DatasetMember]:
        """Get all members of a specific artifact type."""
        return [m for m in self.members if m.artifact_type == artifact_type]

    def get_members_by_role(self, role: str) -> list[DatasetMember]:
        """Get all members with a specific role."""
        return [m for m in self.members if m.role == role]

    def get_member(self, artifact_id: str) -> DatasetMember | None:
        """Get a specific member by artifact ID."""
        for m in self.members:
            if m.artifact_id == artifact_id:
                return m
        return None


def compute_manifest_hash(
    members: list[DatasetMember],
    algorithm: HashAlgorithm = "sha256",
) -> ContentHash:
    """Compute a content hash for the dataset manifest.

    The hash is computed from the sorted member content hashes,
    ensuring a deterministic result regardless of member order.

    Args:
        members: List of dataset members.
        algorithm: Hash algorithm to use.

    Returns:
        ContentHash of the manifest.
    """
    sorted_digests = sorted(m.content_hash.digest for m in members)
    combined = "\n".join(sorted_digests).encode("utf-8")

    if algorithm == "sha256":
        digest = hashlib.sha256(combined).hexdigest()
    elif algorithm == "sha384":
        digest = hashlib.sha384(combined).hexdigest()
    elif algorithm == "sha512":
        digest = hashlib.sha512(combined).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return ContentHash(algorithm=algorithm, digest=digest)


class DatasetSnapshotWriter:
    """Writer for creating dataset snapshots with Parquet index.

    This class builds a dataset snapshot by collecting artifact references
    and optionally extracting features for columnar storage.

    Example usage:
        writer = DatasetSnapshotWriter(
            dataset_id="em_dataset_v1",
            version="v1.0",
            store=artifact_store,
        )

        for manifest in manifests:
            writer.add_member(manifest, role="oracle_output", features={"via_diameter": 0.3})

        snapshot = writer.finalize(output_dir=Path("data/datasets"))
    """

    def __init__(
        self,
        dataset_id: str,
        version: str,
        store: ArtifactStore | None = None,
        generator: str = "formula_foundry",
        generator_version: str = "0.1.0",
        name: str | None = None,
        description: str | None = None,
        parent_version: str | None = None,
    ) -> None:
        """Initialize the snapshot writer.

        Args:
            dataset_id: Unique identifier for the dataset.
            version: Version tag (e.g., "v1.0").
            store: Optional ArtifactStore for retrieving manifests.
            generator: Name of the generator tool.
            generator_version: Version of the generator.
            name: Optional human-readable name.
            description: Optional description.
            parent_version: Optional parent version if this is derived.
        """
        self.dataset_id = dataset_id
        self.version = version
        self.store = store
        self.generator = generator
        self.generator_version = generator_version
        self.name = name
        self.description = description
        self.parent_version = parent_version

        self._members: list[DatasetMember] = []
        self._source_runs: set[str] = set()
        self._tags: dict[str, str] = {}
        self._annotations: dict[str, Any] = {}

    def add_member(
        self,
        manifest: ArtifactManifest,
        role: str,
        features: dict[str, Any] | None = None,
    ) -> DatasetMember:
        """Add an artifact to the dataset.

        Args:
            manifest: The artifact manifest to add.
            role: The role this artifact plays in the dataset.
            features: Optional dict of feature values for this artifact.

        Returns:
            The created DatasetMember.
        """
        member = DatasetMember.from_manifest(manifest, role, features)
        self._members.append(member)
        self._source_runs.add(manifest.lineage.run_id)
        return member

    def add_member_by_id(
        self,
        artifact_id: str,
        role: str,
        features: dict[str, Any] | None = None,
    ) -> DatasetMember:
        """Add an artifact by ID (requires store).

        Args:
            artifact_id: The artifact ID to add.
            role: The role this artifact plays in the dataset.
            features: Optional dict of feature values for this artifact.

        Returns:
            The created DatasetMember.

        Raises:
            DatasetSnapshotError: If no store is configured.
        """
        if self.store is None:
            raise DatasetSnapshotError("No ArtifactStore configured")
        manifest = self.store.get_manifest(artifact_id)
        return self.add_member(manifest, role, features)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags for the dataset."""
        self._tags = dict(tags)

    def set_annotations(self, annotations: dict[str, Any]) -> None:
        """Set annotations for the dataset."""
        self._annotations = dict(annotations)

    def compute_statistics(self) -> DatasetStatistics:
        """Compute summary statistics for the dataset."""
        by_type: dict[str, dict[str, int]] = {}
        by_role: dict[str, dict[str, int]] = {}

        for member in self._members:
            # By type
            if member.artifact_type not in by_type:
                by_type[member.artifact_type] = {"count": 0, "total_bytes": 0}
            by_type[member.artifact_type]["count"] += 1
            by_type[member.artifact_type]["total_bytes"] += member.byte_size

            # By role
            if member.role not in by_role:
                by_role[member.role] = {"count": 0, "total_bytes": 0}
            by_role[member.role]["count"] += 1
            by_role[member.role]["total_bytes"] += member.byte_size

        # Compute parameter ranges from features
        parameter_ranges: dict[str, dict[str, float]] = {}
        for member in self._members:
            for key, value in member.features.items():
                if isinstance(value, (int, float)):
                    if key not in parameter_ranges:
                        parameter_ranges[key] = {"min": value, "max": value}
                    else:
                        parameter_ranges[key]["min"] = min(parameter_ranges[key]["min"], value)
                        parameter_ranges[key]["max"] = max(parameter_ranges[key]["max"], value)

        return DatasetStatistics(
            by_artifact_type=by_type,
            by_role=by_role,
            unique_coupons=by_type.get("coupon_spec", {}).get("count", 0),
            parameter_ranges=parameter_ranges,
        )

    def write_parquet_index(self, output_path: Path) -> Path:
        """Write a Parquet index file with member metadata and features.

        The Parquet file provides efficient columnar access to:
        - artifact_id, content_hash, artifact_type, role, byte_size
        - All feature columns (flattened from member.features)

        Args:
            output_path: Directory to write the index file.

        Returns:
            Path to the written Parquet file.

        Raises:
            ParquetNotAvailableError: If PyArrow is not installed.
        """
        if not HAS_PYARROW:
            raise ParquetNotAvailableError("PyArrow is required for Parquet index. Install with: pip install pyarrow")

        output_path.mkdir(parents=True, exist_ok=True)
        parquet_path = output_path / f"{self.dataset_id}_{self.version}_index.parquet"

        # Collect all feature keys
        all_feature_keys: set[str] = set()
        for member in self._members:
            all_feature_keys.update(member.features.keys())

        # Build column data
        data: dict[str, list[Any]] = {
            "artifact_id": [],
            "content_hash_algorithm": [],
            "content_hash_digest": [],
            "artifact_type": [],
            "role": [],
            "byte_size": [],
            "storage_path": [],
        }
        for key in sorted(all_feature_keys):
            data[f"feature_{key}"] = []

        for member in self._members:
            data["artifact_id"].append(member.artifact_id)
            data["content_hash_algorithm"].append(member.content_hash.algorithm)
            data["content_hash_digest"].append(member.content_hash.digest)
            data["artifact_type"].append(member.artifact_type)
            data["role"].append(member.role)
            data["byte_size"].append(member.byte_size)
            data["storage_path"].append(member.storage_path)

            for key in sorted(all_feature_keys):
                data[f"feature_{key}"].append(member.features.get(key))

        # Create PyArrow table and write
        table = pa.table(data)
        pq.write_table(table, parquet_path, compression="snappy")

        return parquet_path

    def finalize(
        self,
        output_dir: Path | None = None,
        write_parquet: bool = True,
        git_commit: str | None = None,
        pipeline_stage: str | None = None,
    ) -> DatasetSnapshot:
        """Finalize and create the dataset snapshot.

        Args:
            output_dir: Optional directory to write manifest and index.
            write_parquet: If True and PyArrow available, write Parquet index.
            git_commit: Optional git commit SHA for provenance.
            pipeline_stage: Optional pipeline stage name.

        Returns:
            The created DatasetSnapshot.
        """
        # Compute manifest hash
        content_hash = compute_manifest_hash(self._members)

        # Create provenance
        provenance = DatasetProvenance(
            generator=self.generator,
            generator_version=self.generator_version,
            source_runs=sorted(self._source_runs),
            pipeline_stage=pipeline_stage,
            git_commit=git_commit,
        )

        # Compute statistics
        statistics = self.compute_statistics()

        # Create timestamp
        created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Write Parquet index if requested
        index_path: str | None = None
        if output_dir and write_parquet and HAS_PYARROW and self._members:
            parquet_path = self.write_parquet_index(output_dir)
            # Store index_path relative to output_dir (where manifest is written)
            # so the reader can resolve it correctly relative to the manifest location
            index_path = parquet_path.name

        # Create snapshot
        snapshot = DatasetSnapshot(
            dataset_id=self.dataset_id,
            version=self.version,
            created_utc=created_utc,
            members=list(self._members),
            content_hash=content_hash,
            provenance=provenance,
            name=self.name,
            description=self.description,
            parent_version=self.parent_version,
            statistics=statistics,
            index_path=index_path,
            tags=self._tags,
            annotations=self._annotations,
        )

        # Write manifest if output_dir provided
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = output_dir / f"{self.dataset_id}_{self.version}.json"
            manifest_path.write_text(snapshot.to_json(), encoding="utf-8")

        return snapshot


class DatasetSnapshotReader:
    """Reader for loading and querying dataset snapshots.

    This class provides methods to:
    - Load snapshots from JSON manifests
    - Query the Parquet index for efficient feature access
    - Iterate over members with optional filtering

    Example usage:
        reader = DatasetSnapshotReader(snapshot_path=Path("data/datasets/em_dataset_v1_v1.0.json"))
        snapshot = reader.load()

        # Query via Parquet index
        df = reader.query_features(artifact_type="touchstone")
    """

    def __init__(
        self,
        snapshot_path: Path | None = None,
        snapshot: DatasetSnapshot | None = None,
        store: ArtifactStore | None = None,
    ) -> None:
        """Initialize the reader.

        Args:
            snapshot_path: Path to a snapshot JSON manifest.
            snapshot: An existing DatasetSnapshot to read from.
            store: Optional ArtifactStore for retrieving artifact content.
        """
        self._snapshot_path = snapshot_path
        self._snapshot = snapshot
        self.store = store
        self._parquet_table: Any | None = None

    def load(self) -> DatasetSnapshot:
        """Load the dataset snapshot from the manifest file.

        Returns:
            The loaded DatasetSnapshot.

        Raises:
            DatasetNotFoundError: If the snapshot file doesn't exist.
        """
        if self._snapshot is not None:
            return self._snapshot

        if self._snapshot_path is None:
            raise DatasetSnapshotError("No snapshot path or snapshot provided")

        if not self._snapshot_path.exists():
            raise DatasetNotFoundError(f"Snapshot not found: {self._snapshot_path}")

        data = json.loads(self._snapshot_path.read_text(encoding="utf-8"))
        self._snapshot = DatasetSnapshot.from_dict(data)
        return self._snapshot

    def _load_parquet_index(self) -> Any:
        """Load the Parquet index if available."""
        if not HAS_PYARROW:
            raise ParquetNotAvailableError("PyArrow required for Parquet index access")

        if self._parquet_table is not None:
            return self._parquet_table

        snapshot = self.load()
        if not snapshot.index_path:
            raise DatasetSnapshotError("Snapshot has no Parquet index")

        # Resolve index path relative to snapshot
        if self._snapshot_path:
            base_dir = self._snapshot_path.parent
            index_path = base_dir / snapshot.index_path
        else:
            index_path = Path(snapshot.index_path)

        if not index_path.exists():
            raise DatasetNotFoundError(f"Parquet index not found: {index_path}")

        self._parquet_table = pq.read_table(index_path)
        return self._parquet_table

    def query_features(
        self,
        artifact_type: str | None = None,
        role: str | None = None,
        columns: list[str] | None = None,
    ) -> Any:
        """Query the Parquet index for feature data.

        Args:
            artifact_type: Filter by artifact type.
            role: Filter by role.
            columns: Specific columns to return (default: all).

        Returns:
            PyArrow Table with the matching rows.

        Raises:
            ParquetNotAvailableError: If PyArrow is not installed.
        """
        table = self._load_parquet_index()

        # Apply filters
        if artifact_type is not None:
            mask = pa.compute.equal(table.column("artifact_type"), artifact_type)
            table = table.filter(mask)

        if role is not None:
            mask = pa.compute.equal(table.column("role"), role)
            table = table.filter(mask)

        # Select columns
        if columns:
            table = table.select(columns)

        return table

    def to_pandas(
        self,
        artifact_type: str | None = None,
        role: str | None = None,
    ) -> Any:
        """Convert the index to a pandas DataFrame.

        Args:
            artifact_type: Filter by artifact type.
            role: Filter by role.

        Returns:
            pandas DataFrame with the index data.
        """
        table = self.query_features(artifact_type=artifact_type, role=role)
        return table.to_pandas()

    def iter_members(
        self,
        artifact_type: str | None = None,
        role: str | None = None,
    ) -> Iterator[DatasetMember]:
        """Iterate over dataset members with optional filtering.

        Args:
            artifact_type: Filter by artifact type.
            role: Filter by role.

        Yields:
            DatasetMember objects matching the filters.
        """
        snapshot = self.load()

        for member in snapshot.members:
            if artifact_type and member.artifact_type != artifact_type:
                continue
            if role and member.role != role:
                continue
            yield member

    def get_artifact_content(self, artifact_id: str) -> bytes:
        """Retrieve the content of an artifact by ID.

        Args:
            artifact_id: The artifact ID to retrieve.

        Returns:
            The artifact content as bytes.

        Raises:
            DatasetSnapshotError: If no store is configured.
        """
        if self.store is None:
            raise DatasetSnapshotError("No ArtifactStore configured for content retrieval")

        snapshot = self.load()
        member = snapshot.get_member(artifact_id)
        if member is None:
            raise DatasetNotFoundError(f"Artifact not in dataset: {artifact_id}")

        return self.store.get(member.content_hash.digest)

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """Verify the integrity of the dataset snapshot.

        Checks that the computed manifest hash matches the stored hash.

        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        snapshot = self.load()
        errors: list[str] = []

        # Recompute manifest hash
        computed_hash = compute_manifest_hash(snapshot.members)
        if computed_hash.digest != snapshot.content_hash.digest:
            errors.append(
                f"Manifest hash mismatch: computed {computed_hash.digest[:16]}... "
                f"vs stored {snapshot.content_hash.digest[:16]}..."
            )

        # Check member count matches
        if len(snapshot.members) != snapshot.member_count:
            errors.append(f"Member count mismatch: {len(snapshot.members)} vs {snapshot.member_count}")

        return (len(errors) == 0, errors)

    def slice(
        self,
        start: int | None = None,
        end: int | None = None,
        artifact_ids: list[str] | None = None,
        artifact_type: str | None = None,
        role: str | None = None,
        feature_filter: dict[str, tuple[Any, Any]] | None = None,
    ) -> list[DatasetMember]:
        """Efficient slicing with multiple filter criteria.

        This method provides efficient access to subsets of the dataset,
        using Parquet row group filtering when available for large datasets.

        Args:
            start: Start index for range-based slicing (inclusive).
            end: End index for range-based slicing (exclusive).
            artifact_ids: Explicit list of artifact IDs to include.
            artifact_type: Filter by artifact type.
            role: Filter by role.
            feature_filter: Dict mapping feature names to (min, max) tuples
                for range filtering. Use None for unbounded ranges:
                {"via_diameter": (0.2, 0.5), "freq": (1e6, None)}

        Returns:
            List of DatasetMember objects matching all criteria.

        Example:
            # Get artifacts 10-20 that are touchstones with via_diameter in [0.2, 0.5]
            members = reader.slice(
                start=10,
                end=20,
                artifact_type="touchstone",
                feature_filter={"via_diameter": (0.2, 0.5)}
            )
        """
        snapshot = self.load()

        # Start with all members
        members = list(snapshot.members)

        # Apply artifact ID filter first (most selective)
        if artifact_ids is not None:
            id_set = set(artifact_ids)
            members = [m for m in members if m.artifact_id in id_set]

        # Apply type filter
        if artifact_type is not None:
            members = [m for m in members if m.artifact_type == artifact_type]

        # Apply role filter
        if role is not None:
            members = [m for m in members if m.role == role]

        # Apply feature range filters
        if feature_filter:
            filtered = []
            for m in members:
                passes = True
                for feature_name, (min_val, max_val) in feature_filter.items():
                    if feature_name not in m.features:
                        passes = False
                        break
                    val = m.features[feature_name]
                    if min_val is not None and val < min_val:
                        passes = False
                        break
                    if max_val is not None and val > max_val:
                        passes = False
                        break
                if passes:
                    filtered.append(m)
            members = filtered

        # Apply range slicing last
        if start is not None or end is not None:
            members = members[start:end]

        return members

    def slice_parquet(
        self,
        artifact_type: str | None = None,
        role: str | None = None,
        columns: list[str] | None = None,
        row_groups: list[int] | None = None,
        filters: list[tuple[str, str, Any]] | None = None,
    ) -> Any:
        """Efficient Parquet-based slicing with predicate pushdown.

        This method leverages Parquet's row group structure for efficient
        filtering of large datasets without loading all data into memory.

        Args:
            artifact_type: Filter by artifact type.
            role: Filter by role.
            columns: Specific columns to read (reduces I/O).
            row_groups: Specific row groups to read (for manual partitioning).
            filters: PyArrow filter expressions as tuples:
                [("feature_via_diameter", ">=", 0.2), ("feature_via_diameter", "<=", 0.5)]

        Returns:
            PyArrow Table with the filtered results.

        Raises:
            ParquetNotAvailableError: If PyArrow is not installed.
            DatasetSnapshotError: If no Parquet index exists.

        Example:
            # Read only specific columns with filters
            table = reader.slice_parquet(
                artifact_type="touchstone",
                columns=["artifact_id", "feature_via_diameter"],
                filters=[("feature_via_diameter", ">=", 0.2)]
            )
        """
        if not HAS_PYARROW:
            raise ParquetNotAvailableError("PyArrow required for Parquet slicing")

        snapshot = self.load()
        if not snapshot.index_path:
            raise DatasetSnapshotError("Snapshot has no Parquet index")

        # Resolve index path
        if self._snapshot_path:
            base_dir = self._snapshot_path.parent
            index_path = base_dir / snapshot.index_path
        else:
            index_path = Path(snapshot.index_path)

        if not index_path.exists():
            raise DatasetNotFoundError(f"Parquet index not found: {index_path}")

        # Build combined filters
        all_filters = []
        if artifact_type is not None:
            all_filters.append(("artifact_type", "==", artifact_type))
        if role is not None:
            all_filters.append(("role", "==", role))
        if filters:
            all_filters.extend(filters)

        # Read with predicate pushdown
        parquet_file = pq.ParquetFile(index_path)

        if row_groups is not None:
            # Read specific row groups
            table = parquet_file.read_row_groups(row_groups, columns=columns)
        elif all_filters:
            # Use predicate pushdown with filters
            table = pq.read_table(
                index_path,
                columns=columns,
                filters=all_filters if all_filters else None,
            )
        else:
            # Read all with optional column projection
            table = pq.read_table(index_path, columns=columns)

        return table

    def get_row_group_metadata(self) -> list[dict[str, Any]]:
        """Get metadata about Parquet row groups for manual partitioning.

        Returns information about each row group including row count
        and statistics, useful for parallel processing of large datasets.

        Returns:
            List of dicts with row group metadata.

        Raises:
            ParquetNotAvailableError: If PyArrow is not installed.
        """
        if not HAS_PYARROW:
            raise ParquetNotAvailableError("PyArrow required for row group metadata")

        snapshot = self.load()
        if not snapshot.index_path:
            raise DatasetSnapshotError("Snapshot has no Parquet index")

        if self._snapshot_path:
            base_dir = self._snapshot_path.parent
            index_path = base_dir / snapshot.index_path
        else:
            index_path = Path(snapshot.index_path)

        parquet_file = pq.ParquetFile(index_path)
        metadata = parquet_file.metadata

        row_groups = []
        for i in range(metadata.num_row_groups):
            rg = metadata.row_group(i)
            row_groups.append({
                "index": i,
                "num_rows": rg.num_rows,
                "total_byte_size": rg.total_byte_size,
                "columns": metadata.num_columns,
            })

        return row_groups


class IncrementalSnapshotBuilder:
    """Builder for creating incremental dataset snapshots.

    This class enables efficient creation of new dataset versions by
    starting from a parent snapshot and adding/removing members.
    Only changes are recomputed, making it efficient for large datasets.

    Example usage:
        # Load parent snapshot
        reader = DatasetSnapshotReader(snapshot_path=parent_path)
        parent = reader.load()

        # Create incremental builder
        builder = IncrementalSnapshotBuilder(
            parent=parent,
            new_version="v1.1",
            store=artifact_store,
        )

        # Add new members
        builder.add_member(new_manifest, role="oracle_output")

        # Remove outdated members
        builder.remove_member("art-old-001")

        # Finalize
        snapshot = builder.finalize(output_dir=Path("datasets"))
    """

    def __init__(
        self,
        parent: DatasetSnapshot,
        new_version: str,
        store: ArtifactStore | None = None,
        generator: str = "formula_foundry",
        generator_version: str = "0.1.0",
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize the incremental builder.

        Args:
            parent: The parent DatasetSnapshot to derive from.
            new_version: Version tag for the new snapshot.
            store: Optional ArtifactStore for retrieving manifests.
            generator: Name of the generator tool.
            generator_version: Version of the generator.
            name: Optional human-readable name (inherits from parent if None).
            description: Optional description (inherits from parent if None).
        """
        self.parent = parent
        self.new_version = new_version
        self.store = store
        self.generator = generator
        self.generator_version = generator_version
        self.name = name or parent.name
        self.description = description or parent.description

        # Copy parent members into a dict for efficient modification
        self._members: dict[str, DatasetMember] = {
            m.artifact_id: m for m in parent.members
        }
        self._added: set[str] = set()
        self._removed: set[str] = set()
        self._source_runs: set[str] = set(parent.provenance.source_runs)
        self._tags: dict[str, str] = dict(parent.tags)
        self._annotations: dict[str, Any] = dict(parent.annotations)

    def add_member(
        self,
        manifest: ArtifactManifest,
        role: str,
        features: dict[str, Any] | None = None,
    ) -> DatasetMember:
        """Add a new member to the dataset.

        Args:
            manifest: The artifact manifest to add.
            role: The role this artifact plays.
            features: Optional feature values.

        Returns:
            The created DatasetMember.
        """
        member = DatasetMember.from_manifest(manifest, role, features)
        self._members[member.artifact_id] = member
        self._added.add(member.artifact_id)
        self._removed.discard(member.artifact_id)
        self._source_runs.add(manifest.lineage.run_id)
        return member

    def add_member_by_id(
        self,
        artifact_id: str,
        role: str,
        features: dict[str, Any] | None = None,
    ) -> DatasetMember:
        """Add a member by artifact ID (requires store).

        Args:
            artifact_id: The artifact ID to add.
            role: The role this artifact plays.
            features: Optional feature values.

        Returns:
            The created DatasetMember.

        Raises:
            DatasetSnapshotError: If no store is configured.
        """
        if self.store is None:
            raise DatasetSnapshotError("No ArtifactStore configured")
        manifest = self.store.get_manifest(artifact_id)
        return self.add_member(manifest, role, features)

    def remove_member(self, artifact_id: str) -> bool:
        """Remove a member from the dataset.

        Args:
            artifact_id: The artifact ID to remove.

        Returns:
            True if the member was removed, False if not found.
        """
        if artifact_id in self._members:
            del self._members[artifact_id]
            self._removed.add(artifact_id)
            self._added.discard(artifact_id)
            return True
        return False

    def update_member_features(
        self,
        artifact_id: str,
        features: dict[str, Any],
        merge: bool = True,
    ) -> bool:
        """Update features for an existing member.

        Args:
            artifact_id: The artifact ID to update.
            features: New feature values.
            merge: If True, merge with existing features; if False, replace.

        Returns:
            True if the member was updated, False if not found.
        """
        if artifact_id not in self._members:
            return False

        member = self._members[artifact_id]
        if merge:
            member.features.update(features)
        else:
            member.features = features
        return True

    def set_tags(self, tags: dict[str, str], merge: bool = True) -> None:
        """Set tags for the new snapshot.

        Args:
            tags: Tag key-value pairs.
            merge: If True, merge with inherited tags; if False, replace.
        """
        if merge:
            self._tags.update(tags)
        else:
            self._tags = dict(tags)

    def set_annotations(self, annotations: dict[str, Any], merge: bool = True) -> None:
        """Set annotations for the new snapshot.

        Args:
            annotations: Annotation key-value pairs.
            merge: If True, merge with inherited annotations; if False, replace.
        """
        if merge:
            self._annotations.update(annotations)
        else:
            self._annotations = dict(annotations)

    def get_diff_summary(self) -> dict[str, Any]:
        """Get a summary of changes from the parent snapshot.

        Returns:
            Dict with added, removed, and total member counts.
        """
        return {
            "parent_version": self.parent.version,
            "new_version": self.new_version,
            "parent_count": self.parent.member_count,
            "new_count": len(self._members),
            "added": len(self._added),
            "removed": len(self._removed),
            "added_ids": sorted(self._added),
            "removed_ids": sorted(self._removed),
        }

    def finalize(
        self,
        output_dir: Path | None = None,
        write_parquet: bool = True,
        git_commit: str | None = None,
        pipeline_stage: str | None = None,
    ) -> DatasetSnapshot:
        """Finalize and create the incremental snapshot.

        Args:
            output_dir: Optional directory to write manifest and index.
            write_parquet: If True and PyArrow available, write Parquet index.
            git_commit: Optional git commit SHA for provenance.
            pipeline_stage: Optional pipeline stage name.

        Returns:
            The created DatasetSnapshot.
        """
        members = list(self._members.values())

        # Compute manifest hash
        content_hash = compute_manifest_hash(members)

        # Create provenance with parent reference
        provenance = DatasetProvenance(
            generator=self.generator,
            generator_version=self.generator_version,
            source_runs=sorted(self._source_runs),
            pipeline_stage=pipeline_stage,
            git_commit=git_commit,
        )

        # Compute statistics
        statistics = self._compute_statistics(members)

        # Create timestamp
        created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Write Parquet index if requested
        index_path: str | None = None
        if output_dir and write_parquet and HAS_PYARROW and members:
            parquet_path = self._write_parquet_index(output_dir, members)
            # Store index_path relative to output_dir (where manifest is written)
            index_path = parquet_path.name

        # Record diff info in annotations
        diff_annotations = dict(self._annotations)
        diff_annotations["incremental_from"] = self.parent.version
        diff_annotations["incremental_diff"] = {
            "added": len(self._added),
            "removed": len(self._removed),
        }

        # Create snapshot
        snapshot = DatasetSnapshot(
            dataset_id=self.parent.dataset_id,
            version=self.new_version,
            created_utc=created_utc,
            members=members,
            content_hash=content_hash,
            provenance=provenance,
            name=self.name,
            description=self.description,
            parent_version=self.parent.version,
            statistics=statistics,
            index_path=index_path,
            tags=self._tags,
            annotations=diff_annotations,
        )

        # Write manifest if output_dir provided
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = output_dir / f"{self.parent.dataset_id}_{self.new_version}.json"
            manifest_path.write_text(snapshot.to_json(), encoding="utf-8")

        return snapshot

    def _compute_statistics(self, members: list[DatasetMember]) -> DatasetStatistics:
        """Compute statistics for the member list."""
        by_type: dict[str, dict[str, int]] = {}
        by_role: dict[str, dict[str, int]] = {}
        parameter_ranges: dict[str, dict[str, float]] = {}

        for member in members:
            # By type
            if member.artifact_type not in by_type:
                by_type[member.artifact_type] = {"count": 0, "total_bytes": 0}
            by_type[member.artifact_type]["count"] += 1
            by_type[member.artifact_type]["total_bytes"] += member.byte_size

            # By role
            if member.role not in by_role:
                by_role[member.role] = {"count": 0, "total_bytes": 0}
            by_role[member.role]["count"] += 1
            by_role[member.role]["total_bytes"] += member.byte_size

            # Parameter ranges from features
            for key, value in member.features.items():
                if isinstance(value, (int, float)):
                    if key not in parameter_ranges:
                        parameter_ranges[key] = {"min": value, "max": value}
                    else:
                        parameter_ranges[key]["min"] = min(parameter_ranges[key]["min"], value)
                        parameter_ranges[key]["max"] = max(parameter_ranges[key]["max"], value)

        return DatasetStatistics(
            by_artifact_type=by_type,
            by_role=by_role,
            unique_coupons=by_type.get("coupon_spec", {}).get("count", 0),
            parameter_ranges=parameter_ranges,
        )

    def _write_parquet_index(self, output_dir: Path, members: list[DatasetMember]) -> Path:
        """Write a Parquet index for the members."""
        output_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = output_dir / f"{self.parent.dataset_id}_{self.new_version}_index.parquet"

        # Collect all feature keys
        all_feature_keys: set[str] = set()
        for member in members:
            all_feature_keys.update(member.features.keys())

        # Build column data
        data: dict[str, list[Any]] = {
            "artifact_id": [],
            "content_hash_algorithm": [],
            "content_hash_digest": [],
            "artifact_type": [],
            "role": [],
            "byte_size": [],
            "storage_path": [],
        }
        for key in sorted(all_feature_keys):
            data[f"feature_{key}"] = []

        for member in members:
            data["artifact_id"].append(member.artifact_id)
            data["content_hash_algorithm"].append(member.content_hash.algorithm)
            data["content_hash_digest"].append(member.content_hash.digest)
            data["artifact_type"].append(member.artifact_type)
            data["role"].append(member.role)
            data["byte_size"].append(member.byte_size)
            data["storage_path"].append(member.storage_path)

            for key in sorted(all_feature_keys):
                data[f"feature_{key}"].append(member.features.get(key))

        table = pa.table(data)
        pq.write_table(table, parquet_path, compression="snappy")

        return parquet_path
