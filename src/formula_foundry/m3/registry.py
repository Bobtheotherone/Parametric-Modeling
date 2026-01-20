"""SQLite-backed artifact registry for indexing and querying artifacts.

This module provides the ArtifactRegistry class which maintains a SQLite database
index over artifact manifests. The registry is derived (rebuildable from manifests)
and provides fast queries for artifacts, datasets, and runs.

Tables:
    - artifacts: Index of all artifact manifests
    - datasets: Dataset snapshots and their metadata
    - runs: Pipeline/experiment run metadata

Indexes:
    - artifact_id (unique)
    - artifact_type
    - created_utc

The registry is designed to be rebuilt from the manifest files at any time,
making it a cache/index rather than a source of truth.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from formula_foundry.m3.artifact_store import ArtifactManifest, ArtifactStore


# Schema version for migration support
SCHEMA_VERSION = 1


# SQL statements for table creation
_CREATE_ARTIFACTS_TABLE = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    artifact_type TEXT NOT NULL,
    content_hash_algorithm TEXT NOT NULL,
    content_hash_digest TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    created_utc TEXT NOT NULL,
    run_id TEXT,
    stage_name TEXT,
    dataset_id TEXT,
    storage_path TEXT,
    media_type TEXT,
    roles TEXT NOT NULL,
    tags TEXT,
    manifest_path TEXT,
    indexed_utc TEXT NOT NULL
);
"""

_CREATE_DATASETS_TABLE = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    artifact_count INTEGER NOT NULL,
    total_bytes INTEGER NOT NULL,
    created_utc TEXT NOT NULL,
    description TEXT,
    manifest_hash TEXT,
    parquet_index_path TEXT,
    indexed_utc TEXT NOT NULL
);
"""

_CREATE_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    stage_name TEXT,
    started_utc TEXT NOT NULL,
    ended_utc TEXT,
    status TEXT NOT NULL,
    hostname TEXT,
    generator TEXT,
    generator_version TEXT,
    artifact_count INTEGER DEFAULT 0,
    config TEXT,
    indexed_utc TEXT NOT NULL
);
"""

# Indexes for fast queries
_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts (artifact_type);",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_created ON artifacts (created_utc);",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts (run_id);",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_dataset ON artifacts (dataset_id);",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_hash ON artifacts (content_hash_digest);",
    "CREATE INDEX IF NOT EXISTS idx_datasets_created ON datasets (created_utc);",
    "CREATE INDEX IF NOT EXISTS idx_runs_started ON runs (started_utc);",
    "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status);",
]

# Schema version table
_CREATE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""


@dataclass
class ArtifactRecord:
    """Record representing an artifact in the registry."""

    artifact_id: str
    artifact_type: str
    content_hash_algorithm: str
    content_hash_digest: str
    byte_size: int
    created_utc: str
    run_id: str | None
    stage_name: str | None
    dataset_id: str | None
    storage_path: str | None
    media_type: str | None
    roles: list[str]
    tags: dict[str, str]
    manifest_path: str | None
    indexed_utc: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ArtifactRecord:
        """Create an ArtifactRecord from a database row."""
        return cls(
            artifact_id=row["artifact_id"],
            artifact_type=row["artifact_type"],
            content_hash_algorithm=row["content_hash_algorithm"],
            content_hash_digest=row["content_hash_digest"],
            byte_size=row["byte_size"],
            created_utc=row["created_utc"],
            run_id=row["run_id"],
            stage_name=row["stage_name"],
            dataset_id=row["dataset_id"],
            storage_path=row["storage_path"],
            media_type=row["media_type"],
            roles=json.loads(row["roles"]) if row["roles"] else [],
            tags=json.loads(row["tags"]) if row["tags"] else {},
            manifest_path=row["manifest_path"],
            indexed_utc=row["indexed_utc"],
        )


@dataclass
class DatasetRecord:
    """Record representing a dataset in the registry."""

    dataset_id: str
    version: str
    artifact_count: int
    total_bytes: int
    created_utc: str
    description: str | None
    manifest_hash: str | None
    parquet_index_path: str | None
    indexed_utc: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> DatasetRecord:
        """Create a DatasetRecord from a database row."""
        return cls(
            dataset_id=row["dataset_id"],
            version=row["version"],
            artifact_count=row["artifact_count"],
            total_bytes=row["total_bytes"],
            created_utc=row["created_utc"],
            description=row["description"],
            manifest_hash=row["manifest_hash"],
            parquet_index_path=row["parquet_index_path"],
            indexed_utc=row["indexed_utc"],
        )


@dataclass
class RunRecord:
    """Record representing a run in the registry."""

    run_id: str
    stage_name: str | None
    started_utc: str
    ended_utc: str | None
    status: str
    hostname: str | None
    generator: str | None
    generator_version: str | None
    artifact_count: int
    config: dict[str, Any] | None
    indexed_utc: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> RunRecord:
        """Create a RunRecord from a database row."""
        return cls(
            run_id=row["run_id"],
            stage_name=row["stage_name"],
            started_utc=row["started_utc"],
            ended_utc=row["ended_utc"],
            status=row["status"],
            hostname=row["hostname"],
            generator=row["generator"],
            generator_version=row["generator_version"],
            artifact_count=row["artifact_count"],
            config=json.loads(row["config"]) if row["config"] else None,
            indexed_utc=row["indexed_utc"],
        )


class RegistryError(Exception):
    """Base exception for registry errors."""


class ArtifactNotIndexedError(RegistryError):
    """Raised when an artifact is not found in the registry index."""


class ArtifactRegistry:
    """SQLite-backed registry for indexing and querying artifacts.

    The registry maintains an SQLite database that indexes artifact manifests
    for fast queries. It is designed to be derived/rebuildable from the
    source manifest files.

    Thread Safety:
        The registry uses a thread-local connection pool to ensure thread safety.
        Each thread gets its own connection.

    Example usage:
        registry = ArtifactRegistry(Path("data/registry.db"))
        registry.initialize()

        # Index an artifact from a manifest
        registry.index_artifact(manifest)

        # Query artifacts
        records = registry.query_artifacts(artifact_type="touchstone")

        # Rebuild from manifests
        registry.rebuild_from_store(artifact_store)
    """

    def __init__(self, db_path: Path | str) -> None:
        """Initialize the artifact registry.

        Args:
            db_path: Path to the SQLite database file. Will be created if
                    it doesn't exist.
        """
        self.db_path = Path(db_path)
        self._local = threading.local()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Enable foreign keys and WAL mode for better concurrency
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for database transactions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _now_utc_iso(self) -> str:
        """Get current UTC time in ISO 8601 format."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def initialize(self) -> None:
        """Initialize the database schema.

        Creates all required tables and indexes if they don't exist.
        Safe to call multiple times.
        """
        with self._transaction() as cursor:
            # Create schema version table
            cursor.execute(_CREATE_SCHEMA_VERSION_TABLE)

            # Check existing schema version
            cursor.execute("SELECT version FROM schema_version LIMIT 1;")
            row = cursor.fetchone()

            if row is None:
                # Fresh database, create tables
                cursor.execute(_CREATE_ARTIFACTS_TABLE)
                cursor.execute(_CREATE_DATASETS_TABLE)
                cursor.execute(_CREATE_RUNS_TABLE)

                for index_sql in _CREATE_INDEXES:
                    cursor.execute(index_sql)

                cursor.execute(
                    "INSERT INTO schema_version (version) VALUES (?);",
                    (SCHEMA_VERSION,),
                )
            elif row["version"] < SCHEMA_VERSION:
                # Future: handle migrations here
                pass

    def close(self) -> None:
        """Close the database connection for the current thread."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    def index_artifact(
        self,
        manifest: ArtifactManifest,
        manifest_path: str | None = None,
    ) -> None:
        """Index an artifact manifest in the registry.

        Args:
            manifest: The artifact manifest to index.
            manifest_path: Optional path to the manifest file.
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO artifacts (
                    artifact_id, artifact_type, content_hash_algorithm,
                    content_hash_digest, byte_size, created_utc, run_id,
                    stage_name, dataset_id, storage_path, media_type,
                    roles, tags, manifest_path, indexed_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.artifact_id,
                    manifest.artifact_type,
                    manifest.content_hash.algorithm,
                    manifest.content_hash.digest,
                    manifest.byte_size,
                    manifest.created_utc,
                    manifest.lineage.run_id,
                    manifest.lineage.stage_name,
                    manifest.lineage.dataset_id,
                    manifest.storage_path,
                    manifest.media_type,
                    json.dumps(manifest.roles),
                    json.dumps(manifest.tags) if manifest.tags else None,
                    manifest_path,
                    self._now_utc_iso(),
                ),
            )

            # Update or create run record
            self._upsert_run_from_manifest(cursor, manifest)

    def _upsert_run_from_manifest(
        self,
        cursor: sqlite3.Cursor,
        manifest: ArtifactManifest,
    ) -> None:
        """Update or insert a run record based on a manifest."""
        run_id = manifest.lineage.run_id
        if not run_id:
            return

        # Check if run exists
        cursor.execute("SELECT run_id, artifact_count FROM runs WHERE run_id = ?;", (run_id,))
        row = cursor.fetchone()

        if row is None:
            # Create new run record
            cursor.execute(
                """
                INSERT INTO runs (
                    run_id, stage_name, started_utc, status, hostname,
                    generator, generator_version, artifact_count, indexed_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    manifest.lineage.stage_name,
                    manifest.created_utc,  # Use artifact creation as run start
                    "in_progress",
                    manifest.provenance.hostname,
                    manifest.provenance.generator,
                    manifest.provenance.generator_version,
                    1,
                    self._now_utc_iso(),
                ),
            )
        else:
            # Update artifact count
            cursor.execute(
                """
                UPDATE runs SET artifact_count = artifact_count + 1, indexed_utc = ?
                WHERE run_id = ?
                """,
                (self._now_utc_iso(), run_id),
            )

    def index_dataset(
        self,
        dataset_id: str,
        version: str,
        artifact_count: int,
        total_bytes: int,
        created_utc: str,
        description: str | None = None,
        manifest_hash: str | None = None,
        parquet_index_path: str | None = None,
    ) -> None:
        """Index a dataset in the registry.

        Args:
            dataset_id: Unique identifier for the dataset.
            version: Version string for the dataset.
            artifact_count: Number of artifacts in the dataset.
            total_bytes: Total size of all artifacts in bytes.
            created_utc: ISO 8601 creation timestamp.
            description: Optional description of the dataset.
            manifest_hash: Optional hash of the dataset manifest.
            parquet_index_path: Optional path to the parquet index file.
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO datasets (
                    dataset_id, version, artifact_count, total_bytes,
                    created_utc, description, manifest_hash,
                    parquet_index_path, indexed_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    version,
                    artifact_count,
                    total_bytes,
                    created_utc,
                    description,
                    manifest_hash,
                    parquet_index_path,
                    self._now_utc_iso(),
                ),
            )

    def index_dataset_snapshot(self, snapshot: Any) -> None:
        """Index a DatasetSnapshot in the registry.

        This is a convenience method that extracts fields from a DatasetSnapshot
        and calls index_dataset.

        Args:
            snapshot: A DatasetSnapshot instance.
        """
        self.index_dataset(
            dataset_id=snapshot.dataset_id,
            version=snapshot.version,
            artifact_count=snapshot.member_count,
            total_bytes=snapshot.total_bytes,
            created_utc=snapshot.created_utc,
            description=snapshot.description,
            manifest_hash=snapshot.content_hash.digest,
            parquet_index_path=snapshot.index_path,
        )

    def index_run(
        self,
        run_id: str,
        started_utc: str,
        status: Literal["pending", "in_progress", "completed", "failed"] = "pending",
        stage_name: str | None = None,
        ended_utc: str | None = None,
        hostname: str | None = None,
        generator: str | None = None,
        generator_version: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Index or update a run in the registry.

        Args:
            run_id: Unique identifier for the run.
            started_utc: ISO 8601 start timestamp.
            status: Run status.
            stage_name: Optional pipeline stage name.
            ended_utc: Optional ISO 8601 end timestamp.
            hostname: Optional hostname where the run executed.
            generator: Optional generator name.
            generator_version: Optional generator version.
            config: Optional configuration dictionary.
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, stage_name, started_utc, ended_utc, status,
                    hostname, generator, generator_version, artifact_count,
                    config, indexed_utc
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT artifact_count FROM runs WHERE run_id = ?), 0),
                    ?, ?
                )
                """,
                (
                    run_id,
                    stage_name,
                    started_utc,
                    ended_utc,
                    status,
                    hostname,
                    generator,
                    generator_version,
                    run_id,  # For the subquery
                    json.dumps(config) if config else None,
                    self._now_utc_iso(),
                ),
            )

    def update_run_status(
        self,
        run_id: str,
        status: Literal["pending", "in_progress", "completed", "failed"],
        ended_utc: str | None = None,
    ) -> None:
        """Update the status of a run.

        Args:
            run_id: The run ID to update.
            status: New status.
            ended_utc: Optional end timestamp.
        """
        with self._transaction() as cursor:
            if ended_utc:
                cursor.execute(
                    "UPDATE runs SET status = ?, ended_utc = ?, indexed_utc = ? WHERE run_id = ?;",
                    (status, ended_utc, self._now_utc_iso(), run_id),
                )
            else:
                cursor.execute(
                    "UPDATE runs SET status = ?, indexed_utc = ? WHERE run_id = ?;",
                    (status, self._now_utc_iso(), run_id),
                )

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        """Get an artifact record by ID.

        Args:
            artifact_id: The artifact ID to look up.

        Returns:
            The artifact record.

        Raises:
            ArtifactNotIndexedError: If the artifact is not in the index.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?;",
            (artifact_id,),
        )
        row = cursor.fetchone()
        cursor.close()

        if row is None:
            raise ArtifactNotIndexedError(f"Artifact not indexed: {artifact_id}")

        return ArtifactRecord.from_row(row)

    def get_dataset(self, dataset_id: str) -> DatasetRecord:
        """Get a dataset record by ID.

        Args:
            dataset_id: The dataset ID to look up.

        Returns:
            The dataset record.

        Raises:
            ArtifactNotIndexedError: If the dataset is not in the index.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM datasets WHERE dataset_id = ?;",
            (dataset_id,),
        )
        row = cursor.fetchone()
        cursor.close()

        if row is None:
            raise ArtifactNotIndexedError(f"Dataset not indexed: {dataset_id}")

        return DatasetRecord.from_row(row)

    def get_run(self, run_id: str) -> RunRecord:
        """Get a run record by ID.

        Args:
            run_id: The run ID to look up.

        Returns:
            The run record.

        Raises:
            ArtifactNotIndexedError: If the run is not in the index.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?;",
            (run_id,),
        )
        row = cursor.fetchone()
        cursor.close()

        if row is None:
            raise ArtifactNotIndexedError(f"Run not indexed: {run_id}")

        return RunRecord.from_row(row)

    def query_artifacts(
        self,
        artifact_type: str | None = None,
        run_id: str | None = None,
        dataset_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        roles: list[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
        order_by: Literal["created_utc", "artifact_id", "byte_size"] = "created_utc",
        order_desc: bool = True,
    ) -> list[ArtifactRecord]:
        """Query artifacts with optional filters.

        Args:
            artifact_type: Filter by artifact type.
            run_id: Filter by run ID.
            dataset_id: Filter by dataset ID.
            created_after: Filter to artifacts created after this timestamp.
            created_before: Filter to artifacts created before this timestamp.
            roles: Filter to artifacts having any of these roles.
            limit: Maximum number of results.
            offset: Number of results to skip.
            order_by: Field to order by.
            order_desc: If True, order descending; otherwise ascending.

        Returns:
            List of matching artifact records.
        """
        conditions = []
        params: list[Any] = []

        if artifact_type is not None:
            conditions.append("artifact_type = ?")
            params.append(artifact_type)

        if run_id is not None:
            conditions.append("run_id = ?")
            params.append(run_id)

        if dataset_id is not None:
            conditions.append("dataset_id = ?")
            params.append(dataset_id)

        if created_after is not None:
            conditions.append("created_utc >= ?")
            params.append(created_after)

        if created_before is not None:
            conditions.append("created_utc <= ?")
            params.append(created_before)

        if roles:
            # Check if any of the roles is in the JSON array
            role_conditions = []
            for role in roles:
                role_conditions.append("roles LIKE ?")
                params.append(f'%"{role}"%')
            conditions.append(f"({' OR '.join(role_conditions)})")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        order_direction = "DESC" if order_desc else "ASC"

        query = f"""
            SELECT * FROM artifacts
            WHERE {where_clause}
            ORDER BY {order_by} {order_direction}
        """

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        conn = self._get_connection()
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()

        return [ArtifactRecord.from_row(row) for row in rows]

    def query_datasets(
        self,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        order_desc: bool = True,
    ) -> list[DatasetRecord]:
        """Query datasets with optional filters.

        Args:
            created_after: Filter to datasets created after this timestamp.
            created_before: Filter to datasets created before this timestamp.
            limit: Maximum number of results.
            offset: Number of results to skip.
            order_desc: If True, order by created_utc descending.

        Returns:
            List of matching dataset records.
        """
        conditions = []
        params: list[Any] = []

        if created_after is not None:
            conditions.append("created_utc >= ?")
            params.append(created_after)

        if created_before is not None:
            conditions.append("created_utc <= ?")
            params.append(created_before)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        order_direction = "DESC" if order_desc else "ASC"

        query = f"""
            SELECT * FROM datasets
            WHERE {where_clause}
            ORDER BY created_utc {order_direction}
        """

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        conn = self._get_connection()
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()

        return [DatasetRecord.from_row(row) for row in rows]

    def query_runs(
        self,
        status: str | None = None,
        started_after: str | None = None,
        started_before: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        order_desc: bool = True,
    ) -> list[RunRecord]:
        """Query runs with optional filters.

        Args:
            status: Filter by run status.
            started_after: Filter to runs started after this timestamp.
            started_before: Filter to runs started before this timestamp.
            limit: Maximum number of results.
            offset: Number of results to skip.
            order_desc: If True, order by started_utc descending.

        Returns:
            List of matching run records.
        """
        conditions = []
        params: list[Any] = []

        if status is not None:
            conditions.append("status = ?")
            params.append(status)

        if started_after is not None:
            conditions.append("started_utc >= ?")
            params.append(started_after)

        if started_before is not None:
            conditions.append("started_utc <= ?")
            params.append(started_before)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        order_direction = "DESC" if order_desc else "ASC"

        query = f"""
            SELECT * FROM runs
            WHERE {where_clause}
            ORDER BY started_utc {order_direction}
        """

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        conn = self._get_connection()
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()

        return [RunRecord.from_row(row) for row in rows]

    def count_artifacts(
        self,
        artifact_type: str | None = None,
        run_id: str | None = None,
        dataset_id: str | None = None,
    ) -> int:
        """Count artifacts matching filters.

        Args:
            artifact_type: Filter by artifact type.
            run_id: Filter by run ID.
            dataset_id: Filter by dataset ID.

        Returns:
            Count of matching artifacts.
        """
        conditions = []
        params: list[Any] = []

        if artifact_type is not None:
            conditions.append("artifact_type = ?")
            params.append(artifact_type)

        if run_id is not None:
            conditions.append("run_id = ?")
            params.append(run_id)

        if dataset_id is not None:
            conditions.append("dataset_id = ?")
            params.append(dataset_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        conn = self._get_connection()
        cursor = conn.execute(f"SELECT COUNT(*) FROM artifacts WHERE {where_clause}", params)
        count = cursor.fetchone()[0]
        cursor.close()

        return count

    def count_datasets(self) -> int:
        """Count all datasets in the registry."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM datasets")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def count_runs(self, status: str | None = None) -> int:
        """Count runs, optionally filtered by status."""
        conn = self._get_connection()
        if status:
            cursor = conn.execute("SELECT COUNT(*) FROM runs WHERE status = ?", (status,))
        else:
            cursor = conn.execute("SELECT COUNT(*) FROM runs")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def delete_artifact(self, artifact_id: str) -> bool:
        """Remove an artifact from the index.

        Args:
            artifact_id: The artifact ID to remove.

        Returns:
            True if the artifact was removed, False if it wasn't in the index.
        """
        with self._transaction() as cursor:
            cursor.execute("DELETE FROM artifacts WHERE artifact_id = ?;", (artifact_id,))
            return cursor.rowcount > 0

    def delete_dataset(self, dataset_id: str) -> bool:
        """Remove a dataset from the index.

        Args:
            dataset_id: The dataset ID to remove.

        Returns:
            True if the dataset was removed, False if it wasn't in the index.
        """
        with self._transaction() as cursor:
            cursor.execute("DELETE FROM datasets WHERE dataset_id = ?;", (dataset_id,))
            return cursor.rowcount > 0

    def delete_run(self, run_id: str) -> bool:
        """Remove a run from the index.

        Args:
            run_id: The run ID to remove.

        Returns:
            True if the run was removed, False if it wasn't in the index.
        """
        with self._transaction() as cursor:
            cursor.execute("DELETE FROM runs WHERE run_id = ?;", (run_id,))
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Clear all data from the registry.

        This removes all artifacts, datasets, and runs from the index.
        Use with caution.
        """
        with self._transaction() as cursor:
            cursor.execute("DELETE FROM artifacts;")
            cursor.execute("DELETE FROM datasets;")
            cursor.execute("DELETE FROM runs;")

    def rebuild_from_store(
        self,
        store: ArtifactStore,
        clear_first: bool = True,
    ) -> int:
        """Rebuild the registry index from an artifact store.

        This is the primary way to ensure the registry is in sync with
        the source manifests. The registry is derived data and can be
        safely rebuilt at any time.

        Args:
            store: The artifact store to index from.
            clear_first: If True, clear the registry before rebuilding.

        Returns:
            Number of artifacts indexed.
        """
        if clear_first:
            self.clear()

        count = 0
        manifest_ids = store.list_manifests()

        for artifact_id in manifest_ids:
            try:
                manifest = store.get_manifest(artifact_id)
                manifest_path = str(store._manifest_path(artifact_id))
                self.index_artifact(manifest, manifest_path=manifest_path)
                count += 1
            except Exception:
                # Skip manifests that can't be loaded
                continue

        return count

    def get_artifacts_by_hash(self, content_hash_digest: str) -> list[ArtifactRecord]:
        """Find all artifacts with a given content hash.

        This is useful for finding duplicate content or verifying
        deduplication is working.

        Args:
            content_hash_digest: The SHA256 digest to search for.

        Returns:
            List of artifact records with matching content hash.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM artifacts WHERE content_hash_digest = ?;",
            (content_hash_digest,),
        )
        rows = cursor.fetchall()
        cursor.close()

        return [ArtifactRecord.from_row(row) for row in rows]

    def get_artifacts_for_run(self, run_id: str) -> list[ArtifactRecord]:
        """Get all artifacts produced by a specific run.

        Args:
            run_id: The run ID to query.

        Returns:
            List of artifact records for the run.
        """
        return self.query_artifacts(run_id=run_id, order_desc=False)

    def get_artifacts_for_dataset(self, dataset_id: str) -> list[ArtifactRecord]:
        """Get all artifacts belonging to a specific dataset.

        Args:
            dataset_id: The dataset ID to query.

        Returns:
            List of artifact records for the dataset.
        """
        return self.query_artifacts(dataset_id=dataset_id, order_desc=False)

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics from the registry.

        Returns:
            Dictionary with storage statistics including:
            - total_artifacts: Total number of artifacts
            - total_bytes: Total size of all artifacts
            - artifacts_by_type: Count per artifact type
            - unique_hashes: Number of unique content hashes
        """
        conn = self._get_connection()

        # Total artifacts and bytes
        cursor = conn.execute("SELECT COUNT(*), COALESCE(SUM(byte_size), 0) FROM artifacts;")
        row = cursor.fetchone()
        total_artifacts = row[0]
        total_bytes = row[1]
        cursor.close()

        # Artifacts by type
        cursor = conn.execute("SELECT artifact_type, COUNT(*) FROM artifacts GROUP BY artifact_type;")
        artifacts_by_type = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()

        # Unique content hashes
        cursor = conn.execute("SELECT COUNT(DISTINCT content_hash_digest) FROM artifacts;")
        unique_hashes = cursor.fetchone()[0]
        cursor.close()

        return {
            "total_artifacts": total_artifacts,
            "total_bytes": total_bytes,
            "artifacts_by_type": artifacts_by_type,
            "unique_hashes": unique_hashes,
            "deduplication_ratio": (total_artifacts / unique_hashes if unique_hashes > 0 else 1.0),
        }
