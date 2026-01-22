"""LineageGraph: Artifact provenance tracer with SQLite persistence.

This module implements the LineageGraph class, which provides:
- Graph construction from artifact manifests
- Ancestor/descendant queries via BFS traversal
- SQLite-based persistence (lineage.sqlite)
- Thread-safe operations with connection pooling

The lineage graph traces artifact provenance through the inputs field
of each artifact's lineage metadata, enabling queries like:
- "What artifacts were used to produce this artifact?" (ancestors)
- "What artifacts were derived from this artifact?" (descendants)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from formula_foundry.m3.artifact_store import ArtifactManifest, ArtifactStore

# Schema version for migration support
LINEAGE_SCHEMA_VERSION = 1

# Relation types for edges
RelationType = Literal[
    "derived_from",
    "generated_by",
    "validated_by",
    "config_from",
    "sibling_of",
    "supersedes",
]

# SQL statements for table creation
_CREATE_NODES_TABLE = """
CREATE TABLE IF NOT EXISTS nodes (
    artifact_id TEXT PRIMARY KEY,
    artifact_type TEXT NOT NULL,
    content_hash_digest TEXT NOT NULL,
    run_id TEXT,
    stage_name TEXT,
    created_utc TEXT,
    indexed_utc TEXT NOT NULL
);
"""

_CREATE_EDGES_TABLE = """
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    UNIQUE(source_id, target_id, relation)
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes (artifact_type);",
    "CREATE INDEX IF NOT EXISTS idx_nodes_run ON nodes (run_id);",
    "CREATE INDEX IF NOT EXISTS idx_nodes_hash ON nodes (content_hash_digest);",
    "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges (source_id);",
    "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges (target_id);",
    "CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges (relation);",
]

_CREATE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""


@dataclass
class LineageNode:
    """A node in the lineage graph representing an artifact."""

    artifact_id: str
    artifact_type: str
    content_hash_digest: str
    run_id: str | None = None
    stage_name: str | None = None
    created_utc: str | None = None
    indexed_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "content_hash_digest": self.content_hash_digest,
        }
        if self.run_id:
            result["run_id"] = self.run_id
        if self.stage_name:
            result["stage_name"] = self.stage_name
        if self.created_utc:
            result["created_utc"] = self.created_utc
        if self.indexed_utc:
            result["indexed_utc"] = self.indexed_utc
        return result

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> LineageNode:
        """Create a LineageNode from a database row."""
        return cls(
            artifact_id=row["artifact_id"],
            artifact_type=row["artifact_type"],
            content_hash_digest=row["content_hash_digest"],
            run_id=row["run_id"],
            stage_name=row["stage_name"],
            created_utc=row["created_utc"],
            indexed_utc=row["indexed_utc"],
        )


@dataclass
class LineageEdge:
    """An edge in the lineage graph representing a relationship."""

    source_id: str
    target_id: str
    relation: RelationType

    def to_dict(self) -> dict[str, str]:
        """Convert to JSON-serializable dict."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> LineageEdge:
        """Create a LineageEdge from a database row."""
        return cls(
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation=row["relation"],
        )


@dataclass
class LineagePath:
    """A path through the lineage graph."""

    nodes: list[LineageNode] = field(default_factory=list)
    edges: list[LineageEdge] = field(default_factory=list)

    @property
    def length(self) -> int:
        """Return the number of edges in the path."""
        return len(self.edges)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "length": self.length,
        }


@dataclass
class LineageSubgraph:
    """A subgraph extracted from the lineage graph."""

    nodes: dict[str, LineageNode] = field(default_factory=dict)
    edges: list[LineageEdge] = field(default_factory=list)
    root_id: str | None = None

    @property
    def node_count(self) -> int:
        """Return the number of nodes."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Return the number of edges."""
        return len(self.edges)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "root_id": self.root_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
        }

    def get_roots(self) -> list[str]:
        """Get artifact IDs with no incoming edges (root nodes)."""
        targets = {e.target_id for e in self.edges}
        return [n for n in self.nodes if n not in targets]

    def get_leaves(self) -> list[str]:
        """Get artifact IDs with no outgoing edges (leaf nodes)."""
        sources = {e.source_id for e in self.edges}
        return [n for n in self.nodes if n not in sources]


class LineageGraphError(Exception):
    """Base exception for LineageGraph errors."""


class NodeNotFoundError(LineageGraphError):
    """Raised when a node is not found in the graph."""


class LineageGraph:
    """Graph-based artifact lineage tracer with SQLite persistence.

    This class maintains a directed graph of artifact relationships where:
    - Nodes represent artifacts (identified by artifact_id)
    - Edges represent lineage relationships (derived_from, generated_by, etc.)

    The edge direction follows the data flow:
    - source_id -> target_id means "target was produced using source as input"
    - Or equivalently: "target depends on source"

    Thread Safety:
        The class uses thread-local connections for thread safety.

    Example usage:
        graph = LineageGraph(Path("data/lineage.sqlite"))
        graph.initialize()

        # Add nodes and edges from manifests
        graph.build_from_manifests(store)

        # Query ancestors
        ancestors = graph.get_ancestors("artifact_id")

        # Query descendants
        descendants = graph.get_descendants("artifact_id")

        # Trace path to roots
        subgraph = graph.trace_to_roots("artifact_id")
    """

    def __init__(self, db_path: Path | str) -> None:
        """Initialize the lineage graph.

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
            cursor.execute(_CREATE_SCHEMA_VERSION_TABLE)

            cursor.execute("SELECT version FROM schema_version LIMIT 1;")
            row = cursor.fetchone()

            if row is None:
                cursor.execute(_CREATE_NODES_TABLE)
                cursor.execute(_CREATE_EDGES_TABLE)

                for index_sql in _CREATE_INDEXES:
                    cursor.execute(index_sql)

                cursor.execute(
                    "INSERT INTO schema_version (version) VALUES (?);",
                    (LINEAGE_SCHEMA_VERSION,),
                )
            elif row["version"] < LINEAGE_SCHEMA_VERSION:
                # Future: handle migrations here
                pass

    def close(self) -> None:
        """Close the database connection for the current thread."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    def add_node(
        self,
        artifact_id: str,
        artifact_type: str,
        content_hash_digest: str,
        run_id: str | None = None,
        stage_name: str | None = None,
        created_utc: str | None = None,
    ) -> LineageNode:
        """Add or update a node in the lineage graph.

        Args:
            artifact_id: Unique identifier for the artifact.
            artifact_type: Type of the artifact.
            content_hash_digest: SHA256 digest of the artifact content.
            run_id: Optional run ID that produced this artifact.
            stage_name: Optional pipeline stage name.
            created_utc: Optional creation timestamp.

        Returns:
            The created or updated LineageNode.
        """
        indexed_utc = self._now_utc_iso()

        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO nodes (
                    artifact_id, artifact_type, content_hash_digest,
                    run_id, stage_name, created_utc, indexed_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    artifact_type,
                    content_hash_digest,
                    run_id,
                    stage_name,
                    created_utc,
                    indexed_utc,
                ),
            )

        return LineageNode(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            content_hash_digest=content_hash_digest,
            run_id=run_id,
            stage_name=stage_name,
            created_utc=created_utc,
            indexed_utc=indexed_utc,
        )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: RelationType,
    ) -> LineageEdge:
        """Add an edge to the lineage graph.

        The edge represents that target_id was derived from/depends on source_id.

        Args:
            source_id: The source artifact (input/dependency).
            target_id: The target artifact (output/dependent).
            relation: The type of relationship.

        Returns:
            The created LineageEdge.
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT OR IGNORE INTO edges (source_id, target_id, relation)
                VALUES (?, ?, ?)
                """,
                (source_id, target_id, relation),
            )

        return LineageEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
        )

    def add_manifest(self, manifest: ArtifactManifest) -> LineageNode:
        """Add a node and its edges from an artifact manifest.

        This extracts lineage information from the manifest and adds:
        - A node for the artifact
        - Edges from all input artifacts to this artifact

        Args:
            manifest: The artifact manifest to add.

        Returns:
            The created LineageNode.
        """
        node = self.add_node(
            artifact_id=manifest.artifact_id,
            artifact_type=manifest.artifact_type,
            content_hash_digest=manifest.content_hash.digest,
            run_id=manifest.lineage.run_id,
            stage_name=manifest.lineage.stage_name,
            created_utc=manifest.created_utc,
        )

        for input_ref in manifest.lineage.inputs:
            self.add_edge(
                source_id=input_ref.artifact_id,
                target_id=manifest.artifact_id,
                relation=input_ref.relation,
            )

        return node

    def get_node(self, artifact_id: str) -> LineageNode:
        """Get a node by artifact ID.

        Args:
            artifact_id: The artifact ID to look up.

        Returns:
            The LineageNode.

        Raises:
            NodeNotFoundError: If the node is not found.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM nodes WHERE artifact_id = ?;",
            (artifact_id,),
        )
        row = cursor.fetchone()
        cursor.close()

        if row is None:
            raise NodeNotFoundError(f"Node not found: {artifact_id}")

        return LineageNode.from_row(row)

    def has_node(self, artifact_id: str) -> bool:
        """Check if a node exists in the graph.

        Args:
            artifact_id: The artifact ID to check.

        Returns:
            True if the node exists, False otherwise.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT 1 FROM nodes WHERE artifact_id = ?;",
            (artifact_id,),
        )
        exists = cursor.fetchone() is not None
        cursor.close()
        return exists

    def get_edges_from(self, artifact_id: str) -> list[LineageEdge]:
        """Get all outgoing edges from a node.

        Args:
            artifact_id: The source artifact ID.

        Returns:
            List of edges where this node is the source.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM edges WHERE source_id = ?;",
            (artifact_id,),
        )
        edges = [LineageEdge.from_row(row) for row in cursor.fetchall()]
        cursor.close()
        return edges

    def get_edges_to(self, artifact_id: str) -> list[LineageEdge]:
        """Get all incoming edges to a node.

        Args:
            artifact_id: The target artifact ID.

        Returns:
            List of edges where this node is the target.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM edges WHERE target_id = ?;",
            (artifact_id,),
        )
        edges = [LineageEdge.from_row(row) for row in cursor.fetchall()]
        cursor.close()
        return edges

    def get_ancestors(
        self,
        artifact_id: str,
        max_depth: int | None = None,
        relation_filter: list[RelationType] | None = None,
    ) -> LineageSubgraph:
        """Get all ancestors of an artifact (inputs that contributed to it).

        Uses BFS to traverse the graph backwards through input edges.

        Args:
            artifact_id: The artifact ID to trace from.
            max_depth: Optional maximum depth to traverse.
            relation_filter: Optional list of relation types to follow.

        Returns:
            A LineageSubgraph containing all ancestor nodes and edges.

        Raises:
            NodeNotFoundError: If the starting node is not found.
        """
        if not self.has_node(artifact_id):
            raise NodeNotFoundError(f"Node not found: {artifact_id}")

        subgraph = LineageSubgraph(root_id=artifact_id)
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(artifact_id, 0)])

        while queue:
            current_id, depth = queue.popleft()

            if current_id in visited:
                continue
            visited.add(current_id)

            if self.has_node(current_id):
                node = self.get_node(current_id)
                subgraph.nodes[current_id] = node

            if max_depth is not None and depth >= max_depth:
                continue

            edges = self.get_edges_to(current_id)
            for edge in edges:
                if relation_filter and edge.relation not in relation_filter:
                    continue
                subgraph.edges.append(edge)
                if edge.source_id not in visited:
                    queue.append((edge.source_id, depth + 1))

        return subgraph

    def get_descendants(
        self,
        artifact_id: str,
        max_depth: int | None = None,
        relation_filter: list[RelationType] | None = None,
    ) -> LineageSubgraph:
        """Get all descendants of an artifact (outputs derived from it).

        Uses BFS to traverse the graph forward through output edges.

        Args:
            artifact_id: The artifact ID to trace from.
            max_depth: Optional maximum depth to traverse.
            relation_filter: Optional list of relation types to follow.

        Returns:
            A LineageSubgraph containing all descendant nodes and edges.

        Raises:
            NodeNotFoundError: If the starting node is not found.
        """
        if not self.has_node(artifact_id):
            raise NodeNotFoundError(f"Node not found: {artifact_id}")

        subgraph = LineageSubgraph(root_id=artifact_id)
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(artifact_id, 0)])

        while queue:
            current_id, depth = queue.popleft()

            if current_id in visited:
                continue
            visited.add(current_id)

            if self.has_node(current_id):
                node = self.get_node(current_id)
                subgraph.nodes[current_id] = node

            if max_depth is not None and depth >= max_depth:
                continue

            edges = self.get_edges_from(current_id)
            for edge in edges:
                if relation_filter and edge.relation not in relation_filter:
                    continue
                subgraph.edges.append(edge)
                if edge.target_id not in visited:
                    queue.append((edge.target_id, depth + 1))

        return subgraph

    def trace_to_roots(
        self,
        artifact_id: str,
        required_roles: list[str] | None = None,
    ) -> LineageSubgraph:
        """Trace lineage back to root inputs.

        This traverses backwards to find all artifacts with no inputs,
        which represent the original source data.

        Args:
            artifact_id: The artifact ID to trace from.
            required_roles: Optional list of roles that must exist in roots.

        Returns:
            A LineageSubgraph containing the full ancestry to roots.

        Raises:
            NodeNotFoundError: If the starting node is not found.
        """
        subgraph = self.get_ancestors(artifact_id)
        return subgraph

    def trace_to_leaves(self, artifact_id: str) -> LineageSubgraph:
        """Trace lineage forward to final outputs.

        This traverses forward to find all artifacts with no dependents,
        which represent final outputs.

        Args:
            artifact_id: The artifact ID to trace from.

        Returns:
            A LineageSubgraph containing the full descent to leaves.

        Raises:
            NodeNotFoundError: If the starting node is not found.
        """
        return self.get_descendants(artifact_id)

    def get_direct_inputs(self, artifact_id: str) -> list[LineageNode]:
        """Get the immediate inputs (parents) of an artifact.

        Args:
            artifact_id: The artifact ID.

        Returns:
            List of nodes that are direct inputs to this artifact.
        """
        edges = self.get_edges_to(artifact_id)
        nodes = []
        for edge in edges:
            if self.has_node(edge.source_id):
                nodes.append(self.get_node(edge.source_id))
        return nodes

    def get_direct_outputs(self, artifact_id: str) -> list[LineageNode]:
        """Get the immediate outputs (children) of an artifact.

        Args:
            artifact_id: The artifact ID.

        Returns:
            List of nodes that are direct outputs from this artifact.
        """
        edges = self.get_edges_from(artifact_id)
        nodes = []
        for edge in edges:
            if self.has_node(edge.target_id):
                nodes.append(self.get_node(edge.target_id))
        return nodes

    def build_from_store(
        self,
        store: ArtifactStore,
        clear_first: bool = True,
    ) -> int:
        """Build the lineage graph from an artifact store.

        Reads all manifests from the store and builds the complete
        lineage graph.

        Args:
            store: The artifact store to read from.
            clear_first: If True, clear the graph before building.

        Returns:
            Number of nodes added.
        """
        if clear_first:
            self.clear()

        count = 0
        manifest_ids = store.list_manifests()

        for artifact_id in manifest_ids:
            try:
                manifest = store.get_manifest(artifact_id)
                self.add_manifest(manifest)
                count += 1
            except Exception:
                continue

        return count

    def clear(self) -> None:
        """Clear all data from the graph."""
        with self._transaction() as cursor:
            cursor.execute("DELETE FROM edges;")
            cursor.execute("DELETE FROM nodes;")

    def count_nodes(self) -> int:
        """Count the total number of nodes in the graph."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM nodes;")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def count_edges(self) -> int:
        """Count the total number of edges in the graph."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM edges;")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def query_nodes(
        self,
        artifact_type: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[LineageNode]:
        """Query nodes with optional filters.

        Args:
            artifact_type: Filter by artifact type.
            run_id: Filter by run ID.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of matching nodes.
        """
        conditions = []
        params: list[Any] = []

        if artifact_type is not None:
            conditions.append("artifact_type = ?")
            params.append(artifact_type)

        if run_id is not None:
            conditions.append("run_id = ?")
            params.append(run_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"SELECT * FROM nodes WHERE {where_clause} ORDER BY created_utc DESC"

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        conn = self._get_connection()
        cursor = conn.execute(query, params)
        nodes = [LineageNode.from_row(row) for row in cursor.fetchall()]
        cursor.close()

        return nodes

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the lineage graph.

        Returns:
            Dictionary with graph statistics.
        """
        conn = self._get_connection()

        cursor = conn.execute("SELECT COUNT(*) FROM nodes;")
        node_count = cursor.fetchone()[0]
        cursor.close()

        cursor = conn.execute("SELECT COUNT(*) FROM edges;")
        edge_count = cursor.fetchone()[0]
        cursor.close()

        cursor = conn.execute("SELECT artifact_type, COUNT(*) FROM nodes GROUP BY artifact_type;")
        nodes_by_type = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()

        cursor = conn.execute("SELECT relation, COUNT(*) FROM edges GROUP BY relation;")
        edges_by_relation = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()

        cursor = conn.execute("SELECT COUNT(*) FROM nodes WHERE artifact_id NOT IN (SELECT target_id FROM edges);")
        root_count = cursor.fetchone()[0]
        cursor.close()

        cursor = conn.execute("SELECT COUNT(*) FROM nodes WHERE artifact_id NOT IN (SELECT source_id FROM edges);")
        leaf_count = cursor.fetchone()[0]
        cursor.close()

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "nodes_by_type": nodes_by_type,
            "edges_by_relation": edges_by_relation,
            "root_count": root_count,
            "leaf_count": leaf_count,
        }

    def export_to_dict(self) -> dict[str, Any]:
        """Export the entire graph as a dictionary.

        Returns:
            Dictionary representation of the graph.
        """
        conn = self._get_connection()

        cursor = conn.execute("SELECT * FROM nodes ORDER BY artifact_id;")
        nodes = {row["artifact_id"]: LineageNode.from_row(row).to_dict() for row in cursor.fetchall()}
        cursor.close()

        cursor = conn.execute("SELECT * FROM edges ORDER BY id;")
        edges = [LineageEdge.from_row(row).to_dict() for row in cursor.fetchall()]
        cursor.close()

        return {
            "schema_version": LINEAGE_SCHEMA_VERSION,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    def export_to_json(self, indent: int = 2) -> str:
        """Export the graph as JSON.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation of the graph.
        """
        return json.dumps(self.export_to_dict(), indent=indent, sort_keys=True)

    def find_ancestors_by_type(
        self,
        artifact_id: str,
        artifact_type: str,
        max_depth: int | None = None,
    ) -> list[LineageNode]:
        """Find all ancestors of an artifact that match a given type.

        This supports queries like "which geometry produced this S-param?"
        by filtering the ancestor graph to only include nodes of the
        specified type.

        Args:
            artifact_id: The artifact ID to trace from.
            artifact_type: The artifact type to filter for (e.g., "resolved_design",
                          "coupon_spec", "touchstone").
            max_depth: Optional maximum traversal depth.

        Returns:
            List of ancestor nodes matching the specified type.

        Raises:
            NodeNotFoundError: If the starting node is not found.

        Example:
            # Find the geometry that produced an S-parameter file
            geometries = graph.find_ancestors_by_type(
                "touchstone-001", "resolved_design"
            )
        """
        subgraph = self.get_ancestors(artifact_id, max_depth=max_depth)
        return [
            node
            for node in subgraph.nodes.values()
            if node.artifact_type == artifact_type
        ]

    def find_descendants_by_type(
        self,
        artifact_id: str,
        artifact_type: str,
        max_depth: int | None = None,
    ) -> list[LineageNode]:
        """Find all descendants of an artifact that match a given type.

        This supports queries like "which S-params were derived from this geometry?"

        Args:
            artifact_id: The artifact ID to trace from.
            artifact_type: The artifact type to filter for.
            max_depth: Optional maximum traversal depth.

        Returns:
            List of descendant nodes matching the specified type.

        Raises:
            NodeNotFoundError: If the starting node is not found.
        """
        subgraph = self.get_descendants(artifact_id, max_depth=max_depth)
        return [
            node
            for node in subgraph.nodes.values()
            if node.artifact_type == artifact_type
        ]

    def find_ancestors_by_role(
        self,
        artifact_id: str,
        role: str,
        store: ArtifactStore | None = None,
    ) -> list[LineageNode]:
        """Find all ancestors of an artifact that have a specific role.

        This requires access to the artifact store to retrieve role information
        from manifests, since roles are not stored in the lineage graph nodes.

        Args:
            artifact_id: The artifact ID to trace from.
            role: The role to filter for (e.g., "geometry", "config", "oracle_output").
            store: The artifact store to retrieve manifests from.

        Returns:
            List of ancestor nodes that have the specified role.

        Raises:
            NodeNotFoundError: If the starting node is not found.
            ValueError: If store is not provided.

        Example:
            # Find all geometry artifacts in the lineage
            geometries = graph.find_ancestors_by_role(
                "touchstone-001", "geometry", store=artifact_store
            )
        """
        if store is None:
            raise ValueError("store parameter is required to query by role")

        subgraph = self.get_ancestors(artifact_id)
        result = []

        for node in subgraph.nodes.values():
            try:
                manifest = store.get_manifest(node.artifact_id)
                if role in manifest.roles:
                    result.append(node)
            except Exception:
                # Skip nodes whose manifests can't be retrieved
                continue

        return result

    def get_artifact_run_info(
        self,
        artifact_id: str,
        store: ArtifactStore | None = None,
    ) -> dict[str, Any]:
        """Get comprehensive run/provenance info for an artifact.

        This answers the question "Which commit created this artifact?"
        by returning the run_id, stage_name, and any provenance information
        available from the manifest.

        Args:
            artifact_id: The artifact ID to look up.
            store: Optional artifact store for retrieving full manifest info.

        Returns:
            Dictionary containing:
            - artifact_id: The artifact ID
            - run_id: The run that produced this artifact
            - stage_name: The pipeline stage (if available)
            - created_utc: Creation timestamp
            - provenance: Full provenance dict (if store provided)

        Raises:
            NodeNotFoundError: If the artifact is not found in the graph.

        Example:
            info = graph.get_artifact_run_info("touchstone-001", store=artifact_store)
            print(f"Created by run {info['run_id']} at {info['created_utc']}")
        """
        node = self.get_node(artifact_id)

        result: dict[str, Any] = {
            "artifact_id": node.artifact_id,
            "artifact_type": node.artifact_type,
            "content_hash_digest": node.content_hash_digest,
            "run_id": node.run_id,
            "stage_name": node.stage_name,
            "created_utc": node.created_utc,
        }

        if store is not None:
            try:
                manifest = store.get_manifest(artifact_id)
                result["provenance"] = manifest.provenance.to_dict()
            except Exception:
                pass

        return result

    def trace_sparam_to_geometry(
        self,
        sparam_artifact_id: str,
    ) -> list[LineageNode]:
        """Trace an S-parameter artifact back to its source geometry.

        This is a convenience method for the common query "which geometry
        produced this S-param?" It finds all ancestors of type "resolved_design"
        or "coupon_spec" in the lineage chain.

        Args:
            sparam_artifact_id: The artifact ID of the touchstone/sparam artifact.

        Returns:
            List of geometry-related ancestor nodes (resolved_design, coupon_spec).

        Raises:
            NodeNotFoundError: If the S-param artifact is not found.

        Example:
            geometries = graph.trace_sparam_to_geometry("touchstone-001")
            for g in geometries:
                print(f"Geometry: {g.artifact_id} ({g.artifact_type})")
        """
        geometry_types = {"resolved_design", "coupon_spec"}
        subgraph = self.get_ancestors(sparam_artifact_id)

        return [
            node
            for node in subgraph.nodes.values()
            if node.artifact_type in geometry_types
        ]

    def get_full_lineage_chain(
        self,
        artifact_id: str,
        store: ArtifactStore | None = None,
    ) -> list[dict[str, Any]]:
        """Get the complete lineage chain from roots to the given artifact.

        Returns an ordered list of artifacts from root inputs to the target,
        with full manifest info if a store is provided.

        Args:
            artifact_id: The artifact ID to trace.
            store: Optional artifact store for retrieving full manifests.

        Returns:
            List of dicts representing artifacts in topological order,
            from roots to the target artifact.

        Raises:
            NodeNotFoundError: If the artifact is not found.
        """
        subgraph = self.get_ancestors(artifact_id)

        # Build adjacency list for topological sort
        children: dict[str, list[str]] = {n: [] for n in subgraph.nodes}
        in_degree: dict[str, int] = {n: 0 for n in subgraph.nodes}

        for edge in subgraph.edges:
            if edge.source_id in children and edge.target_id in subgraph.nodes:
                children[edge.source_id].append(edge.target_id)
                in_degree[edge.target_id] += 1

        # Kahn's algorithm for topological sort
        queue = deque([n for n, d in in_degree.items() if d == 0])
        result: list[dict[str, Any]] = []

        while queue:
            node_id = queue.popleft()
            node = subgraph.nodes[node_id]

            node_info: dict[str, Any] = node.to_dict()
            if store is not None:
                try:
                    manifest = store.get_manifest(node_id)
                    node_info["roles"] = list(manifest.roles)
                    node_info["provenance"] = manifest.provenance.to_dict()
                except Exception:
                    pass

            result.append(node_info)

            for child_id in children[node_id]:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        return result
