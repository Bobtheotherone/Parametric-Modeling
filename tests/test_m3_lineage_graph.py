"""Tests for the M3 LineageGraph with SQLite-backed storage."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from formula_foundry.m3.artifact_store import ArtifactStore, LineageReference
from formula_foundry.m3.lineage_graph import (
    LineageEdge,
    LineageGraph,
    LineageNode,
    LineagePath,
    LineageSubgraph,
    NodeNotFoundError,
)


class TestLineageGraphInit:
    """Tests for lineage graph initialization."""

    def test_initialize_creates_database(self, tmp_path: Path) -> None:
        """Test that initialize creates the database file."""
        db_path = tmp_path / "lineage.sqlite"
        graph = LineageGraph(db_path)
        graph.initialize()

        assert db_path.exists()
        graph.close()

    def test_initialize_creates_tables(self, tmp_path: Path) -> None:
        """Test that initialize creates all required tables."""
        db_path = tmp_path / "lineage.sqlite"
        graph = LineageGraph(db_path)
        graph.initialize()

        conn = graph._get_connection()
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        cursor.close()

        assert "nodes" in tables
        assert "edges" in tables
        assert "schema_version" in tables

        graph.close()

    def test_initialize_idempotent(self, tmp_path: Path) -> None:
        """Test that initialize can be called multiple times safely."""
        db_path = tmp_path / "lineage.sqlite"
        graph = LineageGraph(db_path)

        graph.initialize()
        graph.initialize()
        graph.initialize()

        assert graph.count_nodes() == 0
        graph.close()


class TestNodeOperations:
    """Tests for node operations."""

    def test_add_node(self, tmp_path: Path) -> None:
        """Test adding a node to the graph."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        node = graph.add_node(
            artifact_id="art-001",
            artifact_type="coupon_spec",
            content_hash_digest="abc123",
            run_id="run-001",
            stage_name="generation",
            created_utc="2026-01-20T09:00:00Z",
        )

        assert node.artifact_id == "art-001"
        assert node.artifact_type == "coupon_spec"
        assert node.content_hash_digest == "abc123"
        assert node.run_id == "run-001"
        assert graph.count_nodes() == 1

        graph.close()

    def test_add_node_upsert(self, tmp_path: Path) -> None:
        """Test that adding an existing node updates it."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node(
            artifact_id="art-001",
            artifact_type="coupon_spec",
            content_hash_digest="abc123",
        )

        graph.add_node(
            artifact_id="art-001",
            artifact_type="kicad_board",
            content_hash_digest="xyz789",
        )

        assert graph.count_nodes() == 1
        node = graph.get_node("art-001")
        assert node.artifact_type == "kicad_board"
        assert node.content_hash_digest == "xyz789"

        graph.close()

    def test_get_node(self, tmp_path: Path) -> None:
        """Test retrieving a node by ID."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node(
            artifact_id="art-001",
            artifact_type="touchstone",
            content_hash_digest="hash123",
            run_id="run-001",
        )

        node = graph.get_node("art-001")
        assert node.artifact_id == "art-001"
        assert node.artifact_type == "touchstone"

        graph.close()

    def test_get_node_not_found(self, tmp_path: Path) -> None:
        """Test that get_node raises for missing node."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        with pytest.raises(NodeNotFoundError):
            graph.get_node("nonexistent")

        graph.close()

    def test_has_node(self, tmp_path: Path) -> None:
        """Test checking if a node exists."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node(
            artifact_id="art-001",
            artifact_type="other",
            content_hash_digest="hash",
        )

        assert graph.has_node("art-001")
        assert not graph.has_node("art-002")

        graph.close()


class TestEdgeOperations:
    """Tests for edge operations."""

    def test_add_edge(self, tmp_path: Path) -> None:
        """Test adding an edge to the graph."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("art-001", "coupon_spec", "hash1")
        graph.add_node("art-002", "kicad_board", "hash2")

        edge = graph.add_edge("art-001", "art-002", "derived_from")

        assert edge.source_id == "art-001"
        assert edge.target_id == "art-002"
        assert edge.relation == "derived_from"
        assert graph.count_edges() == 1

        graph.close()

    def test_add_edge_duplicate_ignored(self, tmp_path: Path) -> None:
        """Test that duplicate edges are ignored."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("art-001", "coupon_spec", "hash1")
        graph.add_node("art-002", "kicad_board", "hash2")

        graph.add_edge("art-001", "art-002", "derived_from")
        graph.add_edge("art-001", "art-002", "derived_from")

        assert graph.count_edges() == 1

        graph.close()

    def test_get_edges_from(self, tmp_path: Path) -> None:
        """Test getting outgoing edges from a node."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("art-001", "coupon_spec", "hash1")
        graph.add_node("art-002", "kicad_board", "hash2")
        graph.add_node("art-003", "gerber", "hash3")

        graph.add_edge("art-001", "art-002", "derived_from")
        graph.add_edge("art-001", "art-003", "config_from")

        edges = graph.get_edges_from("art-001")
        assert len(edges) == 2
        target_ids = {e.target_id for e in edges}
        assert target_ids == {"art-002", "art-003"}

        graph.close()

    def test_get_edges_to(self, tmp_path: Path) -> None:
        """Test getting incoming edges to a node."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("art-001", "coupon_spec", "hash1")
        graph.add_node("art-002", "config", "hash2")
        graph.add_node("art-003", "kicad_board", "hash3")

        graph.add_edge("art-001", "art-003", "derived_from")
        graph.add_edge("art-002", "art-003", "config_from")

        edges = graph.get_edges_to("art-003")
        assert len(edges) == 2
        source_ids = {e.source_id for e in edges}
        assert source_ids == {"art-001", "art-002"}

        graph.close()


class TestAddManifest:
    """Tests for adding manifests to the graph."""

    def test_add_manifest_creates_node(self, tmp_path: Path) -> None:
        """Test that add_manifest creates a node."""
        store = ArtifactStore(tmp_path / "store")
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        manifest = store.put(
            content=b"test content",
            artifact_type="coupon_spec",
            roles=["geometry"],
            run_id="run-001",
        )

        node = graph.add_manifest(manifest)

        assert node.artifact_id == manifest.artifact_id
        assert node.artifact_type == "coupon_spec"
        assert node.content_hash_digest == manifest.content_hash.digest
        assert graph.count_nodes() == 1

        graph.close()

    def test_add_manifest_creates_edges(self, tmp_path: Path) -> None:
        """Test that add_manifest creates edges from inputs."""
        store = ArtifactStore(tmp_path / "store")
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        spec_manifest = store.put(
            content=b"spec",
            artifact_type="coupon_spec",
            roles=["geometry"],
            run_id="run-001",
        )

        graph.add_manifest(spec_manifest)

        board_manifest = store.put(
            content=b"board",
            artifact_type="kicad_board",
            roles=["intermediate"],
            run_id="run-001",
            inputs=[
                LineageReference(
                    artifact_id=spec_manifest.artifact_id,
                    relation="config_from",
                    content_hash=spec_manifest.content_hash,
                )
            ],
        )

        graph.add_manifest(board_manifest)

        assert graph.count_nodes() == 2
        assert graph.count_edges() == 1

        edges = graph.get_edges_to(board_manifest.artifact_id)
        assert len(edges) == 1
        assert edges[0].source_id == spec_manifest.artifact_id
        assert edges[0].relation == "config_from"

        graph.close()


class TestAncestorQueries:
    """Tests for ancestor queries."""

    def test_get_ancestors_simple(self, tmp_path: Path) -> None:
        """Test getting ancestors of an artifact."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("root", "coupon_spec", "hash1")
        graph.add_node("mid", "kicad_board", "hash2")
        graph.add_node("leaf", "gerber", "hash3")

        graph.add_edge("root", "mid", "derived_from")
        graph.add_edge("mid", "leaf", "derived_from")

        subgraph = graph.get_ancestors("leaf")

        assert len(subgraph.nodes) == 3
        assert "root" in subgraph.nodes
        assert "mid" in subgraph.nodes
        assert "leaf" in subgraph.nodes

        graph.close()

    def test_get_ancestors_with_max_depth(self, tmp_path: Path) -> None:
        """Test ancestors with depth limit."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("root", "coupon_spec", "hash1")
        graph.add_node("mid1", "kicad_board", "hash2")
        graph.add_node("mid2", "gerber", "hash3")
        graph.add_node("leaf", "touchstone", "hash4")

        graph.add_edge("root", "mid1", "derived_from")
        graph.add_edge("mid1", "mid2", "derived_from")
        graph.add_edge("mid2", "leaf", "derived_from")

        subgraph = graph.get_ancestors("leaf", max_depth=1)

        assert "leaf" in subgraph.nodes
        assert "mid2" in subgraph.nodes
        assert "mid1" not in subgraph.nodes
        assert "root" not in subgraph.nodes

        graph.close()

    def test_get_ancestors_with_relation_filter(self, tmp_path: Path) -> None:
        """Test ancestors filtered by relation type."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("spec", "coupon_spec", "hash1")
        graph.add_node("config", "config", "hash2")
        graph.add_node("board", "kicad_board", "hash3")

        graph.add_edge("spec", "board", "derived_from")
        graph.add_edge("config", "board", "config_from")

        subgraph = graph.get_ancestors("board", relation_filter=["derived_from"])

        assert len(subgraph.edges) == 1
        assert subgraph.edges[0].relation == "derived_from"
        assert "spec" in subgraph.nodes
        assert "config" not in subgraph.nodes

        graph.close()

    def test_get_ancestors_not_found(self, tmp_path: Path) -> None:
        """Test that get_ancestors raises for missing node."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        with pytest.raises(NodeNotFoundError):
            graph.get_ancestors("nonexistent")

        graph.close()


class TestDescendantQueries:
    """Tests for descendant queries."""

    def test_get_descendants_simple(self, tmp_path: Path) -> None:
        """Test getting descendants of an artifact."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("root", "coupon_spec", "hash1")
        graph.add_node("child1", "kicad_board", "hash2")
        graph.add_node("child2", "gerber", "hash3")

        graph.add_edge("root", "child1", "derived_from")
        graph.add_edge("root", "child2", "derived_from")

        subgraph = graph.get_descendants("root")

        assert len(subgraph.nodes) == 3
        assert "root" in subgraph.nodes
        assert "child1" in subgraph.nodes
        assert "child2" in subgraph.nodes

        graph.close()

    def test_get_descendants_with_max_depth(self, tmp_path: Path) -> None:
        """Test descendants with depth limit."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("root", "coupon_spec", "hash1")
        graph.add_node("mid", "kicad_board", "hash2")
        graph.add_node("leaf", "gerber", "hash3")

        graph.add_edge("root", "mid", "derived_from")
        graph.add_edge("mid", "leaf", "derived_from")

        subgraph = graph.get_descendants("root", max_depth=1)

        assert "root" in subgraph.nodes
        assert "mid" in subgraph.nodes
        assert "leaf" not in subgraph.nodes

        graph.close()

    def test_get_descendants_not_found(self, tmp_path: Path) -> None:
        """Test that get_descendants raises for missing node."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        with pytest.raises(NodeNotFoundError):
            graph.get_descendants("nonexistent")

        graph.close()


class TestDirectQueries:
    """Tests for direct input/output queries."""

    def test_get_direct_inputs(self, tmp_path: Path) -> None:
        """Test getting direct inputs of an artifact."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("input1", "coupon_spec", "hash1")
        graph.add_node("input2", "config", "hash2")
        graph.add_node("output", "kicad_board", "hash3")

        graph.add_edge("input1", "output", "derived_from")
        graph.add_edge("input2", "output", "config_from")

        inputs = graph.get_direct_inputs("output")

        assert len(inputs) == 2
        input_ids = {n.artifact_id for n in inputs}
        assert input_ids == {"input1", "input2"}

        graph.close()

    def test_get_direct_outputs(self, tmp_path: Path) -> None:
        """Test getting direct outputs of an artifact."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("input", "coupon_spec", "hash1")
        graph.add_node("output1", "kicad_board", "hash2")
        graph.add_node("output2", "gerber", "hash3")

        graph.add_edge("input", "output1", "derived_from")
        graph.add_edge("input", "output2", "derived_from")

        outputs = graph.get_direct_outputs("input")

        assert len(outputs) == 2
        output_ids = {n.artifact_id for n in outputs}
        assert output_ids == {"output1", "output2"}

        graph.close()


class TestTraceOperations:
    """Tests for trace operations."""

    def test_trace_to_roots(self, tmp_path: Path) -> None:
        """Test tracing to root nodes."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("root1", "coupon_spec", "hash1")
        graph.add_node("root2", "config", "hash2")
        graph.add_node("mid", "kicad_board", "hash3")
        graph.add_node("leaf", "gerber", "hash4")

        graph.add_edge("root1", "mid", "derived_from")
        graph.add_edge("root2", "mid", "config_from")
        graph.add_edge("mid", "leaf", "derived_from")

        subgraph = graph.trace_to_roots("leaf")
        roots = subgraph.get_roots()

        assert set(roots) == {"root1", "root2"}

        graph.close()

    def test_trace_to_leaves(self, tmp_path: Path) -> None:
        """Test tracing to leaf nodes."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("root", "coupon_spec", "hash1")
        graph.add_node("mid", "kicad_board", "hash2")
        graph.add_node("leaf1", "gerber", "hash3")
        graph.add_node("leaf2", "drill_file", "hash4")

        graph.add_edge("root", "mid", "derived_from")
        graph.add_edge("mid", "leaf1", "derived_from")
        graph.add_edge("mid", "leaf2", "derived_from")

        subgraph = graph.trace_to_leaves("root")
        leaves = subgraph.get_leaves()

        assert set(leaves) == {"leaf1", "leaf2"}

        graph.close()


class TestBuildFromStore:
    """Tests for building the graph from a store."""

    def test_build_from_store(self, tmp_path: Path) -> None:
        """Test building the graph from an artifact store."""
        store = ArtifactStore(tmp_path / "store")
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        m1 = store.put(
            content=b"spec",
            artifact_type="coupon_spec",
            roles=["geometry"],
            run_id="run-001",
        )

        store.put(
            content=b"board",
            artifact_type="kicad_board",
            roles=["intermediate"],
            run_id="run-001",
            inputs=[
                LineageReference(
                    artifact_id=m1.artifact_id,
                    relation="config_from",
                )
            ],
        )

        count = graph.build_from_store(store)

        assert count == 2
        assert graph.count_nodes() == 2
        assert graph.count_edges() == 1

        graph.close()

    def test_build_from_store_clears_existing(self, tmp_path: Path) -> None:
        """Test that build_from_store clears existing data by default."""
        store = ArtifactStore(tmp_path / "store")
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("old-node", "other", "oldhash")

        store.put(
            content=b"new content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        graph.build_from_store(store)

        assert not graph.has_node("old-node")

        graph.close()


class TestGraphClear:
    """Tests for clearing the graph."""

    def test_clear(self, tmp_path: Path) -> None:
        """Test clearing all data from the graph."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("node1", "type1", "hash1")
        graph.add_node("node2", "type2", "hash2")
        graph.add_edge("node1", "node2", "derived_from")

        graph.clear()

        assert graph.count_nodes() == 0
        assert graph.count_edges() == 0

        graph.close()


class TestQueryNodes:
    """Tests for querying nodes."""

    def test_query_by_type(self, tmp_path: Path) -> None:
        """Test querying nodes by type."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("spec1", "coupon_spec", "hash1")
        graph.add_node("spec2", "coupon_spec", "hash2")
        graph.add_node("board", "kicad_board", "hash3")

        specs = graph.query_nodes(artifact_type="coupon_spec")

        assert len(specs) == 2
        spec_ids = {n.artifact_id for n in specs}
        assert spec_ids == {"spec1", "spec2"}

        graph.close()

    def test_query_by_run(self, tmp_path: Path) -> None:
        """Test querying nodes by run ID."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("art1", "other", "hash1", run_id="run-001")
        graph.add_node("art2", "other", "hash2", run_id="run-001")
        graph.add_node("art3", "other", "hash3", run_id="run-002")

        run1_nodes = graph.query_nodes(run_id="run-001")

        assert len(run1_nodes) == 2
        node_ids = {n.artifact_id for n in run1_nodes}
        assert node_ids == {"art1", "art2"}

        graph.close()

    def test_query_with_limit(self, tmp_path: Path) -> None:
        """Test querying with pagination."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        for i in range(10):
            graph.add_node(f"art-{i:03d}", "other", f"hash{i}")

        page1 = graph.query_nodes(limit=3, offset=0)
        page2 = graph.query_nodes(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3

        page1_ids = {n.artifact_id for n in page1}
        page2_ids = {n.artifact_id for n in page2}
        assert page1_ids.isdisjoint(page2_ids)

        graph.close()


class TestGraphStats:
    """Tests for graph statistics."""

    def test_get_stats(self, tmp_path: Path) -> None:
        """Test getting graph statistics."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("spec1", "coupon_spec", "hash1")
        graph.add_node("spec2", "coupon_spec", "hash2")
        graph.add_node("board", "kicad_board", "hash3")
        graph.add_node("gerber", "gerber", "hash4")

        graph.add_edge("spec1", "board", "derived_from")
        graph.add_edge("spec2", "board", "config_from")
        graph.add_edge("board", "gerber", "derived_from")

        stats = graph.get_stats()

        assert stats["node_count"] == 4
        assert stats["edge_count"] == 3
        assert stats["nodes_by_type"]["coupon_spec"] == 2
        assert stats["nodes_by_type"]["kicad_board"] == 1
        assert stats["edges_by_relation"]["derived_from"] == 2
        assert stats["edges_by_relation"]["config_from"] == 1
        assert stats["root_count"] == 2
        assert stats["leaf_count"] == 1

        graph.close()


class TestExport:
    """Tests for graph export."""

    def test_export_to_dict(self, tmp_path: Path) -> None:
        """Test exporting the graph as a dictionary."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("art-001", "coupon_spec", "hash1")
        graph.add_node("art-002", "kicad_board", "hash2")
        graph.add_edge("art-001", "art-002", "derived_from")

        data = graph.export_to_dict()

        assert data["schema_version"] == 1
        assert data["node_count"] == 2
        assert data["edge_count"] == 1
        assert "art-001" in data["nodes"]
        assert "art-002" in data["nodes"]
        assert len(data["edges"]) == 1

        graph.close()

    def test_export_to_json(self, tmp_path: Path) -> None:
        """Test exporting the graph as JSON."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        graph.add_node("art-001", "coupon_spec", "hash1")

        json_str = graph.export_to_json()

        assert '"art-001"' in json_str
        assert '"coupon_spec"' in json_str

        graph.close()


class TestDataclasses:
    """Tests for dataclass functionality."""

    def test_lineage_node_to_dict(self) -> None:
        """Test LineageNode serialization."""
        node = LineageNode(
            artifact_id="art-001",
            artifact_type="coupon_spec",
            content_hash_digest="abc123",
            run_id="run-001",
            stage_name="generation",
        )

        data = node.to_dict()

        assert data["artifact_id"] == "art-001"
        assert data["artifact_type"] == "coupon_spec"
        assert data["content_hash_digest"] == "abc123"
        assert data["run_id"] == "run-001"

    def test_lineage_edge_to_dict(self) -> None:
        """Test LineageEdge serialization."""
        edge = LineageEdge(
            source_id="art-001",
            target_id="art-002",
            relation="derived_from",
        )

        data = edge.to_dict()

        assert data["source_id"] == "art-001"
        assert data["target_id"] == "art-002"
        assert data["relation"] == "derived_from"

    def test_lineage_path(self) -> None:
        """Test LineagePath properties."""
        path = LineagePath(
            nodes=[
                LineageNode("a", "t", "h"),
                LineageNode("b", "t", "h"),
            ],
            edges=[LineageEdge("a", "b", "derived_from")],
        )

        assert path.length == 1
        data = path.to_dict()
        assert len(data["nodes"]) == 2
        assert data["length"] == 1

    def test_lineage_subgraph_roots_and_leaves(self, tmp_path: Path) -> None:
        """Test LineageSubgraph roots and leaves detection."""
        subgraph = LineageSubgraph(
            nodes={
                "root1": LineageNode("root1", "t", "h"),
                "root2": LineageNode("root2", "t", "h"),
                "mid": LineageNode("mid", "t", "h"),
                "leaf": LineageNode("leaf", "t", "h"),
            },
            edges=[
                LineageEdge("root1", "mid", "derived_from"),
                LineageEdge("root2", "mid", "derived_from"),
                LineageEdge("mid", "leaf", "derived_from"),
            ],
        )

        roots = subgraph.get_roots()
        leaves = subgraph.get_leaves()

        assert set(roots) == {"root1", "root2"}
        assert set(leaves) == {"leaf"}


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_operations(self, tmp_path: Path) -> None:
        """Test that concurrent operations work correctly."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        num_threads = 10
        errors: list[Exception] = []

        def add_nodes(i: int) -> None:
            try:
                graph.add_node(f"art-{i:03d}", "other", f"hash{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_nodes, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert graph.count_nodes() == num_threads

        graph.close()


class TestComplexLineage:
    """Tests for complex lineage scenarios."""

    def test_diamond_dependency(self, tmp_path: Path) -> None:
        """Test diamond-shaped dependency graph."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        #     root
        #    /    \
        #   mid1  mid2
        #    \    /
        #     leaf

        graph.add_node("root", "coupon_spec", "h1")
        graph.add_node("mid1", "kicad_board", "h2")
        graph.add_node("mid2", "config", "h3")
        graph.add_node("leaf", "gerber", "h4")

        graph.add_edge("root", "mid1", "derived_from")
        graph.add_edge("root", "mid2", "derived_from")
        graph.add_edge("mid1", "leaf", "derived_from")
        graph.add_edge("mid2", "leaf", "config_from")

        ancestors = graph.get_ancestors("leaf")
        assert len(ancestors.nodes) == 4

        descendants = graph.get_descendants("root")
        assert len(descendants.nodes) == 4

        graph.close()

    def test_multiple_lineages(self, tmp_path: Path) -> None:
        """Test multiple independent lineage chains."""
        graph = LineageGraph(tmp_path / "lineage.sqlite")
        graph.initialize()

        for chain in range(3):
            for i in range(4):
                graph.add_node(f"chain{chain}-{i}", "other", f"h{chain}{i}")
                if i > 0:
                    graph.add_edge(f"chain{chain}-{i - 1}", f"chain{chain}-{i}", "derived_from")

        chain0_ancestors = graph.get_ancestors("chain0-3")
        chain1_ancestors = graph.get_ancestors("chain1-3")

        chain0_ids = set(chain0_ancestors.nodes.keys())
        chain1_ids = set(chain1_ancestors.nodes.keys())

        assert chain0_ids.isdisjoint(chain1_ids)

        graph.close()
