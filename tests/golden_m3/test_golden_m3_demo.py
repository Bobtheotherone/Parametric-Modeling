"""Golden demo test for M3 subsystem: Full capability demonstration.

This golden test demonstrates the complete M3 workflow including:
1. Project initialization with m3 init
2. Artifact creation and storage with content-addressed hashing
3. Lineage tracking through the artifact graph
4. Dataset snapshot creation with versioning
5. Garbage collection with retention policies and pinning
6. Audit and verification of artifact provenance

This file serves as both a comprehensive integration test and as documentation
for how the M3 subsystem should be used.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from formula_foundry.m3.artifact_store import (
    ArtifactManifest,
    ArtifactStore,
    LineageReference,
)
from formula_foundry.m3.cli_main import (
    build_parser,
    cmd_audit,
    cmd_gc,
    cmd_init,
    main,
)
from formula_foundry.m3.dataset_snapshot import (
    DatasetMember,
    DatasetSnapshot,
    DatasetSnapshotReader,
    DatasetSnapshotWriter,
    compute_manifest_hash,
)
from formula_foundry.m3.gc import (
    BUILTIN_POLICIES,
    GarbageCollector,
    RetentionPolicy,
)
from formula_foundry.m3.lineage_graph import LineageGraph
from formula_foundry.m3.registry import ArtifactRegistry

if TYPE_CHECKING:
    pass


# ============================================================================
# Golden Test Data - Simulating a coupon generation pipeline
# ============================================================================

GOLDEN_COUPON_SPEC = {
    "schema_version": 1,
    "coupon_family": "F1_SINGLE_ENDED_VIA",
    "units": "nm",
    "toolchain": {
        "kicad": {
            "version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7@sha256:abc123",
        }
    },
    "fab_profile": {
        "id": "oshpark_4layer",
    },
    "stackup": {
        "copper_layers": 4,
        "thicknesses_nm": {
            "L1_to_L2": 180000,
            "L2_to_L3": 800000,
            "L3_to_L4": 180000,
        },
        "materials": {
            "er": 4.1,
            "loss_tangent": 0.02,
        },
    },
    "board": {
        "outline": {
            "width_nm": 20000000,
            "length_nm": 80000000,
            "corner_radius_nm": 2000000,
        },
        "origin": {
            "mode": "EDGE_L_CENTER",
        },
    },
    "transmission_line": {
        "type": "CPWG",
        "layer": "F.Cu",
        "w_nm": 300000,
        "gap_nm": 180000,
    },
    "discontinuity": {
        "type": "VIA_TRANSITION",
        "signal_via": {
            "drill_nm": 300000,
            "diameter_nm": 650000,
        },
    },
}

GOLDEN_RESOLVED_DESIGN = {
    "design_id": "design-001",
    "spec_id": "spec-001",
    "w_nm": 300000,
    "gap_nm": 180000,
    "via_drill_nm": 300000,
    "via_diameter_nm": 650000,
    "computed_impedance_ohm": 50.2,
    "computed_via_inductance_nh": 0.35,
}

GOLDEN_KICAD_BOARD = b"""(kicad_pcb (version 20231014) (generator "formula_foundry")
  (general
    (thickness 1.6)
  )
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
  )
  ; Golden demo board content - simplified for testing
)
"""

GOLDEN_TOUCHSTONE = b"""! Golden demo S-parameter file
! Frequency: 1 GHz - 40 GHz
# GHz S MA R 50
1.0  -0.1  -89.5  -30.2  -0.8  -30.2  -0.8  -0.1  -89.5
10.0 -0.2  -89.2  -20.1  -1.5  -20.1  -1.5  -0.2  -89.2
20.0 -0.3  -88.8  -15.5  -2.2  -15.5  -2.2  -0.3  -88.8
40.0 -0.5  -87.5  -10.2  -3.8  -10.2  -3.8  -0.5  -87.5
"""


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def golden_project(tmp_path: Path) -> Path:
    """Create a fully initialized M3 project for golden demo testing.

    This fixture sets up:
    - Git repository
    - M3 initialization (data directories, registry, lineage DB)
    - DVC configuration (minimal)

    Returns:
        Path to the project root.
    """
    # Initialize git repository
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "golden-demo@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Golden Demo"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Create initial commit
    (tmp_path / "README.md").write_text("Golden M3 Demo Project")
    subprocess.run(
        ["git", "add", "."],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Create DVC configuration (required for m3 run)
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n"
        "  coupon_generation:\n"
        "    cmd: echo 'Generate coupon'\n"
        "  em_simulation:\n"
        "    cmd: echo 'Run EM simulation'\n"
        "  dataset_build:\n"
        "    cmd: echo 'Build dataset'\n"
    )

    # Initialize M3 subsystem
    cmd_init(root=tmp_path, force=False, quiet=True)

    return tmp_path


@pytest.fixture
def golden_store(golden_project: Path) -> ArtifactStore:
    """Create an ArtifactStore for golden demo testing."""
    data_dir = golden_project / "data"
    return ArtifactStore(
        root=data_dir,
        generator="golden_demo",
        generator_version="1.0.0",
    )


@pytest.fixture
def golden_registry(golden_project: Path) -> ArtifactRegistry:
    """Create an ArtifactRegistry for golden demo testing."""
    return ArtifactRegistry(golden_project / "data" / "registry.db")


@pytest.fixture
def golden_lineage(golden_project: Path) -> LineageGraph:
    """Create a LineageGraph for golden demo testing."""
    graph = LineageGraph(golden_project / "data" / "lineage.sqlite")
    graph.initialize()
    return graph


# ============================================================================
# Phase 1: Project Initialization
# ============================================================================


class TestPhase1Initialization:
    """Phase 1: Verify that m3 init creates the complete project structure."""

    def test_init_creates_data_directories(self, golden_project: Path) -> None:
        """m3 init should create all required data directories."""
        data_dir = golden_project / "data"

        # Core storage directories
        assert (data_dir / "objects").is_dir(), "objects/ directory should exist"
        assert (data_dir / "manifests").is_dir(), "manifests/ directory should exist"
        assert (data_dir / "datasets").is_dir(), "datasets/ directory should exist"

        # MLflow integration
        assert (data_dir / "mlflow").is_dir(), "mlflow/ directory should exist"
        assert (data_dir / "mlflow" / "artifacts").is_dir(), "mlflow/artifacts/ should exist"

    def test_init_creates_databases(self, golden_project: Path) -> None:
        """m3 init should create SQLite databases for registry and lineage."""
        data_dir = golden_project / "data"

        assert (data_dir / "registry.db").exists(), "registry.db should exist"
        # Lineage DB is created on first use, but the init run should create it
        # via the registry recording

    def test_init_records_init_run(self, golden_registry: ArtifactRegistry) -> None:
        """m3 init should record itself as the first run in the registry."""
        runs = golden_registry.query_runs()
        golden_registry.close()

        # Should have at least one run (the init run)
        assert len(runs) >= 1, "Should have at least one run from init"

        # The init run should be identifiable
        init_runs = [r for r in runs if r.run_id.startswith("init-")]
        assert len(init_runs) >= 1, "Should have an init run"

    def test_init_is_idempotent(self, golden_project: Path) -> None:
        """Running m3 init twice without --force should not fail."""
        # First init was done in fixture
        # Second init should succeed silently
        result = cmd_init(root=golden_project, force=False, quiet=True)
        assert result == 0, "Second init without --force should succeed"


# ============================================================================
# Phase 2: Artifact Creation and Storage
# ============================================================================


class TestPhase2ArtifactStorage:
    """Phase 2: Demonstrate artifact creation, storage, and content-addressed hashing."""

    def test_store_coupon_spec_artifact(
        self,
        golden_store: ArtifactStore,
        golden_registry: ArtifactRegistry,
    ) -> None:
        """Store a coupon spec as an artifact with proper metadata."""
        content = json.dumps(GOLDEN_COUPON_SPEC, indent=2, sort_keys=True).encode()

        manifest = golden_store.put(
            content=content,
            artifact_type="coupon_spec",
            roles=["config", "root_input"],
            run_id="golden-run-001",
            artifact_id="golden-art-spec-001",
            stage_name="coupon_generation",
            tags={"demo": "golden"},
        )
        golden_registry.index_artifact(manifest)

        # Verify manifest properties
        assert manifest.artifact_type == "coupon_spec"
        assert "config" in manifest.roles
        assert "root_input" in manifest.roles
        assert manifest.content_hash.algorithm == "sha256"
        assert len(manifest.content_hash.digest) == 64

        # Verify content can be retrieved
        retrieved = golden_store.get(manifest.content_hash.digest)
        assert retrieved == content

        golden_registry.close()

    def test_content_hash_determinism(self, golden_store: ArtifactStore) -> None:
        """Same content should always produce the same hash."""
        content = json.dumps(GOLDEN_COUPON_SPEC, indent=2, sort_keys=True).encode()

        # Create two artifacts with identical content
        manifest1 = golden_store.put(
            content=content,
            artifact_type="coupon_spec",
            roles=["config"],
            run_id="run-001",
        )
        manifest2 = golden_store.put(
            content=content,
            artifact_type="coupon_spec",
            roles=["config"],
            run_id="run-002",
        )

        # Content hashes must be identical
        assert manifest1.content_hash.digest == manifest2.content_hash.digest
        assert manifest1.spec_id == manifest2.spec_id

    def test_spec_id_canonical_format(self, golden_store: ArtifactStore) -> None:
        """Spec ID should be a deterministic 12-char base32 identifier."""
        content = json.dumps(GOLDEN_COUPON_SPEC, indent=2, sort_keys=True).encode()

        manifest = golden_store.put(
            content=content,
            artifact_type="coupon_spec",
            roles=["config"],
            run_id="run-001",
        )

        # Spec ID should be 12 characters, lowercase alphanumeric
        assert len(manifest.spec_id) == 12
        assert manifest.spec_id.isalnum()

        # Should be reproducible from content
        computed_spec_id = golden_store.compute_spec_id(content)
        assert manifest.spec_id == computed_spec_id

    def test_store_artifact_chain(
        self,
        golden_store: ArtifactStore,
        golden_registry: ArtifactRegistry,
        golden_lineage: LineageGraph,
    ) -> None:
        """Store a chain of artifacts with lineage relationships."""
        # 1. Store coupon spec (root)
        spec_content = json.dumps(GOLDEN_COUPON_SPEC, indent=2, sort_keys=True).encode()
        spec_manifest = golden_store.put(
            content=spec_content,
            artifact_type="coupon_spec",
            roles=["config", "root_input"],
            run_id="golden-chain-001",
            artifact_id="chain-spec-001",
            stage_name="coupon_generation",
        )
        golden_registry.index_artifact(spec_manifest)
        golden_lineage.add_manifest(spec_manifest)

        # 2. Store resolved design (derived from spec)
        design_content = json.dumps(GOLDEN_RESOLVED_DESIGN, indent=2, sort_keys=True).encode()
        design_manifest = golden_store.put(
            content=design_content,
            artifact_type="resolved_design",
            roles=["geometry", "intermediate"],
            run_id="golden-chain-001",
            artifact_id="chain-design-001",
            stage_name="coupon_generation",
            inputs=[
                LineageReference(
                    artifact_id="chain-spec-001",
                    relation="config_from",
                )
            ],
        )
        golden_registry.index_artifact(design_manifest)
        golden_lineage.add_manifest(design_manifest)

        # 3. Store KiCad board (derived from design)
        board_manifest = golden_store.put(
            content=GOLDEN_KICAD_BOARD,
            artifact_type="kicad_board",
            roles=["final_output", "cad"],
            run_id="golden-chain-001",
            artifact_id="chain-board-001",
            stage_name="coupon_generation",
            inputs=[
                LineageReference(
                    artifact_id="chain-design-001",
                    relation="derived_from",
                )
            ],
        )
        golden_registry.index_artifact(board_manifest)
        golden_lineage.add_manifest(board_manifest)

        # 4. Store touchstone (simulation output)
        touchstone_manifest = golden_store.put(
            content=GOLDEN_TOUCHSTONE,
            artifact_type="touchstone",
            roles=["oracle_output", "simulation"],
            run_id="golden-sim-001",
            artifact_id="chain-touchstone-001",
            stage_name="em_simulation",
            inputs=[
                LineageReference(
                    artifact_id="chain-board-001",
                    relation="derived_from",
                )
            ],
        )
        golden_registry.index_artifact(touchstone_manifest)
        golden_lineage.add_manifest(touchstone_manifest)

        # Verify the chain exists
        assert golden_store.exists_by_id("chain-spec-001")
        assert golden_store.exists_by_id("chain-design-001")
        assert golden_store.exists_by_id("chain-board-001")
        assert golden_store.exists_by_id("chain-touchstone-001")

        golden_registry.close()
        golden_lineage.close()


# ============================================================================
# Phase 3: Lineage Tracking
# ============================================================================


class TestPhase3LineageTracking:
    """Phase 3: Demonstrate lineage graph operations and provenance tracking."""

    @pytest.fixture
    def populated_lineage(
        self,
        golden_store: ArtifactStore,
        golden_registry: ArtifactRegistry,
        golden_lineage: LineageGraph,
    ) -> tuple[LineageGraph, list[str]]:
        """Create a populated lineage graph for testing.

        Creates the following graph:
            spec-001 ─┬─> design-001 ─> board-001 ─> touchstone-001
            spec-002 ─┘                           └─> gerber-001
        """
        artifact_ids = []

        # Spec 1 (root)
        spec1 = golden_store.put(
            content=b'{"spec": "coupon_parameters"}',
            artifact_type="coupon_spec",
            roles=["config", "root_input"],
            run_id="lineage-run-001",
            artifact_id="lineage-spec-001",
        )
        golden_registry.index_artifact(spec1)
        golden_lineage.add_manifest(spec1)
        artifact_ids.append(spec1.artifact_id)

        # Spec 2 (stackup, also root)
        spec2 = golden_store.put(
            content=b'{"stackup": "4-layer FR4"}',
            artifact_type="coupon_spec",
            roles=["config", "root_input"],
            run_id="lineage-run-001",
            artifact_id="lineage-spec-002",
        )
        golden_registry.index_artifact(spec2)
        golden_lineage.add_manifest(spec2)
        artifact_ids.append(spec2.artifact_id)

        # Design (derived from both specs)
        design = golden_store.put(
            content=b'{"w_nm": 300000, "gap_nm": 180000}',
            artifact_type="resolved_design",
            roles=["geometry"],
            run_id="lineage-run-001",
            artifact_id="lineage-design-001",
            inputs=[
                LineageReference(artifact_id="lineage-spec-001", relation="config_from"),
                LineageReference(artifact_id="lineage-spec-002", relation="config_from"),
            ],
        )
        golden_registry.index_artifact(design)
        golden_lineage.add_manifest(design)
        artifact_ids.append(design.artifact_id)

        # Board (derived from design)
        board = golden_store.put(
            content=GOLDEN_KICAD_BOARD,
            artifact_type="kicad_board",
            roles=["final_output"],
            run_id="lineage-run-001",
            artifact_id="lineage-board-001",
            inputs=[
                LineageReference(artifact_id="lineage-design-001", relation="derived_from"),
            ],
        )
        golden_registry.index_artifact(board)
        golden_lineage.add_manifest(board)
        artifact_ids.append(board.artifact_id)

        # Touchstone (derived from board)
        touchstone = golden_store.put(
            content=GOLDEN_TOUCHSTONE,
            artifact_type="touchstone",
            roles=["oracle_output"],
            run_id="lineage-sim-001",
            artifact_id="lineage-touchstone-001",
            inputs=[
                LineageReference(artifact_id="lineage-board-001", relation="derived_from"),
            ],
        )
        golden_registry.index_artifact(touchstone)
        golden_lineage.add_manifest(touchstone)
        artifact_ids.append(touchstone.artifact_id)

        # Gerber (also derived from board)
        gerber = golden_store.put(
            content=b"G04 Gerber file*\n%FSLAX34Y34*%\n",
            artifact_type="gerber",
            roles=["fabrication"],
            run_id="lineage-run-001",
            artifact_id="lineage-gerber-001",
            inputs=[
                LineageReference(artifact_id="lineage-board-001", relation="derived_from"),
            ],
        )
        golden_registry.index_artifact(gerber)
        golden_lineage.add_manifest(gerber)
        artifact_ids.append(gerber.artifact_id)

        golden_registry.close()

        return golden_lineage, artifact_ids

    def test_get_ancestors(self, populated_lineage: tuple[LineageGraph, list[str]]) -> None:
        """Get all ancestors of an artifact."""
        lineage, artifact_ids = populated_lineage

        # Get ancestors of touchstone
        ancestors = lineage.get_ancestors("lineage-touchstone-001")

        # Should include board, design, and both specs
        assert "lineage-board-001" in ancestors.nodes
        assert "lineage-design-001" in ancestors.nodes
        assert "lineage-spec-001" in ancestors.nodes
        assert "lineage-spec-002" in ancestors.nodes

        lineage.close()

    def test_get_descendants(self, populated_lineage: tuple[LineageGraph, list[str]]) -> None:
        """Get all descendants of an artifact."""
        lineage, artifact_ids = populated_lineage

        # Get descendants of design
        descendants = lineage.get_descendants("lineage-design-001")

        # Should include board, touchstone, and gerber
        assert "lineage-board-001" in descendants.nodes
        assert "lineage-touchstone-001" in descendants.nodes
        assert "lineage-gerber-001" in descendants.nodes

        lineage.close()

    def test_trace_to_roots(self, populated_lineage: tuple[LineageGraph, list[str]]) -> None:
        """Trace lineage back to root artifacts."""
        lineage, artifact_ids = populated_lineage

        # Trace touchstone back to roots
        roots_result = lineage.trace_to_roots("lineage-touchstone-001")
        root_ids = set(roots_result.get_roots())

        # Roots should be both specs (get_roots returns artifact_id strings)
        assert "lineage-spec-001" in root_ids
        assert "lineage-spec-002" in root_ids

        lineage.close()

    def test_graph_statistics(self, populated_lineage: tuple[LineageGraph, list[str]]) -> None:
        """Verify graph statistics."""
        lineage, artifact_ids = populated_lineage

        assert lineage.count_nodes() == 6
        # Edges: spec1->design, spec2->design, design->board, board->touchstone, board->gerber = 5
        assert lineage.count_edges() == 5

        lineage.close()


# ============================================================================
# Phase 4: Dataset Snapshots
# ============================================================================


class TestPhase4DatasetSnapshot:
    """Phase 4: Demonstrate dataset snapshot creation, versioning, and querying."""

    @pytest.fixture
    def dataset_artifacts(
        self,
        golden_store: ArtifactStore,
        golden_registry: ArtifactRegistry,
    ) -> list[ArtifactManifest]:
        """Create artifacts for dataset testing."""
        manifests = []

        # Create 5 touchstone files (simulation outputs)
        for i in range(5):
            content = f"! Touchstone {i}\n# GHz S MA R 50\n{i}.0 -0.{i} -89.{i} -30.{i} -0.{i}\n".encode()
            manifest = golden_store.put(
                content=content,
                artifact_type="touchstone",
                roles=["oracle_output", "simulation"],
                run_id=f"dataset-sim-{i:03d}",
                artifact_id=f"dataset-touchstone-{i:03d}",
                stage_name="em_simulation",
                tags={"frequency_ghz": str(i), "coupon_id": f"coupon-{i:03d}"},
            )
            golden_registry.index_artifact(manifest)
            manifests.append(manifest)

        golden_registry.close()
        return manifests

    def test_create_dataset_snapshot(
        self,
        golden_project: Path,
        golden_store: ArtifactStore,
        dataset_artifacts: list[ArtifactManifest],
    ) -> None:
        """Create a dataset snapshot from artifacts."""
        dataset_dir = golden_project / "data" / "datasets"

        writer = DatasetSnapshotWriter(
            dataset_id="golden_em_dataset",
            version="v1.0",
            store=golden_store,
            generator="golden_demo",
            generator_version="1.0.0",
            name="Golden EM Dataset",
            description="Dataset of EM simulation results for coupon designs",
        )

        # Add members with features
        for i, manifest in enumerate(dataset_artifacts):
            writer.add_member(
                manifest,
                role="oracle_output",
                features={
                    "frequency_ghz": float(i),
                    "coupon_index": i,
                    "s11_db": -0.1 * i,
                    "s21_db": -30.0 - i,
                },
            )

        writer.set_tags({"domain": "electromagnetics", "phase": "training"})
        writer.set_annotations({"split_seed": 42, "model_version": "0.1.0"})

        snapshot = writer.finalize(output_dir=dataset_dir, write_parquet=False)

        # Verify snapshot properties
        assert snapshot.dataset_id == "golden_em_dataset"
        assert snapshot.version == "v1.0"
        assert snapshot.member_count == 5
        assert snapshot.content_hash.algorithm == "sha256"
        assert len(snapshot.content_hash.digest) == 64

        # Verify manifest file was written
        manifest_path = dataset_dir / "golden_em_dataset_v1.0.json"
        assert manifest_path.exists()

    def test_dataset_hash_determinism(
        self,
        golden_store: ArtifactStore,
        dataset_artifacts: list[ArtifactManifest],
    ) -> None:
        """Dataset manifest hash should be deterministic regardless of member order."""
        # Create members in order 0, 1, 2, 3, 4
        members_forward = [
            DatasetMember.from_manifest(m, role="test")
            for m in dataset_artifacts
        ]
        hash_forward = compute_manifest_hash(members_forward)

        # Create members in reverse order 4, 3, 2, 1, 0
        members_reverse = [
            DatasetMember.from_manifest(m, role="test")
            for m in reversed(dataset_artifacts)
        ]
        hash_reverse = compute_manifest_hash(members_reverse)

        # Hashes should be identical (order-independent)
        assert hash_forward.digest == hash_reverse.digest

    def test_read_dataset_snapshot(
        self,
        golden_project: Path,
        golden_store: ArtifactStore,
        dataset_artifacts: list[ArtifactManifest],
    ) -> None:
        """Read and verify a dataset snapshot."""
        dataset_dir = golden_project / "data" / "datasets"

        # Create dataset
        writer = DatasetSnapshotWriter(
            dataset_id="golden_read_test",
            version="v1.0",
            store=golden_store,
            generator="golden_demo",
            generator_version="1.0.0",
        )
        for manifest in dataset_artifacts:
            writer.add_member(manifest, role="data")
        snapshot = writer.finalize(output_dir=dataset_dir, write_parquet=False)

        # Read it back
        manifest_path = dataset_dir / "golden_read_test_v1.0.json"
        reader = DatasetSnapshotReader(snapshot_path=manifest_path, store=golden_store)
        loaded = reader.load()

        # Verify loaded matches original
        assert loaded.dataset_id == snapshot.dataset_id
        assert loaded.version == snapshot.version
        assert loaded.member_count == snapshot.member_count
        assert loaded.content_hash.digest == snapshot.content_hash.digest

        # Verify integrity
        is_valid, errors = reader.verify_integrity()
        assert is_valid
        assert len(errors) == 0

    def test_dataset_versioning(
        self,
        golden_project: Path,
        golden_store: ArtifactStore,
        golden_registry: ArtifactRegistry,
        dataset_artifacts: list[ArtifactManifest],
    ) -> None:
        """Create multiple versions of a dataset."""
        dataset_dir = golden_project / "data" / "datasets"

        # Create v1.0 with first 3 artifacts
        writer_v1 = DatasetSnapshotWriter(
            dataset_id="versioned_dataset",
            version="v1.0",
            store=golden_store,
            generator="golden_demo",
            generator_version="1.0.0",
        )
        for manifest in dataset_artifacts[:3]:
            writer_v1.add_member(manifest, role="data")
        snapshot_v1 = writer_v1.finalize(output_dir=dataset_dir, write_parquet=False)

        # Create v2.0 with all 5 artifacts, referencing v1.0 as parent
        writer_v2 = DatasetSnapshotWriter(
            dataset_id="versioned_dataset",
            version="v2.0",
            store=golden_store,
            generator="golden_demo",
            generator_version="1.0.0",
            parent_version="v1.0",
        )
        for manifest in dataset_artifacts:
            writer_v2.add_member(manifest, role="data")
        snapshot_v2 = writer_v2.finalize(output_dir=dataset_dir, write_parquet=False)

        # Verify versions
        assert snapshot_v1.member_count == 3
        assert snapshot_v2.member_count == 5
        assert snapshot_v2.parent_version == "v1.0"
        assert snapshot_v1.content_hash.digest != snapshot_v2.content_hash.digest

        golden_registry.close()


# ============================================================================
# Phase 5: Garbage Collection
# ============================================================================


class TestPhase5GarbageCollection:
    """Phase 5: Demonstrate garbage collection with retention policies and pinning."""

    @pytest.fixture
    def gc_setup(
        self,
        golden_project: Path,
        golden_store: ArtifactStore,
        golden_registry: ArtifactRegistry,
        golden_lineage: LineageGraph,
    ) -> dict[str, Any]:
        """Set up artifacts for GC testing."""
        # Create 10 artifacts of various types
        manifests = []

        for i in range(10):
            artifact_type = "touchstone" if i % 2 == 0 else "other"
            roles = ["oracle_output"] if i % 3 == 0 else ["intermediate"]

            manifest = golden_store.put(
                content=f"GC test content {i}".encode(),
                artifact_type=artifact_type,
                roles=roles,
                run_id=f"gc-run-{i:03d}",
                artifact_id=f"gc-artifact-{i:03d}",
            )
            golden_registry.index_artifact(manifest)
            golden_lineage.add_manifest(manifest)
            manifests.append(manifest)

        gc = GarbageCollector(
            data_dir=golden_project / "data",
            store=golden_store,
            registry=golden_registry,
            lineage=golden_lineage,
        )

        return {
            "project": golden_project,
            "store": golden_store,
            "registry": golden_registry,
            "lineage": golden_lineage,
            "gc": gc,
            "manifests": manifests,
        }

    def test_list_builtin_policies(self) -> None:
        """Verify builtin policies are available."""
        assert "laptop_default" in BUILTIN_POLICIES
        assert "ci_aggressive" in BUILTIN_POLICIES
        assert "archive" in BUILTIN_POLICIES
        assert "dev_minimal" in BUILTIN_POLICIES

        # Verify laptop_default properties
        laptop = BUILTIN_POLICIES["laptop_default"]
        assert laptop.keep_min_age_days == 14
        assert laptop.keep_pinned is True
        assert laptop.keep_with_descendants is True

    def test_gc_dry_run(self, gc_setup: dict[str, Any]) -> None:
        """GC dry run should not delete anything."""
        gc = gc_setup["gc"]
        registry = gc_setup["registry"]

        initial_count = registry.count_artifacts()

        # Very aggressive policy that would delete everything
        policy = RetentionPolicy(
            name="test_delete_all",
            keep_min_age_days=0,
            keep_min_count=0,
            keep_pinned=False,
            keep_with_descendants=False,
        )

        result = gc.run(policy=policy, dry_run=True, run_dvc_gc=False)

        # Should report what would be deleted
        assert result.dry_run is True
        assert result.artifacts_deleted >= 1
        assert result.artifacts_scanned == initial_count

        # But nothing should actually be deleted
        assert registry.count_artifacts() == initial_count

        registry.close()

    def test_gc_pinning_protection(self, gc_setup: dict[str, Any]) -> None:
        """Pinned artifacts should be protected from GC."""
        gc = gc_setup["gc"]
        manifests = gc_setup["manifests"]

        # Pin the first artifact
        gc.pin_artifact(
            artifact_id=manifests[0].artifact_id,
            reason="Golden demo protected artifact",
        )

        # Verify pin is recorded
        assert gc.is_pinned(manifests[0].artifact_id)

        # Aggressive policy
        policy = RetentionPolicy(
            name="test_aggressive",
            keep_min_age_days=0,
            keep_min_count=0,
            keep_pinned=True,  # Respect pins
            keep_with_descendants=False,
        )

        to_delete, to_keep = gc.compute_candidates(policy)

        # Pinned artifact should be in to_keep
        pinned_kept = [c for c in to_keep if c.artifact_id == manifests[0].artifact_id]
        assert len(pinned_kept) == 1
        assert "pinned" in pinned_kept[0].reasons_to_keep

        gc_setup["registry"].close()
        gc_setup["lineage"].close()

    def test_gc_actual_deletion(self, gc_setup: dict[str, Any]) -> None:
        """GC should actually delete artifacts when not in dry-run mode."""
        gc = gc_setup["gc"]
        store = gc_setup["store"]
        registry = gc_setup["registry"]

        initial_count = registry.count_artifacts()

        # Delete all but keep 2 per type
        policy = RetentionPolicy(
            name="test_keep_2",
            keep_min_age_days=0,
            keep_min_count=2,
            keep_pinned=True,
            keep_with_descendants=False,
        )

        result = gc.run(policy=policy, dry_run=False, run_dvc_gc=False)

        # Should have deleted some
        assert result.dry_run is False
        assert result.artifacts_deleted > 0

        # Remaining count should be less
        final_count = registry.count_artifacts()
        assert final_count < initial_count
        assert final_count == initial_count - result.artifacts_deleted

        registry.close()

    def test_gc_estimate_savings(self, gc_setup: dict[str, Any]) -> None:
        """Estimate space savings before running GC."""
        gc = gc_setup["gc"]

        policy = RetentionPolicy(
            name="test_estimate",
            keep_min_age_days=0,
            keep_min_count=2,
            keep_pinned=False,
            keep_with_descendants=False,
        )

        estimate = gc.estimate_savings(policy)

        assert "artifacts_to_delete" in estimate
        assert "bytes_to_delete" in estimate
        assert estimate["artifacts_to_delete"] > 0
        assert estimate["bytes_to_delete"] > 0

        gc_setup["registry"].close()
        gc_setup["lineage"].close()


# ============================================================================
# Phase 6: Audit and Verification
# ============================================================================


class TestPhase6AuditVerification:
    """Phase 6: Demonstrate audit and verification capabilities."""

    @pytest.fixture
    def auditable_project(
        self,
        golden_project: Path,
        golden_store: ArtifactStore,
        golden_registry: ArtifactRegistry,
        golden_lineage: LineageGraph,
    ) -> tuple[Path, list[str]]:
        """Create a project with artifacts for auditing."""
        # Create a lineage chain
        spec = golden_store.put(
            content=json.dumps(GOLDEN_COUPON_SPEC, sort_keys=True).encode(),
            artifact_type="coupon_spec",
            roles=["config", "root_input"],
            run_id="audit-run-001",
            artifact_id="audit-spec-001",
        )
        golden_registry.index_artifact(spec)
        golden_lineage.add_manifest(spec)

        design = golden_store.put(
            content=json.dumps(GOLDEN_RESOLVED_DESIGN, sort_keys=True).encode(),
            artifact_type="resolved_design",
            roles=["geometry"],
            run_id="audit-run-001",
            artifact_id="audit-design-001",
            inputs=[LineageReference(artifact_id="audit-spec-001", relation="config_from")],
        )
        golden_registry.index_artifact(design)
        golden_lineage.add_manifest(design)

        board = golden_store.put(
            content=GOLDEN_KICAD_BOARD,
            artifact_type="kicad_board",
            roles=["final_output"],
            run_id="audit-run-001",
            artifact_id="audit-board-001",
            inputs=[LineageReference(artifact_id="audit-design-001", relation="derived_from")],
        )
        golden_registry.index_artifact(board)
        golden_lineage.add_manifest(board)

        golden_registry.close()
        golden_lineage.close()

        return golden_project, ["audit-spec-001", "audit-design-001", "audit-board-001"]

    def test_audit_single_artifact(
        self,
        auditable_project: tuple[Path, list[str]],
    ) -> None:
        """Audit a single artifact."""
        project, artifact_ids = auditable_project

        result = cmd_audit(
            artifact_id="audit-board-001",
            root=project,
            output_format="text",
            trace_roots=False,
            verify_hashes=False,
            max_depth=None,
            required_roles=None,
            quiet=True,
        )

        assert result == 0

    def test_audit_with_lineage_trace(
        self,
        auditable_project: tuple[Path, list[str]],
    ) -> None:
        """Audit an artifact with full lineage trace to roots."""
        project, artifact_ids = auditable_project

        result = cmd_audit(
            artifact_id="audit-board-001",
            root=project,
            output_format="text",
            trace_roots=True,
            verify_hashes=False,
            max_depth=None,
            required_roles=None,
            quiet=True,
        )

        assert result == 0

    def test_audit_with_hash_verification(
        self,
        auditable_project: tuple[Path, list[str]],
    ) -> None:
        """Audit with content hash verification."""
        project, artifact_ids = auditable_project

        result = cmd_audit(
            artifact_id="audit-board-001",
            root=project,
            output_format="text",
            trace_roots=False,
            verify_hashes=True,
            max_depth=None,
            required_roles=None,
            quiet=True,
        )

        assert result == 0

    def test_audit_required_roles_success(
        self,
        auditable_project: tuple[Path, list[str]],
    ) -> None:
        """Audit verifies required roles exist in ancestry."""
        project, artifact_ids = auditable_project

        result = cmd_audit(
            artifact_id="audit-board-001",
            root=project,
            output_format="text",
            trace_roots=True,
            verify_hashes=False,
            max_depth=None,
            required_roles="config,root_input",  # Spec has these
            quiet=True,
        )

        assert result == 0

    def test_audit_required_roles_failure(
        self,
        auditable_project: tuple[Path, list[str]],
    ) -> None:
        """Audit fails when required roles are missing in ancestry."""
        project, artifact_ids = auditable_project

        result = cmd_audit(
            artifact_id="audit-board-001",
            root=project,
            output_format="text",
            trace_roots=True,
            verify_hashes=False,
            max_depth=None,
            required_roles="simulation_result",  # None of our artifacts have this
            quiet=True,
        )

        assert result == 2  # Exit code for missing required roles

    def test_audit_json_output(
        self,
        auditable_project: tuple[Path, list[str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Audit produces valid JSON output."""
        project, artifact_ids = auditable_project

        result = cmd_audit(
            artifact_id="audit-board-001",
            root=project,
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

        assert "total_artifacts" in report
        assert "artifacts" in report
        assert report["total_artifacts"] >= 1


# ============================================================================
# Phase 7: Full Workflow Integration
# ============================================================================


class TestPhase7FullWorkflow:
    """Phase 7: Complete end-to-end workflow demonstrating all M3 capabilities."""

    def test_complete_m3_workflow(self, golden_project: Path) -> None:
        """Execute the complete M3 workflow from init to verification.

        This test demonstrates the full capability of the M3 subsystem:
        1. Initialize project
        2. Create artifacts with lineage
        3. Build a dataset
        4. Run garbage collection
        5. Audit and verify

        This serves as the primary golden demo for M3.
        """
        data_dir = golden_project / "data"

        # ---- Step 1: Create components ----
        store = ArtifactStore(
            root=data_dir,
            generator="golden_workflow",
            generator_version="1.0.0",
        )
        registry = ArtifactRegistry(data_dir / "registry.db")
        lineage = LineageGraph(data_dir / "lineage.sqlite")
        lineage.initialize()

        # ---- Step 2: Create coupon spec (root) ----
        spec_content = json.dumps(GOLDEN_COUPON_SPEC, indent=2, sort_keys=True).encode()
        spec = store.put(
            content=spec_content,
            artifact_type="coupon_spec",
            roles=["config", "root_input"],
            run_id="golden-workflow-001",
            artifact_id="workflow-spec-001",
            stage_name="coupon_generation",
        )
        registry.index_artifact(spec)
        lineage.add_manifest(spec)

        # ---- Step 3: Create resolved design (derived from spec) ----
        design_content = json.dumps(GOLDEN_RESOLVED_DESIGN, indent=2, sort_keys=True).encode()
        design = store.put(
            content=design_content,
            artifact_type="resolved_design",
            roles=["geometry", "intermediate"],
            run_id="golden-workflow-001",
            artifact_id="workflow-design-001",
            stage_name="coupon_generation",
            inputs=[LineageReference(artifact_id="workflow-spec-001", relation="config_from")],
        )
        registry.index_artifact(design)
        lineage.add_manifest(design)

        # ---- Step 4: Create KiCad board (derived from design) ----
        board = store.put(
            content=GOLDEN_KICAD_BOARD,
            artifact_type="kicad_board",
            roles=["final_output", "cad"],
            run_id="golden-workflow-001",
            artifact_id="workflow-board-001",
            stage_name="coupon_generation",
            inputs=[LineageReference(artifact_id="workflow-design-001", relation="derived_from")],
        )
        registry.index_artifact(board)
        lineage.add_manifest(board)

        # ---- Step 5: Create simulation outputs ----
        touchstones = []
        for i in range(3):
            ts_content = f"! Workflow touchstone {i}\n# GHz S MA R 50\n".encode()
            ts = store.put(
                content=ts_content,
                artifact_type="touchstone",
                roles=["oracle_output", "simulation"],
                run_id=f"golden-sim-{i:03d}",
                artifact_id=f"workflow-touchstone-{i:03d}",
                stage_name="em_simulation",
                inputs=[LineageReference(artifact_id="workflow-board-001", relation="derived_from")],
            )
            registry.index_artifact(ts)
            lineage.add_manifest(ts)
            touchstones.append(ts)

        # ---- Step 6: Create dataset snapshot ----
        dataset_dir = data_dir / "datasets"
        writer = DatasetSnapshotWriter(
            dataset_id="golden_workflow_dataset",
            version="v1.0",
            store=store,
            generator="golden_workflow",
            generator_version="1.0.0",
            name="Golden Workflow Dataset",
            description="Complete workflow demonstration dataset",
        )
        for i, ts in enumerate(touchstones):
            writer.add_member(
                ts,
                role="oracle_output",
                features={"index": i, "freq_ghz": float(i + 1)},
            )
        writer.set_tags({"workflow": "golden", "phase": "demo"})
        snapshot = writer.finalize(output_dir=dataset_dir, write_parquet=False)
        registry.index_dataset_snapshot(snapshot)

        # ---- Step 7: Verify lineage ----
        ancestors = lineage.get_ancestors("workflow-touchstone-000")
        assert "workflow-board-001" in ancestors.nodes
        assert "workflow-design-001" in ancestors.nodes
        assert "workflow-spec-001" in ancestors.nodes

        roots = lineage.trace_to_roots("workflow-touchstone-000")
        root_ids = set(roots.get_roots())  # get_roots returns artifact_id strings
        assert "workflow-spec-001" in root_ids

        # ---- Step 8: Run GC with pinning ----
        gc = GarbageCollector(
            data_dir=data_dir,
            store=store,
            registry=registry,
            lineage=lineage,
        )

        # Pin the dataset
        gc.pin_artifact(artifact_id=touchstones[0].artifact_id, reason="First simulation")

        # Dry run to see what would be deleted
        policy = RetentionPolicy(
            name="test_policy",
            keep_min_age_days=0,
            keep_min_count=1,
            keep_pinned=True,
            keep_with_descendants=True,
        )
        to_delete, to_keep = gc.compute_candidates(policy)

        # Pinned artifact should be protected
        pinned_kept = [c for c in to_keep if c.artifact_id == touchstones[0].artifact_id]
        assert len(pinned_kept) == 1

        # ---- Step 9: Audit final artifact ----
        result = cmd_audit(
            artifact_id="workflow-touchstone-000",
            root=golden_project,
            output_format="text",
            trace_roots=True,
            verify_hashes=True,
            max_depth=None,
            required_roles="config,root_input",
            quiet=True,
        )
        assert result == 0, "Audit should pass for valid artifact chain"

        # ---- Step 10: Verify dataset integrity ----
        reader = DatasetSnapshotReader(
            snapshot_path=dataset_dir / "golden_workflow_dataset_v1.0.json",
            store=store,
        )
        loaded = reader.load()
        is_valid, errors = reader.verify_integrity()
        assert is_valid, f"Dataset should be valid: {errors}"
        assert loaded.member_count == 3

        # ---- Cleanup ----
        registry.close()
        lineage.close()

        # Final assertions
        assert snapshot.dataset_id == "golden_workflow_dataset"
        assert snapshot.version == "v1.0"
        assert store.exists_by_id("workflow-spec-001")
        assert store.exists_by_id("workflow-touchstone-000")
