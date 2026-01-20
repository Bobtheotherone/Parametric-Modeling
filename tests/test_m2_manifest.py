"""Tests for M2 manifest generation (REQ-M2-018).

This module tests the M2 manifest generation functionality including:
- MeshStatistics computation
- ConvergenceMetrics extraction
- PortConfiguration serialization
- M2ManifestBuilder operations
- Full manifest building
- Manifest validation
- Manifest I/O
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from formula_foundry.coupongen.hashing import coupon_id_from_design_hash
from formula_foundry.coupongen.resolve import design_hash, resolve
from formula_foundry.coupongen.spec import CouponSpec
from formula_foundry.openems import (
    ConvergenceMetrics,
    M2ManifestBuilder,
    MeshStatistics,
    PortConfiguration,
    SimulationRunner,
    build_m2_manifest,
    load_m2_manifest,
    manifest_hash,
    validate_m2_manifest,
    write_m2_manifest,
)
from formula_foundry.openems.convert import build_simulation_spec
from formula_foundry.openems.geometry import GeometrySpec, build_geometry_spec
from formula_foundry.openems.spec import MeshSpec, SimulationSpec
from formula_foundry.substrate import canonical_json_dumps


def _example_spec_data() -> dict[str, Any]:
    """Return example CouponSpec data for testing."""
    return {
        "schema_version": 1,
        "coupon_family": "F1_SINGLE_ENDED_VIA",
        "units": "nm",
        "toolchain": {
            "kicad": {
                "version": "9.0.7",
                "docker_image": "kicad/kicad:9.0.7@sha256:deadbeef",
            }
        },
        "fab_profile": {"id": "oshpark_4layer", "overrides": {}},
        "stackup": {
            "copper_layers": 4,
            "thicknesses_nm": {
                "L1_to_L2": 180000,
                "L2_to_L3": 800000,
                "L3_to_L4": 180000,
            },
            "materials": {"er": 4.1, "loss_tangent": 0.02},
        },
        "board": {
            "outline": {
                "width_nm": 20000000,
                "length_nm": 80000000,
                "corner_radius_nm": 2000000,
            },
            "origin": {"mode": "EDGE_L_CENTER"},
            "text": {"coupon_id": "${COUPON_ID}", "include_manifest_hash": True},
        },
        "connectors": {
            "left": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [5000000, 0],
                "rotation_deg": 180,
            },
            "right": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [75000000, 0],
                "rotation_deg": 0,
            },
        },
        "transmission_line": {
            "type": "CPWG",
            "layer": "F.Cu",
            "w_nm": 300000,
            "gap_nm": 180000,
            "length_left_nm": 25000000,
            "length_right_nm": 25000000,
            "ground_via_fence": None,
        },
        "discontinuity": {
            "type": "VIA_TRANSITION",
            "signal_via": {
                "drill_nm": 300000,
                "diameter_nm": 650000,
                "pad_diameter_nm": 900000,
            },
            "antipads": {},
            "return_vias": None,
            "plane_cutouts": {},
        },
        "constraints": {
            "mode": "REJECT",
            "drc": {"must_pass": True, "severity": "all"},
            "symmetry": {"enforce": True},
            "allow_unconnected_copper": False,
        },
        "export": {
            "gerbers": {"enabled": True, "format": "gerbers"},
            "drill": {"enabled": True, "format": "excellon"},
            "outputs_dir": "artifacts/",
        },
    }


def _manifest_for(spec: CouponSpec) -> dict[str, Any]:
    """Build manifest dictionary from CouponSpec."""
    resolved = resolve(spec)
    design_hash_value = design_hash(resolved)
    return {
        "design_hash": design_hash_value,
        "coupon_id": coupon_id_from_design_hash(design_hash_value),
        "coupon_family": spec.coupon_family,
        "stackup": spec.stackup.model_dump(mode="json"),
    }


def _build_test_inputs() -> tuple[SimulationSpec, GeometrySpec]:
    """Build SimulationSpec and GeometrySpec for testing."""
    spec = CouponSpec.model_validate(_example_spec_data())
    resolved = resolve(spec)
    manifest = _manifest_for(spec)
    geometry = build_geometry_spec(resolved, manifest)
    simulation = build_simulation_spec(resolved, manifest)
    return simulation, geometry


# =============================================================================
# MeshStatistics Tests
# =============================================================================


class TestMeshStatistics:
    """Tests for MeshStatistics dataclass."""

    def test_to_dict(self) -> None:
        """Test MeshStatistics.to_dict serialization."""
        stats = MeshStatistics(
            total_cells=1000000,
            n_lines_x=100,
            n_lines_y=100,
            n_lines_z=100,
            x_cell_min_nm=10000,
            x_cell_max_nm=500000,
            x_cell_mean_nm=150000.0,
            y_cell_min_nm=10000,
            y_cell_max_nm=400000,
            y_cell_mean_nm=120000.0,
            z_cell_min_nm=5000,
            z_cell_max_nm=200000,
            z_cell_mean_nm=80000.0,
        )

        result = stats.to_dict()

        assert result["total_cells"] == 1000000
        assert result["n_lines_x"] == 100
        assert result["n_lines_y"] == 100
        assert result["n_lines_z"] == 100
        assert result["x_cell_min_nm"] == 10000
        assert result["x_cell_max_nm"] == 500000
        assert result["x_cell_mean_nm"] == 150000.0
        assert result["y_cell_min_nm"] == 10000
        assert result["y_cell_max_nm"] == 400000
        assert result["y_cell_mean_nm"] == 120000.0
        assert result["z_cell_min_nm"] == 5000
        assert result["z_cell_max_nm"] == 200000
        assert result["z_cell_mean_nm"] == 80000.0

    def test_from_mesh_spec(self) -> None:
        """Test MeshStatistics.from_mesh_spec computation."""
        mesh_spec = MeshSpec(
            fixed_lines_x_nm=[0, 100000, 200000, 300000, 400000],
            fixed_lines_y_nm=[0, 50000, 100000, 150000, 200000],
            fixed_lines_z_nm=[0, 25000, 50000, 75000, 100000],
        )

        stats = MeshStatistics.from_mesh_spec(mesh_spec)

        assert stats.n_lines_x == 5
        assert stats.n_lines_y == 5
        assert stats.n_lines_z == 5
        # 4 cells in each direction = 64 total
        assert stats.total_cells == 64
        # All cells are 100000 nm in X
        assert stats.x_cell_min_nm == 100000
        assert stats.x_cell_max_nm == 100000
        # All cells are 50000 nm in Y
        assert stats.y_cell_min_nm == 50000
        assert stats.y_cell_max_nm == 50000
        # All cells are 25000 nm in Z
        assert stats.z_cell_min_nm == 25000
        assert stats.z_cell_max_nm == 25000


# =============================================================================
# ConvergenceMetrics Tests
# =============================================================================


class TestConvergenceMetrics:
    """Tests for ConvergenceMetrics dataclass."""

    def test_to_dict_minimal(self) -> None:
        """Test ConvergenceMetrics.to_dict with minimal fields."""
        metrics = ConvergenceMetrics(
            engine_type="multithreaded",
            termination_criteria_db=-50.0,
            max_timesteps=1000000,
        )

        result = metrics.to_dict()

        assert result["engine_type"] == "multithreaded"
        assert result["termination_criteria_db"] == -50.0
        assert result["max_timesteps"] == 1000000
        assert "actual_timesteps" not in result
        assert "final_energy_db" not in result
        assert "converged" not in result

    def test_to_dict_full(self) -> None:
        """Test ConvergenceMetrics.to_dict with all fields."""
        metrics = ConvergenceMetrics(
            engine_type="sse",
            termination_criteria_db=-60.0,
            max_timesteps=500000,
            actual_timesteps=250000,
            final_energy_db=-65.0,
            converged=True,
        )

        result = metrics.to_dict()

        assert result["engine_type"] == "sse"
        assert result["termination_criteria_db"] == -60.0
        assert result["max_timesteps"] == 500000
        assert result["actual_timesteps"] == 250000
        assert result["final_energy_db"] == -65.0
        assert result["converged"] is True

    def test_from_spec(self) -> None:
        """Test ConvergenceMetrics.from_spec extraction."""
        simulation, _ = _build_test_inputs()

        metrics = ConvergenceMetrics.from_spec(simulation)

        assert metrics.engine_type == simulation.control.engine.type
        assert metrics.termination_criteria_db == simulation.control.termination.end_criteria_db
        assert metrics.max_timesteps == simulation.control.termination.max_timesteps


# =============================================================================
# PortConfiguration Tests
# =============================================================================


class TestPortConfiguration:
    """Tests for PortConfiguration dataclass."""

    def test_to_dict_minimal(self) -> None:
        """Test PortConfiguration.to_dict with minimal fields."""
        port = PortConfiguration(
            id="P1",
            type="lumped",
            impedance_ohm=50.0,
            excite=True,
            position_nm=(5000000, 0, 100000),
            direction="x",
        )

        result = port.to_dict()

        assert result["id"] == "P1"
        assert result["type"] == "lumped"
        assert result["impedance_ohm"] == 50.0
        assert result["excite"] is True
        assert result["position_nm"] == [5000000, 0, 100000]
        assert result["direction"] == "x"
        assert result["deembed_enabled"] is False
        assert "deembed_distance_nm" not in result

    def test_to_dict_with_deembed(self) -> None:
        """Test PortConfiguration.to_dict with de-embedding enabled."""
        port = PortConfiguration(
            id="P2",
            type="msl",
            impedance_ohm=50.0,
            excite=False,
            position_nm=(75000000, 0, 100000),
            direction="-x",
            deembed_enabled=True,
            deembed_distance_nm=1000000,
        )

        result = port.to_dict()

        assert result["id"] == "P2"
        assert result["deembed_enabled"] is True
        assert result["deembed_distance_nm"] == 1000000


# =============================================================================
# M2ManifestBuilder Tests
# =============================================================================


class TestM2ManifestBuilder:
    """Tests for M2ManifestBuilder class."""

    def test_builder_basic(self, tmp_path: Path) -> None:
        """Test basic manifest building with builder."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)

        builder = M2ManifestBuilder(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
        )
        manifest = builder.build()

        assert manifest["schema_version"] == 1
        assert manifest["simulation_hash"] == result.simulation_hash
        assert manifest["design_hash"] == geometry.design_hash
        assert manifest["coupon_family"] == geometry.coupon_family
        assert "toolchain" in manifest
        assert "toolchain_hash" in manifest
        assert "frequency_sweep" in manifest
        assert "excitation" in manifest
        assert "boundaries" in manifest
        assert "mesh_config" in manifest
        assert "ports" in manifest
        assert "outputs" in manifest
        assert "lineage" in manifest

    def test_builder_with_mesh_statistics(self, tmp_path: Path) -> None:
        """Test builder with mesh statistics."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)

        mesh_spec = MeshSpec(
            fixed_lines_x_nm=[0, 100000, 200000],
            fixed_lines_y_nm=[0, 50000, 100000],
            fixed_lines_z_nm=[0, 25000, 50000],
        )

        builder = M2ManifestBuilder(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
        )
        builder.with_mesh_spec(mesh_spec)
        manifest = builder.build()

        assert "mesh_statistics" in manifest
        assert manifest["mesh_statistics"]["n_lines_x"] == 3
        assert manifest["mesh_statistics"]["n_lines_y"] == 3
        assert manifest["mesh_statistics"]["n_lines_z"] == 3

    def test_builder_with_convergence_from_spec(self, tmp_path: Path) -> None:
        """Test builder with convergence metrics from spec."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)

        builder = M2ManifestBuilder(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
        )
        builder.with_convergence_from_spec()
        manifest = builder.build()

        assert "convergence" in manifest
        assert manifest["convergence"]["engine_type"] == simulation.control.engine.type
        assert manifest["convergence"]["termination_criteria_db"] == simulation.control.termination.end_criteria_db

    def test_builder_with_m1_manifest(self, tmp_path: Path) -> None:
        """Test builder with M1 manifest hash."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)
        m1_hash = "a" * 64

        builder = M2ManifestBuilder(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
        )
        builder.with_m1_manifest(m1_hash)
        manifest = builder.build()

        assert manifest["lineage"]["m1_manifest_hash"] == m1_hash

    def test_builder_chaining(self, tmp_path: Path) -> None:
        """Test builder method chaining."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)
        mesh_spec = MeshSpec(
            fixed_lines_x_nm=[0, 100000, 200000],
            fixed_lines_y_nm=[0, 50000, 100000],
            fixed_lines_z_nm=[0, 25000, 50000],
        )

        manifest = (
            M2ManifestBuilder(
                spec=simulation,
                geometry=geometry,
                simulation_result=result,
            )
            .with_mesh_spec(mesh_spec)
            .with_convergence_from_spec()
            .with_m1_manifest("b" * 64)
            .with_git_sha("c" * 40)
            .with_timestamp("2025-01-01T00:00:00Z")
            .build()
        )

        assert "mesh_statistics" in manifest
        assert "convergence" in manifest
        assert manifest["lineage"]["m1_manifest_hash"] == "b" * 64
        assert manifest["lineage"]["git_sha"] == "c" * 40
        assert manifest["lineage"]["timestamp_utc"] == "2025-01-01T00:00:00Z"


# =============================================================================
# build_m2_manifest Tests
# =============================================================================


class TestBuildM2Manifest:
    """Tests for build_m2_manifest function."""

    def test_basic_manifest(self, tmp_path: Path) -> None:
        """Test basic manifest building."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)

        manifest = build_m2_manifest(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
        )

        assert manifest["schema_version"] == 1
        assert "simulation_hash" in manifest
        assert "spec_hash" in manifest
        assert "geometry_hash" in manifest
        assert len(manifest["spec_hash"]) == 64
        assert len(manifest["geometry_hash"]) == 64

    def test_manifest_with_mesh_spec(self, tmp_path: Path) -> None:
        """Test manifest building with mesh specification."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)
        mesh_spec = MeshSpec(
            fixed_lines_x_nm=[0, 100000, 200000, 300000],
            fixed_lines_y_nm=[0, 50000, 100000, 150000],
            fixed_lines_z_nm=[0, 25000, 50000, 75000],
        )

        manifest = build_m2_manifest(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
            mesh_spec=mesh_spec,
        )

        assert "mesh_statistics" in manifest
        assert manifest["mesh_statistics"]["total_cells"] == 27  # 3x3x3

    def test_manifest_includes_ports(self, tmp_path: Path) -> None:
        """Test manifest includes port configurations."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)

        manifest = build_m2_manifest(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
        )

        assert "ports" in manifest
        assert len(manifest["ports"]) == len(simulation.ports)
        for port_manifest, port_spec in zip(manifest["ports"], simulation.ports, strict=True):
            assert port_manifest["id"] == port_spec.id
            assert port_manifest["type"] == port_spec.type
            assert port_manifest["impedance_ohm"] == port_spec.impedance_ohm

    def test_manifest_includes_frequency_sweep(self, tmp_path: Path) -> None:
        """Test manifest includes frequency sweep configuration."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)

        manifest = build_m2_manifest(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
        )

        assert "frequency_sweep" in manifest
        assert manifest["frequency_sweep"]["f_start_hz"] == simulation.frequency.f_start_hz
        assert manifest["frequency_sweep"]["f_stop_hz"] == simulation.frequency.f_stop_hz
        assert manifest["frequency_sweep"]["n_points"] == simulation.frequency.n_points

    def test_manifest_includes_execution_time(self, tmp_path: Path) -> None:
        """Test manifest includes execution time."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)

        manifest = build_m2_manifest(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
        )

        assert "execution_time_sec" in manifest
        assert manifest["execution_time_sec"] >= 0

    def test_manifest_includes_sparam_path(self, tmp_path: Path) -> None:
        """Test manifest includes S-parameter file path."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)

        manifest = build_m2_manifest(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
        )

        assert "sparam_path" in manifest
        assert manifest["sparam_path"].endswith(".s2p")

    def test_manifest_deterministic(self, tmp_path: Path) -> None:
        """Test manifest generation is deterministic."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)

        fixed_timestamp = "2025-01-01T00:00:00Z"
        fixed_git_sha = "a" * 40

        manifest1 = build_m2_manifest(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
            git_sha=fixed_git_sha,
            timestamp_utc=fixed_timestamp,
        )

        manifest2 = build_m2_manifest(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
            git_sha=fixed_git_sha,
            timestamp_utc=fixed_timestamp,
        )

        # Compare serialized forms (excluding execution_time which varies)
        manifest1.pop("execution_time_sec", None)
        manifest2.pop("execution_time_sec", None)

        assert canonical_json_dumps(manifest1) == canonical_json_dumps(manifest2)


# =============================================================================
# Manifest I/O Tests
# =============================================================================


class TestManifestIO:
    """Tests for manifest I/O functions."""

    def test_write_and_load(self, tmp_path: Path) -> None:
        """Test writing and loading a manifest."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)

        manifest = build_m2_manifest(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
            git_sha="a" * 40,
            timestamp_utc="2025-01-01T00:00:00Z",
        )

        manifest_path = tmp_path / "manifest.json"
        write_m2_manifest(manifest_path, manifest)

        assert manifest_path.exists()

        loaded = load_m2_manifest(manifest_path)

        assert loaded["schema_version"] == manifest["schema_version"]
        assert loaded["simulation_hash"] == manifest["simulation_hash"]
        assert loaded["design_hash"] == manifest["design_hash"]

    def test_manifest_canonical_json(self, tmp_path: Path) -> None:
        """Test that manifest is written as canonical JSON."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)

        manifest = build_m2_manifest(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
        )

        manifest_path = tmp_path / "manifest.json"
        write_m2_manifest(manifest_path, manifest)

        content = manifest_path.read_text(encoding="utf-8").strip()

        # Canonical JSON has no extra whitespace
        parsed = json.loads(content)
        expected = canonical_json_dumps(parsed)
        assert content == expected


# =============================================================================
# Manifest Hash Tests
# =============================================================================


class TestManifestHash:
    """Tests for manifest hashing."""

    def test_hash_is_deterministic(self, tmp_path: Path) -> None:
        """Test manifest hash is deterministic."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)

        manifest = build_m2_manifest(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
            git_sha="a" * 40,
            timestamp_utc="2025-01-01T00:00:00Z",
        )

        hash1 = manifest_hash(manifest)
        hash2 = manifest_hash(manifest)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_hash_changes_with_content(self, tmp_path: Path) -> None:
        """Test manifest hash changes when content changes."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)

        manifest1 = build_m2_manifest(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
            git_sha="a" * 40,
            timestamp_utc="2025-01-01T00:00:00Z",
        )

        manifest2 = build_m2_manifest(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
            git_sha="b" * 40,  # Different git SHA
            timestamp_utc="2025-01-01T00:00:00Z",
        )

        hash1 = manifest_hash(manifest1)
        hash2 = manifest_hash(manifest2)

        assert hash1 != hash2


# =============================================================================
# Manifest Validation Tests
# =============================================================================


class TestManifestValidation:
    """Tests for manifest validation."""

    def test_valid_manifest(self, tmp_path: Path) -> None:
        """Test validation of a valid manifest."""
        simulation, geometry = _build_test_inputs()
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "run"

        result = runner.run(simulation, geometry, output_dir=output_dir)

        manifest = build_m2_manifest(
            spec=simulation,
            geometry=geometry,
            simulation_result=result,
        )

        errors = validate_m2_manifest(manifest)

        assert errors == []

    def test_missing_required_field(self) -> None:
        """Test validation catches missing required field."""
        manifest = {
            "schema_version": 1,
            # Missing other required fields
        }

        errors = validate_m2_manifest(manifest)

        assert len(errors) > 0
        assert any("simulation_hash" in e for e in errors)

    def test_invalid_schema_version(self) -> None:
        """Test validation catches invalid schema version."""
        manifest = {
            "schema_version": "invalid",
            "simulation_hash": "a" * 64,
            "spec_hash": "b" * 64,
            "geometry_hash": "c" * 64,
            "design_hash": "d" * 64,
            "coupon_family": "F1_SINGLE_ENDED_VIA",
            "toolchain": {},
            "toolchain_hash": "e" * 64,
            "frequency_sweep": {},
            "excitation": {},
            "boundaries": {},
            "mesh_config": {},
            "ports": [{}],
            "outputs": [],
            "lineage": {"git_sha": "f" * 40, "timestamp_utc": "2025-01-01T00:00:00Z"},
        }

        errors = validate_m2_manifest(manifest)

        assert any("schema_version" in e and "integer" in e for e in errors)

    def test_invalid_simulation_hash(self) -> None:
        """Test validation catches invalid simulation hash."""
        manifest = {
            "schema_version": 1,
            "simulation_hash": "tooshort",
            "spec_hash": "b" * 64,
            "geometry_hash": "c" * 64,
            "design_hash": "d" * 64,
            "coupon_family": "F1_SINGLE_ENDED_VIA",
            "toolchain": {},
            "toolchain_hash": "e" * 64,
            "frequency_sweep": {},
            "excitation": {},
            "boundaries": {},
            "mesh_config": {},
            "ports": [{}],
            "outputs": [],
            "lineage": {"git_sha": "f" * 40, "timestamp_utc": "2025-01-01T00:00:00Z"},
        }

        errors = validate_m2_manifest(manifest)

        assert any("simulation_hash" in e and "64-character" in e for e in errors)

    def test_missing_lineage_fields(self) -> None:
        """Test validation catches missing lineage fields."""
        manifest = {
            "schema_version": 1,
            "simulation_hash": "a" * 64,
            "spec_hash": "b" * 64,
            "geometry_hash": "c" * 64,
            "design_hash": "d" * 64,
            "coupon_family": "F1_SINGLE_ENDED_VIA",
            "toolchain": {},
            "toolchain_hash": "e" * 64,
            "frequency_sweep": {},
            "excitation": {},
            "boundaries": {},
            "mesh_config": {},
            "ports": [{}],
            "outputs": [],
            "lineage": {},  # Missing git_sha and timestamp_utc
        }

        errors = validate_m2_manifest(manifest)

        assert any("git_sha" in e for e in errors)
        assert any("timestamp_utc" in e for e in errors)

    def test_empty_ports(self) -> None:
        """Test validation catches empty ports list."""
        manifest = {
            "schema_version": 1,
            "simulation_hash": "a" * 64,
            "spec_hash": "b" * 64,
            "geometry_hash": "c" * 64,
            "design_hash": "d" * 64,
            "coupon_family": "F1_SINGLE_ENDED_VIA",
            "toolchain": {},
            "toolchain_hash": "e" * 64,
            "frequency_sweep": {},
            "excitation": {},
            "boundaries": {},
            "mesh_config": {},
            "ports": [],  # Empty
            "outputs": [],
            "lineage": {"git_sha": "f" * 40, "timestamp_utc": "2025-01-01T00:00:00Z"},
        }

        errors = validate_m2_manifest(manifest)

        assert any("at least one port" in e for e in errors)
