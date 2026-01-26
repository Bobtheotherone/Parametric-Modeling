"""Tests for OracleAdapter and OpenEMSAdapter.

REQ-M2-001: Tests for OracleAdapter base class interface.
REQ-M2-002: Tests for OpenEMSAdapter implementation with mesh generation.

These tests validate:
- Abstract interface contract
- Manifest loading and validation
- Geometry reconstruction from manifest
- FDTD mesh generation with thirds-rule grading
- CSX geometry building
- Complete simulation setup workflow
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.em.mesh import FrequencyRange, create_default_mesh_config
from formula_foundry.openems.oracle_adapter import (
    OpenEMSAdapter,
    OracleAdapter,
    SimulationSetup,
    ThirdsRuleConfig,
)

# =============================================================================
# Fixtures
# =============================================================================


def _create_test_manifest() -> dict[str, Any]:
    """Create a valid test manifest."""
    return {
        "schema_version": 1,
        "coupon_family": "F1_SINGLE_ENDED_VIA",
        "design_hash": "abc123def456",
        "coupon_id": "cpn_abc123",
        "resolved_design": {
            "coupon_family": "F1_SINGLE_ENDED_VIA",
            "parameters_nm": {
                "board.outline.width_nm": 20_000_000,
                "board.outline.length_nm": 80_000_000,
                "board.outline.corner_radius_nm": 2_000_000,
                "transmission_line.w_nm": 300_000,
                "transmission_line.gap_nm": 180_000,
                "transmission_line.length_left_nm": 25_000_000,
                "transmission_line.length_right_nm": 25_000_000,
                "discontinuity.signal_via.drill_nm": 300_000,
                "discontinuity.signal_via.diameter_nm": 650_000,
                "discontinuity.signal_via.pad_diameter_nm": 900_000,
                "discontinuity.antipad.L2.r_nm": 1_200_000,
                "discontinuity.antipad.L3.r_nm": 1_100_000,
            },
            "derived_features": {"total_line_length_nm": 50_000_000},
            "dimensionless_groups": {"pitch_ratio": 0.6},
        },
        "derived_features": {"total_line_length_nm": 50_000_000},
        "dimensionless_groups": {"pitch_ratio": 0.6},
        "stackup": {
            "copper_layers": 4,
            "thicknesses_nm": {
                "L1_to_L2": 180_000,
                "L2_to_L3": 800_000,
                "L3_to_L4": 180_000,
            },
            "materials": {"er": 4.2, "loss_tangent": 0.02},
        },
        "toolchain_hash": "toolchain_hash_value",
        "verification": {
            "constraints": {"passed": True, "failed_ids": []},
            "drc": {"returncode": 0, "summary": {}},
        },
    }


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    """Create a temporary manifest file."""
    manifest = _create_test_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def adapter() -> OpenEMSAdapter:
    """Create an OpenEMSAdapter instance."""
    return OpenEMSAdapter()


# =============================================================================
# OracleAdapter Interface Tests (REQ-M2-001)
# =============================================================================


class TestOracleAdapterInterface:
    """Tests for OracleAdapter abstract interface."""

    def test_oracle_adapter_is_abstract(self) -> None:
        """OracleAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            OracleAdapter()  # type: ignore[abstract]

    def test_openems_adapter_implements_oracle(self) -> None:
        """OpenEMSAdapter is a valid OracleAdapter implementation."""
        adapter = OpenEMSAdapter()
        assert isinstance(adapter, OracleAdapter)

    def test_required_methods_exist(self, adapter: OpenEMSAdapter) -> None:
        """OpenEMSAdapter has all required abstract methods."""
        assert hasattr(adapter, "load_manifest")
        assert hasattr(adapter, "reconstruct_geometry")
        assert hasattr(adapter, "generate_mesh")
        assert hasattr(adapter, "setup_simulation")
        assert callable(adapter.load_manifest)
        assert callable(adapter.reconstruct_geometry)
        assert callable(adapter.generate_mesh)
        assert callable(adapter.setup_simulation)


# =============================================================================
# Manifest Loading Tests (REQ-M2-002)
# =============================================================================


class TestManifestLoading:
    """Tests for manifest loading and validation."""

    def test_load_manifest_success(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """Successfully loads a valid manifest."""
        manifest = adapter.load_manifest(manifest_path)
        assert manifest["schema_version"] == 1
        assert manifest["coupon_family"] == "F1_SINGLE_ENDED_VIA"
        assert manifest["design_hash"] == "abc123def456"

    def test_load_manifest_file_not_found(self, adapter: OpenEMSAdapter) -> None:
        """Raises FileNotFoundError for missing manifest."""
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            adapter.load_manifest(Path("/nonexistent/manifest.json"))

    def test_load_manifest_invalid_json(self, adapter: OpenEMSAdapter, tmp_path: Path) -> None:
        """Raises ValueError for invalid JSON."""
        path = tmp_path / "invalid.json"
        path.write_text("{ invalid json }", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            adapter.load_manifest(path)

    def test_load_manifest_missing_required_fields(self, adapter: OpenEMSAdapter, tmp_path: Path) -> None:
        """Raises ValueError for missing required fields."""
        manifest = {"schema_version": 1}  # Missing other required fields
        path = tmp_path / "incomplete.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required fields"):
            adapter.load_manifest(path)

    def test_load_manifest_missing_resolved_design(self, adapter: OpenEMSAdapter, tmp_path: Path) -> None:
        """Raises ValueError when resolved_design is missing."""
        manifest = {
            "schema_version": 1,
            "coupon_family": "F1",
            "design_hash": "abc",
            "stackup": {"copper_layers": 4, "thicknesses_nm": {}, "materials": {}},
        }
        path = tmp_path / "no_resolved.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required fields"):
            adapter.load_manifest(path)

    def test_load_manifest_missing_parameters_nm(self, adapter: OpenEMSAdapter, tmp_path: Path) -> None:
        """Raises ValueError when parameters_nm is missing."""
        manifest = {
            "schema_version": 1,
            "coupon_family": "F1",
            "design_hash": "abc",
            "resolved_design": {},  # Missing parameters_nm
            "stackup": {
                "copper_layers": 4,
                "thicknesses_nm": {"L1_to_L2": 100},
                "materials": {"er": 4.0, "loss_tangent": 0.02},
            },
        }
        path = tmp_path / "no_params.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="parameters_nm"):
            adapter.load_manifest(path)


# =============================================================================
# Geometry Reconstruction Tests (REQ-M2-002)
# =============================================================================


class TestGeometryReconstruction:
    """Tests for geometry reconstruction from manifest."""

    def test_reconstruct_geometry_success(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """Successfully reconstructs geometry from manifest."""
        manifest = adapter.load_manifest(manifest_path)
        geometry = adapter.reconstruct_geometry(manifest)

        assert geometry.coupon_family == "F1_SINGLE_ENDED_VIA"
        assert geometry.design_hash == "abc123def456"

    def test_geometry_has_board_dimensions(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """Reconstructed geometry includes board dimensions."""
        manifest = adapter.load_manifest(manifest_path)
        geometry = adapter.reconstruct_geometry(manifest)

        assert geometry.board.width_nm == 20_000_000
        assert geometry.board.length_nm == 80_000_000
        assert geometry.board.corner_radius_nm == 2_000_000

    def test_geometry_has_stackup(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """Reconstructed geometry includes stackup."""
        manifest = adapter.load_manifest(manifest_path)
        geometry = adapter.reconstruct_geometry(manifest)

        assert geometry.stackup.copper_layers == 4
        assert geometry.stackup.materials.er == 4.2
        assert geometry.stackup.materials.loss_tangent == 0.02

    def test_geometry_has_transmission_line(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """Reconstructed geometry includes transmission line params."""
        manifest = adapter.load_manifest(manifest_path)
        geometry = adapter.reconstruct_geometry(manifest)

        assert geometry.transmission_line.w_nm == 300_000
        assert geometry.transmission_line.gap_nm == 180_000
        assert geometry.transmission_line.length_left_nm == 25_000_000
        assert geometry.transmission_line.length_right_nm == 25_000_000

    def test_geometry_has_discontinuity(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """Reconstructed geometry includes discontinuity params."""
        manifest = adapter.load_manifest(manifest_path)
        geometry = adapter.reconstruct_geometry(manifest)

        assert geometry.discontinuity is not None
        assert geometry.discontinuity.type == "VIA_TRANSITION"
        assert "signal_via.drill_nm" in geometry.discontinuity.parameters_nm


# =============================================================================
# Mesh Generation Tests (REQ-M2-002)
# =============================================================================


class TestMeshGeneration:
    """Tests for FDTD mesh generation."""

    def test_generate_mesh_returns_mesh_spec(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """Mesh generation returns a MeshSpec."""
        manifest = adapter.load_manifest(manifest_path)
        geometry = adapter.reconstruct_geometry(manifest)
        mesh = adapter.generate_mesh(geometry)

        from formula_foundry.openems.spec import MeshSpec

        assert isinstance(mesh, MeshSpec)

    def test_mesh_has_lines_in_all_axes(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """Generated mesh has lines in X, Y, and Z axes."""
        manifest = adapter.load_manifest(manifest_path)
        geometry = adapter.reconstruct_geometry(manifest)
        mesh = adapter.generate_mesh(geometry)

        assert len(mesh.fixed_lines_x_nm) > 2
        assert len(mesh.fixed_lines_y_nm) > 2
        assert len(mesh.fixed_lines_z_nm) > 2

    def test_mesh_lines_are_sorted(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """Mesh lines are sorted in ascending order."""
        manifest = adapter.load_manifest(manifest_path)
        geometry = adapter.reconstruct_geometry(manifest)
        mesh = adapter.generate_mesh(geometry)

        assert mesh.fixed_lines_x_nm == sorted(mesh.fixed_lines_x_nm)
        assert mesh.fixed_lines_y_nm == sorted(mesh.fixed_lines_y_nm)
        assert mesh.fixed_lines_z_nm == sorted(mesh.fixed_lines_z_nm)

    def test_mesh_lines_are_unique(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """Mesh lines have no duplicates."""
        manifest = adapter.load_manifest(manifest_path)
        geometry = adapter.reconstruct_geometry(manifest)
        mesh = adapter.generate_mesh(geometry)

        assert len(mesh.fixed_lines_x_nm) == len(set(mesh.fixed_lines_x_nm))
        assert len(mesh.fixed_lines_y_nm) == len(set(mesh.fixed_lines_y_nm))
        assert len(mesh.fixed_lines_z_nm) == len(set(mesh.fixed_lines_z_nm))

    def test_mesh_respects_frequency_range(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """Mesh generation respects provided frequency range."""
        manifest = adapter.load_manifest(manifest_path)
        geometry = adapter.reconstruct_geometry(manifest)

        # Higher frequency -> more mesh lines
        low_freq = FrequencyRange(f_min_hz=100_000_000, f_max_hz=5_000_000_000)
        high_freq = FrequencyRange(f_min_hz=100_000_000, f_max_hz=40_000_000_000)

        mesh_low = adapter.generate_mesh(geometry, frequency_range=low_freq)
        mesh_high = adapter.generate_mesh(geometry, frequency_range=high_freq)

        # Higher frequency should require finer mesh
        assert len(mesh_high.fixed_lines_x_nm) >= len(mesh_low.fixed_lines_x_nm)


# =============================================================================
# Thirds-Rule Mesh Grading Tests (REQ-M2-002)
# =============================================================================


class TestThirdsRuleMeshGrading:
    """Tests for thirds-rule mesh grading near discontinuities."""

    def test_thirds_rule_config_defaults(self) -> None:
        """ThirdsRuleConfig has sensible defaults."""
        config = ThirdsRuleConfig()
        assert config.enabled is True
        assert config.divisions == 3
        assert config.min_cell_nm == 5_000
        assert config.max_expansion_ratio == 1.5

    def test_thirds_rule_enabled_by_default(self, adapter: OpenEMSAdapter) -> None:
        """Thirds-rule is enabled by default."""
        assert adapter.thirds_rule.enabled is True

    def test_mesh_with_thirds_rule_has_more_lines(self, manifest_path: Path) -> None:
        """Mesh with thirds-rule has more lines than without."""
        # With thirds-rule
        adapter_with = OpenEMSAdapter(thirds_rule=ThirdsRuleConfig(enabled=True))
        manifest = adapter_with.load_manifest(manifest_path)
        geometry = adapter_with.reconstruct_geometry(manifest)
        mesh_with = adapter_with.generate_mesh(geometry)

        # Without thirds-rule
        adapter_without = OpenEMSAdapter(thirds_rule=ThirdsRuleConfig(enabled=False))
        mesh_without = adapter_without.generate_mesh(geometry)

        # With thirds-rule should have at least as many lines
        assert len(mesh_with.fixed_lines_x_nm) >= len(mesh_without.fixed_lines_x_nm)

    def test_mesh_lines_at_via_position(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """Mesh has lines at and around via transition position."""
        manifest = adapter.load_manifest(manifest_path)
        geometry = adapter.reconstruct_geometry(manifest)
        mesh = adapter.generate_mesh(geometry)

        # Via is at transmission line junction
        via_x = geometry.transmission_line.length_left_nm

        # Should have mesh line at or very near via position
        x_lines = mesh.fixed_lines_x_nm
        closest = min(x_lines, key=lambda x: abs(x - via_x))
        assert abs(closest - via_x) < 100_000  # Within 100um

    def test_mesh_lines_at_trace_edges(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """Mesh has lines at trace edge positions."""
        manifest = adapter.load_manifest(manifest_path)
        geometry = adapter.reconstruct_geometry(manifest)
        mesh = adapter.generate_mesh(geometry)

        trace_half_width = geometry.transmission_line.w_nm // 2
        y_lines = mesh.fixed_lines_y_nm

        # Should have lines near trace edges
        closest_pos = min(y_lines, key=lambda y: abs(y - trace_half_width))
        closest_neg = min(y_lines, key=lambda y: abs(y + trace_half_width))

        assert abs(closest_pos - trace_half_width) < 100_000
        assert abs(closest_neg + trace_half_width) < 100_000

    def test_thirds_rule_custom_divisions(self, manifest_path: Path) -> None:
        """Custom thirds-rule divisions are respected."""
        adapter = OpenEMSAdapter(thirds_rule=ThirdsRuleConfig(enabled=True, divisions=4))
        manifest = adapter.load_manifest(manifest_path)
        geometry = adapter.reconstruct_geometry(manifest)
        mesh = adapter.generate_mesh(geometry)

        # Should still generate valid mesh
        assert len(mesh.fixed_lines_x_nm) > 2


# =============================================================================
# CSX Geometry Building Tests (REQ-M2-002)
# =============================================================================


class TestCSXGeometryBuilding:
    """Tests for CSX geometry building from manifest."""

    def test_build_csx_geometry(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """CSX geometry can be built from geometry spec."""
        manifest = adapter.load_manifest(manifest_path)
        geometry = adapter.reconstruct_geometry(manifest)
        csx = adapter.build_csx_geometry(geometry)

        from formula_foundry.openems.csx_primitives import CSXGeometry

        assert isinstance(csx, CSXGeometry)

    def test_csx_has_materials(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """CSX geometry has copper and substrate materials."""
        manifest = adapter.load_manifest(manifest_path)
        geometry = adapter.reconstruct_geometry(manifest)
        csx = adapter.build_csx_geometry(geometry)

        assert "copper" in csx.materials
        assert "substrate" in csx.materials or "air" in csx.materials

    def test_csx_has_primitives(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """CSX geometry has track and via primitives."""
        manifest = adapter.load_manifest(manifest_path)
        geometry = adapter.reconstruct_geometry(manifest)
        csx = adapter.build_csx_geometry(geometry)

        # Should have at least tracks and via
        assert len(csx.primitives) > 0


# =============================================================================
# Complete Simulation Setup Tests (REQ-M2-002)
# =============================================================================


class TestSimulationSetup:
    """Tests for complete simulation setup workflow."""

    def test_setup_simulation_returns_setup(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """setup_simulation returns a SimulationSetup."""
        setup = adapter.setup_simulation(manifest_path)
        assert isinstance(setup, SimulationSetup)

    def test_setup_has_geometry_spec(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """SimulationSetup includes geometry specification."""
        setup = adapter.setup_simulation(manifest_path)
        assert setup.geometry_spec is not None
        assert setup.geometry_spec.coupon_family == "F1_SINGLE_ENDED_VIA"

    def test_setup_has_csx_geometry(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """SimulationSetup includes CSX geometry."""
        setup = adapter.setup_simulation(manifest_path)
        assert setup.csx_geometry is not None
        assert len(setup.csx_geometry.primitives) > 0

    def test_setup_has_mesh_spec(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """SimulationSetup includes mesh specification."""
        setup = adapter.setup_simulation(manifest_path)
        assert setup.mesh_spec is not None
        assert len(setup.mesh_spec.fixed_lines_x_nm) > 0

    def test_setup_has_mesh_summary(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """SimulationSetup includes mesh summary statistics."""
        setup = adapter.setup_simulation(manifest_path)
        assert setup.mesh_summary is not None
        assert "total_cells" in setup.mesh_summary
        assert "n_lines_x" in setup.mesh_summary

    def test_setup_has_design_hash(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """SimulationSetup includes design hash from manifest."""
        setup = adapter.setup_simulation(manifest_path)
        assert setup.design_hash == "abc123def456"

    def test_setup_has_coupon_family(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """SimulationSetup includes coupon family."""
        setup = adapter.setup_simulation(manifest_path)
        assert setup.coupon_family == "F1_SINGLE_ENDED_VIA"

    def test_setup_with_custom_frequency_range(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """setup_simulation accepts custom frequency range."""
        freq_range = FrequencyRange(f_min_hz=1_000_000_000, f_max_hz=30_000_000_000)
        setup = adapter.setup_simulation(manifest_path, frequency_range=freq_range)

        # Should complete without error
        assert setup.mesh_spec is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestAdapterIntegration:
    """Integration tests for the complete adapter workflow."""

    def test_full_workflow(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """Test complete workflow from manifest to simulation setup."""
        # Step 1: Load manifest
        manifest = adapter.load_manifest(manifest_path)
        assert manifest is not None

        # Step 2: Reconstruct geometry
        geometry = adapter.reconstruct_geometry(manifest)
        assert geometry is not None

        # Step 3: Generate mesh
        mesh = adapter.generate_mesh(geometry)
        assert mesh is not None

        # Step 4: Build CSX geometry
        csx = adapter.build_csx_geometry(geometry)
        assert csx is not None

        # Step 5: Verify mesh quality
        from formula_foundry.openems.mesh_generator import mesh_line_summary

        summary = mesh_line_summary(mesh)
        assert summary["total_cells"] > 0

    def test_deterministic_output(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """Same input produces same output (determinism)."""
        setup1 = adapter.setup_simulation(manifest_path)
        setup2 = adapter.setup_simulation(manifest_path)

        # Mesh lines should be identical
        assert setup1.mesh_spec.fixed_lines_x_nm == setup2.mesh_spec.fixed_lines_x_nm
        assert setup1.mesh_spec.fixed_lines_y_nm == setup2.mesh_spec.fixed_lines_y_nm
        assert setup1.mesh_spec.fixed_lines_z_nm == setup2.mesh_spec.fixed_lines_z_nm

        # Summary should be identical
        assert setup1.mesh_summary == setup2.mesh_summary

    def test_mesh_encompasses_geometry(self, adapter: OpenEMSAdapter, manifest_path: Path) -> None:
        """Generated mesh encompasses the coupon geometry."""
        setup = adapter.setup_simulation(manifest_path)
        geometry = setup.geometry_spec
        mesh = setup.mesh_spec

        # X domain should cover board length
        assert mesh.fixed_lines_x_nm[0] <= 0
        assert mesh.fixed_lines_x_nm[-1] >= geometry.board.length_nm

        # Y domain should be symmetric around 0
        half_width = geometry.board.width_nm // 2
        assert mesh.fixed_lines_y_nm[0] <= -half_width
        assert mesh.fixed_lines_y_nm[-1] >= half_width
