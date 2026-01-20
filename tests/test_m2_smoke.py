"""Smoke tests for M2 openEMS simulation subsystem.

This module provides comprehensive smoke tests covering:
- CLI argument parsing (REQ-M2-010)
- Config validation (REQ-M2-001)
- S-parameter file parsing (REQ-M2-007)
- Manifest generation (REQ-M2-018)
- Batch runner (REQ-M2-009)
- Convergence checking (REQ-M2-007/008)
- Basic end-to-end integration

Tests use pytest fixtures and mocking to avoid requiring actual openEMS installation.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# =============================================================================
# CLI Smoke Tests (REQ-M2-010)
# =============================================================================


class TestCLISmoke:
    """Smoke tests for M2 CLI interface."""

    def test_cli_module_imports(self) -> None:
        """CLI module should import without errors."""
        from formula_foundry.openems.cli_main import build_parser, main

        assert build_parser is not None
        assert main is not None

    def test_parser_builds_successfully(self) -> None:
        """Parser should build without errors."""
        from formula_foundry.openems.cli_main import build_parser

        parser = build_parser()
        assert parser is not None

    def test_parser_has_sim_command(self) -> None:
        """Parser should have 'sim' subcommand."""
        from formula_foundry.openems.cli_main import build_parser

        parser = build_parser()
        args = parser.parse_args(["sim", "run", "test.json", "--out", "output/"])
        assert args.command == "sim"

    def test_parser_has_sparam_command(self) -> None:
        """Parser should have 'sparam' subcommand."""
        from formula_foundry.openems.cli_main import build_parser

        parser = build_parser()
        args = parser.parse_args(["sparam", "extract", "sim_output/"])
        assert args.command == "sparam"

    def test_parser_has_validate_command(self) -> None:
        """Parser should have 'validate' subcommand."""
        from formula_foundry.openems.cli_main import build_parser

        parser = build_parser()
        args = parser.parse_args(["validate", "manifest.json"])
        assert args.command == "validate"

    def test_sim_run_arguments(self) -> None:
        """sim run should accept required arguments."""
        from formula_foundry.openems.cli_main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "sim", "run", "config.json",
            "--out", "/output",
            "--timeout", "1800",
            "--solver-mode", "stub",
        ])
        assert args.config == Path("config.json")
        assert args.out == Path("/output")
        assert args.timeout == 1800.0
        assert args.solver_mode == "stub"

    def test_sim_batch_arguments(self) -> None:
        """sim batch should accept required arguments."""
        from formula_foundry.openems.cli_main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "sim", "batch", "/configs",
            "--out", "/output",
            "--max-workers", "4",
        ])
        assert args.config_dir == Path("/configs")
        assert args.out == Path("/output")
        assert args.max_workers == 4


# =============================================================================
# Config Validation Smoke Tests (REQ-M2-001)
# =============================================================================


class TestConfigValidationSmoke:
    """Smoke tests for simulation config validation."""

    def test_spec_module_imports(self) -> None:
        """Spec module should import without errors."""
        from formula_foundry.openems.spec import (
            ExcitationSpec,
            FrequencySpec,
            GeometryRefSpec,
            PortSpec,
            SimulationSpec,
            ToolchainSpec,
            OpenEMSToolchainSpec,
        )

        assert SimulationSpec is not None
        assert ToolchainSpec is not None

    def test_minimal_simulation_spec_creation(self) -> None:
        """Minimal SimulationSpec should create successfully."""
        from formula_foundry.openems.spec import (
            ExcitationSpec,
            FrequencySpec,
            GeometryRefSpec,
            PortSpec,
            SimulationSpec,
            ToolchainSpec,
            OpenEMSToolchainSpec,
        )

        spec = SimulationSpec(
            schema_version=1,
            simulation_id="smoke_test_sim",
            toolchain=ToolchainSpec(
                openems=OpenEMSToolchainSpec(
                    version="0.0.35",
                    docker_image="ghcr.io/openems:0.0.35",
                )
            ),
            geometry_ref=GeometryRefSpec(design_hash="a" * 64),
            excitation=ExcitationSpec(f0_hz=5e9, fc_hz=10e9),
            frequency=FrequencySpec(f_start_hz=1e9, f_stop_hz=10e9),
            ports=[
                PortSpec(
                    id="P1",
                    type="lumped",
                    excite=True,
                    position_nm=(0, 0, 0),
                    direction="x",
                )
            ],
        )

        assert spec.simulation_id == "smoke_test_sim"
        assert spec.schema_version == 1
        assert len(spec.ports) == 1

    def test_spec_json_serialization(self) -> None:
        """SimulationSpec should serialize to JSON."""
        from formula_foundry.openems.spec import (
            ExcitationSpec,
            FrequencySpec,
            GeometryRefSpec,
            PortSpec,
            SimulationSpec,
            ToolchainSpec,
            OpenEMSToolchainSpec,
        )

        spec = SimulationSpec(
            schema_version=1,
            simulation_id="serialize_test",
            toolchain=ToolchainSpec(
                openems=OpenEMSToolchainSpec(
                    version="0.0.35",
                    docker_image="test",
                )
            ),
            geometry_ref=GeometryRefSpec(design_hash="b" * 64),
            excitation=ExcitationSpec(f0_hz=5e9, fc_hz=10e9),
            frequency=FrequencySpec(f_start_hz=1e9, f_stop_hz=10e9),
            ports=[
                PortSpec(
                    id="P1",
                    type="lumped",
                    excite=True,
                    position_nm=(0, 0, 0),
                    direction="x",
                )
            ],
        )

        # Should not raise
        json_str = spec.model_dump_json()
        assert "serialize_test" in json_str

        # Should round-trip
        data = json.loads(json_str)
        recovered = SimulationSpec.model_validate(data)
        assert recovered.simulation_id == spec.simulation_id


# =============================================================================
# S-Parameter Parsing Smoke Tests (REQ-M2-007)
# =============================================================================


class TestSParamParsingSmoke:
    """Smoke tests for S-parameter parsing."""

    def test_touchstone_module_imports(self) -> None:
        """Touchstone module should import without errors."""
        from formula_foundry.em.touchstone import (
            SParameterData,
            read_touchstone_from_string,
            write_touchstone_to_string,
        )

        assert SParameterData is not None

    def test_create_sparam_data(self) -> None:
        """SParameterData should create successfully."""
        from formula_foundry.em.touchstone import SParameterData

        freqs = np.array([1e9, 2e9, 3e9])
        s_params = np.zeros((3, 2, 2), dtype=np.complex128)
        s_params[:, 0, 0] = 0.1  # S11
        s_params[:, 1, 0] = 0.9  # S21

        data = SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=2,
        )

        assert data.n_ports == 2
        assert data.n_frequencies == 3

    def test_read_touchstone_string(self) -> None:
        """Should read touchstone from string."""
        from formula_foundry.em.touchstone import read_touchstone_from_string

        content = """# GHz S RI R 50
1.0 0.1 0.0 0.9 0.0 0.9 0.0 0.1 0.0
"""
        data = read_touchstone_from_string(content, n_ports=2)
        assert data.n_frequencies == 1
        np.testing.assert_allclose(data.frequencies_hz, [1e9])

    def test_touchstone_round_trip(self) -> None:
        """Touchstone should round-trip correctly."""
        from formula_foundry.em.touchstone import (
            SParameterData,
            read_touchstone_from_string,
            write_touchstone_to_string,
        )

        freqs = np.array([1e9, 2e9])
        s_params = np.zeros((2, 2, 2), dtype=np.complex128)
        s_params[0, 0, 0] = 0.1 + 0.1j
        s_params[1, 0, 0] = 0.2 + 0.2j
        s_params[:, 1, 0] = 0.9
        s_params[:, 0, 1] = 0.9
        s_params[:, 1, 1] = 0.1

        original = SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=2,
        )

        content = write_touchstone_to_string(original)
        recovered = read_touchstone_from_string(content, n_ports=2)

        np.testing.assert_allclose(
            recovered.frequencies_hz,
            original.frequencies_hz,
            rtol=1e-6,
        )


# =============================================================================
# Manifest Generation Smoke Tests (REQ-M2-018)
# =============================================================================


class TestManifestGenerationSmoke:
    """Smoke tests for M2 manifest generation."""

    def test_manifest_module_imports(self) -> None:
        """Manifest module should import without errors."""
        from formula_foundry.openems.manifest import (
            ConvergenceMetrics,
            M2ManifestBuilder,
            MeshStatistics,
            PortConfiguration,
            build_m2_manifest,
            validate_m2_manifest,
        )

        assert M2ManifestBuilder is not None

    def test_mesh_statistics_creation(self) -> None:
        """MeshStatistics should create successfully."""
        from formula_foundry.openems.manifest import MeshStatistics

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

        assert stats.total_cells == 1000000
        assert stats.n_lines_x == 100

        # Should serialize to dict
        d = stats.to_dict()
        assert d["total_cells"] == 1000000

    def test_port_configuration_creation(self) -> None:
        """PortConfiguration should create successfully."""
        from formula_foundry.openems.manifest import PortConfiguration

        port = PortConfiguration(
            id="P1",
            type="lumped",
            impedance_ohm=50.0,
            excite=True,
            position_nm=(5000000, 0, 100000),
            direction="x",
        )

        assert port.id == "P1"
        assert port.impedance_ohm == 50.0

        # Should serialize to dict
        d = port.to_dict()
        assert d["id"] == "P1"

    def test_validate_manifest_missing_fields(self) -> None:
        """validate_m2_manifest should catch missing fields."""
        from formula_foundry.openems.manifest import validate_m2_manifest

        # Missing required fields
        manifest = {"schema_version": 1}
        errors = validate_m2_manifest(manifest)
        assert len(errors) > 0

    def test_validate_manifest_valid(self) -> None:
        """validate_m2_manifest should accept valid manifest."""
        from formula_foundry.openems.manifest import validate_m2_manifest

        manifest = {
            "schema_version": 1,
            "simulation_hash": "a" * 64,
            "spec_hash": "b" * 64,
            "geometry_hash": "c" * 64,
            "design_hash": "d" * 64,
            "coupon_family": "F1_SINGLE_ENDED_VIA",
            "toolchain": {"openems": {"version": "0.0.35", "docker_image": "test"}},
            "toolchain_hash": "e" * 64,
            "frequency_sweep": {"f_start_hz": 1e9, "f_stop_hz": 10e9, "n_points": 101},
            "excitation": {"type": "gaussian", "f0_hz": 5e9, "fc_hz": 5e9},
            "boundaries": {
                "x_min": "PML_8",
                "x_max": "PML_8",
                "y_min": "PEC",
                "y_max": "PEC",
                "z_min": "PEC",
                "z_max": "PML_8",
            },
            "mesh_config": {
                "resolution": {
                    "lambda_resolution": 20,
                    "metal_edge_resolution_nm": 50000,
                    "via_resolution_nm": 25000,
                },
                "smoothing": {"max_ratio": 1.5, "smooth_mesh_lines": True},
            },
            "ports": [{
                "id": "P1",
                "type": "lumped",
                "impedance_ohm": 50.0,
                "excite": True,
                "position_nm": [0, 0, 0],
                "direction": "x",
                "deembed_enabled": False,
            }],
            "outputs": [],
            "lineage": {"git_sha": "f" * 40, "timestamp_utc": "2025-01-20T12:00:00Z"},
        }

        errors = validate_m2_manifest(manifest)
        assert errors == []


# =============================================================================
# Batch Runner Smoke Tests (REQ-M2-009)
# =============================================================================


class TestBatchRunnerSmoke:
    """Smoke tests for batch simulation runner."""

    def test_batch_module_imports(self) -> None:
        """Batch runner module should import without errors."""
        from formula_foundry.openems.batch_runner import (
            BatchConfig,
            BatchProgress,
            BatchResult,
            BatchSimulationRunner,
            SimulationJob,
            SimulationJobResult,
            SimulationStatus,
        )

        assert BatchSimulationRunner is not None
        assert SimulationStatus is not None

    def test_batch_config_defaults(self) -> None:
        """BatchConfig should have sensible defaults."""
        from formula_foundry.openems.batch_runner import BatchConfig

        config = BatchConfig()
        assert config.max_workers > 0
        assert config.ram_limit_gb > 0
        assert config.timeout_per_sim_sec > 0

    def test_batch_config_effective_workers(self) -> None:
        """BatchConfig should compute effective workers correctly."""
        from formula_foundry.openems.batch_runner import BatchConfig

        # RAM-limited case
        config = BatchConfig(
            max_workers=8,
            ram_limit_gb=12.0,
            ram_per_sim_gb=4.0,  # 12/4 = 3 workers
        )
        assert config.effective_max_workers == 3

    def test_batch_progress_calculations(self) -> None:
        """BatchProgress should calculate metrics correctly."""
        from formula_foundry.openems.batch_runner import BatchProgress

        progress = BatchProgress(
            total=10,
            pending=4,
            running=2,
            completed=3,
            failed=1,
        )

        assert progress.finished == 4
        assert progress.percent_complete == 40.0

    def test_simulation_status_enum(self) -> None:
        """SimulationStatus enum should have expected values."""
        from formula_foundry.openems.batch_runner import SimulationStatus

        assert SimulationStatus.PENDING is not None
        assert SimulationStatus.RUNNING is not None
        assert SimulationStatus.COMPLETED is not None
        assert SimulationStatus.FAILED is not None
        assert SimulationStatus.TIMEOUT is not None
        assert SimulationStatus.SKIPPED is not None

    def test_batch_config_validation(self) -> None:
        """BatchConfig should validate inputs."""
        from formula_foundry.openems.batch_runner import BatchConfig

        with pytest.raises(ValueError):
            BatchConfig(max_workers=0)

        with pytest.raises(ValueError):
            BatchConfig(ram_limit_gb=-1)


# =============================================================================
# Convergence Checking Smoke Tests (REQ-M2-007/008)
# =============================================================================


class TestConvergenceSmoke:
    """Smoke tests for convergence checking."""

    def test_convergence_module_imports(self) -> None:
        """Convergence module should import without errors."""
        from formula_foundry.openems.convergence import (
            ConvergenceCheckResult,
            ConvergenceConfig,
            ConvergenceReport,
            ConvergenceStatus,
            EnergyDecayData,
            check_energy_decay,
            validate_convergence,
        )

        assert ConvergenceConfig is not None
        assert ConvergenceStatus is not None

    def test_convergence_status_enum(self) -> None:
        """ConvergenceStatus enum should have expected values."""
        from formula_foundry.openems.convergence import ConvergenceStatus

        assert ConvergenceStatus.PASSED is not None
        assert ConvergenceStatus.FAILED is not None
        assert ConvergenceStatus.WARNING is not None
        assert ConvergenceStatus.SKIPPED is not None

    def test_energy_decay_data_creation(self) -> None:
        """EnergyDecayData should create successfully."""
        from formula_foundry.openems.convergence import EnergyDecayData

        time_ps = np.array([0.0, 10.0, 20.0, 30.0])
        energy_db = np.array([-10.0, -20.0, -30.0, -40.0])

        data = EnergyDecayData(time_ps=time_ps, energy_db=energy_db)

        assert data.n_timesteps == 4
        assert data.final_energy_db == -40.0
        assert data.total_time_ps == 30.0

    def test_energy_decay_check_pass(self) -> None:
        """Energy decay check should pass for converged simulation."""
        from formula_foundry.openems.convergence import (
            ConvergenceConfig,
            ConvergenceStatus,
            EnergyDecayData,
            check_energy_decay,
        )

        # Converged: reaches -55 dB
        time_ps = np.linspace(0, 200, 21)
        energy_db = np.linspace(0, -55, 21)

        data = EnergyDecayData(time_ps=time_ps, energy_db=energy_db)
        config = ConvergenceConfig(energy_decay_target_db=-50.0)

        result = check_energy_decay(data, config)
        assert result.status == ConvergenceStatus.PASSED

    def test_energy_decay_check_fail(self) -> None:
        """Energy decay check should fail for unconverged simulation."""
        from formula_foundry.openems.convergence import (
            ConvergenceConfig,
            ConvergenceStatus,
            EnergyDecayData,
            check_energy_decay,
        )

        # Unconverged: only reaches -30 dB
        time_ps = np.linspace(0, 200, 21)
        energy_db = np.linspace(0, -30, 21)

        data = EnergyDecayData(time_ps=time_ps, energy_db=energy_db)
        config = ConvergenceConfig(energy_decay_target_db=-50.0)

        result = check_energy_decay(data, config)
        assert result.status == ConvergenceStatus.FAILED

    def test_validate_convergence_all_skipped(self) -> None:
        """validate_convergence should skip all checks when no data."""
        from formula_foundry.openems.convergence import (
            ConvergenceStatus,
            validate_convergence,
        )

        report = validate_convergence()
        assert report.overall_status == ConvergenceStatus.SKIPPED

    def test_convergence_report_structure(self) -> None:
        """ConvergenceReport should have expected structure."""
        from formula_foundry.openems.convergence import (
            ConvergenceCheckResult,
            ConvergenceConfig,
            ConvergenceReport,
            ConvergenceStatus,
        )

        checks = [
            ConvergenceCheckResult(
                name="test_check",
                status=ConvergenceStatus.PASSED,
                message="OK",
                value=-55.0,
            ),
        ]

        report = ConvergenceReport(
            checks=checks,
            overall_status=ConvergenceStatus.PASSED,
            simulation_hash="abc123",
            canonical_hash="def456",
            config=ConvergenceConfig(),
        )

        assert report.n_passed == 1
        assert report.n_failed == 0
        assert report.all_passed

        # Should serialize to dict
        d = report.to_dict()
        assert d["overall_status"] == "passed"


# =============================================================================
# End-to-End Integration Smoke Tests (REQ-M2-018)
# =============================================================================


class TestEndToEndSmoke:
    """End-to-end integration smoke tests."""

    def test_simulation_runner_stub_mode(self, tmp_path: Path) -> None:
        """SimulationRunner should work in stub mode."""
        from formula_foundry.openems.sim_runner import SimulationRunner
        from formula_foundry.openems.spec import (
            ExcitationSpec,
            FrequencySpec,
            GeometryRefSpec,
            PortSpec,
            SimulationSpec,
            ToolchainSpec,
            OpenEMSToolchainSpec,
        )
        from formula_foundry.openems.geometry import (
            BoardOutlineSpec,
            DiscontinuitySpec,
            GeometrySpec,
            LayerSpec,
            StackupMaterialsSpec,
            StackupSpec,
            TransmissionLineSpec,
        )

        # Create minimal spec
        spec = SimulationSpec(
            schema_version=1,
            simulation_id="stub_test",
            toolchain=ToolchainSpec(
                openems=OpenEMSToolchainSpec(
                    version="0.0.35",
                    docker_image="ghcr.io/openems:0.0.35",
                )
            ),
            geometry_ref=GeometryRefSpec(design_hash="a" * 64),
            excitation=ExcitationSpec(f0_hz=5e9, fc_hz=10e9),
            frequency=FrequencySpec(f_start_hz=1e9, f_stop_hz=10e9),
            ports=[
                PortSpec(
                    id="P1",
                    type="lumped",
                    excite=True,
                    position_nm=(0, 0, 0),
                    direction="x",
                )
            ],
        )

        # Create minimal geometry
        geometry = GeometrySpec(
            design_hash="a" * 64,
            coupon_family="F1_SINGLE_ENDED_VIA",
            board=BoardOutlineSpec(
                width_nm=20_000_000,
                length_nm=40_000_000,
                corner_radius_nm=2_000_000,
            ),
            stackup=StackupSpec(
                copper_layers=4,
                thicknesses_nm={
                    "L1_to_L2": 180_000,
                    "L2_to_L3": 800_000,
                    "L3_to_L4": 180_000,
                },
                materials=StackupMaterialsSpec(er=4.1, loss_tangent=0.02),
            ),
            layers=[
                LayerSpec(id="L1", z_nm=0, role="signal"),
                LayerSpec(id="L2", z_nm=180_000, role="ground"),
                LayerSpec(id="L3", z_nm=980_000, role="ground"),
                LayerSpec(id="L4", z_nm=1_160_000, role="signal"),
            ],
            transmission_line=TransmissionLineSpec(
                type="CPWG",
                layer="F.Cu",
                w_nm=300_000,
                gap_nm=180_000,
                length_left_nm=10_000_000,
                length_right_nm=10_000_000,
            ),
            discontinuity=DiscontinuitySpec(
                type="VIA_TRANSITION",
                parameters_nm={
                    "signal_via.drill_nm": 300_000,
                    "signal_via.diameter_nm": 650_000,
                },
            ),
        )

        # Run simulation in stub mode
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "sim_output"

        result = runner.run(spec, geometry, output_dir=output_dir)

        assert result.output_dir == output_dir
        assert result.simulation_hash is not None
        assert len(result.simulation_hash) > 0
        assert output_dir.exists()

    def test_cli_integration_stub_mode(self, tmp_path: Path) -> None:
        """CLI should run successfully in stub mode."""
        from formula_foundry.openems.cli_main import main

        # Create config file
        config = {
            "schema_version": 1,
            "simulation_id": "cli_test",
            "toolchain": {
                "openems": {
                    "version": "0.0.35",
                    "docker_image": "ghcr.io/openems:0.0.35",
                }
            },
            "geometry_ref": {"design_hash": "a" * 64},
            "excitation": {"type": "gaussian", "f0_hz": 5e9, "fc_hz": 10e9},
            "frequency": {"f_start_hz": 1e9, "f_stop_hz": 10e9, "n_points": 101},
            "ports": [{
                "id": "P1",
                "type": "lumped",
                "impedance_ohm": 50.0,
                "excite": True,
                "position_nm": [0, 0, 0],
                "direction": "x",
            }],
        }

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        output_dir = tmp_path / "output"

        # Run CLI command
        result = main([
            "sim", "run", str(config_path),
            "--out", str(output_dir),
            "--solver-mode", "stub",
            "--no-convergence",
        ])

        assert result == 0
        assert output_dir.exists()

    def test_batch_runner_stub_mode(self, tmp_path: Path) -> None:
        """Batch runner should work in stub mode."""
        from formula_foundry.openems.batch_runner import (
            BatchConfig,
            BatchSimulationRunner,
            SimulationJob,
        )
        from formula_foundry.openems.sim_runner import SimulationRunner
        from formula_foundry.openems.spec import (
            ExcitationSpec,
            FrequencySpec,
            GeometryRefSpec,
            PortSpec,
            SimulationSpec,
            ToolchainSpec,
            OpenEMSToolchainSpec,
        )
        from formula_foundry.openems.geometry import (
            BoardOutlineSpec,
            DiscontinuitySpec,
            GeometrySpec,
            LayerSpec,
            StackupMaterialsSpec,
            StackupSpec,
            TransmissionLineSpec,
        )

        def make_spec(sim_id: str) -> SimulationSpec:
            return SimulationSpec(
                schema_version=1,
                simulation_id=sim_id,
                toolchain=ToolchainSpec(
                    openems=OpenEMSToolchainSpec(
                        version="0.0.35",
                        docker_image="ghcr.io/openems:0.0.35",
                    )
                ),
                geometry_ref=GeometryRefSpec(design_hash="a" * 64),
                excitation=ExcitationSpec(f0_hz=5e9, fc_hz=10e9),
                frequency=FrequencySpec(f_start_hz=1e9, f_stop_hz=10e9),
                ports=[
                    PortSpec(
                        id="P1",
                        type="lumped",
                        excite=True,
                        position_nm=(0, 0, 0),
                        direction="x",
                    )
                ],
            )

        def make_geometry() -> GeometrySpec:
            return GeometrySpec(
                design_hash="a" * 64,
                coupon_family="F1_SINGLE_ENDED_VIA",
                board=BoardOutlineSpec(
                    width_nm=20_000_000,
                    length_nm=40_000_000,
                    corner_radius_nm=2_000_000,
                ),
                stackup=StackupSpec(
                    copper_layers=4,
                    thicknesses_nm={
                        "L1_to_L2": 180_000,
                        "L2_to_L3": 800_000,
                        "L3_to_L4": 180_000,
                    },
                    materials=StackupMaterialsSpec(er=4.1, loss_tangent=0.02),
                ),
                layers=[
                    LayerSpec(id="L1", z_nm=0, role="signal"),
                    LayerSpec(id="L2", z_nm=180_000, role="ground"),
                    LayerSpec(id="L3", z_nm=980_000, role="ground"),
                    LayerSpec(id="L4", z_nm=1_160_000, role="signal"),
                ],
                transmission_line=TransmissionLineSpec(
                    type="CPWG",
                    layer="F.Cu",
                    w_nm=300_000,
                    gap_nm=180_000,
                    length_left_nm=10_000_000,
                    length_right_nm=10_000_000,
                ),
                discontinuity=DiscontinuitySpec(
                    type="VIA_TRANSITION",
                    parameters_nm={
                        "signal_via.drill_nm": 300_000,
                        "signal_via.diameter_nm": 650_000,
                    },
                ),
            )

        # Create stub runner
        sim_runner = SimulationRunner(mode="stub")
        config = BatchConfig(max_workers=2, validate_convergence=False)
        batch_runner = BatchSimulationRunner(sim_runner, config)

        # Create jobs
        jobs = [
            SimulationJob(
                job_id=f"batch_job_{i}",
                spec=make_spec(f"batch_sim_{i}"),
                geometry=make_geometry(),
                output_dir=tmp_path / f"batch_output_{i}",
            )
            for i in range(3)
        ]

        # Run batch
        result = batch_runner.run(jobs)

        assert result.n_completed == 3
        assert result.n_failed == 0
        assert result.all_passed


# =============================================================================
# Import Smoke Tests
# =============================================================================


class TestImportSmoke:
    """Smoke tests for module imports."""

    def test_openems_package_imports(self) -> None:
        """Main openEMS package should import without errors."""
        from formula_foundry import openems

        assert openems is not None

    def test_key_exports_available(self) -> None:
        """Key exports should be available from openems package."""
        from formula_foundry.openems import (
            # CLI
            build_parser,
            # Simulation runner
            SimulationRunner,
            SimulationResult,
            # Batch runner
            BatchConfig,
            BatchSimulationRunner,
            # Convergence
            ConvergenceConfig,
            ConvergenceReport,
            ConvergenceStatus,
            # Manifest
            M2ManifestBuilder,
            build_m2_manifest,
            validate_m2_manifest,
            # Specs
            SimulationSpec,
            GeometrySpec,
        )

        assert build_parser is not None
        assert SimulationRunner is not None
        assert BatchSimulationRunner is not None
        assert ConvergenceConfig is not None
        assert M2ManifestBuilder is not None

    def test_em_touchstone_imports(self) -> None:
        """EM touchstone module should import without errors."""
        from formula_foundry.em.touchstone import (
            SParameterData,
            read_touchstone,
            write_touchstone,
        )

        assert SParameterData is not None
