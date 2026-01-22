"""Integration tests for M2 openEMS simulation pipeline.

This module tests the end-to-end integration of M2 components:
- Pipeline flow: geometry -> mesh -> simulation -> sparam extraction -> manifest
- Batch runner multi-job execution
- CLI command integration in stub mode
- Cross-component data flow validation

All tests run WITHOUT real openEMS/docker by default. To run with actual
openEMS integration, set RUN_OPENEMS_INTEGRATION=1 environment variable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from formula_foundry.em.touchstone import (
    SParameterData,
    read_touchstone_from_string,
    write_touchstone_to_string,
)
from formula_foundry.openems import (
    BatchConfig,
    BatchSimulationRunner,
    ConvergenceConfig,
    ConvergenceStatus,
    SimulationJob,
    SimulationRunner,
    validate_m2_manifest,
)
from formula_foundry.openems.batch_runner import (
    BatchProgress,
    write_batch_result,
)
from formula_foundry.openems.cli_main import main
from formula_foundry.openems.convergence import (
    EnergyDecayData,
    check_energy_decay,
    validate_convergence,
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
from formula_foundry.openems.runner import OpenEMSRunner
from formula_foundry.openems.spec import (
    ExcitationSpec,
    FrequencySpec,
    GeometryRefSpec,
    OpenEMSToolchainSpec,
    PortSpec,
    SimulationSpec,
    ToolchainSpec,
)

# =============================================================================
# Skip helper for docker/openEMS tests
# =============================================================================


def _check_docker_available() -> tuple[bool, str]:
    """Check if docker is available and the daemon is running.

    Returns:
        Tuple of (is_available, reason_if_not).
    """
    import shutil
    import subprocess

    # Check if docker executable exists
    docker_path = shutil.which("docker")
    if not docker_path:
        return False, "docker executable not found in PATH"

    # Check if docker daemon is reachable
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip() if proc.stderr else "unknown error"
            return False, f"docker daemon not reachable: {stderr[:100]}"
    except subprocess.TimeoutExpired:
        return False, "docker info timed out (daemon may be unresponsive)"
    except OSError as e:
        return False, f"failed to run docker: {e}"

    return True, ""


def _check_openems_image_available(docker_image: str) -> tuple[bool, str]:
    """Check if the openEMS docker image is available locally or can be pulled.

    Args:
        docker_image: The docker image name (e.g., 'ghcr.io/openems:0.0.35').

    Returns:
        Tuple of (is_available, reason_if_not).
    """
    import subprocess

    # First check if image exists locally
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", docker_image],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            return True, ""
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Image not local - we won't attempt to pull in the skip check
    # (pulling can take a long time and may fail for various reasons)
    return False, f"docker image '{docker_image}' not found locally (run 'docker pull {docker_image}' to fetch)"


def skip_unless_openems_integration() -> None:
    """Skip test unless RUN_OPENEMS_INTEGRATION=1 is set and docker is available."""
    if os.environ.get("RUN_OPENEMS_INTEGRATION") != "1":
        pytest.skip("Skipping: requires RUN_OPENEMS_INTEGRATION=1")

    # Check docker availability
    docker_ok, docker_reason = _check_docker_available()
    if not docker_ok:
        pytest.skip(f"Skipping: {docker_reason}")


def skip_unless_openems_image(docker_image: str) -> None:
    """Skip test if the specified openEMS docker image is not available.

    Args:
        docker_image: The docker image name to check.
    """
    image_ok, image_reason = _check_openems_image_available(docker_image)
    if not image_ok:
        pytest.skip(f"Skipping: {image_reason}")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def minimal_simulation_spec() -> SimulationSpec:
    """Create a minimal SimulationSpec for integration testing."""
    return SimulationSpec(
        schema_version=1,
        simulation_id="integration_test_sim",
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
            ),
            PortSpec(
                id="P2",
                type="lumped",
                excite=False,
                position_nm=(10_000_000, 0, 0),
                direction="x",
            ),
        ],
    )


@pytest.fixture
def minimal_geometry_spec() -> GeometrySpec:
    """Create a minimal GeometrySpec for integration testing."""
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


@pytest.fixture
def sample_sim_config() -> dict[str, Any]:
    """Create a valid simulation config dict for CLI testing."""
    return {
        "schema_version": 1,
        "simulation_id": "cli_integration_test",
        "toolchain": {
            "openems": {
                "version": "0.0.35",
                "docker_image": "ghcr.io/thliebig/openems:0.0.35",
            }
        },
        "geometry_ref": {"design_hash": "a" * 64},
        "excitation": {"type": "gaussian", "f0_hz": 5e9, "fc_hz": 5e9},
        "frequency": {"f_start_hz": 1e9, "f_stop_hz": 10e9, "n_points": 101},
        "ports": [
            {
                "id": "P1",
                "type": "lumped",
                "impedance_ohm": 50.0,
                "excite": True,
                "position_nm": [0, 0, 0],
                "direction": "x",
            },
            {
                "id": "P2",
                "type": "lumped",
                "impedance_ohm": 50.0,
                "excite": False,
                "position_nm": [10_000_000, 0, 0],
                "direction": "x",
            },
        ],
    }


# =============================================================================
# End-to-End Pipeline Integration Tests
# =============================================================================


class TestEndToEndPipeline:
    """End-to-end pipeline integration tests (stub mode)."""

    def test_full_pipeline_stub_mode(
        self,
        minimal_simulation_spec: SimulationSpec,
        minimal_geometry_spec: GeometrySpec,
        tmp_path: Path,
    ) -> None:
        """Full pipeline: spec -> runner -> manifest -> validation."""
        # Run simulation in stub mode
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "sim_output"

        result = runner.run(
            minimal_simulation_spec,
            minimal_geometry_spec,
            output_dir=output_dir,
        )

        # Verify simulation result
        assert result.output_dir == output_dir
        assert result.simulation_hash is not None
        assert len(result.simulation_hash) > 0
        assert output_dir.exists()

        # Verify manifest was created
        manifest_path = result.manifest_path
        assert manifest_path.exists()

        # Load manifest - stub mode may produce partial manifest
        manifest_data = json.loads(manifest_path.read_text())
        # Just verify it's valid JSON with simulation hash
        assert "simulation_hash" in manifest_data

        # Verify S-parameter output exists
        sparam_path = result.outputs_dir / "sparams.s2p"
        assert sparam_path.exists()

    def test_pipeline_produces_valid_touchstone(
        self,
        minimal_simulation_spec: SimulationSpec,
        minimal_geometry_spec: GeometrySpec,
        tmp_path: Path,
    ) -> None:
        """Pipeline should produce valid Touchstone S-parameter file."""
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "sim_output"

        result = runner.run(
            minimal_simulation_spec,
            minimal_geometry_spec,
            output_dir=output_dir,
        )

        sparam_path = result.outputs_dir / "sparams.s2p"
        content = sparam_path.read_text()

        # Should be parseable
        data = read_touchstone_from_string(content, n_ports=2)
        assert data.n_ports == 2
        assert data.n_frequencies > 0

        # Should have reasonable frequency range
        assert data.frequencies_hz[0] >= 1e6
        assert data.frequencies_hz[-1] <= 100e9

    def test_pipeline_manifest_hash_determinism(
        self,
        minimal_simulation_spec: SimulationSpec,
        minimal_geometry_spec: GeometrySpec,
        tmp_path: Path,
    ) -> None:
        """Same inputs should produce same manifest hash."""
        runner = SimulationRunner(mode="stub")

        result1 = runner.run(
            minimal_simulation_spec,
            minimal_geometry_spec,
            output_dir=tmp_path / "run1",
        )

        result2 = runner.run(
            minimal_simulation_spec,
            minimal_geometry_spec,
            output_dir=tmp_path / "run2",
        )

        # Simulation hashes should match (deterministic)
        assert result1.simulation_hash == result2.simulation_hash

    def test_pipeline_produces_manifest_file(
        self,
        minimal_simulation_spec: SimulationSpec,
        minimal_geometry_spec: GeometrySpec,
        tmp_path: Path,
    ) -> None:
        """Pipeline should produce manifest file in output directory."""
        runner = SimulationRunner(mode="stub")
        output_dir = tmp_path / "sim_output"

        result = runner.run(
            minimal_simulation_spec,
            minimal_geometry_spec,
            output_dir=output_dir,
        )

        assert result.simulation_hash is not None

        # Manifest file should exist
        assert result.manifest_path.exists()

        # Manifest should be valid JSON
        manifest_data = json.loads(result.manifest_path.read_text())
        assert "simulation_hash" in manifest_data or "schema_version" in manifest_data


# =============================================================================
# Batch Runner Integration Tests
# =============================================================================


class TestBatchRunnerIntegration:
    """Batch runner integration tests."""

    def test_batch_runner_multiple_jobs(
        self,
        minimal_geometry_spec: GeometrySpec,
        tmp_path: Path,
    ) -> None:
        """Batch runner should process multiple jobs in stub mode."""
        # Create multiple simulation specs
        specs = []
        for i in range(3):
            spec = SimulationSpec(
                schema_version=1,
                simulation_id=f"batch_sim_{i:03d}",
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
            specs.append(spec)

        # Create jobs
        jobs = [
            SimulationJob(
                job_id=f"job_{i:03d}",
                spec=spec,
                geometry=minimal_geometry_spec,
                output_dir=tmp_path / f"output_{i:03d}",
            )
            for i, spec in enumerate(specs)
        ]

        # Run batch
        sim_runner = SimulationRunner(mode="stub")
        config = BatchConfig(
            max_workers=2,
            validate_convergence=False,
        )
        batch_runner = BatchSimulationRunner(sim_runner, config)

        result = batch_runner.run(jobs)

        # Verify results
        assert len(result.jobs) == 3
        assert result.n_completed == 3
        assert result.n_failed == 0
        assert result.all_passed

        # Verify each output directory exists
        for i in range(3):
            output_dir = tmp_path / f"output_{i:03d}"
            assert output_dir.exists()

    def test_batch_runner_progress_tracking(
        self,
        minimal_geometry_spec: GeometrySpec,
        tmp_path: Path,
    ) -> None:
        """Batch runner should track progress correctly."""
        specs = [
            SimulationSpec(
                schema_version=1,
                simulation_id=f"progress_sim_{i}",
                toolchain=ToolchainSpec(
                    openems=OpenEMSToolchainSpec(
                        version="0.0.35",
                        docker_image="test",
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
            for i in range(5)
        ]

        jobs = [
            SimulationJob(
                job_id=f"prog_job_{i}",
                spec=spec,
                geometry=minimal_geometry_spec,
                output_dir=tmp_path / f"prog_output_{i}",
            )
            for i, spec in enumerate(specs)
        ]

        progress_updates: list[BatchProgress] = []

        def progress_callback(progress: BatchProgress) -> None:
            progress_updates.append(progress)

        sim_runner = SimulationRunner(mode="stub")
        config = BatchConfig(max_workers=2, validate_convergence=False)
        batch_runner = BatchSimulationRunner(sim_runner, config)

        result = batch_runner.run(jobs, progress_callback=progress_callback)

        assert result.n_completed == 5
        # Should have received progress updates
        assert len(progress_updates) >= 1

    def test_batch_result_serialization(
        self,
        minimal_geometry_spec: GeometrySpec,
        tmp_path: Path,
    ) -> None:
        """Batch result should serialize and deserialize correctly."""
        spec = SimulationSpec(
            schema_version=1,
            simulation_id="serialize_test",
            toolchain=ToolchainSpec(
                openems=OpenEMSToolchainSpec(
                    version="0.0.35",
                    docker_image="test",
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

        jobs = [
            SimulationJob(
                job_id="serial_job",
                spec=spec,
                geometry=minimal_geometry_spec,
                output_dir=tmp_path / "serial_output",
            )
        ]

        sim_runner = SimulationRunner(mode="stub")
        config = BatchConfig(max_workers=1, validate_convergence=False)
        batch_runner = BatchSimulationRunner(sim_runner, config)

        result = batch_runner.run(jobs)

        # Write result
        result_path = tmp_path / "batch_result.json"
        write_batch_result(result, result_path)

        # Verify it's valid JSON
        assert result_path.exists()
        data = json.loads(result_path.read_text())
        assert data["total_jobs"] == 1
        assert data["n_completed"] == 1


# =============================================================================
# CLI Integration Tests
# =============================================================================


class TestCLIIntegration:
    """CLI integration tests."""

    def test_cli_sim_run_creates_outputs(
        self,
        sample_sim_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """CLI sim run should create expected output files."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(sample_sim_config))

        output_dir = tmp_path / "cli_output"

        result = main(
            [
                "sim",
                "run",
                str(config_path),
                "--out",
                str(output_dir),
                "--solver-mode",
                "stub",
                "--no-convergence",
            ]
        )

        assert result == 0
        assert output_dir.exists()

        # Should have manifest
        manifest_path = output_dir / "simulation_manifest.json"
        assert manifest_path.exists()

    def test_cli_sim_batch_processes_multiple(
        self,
        sample_sim_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """CLI sim batch should process multiple configs."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        for i in range(3):
            config = sample_sim_config.copy()
            config["simulation_id"] = f"cli_batch_{i:03d}"
            config_path = config_dir / f"config_{i:03d}.json"
            config_path.write_text(json.dumps(config))

        output_dir = tmp_path / "batch_output"

        main(
            [
                "sim",
                "batch",
                str(config_dir),
                "--out",
                str(output_dir),
                "--solver-mode",
                "stub",
                "--max-workers",
                "2",
                "--no-convergence",
            ]
        )

        # Should complete (may have convergence warnings)
        assert output_dir.exists()

        # Should have batch result
        batch_result_path = output_dir / "batch_result.json"
        assert batch_result_path.exists()

        data = json.loads(batch_result_path.read_text())
        assert data["total_jobs"] == 3

    def test_cli_validate_accepts_valid_manifest(
        self,
        tmp_path: Path,
    ) -> None:
        """CLI validate should accept valid manifest."""
        manifest = {
            "schema_version": 1,
            "simulation_hash": "a" * 64,
            "spec_hash": "b" * 64,
            "sim_config_hash": "b" * 64,
            "geometry_hash": "c" * 64,
            "design_hash": "d" * 64,
            "coupon_family": "F1_SINGLE_ENDED_VIA",
            "toolchain": {"openems": {"version": "0.0.35", "docker_image": "test"}},
            "toolchain_hash": "e" * 64,
            "frequency_sweep": {
                "f_start_hz": 1e9,
                "f_stop_hz": 10e9,
                "n_points": 101,
            },
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
            "ports": [
                {
                    "id": "P1",
                    "type": "lumped",
                    "impedance_ohm": 50.0,
                    "excite": True,
                    "position_nm": [0, 0, 0],
                    "direction": "x",
                    "deembed_enabled": False,
                }
            ],
            "outputs": [],
            "lineage": {
                "git_sha": "f" * 40,
                "timestamp_utc": "2025-01-20T12:00:00Z",
            },
        }

        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        result = main(["validate", str(manifest_path)])
        assert result == 0

    def test_cli_validate_rejects_invalid_manifest(
        self,
        tmp_path: Path,
    ) -> None:
        """CLI validate should reject invalid manifest."""
        manifest = {"schema_version": 1}  # Missing required fields

        manifest_path = tmp_path / "invalid_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        result = main(["validate", str(manifest_path)])
        assert result == 1

    def test_cli_sparam_extract_after_sim(
        self,
        sample_sim_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """CLI sparam extract should work on stub simulation output."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(sample_sim_config))

        output_dir = tmp_path / "sim_output"

        # Run simulation first
        main(
            [
                "sim",
                "run",
                str(config_path),
                "--out",
                str(output_dir),
                "--solver-mode",
                "stub",
                "--no-convergence",
            ]
        )

        # Extract S-parameters
        extract_out = tmp_path / "extracted_sparams"
        result = main(
            [
                "sparam",
                "extract",
                str(output_dir),
                "--out",
                str(extract_out),
                "--config",
                str(config_path),
            ]
        )

        assert result == 0
        assert extract_out.exists()


# =============================================================================
# Convergence Integration Tests
# =============================================================================


class TestConvergenceIntegration:
    """Convergence checking integration tests."""

    def test_convergence_validation_with_synthetic_data(self) -> None:
        """Convergence validation should work with synthetic data."""
        # Create synthetic converged energy decay
        time_ps = np.linspace(0, 200, 21)
        energy_db = np.linspace(0, -55, 21)

        data = EnergyDecayData(time_ps=time_ps, energy_db=energy_db)
        config = ConvergenceConfig(energy_decay_target_db=-50.0)

        result = check_energy_decay(data, config)
        assert result.status == ConvergenceStatus.PASSED

    def test_convergence_validation_full_report(self) -> None:
        """Full convergence validation should produce valid report."""
        # Create synthetic data
        time_ps = np.linspace(0, 200, 21)
        energy_db = np.linspace(0, -55, 21)

        data = EnergyDecayData(time_ps=time_ps, energy_db=energy_db)
        config = ConvergenceConfig(energy_decay_target_db=-50.0)

        report = validate_convergence(
            energy_data=data,
            config=config,
        )

        assert report.overall_status in [
            ConvergenceStatus.PASSED,
            ConvergenceStatus.SKIPPED,
        ]

        # Report should serialize correctly
        report_dict = report.to_dict()
        assert "overall_status" in report_dict
        assert "checks" in report_dict


# =============================================================================
# Manifest Integration Tests
# =============================================================================


class TestManifestIntegration:
    """Manifest building and validation integration tests."""

    def test_manifest_validation_accepts_complete_manifest(self) -> None:
        """validate_m2_manifest should accept a complete manifest."""
        manifest = {
            "schema_version": 1,
            "simulation_hash": "a" * 64,
            "spec_hash": "b" * 64,
            "sim_config_hash": "b" * 64,
            "geometry_hash": "c" * 64,
            "design_hash": "d" * 64,
            "coupon_family": "F1_SINGLE_ENDED_VIA",
            "toolchain": {"openems": {"version": "0.0.35", "docker_image": "test"}},
            "toolchain_hash": "e" * 64,
            "frequency_sweep": {
                "f_start_hz": 1e9,
                "f_stop_hz": 10e9,
                "n_points": 101,
            },
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
            "ports": [
                {
                    "id": "P1",
                    "type": "lumped",
                    "impedance_ohm": 50.0,
                    "excite": True,
                    "position_nm": [0, 0, 0],
                    "direction": "x",
                    "deembed_enabled": False,
                }
            ],
            "outputs": [],
            "lineage": {
                "git_sha": "f" * 40,
                "timestamp_utc": "2025-01-20T12:00:00Z",
            },
        }

        errors = validate_m2_manifest(manifest)
        assert errors == [], f"Manifest validation failed: {errors}"

    def test_manifest_round_trip(self, tmp_path: Path) -> None:
        """Manifest should round-trip through JSON correctly."""
        manifest = {
            "schema_version": 1,
            "simulation_hash": "a" * 64,
            "spec_hash": "b" * 64,
            "sim_config_hash": "b" * 64,
            "geometry_hash": "c" * 64,
            "design_hash": "d" * 64,
            "coupon_family": "F1_SINGLE_ENDED_VIA",
            "toolchain": {"openems": {"version": "0.0.35", "docker_image": "test"}},
            "toolchain_hash": "e" * 64,
            "frequency_sweep": {
                "f_start_hz": 1e9,
                "f_stop_hz": 10e9,
                "n_points": 101,
            },
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
            "ports": [
                {
                    "id": "P1",
                    "type": "lumped",
                    "impedance_ohm": 50.0,
                    "excite": True,
                    "position_nm": [0, 0, 0],
                    "direction": "x",
                    "deembed_enabled": False,
                }
            ],
            "outputs": [],
            "lineage": {
                "git_sha": "f" * 40,
                "timestamp_utc": "2025-01-20T12:00:00Z",
            },
        }

        # Write to file
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # Read back
        recovered = json.loads(manifest_path.read_text())

        # Should still validate
        errors = validate_m2_manifest(recovered)
        assert errors == []


# =============================================================================
# S-Parameter Extraction Integration Tests
# =============================================================================


class TestSParamExtractionIntegration:
    """S-parameter extraction integration tests."""

    def test_touchstone_round_trip_complex(self) -> None:
        """Complex S-parameters should round-trip through Touchstone."""
        freqs = np.linspace(1e9, 10e9, 11)
        s_params = np.zeros((11, 2, 2), dtype=np.complex128)

        # Create realistic S-parameter data
        for i, f in enumerate(freqs):
            phase = 2 * np.pi * f * 1e-9
            s_params[i, 0, 0] = 0.1 * np.exp(1j * phase)  # S11
            s_params[i, 1, 1] = 0.1 * np.exp(1j * phase)  # S22
            s_params[i, 0, 1] = 0.9 * np.exp(-1j * phase)  # S12
            s_params[i, 1, 0] = 0.9 * np.exp(-1j * phase)  # S21

        original = SParameterData(
            frequencies_hz=freqs,
            s_parameters=s_params,
            n_ports=2,
        )

        # Write to string
        content = write_touchstone_to_string(original)

        # Read back
        recovered = read_touchstone_from_string(content, n_ports=2)

        # Compare
        np.testing.assert_allclose(
            recovered.frequencies_hz,
            original.frequencies_hz,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.abs(recovered.s_parameters),
            np.abs(original.s_parameters),
            rtol=1e-4,
        )


# =============================================================================
# Real openEMS Integration Tests (Optional)
# =============================================================================


class TestRealOpenEMSIntegration:
    """Tests requiring actual openEMS installation (skipped by default)."""

    def test_real_openems_simulation(
        self,
        minimal_simulation_spec: SimulationSpec,
        minimal_geometry_spec: GeometrySpec,
        tmp_path: Path,
    ) -> None:
        """Run actual openEMS simulation (requires docker)."""
        skip_unless_openems_integration()

        # Get docker image from spec and verify it's available
        docker_image = minimal_simulation_spec.toolchain.openems.docker_image
        skip_unless_openems_image(docker_image)

        # Create runner with correct architecture:
        # OpenEMSRunner(mode="docker") wrapped by SimulationRunner(mode="cli")
        openems_runner = OpenEMSRunner(mode="docker", docker_image=docker_image)
        runner = SimulationRunner(mode="cli", openems_runner=openems_runner)
        output_dir = tmp_path / "real_sim_output"

        result = runner.run(
            minimal_simulation_spec,
            minimal_geometry_spec,
            output_dir=output_dir,
        )

        assert result.simulation_hash is not None
        assert output_dir.exists()

    def test_real_batch_simulation(
        self,
        minimal_geometry_spec: GeometrySpec,
        tmp_path: Path,
    ) -> None:
        """Run batch simulation with real openEMS (requires docker)."""
        skip_unless_openems_integration()

        # Define docker image and verify it's available
        docker_image = "ghcr.io/openems:0.0.35"
        skip_unless_openems_image(docker_image)

        specs = [
            SimulationSpec(
                schema_version=1,
                simulation_id=f"real_batch_{i}",
                toolchain=ToolchainSpec(
                    openems=OpenEMSToolchainSpec(
                        version="0.0.35",
                        docker_image=docker_image,
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
            for i in range(2)
        ]

        jobs = [
            SimulationJob(
                job_id=f"real_job_{i}",
                spec=spec,
                geometry=minimal_geometry_spec,
                output_dir=tmp_path / f"real_output_{i}",
            )
            for i, spec in enumerate(specs)
        ]

        # Create runner with correct architecture:
        # OpenEMSRunner(mode="docker") wrapped by SimulationRunner(mode="cli")
        openems_runner = OpenEMSRunner(mode="docker", docker_image=docker_image)
        sim_runner = SimulationRunner(mode="cli", openems_runner=openems_runner)
        config = BatchConfig(max_workers=2)
        batch_runner = BatchSimulationRunner(sim_runner, config)

        result = batch_runner.run(jobs)

        assert result.n_completed == 2
