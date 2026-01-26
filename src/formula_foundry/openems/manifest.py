"""Manifest generation for M2 simulation outputs.

This module implements REQ-M2-018: Manifest generation for M2 simulation outputs.
The manifest provides complete provenance information for simulation results including:

- Simulation parameters (frequency, excitation, boundaries, mesh)
- Mesh statistics (cell counts, refinement zones, grading)
- Convergence metrics (energy decay threshold, timesteps)
- S-parameter hashes and metrics
- Toolchain versions (openEMS version, Docker image)
- Lineage information (git SHA, timestamp, M1 manifest reference)

REQ-M2-008 extensions for EM simulation provenance:
- sim_config_hash: Hash of the simulation configuration for provenance tracking
- solver_version: openEMS and CSXCAD version metadata from runner
- gpu_device_info: GPU device information if GPU acceleration was used
- sparam_file_hashes: Individual SHA256 hashes for S-parameter output files
- design_hash: Links to upstream M1 geometry manifest (already present in geometry)

The manifest follows the same canonical JSON serialization patterns as M1
to ensure deterministic hashing and provenance tracking.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from formula_foundry.substrate import canonical_json_dumps, get_git_sha, sha256_bytes

from .geometry import GeometrySpec, geometry_canonical_json
from .mesh_generator import mesh_line_summary
from .sim_runner import SimulationResult
from .sparam_extract import ExtractionResult
from .sparam_extract import build_manifest_entry as build_sparam_manifest_entry
from .spec import MeshSpec, SimulationSpec

logger = logging.getLogger(__name__)

_M2_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MeshStatistics:
    """Statistics about the simulation mesh.

    Attributes:
        total_cells: Total number of cells in the mesh.
        n_lines_x: Number of mesh lines in X direction.
        n_lines_y: Number of mesh lines in Y direction.
        n_lines_z: Number of mesh lines in Z direction.
        x_cell_min_nm: Minimum cell size in X (nm).
        x_cell_max_nm: Maximum cell size in X (nm).
        x_cell_mean_nm: Mean cell size in X (nm).
        y_cell_min_nm: Minimum cell size in Y (nm).
        y_cell_max_nm: Maximum cell size in Y (nm).
        y_cell_mean_nm: Mean cell size in Y (nm).
        z_cell_min_nm: Minimum cell size in Z (nm).
        z_cell_max_nm: Maximum cell size in Z (nm).
        z_cell_mean_nm: Mean cell size in Z (nm).
    """

    total_cells: int
    n_lines_x: int
    n_lines_y: int
    n_lines_z: int
    x_cell_min_nm: int
    x_cell_max_nm: int
    x_cell_mean_nm: float
    y_cell_min_nm: int
    y_cell_max_nm: int
    y_cell_mean_nm: float
    z_cell_min_nm: int
    z_cell_max_nm: int
    z_cell_mean_nm: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_cells": self.total_cells,
            "n_lines_x": self.n_lines_x,
            "n_lines_y": self.n_lines_y,
            "n_lines_z": self.n_lines_z,
            "x_cell_min_nm": self.x_cell_min_nm,
            "x_cell_max_nm": self.x_cell_max_nm,
            "x_cell_mean_nm": self.x_cell_mean_nm,
            "y_cell_min_nm": self.y_cell_min_nm,
            "y_cell_max_nm": self.y_cell_max_nm,
            "y_cell_mean_nm": self.y_cell_mean_nm,
            "z_cell_min_nm": self.z_cell_min_nm,
            "z_cell_max_nm": self.z_cell_max_nm,
            "z_cell_mean_nm": self.z_cell_mean_nm,
        }

    @classmethod
    def from_mesh_spec(cls, mesh_spec: MeshSpec) -> MeshStatistics:
        """Compute mesh statistics from a MeshSpec.

        Args:
            mesh_spec: Mesh specification with fixed lines.

        Returns:
            MeshStatistics with computed values.
        """
        stats = mesh_line_summary(mesh_spec)
        return cls(
            total_cells=stats["total_cells"],
            n_lines_x=stats["n_lines_x"],
            n_lines_y=stats["n_lines_y"],
            n_lines_z=stats["n_lines_z"],
            x_cell_min_nm=stats["x_cell_min_nm"],
            x_cell_max_nm=stats["x_cell_max_nm"],
            x_cell_mean_nm=stats["x_cell_mean_nm"],
            y_cell_min_nm=stats["y_cell_min_nm"],
            y_cell_max_nm=stats["y_cell_max_nm"],
            y_cell_mean_nm=stats["y_cell_mean_nm"],
            z_cell_min_nm=stats["z_cell_min_nm"],
            z_cell_max_nm=stats["z_cell_max_nm"],
            z_cell_mean_nm=stats["z_cell_mean_nm"],
        )


@dataclass(frozen=True, slots=True)
class ConvergenceMetrics:
    """Metrics related to simulation convergence.

    Attributes:
        engine_type: FDTD engine type (basic, sse, multithreaded).
        termination_criteria_db: Energy decay threshold in dB.
        max_timesteps: Maximum number of timesteps allowed.
        actual_timesteps: Actual timesteps executed (if available).
        final_energy_db: Final energy level in dB (if available).
        converged: Whether the simulation converged before max_timesteps.
    """

    engine_type: str
    termination_criteria_db: float
    max_timesteps: int
    actual_timesteps: int | None = None
    final_energy_db: float | None = None
    converged: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "engine_type": self.engine_type,
            "termination_criteria_db": self.termination_criteria_db,
            "max_timesteps": self.max_timesteps,
        }
        if self.actual_timesteps is not None:
            result["actual_timesteps"] = self.actual_timesteps
        if self.final_energy_db is not None:
            result["final_energy_db"] = self.final_energy_db
        if self.converged is not None:
            result["converged"] = self.converged
        return result

    @classmethod
    def from_spec(cls, spec: SimulationSpec) -> ConvergenceMetrics:
        """Create convergence metrics from simulation spec.

        Args:
            spec: Simulation specification.

        Returns:
            ConvergenceMetrics with spec-based values.
        """
        return cls(
            engine_type=spec.control.engine.type,
            termination_criteria_db=spec.control.termination.end_criteria_db,
            max_timesteps=spec.control.termination.max_timesteps,
        )


@dataclass(frozen=True, slots=True)
class SolverVersionInfo:
    """Solver version information for provenance tracking.

    Captures openEMS and CSXCAD version information from the solver runtime.
    This is essential for reproducibility and debugging simulation differences.

    Attributes:
        openems_version: openEMS version string (e.g., "0.0.35").
        csxcad_version: CSXCAD version string (if available).
        mode: Execution mode ("local" or "docker").
        docker_image: Docker image reference if docker mode.

    REQ-M2-008: Solver version metadata for EM simulation provenance.
    """

    openems_version: str | None
    csxcad_version: str | None = None
    mode: str = "local"
    docker_image: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "openems_version": self.openems_version,
            "mode": self.mode,
        }
        if self.csxcad_version is not None:
            result["csxcad_version"] = self.csxcad_version
        if self.docker_image is not None:
            result["docker_image"] = self.docker_image
        return result


@dataclass(frozen=True, slots=True)
class GPUDeviceInfo:
    """GPU device information for simulation provenance.

    Captures GPU device metadata when GPU acceleration is used for simulations.
    This is critical for reproducibility tracking on heterogeneous compute environments.

    Attributes:
        device_id: CUDA device ID (0-indexed).
        device_name: GPU device name (e.g., "NVIDIA A100").
        compute_capability: CUDA compute capability (e.g., "8.0").
        memory_total_mb: Total GPU memory in megabytes.
        driver_version: NVIDIA driver version.
        cuda_version: CUDA runtime version.

    REQ-M2-008: GPU device info for EM simulation provenance.
    """

    device_id: int
    device_name: str
    compute_capability: str | None = None
    memory_total_mb: int | None = None
    driver_version: str | None = None
    cuda_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "device_id": self.device_id,
            "device_name": self.device_name,
        }
        if self.compute_capability is not None:
            result["compute_capability"] = self.compute_capability
        if self.memory_total_mb is not None:
            result["memory_total_mb"] = self.memory_total_mb
        if self.driver_version is not None:
            result["driver_version"] = self.driver_version
        if self.cuda_version is not None:
            result["cuda_version"] = self.cuda_version
        return result


@dataclass(frozen=True, slots=True)
class SParamFileHashes:
    """Hashes of S-parameter output files for provenance.

    Provides individual SHA256 hashes for each S-parameter output file,
    enabling verification of file integrity and change detection.

    Attributes:
        touchstone_hash: SHA256 hash of Touchstone (.sNp) file if present.
        csv_hash: SHA256 hash of CSV export if present.
        additional_files: Dictionary of additional file paths to hashes.

    REQ-M2-008: S-parameter file hashes for EM simulation provenance.
    """

    touchstone_hash: str | None = None
    csv_hash: str | None = None
    additional_files: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {}
        if self.touchstone_hash is not None:
            result["touchstone_hash"] = self.touchstone_hash
        if self.csv_hash is not None:
            result["csv_hash"] = self.csv_hash
        if self.additional_files:
            result["additional_files"] = dict(sorted(self.additional_files.items()))
        return result


@dataclass(frozen=True, slots=True)
class PortConfiguration:
    """Configuration for a simulation port.

    Attributes:
        id: Port identifier.
        type: Port type (lumped, waveguide, msl, cpwg).
        impedance_ohm: Reference impedance in Ohms.
        excite: Whether this port is excited.
        position_nm: Port position [x, y, z] in nm.
        direction: Port excitation direction.
        deembed_enabled: Whether de-embedding is enabled.
        deembed_distance_nm: De-embedding distance if enabled.
    """

    id: str
    type: str
    impedance_ohm: float
    excite: bool
    position_nm: tuple[int, int, int]
    direction: str
    deembed_enabled: bool = False
    deembed_distance_nm: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "impedance_ohm": self.impedance_ohm,
            "excite": self.excite,
            "position_nm": list(self.position_nm),
            "direction": self.direction,
            "deembed_enabled": self.deembed_enabled,
        }
        if self.deembed_distance_nm is not None:
            result["deembed_distance_nm"] = self.deembed_distance_nm
        return result


@dataclass(slots=True)
class M2ManifestBuilder:
    """Builder for M2 simulation manifests.

    This class accumulates all the information needed to build a complete
    simulation manifest. Use the `build()` method to generate the final
    manifest dictionary.

    Attributes:
        spec: Simulation specification.
        geometry: Geometry specification.
        simulation_result: Result from simulation run.
        mesh_statistics: Computed mesh statistics.
        convergence_metrics: Convergence-related metrics.
        extraction_result: S-parameter extraction result (optional).
        m1_manifest_hash: Hash of the M1 coupon manifest (optional).
        solver_version: openEMS solver version info (REQ-M2-008).
        gpu_device_info: GPU device info if GPU used (REQ-M2-008).
        sparam_file_hashes: Hashes of S-param files (REQ-M2-008).
        git_sha: Git commit SHA (auto-detected if not provided).
        timestamp_utc: UTC timestamp (auto-generated if not provided).
    """

    spec: SimulationSpec
    geometry: GeometrySpec
    simulation_result: SimulationResult
    mesh_statistics: MeshStatistics | None = None
    convergence_metrics: ConvergenceMetrics | None = None
    extraction_result: ExtractionResult | None = None
    m1_manifest_hash: str | None = None
    solver_version: SolverVersionInfo | None = None
    gpu_device_info: GPUDeviceInfo | None = None
    sparam_file_hashes: SParamFileHashes | None = None
    git_sha: str | None = None
    timestamp_utc: str | None = None

    def with_mesh_statistics(self, stats: MeshStatistics) -> M2ManifestBuilder:
        """Add mesh statistics to the builder."""
        self.mesh_statistics = stats
        return self

    def with_mesh_spec(self, mesh_spec: MeshSpec) -> M2ManifestBuilder:
        """Compute and add mesh statistics from MeshSpec."""
        self.mesh_statistics = MeshStatistics.from_mesh_spec(mesh_spec)
        return self

    def with_convergence_metrics(self, metrics: ConvergenceMetrics) -> M2ManifestBuilder:
        """Add convergence metrics to the builder."""
        self.convergence_metrics = metrics
        return self

    def with_convergence_from_spec(self) -> M2ManifestBuilder:
        """Compute and add convergence metrics from simulation spec."""
        self.convergence_metrics = ConvergenceMetrics.from_spec(self.spec)
        return self

    def with_extraction_result(self, result: ExtractionResult) -> M2ManifestBuilder:
        """Add S-parameter extraction result to the builder."""
        self.extraction_result = result
        return self

    def with_m1_manifest(self, manifest_hash: str) -> M2ManifestBuilder:
        """Add M1 manifest hash for lineage tracking."""
        self.m1_manifest_hash = manifest_hash
        return self

    def with_solver_version(self, solver_version: SolverVersionInfo) -> M2ManifestBuilder:
        """Add solver version information (REQ-M2-008).

        Args:
            solver_version: openEMS solver version metadata.

        Returns:
            Self for method chaining.
        """
        self.solver_version = solver_version
        return self

    def with_solver_version_from_metadata(self, version_metadata: Mapping[str, Any]) -> M2ManifestBuilder:
        """Add solver version from runner version_metadata() output (REQ-M2-008).

        Args:
            version_metadata: Dictionary from OpenEMSRunner.version_metadata().

        Returns:
            Self for method chaining.
        """
        self.solver_version = SolverVersionInfo(
            openems_version=version_metadata.get("openems_version"),
            csxcad_version=version_metadata.get("csxcad_version"),
            mode=version_metadata.get("mode", "local"),
            docker_image=version_metadata.get("docker_image"),
        )
        return self

    def with_gpu_device_info(self, gpu_info: GPUDeviceInfo) -> M2ManifestBuilder:
        """Add GPU device information (REQ-M2-008).

        Args:
            gpu_info: GPU device metadata.

        Returns:
            Self for method chaining.
        """
        self.gpu_device_info = gpu_info
        return self

    def with_sparam_file_hashes(self, hashes: SParamFileHashes) -> M2ManifestBuilder:
        """Add S-parameter file hashes (REQ-M2-008).

        Args:
            hashes: S-parameter file hash information.

        Returns:
            Self for method chaining.
        """
        self.sparam_file_hashes = hashes
        return self

    def with_git_sha(self, sha: str) -> M2ManifestBuilder:
        """Set explicit git SHA."""
        self.git_sha = sha
        return self

    def with_timestamp(self, timestamp: str) -> M2ManifestBuilder:
        """Set explicit timestamp."""
        self.timestamp_utc = timestamp
        return self

    def build(self) -> dict[str, Any]:
        """Build the complete manifest dictionary.

        Returns:
            Dictionary with all manifest fields ready for JSON serialization.
        """
        # Resolve git SHA and timestamp
        resolved_git_sha = _resolve_git_sha(self.git_sha)
        resolved_timestamp = self.timestamp_utc or _utc_timestamp()

        # Build port configurations
        ports = [
            PortConfiguration(
                id=port.id,
                type=port.type,
                impedance_ohm=port.impedance_ohm,
                excite=port.excite,
                position_nm=port.position_nm,
                direction=port.direction,
                deembed_enabled=port.deembed.enabled,
                deembed_distance_nm=port.deembed.distance_nm,
            ).to_dict()
            for port in self.spec.ports
        ]

        # Build frequency sweep info
        frequency_sweep = {
            "f_start_hz": self.spec.frequency.f_start_hz,
            "f_stop_hz": self.spec.frequency.f_stop_hz,
            "n_points": self.spec.frequency.n_points,
        }

        # Build excitation info
        excitation = {
            "type": self.spec.excitation.type,
            "f0_hz": self.spec.excitation.f0_hz,
            "fc_hz": self.spec.excitation.fc_hz,
        }

        # Build boundary conditions
        boundaries = {
            "x_min": self.spec.boundaries.x_min,
            "x_max": self.spec.boundaries.x_max,
            "y_min": self.spec.boundaries.y_min,
            "y_max": self.spec.boundaries.y_max,
            "z_min": self.spec.boundaries.z_min,
            "z_max": self.spec.boundaries.z_max,
        }

        # Build mesh info
        mesh_config = {
            "resolution": {
                "lambda_resolution": self.spec.mesh.resolution.lambda_resolution,
                "metal_edge_resolution_nm": self.spec.mesh.resolution.metal_edge_resolution_nm,
                "via_resolution_nm": self.spec.mesh.resolution.via_resolution_nm,
            },
            "smoothing": {
                "max_ratio": self.spec.mesh.smoothing.max_ratio,
                "smooth_mesh_lines": self.spec.mesh.smoothing.smooth_mesh_lines,
            },
        }

        # Build output file entries
        outputs = [
            {"path": path, "hash": hash_value} for path, hash_value in sorted(self.simulation_result.output_hashes.items())
        ]

        # Compute hashes
        spec_canonical = canonical_json_dumps(self.spec.model_dump(mode="json"))
        spec_hash = sha256_bytes(spec_canonical.encode("utf-8"))

        geometry_canonical = geometry_canonical_json(self.geometry)
        geometry_hash = sha256_bytes(geometry_canonical.encode("utf-8"))

        toolchain_canonical = canonical_json_dumps(self.spec.toolchain.model_dump(mode="json"))
        toolchain_hash = sha256_bytes(toolchain_canonical.encode("utf-8"))

        # Build manifest
        manifest: dict[str, Any] = {
            "schema_version": _M2_MANIFEST_SCHEMA_VERSION,
            "simulation_id": self.spec.simulation_id,
            "simulation_hash": self.simulation_result.simulation_hash,
            "spec_hash": spec_hash,
            "geometry_hash": geometry_hash,
            # Design provenance
            "design_hash": self.geometry.design_hash,
            "coupon_family": self.geometry.coupon_family,
            # Toolchain
            "toolchain": self.spec.toolchain.model_dump(mode="json"),
            "toolchain_hash": toolchain_hash,
            # Simulation configuration
            "frequency_sweep": frequency_sweep,
            "excitation": excitation,
            "boundaries": boundaries,
            "mesh_config": mesh_config,
            "ports": ports,
            # Outputs
            "outputs": outputs,
            # Lineage
            "lineage": {
                "git_sha": resolved_git_sha,
                "timestamp_utc": resolved_timestamp,
            },
        }

        # Add optional fields
        if self.mesh_statistics is not None:
            manifest["mesh_statistics"] = self.mesh_statistics.to_dict()

        if self.convergence_metrics is not None:
            manifest["convergence"] = self.convergence_metrics.to_dict()

        if self.extraction_result is not None:
            manifest["s_parameters"] = build_sparam_manifest_entry(self.extraction_result)

        if self.m1_manifest_hash is not None:
            manifest["lineage"]["m1_manifest_hash"] = self.m1_manifest_hash

        if self.simulation_result.execution_time_sec is not None:
            manifest["execution_time_sec"] = self.simulation_result.execution_time_sec

        if self.simulation_result.sparam_path is not None:
            manifest["sparam_path"] = str(self.simulation_result.sparam_path.relative_to(self.simulation_result.output_dir))

        # REQ-M2-008: Add sim_config_hash for simulation provenance
        # This is the same as spec_hash but explicitly named for clarity
        manifest["sim_config_hash"] = spec_hash

        # REQ-M2-008: Add solver version information
        if self.solver_version is not None:
            manifest["solver_version"] = self.solver_version.to_dict()

        # REQ-M2-008: Add GPU device information if GPU was used
        if self.gpu_device_info is not None:
            manifest["gpu_device_info"] = self.gpu_device_info.to_dict()

        # REQ-M2-008: Add S-parameter file hashes
        if self.sparam_file_hashes is not None:
            sparam_hashes = self.sparam_file_hashes.to_dict()
            if sparam_hashes:  # Only add if non-empty
                manifest["sparam_file_hashes"] = sparam_hashes

        return manifest


def build_m2_manifest(
    *,
    spec: SimulationSpec,
    geometry: GeometrySpec,
    simulation_result: SimulationResult,
    mesh_spec: MeshSpec | None = None,
    extraction_result: ExtractionResult | None = None,
    m1_manifest_hash: str | None = None,
    solver_version: SolverVersionInfo | None = None,
    gpu_device_info: GPUDeviceInfo | None = None,
    sparam_file_hashes: SParamFileHashes | None = None,
    git_sha: str | None = None,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Build a complete M2 simulation manifest.

    This is the main entry point for manifest generation. It assembles all
    available information into a comprehensive manifest dictionary.

    Args:
        spec: Simulation specification.
        geometry: Geometry specification from M1.
        simulation_result: Result from simulation run.
        mesh_spec: Mesh specification for computing statistics (optional).
        extraction_result: S-parameter extraction result (optional).
        m1_manifest_hash: Hash of the M1 coupon manifest (optional).
        solver_version: openEMS solver version info (REQ-M2-008, optional).
        gpu_device_info: GPU device info if GPU used (REQ-M2-008, optional).
        sparam_file_hashes: Hashes of S-param files (REQ-M2-008, optional).
        git_sha: Git commit SHA (auto-detected if not provided).
        timestamp_utc: UTC timestamp (auto-generated if not provided).

    Returns:
        Dictionary with all manifest fields ready for JSON serialization.

    Example:
        >>> manifest = build_m2_manifest(
        ...     spec=simulation_spec,
        ...     geometry=geometry_spec,
        ...     simulation_result=result,
        ...     mesh_spec=mesh_spec,
        ...     extraction_result=extraction,
        ...     solver_version=SolverVersionInfo(openems_version="0.0.35"),
        ... )
        >>> write_m2_manifest(Path("manifest.json"), manifest)
    """
    builder = M2ManifestBuilder(
        spec=spec,
        geometry=geometry,
        simulation_result=simulation_result,
        git_sha=git_sha,
        timestamp_utc=timestamp_utc,
    )

    # Add convergence metrics from spec
    builder.with_convergence_from_spec()

    # Add mesh statistics if mesh_spec provided
    if mesh_spec is not None:
        builder.with_mesh_spec(mesh_spec)

    # Add S-parameter extraction result
    if extraction_result is not None:
        builder.with_extraction_result(extraction_result)

    # Add M1 manifest hash
    if m1_manifest_hash is not None:
        builder.with_m1_manifest(m1_manifest_hash)

    # REQ-M2-008: Add solver version info
    if solver_version is not None:
        builder.with_solver_version(solver_version)

    # REQ-M2-008: Add GPU device info
    if gpu_device_info is not None:
        builder.with_gpu_device_info(gpu_device_info)

    # REQ-M2-008: Add S-parameter file hashes
    if sparam_file_hashes is not None:
        builder.with_sparam_file_hashes(sparam_file_hashes)

    return builder.build()


def write_m2_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write manifest to file with canonical JSON formatting.

    Args:
        path: Output path for manifest file.
        manifest: Manifest dictionary to write.
    """
    text = canonical_json_dumps(dict(manifest))
    path.write_text(f"{text}\n", encoding="utf-8")
    logger.info("Wrote M2 manifest to %s", path)


def load_m2_manifest(path: Path) -> dict[str, Any]:
    """Load manifest from file.

    Args:
        path: Path to manifest file.

    Returns:
        Manifest dictionary.

    Raises:
        FileNotFoundError: If manifest file doesn't exist.
        json.JSONDecodeError: If manifest is invalid JSON.
    """
    import json

    return dict(json.loads(path.read_text(encoding="utf-8")))


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Compute SHA256 hash of a manifest.

    Args:
        manifest: Manifest dictionary.

    Returns:
        SHA256 hex digest of canonical JSON representation.
    """
    canonical = canonical_json_dumps(dict(manifest))
    return sha256_bytes(canonical.encode("utf-8"))


def validate_m2_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Validate that a manifest has required fields.

    Args:
        manifest: Manifest dictionary to validate.

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[str] = []

    required_fields = [
        "schema_version",
        "simulation_hash",
        "spec_hash",
        "geometry_hash",
        "design_hash",
        "coupon_family",
        "toolchain",
        "toolchain_hash",
        "frequency_sweep",
        "excitation",
        "boundaries",
        "mesh_config",
        "ports",
        "outputs",
        "lineage",
        "sim_config_hash",  # REQ-M2-008: Required for provenance tracking
    ]

    for field in required_fields:
        if field not in manifest:
            errors.append(f"Missing required field: {field}")

    # Validate field types
    if "schema_version" in manifest:
        if not isinstance(manifest["schema_version"], int):
            errors.append("schema_version must be an integer")
        elif manifest["schema_version"] != _M2_MANIFEST_SCHEMA_VERSION:
            errors.append(f"Unsupported schema_version: {manifest['schema_version']} (expected {_M2_MANIFEST_SCHEMA_VERSION})")

    if "simulation_hash" in manifest:
        if not isinstance(manifest["simulation_hash"], str) or len(manifest["simulation_hash"]) != 64:
            errors.append("simulation_hash must be a 64-character hex string")

    # REQ-M2-008: Validate sim_config_hash
    if "sim_config_hash" in manifest:
        if not isinstance(manifest["sim_config_hash"], str) or len(manifest["sim_config_hash"]) != 64:
            errors.append("sim_config_hash must be a 64-character hex string")

    if "lineage" in manifest:
        lineage = manifest["lineage"]
        if not isinstance(lineage, dict):
            errors.append("lineage must be a dictionary")
        else:
            if "git_sha" not in lineage:
                errors.append("lineage.git_sha is required")
            if "timestamp_utc" not in lineage:
                errors.append("lineage.timestamp_utc is required")

    if "ports" in manifest:
        if not isinstance(manifest["ports"], list):
            errors.append("ports must be a list")
        elif len(manifest["ports"]) == 0:
            errors.append("ports must contain at least one port")

    if "outputs" in manifest and not isinstance(manifest["outputs"], list):
        errors.append("outputs must be a list")

    # REQ-M2-008: Validate optional solver_version structure
    if "solver_version" in manifest:
        solver_ver = manifest["solver_version"]
        if not isinstance(solver_ver, dict):
            errors.append("solver_version must be a dictionary")
        else:
            if "mode" not in solver_ver:
                errors.append("solver_version.mode is required")

    # REQ-M2-008: Validate optional gpu_device_info structure
    if "gpu_device_info" in manifest:
        gpu_info = manifest["gpu_device_info"]
        if not isinstance(gpu_info, dict):
            errors.append("gpu_device_info must be a dictionary")
        else:
            if "device_id" not in gpu_info:
                errors.append("gpu_device_info.device_id is required")
            if "device_name" not in gpu_info:
                errors.append("gpu_device_info.device_name is required")

    # REQ-M2-008: Validate optional sparam_file_hashes structure
    if "sparam_file_hashes" in manifest:
        sparam_hashes = manifest["sparam_file_hashes"]
        if not isinstance(sparam_hashes, dict):
            errors.append("sparam_file_hashes must be a dictionary")
        else:
            # Validate hash format for touchstone and csv if present
            for key in ["touchstone_hash", "csv_hash"]:
                if key in sparam_hashes:
                    if not isinstance(sparam_hashes[key], str) or len(sparam_hashes[key]) != 64:
                        errors.append(f"sparam_file_hashes.{key} must be a 64-character hex string")

    return errors


def _utc_timestamp() -> str:
    """Generate ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_git_sha(explicit: str | None) -> str:
    """Resolve git SHA, using explicit value or detecting from environment."""
    if explicit and len(explicit) == 40:
        return explicit
    try:
        return get_git_sha(Path.cwd())
    except Exception:
        return "0" * 40
