"""SimConfig validation for Nyquist compliance and PML adequacy.

This module implements REQ-M2-003 validation requirements:
- Nyquist compliance: Ensures mesh density is sufficient for max frequency
- PML adequacy: Ensures PML layers are thick enough for proper wave absorption

The validators can be run pre-simulation to catch configuration errors early.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from formula_foundry.substrate import canonical_json_dumps, sha256_bytes

from .spec import BoundarySpec, BoundaryType, MeshSpec, SimulationSpec

logger = logging.getLogger(__name__)

# Physical constants
C0_M_PER_S = 299_792_458.0  # Speed of light in vacuum (m/s)
NM_TO_M = 1e-9  # Nanometers to meters conversion

# Default validation thresholds
DEFAULT_MIN_CELLS_PER_WAVELENGTH = 10  # Absolute minimum for Nyquist
RECOMMENDED_CELLS_PER_WAVELENGTH = 20  # Recommended for accuracy
# PML thresholds - these are for the longest wavelength in the simulation.
# FDTD PML is effective even at small fractions of a wavelength because
# it uses polynomial grading. 8 cells with good grading absorbs well.
DEFAULT_MIN_PML_WAVELENGTHS = 0.05  # PML should be at least 0.05 wavelengths
RECOMMENDED_MIN_PML_WAVELENGTHS = 0.1  # Recommended for broadband absorption


class ValidationStatus(str, Enum):
    """Status of a validation check."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of a single validation check.

    Attributes:
        name: Name of the validation check.
        status: Pass/fail/warning status.
        message: Human-readable description of result.
        value: Actual value checked (if applicable).
        threshold: Threshold used for comparison (if applicable).
        details: Additional diagnostic details.
    """

    name: str
    status: ValidationStatus
    message: str
    value: float | None = None
    threshold: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether check passed (or had warning)."""
        return self.status in (ValidationStatus.PASSED, ValidationStatus.WARNING)


@dataclass(slots=True)
class SimConfigValidationReport:
    """Complete validation report for a SimConfig.

    Attributes:
        checks: List of individual check results.
        overall_status: Overall pass/fail status.
        spec_hash: Hash of the validated spec.
        canonical_hash: SHA256 hash of canonical report.
    """

    checks: list[ValidationResult]
    overall_status: ValidationStatus
    spec_hash: str
    canonical_hash: str

    @property
    def all_passed(self) -> bool:
        """Whether all checks passed."""
        return self.overall_status == ValidationStatus.PASSED

    @property
    def n_passed(self) -> int:
        """Number of checks that passed."""
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        """Number of checks that failed."""
        return sum(1 for c in self.checks if c.status == ValidationStatus.FAILED)

    def get_check(self, name: str) -> ValidationResult | None:
        """Get a check result by name."""
        for check in self.checks:
            if check.name == name:
                return check
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "overall_status": self.overall_status.value,
            "spec_hash": self.spec_hash,
            "canonical_hash": self.canonical_hash,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "value": c.value,
                    "threshold": c.threshold,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


# =============================================================================
# Nyquist Compliance Validation
# =============================================================================


def _compute_wavelength_nm(freq_hz: float, epsilon_r: float = 1.0) -> float:
    """Compute wavelength in nanometers.

    Args:
        freq_hz: Frequency in Hz.
        epsilon_r: Relative permittivity of the medium.

    Returns:
        Wavelength in nanometers.
    """
    if freq_hz <= 0:
        raise ValueError("Frequency must be positive")
    lambda_m = C0_M_PER_S / (freq_hz * (epsilon_r**0.5))
    return lambda_m / NM_TO_M


def validate_nyquist_compliance(
    spec: SimulationSpec,
    *,
    epsilon_r: float = 4.0,
    min_cells_per_wavelength: int = DEFAULT_MIN_CELLS_PER_WAVELENGTH,
) -> ValidationResult:
    """Validate Nyquist compliance for mesh resolution.

    The Nyquist criterion requires at least 2 samples per wavelength for
    signal reconstruction, but FDTD accuracy requires significantly more.
    The typical rule is 10-20 cells per wavelength at the maximum frequency.

    Args:
        spec: Simulation specification to validate.
        epsilon_r: Relative permittivity for wavelength calculation.
            Use the highest epsilon_r in your simulation domain.
        min_cells_per_wavelength: Minimum required cells per wavelength.

    Returns:
        ValidationResult with pass/fail status.
    """
    max_freq_hz = float(spec.frequency.f_stop_hz)
    lambda_resolution = spec.mesh.resolution.lambda_resolution

    # Compute wavelength at max frequency in the dielectric
    wavelength_nm = _compute_wavelength_nm(max_freq_hz, epsilon_r)

    # Compute required cell size
    required_cell_size_nm = wavelength_nm / lambda_resolution

    # The lambda_resolution in spec is cells per wavelength
    actual_cells_per_wavelength = lambda_resolution

    # Check metal edge and via resolutions against wavelength
    metal_edge_res_nm = spec.mesh.resolution.metal_edge_resolution_nm
    via_res_nm = spec.mesh.resolution.via_resolution_nm

    # Worst case is the largest cell size
    worst_resolution_nm = max(metal_edge_res_nm, via_res_nm)
    cells_per_wavelength_worst = wavelength_nm / worst_resolution_nm

    details = {
        "max_freq_hz": max_freq_hz,
        "epsilon_r": epsilon_r,
        "wavelength_nm": wavelength_nm,
        "wavelength_um": wavelength_nm / 1000,
        "lambda_resolution": lambda_resolution,
        "required_cell_size_nm": required_cell_size_nm,
        "metal_edge_resolution_nm": metal_edge_res_nm,
        "via_resolution_nm": via_res_nm,
        "worst_resolution_nm": worst_resolution_nm,
        "cells_per_wavelength_worst_case": cells_per_wavelength_worst,
    }

    # Check if specified lambda_resolution meets minimum
    if actual_cells_per_wavelength < min_cells_per_wavelength:
        return ValidationResult(
            name="nyquist_compliance",
            status=ValidationStatus.FAILED,
            message=f"lambda_resolution {actual_cells_per_wavelength} < minimum {min_cells_per_wavelength}",
            value=float(actual_cells_per_wavelength),
            threshold=float(min_cells_per_wavelength),
            details=details,
        )

    # Check worst-case (metal edges/vias) meets minimum
    if cells_per_wavelength_worst < min_cells_per_wavelength:
        return ValidationResult(
            name="nyquist_compliance",
            status=ValidationStatus.WARNING,
            message=f"Metal edge/via resolution may be too coarse: {cells_per_wavelength_worst:.1f} cells/wavelength",
            value=cells_per_wavelength_worst,
            threshold=float(min_cells_per_wavelength),
            details=details,
        )

    # Check if meeting recommended threshold
    if actual_cells_per_wavelength < RECOMMENDED_CELLS_PER_WAVELENGTH:
        return ValidationResult(
            name="nyquist_compliance",
            status=ValidationStatus.WARNING,
            message=f"lambda_resolution {actual_cells_per_wavelength} below recommended {RECOMMENDED_CELLS_PER_WAVELENGTH}",
            value=float(actual_cells_per_wavelength),
            threshold=float(RECOMMENDED_CELLS_PER_WAVELENGTH),
            details=details,
        )

    return ValidationResult(
        name="nyquist_compliance",
        status=ValidationStatus.PASSED,
        message=f"Mesh resolution adequate: {actual_cells_per_wavelength} cells/wavelength",
        value=float(actual_cells_per_wavelength),
        threshold=float(min_cells_per_wavelength),
        details=details,
    )


# =============================================================================
# PML Adequacy Validation
# =============================================================================


def _pml_layers_from_type(boundary_type: BoundaryType) -> int:
    """Extract number of PML layers from boundary type.

    Args:
        boundary_type: Boundary condition type.

    Returns:
        Number of PML layers (0 if not PML).
    """
    if boundary_type == "PML_8":
        return 8
    elif boundary_type == "PML_16":
        return 16
    elif boundary_type == "PML_32":
        return 32
    return 0


def _estimate_pml_thickness_nm(
    n_layers: int,
    mesh_resolution_nm: float,
) -> float:
    """Estimate PML thickness in nanometers.

    The PML thickness depends on the number of layers and the mesh resolution.
    This is a rough estimate assuming uniform mesh in the PML region.

    Args:
        n_layers: Number of PML layers.
        mesh_resolution_nm: Approximate mesh cell size in nm.

    Returns:
        Estimated PML thickness in nm.
    """
    return n_layers * mesh_resolution_nm


def validate_pml_adequacy(
    spec: SimulationSpec,
    *,
    epsilon_r: float = 4.0,
    min_pml_wavelengths: float = DEFAULT_MIN_PML_WAVELENGTHS,
) -> ValidationResult:
    """Validate PML adequacy for wave absorption.

    PML (Perfectly Matched Layer) boundaries need to be thick enough to
    absorb outgoing waves without significant reflection. A typical rule
    is that PML should be at least 0.5-1.0 wavelengths thick.

    The PML needs to be adequate at the *lowest* frequency (longest wavelength).
    The cell size in the PML region is typically based on the *highest* frequency
    mesh requirements.

    Args:
        spec: Simulation specification to validate.
        epsilon_r: Relative permittivity for wavelength calculation.
        min_pml_wavelengths: Minimum required PML thickness in wavelengths.

    Returns:
        ValidationResult with pass/fail status.
    """
    boundaries = spec.boundaries
    min_freq_hz = float(spec.frequency.f_start_hz)
    max_freq_hz = float(spec.frequency.f_stop_hz)

    # Use lowest frequency for PML adequacy (longest wavelength = worst case)
    wavelength_at_fmin = _compute_wavelength_nm(min_freq_hz, epsilon_r)

    # Estimate mesh size from lambda_resolution at max frequency
    # This gives the smallest cell size the mesh will have
    wavelength_at_fmax = _compute_wavelength_nm(max_freq_hz, epsilon_r)
    min_cell_size_nm = wavelength_at_fmax / spec.mesh.resolution.lambda_resolution

    # For PML estimation, we use a more conservative (larger) cell size
    # because PML regions may use coarser mesh. A common approach is to
    # use the cell size at the geometric mean frequency.
    geometric_mean_freq = (min_freq_hz * max_freq_hz) ** 0.5
    wavelength_at_gmean = _compute_wavelength_nm(geometric_mean_freq, epsilon_r)
    estimated_cell_size_nm = wavelength_at_gmean / spec.mesh.resolution.lambda_resolution

    # Check each boundary with PML
    pml_checks: dict[str, dict[str, Any]] = {}
    min_pml_wavelength_ratio = float("inf")
    worst_boundary = ""

    for direction, bc_type in [
        ("x_min", boundaries.x_min),
        ("x_max", boundaries.x_max),
        ("y_min", boundaries.y_min),
        ("y_max", boundaries.y_max),
        ("z_min", boundaries.z_min),
        ("z_max", boundaries.z_max),
    ]:
        n_layers = _pml_layers_from_type(bc_type)
        if n_layers > 0:
            thickness_nm = _estimate_pml_thickness_nm(n_layers, estimated_cell_size_nm)
            # Compare against longest wavelength (worst case for PML absorption)
            wavelength_ratio = thickness_nm / wavelength_at_fmin
            pml_checks[direction] = {
                "type": bc_type,
                "n_layers": n_layers,
                "thickness_nm": thickness_nm,
                "wavelength_ratio": wavelength_ratio,
            }
            if wavelength_ratio < min_pml_wavelength_ratio:
                min_pml_wavelength_ratio = wavelength_ratio
                worst_boundary = direction

    details = {
        "min_freq_hz": min_freq_hz,
        "max_freq_hz": max_freq_hz,
        "epsilon_r": epsilon_r,
        "wavelength_at_fmin_nm": wavelength_at_fmin,
        "wavelength_at_fmax_nm": wavelength_at_fmax,
        "min_cell_size_nm": min_cell_size_nm,
        "estimated_cell_size_nm": estimated_cell_size_nm,
        "pml_boundaries": pml_checks,
        "worst_boundary": worst_boundary,
        "min_pml_wavelength_ratio": min_pml_wavelength_ratio if min_pml_wavelength_ratio != float("inf") else None,
    }

    # If no PML boundaries, check if that's expected (e.g., fully enclosed with PEC)
    if not pml_checks:
        return ValidationResult(
            name="pml_adequacy",
            status=ValidationStatus.WARNING,
            message="No PML boundaries configured - check if this is intentional",
            details=details,
        )

    # Check if worst-case PML meets minimum threshold
    if min_pml_wavelength_ratio < min_pml_wavelengths:
        return ValidationResult(
            name="pml_adequacy",
            status=ValidationStatus.FAILED,
            message=f"PML at {worst_boundary} too thin: {min_pml_wavelength_ratio:.2f} wavelengths < {min_pml_wavelengths}",
            value=min_pml_wavelength_ratio,
            threshold=min_pml_wavelengths,
            details=details,
        )

    # Check if meeting recommended threshold
    if min_pml_wavelength_ratio < RECOMMENDED_MIN_PML_WAVELENGTHS:
        return ValidationResult(
            name="pml_adequacy",
            status=ValidationStatus.WARNING,
            message=f"PML at {worst_boundary} below recommended: {min_pml_wavelength_ratio:.2f} wavelengths",
            value=min_pml_wavelength_ratio,
            threshold=RECOMMENDED_MIN_PML_WAVELENGTHS,
            details=details,
        )

    return ValidationResult(
        name="pml_adequacy",
        status=ValidationStatus.PASSED,
        message=f"PML adequate: minimum {min_pml_wavelength_ratio:.2f} wavelengths thick",
        value=min_pml_wavelength_ratio,
        threshold=min_pml_wavelengths,
        details=details,
    )


# =============================================================================
# GPU Configuration Validation
# =============================================================================


def validate_gpu_config(spec: SimulationSpec) -> ValidationResult:
    """Validate GPU configuration consistency.

    Checks that GPU settings are self-consistent and reasonable.

    Args:
        spec: Simulation specification to validate.

    Returns:
        ValidationResult with pass/fail status.
    """
    engine = spec.control.engine
    details = {
        "use_gpu": engine.use_gpu,
        "gpu_device_id": engine.gpu_device_id,
        "gpu_memory_fraction": engine.gpu_memory_fraction,
        "engine_type": engine.type,
    }

    # If GPU is disabled, nothing more to check
    if not engine.use_gpu:
        return ValidationResult(
            name="gpu_config",
            status=ValidationStatus.PASSED,
            message="GPU acceleration disabled, using CPU",
            details=details,
        )

    # GPU enabled - check for potential issues
    warnings = []

    # Memory fraction too low might cause issues
    if engine.gpu_memory_fraction is not None and engine.gpu_memory_fraction < 0.3:
        warnings.append(f"GPU memory fraction {engine.gpu_memory_fraction} is low, may cause issues")

    # GPU with multithreaded CPU engine is a config smell
    if engine.type == "multithreaded":
        warnings.append("GPU enabled with multithreaded CPU engine - consider dedicated GPU engine")

    if warnings:
        return ValidationResult(
            name="gpu_config",
            status=ValidationStatus.WARNING,
            message="; ".join(warnings),
            details=details,
        )

    return ValidationResult(
        name="gpu_config",
        status=ValidationStatus.PASSED,
        message=f"GPU enabled (device: {engine.gpu_device_id or 'auto'})",
        details=details,
    )


# =============================================================================
# High-Level Validation API
# =============================================================================


def validate_sim_config(
    spec: SimulationSpec,
    *,
    epsilon_r: float = 4.0,
    min_cells_per_wavelength: int = DEFAULT_MIN_CELLS_PER_WAVELENGTH,
    min_pml_wavelengths: float = DEFAULT_MIN_PML_WAVELENGTHS,
    spec_hash: str = "",
) -> SimConfigValidationReport:
    """Validate a SimulationSpec for Nyquist compliance and PML adequacy.

    This is the main entry point for pre-simulation validation. It runs
    all configured checks and produces a comprehensive report.

    Args:
        spec: Simulation specification to validate.
        epsilon_r: Relative permittivity for wavelength calculations.
        min_cells_per_wavelength: Minimum cells per wavelength for Nyquist.
        min_pml_wavelengths: Minimum PML thickness in wavelengths.
        spec_hash: Optional hash of the spec for tracking.

    Returns:
        SimConfigValidationReport with all check results.
    """
    checks: list[ValidationResult] = []

    # Nyquist compliance check
    checks.append(
        validate_nyquist_compliance(
            spec,
            epsilon_r=epsilon_r,
            min_cells_per_wavelength=min_cells_per_wavelength,
        )
    )

    # PML adequacy check
    checks.append(
        validate_pml_adequacy(
            spec,
            epsilon_r=epsilon_r,
            min_pml_wavelengths=min_pml_wavelengths,
        )
    )

    # GPU configuration check
    checks.append(validate_gpu_config(spec))

    # Determine overall status
    has_failure = any(c.status == ValidationStatus.FAILED for c in checks)
    has_warning = any(c.status == ValidationStatus.WARNING for c in checks)

    if has_failure:
        overall_status = ValidationStatus.FAILED
    elif has_warning:
        overall_status = ValidationStatus.WARNING
    else:
        overall_status = ValidationStatus.PASSED

    # Compute canonical hash of the report
    report_data = {
        "spec_hash": spec_hash,
        "checks": [
            {
                "name": c.name,
                "status": c.status.value,
                "value": c.value,
                "threshold": c.threshold,
            }
            for c in checks
        ],
    }
    canonical_hash = sha256_bytes(canonical_json_dumps(report_data).encode("utf-8"))

    return SimConfigValidationReport(
        checks=checks,
        overall_status=overall_status,
        spec_hash=spec_hash,
        canonical_hash=canonical_hash,
    )


# =============================================================================
# Report Writing
# =============================================================================


def write_validation_report(
    report: SimConfigValidationReport,
    output_path: Path,
) -> None:
    """Write validation report to JSON file.

    Args:
        report: Validation report to write.
        output_path: Path for output JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json_dumps(report.to_dict())
    output_path.write_text(f"{text}\n", encoding="utf-8")


# =============================================================================
# sim_config.json Storage
# =============================================================================


def write_sim_config_json(
    spec: SimulationSpec,
    output_dir: Path,
    *,
    validation_report: SimConfigValidationReport | None = None,
) -> Path:
    """Write sim_config.json alongside simulation outputs.

    This function writes the simulation configuration as a JSON file
    in the output directory, making it easy to inspect and reproduce
    the simulation settings.

    Args:
        spec: Simulation specification to write.
        output_dir: Output directory for the simulation.
        validation_report: Optional validation report to include.

    Returns:
        Path to the written sim_config.json file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_config_path = output_dir / "sim_config.json"

    # Build the config data
    config_data: dict[str, Any] = {
        "schema_version": spec.schema_version,
        "simulation_id": spec.simulation_id,
        "spec": spec.model_dump(mode="json"),
    }

    # Include validation report if provided
    if validation_report is not None:
        config_data["validation"] = validation_report.to_dict()

    text = canonical_json_dumps(config_data)
    sim_config_path.write_text(f"{text}\n", encoding="utf-8")

    return sim_config_path


def load_sim_config_json(sim_config_path: Path) -> dict[str, Any]:
    """Load sim_config.json from output directory.

    Args:
        sim_config_path: Path to sim_config.json file.

    Returns:
        Parsed configuration data.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    import json

    return json.loads(sim_config_path.read_text(encoding="utf-8"))
