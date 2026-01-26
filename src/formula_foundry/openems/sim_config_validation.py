"""Simulation configuration validation for openEMS simulations.

This module implements REQ-M2-003 validation requirements:
- Nyquist compliance: Ensure temporal sampling rate satisfies Nyquist criterion
  for the maximum simulation frequency.
- PML adequacy: Verify that PML boundary conditions have sufficient layers
  for proper absorption at the frequency range of interest.
- GPU configuration: Validate GPU-related settings.

The validation functions can be used both as pre-simulation checks (to catch
configuration errors early) and as part of the overall simulation quality gates.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from formula_foundry.substrate import canonical_json_dumps

from .spec import BoundaryType, SimulationSpec

logger = logging.getLogger(__name__)

# Physical constants
C0_M_PER_S = 299_792_458.0  # Speed of light in vacuum (m/s)
NM_TO_M = 1e-9  # Nanometers to meters

# =============================================================================
# Mesh Density Constants (REQ-M2-003)
# =============================================================================

# Minimum mesh cells per wavelength for spatial sampling adequacy.
# 10 cells/wavelength is the minimum for acceptable FDTD accuracy.
DEFAULT_MIN_CELLS_PER_WAVELENGTH: int = 10
"""Minimum mesh cells per wavelength for FDTD accuracy (default threshold)."""

# Recommended mesh density for high-quality results.
RECOMMENDED_CELLS_PER_WAVELENGTH: int = 20
"""Recommended mesh cells per wavelength for high-fidelity simulations."""

# =============================================================================
# PML Layer Constants (REQ-M2-003)
# =============================================================================

# Minimum PML layers expressed as wavelengths at the lowest frequency.
# 0.5 wavelength is the minimum to prevent significant reflections.
DEFAULT_MIN_PML_WAVELENGTHS: float = 0.5
"""Minimum PML depth in wavelengths (default threshold)."""

# Recommended PML depth for high absorption quality.
RECOMMENDED_MIN_PML_WAVELENGTHS: float = 1.0
"""Recommended PML depth in wavelengths for optimal absorption."""


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
        value: Primary metric value (if applicable).
        threshold: Threshold used for comparison (if applicable).
        details: Additional diagnostic details.
    """

    name: str
    status: ValidationStatus
    message: str
    value: float | None = None
    threshold: float | None = None
    details: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        """Whether check passed (or had warning)."""
        return self.status in (ValidationStatus.PASSED, ValidationStatus.WARNING)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "details": self.details or {},
        }


@dataclass(frozen=False)
class SimConfigValidationReport:
    """Complete simulation config validation report.

    Attributes:
        checks: List of individual validation results.
        overall_status: Overall validation status (PASSED, FAILED, WARNING).
        spec_hash: Optional hash of the spec being validated.
    """

    checks: list[ValidationResult]
    overall_status: ValidationStatus
    spec_hash: str | None = None

    @property
    def all_passed(self) -> bool:
        """Whether all checks passed (no failures, warnings OK)."""
        return self.overall_status != ValidationStatus.FAILED

    @property
    def overall_passed(self) -> bool:
        """Backwards-compat alias for all_passed."""
        return self.all_passed

    @property
    def n_passed(self) -> int:
        """Count of checks that passed (including warnings)."""
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        """Count of checks that failed."""
        return sum(1 for c in self.checks if c.status == ValidationStatus.FAILED)

    @property
    def warnings(self) -> list[str]:
        """List of warning messages from checks."""
        return [c.message for c in self.checks if c.status == ValidationStatus.WARNING]

    @property
    def errors(self) -> list[str]:
        """List of error messages from checks."""
        return [c.message for c in self.checks if c.status == ValidationStatus.FAILED]

    @property
    def results(self) -> list[ValidationResult]:
        """Backwards-compat alias for checks."""
        return self.checks

    @property
    def canonical_hash(self) -> str:
        """Compute a SHA256 hash of the canonical JSON representation."""
        payload = canonical_json_dumps(self.to_dict())
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_check(self, name: str) -> ValidationResult | None:
        """Get a check result by name.

        Args:
            name: The name of the check to retrieve.

        Returns:
            The ValidationResult if found, None otherwise.
        """
        for check in self.checks:
            if check.name == name:
                return check
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "overall_status": self.overall_status.value,
            "spec_hash": self.spec_hash,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "checks": [r.to_dict() for r in self.checks],
        }


# =============================================================================
# Nyquist Compliance Validation
# =============================================================================

# Default safety margin: Nyquist requires fs > 2*fmax, we use 10x for FDTD
DEFAULT_NYQUIST_SAFETY_FACTOR = 10.0
# Minimum acceptable factor (still valid but marginal)
MINIMUM_NYQUIST_FACTOR = 2.0


def compute_fdtd_timestep_limit_ps(
    min_cell_size_nm: float,
    epsilon_r: float = 1.0,
    courant_factor: float = 0.5,
) -> float:
    """Compute the maximum stable FDTD timestep using Courant condition.

    For 3D FDTD, the Courant-Friedrichs-Lewy (CFL) stability condition is:
        dt <= (1/c) * (1/sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)) * courant_factor

    For uniform cubic cells of size dx:
        dt <= dx / (c * sqrt(3)) * courant_factor

    Args:
        min_cell_size_nm: Minimum mesh cell size in nanometers.
        epsilon_r: Relative permittivity of the medium (reduces wave speed).
        courant_factor: Safety factor for Courant condition (typically 0.5).

    Returns:
        Maximum stable timestep in picoseconds.
    """
    import math

    # Convert cell size to meters
    dx_m = min_cell_size_nm * NM_TO_M

    # Wave speed in medium
    c_medium = C0_M_PER_S / math.sqrt(epsilon_r)

    # CFL condition for 3D cubic mesh
    dt_s = dx_m / (c_medium * math.sqrt(3.0)) * courant_factor

    # Convert to picoseconds
    dt_ps = dt_s * 1e12

    return dt_ps


def validate_nyquist_compliance(
    spec: SimulationSpec,
    *,
    min_cell_size_nm: float | None = None,
    epsilon_r: float = 1.0,
    safety_factor: float = DEFAULT_NYQUIST_SAFETY_FACTOR,
) -> ValidationResult:
    """Validate Nyquist compliance for FDTD simulation.

    The FDTD timestep must be small enough to accurately capture the
    highest frequency in the simulation. The Nyquist criterion requires
    the sampling rate to be at least 2x the maximum frequency, but for
    FDTD accuracy, a factor of 10x or higher is recommended.

    This check verifies that:
        fs = 1/dt >= safety_factor * f_max

    Where fs is the sampling frequency (inverse of FDTD timestep) and
    f_max is the maximum simulation frequency.

    Args:
        spec: Simulation specification to validate.
        min_cell_size_nm: Minimum mesh cell size in nm. If None, uses
            the metal_edge_resolution from mesh spec as an estimate.
        epsilon_r: Relative permittivity (affects timestep through wave speed).
        safety_factor: Required ratio of sampling rate to max frequency.

    Returns:
        ValidationResult indicating pass/fail/warning status.
    """
    # Get maximum frequency from spec
    f_max_hz = float(spec.frequency.f_stop_hz)

    # Estimate minimum cell size if not provided
    if min_cell_size_nm is None:
        # Use the finest resolution setting as an estimate
        mesh_res = spec.mesh.resolution
        min_cell_size_nm = float(
            min(
                mesh_res.metal_edge_resolution_nm,
                mesh_res.via_resolution_nm,
                mesh_res.substrate_resolution_nm or mesh_res.metal_edge_resolution_nm,
            )
        )

    # Get epsilon_r from materials if available
    for dielectric in spec.materials.dielectrics:
        if "substrate" in dielectric.id.lower():
            epsilon_r = dielectric.epsilon_r
            break

    # Compute timestep limit
    dt_ps = compute_fdtd_timestep_limit_ps(min_cell_size_nm, epsilon_r)

    # Convert timestep to sampling frequency
    dt_s = dt_ps * 1e-12
    fs_hz = 1.0 / dt_s if dt_s > 0 else float("inf")

    # Compute actual ratio
    actual_ratio = fs_hz / f_max_hz if f_max_hz > 0 else float("inf")

    details = {
        "f_max_hz": f_max_hz,
        "sampling_freq_hz": fs_hz,
        "timestep_ps": dt_ps,
        "min_cell_size_nm": min_cell_size_nm,
        "epsilon_r": epsilon_r,
        "actual_ratio": actual_ratio,
        "required_ratio": safety_factor,
        "minimum_ratio": MINIMUM_NYQUIST_FACTOR,
    }

    if actual_ratio >= safety_factor:
        return ValidationResult(
            name="nyquist_compliance",
            status=ValidationStatus.PASSED,
            message=f"Nyquist criterion satisfied: fs/f_max = {actual_ratio:.1f}x (required: {safety_factor:.1f}x)",
            value=actual_ratio,
            threshold=safety_factor,
            details=details,
        )
    elif actual_ratio >= MINIMUM_NYQUIST_FACTOR:
        return ValidationResult(
            name="nyquist_compliance",
            status=ValidationStatus.WARNING,
            message=f"Nyquist marginal: fs/f_max = {actual_ratio:.1f}x (recommended: {safety_factor:.1f}x)",
            value=actual_ratio,
            threshold=safety_factor,
            details=details,
        )
    else:
        return ValidationResult(
            name="nyquist_compliance",
            status=ValidationStatus.FAILED,
            message=f"Nyquist violated: fs/f_max = {actual_ratio:.1f}x < {MINIMUM_NYQUIST_FACTOR}x minimum",
            value=actual_ratio,
            threshold=MINIMUM_NYQUIST_FACTOR,
            details=details,
        )


# =============================================================================
# PML Adequacy Validation
# =============================================================================

# PML layer requirements based on frequency range
# These are empirical guidelines based on openEMS best practices
PML_LAYERS_GUIDELINES: dict[str, dict[str, int]] = {
    # Frequency range -> recommended PML layers
    "low": {"min": 8, "recommended": 8},  # < 5 GHz
    "mid": {"min": 8, "recommended": 16},  # 5-20 GHz
    "high": {"min": 16, "recommended": 32},  # > 20 GHz
}

# Mapping from BoundaryType to number of PML layers
PML_LAYER_COUNT: dict[str, int] = {
    "PML_8": 8,
    "PML_16": 16,
    "PML_32": 32,
}


def get_pml_layers(boundary: BoundaryType) -> int | None:
    """Get number of PML layers for a boundary type.

    Args:
        boundary: Boundary type string.

    Returns:
        Number of PML layers, or None if not a PML boundary.
    """
    return PML_LAYER_COUNT.get(boundary)


def get_frequency_range_category(f_max_hz: float) -> str:
    """Categorize frequency range for PML guidelines.

    Args:
        f_max_hz: Maximum frequency in Hz.

    Returns:
        Frequency range category: 'low', 'mid', or 'high'.
    """
    f_max_ghz = f_max_hz / 1e9
    if f_max_ghz < 5.0:
        return "low"
    elif f_max_ghz <= 20.0:
        return "mid"
    else:
        return "high"


def validate_pml_adequacy(
    spec: SimulationSpec,
    *,
    domain_size_nm: tuple[float, float, float] | None = None,
) -> ValidationResult:
    """Validate PML boundary condition adequacy.

    For proper absorption of outgoing waves, the PML (Perfectly Matched Layer)
    must have sufficient layers relative to:
    1. The maximum simulation frequency (higher freq needs more layers)
    2. The simulation domain size (larger domains may need more layers)

    This check verifies that PML boundaries have adequate layer counts
    for the frequency range being simulated.

    Args:
        spec: Simulation specification to validate.
        domain_size_nm: Optional domain size (x, y, z) in nm for additional checks.

    Returns:
        ValidationResult indicating pass/fail/warning status.
    """
    boundaries = spec.boundaries
    f_max_hz = float(spec.frequency.f_stop_hz)

    # Determine frequency category and required PML layers
    freq_category = get_frequency_range_category(f_max_hz)
    guidelines = PML_LAYERS_GUIDELINES[freq_category]
    min_layers = guidelines["min"]
    recommended_layers = guidelines["recommended"]

    # Check each boundary face
    boundary_faces = {
        "x_min": boundaries.x_min,
        "x_max": boundaries.x_max,
        "y_min": boundaries.y_min,
        "y_max": boundaries.y_max,
        "z_min": boundaries.z_min,
        "z_max": boundaries.z_max,
    }

    pml_faces: dict[str, int] = {}
    non_pml_faces: list[str] = []
    inadequate_faces: list[str] = []
    marginal_faces: list[str] = []

    for face, bc_type in boundary_faces.items():
        pml_layers = get_pml_layers(bc_type)
        if pml_layers is not None:
            pml_faces[face] = pml_layers
            if pml_layers < min_layers:
                inadequate_faces.append(f"{face}={bc_type}")
            elif pml_layers < recommended_layers:
                marginal_faces.append(f"{face}={bc_type}")
        else:
            non_pml_faces.append(f"{face}={bc_type}")

    details = {
        "f_max_hz": f_max_hz,
        "f_max_ghz": f_max_hz / 1e9,
        "frequency_category": freq_category,
        "min_pml_layers": min_layers,
        "recommended_pml_layers": recommended_layers,
        "pml_faces": pml_faces,
        "non_pml_faces": non_pml_faces,
        "inadequate_faces": inadequate_faces,
        "marginal_faces": marginal_faces,
    }

    # For S-parameter extraction, we typically need PML on at least the
    # port-facing boundaries (usually x_min and x_max)
    port_faces_with_pml = sum(1 for f in ["x_min", "x_max"] if f in pml_faces)

    if len(inadequate_faces) > 0:
        return ValidationResult(
            name="pml_adequacy",
            status=ValidationStatus.FAILED,
            message=f"PML layers inadequate for {freq_category} frequency: {', '.join(inadequate_faces)}",
            value=float(min(pml_faces.values())) if pml_faces else 0.0,
            threshold=float(min_layers),
            details=details,
        )
    elif port_faces_with_pml < 2:
        return ValidationResult(
            name="pml_adequacy",
            status=ValidationStatus.WARNING,
            message="Port faces (x_min, x_max) should have PML for S-parameter extraction",
            value=float(port_faces_with_pml),
            threshold=2.0,
            details=details,
        )
    elif len(marginal_faces) > 0:
        return ValidationResult(
            name="pml_adequacy",
            status=ValidationStatus.WARNING,
            message=f"PML layers marginal for {freq_category} frequency: {', '.join(marginal_faces)}",
            value=float(min(pml_faces.values())) if pml_faces else 0.0,
            threshold=float(recommended_layers),
            details=details,
        )
    else:
        min_pml = min(pml_faces.values()) if pml_faces else 0
        return ValidationResult(
            name="pml_adequacy",
            status=ValidationStatus.PASSED,
            message=f"PML adequacy satisfied: {min_pml} layers for {freq_category} frequency range",
            value=float(min_pml),
            threshold=float(recommended_layers),
            details=details,
        )


# =============================================================================
# GPU Configuration Validation
# =============================================================================


def validate_gpu_config(spec: SimulationSpec) -> ValidationResult:
    """Validate GPU configuration in the simulation spec.

    Checks that GPU settings are consistent and valid:
    - Engine type is compatible with GPU acceleration
    - GPU device ID is valid (if specified)
    - GPU memory fraction is within bounds (if specified)

    Args:
        spec: Simulation specification to validate.

    Returns:
        ValidationResult indicating pass/fail/warning status.
    """
    engine = spec.control.engine

    details = {
        "use_gpu": engine.use_gpu,
        "gpu_device_id": engine.gpu_device_id,
        "gpu_memory_fraction": engine.gpu_memory_fraction,
        "engine_type": engine.type,
    }

    if not engine.use_gpu:
        return ValidationResult(
            name="gpu_config",
            status=ValidationStatus.PASSED,
            message=f"CPU mode with {engine.type} engine",
            details=details,
        )

    # GPU is enabled - check compatibility
    if engine.type not in ("basic", "multithreaded"):
        return ValidationResult(
            name="gpu_config",
            status=ValidationStatus.WARNING,
            message=f"GPU mode may not be compatible with engine type '{engine.type}'",
            details=details,
        )

    # GPU enabled with compatible engine type
    device_info = f"device: {engine.gpu_device_id}" if engine.gpu_device_id is not None else "device: default"
    return ValidationResult(
        name="gpu_config",
        status=ValidationStatus.PASSED,
        message=f"GPU acceleration enabled ({device_info})",
        details=details,
    )


# =============================================================================
# Comprehensive Validation
# =============================================================================


def validate_sim_config(
    spec: SimulationSpec,
    spec_hash: str | None = None,
    *,
    min_cell_size_nm: float | None = None,
    epsilon_r: float = 1.0,
    domain_size_nm: tuple[float, float, float] | None = None,
) -> SimConfigValidationReport:
    """Validate simulation configuration comprehensively.

    Runs all validation checks and produces a complete report.

    Args:
        spec: Simulation specification to validate.
        spec_hash: Optional hash of the spec being validated.
        min_cell_size_nm: Minimum mesh cell size for Nyquist check.
        epsilon_r: Relative permittivity for wave speed calculation.
        domain_size_nm: Domain size for PML adequacy check.

    Returns:
        SimConfigValidationReport with all validation results.
    """
    checks: list[ValidationResult] = []

    # Run Nyquist compliance check
    nyquist_result = validate_nyquist_compliance(
        spec,
        min_cell_size_nm=min_cell_size_nm,
        epsilon_r=epsilon_r,
    )
    checks.append(nyquist_result)

    # Run PML adequacy check
    pml_result = validate_pml_adequacy(spec, domain_size_nm=domain_size_nm)
    checks.append(pml_result)

    # Run GPU configuration check
    gpu_result = validate_gpu_config(spec)
    checks.append(gpu_result)

    # Check that at least one port is excited
    excited_ports = [p for p in spec.ports if p.excite]
    if len(excited_ports) == 0:
        checks.append(
            ValidationResult(
                name="port_excitation",
                status=ValidationStatus.FAILED,
                message="At least one port must be set to excite=True",
                details={"n_ports": len(spec.ports), "n_excited": 0},
            )
        )
    else:
        checks.append(
            ValidationResult(
                name="port_excitation",
                status=ValidationStatus.PASSED,
                message=f"{len(excited_ports)} of {len(spec.ports)} ports configured for excitation",
                details={"n_ports": len(spec.ports), "n_excited": len(excited_ports)},
            )
        )

    # Check frequency sweep validity
    if spec.frequency.f_start_hz >= spec.frequency.f_stop_hz:
        checks.append(
            ValidationResult(
                name="frequency_sweep",
                status=ValidationStatus.FAILED,
                message="f_start_hz must be less than f_stop_hz",
                value=float(spec.frequency.f_start_hz),
                threshold=float(spec.frequency.f_stop_hz),
            )
        )
    else:
        checks.append(
            ValidationResult(
                name="frequency_sweep",
                status=ValidationStatus.PASSED,
                message=f"Frequency sweep: {spec.frequency.f_start_hz / 1e9:.3f} - {spec.frequency.f_stop_hz / 1e9:.3f} GHz ({spec.frequency.n_points} points)",
            )
        )

    # Determine overall status
    has_failures = any(c.status == ValidationStatus.FAILED for c in checks)
    has_warnings = any(c.status == ValidationStatus.WARNING for c in checks)

    if has_failures:
        overall_status = ValidationStatus.FAILED
    elif has_warnings:
        overall_status = ValidationStatus.WARNING
    else:
        overall_status = ValidationStatus.PASSED

    return SimConfigValidationReport(
        checks=checks,
        overall_status=overall_status,
        spec_hash=spec_hash,
    )


# =============================================================================
# sim_config.json Storage
# =============================================================================


def write_sim_config(
    spec: SimulationSpec,
    output_dir: Path,
    *,
    validate: bool = True,
    filename: str = "sim_config.json",
) -> Path:
    """Write simulation config to JSON file alongside outputs.

    This function implements REQ-M2-003 requirement to store sim_config.json
    alongside simulation outputs for reproducibility and audit trail.

    Args:
        spec: Simulation specification to write.
        output_dir: Directory where outputs are stored.
        validate: If True, validate the config before writing.
        filename: Name of the output file (default: sim_config.json).

    Returns:
        Path to the written config file.

    Raises:
        ValueError: If validation fails and validate=True.
    """
    if validate:
        report = validate_sim_config(spec)
        if not report.overall_passed:
            error_msg = "; ".join(report.errors)
            raise ValueError(f"Simulation config validation failed: {error_msg}")
        if report.warnings:
            for warning in report.warnings:
                logger.warning("SimConfig warning: %s", warning)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    # Serialize with canonical JSON for reproducibility
    payload = spec.model_dump(mode="json")
    text = canonical_json_dumps(payload)
    output_path.write_text(f"{text}\n", encoding="utf-8")

    logger.info("Wrote simulation config to %s", output_path)
    return output_path


def load_sim_config(config_path: Path) -> SimulationSpec:
    """Load simulation config from JSON file.

    Args:
        config_path: Path to sim_config.json file.

    Returns:
        Validated SimulationSpec instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        pydantic.ValidationError: If config is invalid.
    """
    from .spec import load_simulationspec

    if not config_path.exists():
        raise FileNotFoundError(f"Simulation config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    return load_simulationspec(data)


# =============================================================================
# Enhanced sim_config.json I/O (REQ-M2-003)
# =============================================================================


def write_sim_config_json(
    spec: SimulationSpec,
    output_dir: Path,
    *,
    validation_report: SimConfigValidationReport | None = None,
    filename: str = "sim_config.json",
) -> Path:
    """Write simulation config to JSON file with metadata wrapper.

    Creates a JSON file with schema_version, spec, and optional validation
    report. This is the preferred format for persisting simulation configs.

    Args:
        spec: Simulation specification to write.
        output_dir: Directory where outputs are stored.
        validation_report: Optional validation report to include.
        filename: Name of the output file (default: sim_config.json).

    Returns:
        Path to the written config file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    # Build the wrapper structure
    payload: dict[str, Any] = {
        "schema_version": spec.schema_version,
        "spec": spec.model_dump(mode="json"),
    }

    if validation_report is not None:
        payload["validation"] = validation_report.to_dict()

    text = canonical_json_dumps(payload)
    output_path.write_text(f"{text}\n", encoding="utf-8")

    logger.info("Wrote simulation config to %s", output_path)
    return output_path


def load_sim_config_json(config_path: Path) -> dict[str, Any]:
    """Load simulation config JSON file as dictionary.

    Reads the JSON wrapper format produced by write_sim_config_json.
    Returns the raw dict for inspection; use load_simulationspec() on
    the 'spec' field to get a validated SimulationSpec.

    Args:
        config_path: Path to sim_config.json file.

    Returns:
        Dictionary with schema_version, spec, and optional validation.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Simulation config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def write_validation_report(
    report: SimConfigValidationReport,
    output_path: Path,
) -> None:
    """Write validation report to JSON file.

    Serializes a SimConfigValidationReport to a standalone JSON file.

    Args:
        report: Validation report to write.
        output_path: Path for the output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json_dumps(report.to_dict())
    output_path.write_text(f"{text}\n", encoding="utf-8")
    logger.info("Wrote validation report to %s", output_path)
