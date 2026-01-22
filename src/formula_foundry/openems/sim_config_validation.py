"""Simulation configuration validation for openEMS simulations.

This module implements REQ-M2-003 validation requirements:
- Nyquist compliance: Ensure temporal sampling rate satisfies Nyquist criterion
  for the maximum simulation frequency.
- PML adequacy: Verify that PML boundary conditions have sufficient layers
  for proper absorption at the frequency range of interest.

The validation functions can be used both as pre-simulation checks (to catch
configuration errors early) and as part of the overall simulation quality gates.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from formula_foundry.substrate import canonical_json_dumps

from .spec import BoundarySpec, BoundaryType, FrequencySpec, MeshSpec, SimulationSpec

logger = logging.getLogger(__name__)

# Physical constants
C0_M_PER_S = 299_792_458.0  # Speed of light in vacuum (m/s)
NM_TO_M = 1e-9  # Nanometers to meters


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


@dataclass(frozen=True, slots=True)
class SimConfigValidationReport:
    """Complete simulation config validation report.

    Attributes:
        results: List of individual validation results.
        overall_passed: Whether all validations passed.
        warnings: List of warning messages.
        errors: List of error messages.
    """

    results: list[ValidationResult]
    overall_passed: bool
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "overall_passed": self.overall_passed,
            "warnings": self.warnings,
            "errors": self.errors,
            "results": [r.to_dict() for r in self.results],
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
            message=f"Port faces (x_min, x_max) should have PML for S-parameter extraction",
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
# Comprehensive Validation
# =============================================================================


def validate_sim_config(
    spec: SimulationSpec,
    *,
    min_cell_size_nm: float | None = None,
    epsilon_r: float = 1.0,
    domain_size_nm: tuple[float, float, float] | None = None,
) -> SimConfigValidationReport:
    """Validate simulation configuration comprehensively.

    Runs all validation checks and produces a complete report.

    Args:
        spec: Simulation specification to validate.
        min_cell_size_nm: Minimum mesh cell size for Nyquist check.
        epsilon_r: Relative permittivity for wave speed calculation.
        domain_size_nm: Domain size for PML adequacy check.

    Returns:
        SimConfigValidationReport with all validation results.
    """
    results: list[ValidationResult] = []
    warnings: list[str] = []
    errors: list[str] = []

    # Run Nyquist compliance check
    nyquist_result = validate_nyquist_compliance(
        spec,
        min_cell_size_nm=min_cell_size_nm,
        epsilon_r=epsilon_r,
    )
    results.append(nyquist_result)
    if nyquist_result.status == ValidationStatus.WARNING:
        warnings.append(nyquist_result.message)
    elif nyquist_result.status == ValidationStatus.FAILED:
        errors.append(nyquist_result.message)

    # Run PML adequacy check
    pml_result = validate_pml_adequacy(spec, domain_size_nm=domain_size_nm)
    results.append(pml_result)
    if pml_result.status == ValidationStatus.WARNING:
        warnings.append(pml_result.message)
    elif pml_result.status == ValidationStatus.FAILED:
        errors.append(pml_result.message)

    # Check that at least one port is excited
    excited_ports = [p for p in spec.ports if p.excite]
    if len(excited_ports) == 0:
        no_excite_result = ValidationResult(
            name="port_excitation",
            status=ValidationStatus.FAILED,
            message="At least one port must be set to excite=True",
            details={"n_ports": len(spec.ports), "n_excited": 0},
        )
        results.append(no_excite_result)
        errors.append(no_excite_result.message)
    else:
        results.append(
            ValidationResult(
                name="port_excitation",
                status=ValidationStatus.PASSED,
                message=f"{len(excited_ports)} of {len(spec.ports)} ports configured for excitation",
                details={"n_ports": len(spec.ports), "n_excited": len(excited_ports)},
            )
        )

    # Check frequency sweep validity
    if spec.frequency.f_start_hz >= spec.frequency.f_stop_hz:
        freq_result = ValidationResult(
            name="frequency_sweep",
            status=ValidationStatus.FAILED,
            message="f_start_hz must be less than f_stop_hz",
            value=float(spec.frequency.f_start_hz),
            threshold=float(spec.frequency.f_stop_hz),
        )
        results.append(freq_result)
        errors.append(freq_result.message)
    else:
        results.append(
            ValidationResult(
                name="frequency_sweep",
                status=ValidationStatus.PASSED,
                message=f"Frequency sweep: {spec.frequency.f_start_hz/1e9:.3f} - {spec.frequency.f_stop_hz/1e9:.3f} GHz ({spec.frequency.n_points} points)",
            )
        )

    # Check GPU configuration consistency
    if spec.control.engine.use_gpu:
        if spec.control.engine.type not in ("basic", "multithreaded"):
            gpu_result = ValidationResult(
                name="gpu_configuration",
                status=ValidationStatus.WARNING,
                message=f"GPU mode may not be compatible with engine type '{spec.control.engine.type}'",
                details={"engine_type": spec.control.engine.type, "use_gpu": True},
            )
            results.append(gpu_result)
            warnings.append(gpu_result.message)
        else:
            results.append(
                ValidationResult(
                    name="gpu_configuration",
                    status=ValidationStatus.PASSED,
                    message=f"GPU acceleration enabled (device: {spec.control.engine.gpu_device_id or 'default'})",
                )
            )
    else:
        results.append(
            ValidationResult(
                name="gpu_configuration",
                status=ValidationStatus.PASSED,
                message=f"CPU mode with {spec.control.engine.type} engine",
            )
        )

    overall_passed = len(errors) == 0

    return SimConfigValidationReport(
        results=results,
        overall_passed=overall_passed,
        warnings=warnings,
        errors=errors,
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
