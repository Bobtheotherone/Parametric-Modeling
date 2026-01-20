"""Convergence monitoring and validation for openEMS simulations.

This module implements REQ-M2-008: Convergence checking for openEMS
FDTD simulations, providing pass/fail gates and diagnostic output.

Key convergence metrics:
- Energy decay: Verify simulation energy has decayed to target threshold
- Port power balance: Ensure power conservation (passive device check)
- Frequency resolution adequacy: Validate mesh density for max frequency

The convergence pipeline:
1. Load simulation outputs (energy decay, S-parameters, mesh info)
2. Evaluate each convergence criterion
3. Generate diagnostic report with pass/fail gates
4. Produce structured output for manifest inclusion
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from formula_foundry.em.touchstone import SParameterData
from formula_foundry.substrate import canonical_json_dumps, sha256_bytes

from .spec import (
    FrequencySpec,
    MeshResolutionSpec,
    SimulationSpec,
    TerminationSpec,
)

logger = logging.getLogger(__name__)

# Type aliases
FloatArray = NDArray[np.float64]

# Physical constants
C0_M_PER_S = 299_792_458.0  # Speed of light in m/s
NM_TO_M = 1e-9

# Default thresholds
DEFAULT_ENERGY_DECAY_MARGIN_DB = 3.0  # Must reach target - margin
DEFAULT_PASSIVITY_TOLERANCE = 1e-6  # Tolerance for eigenvalue > 1
DEFAULT_POWER_BALANCE_TOLERANCE = 0.1  # 10% tolerance for power balance
DEFAULT_MIN_CELLS_PER_WAVELENGTH = 10  # Minimum mesh density


class ConvergenceStatus(str, Enum):
    """Status of a convergence check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"  # When data not available
    WARNING = "warning"  # Passed with concerns


@dataclass(frozen=True, slots=True)
class EnergyDecayData:
    """Energy decay time series data.

    Attributes:
        time_ps: Time array in picoseconds.
        energy_db: Energy decay in dB (relative to initial).
    """

    time_ps: FloatArray
    energy_db: FloatArray

    def __post_init__(self) -> None:
        if len(self.time_ps) != len(self.energy_db):
            raise ValueError("time_ps and energy_db must have same length")

    @property
    def final_energy_db(self) -> float:
        """Final energy level in dB."""
        return float(self.energy_db[-1]) if len(self.energy_db) > 0 else 0.0

    @property
    def n_timesteps(self) -> int:
        """Number of time steps."""
        return len(self.time_ps)

    @property
    def total_time_ps(self) -> float:
        """Total simulation time in ps."""
        return float(self.time_ps[-1]) if len(self.time_ps) > 0 else 0.0

    def decay_rate_db_per_ps(self) -> float:
        """Compute average decay rate in dB/ps."""
        if len(self.time_ps) < 2:
            return 0.0
        dt = self.time_ps[-1] - self.time_ps[0]
        if dt <= 0:
            return 0.0
        return float((self.energy_db[-1] - self.energy_db[0]) / dt)


@dataclass(frozen=True, slots=True)
class ConvergenceCheckResult:
    """Result of a single convergence check.

    Attributes:
        name: Name of the check.
        status: Pass/fail/skip status.
        message: Human-readable description of result.
        value: Primary metric value (if applicable).
        threshold: Threshold used for comparison (if applicable).
        details: Additional diagnostic details.
    """

    name: str
    status: ConvergenceStatus
    message: str
    value: float | None = None
    threshold: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether check passed (or had warning)."""
        return self.status in (ConvergenceStatus.PASSED, ConvergenceStatus.WARNING)


@dataclass(frozen=True, slots=True)
class ConvergenceConfig:
    """Configuration for convergence validation.

    Attributes:
        energy_decay_target_db: Target energy decay level in dB.
        energy_decay_margin_db: Margin below target to consider converged.
        passivity_tolerance: Tolerance for S-matrix eigenvalue passivity check.
        power_balance_tolerance: Tolerance for port power balance check.
        min_cells_per_wavelength: Minimum mesh cells per wavelength.
        check_energy_decay: Whether to check energy decay.
        check_port_power: Whether to check port power balance.
        check_frequency_resolution: Whether to check frequency resolution.
        check_passivity: Whether to check S-matrix passivity.
    """

    energy_decay_target_db: float = -50.0
    energy_decay_margin_db: float = DEFAULT_ENERGY_DECAY_MARGIN_DB
    passivity_tolerance: float = DEFAULT_PASSIVITY_TOLERANCE
    power_balance_tolerance: float = DEFAULT_POWER_BALANCE_TOLERANCE
    min_cells_per_wavelength: int = DEFAULT_MIN_CELLS_PER_WAVELENGTH
    check_energy_decay: bool = True
    check_port_power: bool = True
    check_frequency_resolution: bool = True
    check_passivity: bool = True

    @classmethod
    def from_spec(cls, spec: SimulationSpec) -> "ConvergenceConfig":
        """Create config from simulation specification.

        Args:
            spec: Simulation specification.

        Returns:
            ConvergenceConfig with settings derived from spec.
        """
        return cls(
            energy_decay_target_db=spec.control.termination.end_criteria_db,
            min_cells_per_wavelength=spec.mesh.resolution.lambda_resolution,
        )


@dataclass(slots=True)
class ConvergenceReport:
    """Complete convergence validation report.

    Attributes:
        checks: List of individual check results.
        overall_status: Overall pass/fail status.
        simulation_hash: Hash of simulation inputs.
        canonical_hash: SHA256 hash of canonical report.
        config: Configuration used for validation.
    """

    checks: list[ConvergenceCheckResult]
    overall_status: ConvergenceStatus
    simulation_hash: str
    canonical_hash: str
    config: ConvergenceConfig

    @property
    def all_passed(self) -> bool:
        """Whether all checks passed."""
        return self.overall_status == ConvergenceStatus.PASSED

    @property
    def n_passed(self) -> int:
        """Number of checks that passed."""
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        """Number of checks that failed."""
        return sum(1 for c in self.checks if c.status == ConvergenceStatus.FAILED)

    def get_check(self, name: str) -> ConvergenceCheckResult | None:
        """Get a check result by name."""
        for check in self.checks:
            if check.name == name:
                return check
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "overall_status": self.overall_status.value,
            "simulation_hash": self.simulation_hash,
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
# Energy Decay Convergence Check
# =============================================================================


def load_energy_decay_json(json_path: Path) -> EnergyDecayData:
    """Load energy decay data from JSON file.

    Expected JSON format:
    {
        "time_ps": [0, 10, 20, ...],
        "energy_db": [-10, -20, -30, ...]
    }

    Args:
        json_path: Path to JSON file.

    Returns:
        EnergyDecayData parsed from file.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If format is invalid.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Energy decay file not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if "time_ps" not in data or "energy_db" not in data:
        raise ValueError("Energy decay JSON must have 'time_ps' and 'energy_db' keys")

    return EnergyDecayData(
        time_ps=np.array(data["time_ps"], dtype=np.float64),
        energy_db=np.array(data["energy_db"], dtype=np.float64),
    )


def check_energy_decay(
    energy_data: EnergyDecayData,
    config: ConvergenceConfig,
) -> ConvergenceCheckResult:
    """Check if energy has decayed to target threshold.

    The simulation is considered converged if the final energy level is
    at or below (target_db - margin_db).

    Args:
        energy_data: Energy decay time series.
        config: Convergence configuration.

    Returns:
        ConvergenceCheckResult with pass/fail status.
    """
    target = config.energy_decay_target_db
    margin = config.energy_decay_margin_db
    threshold = target - margin
    final_energy = energy_data.final_energy_db

    details = {
        "final_energy_db": final_energy,
        "target_db": target,
        "margin_db": margin,
        "effective_threshold_db": threshold,
        "total_time_ps": energy_data.total_time_ps,
        "n_timesteps": energy_data.n_timesteps,
        "decay_rate_db_per_ps": energy_data.decay_rate_db_per_ps(),
    }

    if final_energy <= threshold:
        return ConvergenceCheckResult(
            name="energy_decay",
            status=ConvergenceStatus.PASSED,
            message=f"Energy decayed to {final_energy:.1f} dB (threshold: {threshold:.1f} dB)",
            value=final_energy,
            threshold=threshold,
            details=details,
        )
    elif final_energy <= target:
        return ConvergenceCheckResult(
            name="energy_decay",
            status=ConvergenceStatus.WARNING,
            message=f"Energy at {final_energy:.1f} dB is close to target {target:.1f} dB",
            value=final_energy,
            threshold=threshold,
            details=details,
        )
    else:
        return ConvergenceCheckResult(
            name="energy_decay",
            status=ConvergenceStatus.FAILED,
            message=f"Energy only decayed to {final_energy:.1f} dB (need: {threshold:.1f} dB)",
            value=final_energy,
            threshold=threshold,
            details=details,
        )


# =============================================================================
# Port Power Balance Check
# =============================================================================


def check_port_power_balance(
    s_parameters: SParameterData,
    config: ConvergenceConfig,
) -> ConvergenceCheckResult:
    """Check port power balance for passivity and conservation.

    For a passive N-port network:
    - Sum of outgoing power <= Sum of incident power
    - S*S^H eigenvalues <= 1 (unitary bound for lossless, <1 for lossy)

    This check computes the power balance at each frequency point:
    - For 2-port: |S11|^2 + |S21|^2 <= 1 (with tolerance)

    Args:
        s_parameters: S-parameter data.
        config: Convergence configuration.

    Returns:
        ConvergenceCheckResult with pass/fail status.
    """
    tolerance = config.power_balance_tolerance
    n_ports = s_parameters.n_ports
    n_freqs = s_parameters.n_frequencies

    # Compute power balance at each frequency
    max_power_excess = 0.0
    worst_freq_hz = 0.0
    violations = 0

    for f_idx in range(n_freqs):
        freq = s_parameters.frequencies_hz[f_idx]
        s_mat = s_parameters.s_parameters[f_idx]

        # For each input port, compute total output power
        for port_in in range(n_ports):
            # Sum |Sij|^2 for all output ports j when exciting port i
            total_power = np.sum(np.abs(s_mat[:, port_in]) ** 2)

            # For passive device, total_power <= 1
            excess = total_power - 1.0
            if excess > max_power_excess:
                max_power_excess = excess
                worst_freq_hz = freq

            if excess > tolerance:
                violations += 1

    details = {
        "max_power_excess": float(max_power_excess),
        "worst_freq_hz": float(worst_freq_hz),
        "n_violations": violations,
        "n_frequencies": n_freqs,
        "n_ports": n_ports,
        "tolerance": tolerance,
    }

    if violations == 0:
        if max_power_excess > 0:
            return ConvergenceCheckResult(
                name="port_power_balance",
                status=ConvergenceStatus.WARNING,
                message=f"Power balance OK (max excess: {max_power_excess:.4f})",
                value=max_power_excess,
                threshold=tolerance,
                details=details,
            )
        return ConvergenceCheckResult(
            name="port_power_balance",
            status=ConvergenceStatus.PASSED,
            message="Power balance satisfied (passive device)",
            value=max_power_excess,
            threshold=tolerance,
            details=details,
        )
    else:
        return ConvergenceCheckResult(
            name="port_power_balance",
            status=ConvergenceStatus.FAILED,
            message=f"Power balance violated at {violations} frequencies "
            f"(max excess: {max_power_excess:.4f})",
            value=max_power_excess,
            threshold=tolerance,
            details=details,
        )


def check_passivity(
    s_parameters: SParameterData,
    config: ConvergenceConfig,
) -> ConvergenceCheckResult:
    """Check S-matrix passivity using eigenvalue criterion.

    A network is passive if the eigenvalues of S*S^H are all <= 1.
    This is a stronger condition than simple power balance.

    Args:
        s_parameters: S-parameter data.
        config: Convergence configuration.

    Returns:
        ConvergenceCheckResult with pass/fail status.
    """
    tolerance = config.passivity_tolerance
    n_freqs = s_parameters.n_frequencies

    max_eigenvalue = 0.0
    worst_freq_hz = 0.0
    violations = 0

    for f_idx in range(n_freqs):
        freq = s_parameters.frequencies_hz[f_idx]
        s_mat = s_parameters.s_parameters[f_idx]

        # Compute eigenvalues of S*S^H
        ssh = s_mat @ s_mat.conj().T
        eigenvalues = np.linalg.eigvalsh(ssh)
        max_eig = float(np.max(eigenvalues))

        if max_eig > max_eigenvalue:
            max_eigenvalue = max_eig
            worst_freq_hz = freq

        if max_eig > 1.0 + tolerance:
            violations += 1

    details = {
        "max_eigenvalue": max_eigenvalue,
        "worst_freq_hz": float(worst_freq_hz),
        "n_violations": violations,
        "n_frequencies": n_freqs,
        "tolerance": tolerance,
    }

    if violations == 0:
        return ConvergenceCheckResult(
            name="passivity",
            status=ConvergenceStatus.PASSED,
            message=f"Passivity satisfied (max eigenvalue: {max_eigenvalue:.6f})",
            value=max_eigenvalue,
            threshold=1.0 + tolerance,
            details=details,
        )
    else:
        return ConvergenceCheckResult(
            name="passivity",
            status=ConvergenceStatus.FAILED,
            message=f"Passivity violated at {violations} frequencies "
            f"(max eigenvalue: {max_eigenvalue:.6f})",
            value=max_eigenvalue,
            threshold=1.0 + tolerance,
            details=details,
        )


# =============================================================================
# Frequency Resolution Adequacy Check
# =============================================================================


@dataclass(frozen=True, slots=True)
class MeshInfo:
    """Mesh information for frequency resolution checking.

    Attributes:
        min_cell_size_nm: Minimum mesh cell size in nm.
        max_cell_size_nm: Maximum mesh cell size in nm.
        n_cells_x: Number of cells in x direction.
        n_cells_y: Number of cells in y direction.
        n_cells_z: Number of cells in z direction.
        total_cells: Total number of mesh cells.
    """

    min_cell_size_nm: float
    max_cell_size_nm: float
    n_cells_x: int
    n_cells_y: int
    n_cells_z: int

    @property
    def total_cells(self) -> int:
        """Total number of mesh cells."""
        return self.n_cells_x * self.n_cells_y * self.n_cells_z


def check_frequency_resolution(
    max_freq_hz: float,
    mesh_info: MeshInfo | None,
    config: ConvergenceConfig,
    epsilon_r: float = 1.0,
) -> ConvergenceCheckResult:
    """Check if mesh resolution is adequate for maximum frequency.

    The mesh cell size should be smaller than lambda/(min_cells_per_wavelength)
    where lambda is the wavelength in the medium.

    Args:
        max_freq_hz: Maximum simulation frequency in Hz.
        mesh_info: Mesh information (cell sizes). If None, check is skipped.
        config: Convergence configuration.
        epsilon_r: Relative permittivity of the medium.

    Returns:
        ConvergenceCheckResult with pass/fail status.
    """
    if mesh_info is None:
        return ConvergenceCheckResult(
            name="frequency_resolution",
            status=ConvergenceStatus.SKIPPED,
            message="Mesh information not available",
        )

    # Compute wavelength in the medium
    lambda_m = C0_M_PER_S / (max_freq_hz * np.sqrt(epsilon_r))
    lambda_nm = lambda_m / NM_TO_M

    # Required cell size
    min_cells = config.min_cells_per_wavelength
    required_cell_size_nm = lambda_nm / min_cells

    # Actual maximum cell size
    max_cell_nm = mesh_info.max_cell_size_nm

    # Compute actual cells per wavelength
    actual_cells_per_lambda = lambda_nm / max_cell_nm if max_cell_nm > 0 else float("inf")

    details = {
        "max_freq_hz": max_freq_hz,
        "wavelength_nm": lambda_nm,
        "epsilon_r": epsilon_r,
        "required_cells_per_wavelength": min_cells,
        "actual_cells_per_wavelength": actual_cells_per_lambda,
        "max_cell_size_nm": max_cell_nm,
        "required_cell_size_nm": required_cell_size_nm,
        "total_mesh_cells": mesh_info.total_cells,
    }

    if actual_cells_per_lambda >= min_cells:
        return ConvergenceCheckResult(
            name="frequency_resolution",
            status=ConvergenceStatus.PASSED,
            message=f"Mesh resolution adequate ({actual_cells_per_lambda:.1f} "
            f"cells/wavelength >= {min_cells})",
            value=actual_cells_per_lambda,
            threshold=float(min_cells),
            details=details,
        )
    elif actual_cells_per_lambda >= min_cells * 0.8:
        return ConvergenceCheckResult(
            name="frequency_resolution",
            status=ConvergenceStatus.WARNING,
            message=f"Mesh resolution marginal ({actual_cells_per_lambda:.1f} "
            f"cells/wavelength, target: {min_cells})",
            value=actual_cells_per_lambda,
            threshold=float(min_cells),
            details=details,
        )
    else:
        return ConvergenceCheckResult(
            name="frequency_resolution",
            status=ConvergenceStatus.FAILED,
            message=f"Mesh resolution insufficient ({actual_cells_per_lambda:.1f} "
            f"cells/wavelength < {min_cells})",
            value=actual_cells_per_lambda,
            threshold=float(min_cells),
            details=details,
        )


# =============================================================================
# High-Level Convergence Validation API
# =============================================================================


def validate_convergence(
    *,
    energy_data: EnergyDecayData | None = None,
    s_parameters: SParameterData | None = None,
    mesh_info: MeshInfo | None = None,
    max_freq_hz: float | None = None,
    epsilon_r: float = 4.0,
    simulation_hash: str = "",
    config: ConvergenceConfig | None = None,
) -> ConvergenceReport:
    """Validate convergence of a simulation using all available data.

    This is the main entry point for convergence checking. It runs all
    configured checks and produces a comprehensive report.

    Args:
        energy_data: Energy decay data (for energy decay check).
        s_parameters: S-parameter data (for power balance and passivity checks).
        mesh_info: Mesh information (for frequency resolution check).
        max_freq_hz: Maximum simulation frequency (for resolution check).
        epsilon_r: Relative permittivity for wavelength calculation.
        simulation_hash: Hash of simulation inputs for tracking.
        config: Convergence configuration (uses defaults if None).

    Returns:
        ConvergenceReport with all check results.
    """
    if config is None:
        config = ConvergenceConfig()

    checks: list[ConvergenceCheckResult] = []

    # Energy decay check
    if config.check_energy_decay:
        if energy_data is not None:
            checks.append(check_energy_decay(energy_data, config))
        else:
            checks.append(
                ConvergenceCheckResult(
                    name="energy_decay",
                    status=ConvergenceStatus.SKIPPED,
                    message="Energy decay data not available",
                )
            )

    # Port power balance check
    if config.check_port_power:
        if s_parameters is not None:
            checks.append(check_port_power_balance(s_parameters, config))
        else:
            checks.append(
                ConvergenceCheckResult(
                    name="port_power_balance",
                    status=ConvergenceStatus.SKIPPED,
                    message="S-parameter data not available",
                )
            )

    # Passivity check
    if config.check_passivity:
        if s_parameters is not None:
            checks.append(check_passivity(s_parameters, config))
        else:
            checks.append(
                ConvergenceCheckResult(
                    name="passivity",
                    status=ConvergenceStatus.SKIPPED,
                    message="S-parameter data not available",
                )
            )

    # Frequency resolution check
    if config.check_frequency_resolution:
        if max_freq_hz is not None:
            checks.append(
                check_frequency_resolution(max_freq_hz, mesh_info, config, epsilon_r)
            )
        else:
            checks.append(
                ConvergenceCheckResult(
                    name="frequency_resolution",
                    status=ConvergenceStatus.SKIPPED,
                    message="Maximum frequency not specified",
                )
            )

    # Determine overall status
    has_failure = any(c.status == ConvergenceStatus.FAILED for c in checks)
    has_warning = any(c.status == ConvergenceStatus.WARNING for c in checks)
    all_skipped = all(c.status == ConvergenceStatus.SKIPPED for c in checks)

    if has_failure:
        overall_status = ConvergenceStatus.FAILED
    elif all_skipped:
        overall_status = ConvergenceStatus.SKIPPED
    elif has_warning:
        overall_status = ConvergenceStatus.WARNING
    else:
        overall_status = ConvergenceStatus.PASSED

    # Compute canonical hash of the report
    report_data = {
        "simulation_hash": simulation_hash,
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

    return ConvergenceReport(
        checks=checks,
        overall_status=overall_status,
        simulation_hash=simulation_hash,
        canonical_hash=canonical_hash,
        config=config,
    )


def validate_simulation_convergence(
    sim_output_dir: Path,
    spec: SimulationSpec,
    *,
    simulation_hash: str = "",
    config: ConvergenceConfig | None = None,
) -> ConvergenceReport:
    """Validate convergence of a completed simulation from output directory.

    This convenience function loads all available data from a simulation
    output directory and runs convergence validation.

    Args:
        sim_output_dir: Directory containing simulation outputs.
        spec: Simulation specification.
        simulation_hash: Hash of simulation inputs.
        config: Convergence configuration (derived from spec if None).

    Returns:
        ConvergenceReport with all check results.
    """
    from formula_foundry.em.touchstone import read_touchstone

    if config is None:
        config = ConvergenceConfig.from_spec(spec)

    # Load energy decay data
    energy_data = None
    energy_path = sim_output_dir / "energy_decay.json"
    if energy_path.exists():
        try:
            energy_data = load_energy_decay_json(energy_path)
        except Exception as e:
            logger.warning("Failed to load energy decay data: %s", e)

    # Load S-parameters
    s_parameters = None
    sparam_patterns = ["*.s?p", "sparams.s2p"]
    for pattern in sparam_patterns:
        matches = list(sim_output_dir.glob(pattern))
        if matches:
            try:
                s_parameters = read_touchstone(matches[0])
                break
            except Exception as e:
                logger.warning("Failed to load S-parameters from %s: %s", matches[0], e)

    # Get max frequency from spec
    max_freq_hz = float(spec.frequency.f_stop_hz)

    # Get epsilon_r from materials (use substrate if available)
    epsilon_r = 4.0  # Default
    for dielectric in spec.materials.dielectrics:
        if "substrate" in dielectric.id.lower() or dielectric.id.lower() == "fr4":
            epsilon_r = dielectric.epsilon_r
            break

    # Mesh info would need to be extracted from simulation setup
    # For now, skip this check unless mesh info is provided separately
    mesh_info = None

    return validate_convergence(
        energy_data=energy_data,
        s_parameters=s_parameters,
        mesh_info=mesh_info,
        max_freq_hz=max_freq_hz,
        epsilon_r=epsilon_r,
        simulation_hash=simulation_hash,
        config=config,
    )


# =============================================================================
# Report Writing and Manifest Integration
# =============================================================================


def write_convergence_report(
    report: ConvergenceReport,
    output_path: Path,
) -> None:
    """Write convergence report to JSON file.

    Args:
        report: Convergence report to write.
        output_path: Path for output JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json_dumps(report.to_dict())
    output_path.write_text(f"{text}\n", encoding="utf-8")


def build_convergence_manifest_entry(report: ConvergenceReport) -> dict[str, Any]:
    """Build manifest entry for convergence report.

    Returns a dictionary suitable for inclusion in simulation manifest.

    Args:
        report: Convergence report.

    Returns:
        Manifest entry dictionary.
    """
    return {
        "convergence": {
            "overall_status": report.overall_status.value,
            "canonical_hash": report.canonical_hash,
            "n_passed": report.n_passed,
            "n_failed": report.n_failed,
            "checks": {
                c.name: {
                    "status": c.status.value,
                    "value": c.value,
                    "threshold": c.threshold,
                }
                for c in report.checks
            },
        }
    }


def convergence_gates_passed(report: ConvergenceReport) -> list[str]:
    """Extract list of passed convergence gates from report.

    Args:
        report: Convergence report.

    Returns:
        List of gate names that passed.
    """
    return [c.name for c in report.checks if c.passed]
