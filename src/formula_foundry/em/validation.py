"""S-parameter validation for Touchstone data.

This module implements REQ-M2-007: Validate passivity (|S| eigenvalues <= 1),
reciprocity (S12 ≈ S21), and causality checks. Include validation results in manifest.

Key validations:
- Passivity: S-matrix eigenvalues must satisfy |λ| <= 1 for physical realizability
- Reciprocity: For passive reciprocal networks, S12 ≈ S21
- Causality: The network must be causal (response doesn't precede excitation)

Validation results are structured for inclusion in simulation manifests to ensure
complete provenance tracking of S-parameter quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .touchstone import SParameterData

logger = logging.getLogger(__name__)

# Type aliases
ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


class ValidationStatus(str, Enum):
    """Status of a validation check."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class PassivityCheckResult:
    """Result of passivity validation.

    Passivity requires that the S-matrix singular values (or eigenvalues of S*S^H)
    be <= 1 at all frequencies. This ensures the network doesn't generate power.

    Attributes:
        status: Overall validation status.
        max_eigenvalue: Maximum eigenvalue of S*S^H across all frequencies.
        n_violations: Number of frequency points with passivity violations.
        violation_frequencies_hz: Frequencies where passivity is violated.
        tolerance: Tolerance used for the check.
        message: Human-readable status message.

    REQ-M2-007: Passivity validation for S-parameters.
    """

    status: ValidationStatus
    max_eigenvalue: float
    n_violations: int
    violation_frequencies_hz: tuple[float, ...]
    tolerance: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for manifest inclusion."""
        return {
            "check": "passivity",
            "status": self.status.value,
            "max_eigenvalue": self.max_eigenvalue,
            "n_violations": self.n_violations,
            "violation_frequencies_hz": list(self.violation_frequencies_hz),
            "tolerance": self.tolerance,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ReciprocityCheckResult:
    """Result of reciprocity validation.

    Reciprocity requires that Sij ≈ Sji for passive reciprocal networks.
    For 2-port networks, this primarily checks S12 ≈ S21.

    Attributes:
        status: Overall validation status.
        max_error: Maximum |Sij - Sji| across all frequencies and port pairs.
        mean_error: Mean reciprocity error.
        max_error_db: Maximum error in dB.
        n_violations: Number of frequency points exceeding tolerance.
        tolerance: Tolerance used for the check.
        message: Human-readable status message.

    REQ-M2-007: Reciprocity validation for S-parameters.
    """

    status: ValidationStatus
    max_error: float
    mean_error: float
    max_error_db: float
    n_violations: int
    tolerance: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for manifest inclusion."""
        return {
            "check": "reciprocity",
            "status": self.status.value,
            "max_error": self.max_error,
            "mean_error": self.mean_error,
            "max_error_db": self.max_error_db,
            "n_violations": self.n_violations,
            "tolerance": self.tolerance,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class CausalityCheckResult:
    """Result of causality validation.

    Causality ensures the network response doesn't precede excitation.
    This is checked by:
    1. Verifying the impulse response is zero for t < 0
    2. Checking Kramers-Kronig consistency (optional)

    Attributes:
        status: Overall validation status.
        is_causal: Whether the network appears causal.
        pre_response_energy_ratio: Ratio of energy before t=0 to total energy.
        max_pre_response_magnitude: Maximum impulse response magnitude for t < 0.
        tolerance: Tolerance for pre-response energy.
        message: Human-readable status message.

    REQ-M2-007: Causality validation for S-parameters.
    """

    status: ValidationStatus
    is_causal: bool
    pre_response_energy_ratio: float
    max_pre_response_magnitude: float
    tolerance: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for manifest inclusion."""
        return {
            "check": "causality",
            "status": self.status.value,
            "is_causal": self.is_causal,
            "pre_response_energy_ratio": self.pre_response_energy_ratio,
            "max_pre_response_magnitude": self.max_pre_response_magnitude,
            "tolerance": self.tolerance,
            "message": self.message,
        }


@dataclass(slots=True)
class SParameterValidationResult:
    """Complete S-parameter validation result.

    Combines all validation checks into a single result structure
    suitable for manifest inclusion.

    Attributes:
        passivity: Passivity check result.
        reciprocity: Reciprocity check result.
        causality: Causality check result.
        overall_status: Overall validation status (worst of individual checks).
        n_frequencies: Number of frequency points validated.
        n_ports: Number of ports in the network.

    REQ-M2-007: Combined validation results for manifest.
    """

    passivity: PassivityCheckResult
    reciprocity: ReciprocityCheckResult
    causality: CausalityCheckResult
    overall_status: ValidationStatus
    n_frequencies: int
    n_ports: int
    _extra_checks: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if all validations passed."""
        return self.overall_status == ValidationStatus.PASS

    @property
    def has_warnings(self) -> bool:
        """Check if any validation has warnings."""
        return (
            self.passivity.status == ValidationStatus.WARN
            or self.reciprocity.status == ValidationStatus.WARN
            or self.causality.status == ValidationStatus.WARN
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for manifest inclusion.

        REQ-M2-007: Validation results in manifest format.
        """
        result: dict[str, Any] = {
            "overall_status": self.overall_status.value,
            "is_valid": self.is_valid,
            "has_warnings": self.has_warnings,
            "n_frequencies": self.n_frequencies,
            "n_ports": self.n_ports,
            "checks": {
                "passivity": self.passivity.to_dict(),
                "reciprocity": self.reciprocity.to_dict(),
                "causality": self.causality.to_dict(),
            },
        }
        if self._extra_checks:
            result["checks"].update(self._extra_checks)
        return result

    def add_extra_check(self, name: str, check_dict: dict[str, Any]) -> None:
        """Add an additional validation check result."""
        self._extra_checks[name] = check_dict


# =============================================================================
# Passivity Validation (REQ-M2-007)
# =============================================================================


def check_passivity(
    sparam_data: SParameterData,
    *,
    tolerance: float = 1e-6,
    warn_threshold: float = 1e-3,
) -> PassivityCheckResult:
    """Check passivity of S-parameter data.

    A network is passive if it doesn't generate power, which requires all
    eigenvalues of S*S^H to be <= 1 at all frequencies.

    For physical networks, passivity must be satisfied. Violations indicate:
    - Numerical errors in simulation
    - Insufficient mesh resolution
    - Non-physical material properties

    Args:
        sparam_data: S-parameter data to validate.
        tolerance: Maximum allowed eigenvalue excess over 1.0.
        warn_threshold: Threshold for issuing a warning vs failure.

    Returns:
        PassivityCheckResult with validation details.

    REQ-M2-007: Validate passivity (|S| eigenvalues <= 1).

    Example:
        >>> result = check_passivity(sparam_data)
        >>> if result.status == ValidationStatus.PASS:
        ...     print("Network is passive")
        >>> else:
        ...     print(f"Passivity violated at {result.n_violations} frequencies")
    """
    n_freq = sparam_data.n_frequencies
    violation_frequencies: list[float] = []
    max_eigenvalue = 0.0

    for f_idx in range(n_freq):
        s_matrix = sparam_data.s_parameters[f_idx]
        # Compute eigenvalues of S*S^H (scattering power matrix)
        eigenvalues = np.linalg.eigvalsh(s_matrix @ s_matrix.conj().T)
        max_eig = float(np.max(eigenvalues))

        if max_eig > max_eigenvalue:
            max_eigenvalue = max_eig

        if max_eig > 1.0 + tolerance:
            violation_frequencies.append(float(sparam_data.frequencies_hz[f_idx]))

    n_violations = len(violation_frequencies)

    # Determine status
    if n_violations == 0 and max_eigenvalue <= 1.0 + tolerance:
        status = ValidationStatus.PASS
        message = f"Network is passive (max eigenvalue = {max_eigenvalue:.6f})"
    elif max_eigenvalue <= 1.0 + warn_threshold:
        status = ValidationStatus.WARN
        message = (
            f"Network is marginally passive (max eigenvalue = {max_eigenvalue:.6f}, "
            f"{n_violations} frequency points exceed tolerance)"
        )
    else:
        status = ValidationStatus.FAIL
        message = (
            f"Passivity violated (max eigenvalue = {max_eigenvalue:.6f}, "
            f"{n_violations} violations)"
        )

    return PassivityCheckResult(
        status=status,
        max_eigenvalue=max_eigenvalue,
        n_violations=n_violations,
        violation_frequencies_hz=tuple(violation_frequencies),
        tolerance=tolerance,
        message=message,
    )


# =============================================================================
# Reciprocity Validation (REQ-M2-007)
# =============================================================================


def check_reciprocity(
    sparam_data: SParameterData,
    *,
    tolerance: float = 1e-6,
    warn_threshold: float = 1e-3,
) -> ReciprocityCheckResult:
    """Check reciprocity of S-parameter data.

    Reciprocity requires Sij = Sji for passive reciprocal networks (i.e., networks
    without ferrites, active devices, or nonreciprocal materials).

    For via transitions in PCBs, reciprocity should be well-satisfied. Violations
    indicate:
    - Numerical asymmetry in simulation
    - Incorrect boundary conditions
    - Non-reciprocal materials (unusual for PCB)

    Args:
        sparam_data: S-parameter data to validate.
        tolerance: Maximum allowed |Sij - Sji| for pass.
        warn_threshold: Threshold for issuing a warning vs failure.

    Returns:
        ReciprocityCheckResult with validation details.

    REQ-M2-007: Validate reciprocity (S12 ≈ S21).

    Example:
        >>> result = check_reciprocity(sparam_data)
        >>> if result.status == ValidationStatus.PASS:
        ...     print(f"Network is reciprocal (max error = {result.max_error})")
    """
    n_freq = sparam_data.n_frequencies
    n_ports = sparam_data.n_ports
    errors: list[float] = []
    n_violations = 0

    for f_idx in range(n_freq):
        s_matrix = sparam_data.s_parameters[f_idx]

        # Check all off-diagonal pairs
        for i in range(n_ports):
            for j in range(i + 1, n_ports):
                error = abs(s_matrix[i, j] - s_matrix[j, i])
                errors.append(error)
                if error > tolerance:
                    n_violations += 1

    max_error = float(np.max(errors)) if errors else 0.0
    mean_error = float(np.mean(errors)) if errors else 0.0

    # Convert to dB (20*log10 of relative error)
    # Use reference as mean magnitude of off-diagonal elements
    if errors:
        max_error_db = 20.0 * np.log10(max(max_error, 1e-15))
    else:
        max_error_db = -300.0  # Very small (perfect)

    # Determine status
    if max_error <= tolerance:
        status = ValidationStatus.PASS
        message = f"Network is reciprocal (max error = {max_error:.2e})"
    elif max_error <= warn_threshold:
        status = ValidationStatus.WARN
        message = (
            f"Network is approximately reciprocal (max error = {max_error:.2e}, "
            f"{n_violations} frequency/pair violations)"
        )
    else:
        status = ValidationStatus.FAIL
        message = (
            f"Reciprocity violated (max error = {max_error:.2e}, "
            f"{n_violations} violations)"
        )

    return ReciprocityCheckResult(
        status=status,
        max_error=max_error,
        mean_error=mean_error,
        max_error_db=max_error_db,
        n_violations=n_violations,
        tolerance=tolerance,
        message=message,
    )


# =============================================================================
# Causality Validation (REQ-M2-007)
# =============================================================================


def check_causality(
    sparam_data: SParameterData,
    *,
    tolerance: float = 1e-3,
    warn_threshold: float = 1e-2,
) -> CausalityCheckResult:
    """Check causality of S-parameter data.

    A network is causal if its impulse response is zero for t < 0. This is
    checked by computing the inverse FFT of the frequency response and
    examining the pre-t=0 region.

    Non-causality can indicate:
    - Insufficient frequency range
    - Aliasing in time domain
    - Data interpolation artifacts
    - Non-physical simulation results

    This check uses the S21 (transmission) parameter for 2-port networks
    as it's most sensitive to causality issues.

    Args:
        sparam_data: S-parameter data to validate.
        tolerance: Maximum allowed pre-response energy ratio for pass.
        warn_threshold: Threshold for issuing a warning vs failure.

    Returns:
        CausalityCheckResult with validation details.

    REQ-M2-007: Validate causality for S-parameters.

    Example:
        >>> result = check_causality(sparam_data)
        >>> if result.is_causal:
        ...     print("Network is causal")
    """
    n_freq = sparam_data.n_frequencies
    n_ports = sparam_data.n_ports

    # Use S21 for 2-port, or S11 for 1-port
    if n_ports >= 2:
        s_response = sparam_data.s21()
    else:
        s_response = sparam_data.s11()

    # Compute impulse response via inverse FFT
    # Pad to improve resolution
    n_fft = max(4 * n_freq, 1024)
    s_padded = np.zeros(n_fft, dtype=np.complex128)
    s_padded[:n_freq] = s_response

    # Inverse FFT to get impulse response
    impulse_response = np.fft.ifft(s_padded)

    # For a causal system, the response should be zero for negative time
    # In the FFT convention, negative time corresponds to indices > n_fft/2
    mid_point = n_fft // 2
    pre_response = impulse_response[mid_point:]  # t < 0 region
    main_response = impulse_response[:mid_point]  # t >= 0 region

    # Compute energy in pre-response and total
    pre_response_energy = float(np.sum(np.abs(pre_response) ** 2))
    total_energy = float(np.sum(np.abs(impulse_response) ** 2))

    if total_energy > 0:
        pre_response_ratio = pre_response_energy / total_energy
    else:
        pre_response_ratio = 0.0

    max_pre_response = float(np.max(np.abs(pre_response)))

    # Determine status
    is_causal = pre_response_ratio <= tolerance
    if pre_response_ratio <= tolerance:
        status = ValidationStatus.PASS
        message = f"Network is causal (pre-response energy ratio = {pre_response_ratio:.2e})"
    elif pre_response_ratio <= warn_threshold:
        status = ValidationStatus.WARN
        is_causal = True  # Still considered causal with warning
        message = (
            f"Network is approximately causal (pre-response energy ratio = {pre_response_ratio:.2e})"
        )
    else:
        status = ValidationStatus.FAIL
        message = (
            f"Causality check failed (pre-response energy ratio = {pre_response_ratio:.2e})"
        )

    return CausalityCheckResult(
        status=status,
        is_causal=is_causal,
        pre_response_energy_ratio=pre_response_ratio,
        max_pre_response_magnitude=max_pre_response,
        tolerance=tolerance,
        message=message,
    )


# =============================================================================
# Combined Validation (REQ-M2-007)
# =============================================================================


def validate_sparam_data(
    sparam_data: SParameterData,
    *,
    passivity_tolerance: float = 1e-6,
    reciprocity_tolerance: float = 1e-6,
    causality_tolerance: float = 1e-3,
    skip_causality: bool = False,
) -> SParameterValidationResult:
    """Validate S-parameter data for passivity, reciprocity, and causality.

    This is the main validation entry point that runs all checks and combines
    results into a single validation result suitable for manifest inclusion.

    Args:
        sparam_data: S-parameter data to validate.
        passivity_tolerance: Tolerance for passivity check.
        reciprocity_tolerance: Tolerance for reciprocity check.
        causality_tolerance: Tolerance for causality check.
        skip_causality: Skip causality check (useful for incomplete frequency data).

    Returns:
        SParameterValidationResult with all check results.

    REQ-M2-007: Combined S-parameter validation with manifest-ready results.

    Example:
        >>> result = validate_sparam_data(sparam_data)
        >>> manifest["validation"] = result.to_dict()
        >>> if not result.is_valid:
        ...     logger.warning(f"S-parameter validation failed: {result.overall_status}")
    """
    # Run passivity check
    passivity_result = check_passivity(sparam_data, tolerance=passivity_tolerance)

    # Run reciprocity check
    reciprocity_result = check_reciprocity(sparam_data, tolerance=reciprocity_tolerance)

    # Run causality check (optional)
    if skip_causality:
        causality_result = CausalityCheckResult(
            status=ValidationStatus.SKIP,
            is_causal=True,
            pre_response_energy_ratio=0.0,
            max_pre_response_magnitude=0.0,
            tolerance=causality_tolerance,
            message="Causality check skipped",
        )
    else:
        causality_result = check_causality(sparam_data, tolerance=causality_tolerance)

    # Determine overall status (worst of individual checks, excluding SKIP)
    statuses = [passivity_result.status, reciprocity_result.status]
    if not skip_causality:
        statuses.append(causality_result.status)

    if ValidationStatus.FAIL in statuses:
        overall_status = ValidationStatus.FAIL
    elif ValidationStatus.WARN in statuses:
        overall_status = ValidationStatus.WARN
    else:
        overall_status = ValidationStatus.PASS

    return SParameterValidationResult(
        passivity=passivity_result,
        reciprocity=reciprocity_result,
        causality=causality_result,
        overall_status=overall_status,
        n_frequencies=sparam_data.n_frequencies,
        n_ports=sparam_data.n_ports,
    )


# =============================================================================
# Touchstone File Validation (REQ-M2-007)
# =============================================================================


def validate_touchstone_file(
    file_path: str | Path,
    *,
    passivity_tolerance: float = 1e-6,
    reciprocity_tolerance: float = 1e-6,
    causality_tolerance: float = 1e-3,
    skip_causality: bool = False,
) -> SParameterValidationResult:
    """Validate a Touchstone file for passivity, reciprocity, and causality.

    Convenience function that loads a Touchstone file and validates it.

    Args:
        file_path: Path to Touchstone file.
        passivity_tolerance: Tolerance for passivity check.
        reciprocity_tolerance: Tolerance for reciprocity check.
        causality_tolerance: Tolerance for causality check.
        skip_causality: Skip causality check.

    Returns:
        SParameterValidationResult with all check results.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is invalid.

    REQ-M2-007: Touchstone file validation.

    Example:
        >>> result = validate_touchstone_file("simulation.s2p")
        >>> if result.is_valid:
        ...     print("Touchstone file is valid")
    """
    from .touchstone import read_touchstone

    sparam_data = read_touchstone(file_path)
    return validate_sparam_data(
        sparam_data,
        passivity_tolerance=passivity_tolerance,
        reciprocity_tolerance=reciprocity_tolerance,
        causality_tolerance=causality_tolerance,
        skip_causality=skip_causality,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def compute_stability_k_factor(sparam_data: SParameterData) -> FloatArray:
    """Compute Rollett stability factor K for 2-port networks.

    The stability factor K is defined as:
    K = (1 - |S11|^2 - |S22|^2 + |Δ|^2) / (2*|S21*S12|)

    where Δ = S11*S22 - S12*S21 (determinant of S-matrix).

    For unconditional stability: K > 1 and |Δ| < 1

    Args:
        sparam_data: S-parameter data (must be 2-port).

    Returns:
        Array of K values at each frequency point.

    Raises:
        ValueError: If not a 2-port network.

    REQ-M2-007: Additional stability metric for 2-port networks.
    """
    if sparam_data.n_ports != 2:
        raise ValueError("Stability K factor requires 2-port network")

    s11 = sparam_data.s11()
    s12 = sparam_data.s12()
    s21 = sparam_data.s21()
    s22 = sparam_data.s22()

    # Determinant
    delta = s11 * s22 - s12 * s21

    # Numerator and denominator
    numerator = 1 - np.abs(s11) ** 2 - np.abs(s22) ** 2 + np.abs(delta) ** 2
    denominator = 2 * np.abs(s21 * s12)

    # Avoid division by zero
    k = np.where(denominator > 1e-15, numerator / denominator, np.inf)

    return k


def check_stability_2port(
    sparam_data: SParameterData,
    *,
    k_threshold: float = 1.0,
) -> dict[str, Any]:
    """Check unconditional stability for 2-port networks.

    A 2-port network is unconditionally stable when:
    - K > 1 at all frequencies
    - |Δ| < 1 at all frequencies

    Args:
        sparam_data: S-parameter data (must be 2-port).
        k_threshold: Minimum K for stability (typically 1.0).

    Returns:
        Dictionary with stability check results.

    REQ-M2-007: Stability check for 2-port networks.
    """
    if sparam_data.n_ports != 2:
        return {
            "check": "stability_2port",
            "status": "skip",
            "message": "Stability check requires 2-port network",
        }

    k = compute_stability_k_factor(sparam_data)

    s11 = sparam_data.s11()
    s12 = sparam_data.s12()
    s21 = sparam_data.s21()
    s22 = sparam_data.s22()
    delta = s11 * s22 - s12 * s21
    delta_mag = np.abs(delta)

    is_k_stable = bool(np.all(k > k_threshold))
    is_delta_stable = bool(np.all(delta_mag < 1.0))
    is_unconditionally_stable = is_k_stable and is_delta_stable

    return {
        "check": "stability_2port",
        "status": "pass" if is_unconditionally_stable else "warn",
        "is_unconditionally_stable": is_unconditionally_stable,
        "k_min": float(np.min(k)),
        "k_max": float(np.max(k)),
        "k_mean": float(np.mean(k)),
        "delta_max": float(np.max(delta_mag)),
        "message": (
            "Network is unconditionally stable"
            if is_unconditionally_stable
            else f"Network may be conditionally stable (K_min={float(np.min(k)):.4f})"
        ),
    }


def build_validation_manifest_entry(
    validation_result: SParameterValidationResult,
) -> dict[str, Any]:
    """Build manifest entry from validation result.

    Convenience function that creates a manifest-compatible dictionary
    from a validation result.

    Args:
        validation_result: Complete validation result.

    Returns:
        Dictionary suitable for manifest inclusion.

    REQ-M2-007: Validation results in manifest format.

    Example:
        >>> validation = validate_sparam_data(sparam_data)
        >>> manifest["s_parameter_validation"] = build_validation_manifest_entry(validation)
    """
    return validation_result.to_dict()
