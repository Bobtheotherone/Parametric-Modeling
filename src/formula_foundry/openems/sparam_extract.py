"""S-parameter extraction and post-processing for openEMS simulation results.

This module implements REQ-M2-006: S-parameter extraction pipeline from openEMS
field data with windowing, FFT, and complex S-parameter computation.

Key functionality:
- Parse simulation output files (Touchstone, CSV)
- Compute S11, S21, S12, S22 matrices from port voltage/current data
- Apply windowing functions (Hann, Hamming, Blackman, Kaiser) to time-domain data
- Handle frequency sweeps with proper interpolation
- Produce structured output for manifest inclusion
- Support de-embedding and reference plane shifting

The extraction pipeline:
1. Load raw port signal data from simulation
2. Apply configurable windowing to time-domain signals
3. Compute frequency-domain transfer functions via FFT
4. Extract S-parameters at specified frequency points
5. Apply de-embedding corrections if configured
6. Package results in structured format for manifest

REQ-M2-006: S-parameter extraction pipeline with windowing and FFT.
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

from formula_foundry.em.touchstone import (
    FrequencyUnit,
    SParameterData,
    SParameterFormat,
    TouchstoneOptions,
    read_touchstone,
    write_touchstone,
)
from formula_foundry.postprocess.renormalize import renormalize_sparameters
from formula_foundry.substrate import canonical_json_dumps, sha256_bytes

from .spec import FrequencySpec, PortSpec

logger = logging.getLogger(__name__)

# Type aliases
ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

# Constants
C0_M_PER_S = 299_792_458.0  # Speed of light in m/s
NM_TO_M = 1e-9


# =============================================================================
# Windowing Functions (REQ-M2-006)
# =============================================================================


class WindowType(str, Enum):
    """Available window functions for time-domain signal processing.

    Window functions are applied to time-domain data before FFT to reduce
    spectral leakage and improve frequency resolution.

    REQ-M2-006: Windowing for S-parameter extraction.
    """

    NONE = "none"
    """No windowing (rectangular window). Best frequency resolution but highest leakage."""

    HANN = "hann"
    """Hann (Hanning) window. Good general-purpose window with moderate leakage."""

    HAMMING = "hamming"
    """Hamming window. Similar to Hann but slightly higher side lobes."""

    BLACKMAN = "blackman"
    """Blackman window. Lower side lobes but wider main lobe."""

    KAISER = "kaiser"
    """Kaiser window with configurable beta parameter for tunable trade-off."""

    TUKEY = "tukey"
    """Tukey (cosine-tapered) window. Useful for transient simulations."""


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Configuration for time-domain windowing.

    Attributes:
        window_type: Type of window function to apply.
        kaiser_beta: Beta parameter for Kaiser window (default 14 for good sidelobe suppression).
        tukey_alpha: Alpha parameter for Tukey window (0.5 is typical).
        normalize: Whether to normalize window to preserve signal energy.

    REQ-M2-006: Windowing configuration for S-parameter extraction.
    """

    window_type: WindowType = WindowType.HANN
    kaiser_beta: float = 14.0
    tukey_alpha: float = 0.5
    normalize: bool = True


def create_window(n_samples: int, config: WindowConfig) -> FloatArray:
    """Create a window function array.

    Args:
        n_samples: Number of samples in the window.
        config: Window configuration.

    Returns:
        Window function as a 1D array.

    REQ-M2-006: Window function generation.
    """
    if config.window_type == WindowType.NONE:
        window = np.ones(n_samples)
    elif config.window_type == WindowType.HANN:
        window = np.hanning(n_samples)
    elif config.window_type == WindowType.HAMMING:
        window = np.hamming(n_samples)
    elif config.window_type == WindowType.BLACKMAN:
        window = np.blackman(n_samples)
    elif config.window_type == WindowType.KAISER:
        window = np.kaiser(n_samples, config.kaiser_beta)
    elif config.window_type == WindowType.TUKEY:
        window = _tukey_window(n_samples, config.tukey_alpha)
    else:
        raise ValueError(f"Unknown window type: {config.window_type}")

    if config.normalize:
        # Normalize to preserve signal energy
        coherent_gain = np.sum(window) / n_samples
        if coherent_gain > 0:
            window = window / coherent_gain

    return window


def _tukey_window(n_samples: int, alpha: float) -> FloatArray:
    """Create a Tukey (cosine-tapered) window.

    The Tukey window is a rectangular window with cosine tapers at the ends.
    Alpha controls the fraction of the window inside the cosine tapers:
    - alpha=0: rectangular window
    - alpha=1: Hann window

    Args:
        n_samples: Number of samples.
        alpha: Taper fraction (0 to 1).

    Returns:
        Tukey window array.
    """
    if alpha <= 0:
        return np.ones(n_samples)
    if alpha >= 1:
        return np.hanning(n_samples)

    window = np.ones(n_samples)
    # Number of points in the tapered region on each side
    n_taper = int(alpha * n_samples / 2)

    if n_taper > 0:
        # Left taper
        t_left = np.arange(n_taper)
        window[:n_taper] = 0.5 * (1 - np.cos(np.pi * t_left / n_taper))

        # Right taper
        t_right = np.arange(n_taper)
        window[-n_taper:] = 0.5 * (1 - np.cos(np.pi * (t_right + 1) / n_taper))[::-1]

    return window


def apply_window(signal: FloatArray, config: WindowConfig) -> FloatArray:
    """Apply windowing to a time-domain signal.

    Args:
        signal: Time-domain signal array.
        config: Window configuration.

    Returns:
        Windowed signal.

    REQ-M2-006: Apply windowing to time-domain data.
    """
    window = create_window(len(signal), config)
    return signal * window


def compute_window_metrics(config: WindowConfig, n_samples: int = 1024) -> dict[str, float]:
    """Compute window function metrics.

    Returns useful metrics about the window function:
    - coherent_gain: Ratio of windowed vs unwindowed DC response
    - noise_bandwidth: Equivalent noise bandwidth (bins)
    - processing_gain: Gain in SNR from windowing (dB)
    - scalloping_loss: Maximum amplitude error for off-bin frequencies (dB)

    Args:
        config: Window configuration.
        n_samples: Number of samples for computation.

    Returns:
        Dictionary of window metrics.

    REQ-M2-006: Window function metrics for analysis.
    """
    window = create_window(n_samples, WindowConfig(
        window_type=config.window_type,
        kaiser_beta=config.kaiser_beta,
        tukey_alpha=config.tukey_alpha,
        normalize=False,  # Use unnormalized for metrics
    ))

    # Coherent gain
    coherent_gain = np.sum(window) / n_samples

    # Noise bandwidth (equivalent noise bandwidth in bins)
    noise_bandwidth = np.sum(window**2) / (coherent_gain**2 * n_samples)

    # Processing gain (dB)
    processing_gain = 10 * np.log10(n_samples / noise_bandwidth) if noise_bandwidth > 0 else 0

    # Scalloping loss (approximate based on window type)
    scalloping_loss_table = {
        WindowType.NONE: 3.92,
        WindowType.HANN: 1.42,
        WindowType.HAMMING: 1.75,
        WindowType.BLACKMAN: 1.10,
        WindowType.KAISER: 1.0,  # Varies with beta
        WindowType.TUKEY: 2.0,   # Varies with alpha
    }
    scalloping_loss = scalloping_loss_table.get(config.window_type, 1.5)

    return {
        "coherent_gain": float(coherent_gain),
        "noise_bandwidth_bins": float(noise_bandwidth),
        "processing_gain_db": float(processing_gain),
        "scalloping_loss_db": float(scalloping_loss),
    }


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Configuration for S-parameter extraction.

    Attributes:
        frequency_spec: Frequency sweep specification.
        port_specs: List of port specifications.
        reference_impedance_ohm: Reference impedance for S-parameters.
        renormalize_to_ohms: Optional impedance for renormalized exports.
        deembed_enabled: Whether de-embedding is enabled.
        output_format: Output format for S-parameters.
        window_config: Configuration for time-domain windowing (REQ-M2-006).

    REQ-M2-006: Enhanced S-parameter extraction with windowing support.
    """

    frequency_spec: FrequencySpec
    port_specs: list[PortSpec]
    reference_impedance_ohm: float = 50.0
    renormalize_to_ohms: float | None = None
    deembed_enabled: bool = False
    output_format: Literal["touchstone", "csv", "both"] = "touchstone"
    window_config: WindowConfig = field(default_factory=lambda: WindowConfig())

    @property
    def n_ports(self) -> int:
        """Number of ports."""
        return len(self.port_specs)

    @property
    def n_frequencies(self) -> int:
        """Number of frequency points."""
        return self.frequency_spec.n_points

    def frequencies_hz(self) -> FloatArray:
        """Generate frequency array in Hz."""
        return np.linspace(
            float(self.frequency_spec.f_start_hz),
            float(self.frequency_spec.f_stop_hz),
            self.frequency_spec.n_points,
        )


@dataclass(slots=True)
class PortSignalData:
    """Time-domain port signal data.

    Attributes:
        port_id: Port identifier.
        time_s: Time array in seconds.
        voltage_v: Voltage signal array.
        current_a: Current signal array.
    """

    port_id: str
    time_s: FloatArray
    voltage_v: FloatArray
    current_a: FloatArray

    def __post_init__(self) -> None:
        if len(self.time_s) != len(self.voltage_v):
            raise ValueError("time_s and voltage_v must have same length")
        if len(self.time_s) != len(self.current_a):
            raise ValueError("time_s and current_a must have same length")


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Result of S-parameter extraction.

    Attributes:
        s_parameters: Extracted S-parameter data.
        extraction_config: Configuration used for extraction.
        source_files: List of source files used.
        canonical_hash: SHA256 hash of canonical representation.
        metrics: Dictionary of extraction metrics.
    """

    s_parameters: SParameterData
    extraction_config: ExtractionConfig
    source_files: list[str]
    canonical_hash: str
    metrics: dict[str, Any] = field(default_factory=dict)


def extract_sparams_from_touchstone(
    touchstone_path: Path,
    config: ExtractionConfig,
) -> ExtractionResult:
    """Extract S-parameters from a Touchstone file.

    This is the simplest extraction path when the simulation directly
    produces Touchstone output.

    Args:
        touchstone_path: Path to Touchstone file.
        config: Extraction configuration.

    Returns:
        ExtractionResult with S-parameter data.

    Raises:
        FileNotFoundError: If touchstone file doesn't exist.
        ValueError: If file format is invalid.
    """
    if not touchstone_path.exists():
        raise FileNotFoundError(f"Touchstone file not found: {touchstone_path}")

    sparam_data = read_touchstone(touchstone_path)

    # Interpolate to requested frequency grid if different
    target_freqs = config.frequencies_hz()
    needs_interpolation = len(sparam_data.frequencies_hz) != len(target_freqs) or not np.allclose(
        sparam_data.frequencies_hz, target_freqs, rtol=1e-6
    )
    if needs_interpolation:
        logger.info(
            "Interpolating S-parameters from %d to %d frequency points",
            sparam_data.n_frequencies,
            len(target_freqs),
        )
        sparam_data = sparam_data.interpolate(target_freqs)

    # Compute canonical hash
    canonical = _sparam_canonical_json(sparam_data)
    canonical_hash = sha256_bytes(canonical.encode("utf-8"))

    # Compute metrics
    metrics = _compute_extraction_metrics(sparam_data)

    return ExtractionResult(
        s_parameters=sparam_data,
        extraction_config=config,
        source_files=[str(touchstone_path)],
        canonical_hash=canonical_hash,
        metrics=metrics,
    )


def extract_sparams_from_csv(
    csv_path: Path,
    config: ExtractionConfig,
) -> ExtractionResult:
    """Extract S-parameters from a CSV file.

    CSV format expected:
    freq_hz,s11_re,s11_im,s21_re,s21_im,s12_re,s12_im,s22_re,s22_im

    Args:
        csv_path: Path to CSV file.
        config: Extraction configuration.

    Returns:
        ExtractionResult with S-parameter data.

    Raises:
        FileNotFoundError: If CSV file doesn't exist.
        ValueError: If file format is invalid.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Parse CSV
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_freq = data.shape[0]
    n_ports = config.n_ports

    # Validate column count
    expected_cols = 1 + 2 * n_ports * n_ports  # freq + 2*(re,im) per S-param
    if data.shape[1] < expected_cols:
        raise ValueError(f"CSV has {data.shape[1]} columns, expected at least {expected_cols} for {n_ports}-port S-parameters")

    frequencies_hz = data[:, 0]
    s_parameters = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)

    # Parse S-parameters (assumed order: S11, S21, S12, S22 for 2-port)
    col = 1
    if n_ports == 2:
        # Standard 2-port order
        order = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for out_idx, in_idx in order:
            s_parameters[:, out_idx, in_idx] = data[:, col] + 1j * data[:, col + 1]
            col += 2
    else:
        # Row-major order for other port counts
        for out_idx in range(n_ports):
            for in_idx in range(n_ports):
                s_parameters[:, out_idx, in_idx] = data[:, col] + 1j * data[:, col + 1]
                col += 2

    sparam_data = SParameterData(
        frequencies_hz=frequencies_hz,
        s_parameters=s_parameters,
        n_ports=n_ports,
        reference_impedance_ohm=config.reference_impedance_ohm,
    )

    # Interpolate to requested frequency grid if different
    target_freqs = config.frequencies_hz()
    needs_interpolation = len(sparam_data.frequencies_hz) != len(target_freqs) or not np.allclose(
        sparam_data.frequencies_hz, target_freqs, rtol=1e-6
    )
    if needs_interpolation:
        sparam_data = sparam_data.interpolate(target_freqs)

    # Compute canonical hash
    canonical = _sparam_canonical_json(sparam_data)
    canonical_hash = sha256_bytes(canonical.encode("utf-8"))

    # Compute metrics
    metrics = _compute_extraction_metrics(sparam_data)

    return ExtractionResult(
        s_parameters=sparam_data,
        extraction_config=config,
        source_files=[str(csv_path)],
        canonical_hash=canonical_hash,
        metrics=metrics,
    )


def extract_sparams_from_port_signals(
    port_signals: list[PortSignalData],
    excitation_port_id: str,
    config: ExtractionConfig,
    *,
    window_config: WindowConfig | None = None,
) -> ExtractionResult:
    """Extract S-parameters from time-domain port signals.

    This function computes S-parameters by:
    1. Applying windowing to time-domain signals (REQ-M2-006)
    2. Computing FFT of incident and reflected waves at each port
    3. Forming the S-matrix from wave ratios

    The windowing reduces spectral leakage when the simulation doesn't
    capture an integer number of periods. For FDTD simulations that
    run until field decay, a Tukey or Hann window is recommended.

    Args:
        port_signals: List of time-domain port signals.
        excitation_port_id: ID of the excited port.
        config: Extraction configuration.
        window_config: Optional override for window configuration.
            If None, uses config.window_config.

    Returns:
        ExtractionResult with S-parameter data.

    Raises:
        ValueError: If port configuration is invalid.

    REQ-M2-006: S-parameter extraction with windowing and FFT.
    """
    n_ports = config.n_ports
    if len(port_signals) != n_ports:
        raise ValueError(f"Expected {n_ports} port signals, got {len(port_signals)}")

    target_freqs = config.frequencies_hz()
    n_freq = len(target_freqs)
    z0 = config.reference_impedance_ohm

    # Use provided window config or config default
    win_config = window_config if window_config is not None else config.window_config

    # Map port IDs to indices
    port_id_to_idx = {spec.id: idx for idx, spec in enumerate(config.port_specs)}

    # Find excitation port index
    if excitation_port_id not in port_id_to_idx:
        raise ValueError(f"Excitation port '{excitation_port_id}' not found")
    excite_idx = port_id_to_idx[excitation_port_id]

    # Compute FFTs for each port with windowing (REQ-M2-006)
    port_ffts: dict[str, tuple[ComplexArray, ComplexArray, FloatArray]] = {}
    for signal in port_signals:
        dt = np.mean(np.diff(signal.time_s))
        n_samples = len(signal.time_s)

        # Create window function
        window = create_window(n_samples, win_config)

        # Apply windowing to voltage and current signals (REQ-M2-006)
        voltage_windowed = signal.voltage_v * window
        current_windowed = signal.current_a * window

        # FFT frequencies
        fft_freqs = np.fft.rfftfreq(n_samples, dt)

        # Compute incident and reflected waves from windowed signals
        # a = (V + Z0*I) / (2*sqrt(Z0)) - incident wave
        # b = (V - Z0*I) / (2*sqrt(Z0)) - reflected wave
        sqrt_z0 = np.sqrt(z0)
        v_fft = np.fft.rfft(voltage_windowed)
        i_fft = np.fft.rfft(current_windowed)

        a_fft = (v_fft + z0 * i_fft) / (2 * sqrt_z0)
        b_fft = (v_fft - z0 * i_fft) / (2 * sqrt_z0)

        port_ffts[signal.port_id] = (a_fft, b_fft, fft_freqs)

        logger.debug(
            "Port %s: applied %s window to %d samples, FFT range %.3g-%.3g Hz",
            signal.port_id,
            win_config.window_type.value,
            n_samples,
            fft_freqs[1] if len(fft_freqs) > 1 else 0,
            fft_freqs[-1] if len(fft_freqs) > 0 else 0,
        )

    # Initialize S-parameter matrix
    s_parameters = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)

    # Get excitation port FFT data
    a_excite, _, fft_freqs = port_ffts[excitation_port_id]

    # Compute S-parameters: Sij = bj / ai (when port i is excited)
    # This gives us one column of the S-matrix for single-port excitation
    for port_spec in config.port_specs:
        j = port_id_to_idx[port_spec.id]
        _, b_j, _ = port_ffts[port_spec.id]

        # Interpolate to target frequencies using linear interpolation
        # in real/imag domain for smoothness
        s_col = _interpolate_complex_spectrum(
            fft_freqs, b_j, a_excite, target_freqs
        )

        s_parameters[:, j, excite_idx] = s_col

    sparam_data = SParameterData(
        frequencies_hz=target_freqs,
        s_parameters=s_parameters,
        n_ports=n_ports,
        reference_impedance_ohm=z0,
    )

    # Compute canonical hash
    canonical = _sparam_canonical_json(sparam_data)
    canonical_hash = sha256_bytes(canonical.encode("utf-8"))

    # Compute metrics including window info
    metrics = _compute_extraction_metrics(sparam_data)
    metrics["windowing"] = {
        "window_type": win_config.window_type.value,
        "normalize": win_config.normalize,
    }
    if win_config.window_type == WindowType.KAISER:
        metrics["windowing"]["kaiser_beta"] = win_config.kaiser_beta
    if win_config.window_type == WindowType.TUKEY:
        metrics["windowing"]["tukey_alpha"] = win_config.tukey_alpha

    return ExtractionResult(
        s_parameters=sparam_data,
        extraction_config=config,
        source_files=[],
        canonical_hash=canonical_hash,
        metrics=metrics,
    )


def _interpolate_complex_spectrum(
    fft_freqs: FloatArray,
    numerator: ComplexArray,
    denominator: ComplexArray,
    target_freqs: FloatArray,
    min_denominator: float = 1e-15,
) -> ComplexArray:
    """Interpolate complex ratio spectrum to target frequencies.

    Performs interpolation of numerator/denominator ratio in the
    real/imaginary domain for smoother results than magnitude/phase.

    Args:
        fft_freqs: FFT frequency bins.
        numerator: Complex numerator spectrum.
        denominator: Complex denominator spectrum.
        target_freqs: Target frequency points.
        min_denominator: Minimum denominator magnitude to avoid division by zero.

    Returns:
        Interpolated complex ratio at target frequencies.

    REQ-M2-006: Proper interpolation for S-parameter extraction.
    """
    n_target = len(target_freqs)
    result = np.zeros(n_target, dtype=np.complex128)

    # Compute ratio at FFT frequencies (where denominator is significant)
    valid_mask = np.abs(denominator) > min_denominator
    ratio_valid = np.zeros_like(numerator)
    ratio_valid[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    # Only interpolate over frequency range with valid data
    if not np.any(valid_mask):
        logger.warning("No valid frequency bins for S-parameter interpolation")
        return result

    valid_freqs = fft_freqs[valid_mask]
    valid_ratio = ratio_valid[valid_mask]

    # Interpolate real and imaginary parts separately
    for f_idx, freq in enumerate(target_freqs):
        if freq < valid_freqs[0] or freq > valid_freqs[-1]:
            # Outside valid range - use nearest or zero
            if freq < valid_freqs[0]:
                result[f_idx] = valid_ratio[0]
            else:
                result[f_idx] = valid_ratio[-1]
        else:
            # Linear interpolation
            real_interp = np.interp(freq, valid_freqs, valid_ratio.real)
            imag_interp = np.interp(freq, valid_freqs, valid_ratio.imag)
            result[f_idx] = real_interp + 1j * imag_interp

    return result


@dataclass(slots=True)
class MultiPortSignalSet:
    """Signal data from multiple simulation runs with different port excitations.

    For complete 2-port S-parameter extraction (S11, S21, S12, S22), you need
    two simulation runs: one with port 1 excited and one with port 2 excited.
    This class holds signal data from all such runs.

    Attributes:
        excitation_sets: Dict mapping excitation_port_id to list of PortSignalData.

    REQ-M2-006: Multi-excitation data structure for complete S-matrix extraction.
    """

    excitation_sets: dict[str, list[PortSignalData]]

    def get_signals_for_excitation(self, excitation_port_id: str) -> list[PortSignalData]:
        """Get port signals for a specific excitation configuration."""
        if excitation_port_id not in self.excitation_sets:
            raise KeyError(f"No signals for excitation at port '{excitation_port_id}'")
        return self.excitation_sets[excitation_port_id]

    @property
    def excitation_port_ids(self) -> list[str]:
        """Get list of excitation port IDs."""
        return list(self.excitation_sets.keys())


def extract_full_sparam_matrix(
    signal_sets: MultiPortSignalSet,
    config: ExtractionConfig,
    *,
    window_config: WindowConfig | None = None,
) -> ExtractionResult:
    """Extract complete S-parameter matrix from multi-excitation simulation data.

    For a 2-port network, this computes all four S-parameters (S11, S21, S12, S22)
    by combining results from simulations with each port excited in turn.

    The S-matrix columns are filled by:
    - Column j from simulation with port j excited

    Args:
        signal_sets: Signal data from multiple simulations.
        config: Extraction configuration.
        window_config: Optional window configuration override.

    Returns:
        ExtractionResult with complete S-parameter matrix.

    Raises:
        ValueError: If signal sets don't match port configuration.

    REQ-M2-006: Complete S-parameter matrix extraction for 2-port networks.

    Example:
        >>> # Run two simulations: one with P1 excited, one with P2 excited
        >>> signal_sets = MultiPortSignalSet({
        ...     "P1": [signals_p1_excited_for_p1, signals_p1_excited_for_p2],
        ...     "P2": [signals_p2_excited_for_p1, signals_p2_excited_for_p2],
        ... })
        >>> result = extract_full_sparam_matrix(signal_sets, config)
        >>> # result.s_parameters contains [S11, S21, S12, S22]
    """
    n_ports = config.n_ports
    target_freqs = config.frequencies_hz()
    n_freq = len(target_freqs)
    z0 = config.reference_impedance_ohm

    # Use provided window config or config default
    win_config = window_config if window_config is not None else config.window_config

    # Map port IDs to indices
    port_id_to_idx = {spec.id: idx for idx, spec in enumerate(config.port_specs)}

    # Initialize complete S-parameter matrix
    s_parameters = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)

    # Process each excitation (each gives one column of S-matrix)
    for excite_port_id in signal_sets.excitation_port_ids:
        if excite_port_id not in port_id_to_idx:
            logger.warning("Excitation port '%s' not in config, skipping", excite_port_id)
            continue

        excite_idx = port_id_to_idx[excite_port_id]
        port_signals = signal_sets.get_signals_for_excitation(excite_port_id)

        if len(port_signals) != n_ports:
            raise ValueError(
                f"Expected {n_ports} signals for excitation at {excite_port_id}, "
                f"got {len(port_signals)}"
            )

        # Extract S-parameters for this excitation
        result = extract_sparams_from_port_signals(
            port_signals,
            excite_port_id,
            config,
            window_config=win_config,
        )

        # Copy this column into the full S-matrix
        s_parameters[:, :, excite_idx] = result.s_parameters.s_parameters[:, :, excite_idx]

        logger.debug("Filled column %d (port %s excitation)", excite_idx, excite_port_id)

    # Create final S-parameter data
    sparam_data = SParameterData(
        frequencies_hz=target_freqs,
        s_parameters=s_parameters,
        n_ports=n_ports,
        reference_impedance_ohm=z0,
        comment=f"Full {n_ports}-port S-matrix from multi-excitation extraction",
    )

    # Compute canonical hash
    canonical = _sparam_canonical_json(sparam_data)
    canonical_hash = sha256_bytes(canonical.encode("utf-8"))

    # Compute metrics
    metrics = _compute_extraction_metrics(sparam_data)
    metrics["extraction_method"] = "multi_excitation"
    metrics["excitation_ports"] = signal_sets.excitation_port_ids
    metrics["windowing"] = {
        "window_type": win_config.window_type.value,
        "normalize": win_config.normalize,
    }

    return ExtractionResult(
        s_parameters=sparam_data,
        extraction_config=config,
        source_files=[],
        canonical_hash=canonical_hash,
        metrics=metrics,
    )


def load_port_signals_json(json_path: Path) -> list[PortSignalData]:
    """Load port signal data from JSON file.

    Expected JSON format:
    {
        "time_ps": [0, 10, 20, ...],
        "ports": {
            "P1": {"voltage_v": [...], "current_a": [...]},
            "P2": {"voltage_v": [...], "current_a": [...]}
        }
    }

    Args:
        json_path: Path to JSON file.

    Returns:
        List of PortSignalData for each port.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Port signals file not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    time_ps = np.array(data["time_ps"], dtype=np.float64)
    time_s = time_ps * 1e-12  # Convert ps to s

    signals = []
    for port_id, port_data in data["ports"].items():
        signals.append(
            PortSignalData(
                port_id=port_id,
                time_s=time_s,
                voltage_v=np.array(port_data["voltage_v"], dtype=np.float64),
                current_a=np.array(port_data["current_a"], dtype=np.float64),
            )
        )

    return signals


def load_multi_port_signals_json(json_paths: dict[str, Path]) -> MultiPortSignalSet:
    """Load multi-port signal data from multiple JSON files.

    Each file contains signals from a simulation with a different port excited.

    Args:
        json_paths: Dict mapping excitation_port_id to JSON file path.

    Returns:
        MultiPortSignalSet with all signal data.

    REQ-M2-006: Load multi-excitation data for complete S-matrix extraction.
    """
    excitation_sets: dict[str, list[PortSignalData]] = {}

    for excite_port_id, json_path in json_paths.items():
        signals = load_port_signals_json(json_path)
        excitation_sets[excite_port_id] = signals

    return MultiPortSignalSet(excitation_sets=excitation_sets)


def apply_deembedding(
    sparam_data: SParameterData,
    deembed_distance_nm: float,
    epsilon_r_eff: float,
    port_indices: list[int] | None = None,
) -> SParameterData:
    """Apply de-embedding to shift reference planes.

    De-embedding removes the effect of transmission line sections between
    the port reference planes and the device under test.

    Args:
        sparam_data: Original S-parameter data.
        deembed_distance_nm: Distance to shift reference plane (nm).
        epsilon_r_eff: Effective dielectric constant.
        port_indices: Port indices to de-embed (1-based). If None, all ports.

    Returns:
        De-embedded S-parameter data.
    """
    if port_indices is None:
        port_indices = list(range(1, sparam_data.n_ports + 1))

    # Convert to 0-based indices
    port_idxs_0 = [p - 1 for p in port_indices]

    # Compute phase shift
    # phase = exp(-j * beta * L) where beta = 2*pi*f*sqrt(eps_eff)/c0
    distance_m = deembed_distance_nm * NM_TO_M
    sqrt_eps = np.sqrt(epsilon_r_eff)

    new_s = sparam_data.s_parameters.copy()

    for f_idx, freq in enumerate(sparam_data.frequencies_hz):
        # Phase constant
        beta = 2 * np.pi * freq * sqrt_eps / C0_M_PER_S
        phase_shift = np.exp(-1j * beta * distance_m)

        for p in port_idxs_0:
            # De-embed by multiplying relevant S-params by phase shift
            # Sij where i or j is the de-embedded port
            for i in range(sparam_data.n_ports):
                if i == p:
                    for j in range(sparam_data.n_ports):
                        new_s[f_idx, i, j] *= phase_shift
                elif p in port_idxs_0:
                    new_s[f_idx, i, p] *= phase_shift

    return SParameterData(
        frequencies_hz=sparam_data.frequencies_hz.copy(),
        s_parameters=new_s,
        n_ports=sparam_data.n_ports,
        reference_impedance_ohm=sparam_data.reference_impedance_ohm,
        comment=f"De-embedded {deembed_distance_nm} nm, eps_eff={epsilon_r_eff}",
    )


def write_extraction_result(
    result: ExtractionResult,
    output_dir: Path,
    base_name: str = "sparams",
) -> dict[str, Path]:
    """Write extraction result to files.

    Args:
        result: Extraction result to write.
        output_dir: Output directory.
        base_name: Base name for output files.

    Returns:
        Dictionary mapping format name to file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}

    fmt = result.extraction_config.output_format

    if fmt in ("touchstone", "both"):
        ts_path = output_dir / f"{base_name}.s{result.s_parameters.n_ports}p"
        write_touchstone(
            result.s_parameters,
            ts_path,
            TouchstoneOptions(
                frequency_unit=FrequencyUnit.HZ,
                parameter_format=SParameterFormat.RI,
                reference_impedance_ohm=result.s_parameters.reference_impedance_ohm,
            ),
        )
        output_paths["touchstone"] = ts_path

    if fmt in ("csv", "both"):
        csv_path = output_dir / f"{base_name}.csv"
        _write_sparam_csv(result.s_parameters, csv_path)
        output_paths["csv"] = csv_path

    # Always write metrics JSON
    metrics_path = output_dir / f"{base_name}_metrics.json"
    _write_metrics_json(result, metrics_path)
    output_paths["metrics"] = metrics_path

    if result.extraction_config.renormalize_to_ohms is not None:
        renorm_target = float(result.extraction_config.renormalize_to_ohms)
        renorm_data = renormalize_sparameters(result.s_parameters, renorm_target)
        renorm_label = _format_ohms_label(renorm_target)
        renorm_base = f"{base_name}_renorm_{renorm_label}ohm"

        renorm_config = ExtractionConfig(
            frequency_spec=result.extraction_config.frequency_spec,
            port_specs=result.extraction_config.port_specs,
            reference_impedance_ohm=renorm_target,
            renormalize_to_ohms=None,
            deembed_enabled=result.extraction_config.deembed_enabled,
            output_format=result.extraction_config.output_format,
            window_config=result.extraction_config.window_config,
        )
        renorm_canonical = _sparam_canonical_json(renorm_data)
        renorm_hash = sha256_bytes(renorm_canonical.encode("utf-8"))
        renorm_result = ExtractionResult(
            s_parameters=renorm_data,
            extraction_config=renorm_config,
            source_files=result.source_files,
            canonical_hash=renorm_hash,
            metrics=_compute_extraction_metrics(renorm_data),
        )

        if fmt in ("touchstone", "both"):
            ts_path = output_dir / f"{renorm_base}.s{renorm_data.n_ports}p"
            write_touchstone(
                renorm_data,
                ts_path,
                TouchstoneOptions(
                    frequency_unit=FrequencyUnit.HZ,
                    parameter_format=SParameterFormat.RI,
                    reference_impedance_ohm=renorm_data.reference_impedance_ohm,
                ),
            )
            output_paths["touchstone_renormalized"] = ts_path

        if fmt in ("csv", "both"):
            csv_path = output_dir / f"{renorm_base}.csv"
            _write_sparam_csv(renorm_data, csv_path)
            output_paths["csv_renormalized"] = csv_path

        renorm_metrics_path = output_dir / f"{renorm_base}_metrics.json"
        _write_metrics_json(renorm_result, renorm_metrics_path)
        output_paths["metrics_renormalized"] = renorm_metrics_path

    return output_paths


def build_manifest_entry(result: ExtractionResult) -> dict[str, Any]:
    """Build a manifest entry for the extraction result.

    Returns a dictionary suitable for inclusion in a simulation manifest.

    Args:
        result: Extraction result.

    Returns:
        Manifest entry dictionary.
    """
    return {
        "s_parameters": {
            "canonical_hash": result.canonical_hash,
            "n_ports": result.s_parameters.n_ports,
            "n_frequencies": result.s_parameters.n_frequencies,
            "f_min_hz": result.s_parameters.f_min_hz,
            "f_max_hz": result.s_parameters.f_max_hz,
            "reference_impedance_ohm": result.s_parameters.reference_impedance_ohm,
        },
        "extraction": {
            "source_files": result.source_files,
            "output_format": result.extraction_config.output_format,
            "deembed_enabled": result.extraction_config.deembed_enabled,
        },
        "metrics": result.metrics,
    }


# =============================================================================
# Private Helper Functions
# =============================================================================


def _format_ohms_label(ohms: float) -> str:
    """Format impedance for file naming."""
    label = f"{ohms:.6g}".replace(".", "p")
    return label


def _sparam_canonical_json(sparam_data: SParameterData) -> str:
    """Generate canonical JSON representation of S-parameter data.

    This is used for hashing to ensure deterministic identification.
    """
    # Round to reasonable precision for canonical form
    precision = 12

    freq_list = [round(float(f), precision) for f in sparam_data.frequencies_hz]

    s_list = []
    for f_idx in range(sparam_data.n_frequencies):
        freq_s = []
        for i in range(sparam_data.n_ports):
            row = []
            for j in range(sparam_data.n_ports):
                val = sparam_data.s_parameters[f_idx, i, j]
                row.append([round(val.real, precision), round(val.imag, precision)])
            freq_s.append(row)
        s_list.append(freq_s)

    payload = {
        "n_ports": sparam_data.n_ports,
        "reference_impedance_ohm": sparam_data.reference_impedance_ohm,
        "frequencies_hz": freq_list,
        "s_parameters": s_list,
    }

    return canonical_json_dumps(payload)


def _compute_extraction_metrics(sparam_data: SParameterData) -> dict[str, Any]:
    """Compute metrics from extracted S-parameters.

    Returns metrics like insertion loss, return loss, and bandwidth.
    """
    metrics: dict[str, Any] = {}

    # Basic counts
    metrics["n_ports"] = sparam_data.n_ports
    metrics["n_frequencies"] = sparam_data.n_frequencies
    metrics["f_min_hz"] = sparam_data.f_min_hz
    metrics["f_max_hz"] = sparam_data.f_max_hz

    # Return loss (S11) statistics
    s11_db = sparam_data.magnitude_db(1, 1)
    metrics["s11_min_db"] = float(np.min(s11_db))
    metrics["s11_max_db"] = float(np.max(s11_db))
    metrics["s11_mean_db"] = float(np.mean(s11_db))

    if sparam_data.n_ports >= 2:
        # Insertion loss (S21) statistics
        s21_db = sparam_data.magnitude_db(2, 1)
        metrics["s21_min_db"] = float(np.min(s21_db))
        metrics["s21_max_db"] = float(np.max(s21_db))
        metrics["s21_mean_db"] = float(np.mean(s21_db))

        # Reverse isolation (S12)
        s12_db = sparam_data.magnitude_db(1, 2)
        metrics["s12_mean_db"] = float(np.mean(s12_db))

        # Symmetry check (S21 vs S12)
        s21 = sparam_data.s21()
        s12 = sparam_data.s12()
        symmetry_error = np.abs(s21 - s12)
        metrics["symmetry_max_error"] = float(np.max(symmetry_error))

    # Passivity check: all eigenvalues of S*S^H <= 1
    passivity_violations = 0
    for f_idx in range(sparam_data.n_frequencies):
        s_mat = sparam_data.s_parameters[f_idx]
        eigenvalues = np.linalg.eigvalsh(s_mat @ s_mat.conj().T)
        if np.any(eigenvalues > 1.0 + 1e-10):
            passivity_violations += 1
    metrics["passivity_violations"] = passivity_violations
    metrics["is_passive"] = passivity_violations == 0

    return metrics


def _write_sparam_csv(sparam_data: SParameterData, path: Path) -> None:
    """Write S-parameters to CSV format."""
    n_ports = sparam_data.n_ports

    # Build header
    header_parts = ["freq_hz"]
    if n_ports == 2:
        order = [(1, 1), (2, 1), (1, 2), (2, 2)]
    else:
        order = [(i, j) for i in range(1, n_ports + 1) for j in range(1, n_ports + 1)]

    for i, j in order:
        header_parts.append(f"s{i}{j}_re")
        header_parts.append(f"s{i}{j}_im")

    header = ",".join(header_parts)

    lines = [header]
    for f_idx, freq in enumerate(sparam_data.frequencies_hz):
        row_parts = [f"{freq:.9g}"]
        for i, j in order:
            s_val = sparam_data.get_s(i, j)[f_idx]
            row_parts.append(f"{s_val.real:.9g}")
            row_parts.append(f"{s_val.imag:.9g}")
        lines.append(",".join(row_parts))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_metrics_json(result: ExtractionResult, path: Path) -> None:
    """Write extraction metrics to JSON file."""
    payload = {
        "canonical_hash": result.canonical_hash,
        "source_files": result.source_files,
        "metrics": result.metrics,
        "config": {
            "n_ports": result.extraction_config.n_ports,
            "n_frequencies": result.extraction_config.n_frequencies,
            "f_start_hz": float(result.extraction_config.frequency_spec.f_start_hz),
            "f_stop_hz": float(result.extraction_config.frequency_spec.f_stop_hz),
            "reference_impedance_ohm": result.extraction_config.reference_impedance_ohm,
            "renormalize_to_ohms": result.extraction_config.renormalize_to_ohms,
            "output_format": result.extraction_config.output_format,
        },
    }
    text = canonical_json_dumps(payload)
    path.write_text(f"{text}\n", encoding="utf-8")


# =============================================================================
# High-Level Extraction API
# =============================================================================


def extract_sparams(
    sim_output_dir: Path,
    config: ExtractionConfig,
    *,
    prefer_touchstone: bool = True,
) -> ExtractionResult:
    """High-level S-parameter extraction from simulation outputs.

    Automatically detects available output format and extracts S-parameters.

    Args:
        sim_output_dir: Directory containing simulation outputs.
        config: Extraction configuration.
        prefer_touchstone: Prefer Touchstone over CSV if both available.

    Returns:
        ExtractionResult with S-parameter data.

    Raises:
        FileNotFoundError: If no S-parameter output found.
    """
    # Look for Touchstone files
    touchstone_patterns = [f"*.s{config.n_ports}p", "*.s?p", "sparams.s?p"]
    touchstone_path = None
    for pattern in touchstone_patterns:
        matches = list(sim_output_dir.glob(pattern))
        if matches:
            touchstone_path = matches[0]
            break

    # Look for CSV files
    csv_path = sim_output_dir / "sparams.csv"
    if not csv_path.exists():
        csv_path = None

    # Extract based on available files
    if prefer_touchstone and touchstone_path:
        logger.info("Extracting S-parameters from Touchstone: %s", touchstone_path)
        return extract_sparams_from_touchstone(touchstone_path, config)
    elif csv_path:
        logger.info("Extracting S-parameters from CSV: %s", csv_path)
        return extract_sparams_from_csv(csv_path, config)
    elif touchstone_path:
        logger.info("Extracting S-parameters from Touchstone: %s", touchstone_path)
        return extract_sparams_from_touchstone(touchstone_path, config)
    else:
        # Try to extract from port signals
        port_signals_path = sim_output_dir / "port_signals.json"
        if port_signals_path.exists():
            logger.info("Extracting S-parameters from port signals: %s", port_signals_path)
            signals = load_port_signals_json(port_signals_path)

            # Find excited port
            excite_port = None
            for spec in config.port_specs:
                if spec.excite:
                    excite_port = spec.id
                    break
            if excite_port is None:
                excite_port = config.port_specs[0].id

            return extract_sparams_from_port_signals(signals, excite_port, config)

        raise FileNotFoundError(
            f"No S-parameter output found in {sim_output_dir}. Expected .s?p, sparams.csv, or port_signals.json"
        )
