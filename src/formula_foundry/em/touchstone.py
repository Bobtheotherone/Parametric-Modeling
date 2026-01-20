"""Touchstone file I/O for S-parameter data.

This module provides reading and writing of Touchstone format files (.sNp)
with support for frequency-dependent S-parameter arrays. The primary focus
is on .s2p (2-port) files commonly used for via transition coupon analysis.

Supported features:
- Touchstone 1.0 and 2.0 formats
- Frequency unit conversion (Hz, kHz, MHz, GHz)
- S-parameter format conversion (dB/ang, mag/ang, real/imag)
- Frequency-dependent complex S-parameter arrays
- Optional scikit-rf integration for validation

All internal representations use:
- Frequency in Hz (integer for determinism when possible)
- S-parameters as complex numpy arrays (magnitude + phase in radians internally)

References:
- Touchstone File Specification: IBIS Open Forum
- scikit-rf documentation: https://scikit-rf.readthedocs.io/
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass

# Type aliases for clarity
ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


class FrequencyUnit(Enum):
    """Frequency unit specifier in Touchstone files."""

    HZ = "Hz"
    KHZ = "kHz"
    MHZ = "MHz"
    GHZ = "GHz"

    @property
    def multiplier(self) -> float:
        """Return the multiplier to convert to Hz."""
        multipliers = {
            FrequencyUnit.HZ: 1.0,
            FrequencyUnit.KHZ: 1e3,
            FrequencyUnit.MHZ: 1e6,
            FrequencyUnit.GHZ: 1e9,
        }
        return multipliers[self]


class SParameterFormat(Enum):
    """S-parameter data format in Touchstone files."""

    DB = "DB"  # dB magnitude and angle in degrees
    MA = "MA"  # Linear magnitude and angle in degrees
    RI = "RI"  # Real and imaginary parts


class NetworkType(Enum):
    """Network parameter type."""

    S = "S"  # S-parameters (scattering)
    Y = "Y"  # Y-parameters (admittance)
    Z = "Z"  # Z-parameters (impedance)
    H = "H"  # H-parameters (hybrid)
    G = "G"  # G-parameters (inverse hybrid)


@dataclass(frozen=True, slots=True)
class TouchstoneOptions:
    """Configuration options for Touchstone file format.

    Attributes:
        frequency_unit: Unit for frequency values in the file.
        parameter_format: Format for S-parameter data.
        network_type: Type of network parameters (typically S).
        reference_impedance_ohm: Reference impedance in ohms (typically 50).
        version: Touchstone format version ("1.0" or "2.0").
    """

    frequency_unit: FrequencyUnit = FrequencyUnit.GHZ
    parameter_format: SParameterFormat = SParameterFormat.RI
    network_type: NetworkType = NetworkType.S
    reference_impedance_ohm: float = 50.0
    version: str = "1.0"

    def __post_init__(self) -> None:
        if self.reference_impedance_ohm <= 0:
            raise ValueError("reference_impedance_ohm must be positive")
        if self.version not in ("1.0", "2.0"):
            raise ValueError("version must be '1.0' or '2.0'")


@dataclass(slots=True)
class SParameterData:
    """Container for frequency-dependent S-parameter data.

    This is the primary data structure for S-parameter storage and manipulation.
    All frequency values are stored in Hz for consistency. S-parameters are
    stored as complex arrays with shape (n_frequencies, n_ports, n_ports).

    Attributes:
        frequencies_hz: 1D array of frequency points in Hz.
        s_parameters: 3D complex array of S-parameters [freq, port_out, port_in].
        n_ports: Number of ports in the network.
        reference_impedance_ohm: Reference impedance in ohms.
        comment: Optional comment/description for the data.
    """

    frequencies_hz: FloatArray
    s_parameters: ComplexArray
    n_ports: int
    reference_impedance_ohm: float = 50.0
    comment: str = ""
    _metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate array shapes and consistency."""
        if self.frequencies_hz.ndim != 1:
            raise ValueError("frequencies_hz must be 1D array")
        if self.s_parameters.ndim != 3:
            raise ValueError("s_parameters must be 3D array [freq, port_out, port_in]")

        n_freq = len(self.frequencies_hz)
        expected_shape = (n_freq, self.n_ports, self.n_ports)
        if self.s_parameters.shape != expected_shape:
            raise ValueError(f"s_parameters shape {self.s_parameters.shape} does not match expected {expected_shape}")

        if self.reference_impedance_ohm <= 0:
            raise ValueError("reference_impedance_ohm must be positive")

        # Ensure frequencies are sorted ascending
        if not np.all(np.diff(self.frequencies_hz) > 0):
            raise ValueError("frequencies_hz must be strictly increasing")

    @property
    def n_frequencies(self) -> int:
        """Return the number of frequency points."""
        return len(self.frequencies_hz)

    @property
    def f_min_hz(self) -> float:
        """Return the minimum frequency in Hz."""
        return float(self.frequencies_hz[0])

    @property
    def f_max_hz(self) -> float:
        """Return the maximum frequency in Hz."""
        return float(self.frequencies_hz[-1])

    def get_s(self, i: int, j: int) -> ComplexArray:
        """Get S[i,j] as a 1D array across all frequencies.

        Port indices are 1-based to match Touchstone convention.

        Args:
            i: Output port (1-based).
            j: Input port (1-based).

        Returns:
            1D complex array of S[i,j] at each frequency.
        """
        if not (1 <= i <= self.n_ports and 1 <= j <= self.n_ports):
            raise ValueError(f"Port indices must be 1 to {self.n_ports}")
        return self.s_parameters[:, i - 1, j - 1]

    def s11(self) -> ComplexArray:
        """Return S11 (input reflection coefficient) across frequencies."""
        return self.get_s(1, 1)

    def s21(self) -> ComplexArray:
        """Return S21 (forward transmission) across frequencies."""
        if self.n_ports < 2:
            raise ValueError("S21 requires at least 2 ports")
        return self.get_s(2, 1)

    def s12(self) -> ComplexArray:
        """Return S12 (reverse transmission) across frequencies."""
        if self.n_ports < 2:
            raise ValueError("S12 requires at least 2 ports")
        return self.get_s(1, 2)

    def s22(self) -> ComplexArray:
        """Return S22 (output reflection coefficient) across frequencies."""
        if self.n_ports < 2:
            raise ValueError("S22 requires at least 2 ports")
        return self.get_s(2, 2)

    def magnitude_db(self, i: int, j: int) -> FloatArray:
        """Get magnitude of S[i,j] in dB.

        Args:
            i: Output port (1-based).
            j: Input port (1-based).

        Returns:
            1D array of magnitudes in dB.
        """
        s_ij = self.get_s(i, j)
        mag = np.abs(s_ij)
        # Avoid log(0) by using a floor
        mag_clipped = np.maximum(mag, 1e-15)
        return 20.0 * np.log10(mag_clipped)

    def phase_deg(self, i: int, j: int) -> FloatArray:
        """Get phase of S[i,j] in degrees.

        Args:
            i: Output port (1-based).
            j: Input port (1-based).

        Returns:
            1D array of phases in degrees.
        """
        s_ij = self.get_s(i, j)
        return np.degrees(np.angle(s_ij))

    def interpolate(self, new_frequencies_hz: FloatArray) -> SParameterData:
        """Interpolate S-parameters to new frequency points.

        Uses linear interpolation in the real/imaginary domain.

        Args:
            new_frequencies_hz: Target frequency points in Hz.

        Returns:
            New SParameterData with interpolated values.
        """
        n_new = len(new_frequencies_hz)
        new_s = np.zeros((n_new, self.n_ports, self.n_ports), dtype=np.complex128)

        for i in range(self.n_ports):
            for j in range(self.n_ports):
                s_ij = self.s_parameters[:, i, j]
                # Interpolate real and imaginary parts separately
                real_interp = np.interp(new_frequencies_hz, self.frequencies_hz, s_ij.real)
                imag_interp = np.interp(new_frequencies_hz, self.frequencies_hz, s_ij.imag)
                new_s[:, i, j] = real_interp + 1j * imag_interp

        return SParameterData(
            frequencies_hz=new_frequencies_hz.copy(),
            s_parameters=new_s,
            n_ports=self.n_ports,
            reference_impedance_ohm=self.reference_impedance_ohm,
            comment=self.comment,
        )


def _parse_option_line(line: str) -> TouchstoneOptions:
    """Parse a Touchstone option line (starts with #).

    The option line format is:
    # <freq_unit> <param_type> <format> R <impedance>

    Args:
        line: The option line (without leading #).

    Returns:
        TouchstoneOptions parsed from the line.
    """
    # Normalize whitespace and case
    tokens = line.upper().split()

    # Defaults per Touchstone spec
    freq_unit = FrequencyUnit.GHZ
    param_format = SParameterFormat.MA
    network_type = NetworkType.S
    ref_impedance = 50.0

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Check for frequency unit
        if token in ("HZ", "KHZ", "MHZ", "GHZ"):
            freq_unit = FrequencyUnit[token]
        # Check for network type
        elif token in ("S", "Y", "Z", "H", "G"):
            network_type = NetworkType[token]
        # Check for format
        elif token in ("DB", "MA", "RI"):
            param_format = SParameterFormat[token]
        # Check for reference impedance
        elif token == "R" and i + 1 < len(tokens):
            try:
                ref_impedance = float(tokens[i + 1])
                i += 1
            except ValueError:
                pass

        i += 1

    return TouchstoneOptions(
        frequency_unit=freq_unit,
        parameter_format=param_format,
        network_type=network_type,
        reference_impedance_ohm=ref_impedance,
    )


def _convert_to_complex(val1: float, val2: float, fmt: SParameterFormat) -> complex:
    """Convert a value pair to complex based on format.

    Args:
        val1: First value (dB/mag/real depending on format).
        val2: Second value (angle/imag depending on format).
        fmt: The S-parameter format.

    Returns:
        Complex S-parameter value.
    """
    if fmt == SParameterFormat.RI:
        return complex(val1, val2)
    elif fmt == SParameterFormat.MA:
        # MA: magnitude, angle in degrees
        return val1 * np.exp(1j * np.radians(val2))
    elif fmt == SParameterFormat.DB:
        # DB: dB magnitude, angle in degrees
        mag = 10 ** (val1 / 20.0)
        return mag * np.exp(1j * np.radians(val2))
    else:
        raise ValueError(f"Unknown format: {fmt}")


def _complex_to_format(value: complex, fmt: SParameterFormat) -> tuple[float, float]:
    """Convert a complex value to the specified format pair.

    Args:
        value: Complex S-parameter value.
        fmt: Target S-parameter format.

    Returns:
        Tuple of (val1, val2) in the specified format.
    """
    if fmt == SParameterFormat.RI:
        return (value.real, value.imag)
    elif fmt == SParameterFormat.MA:
        mag = abs(value)
        ang = np.degrees(np.angle(value))
        return (mag, ang)
    elif fmt == SParameterFormat.DB:
        mag = abs(value)
        mag_db = 20.0 * np.log10(max(mag, 1e-15))
        ang = np.degrees(np.angle(value))
        return (mag_db, ang)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def read_touchstone(file_path: str | Path) -> SParameterData:
    """Read a Touchstone file and return S-parameter data.

    Supports Touchstone 1.0 (.s1p, .s2p, .s3p, .s4p, etc.) format.

    Args:
        file_path: Path to the Touchstone file.

    Returns:
        SParameterData containing the parsed S-parameters.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Touchstone file not found: {path}")

    # Determine number of ports from extension
    ext = path.suffix.lower()
    match = re.match(r"\.s(\d+)p", ext)
    if not match:
        raise ValueError(f"Invalid Touchstone extension: {ext}")
    n_ports = int(match.group(1))

    # Read file content
    with open(path, encoding="utf-8") as f:
        return _parse_touchstone_content(f, n_ports)


def read_touchstone_from_string(content: str, n_ports: int) -> SParameterData:
    """Read Touchstone data from a string.

    Args:
        content: Touchstone file content as string.
        n_ports: Number of ports in the network.

    Returns:
        SParameterData containing the parsed S-parameters.
    """
    import io

    return _parse_touchstone_content(io.StringIO(content), n_ports)


def _parse_touchstone_content(f: TextIO, n_ports: int) -> SParameterData:
    """Parse Touchstone content from a file-like object.

    Args:
        f: File-like object with Touchstone content.
        n_ports: Number of ports.

    Returns:
        SParameterData with parsed values.
    """
    options: TouchstoneOptions | None = None
    comments: list[str] = []
    data_lines: list[str] = []

    for line in f:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Comment line (starts with !)
        if line.startswith("!"):
            comments.append(line[1:].strip())
            continue

        # Option line (starts with #)
        if line.startswith("#"):
            options = _parse_option_line(line[1:])
            continue

        # Data line - remove inline comments
        if "!" in line:
            line = line[: line.index("!")].strip()

        if line:
            data_lines.append(line)

    if options is None:
        # Use defaults
        options = TouchstoneOptions()

    # Parse data
    frequencies: list[float] = []
    s_data: list[list[list[complex]]] = []

    # Number of S-parameters per frequency point
    n_ports * n_ports
    # Number of value pairs per frequency point

    # Join all data lines and split into tokens
    all_tokens: list[str] = []
    for line in data_lines:
        all_tokens.extend(line.split())

    # Parse tokens
    idx = 0
    while idx < len(all_tokens):
        # First token is frequency
        freq = float(all_tokens[idx]) * options.frequency_unit.multiplier
        frequencies.append(freq)
        idx += 1

        # Read n_pairs value pairs
        s_matrix: list[list[complex]] = [[complex(0, 0)] * n_ports for _ in range(n_ports)]

        if n_ports == 1:
            # S11 only
            val1 = float(all_tokens[idx])
            val2 = float(all_tokens[idx + 1])
            s_matrix[0][0] = _convert_to_complex(val1, val2, options.parameter_format)
            idx += 2
        elif n_ports == 2:
            # Order: S11, S21, S12, S22
            order = [(0, 0), (1, 0), (0, 1), (1, 1)]
            for out_idx, in_idx in order:
                val1 = float(all_tokens[idx])
                val2 = float(all_tokens[idx + 1])
                s_matrix[out_idx][in_idx] = _convert_to_complex(val1, val2, options.parameter_format)
                idx += 2
        else:
            # For n>2, data is row-major: S11, S12, ..., S1n, S21, S22, ...
            for out_idx in range(n_ports):
                for in_idx in range(n_ports):
                    val1 = float(all_tokens[idx])
                    val2 = float(all_tokens[idx + 1])
                    s_matrix[out_idx][in_idx] = _convert_to_complex(val1, val2, options.parameter_format)
                    idx += 2

        s_data.append(s_matrix)

    # Convert to numpy arrays
    frequencies_hz = np.array(frequencies, dtype=np.float64)
    n_freq = len(frequencies)
    s_parameters = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)

    for f_idx, s_matrix in enumerate(s_data):
        for i in range(n_ports):
            for j in range(n_ports):
                s_parameters[f_idx, i, j] = s_matrix[i][j]

    return SParameterData(
        frequencies_hz=frequencies_hz,
        s_parameters=s_parameters,
        n_ports=n_ports,
        reference_impedance_ohm=options.reference_impedance_ohm,
        comment="\n".join(comments) if comments else "",
    )


def write_touchstone(
    data: SParameterData,
    file_path: str | Path,
    options: TouchstoneOptions | None = None,
) -> None:
    """Write S-parameter data to a Touchstone file.

    Args:
        data: S-parameter data to write.
        file_path: Output file path.
        options: Formatting options (defaults to GHz, RI format).
    """
    path = Path(file_path)
    if options is None:
        options = TouchstoneOptions()

    with open(path, "w", encoding="utf-8") as f:
        _write_touchstone_content(data, f, options)


def write_touchstone_to_string(
    data: SParameterData,
    options: TouchstoneOptions | None = None,
) -> str:
    """Write S-parameter data to a Touchstone-formatted string.

    Args:
        data: S-parameter data to write.
        options: Formatting options.

    Returns:
        Touchstone file content as string.
    """
    import io

    if options is None:
        options = TouchstoneOptions()

    output = io.StringIO()
    _write_touchstone_content(data, output, options)
    return output.getvalue()


def _write_touchstone_content(
    data: SParameterData,
    f: TextIO,
    options: TouchstoneOptions,
) -> None:
    """Write Touchstone content to a file-like object.

    Args:
        data: S-parameter data.
        f: File-like object for output.
        options: Formatting options.
    """
    n_ports = data.n_ports

    # Write comment header
    if data.comment:
        for line in data.comment.split("\n"):
            f.write(f"! {line}\n")
    f.write(f"! {n_ports}-port S-parameters\n")
    f.write("! Generated by formula_foundry.em.touchstone\n")

    # Write option line
    f.write(
        f"# {options.frequency_unit.value} "
        f"{options.network_type.value} "
        f"{options.parameter_format.value} "
        f"R {options.reference_impedance_ohm:.1f}\n"
    )

    # Write data
    freq_divisor = options.frequency_unit.multiplier

    for f_idx in range(data.n_frequencies):
        freq = data.frequencies_hz[f_idx] / freq_divisor

        if n_ports == 1:
            # Single line: freq S11
            val1, val2 = _complex_to_format(data.s_parameters[f_idx, 0, 0], options.parameter_format)
            f.write(f"{freq:.9g} {val1:.9g} {val2:.9g}\n")

        elif n_ports == 2:
            # Order: S11, S21, S12, S22 on single line
            order = [(0, 0), (1, 0), (0, 1), (1, 1)]
            parts = [f"{freq:.9g}"]
            for out_idx, in_idx in order:
                val1, val2 = _complex_to_format(data.s_parameters[f_idx, out_idx, in_idx], options.parameter_format)
                parts.append(f"{val1:.9g}")
                parts.append(f"{val2:.9g}")
            f.write(" ".join(parts) + "\n")

        else:
            # For n>2, write frequency on first line, then matrix row-by-row
            f.write(f"{freq:.9g}")
            for out_idx in range(n_ports):
                if out_idx > 0:
                    f.write("\n")
                for in_idx in range(n_ports):
                    val1, val2 = _complex_to_format(
                        data.s_parameters[f_idx, out_idx, in_idx],
                        options.parameter_format,
                    )
                    f.write(f" {val1:.9g} {val2:.9g}")
            f.write("\n")


# =============================================================================
# Scikit-RF Integration
# =============================================================================


def validate_with_skrf(data: SParameterData, tolerance_db: float = 0.01) -> bool:
    """Validate S-parameter data using scikit-rf.

    This function creates a scikit-rf Network from the data and performs
    basic validation checks. It's useful for verifying that our I/O
    produces correct results.

    Args:
        data: S-parameter data to validate.
        tolerance_db: Maximum allowed difference in dB for passivity check.

    Returns:
        True if validation passes, False otherwise.

    Raises:
        ImportError: If scikit-rf is not installed.
    """
    try:
        import skrf  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError("scikit-rf is required for validation. Install with: pip install scikit-rf") from e

    # Create skrf Network
    network = to_skrf_network(data)

    # Basic checks
    # 1. Verify frequency array matches
    if len(network.f) != data.n_frequencies:
        return False

    # 2. Verify S-parameters shape
    if network.s.shape != data.s_parameters.shape:
        return False

    # 3. Check for NaN/Inf values
    return not (np.any(np.isnan(network.s)) or np.any(np.isinf(network.s)))


def to_skrf_network(data: SParameterData) -> skrf.Network:  # type: ignore[name-defined]
    """Convert SParameterData to a scikit-rf Network object.

    Args:
        data: S-parameter data to convert.

    Returns:
        scikit-rf Network object.

    Raises:
        ImportError: If scikit-rf is not installed.
    """
    try:
        import skrf  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError("scikit-rf is required. Install with: pip install scikit-rf") from e

    # Create frequency object
    frequency = skrf.Frequency.from_f(data.frequencies_hz, unit="Hz")

    # Create Network
    network = skrf.Network(
        frequency=frequency,
        s=data.s_parameters,
        z0=data.reference_impedance_ohm,
        name=data.comment[:50] if data.comment else "formula_foundry_network",
    )

    return network


def from_skrf_network(network: skrf.Network) -> SParameterData:  # type: ignore[name-defined]
    """Convert a scikit-rf Network to SParameterData.

    Args:
        network: scikit-rf Network object.

    Returns:
        SParameterData with the network's S-parameters.
    """
    # Get frequency in Hz
    frequencies_hz = network.f.copy()

    # Get S-parameters
    s_parameters = network.s.copy()
    n_ports = network.number_of_ports

    # Get reference impedance (may be frequency-dependent in skrf)
    # Take the first value as our reference
    z0 = network.z0
    if hasattr(z0, "__len__"):
        ref_impedance = float(z0[0, 0]) if z0.ndim > 1 else float(z0[0])
    else:
        ref_impedance = float(z0)

    return SParameterData(
        frequencies_hz=frequencies_hz,
        s_parameters=s_parameters,
        n_ports=n_ports,
        reference_impedance_ohm=ref_impedance,
        comment=network.name if network.name else "",
    )


# =============================================================================
# Utility Functions
# =============================================================================


def create_empty_sparam_data(
    frequencies_hz: FloatArray,
    n_ports: int,
    reference_impedance_ohm: float = 50.0,
) -> SParameterData:
    """Create an empty SParameterData container.

    All S-parameters are initialized to zero.

    Args:
        frequencies_hz: Frequency points in Hz.
        n_ports: Number of ports.
        reference_impedance_ohm: Reference impedance in ohms.

    Returns:
        SParameterData with zero S-parameters.
    """
    n_freq = len(frequencies_hz)
    s_parameters = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)

    return SParameterData(
        frequencies_hz=frequencies_hz.copy(),
        s_parameters=s_parameters,
        n_ports=n_ports,
        reference_impedance_ohm=reference_impedance_ohm,
    )


def create_thru_sparam_data(
    frequencies_hz: FloatArray,
    insertion_loss_db: float = 0.0,
    reference_impedance_ohm: float = 50.0,
) -> SParameterData:
    """Create S-parameter data for an ideal thru (2-port transmission line).

    Args:
        frequencies_hz: Frequency points in Hz.
        insertion_loss_db: Insertion loss in dB (negative for loss).
        reference_impedance_ohm: Reference impedance in ohms.

    Returns:
        SParameterData for a 2-port thru.
    """
    n_freq = len(frequencies_hz)
    s_parameters = np.zeros((n_freq, 2, 2), dtype=np.complex128)

    # S21 = S12 = transmission coefficient
    transmission = 10 ** (insertion_loss_db / 20.0)
    s_parameters[:, 1, 0] = transmission  # S21
    s_parameters[:, 0, 1] = transmission  # S12

    # S11 = S22 = 0 (perfect match)

    return SParameterData(
        frequencies_hz=frequencies_hz.copy(),
        s_parameters=s_parameters,
        n_ports=2,
        reference_impedance_ohm=reference_impedance_ohm,
        comment="Ideal thru",
    )


def merge_sparam_data(
    data_list: list[SParameterData],
    common_frequencies_hz: FloatArray | None = None,
) -> list[SParameterData]:
    """Interpolate multiple SParameterData to common frequency points.

    This is useful for comparing S-parameters from different sources
    (e.g., simulation vs measurement) that may have different frequency grids.

    Args:
        data_list: List of SParameterData to merge.
        common_frequencies_hz: Target frequency grid (if None, uses first data's grid).

    Returns:
        List of SParameterData all on the same frequency grid.
    """
    if not data_list:
        return []

    if common_frequencies_hz is None:
        common_frequencies_hz = data_list[0].frequencies_hz.copy()

    return [d.interpolate(common_frequencies_hz) for d in data_list]
