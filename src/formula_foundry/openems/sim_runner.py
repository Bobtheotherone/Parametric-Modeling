from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from formula_foundry.substrate import canonical_json_dumps, sha256_bytes

from .convert import simulation_canonical_json
from .geometry import GeometrySpec, geometry_canonical_json
from .runner import OpenEMSRunner
from .spec import FrequencySpec, SimulationSpec

SimulationSolverMode = Literal["stub", "cli"]

_MANIFEST_SCHEMA_VERSION = 1
_HASH_CHUNK_SIZE = 1024 * 1024
_DEFAULT_TIMEOUT_SEC = 3600  # 1 hour default timeout

logger = logging.getLogger(__name__)


class SimulationError(Exception):
    """Base exception for simulation errors."""

    pass


class SimulationTimeoutError(SimulationError):
    """Raised when simulation exceeds timeout."""

    pass


class SimulationExecutionError(SimulationError):
    """Raised when openEMS execution fails."""

    def __init__(self, message: str, returncode: int, stdout: str, stderr: str) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True)
class SimulationResult:
    """Result of a simulation run.

    Attributes:
        output_dir: Root output directory.
        outputs_dir: Directory containing simulation outputs.
        manifest_path: Path to the simulation manifest JSON.
        cache_hit: Whether the result was retrieved from cache.
        simulation_hash: SHA256 hash of the simulation inputs.
        manifest_hash: SHA256 hash of the manifest content.
        output_hashes: Dictionary mapping output file paths to their SHA256 hashes.
        execution_time_sec: Execution time in seconds (None for cached results).
        sparam_path: Path to S-parameter output file if generated.
    """

    output_dir: Path
    outputs_dir: Path
    manifest_path: Path
    cache_hit: bool
    simulation_hash: str
    manifest_hash: str
    output_hashes: dict[str, str]
    execution_time_sec: float | None = None
    sparam_path: Path | None = None


class SimulationRunner:
    """Runner for openEMS FDTD simulations.

    This class manages simulation execution with:
    - Stub mode for testing without actual openEMS
    - CLI mode for real openEMS execution
    - Caching based on input hashes
    - Manifest generation for reproducibility
    - S-parameter output parsing

    Attributes:
        mode: Simulation mode ("stub" or "cli").
        openems_runner: OpenEMSRunner instance for CLI mode.
        timeout_sec: Timeout in seconds for simulation execution.
    """

    def __init__(
        self,
        *,
        mode: SimulationSolverMode = "stub",
        openems_runner: OpenEMSRunner | None = None,
        timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    ) -> None:
        """Initialize the simulation runner.

        Args:
            mode: Simulation mode ("stub" or "cli").
            openems_runner: OpenEMSRunner instance required for CLI mode.
            timeout_sec: Timeout in seconds for simulation execution.

        Raises:
            ValueError: If mode is invalid or openems_runner missing for CLI mode.
        """
        if mode not in ("stub", "cli"):
            raise ValueError(f"Unsupported simulation runner mode: {mode}")
        if mode == "cli" and openems_runner is None:
            raise ValueError("openems_runner is required for cli mode")
        self.mode = mode
        self.openems_runner = openems_runner
        self.timeout_sec = timeout_sec

    def run(
        self,
        spec: SimulationSpec,
        geometry: GeometrySpec,
        *,
        output_dir: Path,
        openems_args: Sequence[str] | None = None,
        timeout_sec: float | None = None,
    ) -> SimulationResult:
        """Run an openEMS simulation.

        Args:
            spec: Simulation specification.
            geometry: Geometry specification for the simulation.
            output_dir: Directory for simulation outputs.
            openems_args: Command-line arguments for openEMS.
            timeout_sec: Override timeout for this run (None uses default).

        Returns:
            SimulationResult containing output paths, hashes, and metadata.

        Raises:
            SimulationTimeoutError: If simulation exceeds timeout.
            SimulationExecutionError: If openEMS execution fails.
        """
        import time

        resolved_output_dir = output_dir.resolve()
        outputs_dir = resolved_output_dir / spec.output.outputs_dir
        manifest_path = resolved_output_dir / "simulation_manifest.json"

        solver_payload = self._solver_payload(openems_args)
        spec_hash = _spec_hash(spec)
        geometry_hash = _geometry_hash(geometry)
        simulation_hash = _simulation_hash(spec_hash, geometry_hash, solver_payload)

        # Check cache
        cached_manifest = _load_manifest(manifest_path)
        if cached_manifest is not None and _is_cache_hit(
            cached_manifest,
            simulation_hash=simulation_hash,
            spec_hash=spec_hash,
            geometry_hash=geometry_hash,
            solver_payload=solver_payload,
            output_dir=resolved_output_dir,
            outputs_dir=outputs_dir,
        ):
            output_hashes = _hash_outputs(resolved_output_dir, outputs_dir)
            manifest_hash = _manifest_hash(cached_manifest)
            sparam_path = _find_sparam_file(outputs_dir, spec)
            logger.info("Cache hit for simulation %s", simulation_hash[:12])
            return SimulationResult(
                output_dir=resolved_output_dir,
                outputs_dir=outputs_dir,
                manifest_path=manifest_path,
                cache_hit=True,
                simulation_hash=simulation_hash,
                manifest_hash=manifest_hash,
                output_hashes=output_hashes,
                sparam_path=sparam_path,
            )

        # Run simulation
        _clear_outputs_dir(outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.perf_counter()
        effective_timeout = timeout_sec if timeout_sec is not None else self.timeout_sec

        if self.mode == "stub":
            self._run_stub(spec, outputs_dir, simulation_hash)
        else:
            self._run_cli(outputs_dir, openems_args, timeout_sec=effective_timeout)

        execution_time = time.perf_counter() - start_time
        logger.info(
            "Simulation %s completed in %.2f seconds",
            simulation_hash[:12],
            execution_time,
        )

        # Build manifest and result
        output_hashes = _hash_outputs(resolved_output_dir, outputs_dir)
        manifest = _build_manifest(
            spec=spec,
            simulation_hash=simulation_hash,
            spec_hash=spec_hash,
            geometry_hash=geometry_hash,
            solver_payload=solver_payload,
            output_hashes=output_hashes,
            execution_time_sec=execution_time,
        )
        _write_manifest(manifest_path, manifest)
        manifest_hash = _manifest_hash(manifest)
        sparam_path = _find_sparam_file(outputs_dir, spec)

        return SimulationResult(
            output_dir=resolved_output_dir,
            outputs_dir=outputs_dir,
            manifest_path=manifest_path,
            cache_hit=False,
            simulation_hash=simulation_hash,
            manifest_hash=manifest_hash,
            output_hashes=output_hashes,
            execution_time_sec=execution_time,
            sparam_path=sparam_path,
        )

    def _solver_payload(self, openems_args: Sequence[str] | None) -> dict[str, Any]:
        if self.mode == "stub":
            return {"mode": "stub"}
        if self.openems_runner is None:
            raise ValueError("openems_runner is required for cli mode")
        return {
            "mode": "cli",
            "runner_mode": self.openems_runner.mode,
            "openems_bin": self.openems_runner.openems_bin,
            "docker_image": self.openems_runner.docker_image,
            "openems_args": list(openems_args or []),
        }

    def _run_stub(self, spec: SimulationSpec, outputs_dir: Path, simulation_hash: str) -> None:
        seed = int(simulation_hash[:8], 16)
        _write_stub_outputs(spec, outputs_dir, seed)

    def _run_cli(
        self,
        outputs_dir: Path,
        openems_args: Sequence[str] | None,
        *,
        timeout_sec: float | None = None,
    ) -> None:
        """Execute openEMS via CLI with timeout support.

        Args:
            outputs_dir: Working directory for the simulation.
            openems_args: Command-line arguments for openEMS.
            timeout_sec: Timeout in seconds (None for no timeout).

        Raises:
            SimulationTimeoutError: If simulation exceeds timeout.
            SimulationExecutionError: If openEMS returns non-zero exit code.
        """
        if self.openems_runner is None:
            raise ValueError("openems_runner is required for cli mode")
        args = list(openems_args or [])
        if not args:
            raise ValueError("openems_args must be provided for cli mode")

        try:
            proc = self.openems_runner.run_with_timeout(args, workdir=outputs_dir, timeout_sec=timeout_sec)
        except subprocess.TimeoutExpired as e:
            raise SimulationTimeoutError(f"openEMS simulation timed out after {timeout_sec} seconds") from e

        if proc.returncode != 0:
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            error_msg = (stderr or stdout).strip()
            logger.error(
                "openEMS failed with code %d: %s",
                proc.returncode,
                error_msg[:500],
            )
            raise SimulationExecutionError(
                f"openEMS CLI failed with returncode {proc.returncode}: {error_msg}",
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
            )


def _spec_hash(spec: SimulationSpec) -> str:
    canonical = simulation_canonical_json(spec)
    return sha256_bytes(canonical.encode("utf-8"))


def _geometry_hash(geometry: GeometrySpec) -> str:
    canonical = geometry_canonical_json(geometry)
    return sha256_bytes(canonical.encode("utf-8"))


def _simulation_hash(spec_hash: str, geometry_hash: str, solver_payload: dict[str, Any]) -> str:
    payload = {
        "spec_hash": spec_hash,
        "geometry_hash": geometry_hash,
        "solver": solver_payload,
    }
    return sha256_bytes(canonical_json_dumps(payload).encode("utf-8"))


def _build_manifest(
    *,
    spec: SimulationSpec,
    simulation_hash: str,
    spec_hash: str,
    geometry_hash: str,
    solver_payload: dict[str, Any],
    output_hashes: dict[str, str],
    execution_time_sec: float | None = None,
) -> dict[str, Any]:
    """Build a simulation manifest dictionary.

    Args:
        spec: Simulation specification.
        simulation_hash: Hash of simulation inputs.
        spec_hash: Hash of the simulation spec.
        geometry_hash: Hash of the geometry spec.
        solver_payload: Solver configuration details.
        output_hashes: Map of output file paths to hashes.
        execution_time_sec: Execution time in seconds (optional).

    Returns:
        Manifest dictionary.
    """
    outputs = [{"path": path, "hash": output_hashes[path]} for path in sorted(output_hashes.keys())]
    manifest: dict[str, Any] = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "simulation_id": spec.simulation_id,
        "simulation_hash": simulation_hash,
        "spec_hash": spec_hash,
        "geometry_hash": geometry_hash,
        "toolchain": spec.toolchain.model_dump(mode="json"),
        "solver": solver_payload,
        "outputs": outputs,
    }
    if execution_time_sec is not None:
        manifest["execution_time_sec"] = execution_time_sec
    return manifest


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    text = canonical_json_dumps(manifest)
    path.write_text(f"{text}\n", encoding="utf-8")


def _manifest_hash(manifest: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_dumps(manifest).encode("utf-8"))


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _is_cache_hit(
    manifest: dict[str, Any],
    *,
    simulation_hash: str,
    spec_hash: str,
    geometry_hash: str,
    solver_payload: dict[str, Any],
    output_dir: Path,
    outputs_dir: Path,
) -> bool:
    """Check if the cached manifest matches the current simulation inputs.

    Args:
        manifest: Cached manifest dictionary.
        simulation_hash: Hash of current simulation inputs.
        spec_hash: Hash of current simulation spec.
        geometry_hash: Hash of current geometry spec.
        solver_payload: Current solver configuration.
        output_dir: Root output directory.
        outputs_dir: Directory containing simulation outputs.

    Returns:
        True if cache is valid, False otherwise.
    """
    if manifest.get("simulation_hash") != simulation_hash:
        return False
    output_hashes = _hash_outputs(output_dir, outputs_dir)
    if not output_hashes:
        return False
    try:
        expected = _build_manifest(
            spec=_manifest_stub_spec(manifest),
            simulation_hash=simulation_hash,
            spec_hash=spec_hash,
            geometry_hash=geometry_hash,
            solver_payload=solver_payload,
            output_hashes=output_hashes,
        )
    except Exception:
        return False
    # Compare manifests excluding execution_time_sec (which varies between runs)
    manifest_copy = dict(manifest)
    expected_copy = dict(expected)
    manifest_copy.pop("execution_time_sec", None)
    expected_copy.pop("execution_time_sec", None)
    return canonical_json_dumps(expected_copy) == canonical_json_dumps(manifest_copy)


def _manifest_stub_spec(manifest: dict[str, Any]) -> SimulationSpec:
    toolchain = manifest.get("toolchain")
    if not isinstance(toolchain, dict):
        raise ValueError("Manifest toolchain must be a mapping")
    payload = {
        "schema_version": 1,
        "simulation_id": manifest.get("simulation_id"),
        "toolchain": toolchain,
        "geometry_ref": {
            "design_hash": "stub",
            "coupon_id": None,
        },
        "excitation": {"type": "gaussian", "f0_hz": 1, "fc_hz": 1},
        "frequency": {"f_start_hz": 1, "f_stop_hz": 1, "n_points": 2},
        "ports": [
            {
                "id": "P1",
                "type": "lumped",
                "impedance_ohm": 50.0,
                "excite": True,
                "position_nm": [0, 0, 0],
                "direction": "x",
            }
        ],
    }
    payload["output"] = {"outputs_dir": "sim_outputs/"}
    payload["mesh"] = {}
    payload["boundaries"] = {}
    payload["materials"] = {"dielectrics": [], "conductors": []}
    payload["control"] = {}
    return SimulationSpec.model_validate(payload)


def _hash_outputs(output_dir: Path, outputs_dir: Path) -> dict[str, str]:
    if not outputs_dir.exists():
        return {}
    hashes: dict[str, str] = {}
    for path in sorted(outputs_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(output_dir).as_posix()
            hashes[rel] = _hash_file(path)
    return hashes


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _clear_outputs_dir(outputs_dir: Path) -> None:
    if not outputs_dir.exists():
        return
    for path in sorted(outputs_dir.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def _find_sparam_file(outputs_dir: Path, spec: SimulationSpec) -> Path | None:
    """Find the S-parameter output file in the outputs directory.

    Args:
        outputs_dir: Directory containing simulation outputs.
        spec: Simulation specification.

    Returns:
        Path to S-parameter file if found, None otherwise.
    """
    if not spec.output.s_params:
        return None

    # Check for touchstone files first
    if spec.output.s_params_format in ("touchstone", "both"):
        # Look for .s2p file (2-port S-parameters)
        s2p_path = outputs_dir / "sparams.s2p"
        if s2p_path.exists():
            return s2p_path
        # Also check for any .sNp file
        for path in outputs_dir.glob("*.s?p"):
            return path

    # Check for CSV format
    if spec.output.s_params_format in ("csv", "both"):
        csv_path = outputs_dir / "sparams.csv"
        if csv_path.exists():
            return csv_path

    return None


def _write_stub_outputs(spec: SimulationSpec, outputs_dir: Path, seed: int) -> None:
    if spec.output.s_params:
        if spec.output.s_params_format in ("touchstone", "both"):
            _write_stub_touchstone(outputs_dir / "sparams.s2p", spec.frequency, seed)
        if spec.output.s_params_format in ("csv", "both"):
            _write_stub_csv(outputs_dir / "sparams.csv", spec.frequency, seed)
    if spec.output.port_signals:
        _write_stub_port_signals(outputs_dir / "port_signals.json", seed)
    if spec.output.energy_decay:
        _write_stub_energy_decay(outputs_dir / "energy_decay.json", seed)
    if spec.output.nf2ff:
        _write_stub_nf2ff(outputs_dir / "nf2ff.json", seed)


def _frequency_axis(frequency: FrequencySpec) -> list[int]:
    f_start = int(frequency.f_start_hz)
    f_stop = int(frequency.f_stop_hz)
    n_points = int(frequency.n_points)
    if n_points <= 1:
        return [f_start]
    step_num = f_stop - f_start
    step_den = n_points - 1
    freqs = [f_start + (step_num * i) // step_den for i in range(n_points)]
    freqs[-1] = f_stop
    return freqs


def _stub_sparams(seed: int, index: int) -> tuple[int, int, int, int, int, int, int, int]:
    base = (seed % 1000) + 1
    step = index + 1
    s11_re = -(100_000 + base + step)
    s11_im = -(5_000 + step)
    s21_re = 900_000 - base - step * 10
    s21_im = 2_000 + step
    s12_re = s21_re // 100
    s12_im = -(s21_im // 100)
    s22_re = s11_re * 9 // 10
    s22_im = s11_im * 9 // 10
    return s11_re, s11_im, s21_re, s21_im, s12_re, s12_im, s22_re, s22_im


def _format_micro(value_micro: int) -> str:
    return f"{value_micro / 1_000_000:.6e}"


def _write_stub_touchstone(path: Path, frequency: FrequencySpec, seed: int) -> None:
    freqs = _frequency_axis(frequency)
    lines = ["# Hz S RI R 50"]
    for index, freq in enumerate(freqs):
        values = _stub_sparams(seed, index)
        rendered = " ".join([str(freq)] + [_format_micro(value) for value in values])
        lines.append(rendered)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stub_csv(path: Path, frequency: FrequencySpec, seed: int) -> None:
    freqs = _frequency_axis(frequency)
    header = "freq_hz,s11_re,s11_im,s21_re,s21_im,s12_re,s12_im,s22_re,s22_im"
    lines = [header]
    for index, freq in enumerate(freqs):
        values = _stub_sparams(seed, index)
        rendered = ",".join([str(freq)] + [_format_micro(value) for value in values])
        lines.append(rendered)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stub_port_signals(path: Path, seed: int) -> None:
    base = seed % 10
    times = [0, 10, 20, 30, 40]
    payload = {
        "time_ps": times,
        "ports": {
            "P1": {
                "voltage_v": [float(base + idx) for idx in range(len(times))],
                "current_a": [float(base - idx) for idx in range(len(times))],
            }
        },
    }
    text = canonical_json_dumps(payload)
    path.write_text(f"{text}\n", encoding="utf-8")


def _write_stub_energy_decay(path: Path, seed: int) -> None:
    base = seed % 5
    times = [0, 50, 100, 150, 200]
    energy = [float(-10 * idx - base) for idx in range(len(times))]
    payload = {"time_ps": times, "energy_db": energy}
    text = canonical_json_dumps(payload)
    path.write_text(f"{text}\n", encoding="utf-8")


def _write_stub_nf2ff(path: Path, seed: int) -> None:
    payload = {"seed": seed, "fields": []}
    text = canonical_json_dumps(payload)
    path.write_text(f"{text}\n", encoding="utf-8")


# =============================================================================
# S-Parameter Result Loading
# =============================================================================


def load_sparam_result(result: SimulationResult) -> SParameterData | None:
    """Load S-parameter data from a simulation result.

    This function reads the S-parameter output file (Touchstone format)
    from a completed simulation result.

    Args:
        result: SimulationResult from a completed simulation.

    Returns:
        SParameterData if S-parameter file exists and is valid, None otherwise.

    Example:
        >>> runner = SimulationRunner(mode="stub")
        >>> result = runner.run(spec, geometry, output_dir=Path("./output"))
        >>> sparam_data = load_sparam_result(result)
        >>> if sparam_data is not None:
        ...     print(f"S21 at {sparam_data.f_min_hz} Hz: {sparam_data.s21()[0]}")
    """
    from formula_foundry.em.touchstone import read_touchstone

    if result.sparam_path is None:
        return None

    if not result.sparam_path.exists():
        logger.warning("S-parameter file not found: %s", result.sparam_path)
        return None

    try:
        return read_touchstone(result.sparam_path)
    except Exception as e:
        logger.warning("Failed to parse S-parameter file %s: %s", result.sparam_path, e)
        return None
