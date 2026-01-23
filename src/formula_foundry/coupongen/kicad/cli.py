"""KiCad CLI runner with local binary and Docker support.

This module provides a runner for kicad-cli that can execute:
- Locally using the system-installed kicad-cli binary
- Via Docker using a pinned KiCad Docker image (e.g., kicad/kicad:9.0.7)

Satisfies REQ-M1-015 through REQ-M1-017:
- REQ-M1-015: Runner supports local binary or pinned Docker image execution
              with variable injection via --define-var and timeout handling
- REQ-M1-016: DRC with severity-all, JSON report output, and exit-code gating
- REQ-M1-017: Gerber and drill file export via KiCad CLI
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Iterator, Literal

KicadCliMode = Literal["local", "docker"]

# Default timeout for kicad-cli operations (5 minutes)
DEFAULT_TIMEOUT_SEC: float = 300.0

# Docker-accessible temp directory for WSL environments
_WSL_DOCKER_TMP_BASE: Path | None = None

# Environment variable overrides for WSL workaround
# COUPONGEN_DOCKER_TMP_BASE: Override the temp directory location for Docker-accessible workdirs
# COUPONGEN_DISABLE_WSL_WORKDIR_COPY: Set to "1" to disable the WSL workdir copy workaround
_ENV_DOCKER_TMP_BASE = "COUPONGEN_DOCKER_TMP_BASE"
_ENV_DISABLE_WSL_COPY = "COUPONGEN_DISABLE_WSL_WORKDIR_COPY"


def _is_wsl() -> bool:
    """Detect if running in WSL environment."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except (FileNotFoundError, PermissionError):
        return False


def _is_path_docker_accessible(path: Path) -> bool:
    """Check if a path is accessible to Docker Desktop on WSL.

    In WSL2 with Docker Desktop, paths under /tmp are not accessible because
    Docker Desktop runs in a separate namespace and can only access paths
    under the Windows filesystem (/mnt/c, /mnt/d, etc.) or the user's home
    directory.

    Can be disabled via COUPONGEN_DISABLE_WSL_WORKDIR_COPY=1 environment variable.
    """
    # Allow disabling the WSL workaround via env var
    if os.environ.get(_ENV_DISABLE_WSL_COPY) == "1":
        return True

    if not _is_wsl():
        return True  # Not WSL, assume Docker can access all paths

    resolved = path.resolve()
    path_str = str(resolved)

    # Paths under /mnt/ are accessible (Windows filesystem)
    if path_str.startswith("/mnt/"):
        return True

    # Paths under user's home directory are accessible
    home = Path.home()
    try:
        resolved.relative_to(home)
        return True
    except ValueError:
        pass

    # /tmp and other system paths are not accessible to Docker Desktop
    return False


def _get_wsl_docker_tmp() -> Path:
    """Get a Docker-accessible temp directory for WSL environments.

    Can be overridden via COUPONGEN_DOCKER_TMP_BASE environment variable.
    """
    global _WSL_DOCKER_TMP_BASE
    if _WSL_DOCKER_TMP_BASE is None:
        env_override = os.environ.get(_ENV_DOCKER_TMP_BASE)
        if env_override:
            _WSL_DOCKER_TMP_BASE = Path(env_override)
        else:
            _WSL_DOCKER_TMP_BASE = Path.home() / ".coupongen_docker_tmp"
        _WSL_DOCKER_TMP_BASE.mkdir(parents=True, exist_ok=True)
    return _WSL_DOCKER_TMP_BASE


@contextmanager
def _docker_accessible_workdir(workdir: Path) -> Iterator[Path]:
    """Context manager that provides a Docker-accessible working directory.

    If the workdir is already Docker-accessible, yields it directly.
    Otherwise, copies contents to a Docker-accessible temp directory,
    yields that, and copies results back on exit.

    Args:
        workdir: Original working directory.

    Yields:
        Docker-accessible working directory path.
    """
    if _is_path_docker_accessible(workdir):
        yield workdir
        return

    # Need to copy to a Docker-accessible location
    tmp_base = _get_wsl_docker_tmp()
    tmp_workdir = Path(tempfile.mkdtemp(dir=tmp_base, prefix="kicad_"))

    try:
        # Copy all files from workdir to tmp_workdir
        for item in workdir.iterdir():
            src = workdir / item.name
            dst = tmp_workdir / item.name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        yield tmp_workdir

        # Copy new/modified files back to original workdir
        for item in tmp_workdir.iterdir():
            src = tmp_workdir / item.name
            dst = workdir / item.name
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_workdir, ignore_errors=True)


class KicadCliError(Exception):
    """Base exception for KiCad CLI errors."""

    def __init__(
        self,
        message: str,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
        command: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.command = command or []


class KicadCliTimeoutError(KicadCliError):
    """Raised when kicad-cli command exceeds the timeout."""

    def __init__(
        self,
        timeout_sec: float,
        command: list[str] | None = None,
    ) -> None:
        super().__init__(
            f"kicad-cli command timed out after {timeout_sec} seconds",
            returncode=None,
            command=command,
        )
        self.timeout_sec = timeout_sec


class KicadErrorCode(IntEnum):
    """Known kicad-cli exit codes and their meanings.

    Reference: KiCad CLI documentation and observed behavior.
    """

    SUCCESS = 0
    GENERAL_ERROR = 1
    INVALID_ARGUMENTS = 2
    FILE_LOAD_ERROR = 3
    FILE_WRITE_ERROR = 4
    DRC_VIOLATIONS = 5


@dataclass(frozen=True)
class ParsedKicadError:
    """Structured representation of a parsed KiCad CLI error.

    Attributes:
        error_code: The exit code from kicad-cli.
        error_type: Human-readable error type string.
        message: Extracted error message from stderr/stdout.
        file_path: File path mentioned in the error, if any.
        details: Additional error details extracted from output.
    """

    error_code: int
    error_type: str
    message: str
    file_path: str | None = None
    details: list[str] = field(default_factory=list)

    @property
    def is_file_error(self) -> bool:
        """Return True if this is a file-related error."""
        return self.error_code in (
            KicadErrorCode.FILE_LOAD_ERROR,
            KicadErrorCode.FILE_WRITE_ERROR,
        )

    @property
    def is_drc_error(self) -> bool:
        """Return True if this indicates DRC violations."""
        return self.error_code == KicadErrorCode.DRC_VIOLATIONS


def parse_kicad_error(
    returncode: int,
    stdout: str,
    stderr: str,
) -> ParsedKicadError:
    """Parse kicad-cli error output into a structured format.

    Args:
        returncode: Exit code from kicad-cli.
        stdout: Standard output from the command.
        stderr: Standard error from the command.

    Returns:
        ParsedKicadError with structured error information.
    """
    error_type_map = {
        KicadErrorCode.SUCCESS: "success",
        KicadErrorCode.GENERAL_ERROR: "general_error",
        KicadErrorCode.INVALID_ARGUMENTS: "invalid_arguments",
        KicadErrorCode.FILE_LOAD_ERROR: "file_load_error",
        KicadErrorCode.FILE_WRITE_ERROR: "file_write_error",
        KicadErrorCode.DRC_VIOLATIONS: "drc_violations",
    }

    error_type = error_type_map.get(returncode, f"unknown_error_{returncode}")
    combined_output = f"{stderr}\n{stdout}".strip()

    # Extract file path from common error patterns
    file_path = None
    file_patterns = [
        r"Failed to load\s+['\"]?([^'\"]+)['\"]?",
        r"Cannot open\s+['\"]?([^'\"]+)['\"]?",
        r"Error loading\s+['\"]?([^'\"]+)['\"]?",
        r"File not found:\s*['\"]?([^'\"]+)['\"]?",
        r"Could not read\s+['\"]?([^'\"]+)['\"]?",
    ]
    for pattern in file_patterns:
        match = re.search(pattern, combined_output, re.IGNORECASE)
        if match:
            file_path = match.group(1).strip()
            break

    # Extract error message (first non-empty line of stderr, or summary)
    message_lines = [line.strip() for line in stderr.split("\n") if line.strip()]
    if not message_lines:
        message_lines = [line.strip() for line in stdout.split("\n") if line.strip()]

    message = message_lines[0] if message_lines else f"kicad-cli exited with code {returncode}"

    # Collect additional details
    details = []
    if returncode == KicadErrorCode.DRC_VIOLATIONS:
        # For DRC errors, look for violation counts
        violation_match = re.search(r"(\d+)\s+violation", combined_output, re.IGNORECASE)
        if violation_match:
            details.append(f"Violations found: {violation_match.group(1)}")

    return ParsedKicadError(
        error_code=returncode,
        error_type=error_type,
        message=message,
        file_path=file_path,
        details=details,
    )


def build_define_var_args(variables: Mapping[str, str] | None) -> list[str]:
    """Build --define-var arguments for kicad-cli.

    KiCad supports text variable substitution via --define-var NAME=VALUE.
    These can be used in board text elements like ${COUPON_ID}.

    Args:
        variables: Mapping of variable names to values. If None or empty,
            returns an empty list.

    Returns:
        List of command-line arguments (e.g., ["--define-var", "COUPON_ID=test-001"]).

    Example:
        >>> build_define_var_args({"COUPON_ID": "test-001", "VERSION": "1.0"})
        ['--define-var', 'COUPON_ID=test-001', '--define-var', 'VERSION=1.0']
    """
    if not variables:
        return []

    args: list[str] = []
    for name, value in variables.items():
        # Validate variable name (alphanumeric and underscores only)
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            raise ValueError(
                f"Invalid variable name '{name}': must start with a letter or underscore "
                "and contain only alphanumeric characters and underscores"
            )
        args.extend(["--define-var", f"{name}={value}"])

    return args


@dataclass(frozen=True)
class KicadCliRunner:
    """Runner for kicad-cli supporting local and Docker execution modes.

    This class provides a unified interface for executing kicad-cli commands
    either locally (using system-installed binary) or via Docker container.

    Attributes:
        mode: Execution mode - "local" or "docker".
        docker_image: Docker image reference (required for docker mode).
        kicad_bin: Name of the kicad-cli binary. Defaults to "kicad-cli".
        default_timeout: Default timeout in seconds for commands.

    Example:
        >>> runner = KicadCliRunner(mode="local")
        >>> result = runner.run(["--version"], workdir=Path.cwd())
        >>> print(result.stdout)
    """

    mode: KicadCliMode
    docker_image: str | None = None
    kicad_bin: str = "kicad-cli"
    default_timeout: float | None = DEFAULT_TIMEOUT_SEC

    def _to_container_path(self, host_path: Path, workdir: Path) -> str:
        """Convert a host path to a container path relative to /workspace.

        When running in docker mode, paths must be translated from host
        absolute paths to container paths inside /workspace.

        Args:
            host_path: Absolute host path to convert.
            workdir: The host directory mounted as /workspace.

        Returns:
            Container-relative path string (e.g., "board.kicad_pcb" or
            "subdir/board.kicad_pcb" for paths within workdir).

        Raises:
            ValueError: If host_path is not within workdir.
        """
        host_abs = host_path.resolve()
        workdir_abs = workdir.resolve()

        try:
            relative = host_abs.relative_to(workdir_abs)
            # Return the relative path as a string (works as /workspace/relative)
            return str(relative)
        except ValueError as e:
            raise ValueError(
                f"Path {host_path} is not within workdir {workdir}. "
                f"Docker mount mapping requires all paths to be within the mounted directory."
            ) from e

    def build_command(self, args: Iterable[str], *, workdir: Path) -> list[str]:
        if self.mode == "local":
            return [self.kicad_bin, *args]
        if not self.docker_image:
            raise ValueError("docker_image is required for docker mode")
        return [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{workdir.resolve()}:/workspace",
            "-w",
            "/workspace",
            self.docker_image,
            self.kicad_bin,
            *args,
        ]

    def _collect_debug_diagnostics(self, workdir: Path) -> str:
        """Collect debug diagnostics when KiCad fails to load a board file.

        Runs diagnostic commands inside the container to help diagnose
        why kicad-cli failed to load the board file (returncode 3).

        Args:
            workdir: Working directory mounted as /workspace.

        Returns:
            Formatted diagnostic output string.
        """
        if self.mode != "docker" or not self.docker_image:
            return ""

        workdir_abs = workdir.resolve()
        diagnostics: list[str] = [
            "\n" + "=" * 60,
            "DEBUG DIAGNOSTICS (returncode 3: Failed to load board)",
            "=" * 60,
        ]

        # Define diagnostic commands to run
        diag_commands = [
            (["kicad-cli", "--version"], "kicad-cli --version"),
            (["ls", "-la", "/workspace"], "ls -la /workspace"),
            (["stat", "coupon.kicad_pcb"], "stat coupon.kicad_pcb"),
            (["head", "-n", "40", "coupon.kicad_pcb"], "head -n 40 coupon.kicad_pcb"),
            (["tail", "-n", "40", "coupon.kicad_pcb"], "tail -n 40 coupon.kicad_pcb"),
            (["wc", "-c", "coupon.kicad_pcb"], "wc -c coupon.kicad_pcb"),
            (["file", "coupon.kicad_pcb"], "file coupon.kicad_pcb"),
        ]

        for cmd_args, description in diag_commands:
            diagnostics.append(f"\n--- {description} ---")
            try:
                docker_cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{workdir_abs}:/workspace",
                    "-w",
                    "/workspace",
                    self.docker_image,
                    *cmd_args,
                ]
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                if result.stdout:
                    diagnostics.append(result.stdout.rstrip())
                if result.stderr:
                    diagnostics.append(f"[stderr] {result.stderr.rstrip()}")
                if result.returncode != 0:
                    diagnostics.append(f"[exit code: {result.returncode}]")
            except subprocess.TimeoutExpired:
                diagnostics.append("[TIMEOUT]")
            except Exception as e:
                diagnostics.append(f"[ERROR: {e}]")

        # Check for WSL mount issue
        if _is_wsl():
            diagnostics.append("\n--- WSL Mount Check ---")
            diagnostics.append(
                "Running in WSL. If /workspace appears empty, this indicates a Docker "
                "Desktop mount issue. Paths under /tmp are not accessible to Docker Desktop "
                "in WSL2. The workaround should copy files to ~/.coupongen_docker_tmp/, "
                "but if you see this error, the workaround may not have triggered."
            )
            diagnostics.append(f"Original workdir: {workdir}")
            diagnostics.append(f"Is docker accessible: {_is_path_docker_accessible(workdir)}")

        diagnostics.append("\n" + "=" * 60)
        return "\n".join(diagnostics)

    def run(
        self,
        args: Iterable[str],
        *,
        workdir: Path,
        timeout: float | None = None,
        variables: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Execute a kicad-cli command.

        Args:
            args: Command arguments to pass to kicad-cli.
            workdir: Working directory for the command. For Docker mode,
                this directory is mounted into the container at /workspace.
            timeout: Timeout in seconds. If None, uses default_timeout.
                Set to 0 or negative for no timeout.
            variables: Optional mapping of text variables to inject via
                --define-var. Used for board text substitution.

        Returns:
            CompletedProcess with stdout, stderr, and returncode.

        Raises:
            KicadCliTimeoutError: If the command exceeds the timeout.

        Example:
            >>> runner = KicadCliRunner(mode="local")
            >>> result = runner.run(
            ...     ["pcb", "export", "gerbers", "board.kicad_pcb"],
            ...     workdir=Path("/project"),
            ...     variables={"COUPON_ID": "test-001"},
            ... )
        """
        # Build define-var arguments if variables provided
        var_args = build_define_var_args(variables)

        # Combine variable args with command args
        full_args = [*var_args, *args]

        # Determine effective timeout
        effective_timeout: float | None = timeout if timeout is not None else self.default_timeout
        if effective_timeout is not None and effective_timeout <= 0:
            effective_timeout = None

        # Use Docker-accessible workdir for Docker mode in WSL environments
        if self.mode == "docker":
            with _docker_accessible_workdir(workdir) as accessible_workdir:
                cmd = self.build_command(full_args, workdir=accessible_workdir)
                try:
                    result = subprocess.run(
                        cmd,
                        cwd=accessible_workdir,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=effective_timeout,
                    )
                except subprocess.TimeoutExpired as e:
                    raise KicadCliTimeoutError(
                        timeout_sec=effective_timeout or 0,
                        command=cmd,
                    ) from e

                # On returncode 3 (Failed to load board), collect debug diagnostics
                if result.returncode == 3:
                    diagnostics = self._collect_debug_diagnostics(accessible_workdir)
                    result = subprocess.CompletedProcess(
                        args=result.args,
                        returncode=result.returncode,
                        stdout=result.stdout,
                        stderr=(result.stderr or "") + diagnostics,
                    )
        else:
            # Local mode - no path translation needed
            cmd = self.build_command(full_args, workdir=workdir)
            try:
                result = subprocess.run(
                    cmd,
                    cwd=workdir,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=effective_timeout,
                )
            except subprocess.TimeoutExpired as e:
                raise KicadCliTimeoutError(
                    timeout_sec=effective_timeout or 0,
                    command=cmd,
                ) from e

        return result

    def run_drc(
        self,
        board_path: Path,
        report_path: Path,
        *,
        timeout: float | None = None,
        variables: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run DRC check on a board file.

        Args:
            board_path: Path to the .kicad_pcb file.
            report_path: Path where the JSON DRC report will be written.
            timeout: Timeout in seconds. If None, uses default_timeout.
            variables: Optional text variables for board substitution.

        Returns:
            CompletedProcess with DRC results. Exit code 5 indicates violations.

        Raises:
            KicadCliTimeoutError: If the command exceeds the timeout.
        """
        workdir = board_path.parent.resolve()

        if self.mode == "docker":
            # Translate paths to container-relative paths
            board_container = self._to_container_path(board_path, workdir)
            report_container = self._to_container_path(report_path, workdir)
            args = build_drc_args(Path(board_container), Path(report_container))
        else:
            args = build_drc_args(board_path, report_path)

        return self.run(args, workdir=workdir, timeout=timeout, variables=variables)

    def export_gerbers(
        self,
        board_path: Path,
        out_dir: Path,
        *,
        timeout: float | None = None,
        variables: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Export Gerber files from a board.

        Args:
            board_path: Path to the .kicad_pcb file.
            out_dir: Output directory for Gerber files.
            timeout: Timeout in seconds. If None, uses default_timeout.
            variables: Optional text variables for board substitution.

        Returns:
            CompletedProcess with export results.

        Raises:
            KicadCliTimeoutError: If the command exceeds the timeout.
        """
        workdir = board_path.parent.resolve()

        if self.mode == "docker":
            # Translate paths to container-relative paths
            board_container = self._to_container_path(board_path, workdir)
            out_container = self._to_container_path(out_dir, workdir)
            args = [
                "pcb",
                "export",
                "gerbers",
                "--output",
                out_container,
                board_container,
            ]
        else:
            args = [
                "pcb",
                "export",
                "gerbers",
                "--output",
                str(out_dir),
                str(board_path),
            ]

        return self.run(args, workdir=workdir, timeout=timeout, variables=variables)

    def export_drill(
        self,
        board_path: Path,
        out_dir: Path,
        *,
        timeout: float | None = None,
        variables: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Export drill files from a board.

        Args:
            board_path: Path to the .kicad_pcb file.
            out_dir: Output directory for drill files.
            timeout: Timeout in seconds. If None, uses default_timeout.
            variables: Optional text variables for board substitution.

        Returns:
            CompletedProcess with export results.

        Raises:
            KicadCliTimeoutError: If the command exceeds the timeout.
        """
        workdir = board_path.parent.resolve()

        if self.mode == "docker":
            # Translate paths to container-relative paths
            board_container = self._to_container_path(board_path, workdir)
            out_container = self._to_container_path(out_dir, workdir)
            args = [
                "pcb",
                "export",
                "drill",
                "--output",
                out_container,
                board_container,
            ]
        else:
            args = [
                "pcb",
                "export",
                "drill",
                "--output",
                str(out_dir),
                str(board_path),
            ]

        return self.run(args, workdir=workdir, timeout=timeout, variables=variables)


def build_drc_args(
    board_path: Path,
    report_path: Path,
    *,
    severity: str = "all",
) -> list[str]:
    """Build kicad-cli DRC command arguments.

    Satisfies REQ-M1-016:
    - --severity-all: Report all violations including warnings (default for M1)
    - --format json: Output in JSON format for programmatic parsing
    - --exit-code-violations: Return non-zero exit code if violations exist

    Note: M1 uses --severity-all to catch all DRC issues including warnings.
    Use severity="error" if you need to ignore warnings.

    Args:
        board_path: Path to the .kicad_pcb file to check.
        report_path: Path where the JSON DRC report will be written.
        severity: Severity level to check ("error", "warning", "all").
            Default is "all" to check all violations.

    Returns:
        List of command-line arguments for kicad-cli pcb drc.
    """
    severity_arg = f"--severity-{severity}"
    return [
        "pcb",
        "drc",
        severity_arg,
        "--exit-code-violations",
        "--format",
        "json",
        "--output",
        str(report_path),
        str(board_path),
    ]


def get_kicad_cli_version(runner: KicadCliRunner, workdir: Path | None = None) -> str:
    """Get the kicad-cli version string.

    This helper function executes `kicad-cli --version` and parses the output
    to extract the version string. It works with both local and Docker runners.

    Args:
        runner: A KicadCliRunner instance (local or Docker mode).
        workdir: Working directory for the command. If None, uses current directory.
            Required for Docker mode to mount the volume.

    Returns:
        Version string (e.g., "9.0.7").

    Raises:
        RuntimeError: If version cannot be determined.

    Example:
        >>> runner = KicadCliRunner(mode="local")
        >>> version = get_kicad_cli_version(runner)
        >>> print(version)
        '9.0.7'
    """
    if workdir is None:
        workdir = Path.cwd()

    result = runner.run(["--version"], workdir=workdir)

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to get kicad-cli version: {result.stderr or result.stdout}"
        )

    return _parse_version_output(result.stdout)


def _parse_version_output(version_output: str) -> str:
    """Parse the version string from kicad-cli --version output.

    Args:
        version_output: Raw output from kicad-cli --version.

    Returns:
        Cleaned version string (e.g., "9.0.7").
    """
    output = version_output.strip()

    # Try to extract version pattern (e.g., "9.0.7" or "9.0.7-1")
    match = re.search(r"(\d+\.\d+\.\d+(?:-\d+)?)", output)
    if match:
        return match.group(1)

    # Fall back to returning the full output if no pattern matches
    return output


__all__ = [
    # Constants
    "DEFAULT_TIMEOUT_SEC",
    # Types and Enums
    "KicadCliMode",
    "KicadErrorCode",
    # Exceptions
    "KicadCliError",
    "KicadCliTimeoutError",
    # Data classes
    "ParsedKicadError",
    "KicadCliRunner",
    # Functions
    "build_define_var_args",
    "build_drc_args",
    "get_kicad_cli_version",
    "parse_kicad_error",
]
