"""Docker-based KiCad CLI runner.

This module provides DockerKicadRunner, which executes kicad-cli commands
inside a pinned Docker container (e.g., kicad/kicad:9.0.7).

Satisfies CP-1.2 and REQ-M1-015 requirements:
- Executes kicad-cli inside pinned Docker container
- Handles DRC with --severity-all --exit-code-violations --format json
- Provides kicad_cli_version() for toolchain provenance
- Supports timeout handling and --define-var variable injection

See Section 13.1.2 of the design document.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .protocol import IKicadRunner, KicadRunResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _get_host_uid_gid() -> tuple[int, int]:
    """Get the current host user's UID and GID.

    Returns:
        Tuple of (uid, gid).
    """
    return os.getuid(), os.getgid()

# Default timeout for Docker kicad-cli operations (5 minutes)
DEFAULT_DOCKER_TIMEOUT_SEC: float = 300.0


class DockerKicadTimeoutError(RuntimeError):
    """Raised when a Docker kicad-cli command exceeds the timeout."""

    def __init__(
        self,
        timeout_sec: float,
        command: list[str] | None = None,
    ) -> None:
        self.timeout_sec = timeout_sec
        self.command = command or []
        super().__init__(
            f"Docker kicad-cli command timed out after {timeout_sec} seconds"
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

# Default path to the toolchain lock file
DEFAULT_LOCK_FILE = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "toolchain" / "kicad.lock.json"

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_PLACEHOLDER_HEX = {
    "0" * 64,
    "0" * 63 + "1",
}


def _is_placeholder_digest(digest: str) -> bool:
    digest_text = str(digest)
    upper = digest_text.upper()
    if "PLACEHOLDER" in upper or "UNKNOWN" in upper:
        return True
    if digest_text.startswith("sha256:"):
        hex_part = digest_text.split("sha256:", 1)[1]
        if hex_part in _PLACEHOLDER_HEX:
            return True
    return False


def _validate_digest(digest: str, *, field: str) -> None:
    if not digest:
        raise ValueError(f"{field} missing from toolchain lock file")
    if _is_placeholder_digest(digest):
        raise ValueError(f"{field} is a placeholder; run tools/pin_kicad_image.py")
    if not _DIGEST_PATTERN.match(str(digest)):
        raise ValueError(f"{field} must be a sha256: 64-hex digest")


def _parse_docker_ref(docker_ref: str) -> tuple[str, str]:
    if "@sha256:" not in docker_ref:
        raise ValueError("docker_ref must include sha256 digest")
    image_base, digest = docker_ref.split("@", 1)
    _validate_digest(digest, field="docker_ref")
    return image_base, digest


def load_docker_image_ref(lock_file: Path | None = None) -> str:
    """Load the pinned Docker image reference from the toolchain lock file.

    Args:
        lock_file: Path to kicad.lock.json. Defaults to toolchain/kicad.lock.json.

    Returns:
        Docker image reference string (e.g., "kicad/kicad:9.0.7@sha256:...").

    Raises:
        FileNotFoundError: If the lock file doesn't exist.
        ValueError: If the lock file is malformed.
    """
    if lock_file is None:
        lock_file = DEFAULT_LOCK_FILE

    if not lock_file.exists():
        raise FileNotFoundError(f"Toolchain lock file not found: {lock_file}")

    with lock_file.open() as f:
        lock_data = json.load(f)

    docker_image = lock_data.get("docker_image")
    docker_digest = lock_data.get("docker_digest")
    docker_ref = lock_data.get("docker_ref")

    if not docker_image:
        raise ValueError(f"docker_image not found in {lock_file}")

    image_base = docker_image
    embedded_digest = None
    if "@sha256:" in docker_image:
        image_base, embedded_digest = docker_image.split("@", 1)

    if embedded_digest and docker_digest and embedded_digest != docker_digest:
        raise ValueError("docker_image digest does not match docker_digest")

    _validate_digest(docker_digest, field="docker_digest")

    if docker_ref:
        ref_base, ref_digest = _parse_docker_ref(docker_ref)
        if ref_digest != docker_digest:
            raise ValueError("docker_ref digest does not match docker_digest")
        if ref_base != image_base:
            raise ValueError("docker_ref does not match docker_image")

    return f"{image_base}@{docker_digest}"


class DockerMountError(RuntimeError):
    """Raised when Docker bind mount fails to expose files inside the container."""

    def __init__(self, host_path: Path, expected_file: str | None = None) -> None:
        self.host_path = host_path
        self.expected_file = expected_file
        msg = (
            f"Docker bind mount visibility issue:\n"
            f"  Host path: {host_path}\n"
            f"  Container mount: /workspace\n"
        )
        if expected_file:
            msg += f"  Expected file: {expected_file} (not visible in container)\n"
        msg += (
            "This is usually a Docker daemon path visibility issue.\n"
            "Common causes:\n"
            "  - WSL2: /tmp paths may not be visible to Docker Desktop\n"
            "  - Use a path under your home directory or repo instead"
        )
        super().__init__(msg)


class DockerKicadRunner(IKicadRunner):
    """KiCad CLI runner that executes commands inside a Docker container.

    This runner mounts the working directory into the container and executes
    kicad-cli commands using the pinned Docker image from the toolchain lock file.

    Attributes:
        docker_image: Docker image reference (tag or tag@digest).
        kicad_bin: Name of the kicad-cli binary inside the container.
        default_timeout: Default timeout in seconds for commands.

    Example:
        >>> runner = DockerKicadRunner.from_lock_file()
        >>> result = runner.run(["pcb", "drc", "--help"], cwd=Path("/workspace"))
        >>> print(result.stdout)
    """

    def __init__(
        self,
        docker_image: str,
        kicad_bin: str = "kicad-cli",
        default_timeout: float | None = DEFAULT_DOCKER_TIMEOUT_SEC,
    ) -> None:
        """Initialize the Docker runner.

        Args:
            docker_image: Docker image reference (e.g., "kicad/kicad:9.0.7").
            kicad_bin: Name of the kicad-cli binary. Defaults to "kicad-cli".
            default_timeout: Default timeout in seconds for commands.
                Set to None or 0 for no timeout.
        """
        self._docker_image = docker_image
        self._kicad_bin = kicad_bin
        self._default_timeout = default_timeout

    @classmethod
    def from_lock_file(cls, lock_file: Path | None = None) -> DockerKicadRunner:
        """Create a runner from the toolchain lock file.

        Args:
            lock_file: Path to kicad.lock.json. Defaults to toolchain/kicad.lock.json.

        Returns:
            DockerKicadRunner configured with the pinned image.
        """
        docker_image = load_docker_image_ref(lock_file)
        return cls(docker_image=docker_image)

    @property
    def docker_image(self) -> str:
        """Return the Docker image reference."""
        return self._docker_image

    @property
    def kicad_bin(self) -> str:
        """Return the kicad-cli binary name."""
        return self._kicad_bin

    @property
    def default_timeout(self) -> float | None:
        """Return the default timeout in seconds."""
        return self._default_timeout

    def _build_docker_command(
        self,
        args: Sequence[str],
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> list[str]:
        """Build the full Docker command.

        Args:
            args: Arguments to pass to kicad-cli.
            cwd: Working directory to mount as /workspace.
            env: Environment variables to pass to the container.

        Returns:
            Full Docker command as a list of strings.

        Note:
            This command includes --user to run as the host user's UID:GID,
            which is essential for bind-mounted directories to be writable.
            Without this, the container runs as uid 1000 (kicad) which cannot
            write to directories owned by different UIDs (e.g., uid 1001 on
            GitHub Actions runners).
        """
        # Resolve to absolute path for volume mounting
        cwd_abs = cwd.resolve()

        # Get host UID/GID for container user mapping
        # This ensures the container can write to bind-mounted directories
        uid, gid = _get_host_uid_gid()

        cmd = [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{uid}:{gid}",
            "-v",
            f"{cwd_abs}:/workspace",
            "-w",
            "/workspace",
        ]

        # Set HOME to /tmp since the numeric UID may not have a passwd entry
        # in the container. KiCad needs a writable home for config files.
        cmd.extend(["-e", "HOME=/tmp"])

        # Add additional environment variables if provided
        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])

        cmd.extend([
            self._docker_image,
            self._kicad_bin,
            *args,
        ])

        return cmd

    def _verify_mount(
        self,
        cwd: Path,
        expected_file: str | None = None,
    ) -> None:
        """Verify Docker bind mount is working before running kicad-cli.

        Args:
            cwd: Working directory being mounted as /workspace.
            expected_file: Optional filename expected to exist in /workspace.

        Raises:
            DockerMountError: If mount verification fails.
        """
        cwd_abs = cwd.resolve()
        uid, gid = _get_host_uid_gid()

        # Run a quick ls to check if mount is visible
        check_cmd = [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{uid}:{gid}",
            "-e",
            "HOME=/tmp",
            "-v",
            f"{cwd_abs}:/workspace",
            "-w",
            "/workspace",
            self._docker_image,
            "ls",
            "-la",
            "/workspace",
        ]

        try:
            result = subprocess.run(
                check_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )

            # Check if directory appears empty (only . and .. entries)
            lines = result.stdout.strip().split("\n")
            # Filter out "total" line and count actual entries
            entries = [l for l in lines if not l.startswith("total")]
            if len(entries) <= 2:  # Only . and ..
                raise DockerMountError(cwd_abs, expected_file)

            # If specific file expected, check it exists
            if expected_file:
                file_check_cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "--user",
                    f"{uid}:{gid}",
                    "-e",
                    "HOME=/tmp",
                    "-v",
                    f"{cwd_abs}:/workspace",
                    "-w",
                    "/workspace",
                    self._docker_image,
                    "test",
                    "-f",
                    f"/workspace/{expected_file}",
                ]
                file_result = subprocess.run(
                    file_check_cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                if file_result.returncode != 0:
                    raise DockerMountError(cwd_abs, expected_file)

        except subprocess.TimeoutExpired as e:
            raise DockerMountError(cwd_abs, expected_file) from e

    def _summarize_drc_violations(self, drc_json_path: Path, cwd: Path) -> str:
        """Parse DRC JSON and return a summary of violations.

        Args:
            drc_json_path: Path to the drc.json file.
            cwd: Working directory (for reading from container or host).

        Returns:
            Formatted summary string.
        """
        try:
            # Try to read from host path first
            host_path = cwd / drc_json_path.name
            if host_path.exists():
                with host_path.open() as f:
                    drc_data = json.load(f)
            else:
                return "[DRC summary unavailable: drc.json not found]"

            summary_lines = [
                "\n" + "=" * 60,
                "DRC VIOLATION SUMMARY (rc=5)",
                "=" * 60,
            ]

            # Count by type and severity
            violations = drc_data.get("violations", [])
            unconnected = drc_data.get("unconnected_items", [])

            type_counts: dict[str, int] = {}
            severity_counts: dict[str, int] = {}

            for v in violations + unconnected:
                vtype = v.get("type", "unknown")
                severity = v.get("severity", "unknown")
                type_counts[vtype] = type_counts.get(vtype, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            summary_lines.append(f"\nTotal violations: {len(violations)}")
            summary_lines.append(f"Unconnected items: {len(unconnected)}")

            if severity_counts:
                summary_lines.append("\nBy severity:")
                for sev, count in sorted(severity_counts.items()):
                    summary_lines.append(f"  {sev}: {count}")

            if type_counts:
                summary_lines.append("\nBy type:")
                for vtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                    summary_lines.append(f"  {vtype}: {count}")

            # Show first few violations
            all_items = violations + unconnected
            if all_items:
                summary_lines.append(f"\nFirst {min(10, len(all_items))} items:")
                for item in all_items[:10]:
                    desc = item.get("description", "")
                    severity = item.get("severity", "")
                    vtype = item.get("type", "")
                    summary_lines.append(f"  [{severity}] {vtype}: {desc}")

            summary_lines.append("\n" + "=" * 60)
            return "\n".join(summary_lines)

        except Exception as e:
            return f"[DRC summary error: {e}]"

    def _collect_debug_diagnostics(self, cwd: Path) -> str:
        """Collect debug diagnostics when KiCad fails to load a board file.

        Runs diagnostic commands inside the container to help diagnose
        why kicad-cli failed to load the board file (returncode 3).

        Args:
            cwd: Working directory mounted as /workspace.

        Returns:
            Formatted diagnostic output string.
        """
        cwd_abs = cwd.resolve()
        uid, gid = _get_host_uid_gid()
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
                    "--user",
                    f"{uid}:{gid}",
                    "-e",
                    "HOME=/tmp",
                    "-v",
                    f"{cwd_abs}:/workspace",
                    "-w",
                    "/workspace",
                    self._docker_image,
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

        diagnostics.append("\n" + "=" * 60)
        return "\n".join(diagnostics)

    def run(
        self,
        args: Sequence[str],
        cwd: Path,
        env: Mapping[str, str] | None = None,
        *,
        timeout: float | None = None,
        variables: Mapping[str, str] | None = None,
        verify_mount: bool = False,
        expected_file: str | None = None,
        drc_report_path: Path | None = None,
    ) -> KicadRunResult:
        """Execute a kicad-cli command inside the Docker container.

        Args:
            args: Command arguments to pass to kicad-cli (e.g., ["pcb", "drc", ...]).
            cwd: Working directory to mount into the container at /workspace.
            env: Optional environment variables to set in the container.
            timeout: Timeout in seconds. If None, uses default_timeout.
                Set to 0 or negative for no timeout.
            variables: Optional mapping of text variables to inject via
                --define-var. Used for board text substitution (e.g., ${COUPON_ID}).
            verify_mount: If True, verify bind mount visibility before running.
            expected_file: File expected to exist in /workspace (for mount verification).
            drc_report_path: Path to DRC JSON report (for violation summary on rc=5).

        Returns:
            KicadRunResult containing exit code, stdout, stderr, and the full command.

        Raises:
            DockerMountError: If verify_mount=True and mount verification fails.
            DockerKicadTimeoutError: If the command exceeds the timeout.

        Example:
            >>> runner = DockerKicadRunner.from_lock_file()
            >>> result = runner.run(
            ...     ["pcb", "export", "gerbers", "board.kicad_pcb"],
            ...     cwd=Path("/project"),
            ...     timeout=60,
            ...     variables={"COUPON_ID": "test-001"},
            ... )
        """
        # Optionally verify mount before running
        if verify_mount:
            self._verify_mount(cwd, expected_file)

        # Build define-var arguments if variables provided
        var_args = build_define_var_args(variables)

        # Combine variable args with command args
        full_args = [*var_args, *args]
        cmd = self._build_docker_command(full_args, cwd, env)

        # Determine effective timeout
        effective_timeout: float | None = (
            timeout if timeout is not None else self._default_timeout
        )
        if effective_timeout is not None and effective_timeout <= 0:
            effective_timeout = None

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise DockerKicadTimeoutError(
                timeout_sec=effective_timeout or 0,
                command=cmd,
            ) from e

        stderr = result.stderr or ""

        # On returncode 3 (Failed to load board), collect debug diagnostics
        if result.returncode == 3:
            diagnostics = self._collect_debug_diagnostics(cwd)
            stderr = stderr + diagnostics

        # On returncode 5 (DRC violations), add violation summary if report available
        if result.returncode == 5 and drc_report_path:
            summary = self._summarize_drc_violations(drc_report_path, cwd)
            stderr = stderr + summary

        return KicadRunResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=stderr,
            command=cmd,
        )

    def kicad_cli_version(self, cwd: Path) -> str:
        """Get the kicad-cli version string from the container.

        Args:
            cwd: Working directory (used for Docker volume mounting).

        Returns:
            Version string from kicad-cli --version output.

        Raises:
            RuntimeError: If version cannot be determined.
        """
        result = self.run(["--version"], cwd)

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to get kicad-cli version: {result.stderr or result.stdout}"
            )

        return parse_kicad_version(result.stdout)


def parse_kicad_version(version_output: str) -> str:
    """Parse the version string from kicad-cli --version output.

    Args:
        version_output: Raw output from kicad-cli --version.

    Returns:
        Cleaned version string (e.g., "9.0.7").

    Example:
        >>> parse_kicad_version("kicad-cli 9.0.7\\n")
        '9.0.7'
    """
    # Strip whitespace and extract version number
    output = version_output.strip()

    # Try to extract version pattern (e.g., "9.0.7" or "9.0.7-1")
    match = re.search(r"(\d+\.\d+\.\d+(?:-\d+)?)", output)
    if match:
        return match.group(1)

    # Fall back to returning the full output if no pattern matches
    return output


__all__ = [
    # Constants
    "DEFAULT_DOCKER_TIMEOUT_SEC",
    # Exceptions
    "DockerKicadTimeoutError",
    "DockerMountError",
    # Classes
    "DockerKicadRunner",
    # Functions
    "build_define_var_args",
    "load_docker_image_ref",
    "parse_kicad_version",
]
