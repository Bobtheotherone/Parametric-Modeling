"""Docker-based KiCad CLI runner.

This module provides DockerKicadRunner, which executes kicad-cli commands
inside a pinned Docker container (e.g., kicad/kicad:9.0.7).

Satisfies CP-1.2 requirements:
- Executes kicad-cli inside pinned Docker container
- Handles DRC with --severity-all --exit-code-violations --format json
- Provides kicad_cli_version() for toolchain provenance

See Section 13.1.2 of the design document.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .protocol import IKicadRunner, KicadRunResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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


class DockerKicadRunner(IKicadRunner):
    """KiCad CLI runner that executes commands inside a Docker container.

    This runner mounts the working directory into the container and executes
    kicad-cli commands using the pinned Docker image from the toolchain lock file.

    Attributes:
        docker_image: Docker image reference (tag or tag@digest).
        kicad_bin: Name of the kicad-cli binary inside the container.

    Example:
        >>> runner = DockerKicadRunner.from_lock_file()
        >>> result = runner.run(["pcb", "drc", "--help"], cwd=Path("/workspace"))
        >>> print(result.stdout)
    """

    def __init__(
        self,
        docker_image: str,
        kicad_bin: str = "kicad-cli",
    ) -> None:
        """Initialize the Docker runner.

        Args:
            docker_image: Docker image reference (e.g., "kicad/kicad:9.0.7").
            kicad_bin: Name of the kicad-cli binary. Defaults to "kicad-cli".
        """
        self._docker_image = docker_image
        self._kicad_bin = kicad_bin

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
        """
        # Resolve to absolute path for volume mounting
        cwd_abs = cwd.resolve()

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{cwd_abs}:/workspace",
            "-w",
            "/workspace",
        ]

        # Add environment variables if provided
        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])

        cmd.extend([
            self._docker_image,
            self._kicad_bin,
            *args,
        ])

        return cmd

    def run(
        self,
        args: Sequence[str],
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> KicadRunResult:
        """Execute a kicad-cli command inside the Docker container.

        Args:
            args: Command arguments to pass to kicad-cli (e.g., ["pcb", "drc", ...]).
            cwd: Working directory to mount into the container at /workspace.
            env: Optional environment variables to set in the container.

        Returns:
            KicadRunResult containing exit code, stdout, stderr, and the full command.
        """
        cmd = self._build_docker_command(args, cwd, env)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        return KicadRunResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
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
    "DockerKicadRunner",
    "load_docker_image_ref",
    "parse_kicad_version",
]
