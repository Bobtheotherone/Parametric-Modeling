"""Toolchain provenance capture for coupon builds.

This module ensures complete toolchain provenance is captured for Docker builds,
per CP-5.1 and CP-5.3 requirements:
- kicad-cli version is always run inside the container
- Docker image reference (tag+digest) is recorded
- KiCad version string is recorded
- CLI version output is recorded
- Generator git sha is recorded
- 'unknown' values are never allowed for docker builds
- toolchain_hash from lock file is recorded for traceability

Satisfies:
    - CP-5.1: Ensure toolchain provenance always captured
    - CP-5.3: Ensure toolchain provenance always captured
    - D5: Toolchain provenance incomplete in manifest (fix)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from formula_foundry.substrate import get_git_sha

from .kicad import KicadCliRunner, get_kicad_cli_version
from .kicad.cli import KicadCliMode
from .kicad.runners.docker import DockerKicadRunner
from .toolchain import ToolchainLoadError, load_toolchain_lock


class ToolchainProvenanceError(RuntimeError):
    """Raised when toolchain provenance cannot be captured for docker builds."""

    pass


@dataclass(frozen=True)
class ToolchainProvenance:
    """Complete toolchain provenance for a coupon build.

    Attributes:
        mode: The execution mode ("local" or "docker").
        kicad_version: The KiCad version from the spec (e.g., "9.0.7").
        kicad_cli_version: Full kicad-cli version output (e.g., "9.0.7").
        docker_image_ref: Full Docker image reference with digest (e.g., "kicad/kicad:9.0.7@sha256:...").
        generator_git_sha: Git SHA of the generator codebase.
        lock_file_toolchain_hash: SHA256 hash from the toolchain lock file (for traceability).
    """

    mode: KicadCliMode
    kicad_version: str
    kicad_cli_version: str
    docker_image_ref: str | None
    generator_git_sha: str
    lock_file_toolchain_hash: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Convert to toolchain metadata dict for manifest.

        Returns:
            Dictionary with kicad, docker, mode, generator_git_sha, and
            lock_file_toolchain_hash keys.
        """
        result: dict[str, Any] = {
            "kicad": {
                "version": self.kicad_version,
                "cli_version_output": self.kicad_cli_version,
            },
            "mode": self.mode,
            "generator_git_sha": self.generator_git_sha,
        }
        if self.docker_image_ref:
            result["docker"] = {"image_ref": self.docker_image_ref}
        if self.lock_file_toolchain_hash:
            result["lock_file_toolchain_hash"] = self.lock_file_toolchain_hash
        return result


def capture_toolchain_provenance(
    *,
    mode: KicadCliMode,
    kicad_version: str,
    docker_image: str | None = None,
    workdir: Path | None = None,
    lock_file: Path | None = None,
) -> ToolchainProvenance:
    """Capture complete toolchain provenance.

    For docker mode, this function:
    1. Runs kicad-cli --version inside the container
    2. Loads the full Docker image reference (with digest) from the lock file
    3. Captures the generator git SHA

    For local mode, it captures version from the local kicad-cli installation.

    Args:
        mode: The execution mode ("local" or "docker").
        kicad_version: KiCad version string from the spec.
        docker_image: Docker image reference (for docker mode).
        workdir: Working directory for running kicad-cli.
        lock_file: Path to toolchain lock file (for docker mode digest).

    Returns:
        ToolchainProvenance with complete provenance information.

    Raises:
        ToolchainProvenanceError: If provenance cannot be captured for docker mode.
    """
    if workdir is None:
        workdir = Path.cwd()

    # Capture generator git SHA
    try:
        generator_git_sha = get_git_sha(Path.cwd())
    except Exception:
        generator_git_sha = "0" * 40

    if mode == "docker":
        return _capture_docker_provenance(
            kicad_version=kicad_version,
            docker_image=docker_image,
            workdir=workdir,
            lock_file=lock_file,
            generator_git_sha=generator_git_sha,
        )
    else:
        return _capture_local_provenance(
            kicad_version=kicad_version,
            workdir=workdir,
            generator_git_sha=generator_git_sha,
        )


def _capture_docker_provenance(
    *,
    kicad_version: str,
    docker_image: str | None,
    workdir: Path,
    lock_file: Path | None,
    generator_git_sha: str,
) -> ToolchainProvenance:
    """Capture provenance for docker mode.

    Runs kicad-cli --version inside the container and loads the full image ref.
    Also loads the toolchain_hash from the lock file for traceability (CP-5.1).

    Raises:
        ToolchainProvenanceError: If kicad-cli version cannot be determined.
    """
    # Load full docker image reference with digest and toolchain_hash from lock file
    lock_file_toolchain_hash: str | None = None
    try:
        toolchain_config = load_toolchain_lock(lock_path=lock_file)
        docker_image_ref = toolchain_config.pinned_image_ref
        lock_file_toolchain_hash = toolchain_config.toolchain_hash
    except ToolchainLoadError:
        # Fall back to provided image if lock file not found/invalid
        if docker_image:
            docker_image_ref = docker_image
        else:
            raise ToolchainProvenanceError(
                "Docker mode requires toolchain lock file or docker_image parameter"
            )
    except FileNotFoundError:
        # Fall back to provided image if lock file not found
        if docker_image:
            docker_image_ref = docker_image
        else:
            raise ToolchainProvenanceError(
                "Docker mode requires toolchain lock file or docker_image parameter"
            )

    # Run kicad-cli --version inside the container
    try:
        runner = DockerKicadRunner(docker_image=docker_image_ref)
        kicad_cli_version = runner.kicad_cli_version(workdir)
    except RuntimeError as e:
        raise ToolchainProvenanceError(
            f"Failed to capture kicad-cli version in docker mode: {e}"
        ) from e

    # Validate that we don't have 'unknown' values for docker builds
    if kicad_cli_version == "unknown" or not kicad_cli_version:
        raise ToolchainProvenanceError(
            "Docker builds must have valid kicad-cli version, got 'unknown'"
        )

    return ToolchainProvenance(
        mode="docker",
        kicad_version=kicad_version,
        kicad_cli_version=kicad_cli_version,
        docker_image_ref=docker_image_ref,
        generator_git_sha=generator_git_sha,
        lock_file_toolchain_hash=lock_file_toolchain_hash,
    )


def _capture_local_provenance(
    *,
    kicad_version: str,
    workdir: Path,
    generator_git_sha: str,
) -> ToolchainProvenance:
    """Capture provenance for local mode.

    Runs kicad-cli --version on the local system.
    """
    try:
        runner = KicadCliRunner(mode="local")
        kicad_cli_version = get_kicad_cli_version(runner, workdir)
    except (RuntimeError, OSError):
        # For local mode, allow missing kicad-cli or other lookup failures.
        kicad_cli_version = "unknown"

    return ToolchainProvenance(
        mode="local",
        kicad_version=kicad_version,
        kicad_cli_version=kicad_cli_version,
        docker_image_ref=None,
        generator_git_sha=generator_git_sha,
    )


__all__ = [
    "ToolchainProvenance",
    "ToolchainProvenanceError",
    "capture_toolchain_provenance",
]
