"""Toolchain lock file loader and hash computation.

This module provides functions to load toolchain/kicad.lock.json, compute
toolchain_hash from canonical JSON bytes, and return a ToolchainConfig dataclass.

Used by:
    - Docker runner for pinned image reference
    - Manifest generation for toolchain provenance

Satisfies:
    - CP-1.1: Add toolchain lock + explicit pinning strategy
    - REQ-M1-018: Complete toolchain provenance in manifest
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from formula_foundry.substrate import canonical_json_dumps, sha256_bytes

# Default path to the toolchain lock file relative to repository root
DEFAULT_LOCK_PATH = Path("toolchain/kicad.lock.json")


class ToolchainLoadError(Exception):
    """Raised when toolchain lock file cannot be loaded or is invalid."""

    pass


@dataclass(frozen=True)
class ToolchainConfig:
    """Immutable configuration for pinned KiCad toolchain.

    Attributes:
        schema_version: Version of the lock file schema.
        kicad_version: Pinned KiCad version (e.g., "9.0.7").
        docker_image: Docker image reference (e.g., "kicad/kicad:9.0.7").
        docker_digest: SHA256 digest of the pinned docker image.
        toolchain_hash: SHA256 hash computed from canonical JSON of lock file.
    """

    schema_version: str
    kicad_version: str
    docker_image: str
    docker_digest: str
    toolchain_hash: str

    @property
    def pinned_image_ref(self) -> str:
        """Return fully pinned docker image reference with digest if available.

        Returns:
            Image reference like "kicad/kicad:9.0.7@sha256:...".
        """
        if "@sha256:" in self.docker_image:
            return self.docker_image
        return f"{self.docker_image}@{self.docker_digest}"

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return dict suitable for inclusion in build manifest.

        Returns:
            Dictionary with toolchain metadata for manifest.json.
        """
        return {
            "kicad_version": self.kicad_version,
            "docker_image": self.docker_image,
            "docker_digest": self.docker_digest,
            "toolchain_hash": self.toolchain_hash,
        }


_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_PLACEHOLDER_HEX = {
    "0" * 64,
    "0" * 63 + "1",
}


def _is_placeholder_digest(digest: str) -> bool:
    """Return True if digest is an explicit placeholder value."""
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
    """Validate a sha256 digest string and reject placeholders."""
    if not digest:
        raise ToolchainLoadError(f"{field} must be set in lock file")
    if _is_placeholder_digest(digest):
        raise ToolchainLoadError(f"{field} must be resolved (not placeholder)")
    if not _DIGEST_PATTERN.match(str(digest)):
        raise ToolchainLoadError(f"{field} must be a sha256: 64-hex digest")


def _parse_docker_ref(docker_ref: str) -> tuple[str, str]:
    """Parse docker_ref into (image_base, digest) and validate digest."""
    if "@sha256:" not in docker_ref:
        raise ToolchainLoadError("docker_ref must include sha256 digest")
    image_base, digest = docker_ref.split("@", 1)
    _validate_digest(digest, field="docker_ref")
    return image_base, digest


def compute_toolchain_hash(lock_data: dict[str, Any]) -> str:
    """Compute SHA256 hash from canonical JSON representation of lock data.

    The hash is computed over the lock file contents (excluding the
    toolchain_hash field itself) to provide a stable fingerprint of the
    toolchain configuration.

    Args:
        lock_data: Dictionary containing lock file data. The 'toolchain_hash'
            key is excluded from the hash computation if present.

    Returns:
        SHA256 hash as lowercase hex string.
    """
    # Create copy without toolchain_hash for deterministic hashing
    hashable_data = {k: v for k, v in lock_data.items() if k != "toolchain_hash"}
    canonical = canonical_json_dumps(hashable_data)
    return sha256_bytes(canonical.encode("utf-8"))


def load_toolchain_lock(
    lock_path: Path | None = None,
    repo_root: Path | None = None,
) -> ToolchainConfig:
    """Load toolchain lock file and return validated ToolchainConfig.

    Args:
        lock_path: Explicit path to lock file. If None, uses DEFAULT_LOCK_PATH
            relative to repo_root.
        repo_root: Repository root directory. If None, uses current working
            directory.

    Returns:
        ToolchainConfig with all fields populated including computed hash.

    Raises:
        ToolchainLoadError: If lock file doesn't exist, is invalid JSON, or
            is missing required fields.
    """
    if repo_root is None:
        repo_root = Path.cwd()

    if lock_path is None:
        lock_path = repo_root / DEFAULT_LOCK_PATH

    if not lock_path.exists():
        raise ToolchainLoadError(f"Toolchain lock file not found: {lock_path}")

    try:
        lock_text = lock_path.read_text(encoding="utf-8")
        lock_data = json.loads(lock_text)
    except json.JSONDecodeError as e:
        raise ToolchainLoadError(f"Invalid JSON in toolchain lock file: {e}") from e
    except OSError as e:
        raise ToolchainLoadError(f"Failed to read toolchain lock file: {e}") from e

    return _parse_lock_data(lock_data)


def _parse_lock_data(lock_data: dict[str, Any]) -> ToolchainConfig:
    """Parse and validate lock data dictionary into ToolchainConfig.

    Args:
        lock_data: Dictionary from parsed lock file JSON.

    Returns:
        Validated ToolchainConfig.

    Raises:
        ToolchainLoadError: If required fields are missing.
    """
    required_fields = ["kicad_version", "docker_image", "docker_digest"]
    missing = [f for f in required_fields if f not in lock_data]
    if missing:
        raise ToolchainLoadError(f"Missing required fields in lock file: {missing}")

    docker_image = lock_data["docker_image"]
    docker_digest = lock_data["docker_digest"]
    docker_ref = lock_data.get("docker_ref")

    _validate_digest(docker_digest, field="docker_digest")

    if "@sha256:" in docker_image:
        image_base, embedded_digest = docker_image.split("@", 1)
        if embedded_digest != docker_digest:
            raise ToolchainLoadError("docker_image digest does not match docker_digest")
        docker_image = image_base

    if docker_ref:
        ref_base, ref_digest = _parse_docker_ref(docker_ref)
        if ref_digest != docker_digest:
            raise ToolchainLoadError("docker_ref digest does not match docker_digest")
        if ref_base != docker_image:
            raise ToolchainLoadError("docker_ref does not match docker_image")

    # Compute hash from lock data
    toolchain_hash = compute_toolchain_hash(lock_data)

    return ToolchainConfig(
        schema_version=lock_data.get("schema_version", "1.0"),
        kicad_version=lock_data["kicad_version"],
        docker_image=docker_image,
        docker_digest=docker_digest,
        toolchain_hash=toolchain_hash,
    )


def load_toolchain_lock_from_dict(lock_data: dict[str, Any]) -> ToolchainConfig:
    """Load ToolchainConfig from a dictionary (for testing or embedded configs).

    Args:
        lock_data: Dictionary with lock file structure.

    Returns:
        Validated ToolchainConfig.

    Raises:
        ToolchainLoadError: If required fields are missing.
    """
    return _parse_lock_data(lock_data)


def resolve_docker_image_ref(
    spec_docker_image: str,
    *,
    lock_path: Path | None = None,
    repo_root: Path | None = None,
    mode: str = "docker",
) -> str:
    """Resolve the Docker image reference to use for execution.

    This function implements a single-source-of-truth resolution rule:
    - If the spec's docker digest is a placeholder, substitute from lock file
    - If the spec has a real digest that differs from lock file, fail-fast
    - Local mode returns the spec's docker_image as-is (no validation)

    Args:
        spec_docker_image: Docker image reference from the spec (e.g.,
            "kicad/kicad:9.0.7@sha256:...").
        lock_path: Path to the toolchain lock file. If None, uses default.
        repo_root: Repository root for finding lock file. If None, uses cwd.
        mode: Execution mode ("local" or "docker"). Placeholder validation
            only applies to "docker" mode.

    Returns:
        Resolved Docker image reference (from lock file if spec has placeholder).

    Raises:
        ToolchainLoadError: If mode is "docker" and:
            - Lock file cannot be loaded
            - Spec has a real (non-placeholder) digest that differs from lock file
    """
    if mode == "local":
        # Local mode doesn't need docker image resolution
        return spec_docker_image

    # Load the lock file
    try:
        lock_config = load_toolchain_lock(lock_path=lock_path, repo_root=repo_root)
    except ToolchainLoadError as e:
        raise ToolchainLoadError(f"Cannot resolve docker image ref in docker mode: {e}") from e

    # Check if the spec's docker_image contains a digest
    spec_digest = None
    if "@sha256:" in spec_docker_image:
        _, spec_digest = spec_docker_image.split("@", 1)

    # If spec has no digest or has a placeholder, use lock file
    if spec_digest is None or _is_placeholder_digest(spec_digest):
        return lock_config.pinned_image_ref

    # Spec has a real digest - verify it matches the lock file
    if spec_digest != lock_config.docker_digest:
        raise ToolchainLoadError(
            f"Spec docker digest ({spec_digest[:20]}...) does not match "
            f"lock file digest ({lock_config.docker_digest[:20]}...). "
            f"Update the spec or lock file to ensure reproducibility."
        )

    # Spec digest matches lock file - use spec's reference
    return spec_docker_image


def is_placeholder_digest(digest: str) -> bool:
    """Public interface to check if a digest is a placeholder.

    Args:
        digest: Digest string to check (e.g., "sha256:0000...0001").

    Returns:
        True if the digest is a recognized placeholder value.
    """
    return _is_placeholder_digest(digest)
