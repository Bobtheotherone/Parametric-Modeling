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
    docker_digest: str | None
    toolchain_hash: str

    @property
    def pinned_image_ref(self) -> str:
        """Return fully pinned docker image reference with digest if available.

        Returns:
            Image reference like "kicad/kicad:9.0.7@sha256:..." if digest is
            available, otherwise just the tag reference.
        """
        if self.docker_digest and "PLACEHOLDER" not in self.docker_digest:
            return f"{self.docker_image}@{self.docker_digest}"
        return self.docker_image

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
    required_fields = ["kicad_version", "docker_image"]
    missing = [f for f in required_fields if f not in lock_data]
    if missing:
        raise ToolchainLoadError(f"Missing required fields in lock file: {missing}")

    # Compute hash from lock data
    toolchain_hash = compute_toolchain_hash(lock_data)

    return ToolchainConfig(
        schema_version=lock_data.get("schema_version", "1.0"),
        kicad_version=lock_data["kicad_version"],
        docker_image=lock_data["docker_image"],
        docker_digest=lock_data.get("docker_digest"),
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
