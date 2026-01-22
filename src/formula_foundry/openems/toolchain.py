from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_OPENEMS_TOOLCHAIN_PATH = Path(__file__).resolve().parents[3] / "config" / "openems_toolchain.json"
DEFAULT_OPENEMS_LOCKFILE_PATH = Path(__file__).resolve().parents[3] / "toolchain" / "openems.lock.json"


@dataclass(frozen=True)
class OpenEMSToolchain:
    version: str
    docker_image: str
    csxcad_version: str | None = None
    appcsxcad_version: str | None = None
    base_image_digest: str | None = None
    toolchain_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"version": self.version, "docker_image": self.docker_image}
        if self.csxcad_version:
            result["csxcad_version"] = self.csxcad_version
        if self.appcsxcad_version:
            result["appcsxcad_version"] = self.appcsxcad_version
        if self.base_image_digest:
            result["base_image_digest"] = self.base_image_digest
        if self.toolchain_hash:
            result["toolchain_hash"] = self.toolchain_hash
        return result


def load_openems_toolchain(path: Path | None = None) -> OpenEMSToolchain:
    """Load openEMS toolchain configuration from config file."""
    toolchain_path = path or DEFAULT_OPENEMS_TOOLCHAIN_PATH
    if not toolchain_path.exists():
        raise FileNotFoundError(f"openEMS toolchain not found: {toolchain_path}")

    payload = json.loads(toolchain_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("openEMS toolchain config must be a JSON object")

    version = payload.get("version")
    docker_image = payload.get("docker_image")
    if not isinstance(version, str) or not version:
        raise ValueError("openEMS toolchain version must be a non-empty string")
    if not isinstance(docker_image, str) or not docker_image:
        raise ValueError("openEMS toolchain docker_image must be a non-empty string")

    return OpenEMSToolchain(version=version, docker_image=docker_image)


def load_openems_lockfile(path: Path | None = None) -> OpenEMSToolchain:
    """Load openEMS toolchain configuration from lockfile with extended metadata."""
    lockfile_path = path or DEFAULT_OPENEMS_LOCKFILE_PATH
    if not lockfile_path.exists():
        raise FileNotFoundError(f"openEMS lockfile not found: {lockfile_path}")

    payload = json.loads(lockfile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("openEMS lockfile must be a JSON object")

    version = payload.get("openems_version")
    docker_image = payload.get("docker_image")
    if not isinstance(version, str) or not version:
        raise ValueError("openEMS lockfile openems_version must be a non-empty string")
    if not isinstance(docker_image, str) or not docker_image:
        raise ValueError("openEMS lockfile docker_image must be a non-empty string")

    return OpenEMSToolchain(
        version=version,
        docker_image=docker_image,
        csxcad_version=payload.get("csxcad_version"),
        appcsxcad_version=payload.get("appcsxcad_version"),
        base_image_digest=payload.get("base_image_digest"),
        toolchain_hash=payload.get("toolchain_hash"),
    )


def validate_openems_version(version_str: str, expected: str) -> bool:
    """Validate that a version string matches the expected version.

    Args:
        version_str: The version string to validate (e.g., "0.0.35").
        expected: The expected version (e.g., "0.0.35").

    Returns:
        True if versions match, False otherwise.
    """
    return version_str.strip() == expected.strip()


def parse_docker_image_ref(image_ref: str) -> dict[str, str | None]:
    """Parse a Docker image reference into components.

    Args:
        image_ref: Docker image reference (e.g., "registry/image:tag@sha256:digest").

    Returns:
        Dict with keys: registry, repository, tag, digest.
    """
    result: dict[str, str | None] = {
        "registry": None,
        "repository": None,
        "tag": None,
        "digest": None,
    }

    # Split off digest if present
    if "@" in image_ref:
        image_ref, digest = image_ref.rsplit("@", 1)
        result["digest"] = digest

    # Split off tag if present
    if ":" in image_ref and not image_ref.startswith("sha256:"):
        image_ref, tag = image_ref.rsplit(":", 1)
        result["tag"] = tag

    # Split registry and repository
    parts = image_ref.split("/")
    if len(parts) >= 2 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        result["registry"] = parts[0]
        result["repository"] = "/".join(parts[1:])
    else:
        result["repository"] = image_ref

    return result


def is_digest_pinned(image_ref: str) -> bool:
    """Check if a Docker image reference is pinned by digest.

    Args:
        image_ref: Docker image reference to check.

    Returns:
        True if the reference includes a sha256 digest.
    """
    return bool(re.search(r"@sha256:[0-9a-f]{64}$", image_ref))
