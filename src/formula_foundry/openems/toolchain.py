from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OPENEMS_TOOLCHAIN_PATH = Path(__file__).resolve().parents[3] / "config" / "openems_toolchain.json"


@dataclass(frozen=True)
class OpenEMSToolchain:
    version: str
    docker_image: str

    def to_dict(self) -> dict[str, str]:
        return {"version": self.version, "docker_image": self.docker_image}


def load_openems_toolchain(path: Path | None = None) -> OpenEMSToolchain:
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
