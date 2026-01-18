from __future__ import annotations

import hashlib
import json
import platform
import string
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .determinism import DeterminismConfig

MANIFEST_REQUIRED_FIELDS = (
    "git_sha",
    "design_doc_sha256",
    "environment_fingerprint",
    "determinism",
    "command_line",
    "artifacts",
)


class ManifestValidationError(ValueError):
    """Raised when a manifest fails validation."""


@dataclass(frozen=True)
class Manifest:
    git_sha: str
    design_doc_sha256: str
    environment_fingerprint: str
    determinism: dict[str, Any]
    command_line: list[str]
    artifacts: dict[str, str]

    @classmethod
    def from_environment(
        cls,
        determinism: DeterminismConfig | Mapping[str, Any],
        *,
        command_line: Sequence[str] | None = None,
        artifacts: Mapping[str, str] | None = None,
        project_root: Path | None = None,
        design_doc_path: Path | None = None,
        environment_payload: Mapping[str, Any] | None = None,
        git_sha: str | None = None,
    ) -> Manifest:
        root = project_root or Path.cwd()
        doc_path = design_doc_path or root / "DESIGN_DOCUMENT.md"
        resolved_git_sha = git_sha or get_git_sha(root)
        resolved_doc_sha = sha256_file(doc_path)
        resolved_payload = environment_payload or build_environment_payload(root)
        resolved_fingerprint = build_environment_fingerprint(resolved_payload)
        return cls(
            git_sha=resolved_git_sha,
            design_doc_sha256=resolved_doc_sha,
            environment_fingerprint=resolved_fingerprint,
            determinism=normalize_determinism_entry(determinism),
            command_line=list(command_line) if command_line is not None else list(sys.argv),
            artifacts=dict(artifacts) if artifacts is not None else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_sha": self.git_sha,
            "design_doc_sha256": self.design_doc_sha256,
            "environment_fingerprint": self.environment_fingerprint,
            "determinism": self.determinism,
            "command_line": self.command_line,
            "artifacts": self.artifacts,
        }

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    def validate(self) -> None:
        validate_manifest(self.to_dict())

    def write(self, path: Path) -> None:
        write_manifest(path, self)


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    manifest_path: Path
    logs_path: Path
    artifacts_dir: Path


def normalize_determinism_entry(entry: DeterminismConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(entry, DeterminismConfig):
        return entry.manifest_entry()
    return dict(entry)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def get_git_sha(project_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(project_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to resolve git SHA: {proc.stderr.strip()}")
    return proc.stdout.strip()


def build_environment_payload(project_root: Path) -> dict[str, Any]:
    uv_lock = project_root / "uv.lock"
    pyproject = project_root / "pyproject.toml"
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "uv_lock_sha256": sha256_file(uv_lock) if uv_lock.exists() else None,
        "pyproject_sha256": sha256_file(pyproject) if pyproject.exists() else None,
    }


def build_environment_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = canonical_json_dumps(dict(payload))
    return sha256_bytes(canonical.encode("utf-8"))


def validate_manifest(data: Mapping[str, Any]) -> None:
    missing = [field for field in MANIFEST_REQUIRED_FIELDS if field not in data]
    if missing:
        raise ManifestValidationError(f"Manifest missing required fields: {missing}")
    git_sha = data["git_sha"]
    if not isinstance(git_sha, str) or not git_sha:
        raise ManifestValidationError("git_sha must be a non-empty string")
    if not _is_hex_digest(git_sha, length=40):
        raise ManifestValidationError("git_sha must be a 40-character hex string")
    design_doc_sha = data["design_doc_sha256"]
    if not _is_hex_digest(design_doc_sha, length=64):
        raise ManifestValidationError("design_doc_sha256 must be a 64-character hex string")
    env_fp = data["environment_fingerprint"]
    if not _is_hex_digest(env_fp, length=64):
        raise ManifestValidationError("environment_fingerprint must be a 64-character hex string")
    determinism = data["determinism"]
    if not isinstance(determinism, Mapping):
        raise ManifestValidationError("determinism must be a mapping")
    if "mode" not in determinism or "seeds" not in determinism:
        raise ManifestValidationError("determinism must include mode and seeds entries")
    command_line = data["command_line"]
    if not isinstance(command_line, list) or not all(isinstance(item, str) for item in command_line):
        raise ManifestValidationError("command_line must be a list of strings")
    artifacts = data["artifacts"]
    if not isinstance(artifacts, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in artifacts.items()
    ):
        raise ManifestValidationError("artifacts must be a mapping of string paths to string hashes")


def canonical_json_dumps(data: Any) -> str:
    """Serialize JSON with sorted keys and stable float formatting (repr-based, 17-digit)."""
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def write_manifest(path: Path, manifest: Manifest | Mapping[str, Any]) -> None:
    payload = manifest.to_dict() if isinstance(manifest, Manifest) else dict(manifest)
    validate_manifest(payload)
    text = canonical_json_dumps(payload)
    path.write_text(f"{text}\n", encoding="utf-8")


def init_run_dir(run_root: Path, run_id: str) -> RunArtifacts:
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_path = run_dir / "logs.jsonl"
    logs_path.touch(exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        logs_path=logs_path,
        artifacts_dir=artifacts_dir,
    )


def create_run(run_root: Path, run_id: str, manifest: Manifest) -> RunArtifacts:
    artifacts = init_run_dir(run_root, run_id)
    write_manifest(artifacts.manifest_path, manifest)
    return artifacts


def _is_hex_digest(value: object, *, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    return all(ch in string.hexdigits for ch in value)
