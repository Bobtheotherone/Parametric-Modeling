from __future__ import annotations

import hashlib
import json
import os
import string
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import canonical_json_dumps, sha256_bytes

_HASH_CHUNK_SIZE = 1024 * 1024


class ArtifactStoreError(RuntimeError):
    """Raised when artifact store operations fail."""


class ArtifactManifestError(ValueError):
    """Raised when an artifact manifest fails validation."""


@dataclass(frozen=True)
class ArtifactEntry:
    path: str
    digest: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "hash": self.digest, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: int
    artifacts: tuple[ArtifactEntry, ...]

    @classmethod
    def from_entries(cls, entries: Sequence[ArtifactEntry], *, schema_version: int = 1) -> ArtifactManifest:
        _ensure_unique_paths(entries)
        sorted_entries = tuple(sorted(entries, key=lambda entry: entry.path))
        return cls(schema_version=schema_version, artifacts=sorted_entries)

    @classmethod
    def load(cls, path: Path) -> ArtifactManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_artifact_manifest(payload)
        entries = [
            ArtifactEntry(
                path=entry["path"],
                digest=entry["hash"],
                size_bytes=entry["size_bytes"],
            )
            for entry in payload["artifacts"]
        ]
        return cls.from_entries(entries, schema_version=payload["schema_version"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifacts": [entry.to_dict() for entry in self.artifacts],
        }

    def to_json(self) -> str:
        payload = self.to_dict()
        validate_artifact_manifest(payload)
        return canonical_json_dumps(payload)

    def write(self, path: Path) -> None:
        text = f"{self.to_json()}\n"
        atomic_write_text(path, text)


class ArtifactTransaction:
    def __init__(self, store: ArtifactStore, manifest_path: Path) -> None:
        self._store = store
        self._manifest_path = manifest_path
        self._entries: list[ArtifactEntry] = []

    def add_bytes(self, logical_path: str, data: bytes) -> ArtifactEntry:
        entry = self._store.add_bytes(logical_path, data)
        self._entries.append(entry)
        return entry

    def add_file(self, logical_path: str, src_path: Path) -> ArtifactEntry:
        entry = self._store.add_file(src_path, logical_path=logical_path)
        self._entries.append(entry)
        return entry

    def commit(self, *, schema_version: int = 1) -> ArtifactManifest:
        manifest = ArtifactManifest.from_entries(self._entries, schema_version=schema_version)
        self._store.write_manifest(self._manifest_path, manifest)
        return manifest


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.objects_dir = root / "objects"
        self.tmp_dir = root / "tmp"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def object_path(self, digest: str) -> Path:
        if not _is_hex_digest(digest, length=64):
            raise ArtifactStoreError(f"Invalid digest: {digest!r}")
        return self.objects_dir / digest[:2] / digest[2:]

    def add_bytes(self, logical_path: str, data: bytes) -> ArtifactEntry:
        digest = sha256_bytes(data)
        size_bytes = len(data)
        final_path = self.object_path(digest)
        if not final_path.exists():
            atomic_write_bytes(final_path, data)
        return ArtifactEntry(path=logical_path, digest=digest, size_bytes=size_bytes)

    def add_file(self, src_path: Path, *, logical_path: str | None = None) -> ArtifactEntry:
        temp_path = self._new_temp_path()
        try:
            digest, size_bytes = _copy_with_hash(src_path, temp_path)
            final_path = self.object_path(digest)
            self._commit_temp(temp_path, final_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        entry_path = logical_path or src_path.name
        return ArtifactEntry(path=entry_path, digest=digest, size_bytes=size_bytes)

    def verify_digest(self, digest: str) -> bool:
        try:
            path = self.object_path(digest)
        except ArtifactStoreError:
            return False
        if not path.exists():
            return False
        return _hash_file(path) == digest

    def verify_entry(self, entry: ArtifactEntry) -> bool:
        try:
            path = self.object_path(entry.digest)
        except ArtifactStoreError:
            return False
        if not path.exists():
            return False
        if path.stat().st_size != entry.size_bytes:
            return False
        return _hash_file(path) == entry.digest

    def verify_manifest(self, manifest: ArtifactManifest) -> list[str]:
        failures: list[str] = []
        for entry in manifest.artifacts:
            if not self.verify_entry(entry):
                failures.append(entry.path)
        return failures

    def start_transaction(self, manifest_path: Path) -> ArtifactTransaction:
        return ArtifactTransaction(self, manifest_path)

    def write_manifest(self, manifest_path: Path, manifest: ArtifactManifest) -> None:
        manifest.write(manifest_path)

    def _new_temp_path(self) -> Path:
        fd, path = tempfile.mkstemp(prefix="artifact-", dir=str(self.tmp_dir))
        os.close(fd)
        return Path(path)

    def _commit_temp(self, temp_path: Path, final_path: Path) -> None:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_path, final_path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def validate_artifact_manifest(payload: Mapping[str, Any]) -> None:
    if "schema_version" not in payload:
        raise ArtifactManifestError("manifest missing schema_version")
    schema_version = payload["schema_version"]
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 1:
        raise ArtifactManifestError("schema_version must be an integer >= 1")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ArtifactManifestError("artifacts must be a list")
    paths: list[str] = []
    for entry in artifacts:
        if not isinstance(entry, Mapping):
            raise ArtifactManifestError("artifact entries must be mappings")
        path = entry.get("path")
        digest = entry.get("hash")
        size_bytes = entry.get("size_bytes")
        if not isinstance(path, str) or not path:
            raise ArtifactManifestError("artifact path must be a non-empty string")
        if not isinstance(digest, str) or not _is_hex_digest(digest, length=64):
            raise ArtifactManifestError("artifact hash must be a 64-character hex string")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise ArtifactManifestError("artifact size_bytes must be a non-negative integer")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ArtifactManifestError("artifact paths must be unique")


def _ensure_unique_paths(entries: Sequence[ArtifactEntry]) -> None:
    paths = [entry.path for entry in entries]
    if len(set(paths)) != len(paths):
        raise ArtifactManifestError("artifact paths must be unique")


def _copy_with_hash(src_path: Path, dest_path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size_bytes = 0
    with src_path.open("rb") as src, dest_path.open("wb") as dest:
        while True:
            chunk = src.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
            dest.write(chunk)
            size_bytes += len(chunk)
        dest.flush()
        os.fsync(dest.fileno())
    return hasher.hexdigest(), size_bytes


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_hex_digest(value: object, *, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    return all(ch in string.hexdigits for ch in value)
