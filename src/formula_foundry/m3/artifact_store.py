"""Content-addressed artifact storage with atomic writes.

This module implements the ArtifactStore class, which provides:
- Content-addressed storage under data/objects/ using SHA256 hashing
- Atomic write pattern (write to .tmp then rename)
- Manifest generation conforming to artifact.v1 schema
- Spec ID computation for canonical artifact identification
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Literal

# Type aliases for clarity
HashAlgorithm = Literal["sha256", "sha384", "sha512", "blake3"]
ArtifactType = Literal[
    "coupon_spec",
    "resolved_design",
    "kicad_board",
    "kicad_project",
    "gerber",
    "drill_file",
    "fab_package",
    "em_simulation_config",
    "em_simulation_result",
    "touchstone",
    "sparam_matrix",
    "dataset_index",
    "dataset_snapshot",
    "model_checkpoint",
    "formula_candidate",
    "validation_report",
    "drc_report",
    "manifest",
    "log",
    "other",
]
ArtifactRole = Literal[
    "geometry",
    "config",
    "oracle_output",
    "intermediate",
    "final_output",
    "validation",
    "metadata",
    "cache",
    "checkpoint",
    "dataset_member",
    "root_input",
]


@dataclass
class ContentHash:
    """Represents a cryptographic content hash."""

    algorithm: HashAlgorithm
    digest: str

    def to_dict(self) -> dict[str, str]:
        """Convert to JSON-serializable dict."""
        return {"algorithm": self.algorithm, "digest": self.digest}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContentHash:
        """Create from a dict (e.g., from JSON)."""
        return cls(algorithm=data["algorithm"], digest=data["digest"])


@dataclass
class LineageReference:
    """Reference to another artifact in the lineage graph."""

    artifact_id: str
    relation: Literal[
        "derived_from",
        "generated_by",
        "validated_by",
        "config_from",
        "sibling_of",
        "supersedes",
    ]
    content_hash: ContentHash | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "relation": self.relation,
        }
        if self.content_hash:
            result["content_hash"] = self.content_hash.to_dict()
        return result


@dataclass
class Lineage:
    """Lineage information for an artifact."""

    run_id: str
    inputs: list[LineageReference] = field(default_factory=list)
    stage_name: str | None = None
    outputs: list[LineageReference] = field(default_factory=list)
    dataset_id: str | None = None
    dataset_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "run_id": self.run_id,
            "inputs": [inp.to_dict() for inp in self.inputs],
        }
        if self.stage_name:
            result["stage_name"] = self.stage_name
        if self.outputs:
            result["outputs"] = [out.to_dict() for out in self.outputs]
        if self.dataset_id:
            result["dataset_id"] = self.dataset_id
        if self.dataset_version:
            result["dataset_version"] = self.dataset_version
        return result


@dataclass
class Provenance:
    """Provenance information for an artifact."""

    generator: str
    generator_version: str
    hostname: str
    username: str | None = None
    command: str | None = None
    working_directory: str | None = None
    ci_run_id: str | None = None
    ci_job_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "generator": self.generator,
            "generator_version": self.generator_version,
            "hostname": self.hostname,
        }
        if self.username:
            result["username"] = self.username
        if self.command:
            result["command"] = self.command
        if self.working_directory:
            result["working_directory"] = self.working_directory
        if self.ci_run_id:
            result["ci_run_id"] = self.ci_run_id
        if self.ci_job_url:
            result["ci_job_url"] = self.ci_job_url
        return result


@dataclass
class ArtifactManifest:
    """Full artifact manifest conforming to artifact.v1 schema."""

    artifact_id: str
    artifact_type: ArtifactType
    content_hash: ContentHash
    byte_size: int
    created_utc: str
    provenance: Provenance
    roles: list[ArtifactRole]
    lineage: Lineage
    storage_path: str | None = None
    media_type: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict conforming to artifact.v1 schema."""
        result: dict[str, Any] = {
            "schema_version": 1,
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "content_hash": self.content_hash.to_dict(),
            "byte_size": self.byte_size,
            "created_utc": self.created_utc,
            "provenance": self.provenance.to_dict(),
            "roles": list(self.roles),
            "lineage": self.lineage.to_dict(),
        }
        if self.storage_path:
            result["storage_path"] = self.storage_path
        if self.media_type:
            result["media_type"] = self.media_type
        if self.tags:
            result["tags"] = dict(self.tags)
        if self.annotations:
            result["annotations"] = dict(self.annotations)
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @property
    def spec_id(self) -> str:
        """Compute a short spec ID from the content hash.

        Returns:
            A 12-character base32-encoded identifier derived from the content hash.
        """
        return compute_spec_id(self.content_hash.digest)


def compute_spec_id(digest: str, length: int = 12) -> str:
    """Compute a short, human-friendly spec ID from a SHA256 digest.

    This generates a canonical identifier suitable for use as a spec_id,
    following the design pattern: spec_id = base32(sha256_digest)[0:length]

    The result uses lowercase alphanumeric characters without padding,
    making it suitable for use in filenames, URLs, and CLI arguments.

    Args:
        digest: A SHA256 hex digest (64 characters).
        length: Desired length of the spec ID (default 12).

    Returns:
        A short base32-encoded identifier.

    Example:
        >>> digest = hashlib.sha256(b"hello").hexdigest()
        >>> compute_spec_id(digest)
        'l5ua4w36g7wa'

    Note:
        This follows Section 9.1 of the design document:
        design_hash = SHA256(canonical_resolved_design_json_bytes)
        coupon_id = base32(design_hash)[0:12]
    """
    digest_bytes = bytes.fromhex(digest)
    encoded = base64.b32encode(digest_bytes).decode("ascii").lower().rstrip("=")
    return encoded[:length]


class ArtifactStoreError(Exception):
    """Base exception for ArtifactStore errors."""


class ArtifactNotFoundError(ArtifactStoreError):
    """Raised when an artifact is not found in the store."""


class ArtifactExistsError(ArtifactStoreError):
    """Raised when attempting to write an artifact that already exists."""


class ArtifactStore:
    """Content-addressed artifact storage with atomic writes.

    This class manages artifacts in a content-addressed store where each artifact
    is stored by its SHA256 hash. The store uses atomic rename operations to ensure
    consistency even in case of crashes or concurrent access.

    Directory structure:
        data/
            objects/
                <2-char prefix>/
                    <sha256 hex digest>   # artifact content
            manifests/
                <artifact_id>.json         # artifact manifest

    Example usage:
        store = ArtifactStore(Path("data"))
        manifest = store.put(
            content=b"hello world",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
            generator="test",
            generator_version="1.0.0",
        )
        content = store.get(manifest.content_hash.digest)
    """

    HASH_ALGORITHM: HashAlgorithm = "sha256"
    OBJECTS_DIR = "objects"
    MANIFESTS_DIR = "manifests"
    TMP_SUFFIX = ".tmp"

    def __init__(
        self,
        root: Path | str,
        generator: str = "formula_foundry",
        generator_version: str = "0.1.0",
    ) -> None:
        """Initialize the artifact store.

        Args:
            root: Root directory for the data store.
            generator: Name of the generator tool for provenance.
            generator_version: Version of the generator for provenance.
        """
        self.root = Path(root)
        self.objects_dir = self.root / self.OBJECTS_DIR
        self.manifests_dir = self.root / self.MANIFESTS_DIR
        self.generator = generator
        self.generator_version = generator_version
        self._hostname = socket.gethostname()
        self._username = os.environ.get("USER") or os.environ.get("USERNAME")

    def _ensure_dirs(self) -> None:
        """Ensure the store directories exist."""
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def _object_path(self, digest: str) -> Path:
        """Get the path for an object by its hash digest.

        Uses a 2-character prefix directory for sharding (like Git).
        """
        prefix = digest[:2]
        return self.objects_dir / prefix / digest

    def _manifest_path(self, artifact_id: str) -> Path:
        """Get the path for an artifact manifest."""
        return self.manifests_dir / f"{artifact_id}.json"

    def _compute_hash(self, content: bytes) -> ContentHash:
        """Compute the SHA256 hash of content."""
        digest = hashlib.sha256(content).hexdigest()
        return ContentHash(algorithm=self.HASH_ALGORITHM, digest=digest)

    def _compute_hash_streaming(self, stream: BinaryIO, chunk_size: int = 65536) -> tuple[ContentHash, int]:
        """Compute hash from a stream, returning hash and total bytes."""
        hasher = hashlib.sha256()
        total_bytes = 0
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            total_bytes += len(chunk)
        return ContentHash(algorithm=self.HASH_ALGORITHM, digest=hasher.hexdigest()), total_bytes

    def _atomic_write(self, path: Path, content: bytes) -> None:
        """Write content atomically using tmp file + rename pattern.

        Args:
            path: Target path for the file.
            content: Content to write.

        The write is performed to a temporary file in the same directory,
        then atomically renamed to the target path. This ensures that the
        target file is either fully written or not present at all.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create temp file in the same directory to ensure same filesystem
        fd, tmp_path_str = tempfile.mkstemp(
            suffix=self.TMP_SUFFIX,
            prefix=path.name + ".",
            dir=path.parent,
        )
        tmp_path = Path(tmp_path_str)

        try:
            with os.fdopen(fd, "wb") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())  # Ensure data is on disk

            # Atomic rename
            tmp_path.rename(path)
        except Exception:
            # Clean up temp file on failure
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _atomic_write_stream(self, path: Path, stream: BinaryIO, chunk_size: int = 65536) -> tuple[ContentHash, int]:
        """Write stream content atomically, computing hash during write.

        Args:
            path: Target path for the file.
            stream: Binary stream to read content from.
            chunk_size: Size of chunks to read.

        Returns:
            Tuple of (content_hash, byte_size).
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path_str = tempfile.mkstemp(
            suffix=self.TMP_SUFFIX,
            prefix=path.name + ".",
            dir=path.parent,
        )
        tmp_path = Path(tmp_path_str)

        hasher = hashlib.sha256()
        total_bytes = 0

        try:
            with os.fdopen(fd, "wb") as f:
                while True:
                    chunk = stream.read(chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)
                    f.write(chunk)
                    total_bytes += len(chunk)
                f.flush()
                os.fsync(f.fileno())

            content_hash = ContentHash(algorithm=self.HASH_ALGORITHM, digest=hasher.hexdigest())

            # Now we know the final path based on the hash
            final_path = self._object_path(content_hash.digest)
            final_path.parent.mkdir(parents=True, exist_ok=True)

            if final_path.exists():
                # Content already exists, discard temp file
                tmp_path.unlink()
            else:
                tmp_path.rename(final_path)

            return content_hash, total_bytes
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _generate_artifact_id(self) -> str:
        """Generate a unique artifact ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        unique = uuid.uuid4().hex[:8]
        return f"art-{timestamp}-{unique}"

    def _now_utc_iso(self) -> str:
        """Get current UTC time in ISO 8601 format."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def put(
        self,
        content: bytes,
        artifact_type: ArtifactType,
        roles: list[ArtifactRole],
        run_id: str,
        generator: str | None = None,
        generator_version: str | None = None,
        artifact_id: str | None = None,
        stage_name: str | None = None,
        inputs: list[LineageReference] | None = None,
        media_type: str | None = None,
        tags: dict[str, str] | None = None,
        annotations: dict[str, Any] | None = None,
        allow_overwrite: bool = False,
    ) -> ArtifactManifest:
        """Store content as a new artifact.

        Args:
            content: The binary content to store.
            artifact_type: Type of the artifact.
            roles: Semantic roles the artifact serves.
            run_id: ID of the run producing this artifact.
            generator: Name of the generator (defaults to store default).
            generator_version: Version of the generator (defaults to store default).
            artifact_id: Optional explicit artifact ID (auto-generated if not provided).
            stage_name: Optional pipeline stage name.
            inputs: Optional list of input artifact references.
            media_type: Optional MIME type.
            tags: Optional key-value tags.
            annotations: Optional free-form annotations.
            allow_overwrite: If True, allow overwriting existing manifest.

        Returns:
            ArtifactManifest describing the stored artifact.

        Raises:
            ArtifactExistsError: If artifact_id already exists and allow_overwrite is False.
        """
        self._ensure_dirs()

        # Compute hash
        content_hash = self._compute_hash(content)
        byte_size = len(content)

        # Generate or validate artifact ID
        if artifact_id is None:
            artifact_id = self._generate_artifact_id()
        else:
            # Check for existing manifest
            manifest_path = self._manifest_path(artifact_id)
            if manifest_path.exists() and not allow_overwrite:
                raise ArtifactExistsError(f"Artifact already exists: {artifact_id}")

        # Write object (content-addressed, so duplicates are harmless)
        object_path = self._object_path(content_hash.digest)
        if not object_path.exists():
            self._atomic_write(object_path, content)

        # Build manifest
        provenance = Provenance(
            generator=generator or self.generator,
            generator_version=generator_version or self.generator_version,
            hostname=self._hostname,
            username=self._username,
        )

        lineage = Lineage(
            run_id=run_id,
            inputs=inputs or [],
            stage_name=stage_name,
        )

        storage_path = str(object_path.relative_to(self.root))

        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            content_hash=content_hash,
            byte_size=byte_size,
            created_utc=self._now_utc_iso(),
            provenance=provenance,
            roles=list(roles),
            lineage=lineage,
            storage_path=storage_path,
            media_type=media_type,
            tags=tags or {},
            annotations=annotations or {},
        )

        # Write manifest atomically
        manifest_path = self._manifest_path(artifact_id)
        self._atomic_write(manifest_path, manifest.to_json().encode("utf-8"))

        return manifest

    def put_file(
        self,
        file_path: Path | str,
        artifact_type: ArtifactType,
        roles: list[ArtifactRole],
        run_id: str,
        **kwargs: Any,
    ) -> ArtifactManifest:
        """Store a file as a new artifact.

        Args:
            file_path: Path to the file to store.
            artifact_type: Type of the artifact.
            roles: Semantic roles the artifact serves.
            run_id: ID of the run producing this artifact.
            **kwargs: Additional arguments passed to put().

        Returns:
            ArtifactManifest describing the stored artifact.
        """
        file_path = Path(file_path)
        content = file_path.read_bytes()
        return self.put(
            content=content,
            artifact_type=artifact_type,
            roles=roles,
            run_id=run_id,
            **kwargs,
        )

    def get(self, digest: str) -> bytes:
        """Retrieve content by its hash digest.

        Args:
            digest: SHA256 hex digest of the content.

        Returns:
            The stored content as bytes.

        Raises:
            ArtifactNotFoundError: If the content is not found.
        """
        object_path = self._object_path(digest)
        if not object_path.exists():
            raise ArtifactNotFoundError(f"Object not found: {digest}")
        return object_path.read_bytes()

    def get_by_id(self, artifact_id: str) -> bytes:
        """Retrieve artifact content by its artifact ID.

        Args:
            artifact_id: The artifact ID.

        Returns:
            The stored content as bytes.

        Raises:
            ArtifactNotFoundError: If the artifact is not found.
        """
        manifest = self.get_manifest(artifact_id)
        return self.get(manifest.content_hash.digest)

    def get_manifest(self, artifact_id: str) -> ArtifactManifest:
        """Retrieve an artifact manifest by ID.

        Args:
            artifact_id: The artifact ID.

        Returns:
            The artifact manifest.

        Raises:
            ArtifactNotFoundError: If the manifest is not found.
        """
        manifest_path = self._manifest_path(artifact_id)
        if not manifest_path.exists():
            raise ArtifactNotFoundError(f"Manifest not found: {artifact_id}")

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return self._manifest_from_dict(data)

    def _manifest_from_dict(self, data: dict[str, Any]) -> ArtifactManifest:
        """Parse a manifest from a dict."""
        content_hash = ContentHash.from_dict(data["content_hash"])

        prov_data = data["provenance"]
        provenance = Provenance(
            generator=prov_data["generator"],
            generator_version=prov_data["generator_version"],
            hostname=prov_data["hostname"],
            username=prov_data.get("username"),
            command=prov_data.get("command"),
            working_directory=prov_data.get("working_directory"),
            ci_run_id=prov_data.get("ci_run_id"),
            ci_job_url=prov_data.get("ci_job_url"),
        )

        lin_data = data["lineage"]
        inputs = []
        for inp in lin_data.get("inputs", []):
            inp_hash = ContentHash.from_dict(inp["content_hash"]) if "content_hash" in inp else None
            inputs.append(
                LineageReference(
                    artifact_id=inp["artifact_id"],
                    relation=inp["relation"],
                    content_hash=inp_hash,
                )
            )

        outputs = []
        for out in lin_data.get("outputs", []):
            out_hash = ContentHash.from_dict(out["content_hash"]) if "content_hash" in out else None
            outputs.append(
                LineageReference(
                    artifact_id=out["artifact_id"],
                    relation=out["relation"],
                    content_hash=out_hash,
                )
            )

        lineage = Lineage(
            run_id=lin_data["run_id"],
            inputs=inputs,
            stage_name=lin_data.get("stage_name"),
            outputs=outputs,
            dataset_id=lin_data.get("dataset_id"),
            dataset_version=lin_data.get("dataset_version"),
        )

        return ArtifactManifest(
            artifact_id=data["artifact_id"],
            artifact_type=data["artifact_type"],
            content_hash=content_hash,
            byte_size=data["byte_size"],
            created_utc=data["created_utc"],
            provenance=provenance,
            roles=data["roles"],
            lineage=lineage,
            storage_path=data.get("storage_path"),
            media_type=data.get("media_type"),
            tags=data.get("tags", {}),
            annotations=data.get("annotations", {}),
        )

    def exists(self, digest: str) -> bool:
        """Check if content with the given hash exists in the store.

        Args:
            digest: SHA256 hex digest.

        Returns:
            True if the content exists, False otherwise.
        """
        return self._object_path(digest).exists()

    def exists_by_id(self, artifact_id: str) -> bool:
        """Check if an artifact with the given ID exists.

        Args:
            artifact_id: The artifact ID.

        Returns:
            True if the manifest exists, False otherwise.
        """
        return self._manifest_path(artifact_id).exists()

    def list_manifests(self) -> list[str]:
        """List all artifact IDs in the store.

        Returns:
            List of artifact IDs.
        """
        if not self.manifests_dir.exists():
            return []
        return [p.stem for p in self.manifests_dir.glob("*.json")]

    def verify(self, artifact_id: str) -> bool:
        """Verify the integrity of a stored artifact.

        Args:
            artifact_id: The artifact ID to verify.

        Returns:
            True if the stored content matches its recorded hash.

        Raises:
            ArtifactNotFoundError: If the artifact is not found.
        """
        manifest = self.get_manifest(artifact_id)
        content = self.get(manifest.content_hash.digest)
        actual_hash = self._compute_hash(content)
        return actual_hash.digest == manifest.content_hash.digest

    def delete(self, artifact_id: str, delete_content: bool = False) -> None:
        """Delete an artifact manifest.

        Args:
            artifact_id: The artifact ID to delete.
            delete_content: If True, also delete the content object if no other
                          manifests reference it.

        Raises:
            ArtifactNotFoundError: If the artifact is not found.
        """
        manifest = self.get_manifest(artifact_id)
        manifest_path = self._manifest_path(artifact_id)
        manifest_path.unlink()

        if delete_content:
            # Check if any other manifest references this content
            digest = manifest.content_hash.digest
            still_referenced = False
            for other_id in self.list_manifests():
                other_manifest = self.get_manifest(other_id)
                if other_manifest.content_hash.digest == digest:
                    still_referenced = True
                    break

            if not still_referenced:
                object_path = self._object_path(digest)
                if object_path.exists():
                    object_path.unlink()
                # Try to remove empty prefix directory
                try:
                    object_path.parent.rmdir()
                except OSError:
                    pass  # Directory not empty

    def compute_spec_id(self, content: bytes, length: int = 12) -> str:
        """Compute a spec ID from content without storing it.

        This is useful for computing a canonical identifier for content
        before deciding whether to store it.

        Args:
            content: The binary content to hash.
            length: Desired length of the spec ID (default 12).

        Returns:
            A short base32-encoded identifier derived from the content hash.
        """
        content_hash = self._compute_hash(content)
        return compute_spec_id(content_hash.digest, length)

    def get_spec_id(self, artifact_id: str) -> str:
        """Get the spec ID for an existing artifact.

        Args:
            artifact_id: The artifact ID.

        Returns:
            The spec ID derived from the artifact's content hash.

        Raises:
            ArtifactNotFoundError: If the artifact is not found.
        """
        manifest = self.get_manifest(artifact_id)
        return manifest.spec_id
