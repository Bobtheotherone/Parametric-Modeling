"""Tests for toolchain lock loader module.

Satisfies:
    - CP-1.1: Toolchain lock file loading and validation
    - CP-1.3: toolchain_hash computation and ToolchainConfig dataclass
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from formula_foundry.coupongen.toolchain import (
    DEFAULT_LOCK_PATH,
    ToolchainConfig,
    ToolchainLoadError,
    compute_toolchain_hash,
    load_toolchain_lock,
    load_toolchain_lock_from_dict,
)


class TestToolchainConfig:
    """Tests for ToolchainConfig dataclass."""

    def test_frozen(self) -> None:
        """ToolchainConfig should be immutable."""
        config = ToolchainConfig(
            schema_version="1.0",
            kicad_version="9.0.7",
            docker_image="kicad/kicad:9.0.7",
            docker_digest="sha256:" + "a" * 64,
            toolchain_hash="deadbeef",
        )
        with pytest.raises(AttributeError):
            config.kicad_version = "9.0.8"  # type: ignore[misc]

    def test_pinned_image_ref_with_digest(self) -> None:
        """pinned_image_ref should include digest when available."""
        config = ToolchainConfig(
            schema_version="1.0",
            kicad_version="9.0.7",
            docker_image="kicad/kicad:9.0.7",
            docker_digest="sha256:" + "b" * 64,
            toolchain_hash="deadbeef",
        )
        assert config.pinned_image_ref == f"kicad/kicad:9.0.7@{'sha256:' + 'b' * 64}"

    def test_pinned_image_ref_with_embedded_digest(self) -> None:
        """pinned_image_ref should honor embedded digest to avoid double tags."""
        digest = "sha256:" + "c" * 64
        config = ToolchainConfig(
            schema_version="1.0",
            kicad_version="9.0.7",
            docker_image=f"kicad/kicad:9.0.7@{digest}",
            docker_digest=digest,
            toolchain_hash="deadbeef",
        )
        assert config.pinned_image_ref == f"kicad/kicad:9.0.7@{digest}"

    def test_to_manifest_dict(self) -> None:
        """to_manifest_dict should return proper manifest structure."""
        config = ToolchainConfig(
            schema_version="1.0",
            kicad_version="9.0.7",
            docker_image="kicad/kicad:9.0.7",
            docker_digest="sha256:" + "d" * 64,
            toolchain_hash="deadbeef",
        )
        manifest = config.to_manifest_dict()
        assert manifest == {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:" + "d" * 64,
            "toolchain_hash": "deadbeef",
        }


class TestComputeToolchainHash:
    """Tests for compute_toolchain_hash function."""

    def test_deterministic(self) -> None:
        """Hash should be deterministic for same input."""
        lock_data = {
            "schema_version": "1.0",
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
        }
        hash1 = compute_toolchain_hash(lock_data)
        hash2 = compute_toolchain_hash(lock_data)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex string

    def test_excludes_toolchain_hash_field(self) -> None:
        """Hash computation should exclude toolchain_hash field."""
        lock_data_without = {
            "schema_version": "1.0",
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
        }
        lock_data_with = {
            "schema_version": "1.0",
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "toolchain_hash": "some_existing_hash",
        }
        assert compute_toolchain_hash(lock_data_without) == compute_toolchain_hash(lock_data_with)

    def test_different_inputs_different_hash(self) -> None:
        """Different inputs should produce different hashes."""
        lock_data1 = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
        }
        lock_data2 = {
            "kicad_version": "9.0.8",
            "docker_image": "kicad/kicad:9.0.8",
        }
        assert compute_toolchain_hash(lock_data1) != compute_toolchain_hash(lock_data2)

    def test_canonical_json_ordering(self) -> None:
        """Hash should be stable regardless of key order."""
        lock_data1 = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "schema_version": "1.0",
        }
        lock_data2 = {
            "schema_version": "1.0",
            "docker_image": "kicad/kicad:9.0.7",
            "kicad_version": "9.0.7",
        }
        assert compute_toolchain_hash(lock_data1) == compute_toolchain_hash(lock_data2)


class TestLoadToolchainLock:
    """Tests for load_toolchain_lock function."""

    def test_load_valid_lock_file(self, tmp_path: Path) -> None:
        """Should load a valid lock file successfully."""
        lock_data = {
            "schema_version": "1.0",
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:" + "e" * 64,
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        config = load_toolchain_lock(lock_path=lock_path)

        assert config.schema_version == "1.0"
        assert config.kicad_version == "9.0.7"
        assert config.docker_image == "kicad/kicad:9.0.7"
        assert config.docker_digest == "sha256:" + "e" * 64
        assert len(config.toolchain_hash) == 64

    def test_load_with_repo_root(self, tmp_path: Path) -> None:
        """Should load from default path relative to repo_root."""
        toolchain_dir = tmp_path / "toolchain"
        toolchain_dir.mkdir()
        lock_data = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:" + "f" * 64,
        }
        lock_path = toolchain_dir / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        config = load_toolchain_lock(repo_root=tmp_path)

        assert config.kicad_version == "9.0.7"

    def test_missing_lock_file(self, tmp_path: Path) -> None:
        """Should raise ToolchainLoadError for missing file."""
        lock_path = tmp_path / "nonexistent.lock.json"
        with pytest.raises(ToolchainLoadError, match="not found"):
            load_toolchain_lock(lock_path=lock_path)

    def test_invalid_json(self, tmp_path: Path) -> None:
        """Should raise ToolchainLoadError for invalid JSON."""
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text("not valid json {")
        with pytest.raises(ToolchainLoadError, match="Invalid JSON"):
            load_toolchain_lock(lock_path=lock_path)

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        """Should raise ToolchainLoadError when required fields missing."""
        lock_data = {
            "schema_version": "1.0",
            # Missing kicad_version, docker_image, and docker_digest
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        with pytest.raises(ToolchainLoadError, match="Missing required fields"):
            load_toolchain_lock(lock_path=lock_path)

    def test_default_schema_version(self, tmp_path: Path) -> None:
        """Should use default schema_version if not specified."""
        lock_data = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:" + "1" * 64,
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        config = load_toolchain_lock(lock_path=lock_path)
        assert config.schema_version == "1.0"

    def test_missing_docker_digest(self, tmp_path: Path) -> None:
        """Should raise ToolchainLoadError when docker_digest missing."""
        lock_data = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        with pytest.raises(ToolchainLoadError, match="docker_digest"):
            load_toolchain_lock(lock_path=lock_path)

    def test_placeholder_docker_digest(self, tmp_path: Path) -> None:
        """Should reject placeholder docker_digest."""
        lock_data = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:PLACEHOLDER",
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        with pytest.raises(ToolchainLoadError, match="PLACEHOLDER"):
            load_toolchain_lock(lock_path=lock_path)

    def test_invalid_sha256_format_too_short(self, tmp_path: Path) -> None:
        """Should reject digest not matching sha256:<64-hex> format (too short)."""
        lock_data = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:abc123",  # Too short - not 64 hex chars
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        with pytest.raises(ToolchainLoadError, match="64-hex"):
            load_toolchain_lock(lock_path=lock_path)

    def test_invalid_sha256_format_missing_prefix(self, tmp_path: Path) -> None:
        """Should reject digest without sha256: prefix."""
        lock_data = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "a" * 64,  # No sha256: prefix
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        with pytest.raises(ToolchainLoadError, match="sha256"):
            load_toolchain_lock(lock_path=lock_path)

    def test_invalid_sha256_format_wrong_chars(self, tmp_path: Path) -> None:
        """Should reject digest with non-hex characters."""
        lock_data = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:" + "g" * 64,  # 'g' is not valid hex
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        with pytest.raises(ToolchainLoadError, match="64-hex"):
            load_toolchain_lock(lock_path=lock_path)

    def test_docker_image_digest_mismatch(self, tmp_path: Path) -> None:
        """Should reject when docker_image has embedded digest that differs from docker_digest."""
        lock_data = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7@sha256:" + "a" * 64,
            "docker_digest": "sha256:" + "b" * 64,  # Different from embedded
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        with pytest.raises(ToolchainLoadError, match="does not match"):
            load_toolchain_lock(lock_path=lock_path)

    def test_docker_image_with_matching_embedded_digest(self, tmp_path: Path) -> None:
        """Should accept docker_image with embedded digest that matches docker_digest."""
        digest = "sha256:" + "c" * 64
        lock_data = {
            "kicad_version": "9.0.7",
            "docker_image": f"kicad/kicad:9.0.7@{digest}",
            "docker_digest": digest,
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        config = load_toolchain_lock(lock_path=lock_path)
        # docker_image should be normalized to remove embedded digest
        assert config.docker_image == "kicad/kicad:9.0.7"
        assert config.docker_digest == digest


class TestLoadToolchainLockFromDict:
    """Tests for load_toolchain_lock_from_dict function."""

    def test_load_from_dict(self) -> None:
        """Should create config from dictionary."""
        lock_data = {
            "schema_version": "1.0",
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:" + "2" * 64,
        }
        config = load_toolchain_lock_from_dict(lock_data)

        assert config.kicad_version == "9.0.7"
        assert config.docker_image == "kicad/kicad:9.0.7"

    def test_missing_required_fields_from_dict(self) -> None:
        """Should raise ToolchainLoadError for invalid dict."""
        lock_data = {"schema_version": "1.0"}
        with pytest.raises(ToolchainLoadError, match="Missing required fields"):
            load_toolchain_lock_from_dict(lock_data)


class TestDefaultLockPath:
    """Tests for DEFAULT_LOCK_PATH constant."""

    def test_default_path_structure(self) -> None:
        """DEFAULT_LOCK_PATH should point to toolchain directory."""
        assert DEFAULT_LOCK_PATH == Path("toolchain/kicad.lock.json")


class TestRealLockFile:
    """Tests against the actual kicad.lock.json in the repo."""

    def test_load_repo_lock_file(self) -> None:
        """Should successfully load the actual lock file from the repo."""
        # Find repo root by looking for toolchain directory
        test_file = Path(__file__)
        repo_root = test_file.parent.parent

        lock_path = repo_root / "toolchain" / "kicad.lock.json"
        if lock_path.exists():
            config = load_toolchain_lock(lock_path=lock_path)
            assert config.kicad_version == "9.0.7"
            assert config.docker_image == "kicad/kicad:9.0.7"
            assert len(config.toolchain_hash) == 64
            assert config.docker_digest.startswith("sha256:")
        else:
            pytest.skip("Lock file not found at expected location")
