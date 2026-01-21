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
    is_placeholder_digest,
    load_toolchain_lock,
    load_toolchain_lock_from_dict,
    resolve_docker_image_ref,
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

        with pytest.raises(ToolchainLoadError, match="placeholder"):
            load_toolchain_lock(lock_path=lock_path)

    def test_zeroed_docker_digest_placeholder(self, tmp_path: Path) -> None:
        """Should reject zeroed placeholder docker_digest."""
        lock_data = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:" + "0" * 63 + "1",
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        with pytest.raises(ToolchainLoadError, match="placeholder"):
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

    def test_docker_ref_missing_digest(self, tmp_path: Path) -> None:
        """Should reject docker_ref without embedded digest."""
        lock_data = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:" + "d" * 64,
            "docker_ref": "kicad/kicad:9.0.7",
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        with pytest.raises(ToolchainLoadError, match="docker_ref must include"):
            load_toolchain_lock(lock_path=lock_path)

    def test_docker_ref_digest_mismatch(self, tmp_path: Path) -> None:
        """Should reject docker_ref digest mismatching docker_digest."""
        lock_data = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:" + "e" * 64,
            "docker_ref": "kicad/kicad:9.0.7@sha256:" + "f" * 64,
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        with pytest.raises(ToolchainLoadError, match="docker_ref digest"):
            load_toolchain_lock(lock_path=lock_path)


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
        assert Path("toolchain/kicad.lock.json") == DEFAULT_LOCK_PATH


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


class TestIsPlaceholderDigest:
    """Tests for is_placeholder_digest function."""

    def test_zeroed_placeholder(self) -> None:
        """Should detect all-zeros as placeholder."""
        assert is_placeholder_digest("sha256:" + "0" * 64)

    def test_zeroed_with_one_placeholder(self) -> None:
        """Should detect zeros-with-trailing-1 as placeholder."""
        assert is_placeholder_digest("sha256:" + "0" * 63 + "1")

    def test_placeholder_literal(self) -> None:
        """Should detect PLACEHOLDER keyword."""
        assert is_placeholder_digest("sha256:PLACEHOLDER")

    def test_unknown_literal(self) -> None:
        """Should detect UNKNOWN keyword."""
        assert is_placeholder_digest("sha256:UNKNOWN_VALUE")

    def test_real_digest_not_placeholder(self) -> None:
        """Should not flag a real digest as placeholder."""
        real_digest = "sha256:4ddaa54d9ead1f1b453e10a8420e0fcfba693e2143ee14b8b9c3b3c63b2a320f"
        assert not is_placeholder_digest(real_digest)

    def test_random_hex_not_placeholder(self) -> None:
        """Should not flag random hex as placeholder."""
        random_digest = "sha256:a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        assert not is_placeholder_digest(random_digest)


class TestResolveDockerImageRef:
    """Tests for resolve_docker_image_ref function."""

    def test_local_mode_returns_spec_as_is(self, tmp_path: Path) -> None:
        """Local mode should return spec's docker_image unchanged."""
        spec_image = "kicad/kicad:9.0.7@sha256:" + "0" * 63 + "1"
        result = resolve_docker_image_ref(spec_image, lock_path=tmp_path / "nonexistent.json", mode="local")
        assert result == spec_image

    def test_docker_mode_placeholder_uses_lock_file(self, tmp_path: Path) -> None:
        """Docker mode with placeholder should use lock file digest."""
        lock_digest = "sha256:" + "a" * 64
        lock_data = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": lock_digest,
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        spec_image = "kicad/kicad:9.0.7@sha256:" + "0" * 63 + "1"  # placeholder
        result = resolve_docker_image_ref(spec_image, lock_path=lock_path, mode="docker")
        assert result == f"kicad/kicad:9.0.7@{lock_digest}"

    def test_docker_mode_no_digest_uses_lock_file(self, tmp_path: Path) -> None:
        """Docker mode with no digest in spec should use lock file digest."""
        lock_digest = "sha256:" + "b" * 64
        lock_data = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": lock_digest,
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        spec_image = "kicad/kicad:9.0.7"  # no digest
        result = resolve_docker_image_ref(spec_image, lock_path=lock_path, mode="docker")
        assert result == f"kicad/kicad:9.0.7@{lock_digest}"

    def test_docker_mode_matching_digest_uses_spec(self, tmp_path: Path) -> None:
        """Docker mode with matching digest should use spec's reference."""
        digest = "sha256:" + "c" * 64
        lock_data = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": digest,
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        spec_image = f"kicad/kicad:9.0.7@{digest}"
        result = resolve_docker_image_ref(spec_image, lock_path=lock_path, mode="docker")
        assert result == spec_image

    def test_docker_mode_mismatched_digest_fails(self, tmp_path: Path) -> None:
        """Docker mode with non-placeholder digest that differs should fail."""
        lock_digest = "sha256:" + "d" * 64
        spec_digest = "sha256:" + "e" * 64
        lock_data = {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": lock_digest,
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        spec_image = f"kicad/kicad:9.0.7@{spec_digest}"
        with pytest.raises(ToolchainLoadError, match="does not match"):
            resolve_docker_image_ref(spec_image, lock_path=lock_path, mode="docker")

    def test_docker_mode_missing_lock_file_fails(self, tmp_path: Path) -> None:
        """Docker mode without valid lock file should fail."""
        spec_image = "kicad/kicad:9.0.7@sha256:" + "0" * 63 + "1"
        with pytest.raises(ToolchainLoadError, match="Cannot resolve"):
            resolve_docker_image_ref(spec_image, lock_path=tmp_path / "nonexistent.json", mode="docker")


class TestGoldenSpecPlaceholderResolution:
    """Integration tests for golden specs with placeholder digests."""

    def test_golden_spec_placeholder_resolved_from_lock_file(self, tmp_path: Path) -> None:
        """Golden specs with placeholder should resolve to lock file digest.

        This test verifies REQ-M1: placeholder digests in specs are replaced
        with real digests from the lock file when running in docker mode.
        """
        # Create a mock lock file with real digest
        real_digest = "sha256:4ddaa54d9ead1f1b453e10a8420e0fcfba693e2143ee14b8b9c3b3c63b2a320f"
        lock_data = {
            "schema_version": "1.0",
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": real_digest,
        }
        lock_path = tmp_path / "kicad.lock.json"
        lock_path.write_text(json.dumps(lock_data))

        # Simulate a golden spec with placeholder digest
        placeholder_spec_image = "kicad/kicad:9.0.7@sha256:" + "0" * 63 + "1"

        # Resolve should use lock file digest
        resolved = resolve_docker_image_ref(
            placeholder_spec_image, lock_path=lock_path, mode="docker"
        )

        assert resolved == f"kicad/kicad:9.0.7@{real_digest}"
        assert "000000" not in resolved  # No placeholder in result
