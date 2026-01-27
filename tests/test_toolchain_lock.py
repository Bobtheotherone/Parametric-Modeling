"""Tests for formula_foundry.toolchain.lock module.

Covers: deterministic hashing, digest validation, lock file loading,
writing, integrity verification, and placeholder detection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from formula_foundry.toolchain.lock import (
    ToolchainLockEntry,
    ToolchainLockError,
    compute_lock_hash,
    is_placeholder_digest,
    is_valid_sha256_digest,
    load_toolchain_lock,
    validate_digest,
    verify_lock_integrity,
    write_toolchain_lock,
)

# --- Digest validation ---


class TestDigestValidation:
    def test_valid_digest(self) -> None:
        assert is_valid_sha256_digest("sha256:" + "a" * 64)

    def test_invalid_no_prefix(self) -> None:
        assert not is_valid_sha256_digest("a" * 64)

    def test_invalid_too_short(self) -> None:
        assert not is_valid_sha256_digest("sha256:abc")

    def test_invalid_non_hex(self) -> None:
        assert not is_valid_sha256_digest("sha256:" + "g" * 64)

    def test_validate_digest_ok(self) -> None:
        validate_digest("sha256:" + "abcdef12" * 8)

    def test_validate_digest_no_prefix_raises(self) -> None:
        with pytest.raises(ToolchainLockError, match="sha256:"):
            validate_digest("abc123")

    def test_validate_digest_too_short_raises(self) -> None:
        with pytest.raises(ToolchainLockError, match="64-hex"):
            validate_digest("sha256:abc")

    def test_validate_digest_placeholder_raises(self) -> None:
        with pytest.raises(ToolchainLockError, match="placeholder"):
            validate_digest("sha256:" + "0" * 64)

    def test_validate_digest_context(self) -> None:
        with pytest.raises(ToolchainLockError, match="myctx"):
            validate_digest("sha256:short", context="myctx")


class TestPlaceholderDetection:
    def test_all_zeros(self) -> None:
        assert is_placeholder_digest("sha256:" + "0" * 64)

    def test_zeros_plus_one(self) -> None:
        assert is_placeholder_digest("sha256:" + "0" * 63 + "1")

    def test_placeholder_keyword(self) -> None:
        assert is_placeholder_digest("sha256:PLACEHOLDER")

    def test_unknown_keyword(self) -> None:
        assert is_placeholder_digest("sha256:UNKNOWN_VALUE")

    def test_real_digest_not_placeholder(self) -> None:
        assert not is_placeholder_digest("sha256:4ddaa54d9ead1f1b453e10a8420e0fcfba693e2143ee14b8b9c3b3c63b2a320f")

    def test_mostly_zero_is_placeholder(self) -> None:
        # 60 zeros + 4 hex chars = placeholder
        assert is_placeholder_digest("sha256:" + "0" * 60 + "abcd")


# --- Deterministic hashing ---


class TestComputeLockHash:
    def test_deterministic(self) -> None:
        data = {"version": "1.0", "tools": [{"name": "x"}]}
        assert compute_lock_hash(data) == compute_lock_hash(data)

    def test_key_order_independent(self) -> None:
        d1 = {"b": 2, "a": 1}
        d2 = {"a": 1, "b": 2}
        assert compute_lock_hash(d1) == compute_lock_hash(d2)

    def test_excludes_lock_hash(self) -> None:
        d1 = {"a": 1}
        d2 = {"a": 1, "lock_hash": "should_be_ignored"}
        assert compute_lock_hash(d1) == compute_lock_hash(d2)

    def test_different_data_different_hash(self) -> None:
        assert compute_lock_hash({"v": "1"}) != compute_lock_hash({"v": "2"})

    def test_hash_is_64_hex(self) -> None:
        h = compute_lock_hash({"x": 1})
        assert len(h) == 64
        int(h, 16)  # must be valid hex


# --- ToolchainLockEntry ---


class TestToolchainLockEntry:
    def test_pinned_ref(self) -> None:
        e = ToolchainLockEntry(
            name="openems",
            version="0.0.35",
            docker_image="ghcr.io/openems:0.0.35",
            docker_digest="sha256:" + "a" * 64,
        )
        assert e.pinned_ref == f"ghcr.io/openems:0.0.35@sha256:{'a' * 64}"

    def test_pinned_ref_strips_embedded_digest(self) -> None:
        digest = "sha256:" + "b" * 64
        e = ToolchainLockEntry(
            name="x",
            version="1",
            docker_image=f"img:tag@{digest}",
            docker_digest=digest,
        )
        assert e.pinned_ref == f"img:tag@{digest}"

    def test_to_dict(self) -> None:
        e = ToolchainLockEntry(name="t", version="1", docker_image="i", docker_digest="d")
        d = e.to_dict()
        assert d == {
            "name": "t",
            "version": "1",
            "docker_image": "i",
            "docker_digest": "d",
        }

    def test_to_dict_with_extras(self) -> None:
        e = ToolchainLockEntry(
            name="t",
            version="1",
            docker_image="i",
            docker_digest="d",
            extras={"foo": "bar"},
        )
        assert e.to_dict()["extras"] == {"foo": "bar"}

    def test_frozen(self) -> None:
        e = ToolchainLockEntry(name="t", version="1", docker_image="i", docker_digest="d")
        with pytest.raises(AttributeError):
            e.name = "other"  # type: ignore[misc]


# --- Loading ---


class TestLoadToolchainLock:
    def test_load_multi_tool(self, tmp_path: Path) -> None:
        lock = {
            "schema_version": "1.0",
            "tools": [
                {
                    "name": "openems",
                    "version": "0.0.35",
                    "docker_image": "img:0.0.35",
                    "docker_digest": "sha256:" + "a" * 64,
                },
                {
                    "name": "csxcad",
                    "version": "0.6.3",
                    "docker_image": "img2:0.6.3",
                    "docker_digest": "sha256:" + "b" * 64,
                },
            ],
        }
        p = tmp_path / "lock.json"
        p.write_text(json.dumps(lock))
        entries = load_toolchain_lock(p)
        assert len(entries) == 2
        assert entries[0].name == "openems"
        assert entries[1].name == "csxcad"

    def test_load_single_tool_shorthand(self, tmp_path: Path) -> None:
        lock = {
            "version": "0.0.35",
            "docker_image": "img:0.0.35",
            "docker_digest": "sha256:" + "c" * 64,
        }
        p = tmp_path / "openems.lock.json"
        p.write_text(json.dumps(lock))
        entries = load_toolchain_lock(p)
        assert len(entries) == 1
        assert entries[0].name == "openems"
        assert entries[0].version == "0.0.35"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ToolchainLockError, match="not found"):
            load_toolchain_lock(tmp_path / "nope.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{invalid")
        with pytest.raises(ToolchainLockError, match="Invalid JSON"):
            load_toolchain_lock(p)

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        p = tmp_path / "lock.json"
        p.write_text(json.dumps({"tools": [{"name": "x"}]}))
        with pytest.raises(ToolchainLockError, match="missing required"):
            load_toolchain_lock(p)

    def test_placeholder_digest_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "lock.json"
        p.write_text(
            json.dumps(
                {
                    "tools": [
                        {
                            "name": "x",
                            "version": "1",
                            "docker_image": "i:1",
                            "docker_digest": "sha256:" + "0" * 64,
                        }
                    ]
                }
            )
        )
        with pytest.raises(ToolchainLockError, match="placeholder"):
            load_toolchain_lock(p)

    def test_embedded_digest_mismatch(self, tmp_path: Path) -> None:
        p = tmp_path / "lock.json"
        p.write_text(
            json.dumps(
                {
                    "tools": [
                        {
                            "name": "x",
                            "version": "1",
                            "docker_image": "i:1@sha256:" + "a" * 64,
                            "docker_digest": "sha256:" + "b" * 64,
                        }
                    ]
                }
            )
        )
        with pytest.raises(ToolchainLockError, match="does not match"):
            load_toolchain_lock(p)

    def test_embedded_digest_match_normalizes(self, tmp_path: Path) -> None:
        digest = "sha256:" + "d" * 64
        p = tmp_path / "lock.json"
        p.write_text(
            json.dumps(
                {
                    "tools": [
                        {
                            "name": "x",
                            "version": "1",
                            "docker_image": f"i:1@{digest}",
                            "docker_digest": digest,
                        }
                    ]
                }
            )
        )
        entries = load_toolchain_lock(p)
        assert entries[0].docker_image == "i:1"


# --- Writing and integrity ---


class TestWriteAndVerify:
    def test_round_trip(self, tmp_path: Path) -> None:
        entries = [
            ToolchainLockEntry(
                name="openems",
                version="0.0.35",
                docker_image="img:0.0.35",
                docker_digest="sha256:" + "a" * 64,
            )
        ]
        p = tmp_path / "lock.json"
        write_toolchain_lock(p, entries)

        assert p.exists()
        raw = json.loads(p.read_text())
        assert "lock_hash" in raw
        assert raw["schema_version"] == "1.0"

        # Verify integrity
        assert verify_lock_integrity(p) is True

    def test_integrity_mismatch(self, tmp_path: Path) -> None:
        entries = [ToolchainLockEntry(name="t", version="1", docker_image="i", docker_digest="sha256:" + "e" * 64)]
        p = tmp_path / "lock.json"
        write_toolchain_lock(p, entries)

        # Tamper with file
        raw = json.loads(p.read_text())
        raw["tools"][0]["version"] = "2"
        p.write_text(json.dumps(raw))

        with pytest.raises(ToolchainLockError, match="mismatch"):
            verify_lock_integrity(p)

    def test_deterministic_output(self, tmp_path: Path) -> None:
        entries = [ToolchainLockEntry(name="a", version="1", docker_image="x", docker_digest="sha256:" + "f" * 64)]
        p1 = tmp_path / "lock1.json"
        p2 = tmp_path / "lock2.json"
        write_toolchain_lock(p1, entries)
        write_toolchain_lock(p2, entries)
        assert p1.read_text() == p2.read_text()

    def test_verify_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ToolchainLockError, match="not found"):
            verify_lock_integrity(tmp_path / "nope.json")

    def test_verify_missing_lock_hash(self, tmp_path: Path) -> None:
        p = tmp_path / "lock.json"
        p.write_text(json.dumps({"tools": []}))
        with pytest.raises(ToolchainLockError, match="lock_hash"):
            verify_lock_integrity(p)
