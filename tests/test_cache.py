"""Tests for the three-level structural caching system (REQ-M1-020)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.coupongen.cache import (
    CacheError,
    CacheKey,
    CacheStats,
    StructuralCache,
    compute_cache_key,
    should_invalidate_cache,
    toolchain_hash_from_config,
)


# -----------------------------------------------------------------------------
# Fixtures and helpers
# -----------------------------------------------------------------------------


def _make_design_hash() -> str:
    """Create a valid 64-character hex hash."""
    return "a" * 64


def _make_toolchain_hash() -> str:
    """Create a valid 64-character hex hash."""
    return "b" * 64


def _make_alternate_toolchain_hash() -> str:
    """Create an alternate 64-character hex hash."""
    return "c" * 64


def _sample_resolved_design_data() -> dict[str, Any]:
    """Return sample resolved design data."""
    return {
        "schema_version": 1,
        "coupon_family": "F1_SINGLE_ENDED_VIA",
        "units": "nm",
        "parameters_nm": {
            "board.outline.width_nm": 20000000,
            "board.outline.length_nm": 80000000,
        },
        "derived_features": {
            "board_area_nm2": 1600000000000000,
            "trace_total_length_nm": 50000000,
        },
        "dimensionless_groups": {
            "board_aspect_ratio": 4.0,
            "cpwg_w_over_gap": 1.67,
        },
    }


def _create_sample_board_file(path: Path) -> None:
    """Create a sample KiCad board file."""
    content = """(kicad_pcb (version 20240101) (generator "test")
  (general (thickness 1.6))
  (paper "A4")
)
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _create_sample_fab_dir(path: Path) -> None:
    """Create a sample fab output directory."""
    gerber_dir = path / "gerbers"
    drill_dir = path / "drill"
    gerber_dir.mkdir(parents=True, exist_ok=True)
    drill_dir.mkdir(parents=True, exist_ok=True)

    (gerber_dir / "F.Cu.gbr").write_text("G04 Test Gerber*\n", encoding="utf-8")
    (gerber_dir / "B.Cu.gbr").write_text("G04 Test Gerber*\n", encoding="utf-8")
    (drill_dir / "drill.drl").write_text("M48\n%\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CacheKey tests
# -----------------------------------------------------------------------------


class TestCacheKey:
    """Tests for CacheKey dataclass."""

    def test_cache_key_creation(self) -> None:
        """Test creating a valid cache key."""
        key = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )
        assert key.design_hash == _make_design_hash()
        assert key.toolchain_hash == _make_toolchain_hash()

    def test_cache_key_combined_hash(self) -> None:
        """Test combined hash computation is deterministic."""
        key = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )
        # Combined hash should be consistent
        assert key.combined_hash == key.combined_hash
        assert len(key.combined_hash) == 64

    def test_cache_key_different_inputs_different_combined(self) -> None:
        """Test different inputs produce different combined hashes."""
        key1 = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )
        key2 = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_alternate_toolchain_hash(),
        )
        assert key1.combined_hash != key2.combined_hash

    def test_cache_key_short_key(self) -> None:
        """Test short key format for display."""
        key = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )
        assert key.short_key == f"{_make_design_hash()[:8]}_{_make_toolchain_hash()[:8]}"

    def test_cache_key_to_dict_roundtrip(self) -> None:
        """Test serialization roundtrip."""
        key = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )
        data = key.to_dict()
        restored = CacheKey.from_dict(data)
        assert restored == key

    def test_cache_key_matches_manifest(self) -> None:
        """Test manifest matching."""
        key = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )
        matching_manifest = {
            "design_hash": _make_design_hash(),
            "toolchain_hash": _make_toolchain_hash(),
        }
        non_matching_manifest = {
            "design_hash": _make_design_hash(),
            "toolchain_hash": _make_alternate_toolchain_hash(),
        }

        assert key.matches_manifest(matching_manifest)
        assert not key.matches_manifest(non_matching_manifest)

    def test_cache_key_invalid_design_hash(self) -> None:
        """Test validation rejects invalid design hash."""
        with pytest.raises(ValueError, match="design_hash"):
            CacheKey(design_hash="short", toolchain_hash=_make_toolchain_hash())

    def test_cache_key_invalid_toolchain_hash(self) -> None:
        """Test validation rejects invalid toolchain hash."""
        with pytest.raises(ValueError, match="toolchain_hash"):
            CacheKey(design_hash=_make_design_hash(), toolchain_hash="short")


# -----------------------------------------------------------------------------
# StructuralCache tests - Level 1: ResolvedDesign
# -----------------------------------------------------------------------------


class TestStructuralCacheResolvedDesign:
    """Tests for ResolvedDesign level caching."""

    def test_put_and_get_resolved_design(self, tmp_path: Path) -> None:
        """Test storing and retrieving resolved design."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )
        data = _sample_resolved_design_data()

        # Store
        path = cache.put_resolved_design(key, data)
        assert path.exists()
        assert cache.has_resolved_design(key)

        # Retrieve
        retrieved = cache.get_resolved_design(key)
        assert retrieved == data

    def test_resolved_design_not_found(self, tmp_path: Path) -> None:
        """Test error when resolved design not cached."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )

        assert not cache.has_resolved_design(key)
        with pytest.raises(CacheError, match="not found"):
            cache.get_resolved_design(key)


# -----------------------------------------------------------------------------
# StructuralCache tests - Level 2: KiCad Board
# -----------------------------------------------------------------------------


class TestStructuralCacheBoard:
    """Tests for KiCad board level caching."""

    def test_put_and_get_board(self, tmp_path: Path) -> None:
        """Test storing and retrieving board file."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )

        # Create sample board
        board_path = tmp_path / "work" / "coupon.kicad_pcb"
        _create_sample_board_file(board_path)

        # Store
        cached_path = cache.put_board(key, board_path)
        assert cached_path.exists()
        assert cache.has_board(key)

        # Retrieve
        retrieved_path = cache.get_board_path(key)
        assert retrieved_path == cached_path

    def test_restore_board(self, tmp_path: Path) -> None:
        """Test restoring board to destination."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )

        # Create and cache board
        board_path = tmp_path / "work" / "coupon.kicad_pcb"
        _create_sample_board_file(board_path)
        cache.put_board(key, board_path)

        # Restore to new location
        dest_path = tmp_path / "restored" / "coupon.kicad_pcb"
        cache.restore_board(key, dest_path)
        assert dest_path.exists()
        assert dest_path.read_text() == board_path.read_text()

    def test_board_not_found(self, tmp_path: Path) -> None:
        """Test error when board not cached."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )

        assert not cache.has_board(key)
        with pytest.raises(CacheError, match="not found"):
            cache.get_board_path(key)


# -----------------------------------------------------------------------------
# StructuralCache tests - Level 3: Fab Outputs
# -----------------------------------------------------------------------------


class TestStructuralCacheFab:
    """Tests for fab output level caching."""

    def test_put_and_get_fab(self, tmp_path: Path) -> None:
        """Test storing and retrieving fab outputs."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )

        # Create sample fab outputs
        fab_dir = tmp_path / "work" / "fab"
        _create_sample_fab_dir(fab_dir)

        # Store
        cached_path = cache.put_fab(key, fab_dir)
        assert cached_path.exists()
        assert cache.has_fab(key)

        # Retrieve
        retrieved_path = cache.get_fab_path(key)
        assert retrieved_path == cached_path

    def test_restore_fab(self, tmp_path: Path) -> None:
        """Test restoring fab outputs to destination."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )

        # Create and cache fab outputs
        fab_dir = tmp_path / "work" / "fab"
        _create_sample_fab_dir(fab_dir)
        cache.put_fab(key, fab_dir)

        # Restore to new location
        dest_dir = tmp_path / "restored" / "fab"
        cache.restore_fab(key, dest_dir)
        assert dest_dir.exists()
        assert (dest_dir / "gerbers" / "F.Cu.gbr").exists()
        assert (dest_dir / "drill" / "drill.drl").exists()

    def test_fab_not_found(self, tmp_path: Path) -> None:
        """Test error when fab not cached."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )

        assert not cache.has_fab(key)
        with pytest.raises(CacheError, match="not found"):
            cache.get_fab_path(key)


# -----------------------------------------------------------------------------
# StructuralCache tests - Cache Management
# -----------------------------------------------------------------------------


class TestStructuralCacheManagement:
    """Tests for cache management operations."""

    def test_invalidate_key(self, tmp_path: Path) -> None:
        """Test invalidating all levels for a key."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )

        # Populate all levels
        cache.put_resolved_design(key, _sample_resolved_design_data())

        board_path = tmp_path / "work" / "coupon.kicad_pcb"
        _create_sample_board_file(board_path)
        cache.put_board(key, board_path)

        fab_dir = tmp_path / "work" / "fab"
        _create_sample_fab_dir(fab_dir)
        cache.put_fab(key, fab_dir)

        # Verify all cached
        assert cache.has_resolved_design(key)
        assert cache.has_board(key)
        assert cache.has_fab(key)

        # Invalidate
        count = cache.invalidate(key)
        assert count == 3

        # Verify all gone
        assert not cache.has_resolved_design(key)
        assert not cache.has_board(key)
        assert not cache.has_fab(key)

    def test_clear_cache(self, tmp_path: Path) -> None:
        """Test clearing all cache entries."""
        cache = StructuralCache(root=tmp_path / "cache")

        # Create multiple cache entries
        for i in range(3):
            key = CacheKey(
                design_hash=f"{i:064d}",
                toolchain_hash=_make_toolchain_hash(),
            )
            cache.put_resolved_design(key, _sample_resolved_design_data())

        # Clear
        count = cache.clear()
        assert count == 3

        # Verify stats
        stats = cache.stats()
        assert stats.resolved_design_entries == 0

    def test_cache_stats(self, tmp_path: Path) -> None:
        """Test cache statistics."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )

        # Initial stats
        stats = cache.stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.resolved_design_entries == 0

        # Add entry
        cache.put_resolved_design(key, _sample_resolved_design_data())

        # Stats after add
        stats = cache.stats()
        assert stats.resolved_design_entries == 1
        assert stats.total_size_bytes > 0

        # Hit
        cache.get_resolved_design(key)
        stats = cache.stats()
        assert stats.hits == 1

        # Miss
        miss_key = CacheKey(
            design_hash=_make_alternate_toolchain_hash(),  # Different hash
            toolchain_hash=_make_toolchain_hash(),
        )
        try:
            cache.get_resolved_design(miss_key)
        except CacheError:
            pass
        stats = cache.stats()
        assert stats.misses == 1

    def test_verify_integrity(self, tmp_path: Path) -> None:
        """Test integrity verification."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )

        # Only resolved design cached
        cache.put_resolved_design(key, _sample_resolved_design_data())

        results = cache.verify_integrity(key)
        assert results[StructuralCache.LEVEL_RESOLVED_DESIGN] is True
        assert results[StructuralCache.LEVEL_BOARD] is False
        assert results[StructuralCache.LEVEL_FAB] is False


# -----------------------------------------------------------------------------
# Cache Invalidation tests (REQ-M1-020)
# -----------------------------------------------------------------------------


class TestCacheInvalidation:
    """Tests for cache invalidation on toolchain changes."""

    def test_should_invalidate_on_toolchain_change(self) -> None:
        """Test invalidation flag when toolchain changes."""
        cached_hash = _make_toolchain_hash()
        current_hash = _make_alternate_toolchain_hash()

        assert should_invalidate_cache(cached_hash, current_hash)

    def test_should_not_invalidate_same_toolchain(self) -> None:
        """Test no invalidation when toolchain is same."""
        hash_value = _make_toolchain_hash()

        assert not should_invalidate_cache(hash_value, hash_value)

    def test_toolchain_hash_from_config(self) -> None:
        """Test toolchain hash computation."""
        hash1 = toolchain_hash_from_config(
            kicad_version="9.0.7",
            docker_image="kicad/kicad:9.0.7@sha256:abc",
            mode="docker",
        )
        hash2 = toolchain_hash_from_config(
            kicad_version="9.0.7",
            docker_image="kicad/kicad:9.0.7@sha256:def",  # Different digest
            mode="docker",
        )

        assert len(hash1) == 64
        assert len(hash2) == 64
        assert hash1 != hash2  # Different configs should produce different hashes

    def test_toolchain_hash_with_lock_file(self) -> None:
        """Test toolchain hash includes lock file hash when provided."""
        hash_without = toolchain_hash_from_config(
            kicad_version="9.0.7",
            docker_image="kicad/kicad:9.0.7@sha256:abc",
            mode="docker",
        )
        hash_with = toolchain_hash_from_config(
            kicad_version="9.0.7",
            docker_image="kicad/kicad:9.0.7@sha256:abc",
            mode="docker",
            lock_file_hash="d" * 64,
        )

        assert hash_without != hash_with

    def test_cache_invalidation_scenario(self, tmp_path: Path) -> None:
        """Test full cache invalidation scenario (REQ-M1-020)."""
        cache = StructuralCache(root=tmp_path / "cache")

        # Initial build
        key1 = CacheKey(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )
        cache.put_resolved_design(key1, _sample_resolved_design_data())
        assert cache.has_resolved_design(key1)

        # Same design, different toolchain (e.g., KiCad version upgrade)
        key2 = CacheKey(
            design_hash=_make_design_hash(),  # Same design
            toolchain_hash=_make_alternate_toolchain_hash(),  # Different toolchain
        )

        # Should NOT find cached entry for new toolchain
        assert not cache.has_resolved_design(key2)

        # Verify invalidation logic
        assert should_invalidate_cache(key1.toolchain_hash, key2.toolchain_hash)


# -----------------------------------------------------------------------------
# Helper function tests
# -----------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_compute_cache_key(self) -> None:
        """Test compute_cache_key helper."""
        key = compute_cache_key(
            design_hash=_make_design_hash(),
            toolchain_hash=_make_toolchain_hash(),
        )
        assert isinstance(key, CacheKey)
        assert key.design_hash == _make_design_hash()
        assert key.toolchain_hash == _make_toolchain_hash()
