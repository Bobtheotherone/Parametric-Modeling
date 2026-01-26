"""Additional unit tests for cache module edge cases.

Supplements test_cache.py with coverage for:
- CacheStats.hit_rate property edge cases
- CacheEntry.to_dict serialization
- CacheKey validation edge cases
- StructuralCache integrity verification edge cases
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.coupongen.cache import (
    CacheEntry,
    CacheError,
    CacheKey,
    CacheStats,
    StructuralCache,
)

# -----------------------------------------------------------------------------
# CacheStats edge case tests
# -----------------------------------------------------------------------------


class TestCacheStatsHitRate:
    """Tests for CacheStats.hit_rate property edge cases."""

    def test_hit_rate_zero_when_no_operations(self) -> None:
        """Test hit_rate returns 0.0 when no hits or misses."""
        stats = CacheStats(
            hits=0,
            misses=0,
            resolved_design_entries=0,
            board_entries=0,
            fab_entries=0,
            total_size_bytes=0,
        )
        assert stats.hit_rate == 0.0

    def test_hit_rate_100_percent_all_hits(self) -> None:
        """Test hit_rate returns 100.0 when all operations are hits."""
        stats = CacheStats(
            hits=10,
            misses=0,
            resolved_design_entries=5,
            board_entries=3,
            fab_entries=2,
            total_size_bytes=1000,
        )
        assert stats.hit_rate == 100.0

    def test_hit_rate_0_percent_all_misses(self) -> None:
        """Test hit_rate returns 0.0 when all operations are misses."""
        stats = CacheStats(
            hits=0,
            misses=10,
            resolved_design_entries=0,
            board_entries=0,
            fab_entries=0,
            total_size_bytes=0,
        )
        assert stats.hit_rate == 0.0

    def test_hit_rate_50_percent(self) -> None:
        """Test hit_rate calculates correctly for mixed hits/misses."""
        stats = CacheStats(
            hits=5,
            misses=5,
            resolved_design_entries=3,
            board_entries=2,
            fab_entries=0,
            total_size_bytes=500,
        )
        assert stats.hit_rate == 50.0

    def test_hit_rate_fractional(self) -> None:
        """Test hit_rate handles fractional percentages."""
        stats = CacheStats(
            hits=1,
            misses=2,
            resolved_design_entries=1,
            board_entries=0,
            fab_entries=0,
            total_size_bytes=100,
        )
        assert abs(stats.hit_rate - 33.333333) < 0.001


# -----------------------------------------------------------------------------
# CacheEntry tests
# -----------------------------------------------------------------------------


def _make_valid_key() -> CacheKey:
    """Create a valid CacheKey for testing."""
    return CacheKey(design_hash="a" * 64, toolchain_hash="b" * 64)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_to_dict_serialization(self) -> None:
        """Test CacheEntry.to_dict returns correct dictionary structure."""
        key = _make_valid_key()
        entry = CacheEntry(
            level="resolved_design",
            key=key,
            path=Path("/cache/resolved_design/ab/cd.json"),
            size_bytes=1024,
            content_hash="c" * 64,
        )

        result = entry.to_dict()

        assert result["level"] == "resolved_design"
        assert result["path"] == "/cache/resolved_design/ab/cd.json"
        assert result["size_bytes"] == 1024
        assert result["content_hash"] == "c" * 64
        assert "key" in result
        assert result["key"]["design_hash"] == "a" * 64
        assert result["key"]["toolchain_hash"] == "b" * 64

    def test_to_dict_json_serializable(self) -> None:
        """Test CacheEntry.to_dict is JSON serializable."""
        key = _make_valid_key()
        entry = CacheEntry(
            level="board",
            key=key,
            path=Path("/cache/board/xy/file.kicad_pcb"),
            size_bytes=2048,
            content_hash="d" * 64,
        )

        result = entry.to_dict()
        # Should not raise
        json_str = json.dumps(result)
        parsed = json.loads(json_str)

        assert parsed["level"] == "board"
        assert parsed["size_bytes"] == 2048

    def test_to_dict_all_levels(self) -> None:
        """Test CacheEntry.to_dict works for all cache levels."""
        key = _make_valid_key()

        for level in ["resolved_design", "board", "fab"]:
            entry = CacheEntry(
                level=level,
                key=key,
                path=Path(f"/cache/{level}/test"),
                size_bytes=512,
                content_hash="e" * 64,
            )
            result = entry.to_dict()
            assert result["level"] == level


# -----------------------------------------------------------------------------
# CacheKey validation edge cases
# -----------------------------------------------------------------------------


class TestCacheKeyValidation:
    """Additional tests for CacheKey validation edge cases."""

    def test_empty_design_hash_raises(self) -> None:
        """Test that empty design_hash raises ValueError."""
        with pytest.raises(ValueError, match="design_hash"):
            CacheKey(design_hash="", toolchain_hash="b" * 64)

    def test_empty_toolchain_hash_raises(self) -> None:
        """Test that empty toolchain_hash raises ValueError."""
        with pytest.raises(ValueError, match="toolchain_hash"):
            CacheKey(design_hash="a" * 64, toolchain_hash="")

    def test_63_char_design_hash_raises(self) -> None:
        """Test that 63-character design_hash raises ValueError."""
        with pytest.raises(ValueError, match="design_hash"):
            CacheKey(design_hash="a" * 63, toolchain_hash="b" * 64)

    def test_65_char_toolchain_hash_raises(self) -> None:
        """Test that 65-character toolchain_hash raises ValueError."""
        with pytest.raises(ValueError, match="toolchain_hash"):
            CacheKey(design_hash="a" * 64, toolchain_hash="b" * 65)

    def test_combined_hash_deterministic(self) -> None:
        """Test combined_hash is deterministic across multiple calls."""
        key = CacheKey(design_hash="a" * 64, toolchain_hash="b" * 64)
        hash1 = key.combined_hash
        hash2 = key.combined_hash
        hash3 = key.combined_hash
        assert hash1 == hash2 == hash3

    def test_to_dict_contains_combined_hash(self) -> None:
        """Test to_dict includes the combined_hash."""
        key = CacheKey(design_hash="a" * 64, toolchain_hash="b" * 64)
        result = key.to_dict()
        assert "combined_hash" in result
        assert len(result["combined_hash"]) == 64


# -----------------------------------------------------------------------------
# StructuralCache integrity verification edge cases
# -----------------------------------------------------------------------------


class TestStructuralCacheIntegrity:
    """Additional tests for StructuralCache integrity verification."""

    def test_verify_integrity_empty_cache(self, tmp_path: Path) -> None:
        """Test verify_integrity on empty cache returns all False."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(design_hash="a" * 64, toolchain_hash="b" * 64)

        results = cache.verify_integrity(key)

        assert results[StructuralCache.LEVEL_RESOLVED_DESIGN] is False
        assert results[StructuralCache.LEVEL_BOARD] is False
        assert results[StructuralCache.LEVEL_FAB] is False

    def test_verify_integrity_partial_cache(self, tmp_path: Path) -> None:
        """Test verify_integrity with only some levels populated."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(design_hash="a" * 64, toolchain_hash="b" * 64)

        # Only add resolved design
        cache.put_resolved_design(key, {"test": "data"})

        results = cache.verify_integrity(key)

        assert results[StructuralCache.LEVEL_RESOLVED_DESIGN] is True
        assert results[StructuralCache.LEVEL_BOARD] is False
        assert results[StructuralCache.LEVEL_FAB] is False

    def test_verify_integrity_corrupted_json(self, tmp_path: Path) -> None:
        """Test verify_integrity detects corrupted JSON."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(design_hash="a" * 64, toolchain_hash="b" * 64)

        # Add valid resolved design first
        path = cache.put_resolved_design(key, {"test": "data"})

        # Corrupt the file
        path.write_text("not valid json {{{", encoding="utf-8")

        results = cache.verify_integrity(key)
        assert results[StructuralCache.LEVEL_RESOLVED_DESIGN] is False

    def test_verify_integrity_empty_board_file(self, tmp_path: Path) -> None:
        """Test verify_integrity detects empty board file."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(design_hash="a" * 64, toolchain_hash="b" * 64)

        # Create board file path and write empty content
        board_path = cache._level_path(StructuralCache.LEVEL_BOARD, key)
        board_path.parent.mkdir(parents=True, exist_ok=True)
        board_path.write_text("", encoding="utf-8")

        results = cache.verify_integrity(key)
        assert results[StructuralCache.LEVEL_BOARD] is False

    def test_verify_integrity_empty_fab_dir(self, tmp_path: Path) -> None:
        """Test verify_integrity detects empty fab directory."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(design_hash="a" * 64, toolchain_hash="b" * 64)

        # Create empty fab directory
        fab_path = cache._level_path(StructuralCache.LEVEL_FAB, key)
        fab_path.mkdir(parents=True, exist_ok=True)

        results = cache.verify_integrity(key)
        assert results[StructuralCache.LEVEL_FAB] is False


# -----------------------------------------------------------------------------
# StructuralCache stats edge cases
# -----------------------------------------------------------------------------


class TestStructuralCacheStatsEdgeCases:
    """Additional tests for StructuralCache stats edge cases."""

    def test_stats_after_clear_resets_counters(self, tmp_path: Path) -> None:
        """Test that clear() resets hit/miss counters."""
        cache = StructuralCache(root=tmp_path / "cache")
        key = CacheKey(design_hash="a" * 64, toolchain_hash="b" * 64)

        # Generate some hits and misses
        cache.put_resolved_design(key, {"test": "data"})
        cache.get_resolved_design(key)  # Hit

        try:
            cache.get_resolved_design(CacheKey(design_hash="c" * 64, toolchain_hash="d" * 64))
        except CacheError:
            pass  # Miss

        stats_before = cache.stats()
        assert stats_before.hits == 1
        assert stats_before.misses == 1

        cache.clear()

        stats_after = cache.stats()
        assert stats_after.hits == 0
        assert stats_after.misses == 0

    def test_stats_counts_all_entry_types(self, tmp_path: Path) -> None:
        """Test stats counts entries across all cache levels."""
        cache = StructuralCache(root=tmp_path / "cache")

        # Add entries at different levels with different keys
        for i in range(3):
            key = CacheKey(design_hash=f"{i:064d}", toolchain_hash="b" * 64)
            cache.put_resolved_design(key, {"index": i})

        for i in range(2):
            key = CacheKey(design_hash=f"{i:064d}", toolchain_hash="c" * 64)
            board_path = tmp_path / f"board_{i}.kicad_pcb"
            board_path.write_text("(kicad_pcb)", encoding="utf-8")
            cache.put_board(key, board_path)

        stats = cache.stats()
        assert stats.resolved_design_entries == 3
        assert stats.board_entries == 2
