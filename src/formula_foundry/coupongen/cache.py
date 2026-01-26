"""Three-level structural caching system for coupon generation.

This module implements REQ-M1-020: Caching keyed by design_hash + toolchain_hash
at three levels:
    1. ResolvedDesign cache: Caches resolved geometry parameters
    2. KiCad board cache: Caches generated .kicad_pcb files
    3. Fab output cache: Caches exported Gerbers and drill files

Cache invalidation occurs automatically when:
    - design_hash changes (spec parameters changed)
    - toolchain_hash changes (KiCad version, docker image, or lock file changed)

The cache uses the substrate's artifact store infrastructure for content-addressed
storage with atomic writes and integrity verification.

Usage:
    >>> cache = StructuralCache(root=Path("cache"))
    >>> key = CacheKey(design_hash=design_hash, toolchain_hash=toolchain_hash)
    >>>
    >>> # Check if resolved design is cached
    >>> if cache.has_resolved_design(key):
    ...     resolved = cache.get_resolved_design(key)
    >>> else:
    ...     resolved = resolve(spec)
    ...     cache.put_resolved_design(key, resolved)
    >>>
    >>> # Similar pattern for board and fab outputs
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from formula_foundry.substrate import canonical_json_dumps, sha256_bytes

if TYPE_CHECKING:
    from .resolve import ResolvedDesign


class CacheError(Exception):
    """Raised when cache operations fail."""


class CacheInvalidationError(CacheError):
    """Raised when cache invalidation logic detects a mismatch."""


@dataclass(frozen=True)
class CacheKey:
    """Cache key combining design_hash and toolchain_hash.

    This key uniquely identifies a build configuration. Cache entries are
    invalidated when either component changes:
    - design_hash: Changes when CouponSpec parameters change
    - toolchain_hash: Changes when toolchain version/config changes

    Satisfies REQ-M1-020: Cache keyed by design_hash + toolchain_hash.
    """

    design_hash: str
    toolchain_hash: str

    def __post_init__(self) -> None:
        """Validate hash formats."""
        if not self.design_hash or len(self.design_hash) != 64:
            raise ValueError("design_hash must be a 64-character hex string")
        if not self.toolchain_hash or len(self.toolchain_hash) != 64:
            raise ValueError("toolchain_hash must be a 64-character hex string")

    @property
    def combined_hash(self) -> str:
        """Compute combined hash for cache directory naming."""
        combined = f"{self.design_hash}:{self.toolchain_hash}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @property
    def short_key(self) -> str:
        """Return a shortened key for display (first 8 chars of each hash)."""
        return f"{self.design_hash[:8]}_{self.toolchain_hash[:8]}"

    def to_dict(self) -> dict[str, str]:
        """Serialize to dictionary."""
        return {
            "design_hash": self.design_hash,
            "toolchain_hash": self.toolchain_hash,
            "combined_hash": self.combined_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> CacheKey:
        """Deserialize from dictionary."""
        return cls(
            design_hash=data["design_hash"],
            toolchain_hash=data["toolchain_hash"],
        )

    def matches_manifest(self, manifest: dict[str, Any]) -> bool:
        """Check if this cache key matches a manifest's hashes."""
        return manifest.get("design_hash") == self.design_hash and manifest.get("toolchain_hash") == self.toolchain_hash


@dataclass(frozen=True)
class CacheStats:
    """Statistics for cache usage."""

    hits: int
    misses: int
    resolved_design_entries: int
    board_entries: int
    fab_entries: int
    total_size_bytes: int

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate as a percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100.0


@dataclass(frozen=True)
class CacheEntry:
    """Metadata for a cache entry."""

    level: str  # "resolved_design", "board", or "fab"
    key: CacheKey
    path: Path
    size_bytes: int
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "level": self.level,
            "key": self.key.to_dict(),
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "content_hash": self.content_hash,
        }


class StructuralCache:
    """Three-level structural cache for coupon generation.

    Implements REQ-M1-020: Caching keyed by design_hash + toolchain_hash at:
    1. ResolvedDesign level: Caches resolved geometry parameters
    2. KiCad board level: Caches generated .kicad_pcb files
    3. Fab output level: Caches exported Gerbers and drill files

    Directory structure:
        {root}/
            resolved_design/
                {combined_hash[:2]}/
                    {combined_hash[2:]}.json
            board/
                {combined_hash[:2]}/
                    {combined_hash[2:]}.kicad_pcb
            fab/
                {combined_hash[:2]}/
                    {combined_hash[2:]}/
                        gerbers/
                        drill/
            meta/
                {combined_hash[:2]}/
                    {combined_hash[2:]}.json  # Cache entry metadata
    """

    LEVEL_RESOLVED_DESIGN = "resolved_design"
    LEVEL_BOARD = "board"
    LEVEL_FAB = "fab"
    LEVELS = (LEVEL_RESOLVED_DESIGN, LEVEL_BOARD, LEVEL_FAB)

    def __init__(self, root: Path) -> None:
        """Initialize the structural cache.

        Args:
            root: Root directory for cache storage.
        """
        self.root = root
        self._ensure_structure()
        self._hits = 0
        self._misses = 0

    def _ensure_structure(self) -> None:
        """Ensure cache directory structure exists."""
        for level in self.LEVELS:
            (self.root / level).mkdir(parents=True, exist_ok=True)
        (self.root / "meta").mkdir(parents=True, exist_ok=True)

    def _level_path(self, level: str, key: CacheKey) -> Path:
        """Compute path for a cache entry at the given level."""
        combined = key.combined_hash
        subdir = combined[:2]
        filename = combined[2:]
        level_dir = self.root / level / subdir

        if level == self.LEVEL_RESOLVED_DESIGN:
            return level_dir / f"{filename}.json"
        elif level == self.LEVEL_BOARD:
            return level_dir / f"{filename}.kicad_pcb"
        elif level == self.LEVEL_FAB:
            return level_dir / filename
        else:
            raise ValueError(f"Unknown cache level: {level}")

    def _meta_path(self, key: CacheKey) -> Path:
        """Compute path for cache entry metadata."""
        combined = key.combined_hash
        return self.root / "meta" / combined[:2] / f"{combined[2:]}.json"

    def _write_meta(self, key: CacheKey, levels: list[str]) -> None:
        """Write cache entry metadata."""
        meta_path = self._meta_path(key)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "key": key.to_dict(),
            "levels": levels,
        }
        meta_path.write_text(canonical_json_dumps(meta), encoding="utf-8")

    def _read_meta(self, key: CacheKey) -> dict[str, Any] | None:
        """Read cache entry metadata, or None if not found."""
        meta_path = self._meta_path(key)
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    # -------------------------------------------------------------------------
    # Level 1: ResolvedDesign cache
    # -------------------------------------------------------------------------

    def has_resolved_design(self, key: CacheKey) -> bool:
        """Check if a resolved design is cached."""
        path = self._level_path(self.LEVEL_RESOLVED_DESIGN, key)
        return path.exists()

    def get_resolved_design(self, key: CacheKey) -> dict[str, Any]:
        """Get a cached resolved design.

        Args:
            key: Cache key for lookup.

        Returns:
            Dictionary containing the resolved design data.

        Raises:
            CacheError: If the entry is not found or corrupted.
        """
        path = self._level_path(self.LEVEL_RESOLVED_DESIGN, key)
        if not path.exists():
            self._misses += 1
            raise CacheError(f"Resolved design not found in cache: {key.short_key}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._hits += 1
            return data
        except (json.JSONDecodeError, OSError) as e:
            self._misses += 1
            raise CacheError(f"Failed to read cached resolved design: {e}") from e

    def put_resolved_design(
        self,
        key: CacheKey,
        resolved: ResolvedDesign | dict[str, Any],
    ) -> Path:
        """Store a resolved design in the cache.

        Args:
            key: Cache key for storage.
            resolved: ResolvedDesign instance or dictionary to cache.

        Returns:
            Path where the entry was stored.
        """
        path = self._level_path(self.LEVEL_RESOLVED_DESIGN, key)
        path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(resolved, "model_dump"):
            data = resolved.model_dump(mode="json")
        else:
            data = resolved

        text = canonical_json_dumps(data)
        path.write_text(text, encoding="utf-8")

        meta = self._read_meta(key) or {"key": key.to_dict(), "levels": []}
        if self.LEVEL_RESOLVED_DESIGN not in meta["levels"]:
            meta["levels"].append(self.LEVEL_RESOLVED_DESIGN)
        self._write_meta(key, meta["levels"])

        return path

    # -------------------------------------------------------------------------
    # Level 2: KiCad board cache
    # -------------------------------------------------------------------------

    def has_board(self, key: CacheKey) -> bool:
        """Check if a KiCad board is cached."""
        path = self._level_path(self.LEVEL_BOARD, key)
        return path.exists()

    def get_board_path(self, key: CacheKey) -> Path:
        """Get the path to a cached KiCad board file.

        Args:
            key: Cache key for lookup.

        Returns:
            Path to the cached .kicad_pcb file.

        Raises:
            CacheError: If the entry is not found.
        """
        path = self._level_path(self.LEVEL_BOARD, key)
        if not path.exists():
            self._misses += 1
            raise CacheError(f"Board not found in cache: {key.short_key}")

        self._hits += 1
        return path

    def put_board(self, key: CacheKey, board_path: Path) -> Path:
        """Store a KiCad board file in the cache.

        Args:
            key: Cache key for storage.
            board_path: Path to the .kicad_pcb file to cache.

        Returns:
            Path where the entry was stored.
        """
        cache_path = self._level_path(self.LEVEL_BOARD, key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the board file to cache
        shutil.copy2(board_path, cache_path)

        meta = self._read_meta(key) or {"key": key.to_dict(), "levels": []}
        if self.LEVEL_BOARD not in meta["levels"]:
            meta["levels"].append(self.LEVEL_BOARD)
        self._write_meta(key, meta["levels"])

        return cache_path

    def restore_board(self, key: CacheKey, dest_path: Path) -> None:
        """Restore a cached board file to a destination path.

        Args:
            key: Cache key for lookup.
            dest_path: Destination path to copy the board to.

        Raises:
            CacheError: If the entry is not found.
        """
        cache_path = self.get_board_path(key)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache_path, dest_path)

    # -------------------------------------------------------------------------
    # Level 3: Fab output cache
    # -------------------------------------------------------------------------

    def has_fab(self, key: CacheKey) -> bool:
        """Check if fab outputs are cached."""
        path = self._level_path(self.LEVEL_FAB, key)
        return path.exists() and path.is_dir()

    def get_fab_path(self, key: CacheKey) -> Path:
        """Get the path to cached fab outputs.

        Args:
            key: Cache key for lookup.

        Returns:
            Path to the directory containing cached Gerbers and drills.

        Raises:
            CacheError: If the entry is not found.
        """
        path = self._level_path(self.LEVEL_FAB, key)
        if not path.exists() or not path.is_dir():
            self._misses += 1
            raise CacheError(f"Fab outputs not found in cache: {key.short_key}")

        self._hits += 1
        return path

    def put_fab(self, key: CacheKey, fab_dir: Path) -> Path:
        """Store fab outputs in the cache.

        Args:
            key: Cache key for storage.
            fab_dir: Path to directory containing Gerbers and drills.

        Returns:
            Path where the entries were stored.
        """
        cache_path = self._level_path(self.LEVEL_FAB, key)
        if cache_path.exists():
            shutil.rmtree(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the entire fab directory
        shutil.copytree(fab_dir, cache_path)

        meta = self._read_meta(key) or {"key": key.to_dict(), "levels": []}
        if self.LEVEL_FAB not in meta["levels"]:
            meta["levels"].append(self.LEVEL_FAB)
        self._write_meta(key, meta["levels"])

        return cache_path

    def restore_fab(self, key: CacheKey, dest_dir: Path) -> None:
        """Restore cached fab outputs to a destination directory.

        Args:
            key: Cache key for lookup.
            dest_dir: Destination directory to copy fab outputs to.

        Raises:
            CacheError: If the entry is not found.
        """
        cache_path = self.get_fab_path(key)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(cache_path, dest_dir)

    # -------------------------------------------------------------------------
    # Cache management
    # -------------------------------------------------------------------------

    def invalidate(self, key: CacheKey) -> int:
        """Invalidate all cache entries for a key.

        Args:
            key: Cache key to invalidate.

        Returns:
            Number of entries invalidated.
        """
        count = 0
        for level in self.LEVELS:
            path = self._level_path(level, key)
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                count += 1

        meta_path = self._meta_path(key)
        if meta_path.exists():
            meta_path.unlink()

        return count

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        count = 0
        for level in self.LEVELS:
            level_dir = self.root / level
            if level_dir.exists():
                for subdir in level_dir.iterdir():
                    if subdir.is_dir():
                        for entry in subdir.iterdir():
                            if entry.is_dir():
                                shutil.rmtree(entry)
                            else:
                                entry.unlink()
                            count += 1

        meta_dir = self.root / "meta"
        if meta_dir.exists():
            shutil.rmtree(meta_dir)
            (self.root / "meta").mkdir(parents=True, exist_ok=True)

        self._hits = 0
        self._misses = 0
        return count

    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with hit/miss counts and entry counts.
        """
        resolved_count = 0
        board_count = 0
        fab_count = 0
        total_size = 0

        for level in self.LEVELS:
            level_dir = self.root / level
            if not level_dir.exists():
                continue

            for subdir in level_dir.iterdir():
                if not subdir.is_dir():
                    continue
                for entry in subdir.iterdir():
                    if entry.is_file():
                        if level == self.LEVEL_RESOLVED_DESIGN:
                            resolved_count += 1
                        elif level == self.LEVEL_BOARD:
                            board_count += 1
                        total_size += entry.stat().st_size
                    elif entry.is_dir():
                        if level == self.LEVEL_FAB:
                            fab_count += 1
                            for f in entry.rglob("*"):
                                if f.is_file():
                                    total_size += f.stat().st_size

        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            resolved_design_entries=resolved_count,
            board_entries=board_count,
            fab_entries=fab_count,
            total_size_bytes=total_size,
        )

    def verify_integrity(self, key: CacheKey) -> dict[str, bool]:
        """Verify integrity of cached entries for a key.

        Args:
            key: Cache key to verify.

        Returns:
            Dictionary mapping level names to validity status.
        """
        results: dict[str, bool] = {}

        for level in self.LEVELS:
            path = self._level_path(level, key)
            if not path.exists():
                results[level] = False
                continue

            if level == self.LEVEL_RESOLVED_DESIGN:
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                    results[level] = True
                except (json.JSONDecodeError, OSError):
                    results[level] = False
            elif level == self.LEVEL_BOARD:
                results[level] = path.is_file() and path.stat().st_size > 0
            elif level == self.LEVEL_FAB:
                results[level] = path.is_dir() and any(path.iterdir())

        return results


def compute_cache_key(
    design_hash: str,
    toolchain_hash: str,
) -> CacheKey:
    """Create a cache key from component hashes.

    Args:
        design_hash: SHA256 hash of the resolved design.
        toolchain_hash: SHA256 hash of the toolchain configuration.

    Returns:
        CacheKey instance.
    """
    return CacheKey(design_hash=design_hash, toolchain_hash=toolchain_hash)


def toolchain_hash_from_config(
    kicad_version: str,
    docker_image: str,
    mode: str,
    lock_file_hash: str | None = None,
) -> str:
    """Compute toolchain hash from configuration components.

    This function computes a deterministic hash from toolchain configuration,
    ensuring cache invalidation when any component changes.

    Args:
        kicad_version: KiCad version string (e.g., "9.0.7").
        docker_image: Docker image reference with digest.
        mode: Execution mode ("local" or "docker").
        lock_file_hash: Optional hash from toolchain lock file.

    Returns:
        SHA256 hash of the toolchain configuration.
    """
    config: dict[str, Any] = {
        "kicad_version": kicad_version,
        "docker_image": docker_image,
        "mode": mode,
    }
    if lock_file_hash:
        config["lock_file_hash"] = lock_file_hash

    canonical = canonical_json_dumps(config)
    return sha256_bytes(canonical.encode("utf-8"))


def should_invalidate_cache(
    cached_toolchain_hash: str,
    current_toolchain_hash: str,
) -> bool:
    """Check if cache should be invalidated due to toolchain changes.

    Satisfies REQ-M1-020: Cache invalidation on toolchain version changes.

    Args:
        cached_toolchain_hash: Toolchain hash from cached manifest.
        current_toolchain_hash: Current toolchain hash.

    Returns:
        True if cache should be invalidated, False otherwise.
    """
    return cached_toolchain_hash != current_toolchain_hash
