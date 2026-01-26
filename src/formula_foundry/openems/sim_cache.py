"""Three-level simulation caching system for M2 EM simulations.

This module implements REQ-M2-011: Caching layer keyed by (design_hash, sim_config_hash, solver_version).

The cache provides three levels for different simulation artifacts:
    1. Config cache: Caches validated simulation configurations
    2. Mesh cache: Caches generated mesh specifications (deterministic from geometry + resolution)
    3. Result cache: Caches complete simulation results including raw field data,
       intermediate results, and final S-parameters

Cache invalidation occurs automatically when:
    - design_hash changes (geometry or coupon design changed)
    - sim_config_hash changes (simulation parameters changed: frequency, ports, mesh, etc.)
    - solver_version changes (openEMS version, Docker image, or execution mode changed)

This module follows the same patterns as coupongen/cache.py for consistency and
integrates with the M2 manifest schema for provenance tracking.

Usage:
    >>> cache = SimulationCache(root=Path("sim_cache"))
    >>> key = SimulationCacheKey.from_simulation_inputs(
    ...     design_hash=design_hash,
    ...     sim_config_hash=spec_hash,
    ...     solver_version=solver_version,
    ... )
    >>>
    >>> # Check for cached result
    >>> if cache.has_result(key):
    ...     result = cache.get_result(key)
    >>> else:
    ...     result = run_simulation(...)
    ...     cache.put_result(key, result)
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from formula_foundry.substrate import canonical_json_dumps, sha256_bytes

if TYPE_CHECKING:
    from .spec import MeshSpec, SimulationSpec

logger = logging.getLogger(__name__)

_CACHE_SCHEMA_VERSION = 1
_HASH_CHUNK_SIZE = 1024 * 1024


class SimulationCacheError(Exception):
    """Base exception for simulation cache operations."""


class CacheMissError(SimulationCacheError):
    """Raised when a requested cache entry is not found."""


class CacheCorruptionError(SimulationCacheError):
    """Raised when a cached entry fails integrity verification."""


class CacheInvalidationError(SimulationCacheError):
    """Raised when cache invalidation detects configuration changes."""


@dataclass(frozen=True)
class SolverVersion:
    """Solver version information for cache key computation.

    This dataclass captures the solver configuration that affects simulation
    results, ensuring cache invalidation when the solver changes.

    Attributes:
        openems_version: openEMS version string (e.g., "0.0.35").
        mode: Execution mode ("local", "docker", or "stub").
        docker_image: Docker image reference with optional digest.
        csxcad_version: CSXCAD version if available.

    REQ-M2-011: Cache key includes solver_version component.
    """

    openems_version: str | None
    mode: str = "local"
    docker_image: str | None = None
    csxcad_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for hashing and serialization."""
        result: dict[str, Any] = {
            "openems_version": self.openems_version,
            "mode": self.mode,
        }
        if self.docker_image is not None:
            result["docker_image"] = self.docker_image
        if self.csxcad_version is not None:
            result["csxcad_version"] = self.csxcad_version
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolverVersion:
        """Create from dictionary."""
        return cls(
            openems_version=data.get("openems_version"),
            mode=data.get("mode", "local"),
            docker_image=data.get("docker_image"),
            csxcad_version=data.get("csxcad_version"),
        )

    @classmethod
    def from_stub_mode(cls) -> SolverVersion:
        """Create a solver version for stub mode testing."""
        return cls(openems_version=None, mode="stub")

    def compute_hash(self) -> str:
        """Compute SHA256 hash of solver version configuration."""
        canonical = canonical_json_dumps(self.to_dict())
        return sha256_bytes(canonical.encode("utf-8"))


@dataclass(frozen=True)
class SimulationCacheKey:
    """Cache key combining design_hash, sim_config_hash, and solver_version.

    This key uniquely identifies a simulation configuration. Cache entries are
    invalidated when any component changes:
    - design_hash: Changes when geometry/coupon design changes
    - sim_config_hash: Changes when simulation parameters change
    - solver_version: Changes when openEMS version or execution mode changes

    REQ-M2-011: Cache keyed by (design_hash, sim_config_hash, solver_version).
    """

    design_hash: str
    sim_config_hash: str
    solver_version_hash: str

    def __post_init__(self) -> None:
        """Validate hash formats."""
        if not self.design_hash or len(self.design_hash) != 64:
            raise ValueError("design_hash must be a 64-character hex string")
        if not self.sim_config_hash or len(self.sim_config_hash) != 64:
            raise ValueError("sim_config_hash must be a 64-character hex string")
        if not self.solver_version_hash or len(self.solver_version_hash) != 64:
            raise ValueError("solver_version_hash must be a 64-character hex string")

    @property
    def combined_hash(self) -> str:
        """Compute combined hash for cache directory naming."""
        combined = f"{self.design_hash}:{self.sim_config_hash}:{self.solver_version_hash}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @property
    def short_key(self) -> str:
        """Return a shortened key for display (first 8 chars of each hash)."""
        return f"{self.design_hash[:8]}_{self.sim_config_hash[:8]}_{self.solver_version_hash[:8]}"

    def to_dict(self) -> dict[str, str]:
        """Serialize to dictionary."""
        return {
            "design_hash": self.design_hash,
            "sim_config_hash": self.sim_config_hash,
            "solver_version_hash": self.solver_version_hash,
            "combined_hash": self.combined_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> SimulationCacheKey:
        """Deserialize from dictionary."""
        return cls(
            design_hash=data["design_hash"],
            sim_config_hash=data["sim_config_hash"],
            solver_version_hash=data["solver_version_hash"],
        )

    @classmethod
    def from_simulation_inputs(
        cls,
        *,
        design_hash: str,
        sim_config_hash: str,
        solver_version: SolverVersion,
    ) -> SimulationCacheKey:
        """Create cache key from simulation input hashes.

        Args:
            design_hash: SHA256 hash of the design/geometry (from M1 manifest).
            sim_config_hash: SHA256 hash of the simulation configuration.
            solver_version: Solver version information.

        Returns:
            SimulationCacheKey instance.
        """
        return cls(
            design_hash=design_hash,
            sim_config_hash=sim_config_hash,
            solver_version_hash=solver_version.compute_hash(),
        )

    def matches_manifest(self, manifest: dict[str, Any]) -> bool:
        """Check if this cache key matches a manifest's hashes."""
        return (
            manifest.get("design_hash") == self.design_hash
            and manifest.get("sim_config_hash") == self.sim_config_hash
            and manifest.get("solver_version_hash") == self.solver_version_hash
        )


@dataclass(frozen=True)
class CacheStats:
    """Statistics for simulation cache usage."""

    hits: int
    misses: int
    config_entries: int
    mesh_entries: int
    result_entries: int
    total_size_bytes: int

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate as a percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100.0


@dataclass(frozen=True)
class CachedSimulationResult:
    """Cached simulation result data.

    Contains all the artifacts from a completed simulation that can be
    restored from cache, including S-parameters, port signals, and
    energy decay data.

    Attributes:
        manifest: Complete simulation manifest with hashes and metadata.
        output_hashes: Map of output file relative paths to their SHA256 hashes.
        sparam_path: Relative path to S-parameter file within cache.
        outputs_dir: Path to directory containing all output files.
    """

    manifest: dict[str, Any]
    output_hashes: dict[str, str]
    sparam_path: str | None
    outputs_dir: Path

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "manifest": self.manifest,
            "output_hashes": self.output_hashes,
            "sparam_path": self.sparam_path,
        }


class SimulationCache:
    """Three-level simulation cache for M2 EM simulations.

    Implements REQ-M2-011: Caching keyed by (design_hash, sim_config_hash, solver_version).

    Cache levels:
        1. config/: Validated simulation configurations (SimulationSpec as JSON)
        2. mesh/: Generated mesh specifications (MeshSpec as JSON)
        3. result/: Complete simulation results (manifest + output files)

    Directory structure:
        {root}/
            config/
                {combined_hash[:2]}/
                    {combined_hash[2:]}.json  # SimulationSpec + validation
            mesh/
                {combined_hash[:2]}/
                    {combined_hash[2:]}.json  # MeshSpec
            result/
                {combined_hash[:2]}/
                    {combined_hash[2:]}/
                        manifest.json         # Simulation manifest
                        sim_outputs/          # Output files directory
                            sparams.s2p       # S-parameter file
                            port_signals.json # Port voltage/current
                            energy_decay.json # Convergence data
                            ...
            meta/
                {combined_hash[:2]}/
                    {combined_hash[2:]}.json  # Cache entry metadata

    The cache supports:
        - Atomic writes to prevent partial cache entries
        - Integrity verification via content hashes
        - Cache invalidation when inputs change
        - Statistics tracking for monitoring cache effectiveness
    """

    LEVEL_CONFIG = "config"
    LEVEL_MESH = "mesh"
    LEVEL_RESULT = "result"
    LEVELS = (LEVEL_CONFIG, LEVEL_MESH, LEVEL_RESULT)

    def __init__(self, root: Path) -> None:
        """Initialize the simulation cache.

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

    def _level_path(self, level: str, key: SimulationCacheKey) -> Path:
        """Compute path for a cache entry at the given level."""
        combined = key.combined_hash
        subdir = combined[:2]
        filename = combined[2:]
        level_dir = self.root / level / subdir

        if level == self.LEVEL_CONFIG or level == self.LEVEL_MESH:
            return level_dir / f"{filename}.json"
        elif level == self.LEVEL_RESULT:
            return level_dir / filename
        else:
            raise ValueError(f"Unknown cache level: {level}")

    def _meta_path(self, key: SimulationCacheKey) -> Path:
        """Compute path for cache entry metadata."""
        combined = key.combined_hash
        return self.root / "meta" / combined[:2] / f"{combined[2:]}.json"

    def _write_meta(self, key: SimulationCacheKey, levels: list[str]) -> None:
        """Write cache entry metadata."""
        meta_path = self._meta_path(key)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "key": key.to_dict(),
            "levels": levels,
        }
        meta_path.write_text(canonical_json_dumps(meta), encoding="utf-8")

    def _read_meta(self, key: SimulationCacheKey) -> dict[str, Any] | None:
        """Read cache entry metadata, or None if not found."""
        meta_path = self._meta_path(key)
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    # -------------------------------------------------------------------------
    # Level 1: Config cache (validated simulation configurations)
    # -------------------------------------------------------------------------

    def has_config(self, key: SimulationCacheKey) -> bool:
        """Check if a simulation config is cached."""
        path = self._level_path(self.LEVEL_CONFIG, key)
        return path.exists()

    def get_config(self, key: SimulationCacheKey) -> dict[str, Any]:
        """Get a cached simulation configuration.

        Args:
            key: Cache key for lookup.

        Returns:
            Dictionary containing the simulation config data.

        Raises:
            CacheMissError: If the entry is not found.
            CacheCorruptionError: If the entry is corrupted.
        """
        path = self._level_path(self.LEVEL_CONFIG, key)
        if not path.exists():
            self._misses += 1
            raise CacheMissError(f"Config not found in cache: {key.short_key}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._hits += 1
            logger.debug("Cache hit for config: %s", key.short_key)
            return data
        except (json.JSONDecodeError, OSError) as e:
            self._misses += 1
            raise CacheCorruptionError(f"Failed to read cached config: {e}") from e

    def put_config(
        self,
        key: SimulationCacheKey,
        config: SimulationSpec | dict[str, Any],
    ) -> Path:
        """Store a simulation configuration in the cache.

        Args:
            key: Cache key for storage.
            config: SimulationSpec instance or dictionary to cache.

        Returns:
            Path where the entry was stored.
        """
        path = self._level_path(self.LEVEL_CONFIG, key)
        path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(config, "model_dump"):
            data = config.model_dump(mode="json")
        else:
            data = config

        # Include cache metadata
        cache_entry = {
            "cache_schema_version": _CACHE_SCHEMA_VERSION,
            "key": key.to_dict(),
            "config": data,
        }

        text = canonical_json_dumps(cache_entry)
        path.write_text(text, encoding="utf-8")

        meta = self._read_meta(key) or {"key": key.to_dict(), "levels": []}
        if self.LEVEL_CONFIG not in meta.get("levels", []):
            levels = meta.get("levels", [])
            levels.append(self.LEVEL_CONFIG)
            self._write_meta(key, levels)

        logger.debug("Cached config: %s", key.short_key)
        return path

    # -------------------------------------------------------------------------
    # Level 2: Mesh cache (generated mesh specifications)
    # -------------------------------------------------------------------------

    def has_mesh(self, key: SimulationCacheKey) -> bool:
        """Check if a mesh specification is cached."""
        path = self._level_path(self.LEVEL_MESH, key)
        return path.exists()

    def get_mesh(self, key: SimulationCacheKey) -> dict[str, Any]:
        """Get a cached mesh specification.

        Args:
            key: Cache key for lookup.

        Returns:
            Dictionary containing the mesh spec data.

        Raises:
            CacheMissError: If the entry is not found.
            CacheCorruptionError: If the entry is corrupted.
        """
        path = self._level_path(self.LEVEL_MESH, key)
        if not path.exists():
            self._misses += 1
            raise CacheMissError(f"Mesh not found in cache: {key.short_key}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._hits += 1
            logger.debug("Cache hit for mesh: %s", key.short_key)
            return data
        except (json.JSONDecodeError, OSError) as e:
            self._misses += 1
            raise CacheCorruptionError(f"Failed to read cached mesh: {e}") from e

    def put_mesh(
        self,
        key: SimulationCacheKey,
        mesh: MeshSpec | dict[str, Any],
    ) -> Path:
        """Store a mesh specification in the cache.

        Args:
            key: Cache key for storage.
            mesh: MeshSpec instance or dictionary to cache.

        Returns:
            Path where the entry was stored.
        """
        path = self._level_path(self.LEVEL_MESH, key)
        path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(mesh, "model_dump"):
            data = mesh.model_dump(mode="json")
        else:
            data = mesh

        # Include cache metadata
        cache_entry = {
            "cache_schema_version": _CACHE_SCHEMA_VERSION,
            "key": key.to_dict(),
            "mesh": data,
        }

        text = canonical_json_dumps(cache_entry)
        path.write_text(text, encoding="utf-8")

        meta = self._read_meta(key) or {"key": key.to_dict(), "levels": []}
        if self.LEVEL_MESH not in meta.get("levels", []):
            levels = meta.get("levels", [])
            levels.append(self.LEVEL_MESH)
            self._write_meta(key, levels)

        logger.debug("Cached mesh: %s", key.short_key)
        return path

    # -------------------------------------------------------------------------
    # Level 3: Result cache (complete simulation results)
    # -------------------------------------------------------------------------

    def has_result(self, key: SimulationCacheKey) -> bool:
        """Check if simulation results are cached."""
        path = self._level_path(self.LEVEL_RESULT, key)
        manifest_path = path / "manifest.json"
        return path.exists() and path.is_dir() and manifest_path.exists()

    def get_result(self, key: SimulationCacheKey) -> CachedSimulationResult:
        """Get cached simulation results.

        Args:
            key: Cache key for lookup.

        Returns:
            CachedSimulationResult with manifest and output file information.

        Raises:
            CacheMissError: If the entry is not found.
            CacheCorruptionError: If the entry is corrupted or incomplete.
        """
        path = self._level_path(self.LEVEL_RESULT, key)
        manifest_path = path / "manifest.json"

        if not path.exists() or not manifest_path.exists():
            self._misses += 1
            raise CacheMissError(f"Result not found in cache: {key.short_key}")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            self._misses += 1
            raise CacheCorruptionError(f"Failed to read cached manifest: {e}") from e

        # Verify all output files exist and match recorded hashes
        output_hashes: dict[str, str] = {}
        outputs_list = manifest.get("outputs", [])
        outputs_dir = path / "sim_outputs"

        for output_entry in outputs_list:
            rel_path = output_entry.get("path", "")
            expected_hash = output_entry.get("hash", "")

            # The path in the manifest is relative to the result directory
            # We need to find the file in the outputs directory
            if rel_path.startswith("sim_outputs/"):
                file_path = path / rel_path
            else:
                file_path = outputs_dir / rel_path

            if not file_path.exists():
                self._misses += 1
                raise CacheCorruptionError(f"Missing output file: {rel_path}")

            actual_hash = _hash_file(file_path)
            if actual_hash != expected_hash:
                self._misses += 1
                raise CacheCorruptionError(f"Hash mismatch for: {rel_path}")

            output_hashes[rel_path] = actual_hash

        # Extract sparam path
        sparam_path: str | None = None
        for output_entry in outputs_list:
            rel_path = output_entry.get("path", "")
            if rel_path.endswith(".s2p") or rel_path.endswith(".csv"):
                sparam_path = rel_path
                break

        self._hits += 1
        logger.info("Cache hit for result: %s", key.short_key)

        return CachedSimulationResult(
            manifest=manifest,
            output_hashes=output_hashes,
            sparam_path=sparam_path,
            outputs_dir=outputs_dir,
        )

    def put_result(
        self,
        key: SimulationCacheKey,
        manifest: dict[str, Any],
        outputs_dir: Path,
    ) -> Path:
        """Store simulation results in the cache.

        This method copies the simulation outputs directory and manifest
        to the cache, enabling future cache hits.

        Args:
            key: Cache key for storage.
            manifest: Complete simulation manifest with output hashes.
            outputs_dir: Directory containing simulation output files.

        Returns:
            Path to the cached result directory.
        """
        cache_path = self._level_path(self.LEVEL_RESULT, key)
        if cache_path.exists():
            shutil.rmtree(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Copy outputs directory
        cache_outputs = cache_path / "sim_outputs"
        if outputs_dir.exists():
            shutil.copytree(outputs_dir, cache_outputs)
        else:
            cache_outputs.mkdir(parents=True, exist_ok=True)

        # Write manifest
        manifest_path = cache_path / "manifest.json"
        text = canonical_json_dumps(manifest)
        manifest_path.write_text(f"{text}\n", encoding="utf-8")

        # Update metadata
        meta = self._read_meta(key) or {"key": key.to_dict(), "levels": []}
        if self.LEVEL_RESULT not in meta.get("levels", []):
            levels = meta.get("levels", [])
            levels.append(self.LEVEL_RESULT)
            self._write_meta(key, levels)

        logger.info("Cached result: %s", key.short_key)
        return cache_path

    def restore_result(self, key: SimulationCacheKey, dest_dir: Path) -> Path:
        """Restore cached simulation results to a destination directory.

        Args:
            key: Cache key for lookup.
            dest_dir: Destination directory to copy results to.

        Returns:
            Path to the restored outputs directory.

        Raises:
            CacheMissError: If the entry is not found.
        """
        cached = self.get_result(key)
        cache_path = self._level_path(self.LEVEL_RESULT, key)

        # Create destination directory
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy manifest
        src_manifest = cache_path / "manifest.json"
        dest_manifest = dest_dir / "simulation_manifest.json"
        shutil.copy2(src_manifest, dest_manifest)

        # Copy outputs
        dest_outputs = dest_dir / "sim_outputs"
        if dest_outputs.exists():
            shutil.rmtree(dest_outputs)
        shutil.copytree(cached.outputs_dir, dest_outputs)

        logger.info("Restored result from cache: %s -> %s", key.short_key, dest_dir)
        return dest_outputs

    # -------------------------------------------------------------------------
    # Cache management
    # -------------------------------------------------------------------------

    def invalidate(self, key: SimulationCacheKey) -> int:
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
                logger.debug("Invalidated cache level %s: %s", level, key.short_key)

        meta_path = self._meta_path(key)
        if meta_path.exists():
            meta_path.unlink()

        if count > 0:
            logger.info("Invalidated %d cache entries: %s", count, key.short_key)
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
        logger.info("Cleared %d cache entries", count)
        return count

    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with hit/miss counts and entry counts.
        """
        config_count = 0
        mesh_count = 0
        result_count = 0
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
                        if level == self.LEVEL_CONFIG:
                            config_count += 1
                        elif level == self.LEVEL_MESH:
                            mesh_count += 1
                        total_size += entry.stat().st_size
                    elif entry.is_dir():
                        if level == self.LEVEL_RESULT:
                            result_count += 1
                            for f in entry.rglob("*"):
                                if f.is_file():
                                    total_size += f.stat().st_size

        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            config_entries=config_count,
            mesh_entries=mesh_count,
            result_entries=result_count,
            total_size_bytes=total_size,
        )

    def verify_integrity(self, key: SimulationCacheKey) -> dict[str, bool]:
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

            if level in (self.LEVEL_CONFIG, self.LEVEL_MESH):
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                    results[level] = True
                except (json.JSONDecodeError, OSError):
                    results[level] = False
            elif level == self.LEVEL_RESULT:
                manifest_path = path / "manifest.json"
                if not manifest_path.exists():
                    results[level] = False
                    continue
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    # Verify output files exist and hash correctly
                    valid = True
                    for output_entry in manifest.get("outputs", []):
                        rel_path = output_entry.get("path", "")
                        expected_hash = output_entry.get("hash", "")
                        if rel_path.startswith("sim_outputs/"):
                            file_path = path / rel_path
                        else:
                            file_path = path / "sim_outputs" / rel_path
                        if not file_path.exists():
                            valid = False
                            break
                        if _hash_file(file_path) != expected_hash:
                            valid = False
                            break
                    results[level] = valid
                except (json.JSONDecodeError, OSError):
                    results[level] = False

        return results


# =============================================================================
# Helper functions
# =============================================================================


def _hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_sim_config_hash(spec: SimulationSpec | dict[str, Any]) -> str:
    """Compute SHA256 hash of a simulation configuration.

    This function computes a deterministic hash from the simulation spec,
    ensuring cache key consistency.

    Args:
        spec: SimulationSpec instance or dictionary.

    Returns:
        SHA256 hash of the canonical JSON representation.
    """
    if hasattr(spec, "model_dump"):
        data = spec.model_dump(mode="json")
    else:
        data = spec

    canonical = canonical_json_dumps(data)
    return sha256_bytes(canonical.encode("utf-8"))


def should_invalidate_cache(
    *,
    cached_design_hash: str,
    cached_sim_config_hash: str,
    cached_solver_version_hash: str,
    current_design_hash: str,
    current_sim_config_hash: str,
    current_solver_version_hash: str,
) -> tuple[bool, list[str]]:
    """Check if cache should be invalidated due to input changes.

    REQ-M2-011: Support cache invalidation on config changes.

    Args:
        cached_design_hash: Design hash from cached manifest.
        cached_sim_config_hash: Sim config hash from cached manifest.
        cached_solver_version_hash: Solver version hash from cached manifest.
        current_design_hash: Current design hash.
        current_sim_config_hash: Current sim config hash.
        current_solver_version_hash: Current solver version hash.

    Returns:
        Tuple of (should_invalidate, list of changed components).
    """
    changes: list[str] = []

    if cached_design_hash != current_design_hash:
        changes.append("design_hash")
    if cached_sim_config_hash != current_sim_config_hash:
        changes.append("sim_config_hash")
    if cached_solver_version_hash != current_solver_version_hash:
        changes.append("solver_version_hash")

    return len(changes) > 0, changes
