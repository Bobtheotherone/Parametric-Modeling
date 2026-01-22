"""Tests for M2 simulation caching system (REQ-M2-011).

This module tests the SimulationCache class which provides:
- Three-level caching (config, mesh, result)
- Cache keyed by (design_hash, sim_config_hash, solver_version)
- Cache invalidation on config changes
- Integrity verification
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.openems.sim_cache import (
    CachedSimulationResult,
    CacheCorruptionError,
    CacheMissError,
    CacheStats,
    SimulationCache,
    SimulationCacheKey,
    SolverVersion,
    compute_sim_config_hash,
    should_invalidate_cache,
)
from formula_foundry.substrate import canonical_json_dumps


# =============================================================================
# Test fixtures
# =============================================================================


def _example_design_hash() -> str:
    """Return a valid 64-character design hash."""
    return "a" * 64


def _example_sim_config_hash() -> str:
    """Return a valid 64-character simulation config hash."""
    return "b" * 64


def _example_solver_version() -> SolverVersion:
    """Return example solver version info."""
    return SolverVersion(
        openems_version="0.0.35",
        mode="docker",
        docker_image="ghcr.io/thliebig/openems:0.0.35@sha256:abc123",
    )


def _example_cache_key() -> SimulationCacheKey:
    """Return an example cache key."""
    return SimulationCacheKey.from_simulation_inputs(
        design_hash=_example_design_hash(),
        sim_config_hash=_example_sim_config_hash(),
        solver_version=_example_solver_version(),
    )


def _example_config_data() -> dict[str, Any]:
    """Return example simulation config data."""
    return {
        "schema_version": 1,
        "simulation_id": "test_sim_001",
        "toolchain": {
            "openems": {
                "version": "0.0.35",
                "docker_image": "ghcr.io/thliebig/openems:0.0.35@sha256:abc123",
            }
        },
        "geometry_ref": {"design_hash": "a" * 64, "coupon_id": "TEST001"},
        "excitation": {"type": "gaussian", "f0_hz": 5_000_000_000, "fc_hz": 10_000_000_000},
        "frequency": {"f_start_hz": 1_000_000_000, "f_stop_hz": 20_000_000_000, "n_points": 201},
        "ports": [
            {
                "id": "P1",
                "type": "lumped",
                "impedance_ohm": 50.0,
                "excite": True,
                "position_nm": [0, 0, 0],
                "direction": "x",
                "deembed": {"enabled": False},
            }
        ],
    }


def _example_mesh_data() -> dict[str, Any]:
    """Return example mesh spec data."""
    return {
        "resolution": {
            "lambda_resolution": 20,
            "metal_edge_resolution_nm": 50000,
            "via_resolution_nm": 25000,
        },
        "smoothing": {"max_ratio": 1.5, "smooth_mesh_lines": True},
        "fixed_lines_x_nm": [0, 1000000, 2000000],
        "fixed_lines_y_nm": [0, 500000],
        "fixed_lines_z_nm": [0, 180000, 980000],
    }


def _example_manifest() -> dict[str, Any]:
    """Return example simulation manifest."""
    return {
        "schema_version": 1,
        "simulation_id": "test_sim_001",
        "simulation_hash": "c" * 64,
        "spec_hash": "b" * 64,
        "geometry_hash": "d" * 64,
        "design_hash": "a" * 64,
        "coupon_family": "F1_SINGLE_ENDED_VIA",
        "toolchain": {
            "openems": {
                "version": "0.0.35",
                "docker_image": "ghcr.io/thliebig/openems:0.0.35@sha256:abc123",
            }
        },
        "toolchain_hash": "e" * 64,
        "frequency_sweep": {"f_start_hz": 1e9, "f_stop_hz": 20e9, "n_points": 201},
        "excitation": {"type": "gaussian", "f0_hz": 5e9, "fc_hz": 10e9},
        "boundaries": {"x_min": "PML_8", "x_max": "PML_8", "y_min": "PEC", "y_max": "PEC", "z_min": "PEC", "z_max": "PML_8"},
        "mesh_config": {
            "resolution": {"lambda_resolution": 20, "metal_edge_resolution_nm": 50000, "via_resolution_nm": 25000},
            "smoothing": {"max_ratio": 1.5, "smooth_mesh_lines": True},
        },
        "ports": [{"id": "P1", "type": "lumped", "impedance_ohm": 50.0, "excite": True, "position_nm": [0, 0, 0], "direction": "x", "deembed_enabled": False}],
        "outputs": [],
        "lineage": {"git_sha": "f" * 40, "timestamp_utc": "2026-01-21T00:00:00Z"},
        "sim_config_hash": "b" * 64,
    }


def _create_test_outputs(outputs_dir: Path) -> dict[str, str]:
    """Create test output files and return their hashes."""
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Create S-parameter file
    sparam_content = "# Hz S RI R 50\n1000000000 -0.1 0.0 0.9 0.0 0.9 0.0 -0.1 0.0\n"
    sparam_path = outputs_dir / "sparams.s2p"
    sparam_path.write_text(sparam_content, encoding="utf-8")

    # Create port signals file
    port_signals = {"time_ps": [0, 10, 20], "ports": {"P1": {"voltage_v": [0.0, 1.0, 0.5], "current_a": [0.0, 0.02, 0.01]}}}
    port_signals_path = outputs_dir / "port_signals.json"
    port_signals_path.write_text(canonical_json_dumps(port_signals), encoding="utf-8")

    # Create energy decay file
    energy_decay = {"time_ps": [0, 100, 200], "energy_db": [-10, -30, -50]}
    energy_decay_path = outputs_dir / "energy_decay.json"
    energy_decay_path.write_text(canonical_json_dumps(energy_decay), encoding="utf-8")

    # Compute hashes
    import hashlib

    def hash_file(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    return {
        "sim_outputs/sparams.s2p": hash_file(sparam_path),
        "sim_outputs/port_signals.json": hash_file(port_signals_path),
        "sim_outputs/energy_decay.json": hash_file(energy_decay_path),
    }


# =============================================================================
# SolverVersion tests
# =============================================================================


class TestSolverVersion:
    """Tests for SolverVersion dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        version = SolverVersion(
            openems_version="0.0.35",
            mode="docker",
            docker_image="ghcr.io/thliebig/openems:0.0.35",
            csxcad_version="0.6.2",
        )
        data = version.to_dict()
        assert data["openems_version"] == "0.0.35"
        assert data["mode"] == "docker"
        assert data["docker_image"] == "ghcr.io/thliebig/openems:0.0.35"
        assert data["csxcad_version"] == "0.6.2"

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "openems_version": "0.0.35",
            "mode": "local",
            "docker_image": None,
        }
        version = SolverVersion.from_dict(data)
        assert version.openems_version == "0.0.35"
        assert version.mode == "local"
        assert version.docker_image is None

    def test_from_stub_mode(self) -> None:
        """Test stub mode factory."""
        version = SolverVersion.from_stub_mode()
        assert version.openems_version is None
        assert version.mode == "stub"

    def test_compute_hash_deterministic(self) -> None:
        """Test that hash computation is deterministic."""
        version1 = SolverVersion(openems_version="0.0.35", mode="docker")
        version2 = SolverVersion(openems_version="0.0.35", mode="docker")
        assert version1.compute_hash() == version2.compute_hash()

    def test_compute_hash_changes_with_version(self) -> None:
        """Test that hash changes when version changes."""
        version1 = SolverVersion(openems_version="0.0.35", mode="docker")
        version2 = SolverVersion(openems_version="0.0.36", mode="docker")
        assert version1.compute_hash() != version2.compute_hash()

    def test_compute_hash_changes_with_mode(self) -> None:
        """Test that hash changes when mode changes."""
        version1 = SolverVersion(openems_version="0.0.35", mode="docker")
        version2 = SolverVersion(openems_version="0.0.35", mode="local")
        assert version1.compute_hash() != version2.compute_hash()


# =============================================================================
# SimulationCacheKey tests
# =============================================================================


class TestSimulationCacheKey:
    """Tests for SimulationCacheKey dataclass."""

    def test_valid_key_creation(self) -> None:
        """Test creating a valid cache key."""
        key = SimulationCacheKey(
            design_hash="a" * 64,
            sim_config_hash="b" * 64,
            solver_version_hash="c" * 64,
        )
        assert len(key.combined_hash) == 64
        assert len(key.short_key) > 0

    def test_invalid_design_hash_length(self) -> None:
        """Test that invalid design hash length raises error."""
        with pytest.raises(ValueError, match="design_hash must be a 64-character"):
            SimulationCacheKey(
                design_hash="short",
                sim_config_hash="b" * 64,
                solver_version_hash="c" * 64,
            )

    def test_invalid_sim_config_hash_length(self) -> None:
        """Test that invalid sim_config hash length raises error."""
        with pytest.raises(ValueError, match="sim_config_hash must be a 64-character"):
            SimulationCacheKey(
                design_hash="a" * 64,
                sim_config_hash="short",
                solver_version_hash="c" * 64,
            )

    def test_from_simulation_inputs(self) -> None:
        """Test factory method from simulation inputs."""
        key = SimulationCacheKey.from_simulation_inputs(
            design_hash="a" * 64,
            sim_config_hash="b" * 64,
            solver_version=SolverVersion(openems_version="0.0.35", mode="docker"),
        )
        assert key.design_hash == "a" * 64
        assert key.sim_config_hash == "b" * 64
        assert len(key.solver_version_hash) == 64

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        key1 = _example_cache_key()
        data = key1.to_dict()
        key2 = SimulationCacheKey.from_dict(data)
        assert key1 == key2

    def test_combined_hash_deterministic(self) -> None:
        """Test that combined hash is deterministic."""
        key1 = _example_cache_key()
        key2 = _example_cache_key()
        assert key1.combined_hash == key2.combined_hash

    def test_combined_hash_changes_with_design(self) -> None:
        """Test that combined hash changes when design changes."""
        key1 = SimulationCacheKey(design_hash="a" * 64, sim_config_hash="b" * 64, solver_version_hash="c" * 64)
        key2 = SimulationCacheKey(design_hash="d" * 64, sim_config_hash="b" * 64, solver_version_hash="c" * 64)
        assert key1.combined_hash != key2.combined_hash


# =============================================================================
# SimulationCache Level 1 (Config) tests
# =============================================================================


class TestSimulationCacheConfig:
    """Tests for config-level caching."""

    def test_put_and_get_config(self, tmp_path: Path) -> None:
        """Test storing and retrieving config from cache."""
        cache = SimulationCache(root=tmp_path / "sim_cache")
        key = _example_cache_key()
        config_data = _example_config_data()

        # Put config
        path = cache.put_config(key, config_data)
        assert path.exists()

        # Get config
        retrieved = cache.get_config(key)
        assert retrieved["config"] == config_data

    def test_has_config(self, tmp_path: Path) -> None:
        """Test checking if config exists in cache."""
        cache = SimulationCache(root=tmp_path / "sim_cache")
        key = _example_cache_key()

        assert not cache.has_config(key)

        cache.put_config(key, _example_config_data())

        assert cache.has_config(key)

    def test_get_config_miss(self, tmp_path: Path) -> None:
        """Test cache miss raises appropriate error."""
        cache = SimulationCache(root=tmp_path / "sim_cache")
        key = _example_cache_key()

        with pytest.raises(CacheMissError):
            cache.get_config(key)


# =============================================================================
# SimulationCache Level 2 (Mesh) tests
# =============================================================================


class TestSimulationCacheMesh:
    """Tests for mesh-level caching."""

    def test_put_and_get_mesh(self, tmp_path: Path) -> None:
        """Test storing and retrieving mesh from cache."""
        cache = SimulationCache(root=tmp_path / "sim_cache")
        key = _example_cache_key()
        mesh_data = _example_mesh_data()

        # Put mesh
        path = cache.put_mesh(key, mesh_data)
        assert path.exists()

        # Get mesh
        retrieved = cache.get_mesh(key)
        assert retrieved["mesh"] == mesh_data

    def test_has_mesh(self, tmp_path: Path) -> None:
        """Test checking if mesh exists in cache."""
        cache = SimulationCache(root=tmp_path / "sim_cache")
        key = _example_cache_key()

        assert not cache.has_mesh(key)

        cache.put_mesh(key, _example_mesh_data())

        assert cache.has_mesh(key)

    def test_get_mesh_miss(self, tmp_path: Path) -> None:
        """Test mesh cache miss raises appropriate error."""
        cache = SimulationCache(root=tmp_path / "sim_cache")
        key = _example_cache_key()

        with pytest.raises(CacheMissError):
            cache.get_mesh(key)


# =============================================================================
# SimulationCache Level 3 (Result) tests
# =============================================================================


class TestSimulationCacheResult:
    """Tests for result-level caching (raw field data, intermediate, S-params)."""

    def test_put_and_get_result(self, tmp_path: Path) -> None:
        """REQ-M2-011: Cache raw field data, intermediate results, and final S-params."""
        cache = SimulationCache(root=tmp_path / "sim_cache")
        key = _example_cache_key()

        # Create test outputs
        outputs_dir = tmp_path / "outputs" / "sim_outputs"
        output_hashes = _create_test_outputs(outputs_dir)

        # Create manifest with output hashes
        manifest = _example_manifest()
        manifest["outputs"] = [{"path": path, "hash": hash_val} for path, hash_val in output_hashes.items()]

        # Put result
        cache_path = cache.put_result(key, manifest, outputs_dir)
        assert cache_path.exists()
        assert (cache_path / "manifest.json").exists()
        assert (cache_path / "sim_outputs" / "sparams.s2p").exists()

        # Get result
        result = cache.get_result(key)
        assert isinstance(result, CachedSimulationResult)
        assert result.manifest["simulation_id"] == "test_sim_001"
        assert len(result.output_hashes) == 3

    def test_has_result(self, tmp_path: Path) -> None:
        """Test checking if result exists in cache."""
        cache = SimulationCache(root=tmp_path / "sim_cache")
        key = _example_cache_key()

        assert not cache.has_result(key)

        # Create and store result
        outputs_dir = tmp_path / "outputs" / "sim_outputs"
        output_hashes = _create_test_outputs(outputs_dir)
        manifest = _example_manifest()
        manifest["outputs"] = [{"path": path, "hash": hash_val} for path, hash_val in output_hashes.items()]
        cache.put_result(key, manifest, outputs_dir)

        assert cache.has_result(key)

    def test_get_result_miss(self, tmp_path: Path) -> None:
        """Test result cache miss raises appropriate error."""
        cache = SimulationCache(root=tmp_path / "sim_cache")
        key = _example_cache_key()

        with pytest.raises(CacheMissError):
            cache.get_result(key)

    def test_get_result_verifies_file_hashes(self, tmp_path: Path) -> None:
        """REQ-M2-011: Cache validates integrity via content hashes."""
        cache = SimulationCache(root=tmp_path / "sim_cache")
        key = _example_cache_key()

        # Create and store result
        outputs_dir = tmp_path / "outputs" / "sim_outputs"
        output_hashes = _create_test_outputs(outputs_dir)
        manifest = _example_manifest()
        manifest["outputs"] = [{"path": path, "hash": hash_val} for path, hash_val in output_hashes.items()]
        cache_path = cache.put_result(key, manifest, outputs_dir)

        # Corrupt a cached file
        corrupt_file = cache_path / "sim_outputs" / "sparams.s2p"
        corrupt_file.write_text("corrupted content", encoding="utf-8")

        # Should raise corruption error on get
        with pytest.raises(CacheCorruptionError, match="Hash mismatch"):
            cache.get_result(key)

    def test_restore_result(self, tmp_path: Path) -> None:
        """Test restoring cached result to destination directory."""
        cache = SimulationCache(root=tmp_path / "sim_cache")
        key = _example_cache_key()

        # Create and store result
        outputs_dir = tmp_path / "outputs" / "sim_outputs"
        output_hashes = _create_test_outputs(outputs_dir)
        manifest = _example_manifest()
        manifest["outputs"] = [{"path": path, "hash": hash_val} for path, hash_val in output_hashes.items()]
        cache.put_result(key, manifest, outputs_dir)

        # Restore to new location
        dest_dir = tmp_path / "restored"
        restored_outputs = cache.restore_result(key, dest_dir)

        assert restored_outputs.exists()
        assert (dest_dir / "simulation_manifest.json").exists()
        assert (restored_outputs / "sparams.s2p").exists()
        assert (restored_outputs / "port_signals.json").exists()
        assert (restored_outputs / "energy_decay.json").exists()


# =============================================================================
# Cache management tests
# =============================================================================


class TestSimulationCacheManagement:
    """Tests for cache management operations."""

    def test_invalidate_removes_all_levels(self, tmp_path: Path) -> None:
        """REQ-M2-011: Support cache invalidation on config changes."""
        cache = SimulationCache(root=tmp_path / "sim_cache")
        key = _example_cache_key()

        # Store entries at all levels
        cache.put_config(key, _example_config_data())
        cache.put_mesh(key, _example_mesh_data())

        outputs_dir = tmp_path / "outputs" / "sim_outputs"
        output_hashes = _create_test_outputs(outputs_dir)
        manifest = _example_manifest()
        manifest["outputs"] = [{"path": path, "hash": hash_val} for path, hash_val in output_hashes.items()]
        cache.put_result(key, manifest, outputs_dir)

        # Verify all exist
        assert cache.has_config(key)
        assert cache.has_mesh(key)
        assert cache.has_result(key)

        # Invalidate
        count = cache.invalidate(key)
        assert count == 3

        # Verify all removed
        assert not cache.has_config(key)
        assert not cache.has_mesh(key)
        assert not cache.has_result(key)

    def test_clear_removes_all_entries(self, tmp_path: Path) -> None:
        """Test clearing entire cache."""
        cache = SimulationCache(root=tmp_path / "sim_cache")
        key1 = _example_cache_key()
        key2 = SimulationCacheKey(design_hash="d" * 64, sim_config_hash="e" * 64, solver_version_hash="f" * 64)

        # Store entries for different keys
        cache.put_config(key1, _example_config_data())
        cache.put_config(key2, _example_config_data())

        # Clear all
        count = cache.clear()
        assert count == 2

        assert not cache.has_config(key1)
        assert not cache.has_config(key2)

    def test_stats_tracking(self, tmp_path: Path) -> None:
        """Test cache statistics tracking."""
        cache = SimulationCache(root=tmp_path / "sim_cache")
        key = _example_cache_key()

        # Initial stats
        stats = cache.stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.config_entries == 0

        # Store config
        cache.put_config(key, _example_config_data())

        # Get config (hit)
        cache.get_config(key)

        stats = cache.stats()
        assert stats.hits == 1
        assert stats.config_entries == 1

        # Try to get non-existent config (miss)
        other_key = SimulationCacheKey(design_hash="x" * 64, sim_config_hash="y" * 64, solver_version_hash="z" * 64)
        try:
            cache.get_config(other_key)
        except CacheMissError:
            pass

        stats = cache.stats()
        assert stats.misses == 1

    def test_verify_integrity(self, tmp_path: Path) -> None:
        """Test integrity verification."""
        cache = SimulationCache(root=tmp_path / "sim_cache")
        key = _example_cache_key()

        # Store config and mesh
        cache.put_config(key, _example_config_data())
        cache.put_mesh(key, _example_mesh_data())

        # Verify integrity
        integrity = cache.verify_integrity(key)
        assert integrity[cache.LEVEL_CONFIG] is True
        assert integrity[cache.LEVEL_MESH] is True
        assert integrity[cache.LEVEL_RESULT] is False  # Not stored yet

        # Corrupt config file
        config_path = cache._level_path(cache.LEVEL_CONFIG, key)
        config_path.write_text("invalid json", encoding="utf-8")

        integrity = cache.verify_integrity(key)
        assert integrity[cache.LEVEL_CONFIG] is False


# =============================================================================
# Cache invalidation helper tests
# =============================================================================


class TestCacheInvalidationHelpers:
    """Tests for cache invalidation helper functions."""

    def test_should_invalidate_on_design_change(self) -> None:
        """REQ-M2-011: Invalidate when design_hash changes."""
        should_inv, changes = should_invalidate_cache(
            cached_design_hash="a" * 64,
            cached_sim_config_hash="b" * 64,
            cached_solver_version_hash="c" * 64,
            current_design_hash="d" * 64,  # Changed
            current_sim_config_hash="b" * 64,
            current_solver_version_hash="c" * 64,
        )
        assert should_inv is True
        assert "design_hash" in changes

    def test_should_invalidate_on_config_change(self) -> None:
        """REQ-M2-011: Invalidate when sim_config_hash changes."""
        should_inv, changes = should_invalidate_cache(
            cached_design_hash="a" * 64,
            cached_sim_config_hash="b" * 64,
            cached_solver_version_hash="c" * 64,
            current_design_hash="a" * 64,
            current_sim_config_hash="e" * 64,  # Changed
            current_solver_version_hash="c" * 64,
        )
        assert should_inv is True
        assert "sim_config_hash" in changes

    def test_should_invalidate_on_solver_change(self) -> None:
        """REQ-M2-011: Invalidate when solver_version changes."""
        should_inv, changes = should_invalidate_cache(
            cached_design_hash="a" * 64,
            cached_sim_config_hash="b" * 64,
            cached_solver_version_hash="c" * 64,
            current_design_hash="a" * 64,
            current_sim_config_hash="b" * 64,
            current_solver_version_hash="f" * 64,  # Changed
        )
        assert should_inv is True
        assert "solver_version_hash" in changes

    def test_no_invalidation_when_unchanged(self) -> None:
        """Test no invalidation when all hashes match."""
        should_inv, changes = should_invalidate_cache(
            cached_design_hash="a" * 64,
            cached_sim_config_hash="b" * 64,
            cached_solver_version_hash="c" * 64,
            current_design_hash="a" * 64,
            current_sim_config_hash="b" * 64,
            current_solver_version_hash="c" * 64,
        )
        assert should_inv is False
        assert len(changes) == 0


# =============================================================================
# compute_sim_config_hash tests
# =============================================================================


class TestComputeSimConfigHash:
    """Tests for compute_sim_config_hash function."""

    def test_hash_is_deterministic(self) -> None:
        """Test that config hash is deterministic."""
        config = _example_config_data()
        hash1 = compute_sim_config_hash(config)
        hash2 = compute_sim_config_hash(config)
        assert hash1 == hash2

    def test_hash_changes_with_config(self) -> None:
        """Test that hash changes when config changes."""
        config1 = _example_config_data()
        config2 = _example_config_data()
        config2["frequency"]["n_points"] = 401  # Change

        hash1 = compute_sim_config_hash(config1)
        hash2 = compute_sim_config_hash(config2)
        assert hash1 != hash2

    def test_hash_format(self) -> None:
        """Test that hash is 64-character hex string."""
        config = _example_config_data()
        hash_val = compute_sim_config_hash(config)
        assert len(hash_val) == 64
        assert all(c in "0123456789abcdef" for c in hash_val)


# =============================================================================
# Integration tests
# =============================================================================


class TestSimulationCacheIntegration:
    """Integration tests for the complete caching workflow."""

    def test_full_cache_workflow(self, tmp_path: Path) -> None:
        """REQ-M2-011: Test complete cache lifecycle."""
        cache = SimulationCache(root=tmp_path / "sim_cache")

        # Create simulation inputs
        design_hash = "a" * 64
        config = _example_config_data()
        sim_config_hash = compute_sim_config_hash(config)
        solver_version = SolverVersion(openems_version="0.0.35", mode="docker")

        key = SimulationCacheKey.from_simulation_inputs(
            design_hash=design_hash,
            sim_config_hash=sim_config_hash,
            solver_version=solver_version,
        )

        # Store config
        cache.put_config(key, config)

        # Store mesh
        cache.put_mesh(key, _example_mesh_data())

        # Simulate running simulation and store result
        outputs_dir = tmp_path / "outputs" / "sim_outputs"
        output_hashes = _create_test_outputs(outputs_dir)
        manifest = _example_manifest()
        manifest["outputs"] = [{"path": path, "hash": hash_val} for path, hash_val in output_hashes.items()]
        cache.put_result(key, manifest, outputs_dir)

        # On subsequent run, check cache
        assert cache.has_config(key)
        assert cache.has_mesh(key)
        assert cache.has_result(key)

        # Get cached result
        result = cache.get_result(key)
        assert result.sparam_path == "sim_outputs/sparams.s2p"

        # Stats should reflect cache activity
        stats = cache.stats()
        assert stats.config_entries == 1
        assert stats.mesh_entries == 1
        assert stats.result_entries == 1

    def test_cache_invalidation_on_config_change(self, tmp_path: Path) -> None:
        """REQ-M2-011: Test that changing config invalidates cache."""
        cache = SimulationCache(root=tmp_path / "sim_cache")

        # Initial configuration
        config_v1 = _example_config_data()
        solver_version = SolverVersion(openems_version="0.0.35", mode="docker")

        key_v1 = SimulationCacheKey.from_simulation_inputs(
            design_hash="a" * 64,
            sim_config_hash=compute_sim_config_hash(config_v1),
            solver_version=solver_version,
        )

        # Store result for v1
        cache.put_config(key_v1, config_v1)
        outputs_dir = tmp_path / "outputs" / "sim_outputs"
        output_hashes = _create_test_outputs(outputs_dir)
        manifest = _example_manifest()
        manifest["outputs"] = [{"path": path, "hash": hash_val} for path, hash_val in output_hashes.items()]
        cache.put_result(key_v1, manifest, outputs_dir)

        # Modify configuration
        config_v2 = _example_config_data()
        config_v2["frequency"]["n_points"] = 401  # Changed

        key_v2 = SimulationCacheKey.from_simulation_inputs(
            design_hash="a" * 64,
            sim_config_hash=compute_sim_config_hash(config_v2),
            solver_version=solver_version,
        )

        # v2 key should not find cached result
        assert not cache.has_result(key_v2)

        # Check invalidation detection
        should_inv, changes = should_invalidate_cache(
            cached_design_hash="a" * 64,
            cached_sim_config_hash=compute_sim_config_hash(config_v1),
            cached_solver_version_hash=solver_version.compute_hash(),
            current_design_hash="a" * 64,
            current_sim_config_hash=compute_sim_config_hash(config_v2),
            current_solver_version_hash=solver_version.compute_hash(),
        )
        assert should_inv is True
        assert "sim_config_hash" in changes
