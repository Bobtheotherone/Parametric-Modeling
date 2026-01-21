"""Toolchain lock and configuration module.

Provides functions to load toolchain lock files and compute toolchain hashes
for deterministic, pinned KiCad toolchain usage in CI and manifest generation.

Satisfies:
    - CP-1.1: Add toolchain lock + explicit pinning strategy
    - CP-1.3: Used by docker runner and manifest generation
"""

from .lock import (
    DEFAULT_LOCK_PATH,
    ToolchainConfig,
    ToolchainLoadError,
    compute_toolchain_hash,
    load_toolchain_lock,
    load_toolchain_lock_from_dict,
)

__all__ = [
    "DEFAULT_LOCK_PATH",
    "ToolchainConfig",
    "ToolchainLoadError",
    "compute_toolchain_hash",
    "load_toolchain_lock",
    "load_toolchain_lock_from_dict",
]
