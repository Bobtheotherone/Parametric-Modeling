"""Substrate utilities for deterministic, GPU-first execution."""

from .determinism import (
    DEFAULT_CUBLAS_WORKSPACE_CONFIG,
    VALID_CUBLAS_WORKSPACE_CONFIGS,
    DeterminismConfig,
    DeterminismMode,
    apply_determinism,
    determinism_context,
    determinism_manifest,
)

__all__ = [
    "DEFAULT_CUBLAS_WORKSPACE_CONFIG",
    "VALID_CUBLAS_WORKSPACE_CONFIGS",
    "DeterminismConfig",
    "DeterminismMode",
    "apply_determinism",
    "determinism_context",
    "determinism_manifest",
]
