"""Substrate utilities for deterministic, GPU-first execution."""

from . import backends
from .backends import (
    Backend,
    BackendName,
    BackendSelectionError,
    GPUNotAvailableError,
    HostTransferError,
    cupy_to_torch,
    host_transfer_guard,
    select_backend,
    torch_to_cupy,
)
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
    "Backend",
    "BackendName",
    "BackendSelectionError",
    "DEFAULT_CUBLAS_WORKSPACE_CONFIG",
    "GPUNotAvailableError",
    "HostTransferError",
    "VALID_CUBLAS_WORKSPACE_CONFIGS",
    "DeterminismConfig",
    "DeterminismMode",
    "apply_determinism",
    "backends",
    "cupy_to_torch",
    "determinism_context",
    "determinism_manifest",
    "host_transfer_guard",
    "select_backend",
    "torch_to_cupy",
]
