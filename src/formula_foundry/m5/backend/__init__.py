"""M5 Backend package: Array backend and chunking utilities."""

from formula_foundry.m5.backend.array_backend import (
    ArrayBackend,
    BackendConfig,
    DeviceType,
    DTypePolicy,
    select_array_backend,
)
from formula_foundry.m5.backend.chunking import ChunkedEvaluator, ChunkingPolicy, max_elements_for_bytes

__all__ = [
    "ArrayBackend",
    "BackendConfig",
    "ChunkedEvaluator",
    "ChunkingPolicy",
    "DeviceType",
    "DTypePolicy",
    "max_elements_for_bytes",
    "select_array_backend",
]
