"""M5 Chunking: Memory-aware batched evaluation utilities.

This module provides chunking utilities for splitting large computations
into memory-efficient chunks for both CPU and GPU execution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from formula_foundry.m5.backend.array_backend import ArrayBackend


def max_elements_for_bytes(bytes_limit: int, dtype: Any = np.float64) -> int:
    """Calculate maximum number of array elements for given memory limit.

    Args:
        bytes_limit: Maximum memory in bytes
        dtype: Array dtype

    Returns:
        Maximum number of elements that fit in memory
    """
    dtype = np.dtype(dtype)
    return bytes_limit // dtype.itemsize


@dataclass
class ChunkingPolicy:
    """Configuration for memory-aware chunking.

    Attributes:
        max_elements: Maximum total elements per chunk (n_params * n_freqs)
        max_bytes: Maximum memory in bytes per chunk (alternative to max_elements)
        chunk_n: Fixed chunk size for param dimension (overrides auto-sizing)
        chunk_f: Fixed chunk size for freq dimension (overrides auto-sizing)
        max_param_chunk: Alias for chunk_n (legacy)
        max_freq_chunk: Alias for chunk_f (legacy)
        max_memory_bytes: Alias for max_bytes (legacy)
        auto_size: Automatically determine chunk sizes from memory limit
    """

    max_elements: int | None = None
    max_bytes: int | None = None
    chunk_n: int | None = None
    chunk_f: int | None = None
    max_param_chunk: int | None = None
    max_freq_chunk: int | None = None
    max_memory_bytes: int | None = None
    auto_size: bool = True

    def resolve_chunk_shape(
        self,
        n_params: int,
        n_freqs: int,
        bytes_per_element: int | None = None,
    ) -> tuple[int, int]:
        """Determine chunk shape for given dimensions.

        The chunking strategy prioritizes maximizing the frequency chunk size
        while keeping param chunk size minimal (1), to optimize for typical
        workloads where frequency iteration is the inner loop.

        Args:
            n_params: Total number of parameters
            n_freqs: Total number of frequencies
            bytes_per_element: Bytes per element (required if max_bytes is set)

        Returns:
            Tuple of (param_chunk_size, freq_chunk_size)

        Raises:
            ValueError: If max_bytes is set but bytes_per_element is not provided
        """
        # Handle explicit chunk sizes first
        param_chunk = self.chunk_n or self.max_param_chunk
        freq_chunk = self.chunk_f or self.max_freq_chunk

        if param_chunk is not None and freq_chunk is not None:
            return min(param_chunk, n_params), min(freq_chunk, n_freqs)

        # Handle max_elements constraint
        if self.max_elements is not None:
            # Strategy: minimize param chunk (1), maximize freq chunk
            param_size = 1
            freq_size = min(self.max_elements, n_freqs)
            return param_size, freq_size

        # Handle max_bytes constraint
        max_bytes = self.max_bytes or self.max_memory_bytes
        if max_bytes is not None:
            if bytes_per_element is None:
                raise ValueError("bytes_per_element required when max_bytes is set")
            max_elems = max_bytes // bytes_per_element
            # Strategy: minimize param chunk (1), maximize freq chunk
            param_size = 1
            freq_size = min(max_elems, n_freqs)
            return param_size, freq_size

        # Default: process all at once
        return n_params, n_freqs

    def get_chunk_sizes(
        self,
        n_params: int,
        n_freqs: int,
        dtype: Any = np.float64,
    ) -> tuple[int, int]:
        """Determine chunk sizes for given array dimensions.

        Args:
            n_params: Total number of parameters
            n_freqs: Total number of frequencies
            dtype: Array dtype

        Returns:
            Tuple of (param_chunk_size, freq_chunk_size)
        """
        dtype_obj = np.dtype(dtype)
        bytes_per_element = dtype_obj.itemsize
        return self.resolve_chunk_shape(n_params, n_freqs, bytes_per_element)


@dataclass
class ChunkedEvaluator:
    """Memory-aware chunked evaluation executor.

    Splits large computations into chunks to fit within memory limits.

    Attributes:
        backend: Array backend to use
        eval_fn: Evaluation function (param_chunk, freq_chunk) -> result
        chunking: Chunking policy
    """

    backend: ArrayBackend
    eval_fn: Callable[[Any, Any], Any]
    chunking: ChunkingPolicy | None = None

    def _check_and_cast_dtype(self, arr: Any) -> Any:
        """Check dtype policy and cast if needed.

        Args:
            arr: Input array

        Returns:
            Array with correct dtype (possibly cast)

        Raises:
            TypeError: If dtype mismatch and allow_cast is False
        """
        policy = self.backend.dtype_policy
        if policy is None:
            return arr

        xp = self.backend.xp
        target_dtype = getattr(xp, policy.dtype, None)
        if target_dtype is None:
            target_dtype = np.dtype(policy.dtype)

        if hasattr(arr, "dtype") and arr.dtype != target_dtype:
            if not policy.allow_cast:
                raise TypeError(
                    f"dtype mismatch: expected {policy.dtype}, got {arr.dtype}. "
                    f"Set allow_cast=True to enable automatic casting."
                )
            return arr.astype(target_dtype)
        return arr

    def evaluate(self, params: Any, freqs: Any) -> Any:
        """Evaluate function over parameters and frequencies.

        Args:
            params: Parameter array [n_params, n_dims]
            freqs: Frequency array [n_freqs]

        Returns:
            Result array [n_params, n_freqs]
        """
        xp = self.backend.xp

        params = xp.asarray(params)
        freqs = xp.asarray(freqs)

        # Apply dtype policy
        params = self._check_and_cast_dtype(params)
        freqs = self._check_and_cast_dtype(freqs)

        n_params = params.shape[0]
        n_freqs = freqs.shape[0]

        if self.chunking is None:
            # No chunking - evaluate all at once
            return self.eval_fn(params, freqs)

        param_chunk_size, freq_chunk_size = self.chunking.get_chunk_sizes(
            n_params, n_freqs, params.dtype
        )

        # Allocate result array
        result = None

        for p_start in range(0, n_params, param_chunk_size):
            p_end = min(p_start + param_chunk_size, n_params)
            param_chunk = params[p_start:p_end]

            for f_start in range(0, n_freqs, freq_chunk_size):
                f_end = min(f_start + freq_chunk_size, n_freqs)
                freq_chunk = freqs[f_start:f_end]

                chunk_result = self.eval_fn(param_chunk, freq_chunk)

                if result is None:
                    # Initialize result array based on first chunk
                    result = xp.zeros((n_params, n_freqs), dtype=chunk_result.dtype)

                result[p_start:p_end, f_start:f_end] = chunk_result

        return result


__all__ = [
    "ChunkedEvaluator",
    "ChunkingPolicy",
    "max_elements_for_bytes",
]
