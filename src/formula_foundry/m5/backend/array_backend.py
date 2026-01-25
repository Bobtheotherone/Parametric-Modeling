"""M5 Array Backend: CPU/GPU array computation abstraction.

This module provides backend abstraction for array computations,
supporting both CPU (NumPy) and GPU (CuPy) backends.

Environment variables:
    FF_M5_BACKEND: Force backend selection ("numpy" or "cupy")
    FF_M5_DTYPE: Default dtype for arrays ("float32" or "float64")
    FF_M5_DTYPE_ALLOW_CAST: Allow automatic dtype casting ("1" or "0")
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from formula_foundry.substrate.backends import BackendSelectionError, GPUNotAvailableError

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DeviceType(str, Enum):
    """Device type for array backend."""

    CPU = "cpu"
    GPU = "gpu"


@dataclass
class DTypePolicy:
    """Policy for dtype handling in backend computations.

    Attributes:
        dtype: Target dtype for arrays (e.g., "float32", "float64")
        allow_cast: If True, inputs with mismatched dtypes will be cast to target.
                    If False, a TypeError is raised on dtype mismatch.
    """

    dtype: str = "float64"
    allow_cast: bool = True


@dataclass
class BackendConfig:
    """Configuration for array backend selection.

    Attributes:
        prefer_gpu: Prefer GPU backend if available
        require_gpu: Require GPU backend (raise if unavailable)
        device_id: GPU device ID (if multiple GPUs)
        dtype: Default dtype for arrays
        dtype_policy: Optional dtype policy for stricter type handling
        backend: Force specific backend ("numpy" or "cupy")
    """

    prefer_gpu: bool = True
    require_gpu: bool = False
    device_id: int | None = None
    dtype: str = "float64"
    dtype_policy: DTypePolicy | None = None
    backend: str | None = None


@dataclass
class ArrayBackend:
    """Abstract array backend for CPU/GPU computation.

    Provides a common interface for array operations that can be
    executed on either CPU (NumPy) or GPU (CuPy).

    Attributes:
        device_type: The device type (cpu or gpu)
        xp: The array module (numpy or cupy)
        device_id: GPU device ID (None for CPU)
        require_gpu: Whether GPU was required
        dtype_policy: Optional dtype policy for type handling
    """

    device_type: DeviceType = DeviceType.CPU
    xp: Any = field(default_factory=lambda: np)
    device_id: int | None = None
    require_gpu: bool = False
    dtype_policy: DTypePolicy | None = None

    @property
    def name(self) -> str:
        """Return backend name ('numpy' or 'cupy')."""
        return "cupy" if self.device_type == DeviceType.GPU else "numpy"

    @property
    def gpu_available(self) -> bool:
        """Return whether GPU is available for this backend."""
        return self.device_type == DeviceType.GPU

    def to_device(self, arr: NDArray[Any]) -> Any:
        """Transfer array to this backend's device."""
        if self.device_type == DeviceType.CPU:
            return np.asarray(arr)
        # For GPU, would use cupy
        return np.asarray(arr)

    def to_host(self, arr: Any) -> NDArray[Any]:
        """Transfer array from device to host (numpy)."""
        if hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """Create zero array on device."""
        return self.xp.zeros(shape, dtype=dtype or np.float64)

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """Create ones array on device."""
        return self.xp.ones(shape, dtype=dtype or np.float64)


def _get_default_dtype_policy() -> DTypePolicy:
    """Get default dtype policy from environment or defaults."""
    dtype = os.environ.get("FF_M5_DTYPE", "float64")
    allow_cast_str = os.environ.get("FF_M5_DTYPE_ALLOW_CAST", "1")
    allow_cast = allow_cast_str not in ("0", "false", "False", "no", "No")
    return DTypePolicy(dtype=dtype, allow_cast=allow_cast)


def _check_cupy_available(cp: Any) -> bool:
    """Check if CuPy CUDA is available."""
    try:
        cuda = getattr(cp, "cuda", None)
        if cuda is None:
            return False
        is_available = getattr(cuda, "is_available", None)
        if callable(is_available):
            return is_available()
        return True
    except Exception:
        return False


def select_array_backend(config: BackendConfig | None = None) -> ArrayBackend:
    """Select the appropriate array backend based on configuration.

    Priority:
    1. FF_M5_BACKEND environment variable
    2. config.backend parameter
    3. config.prefer_gpu / config.require_gpu settings
    4. Default: prefer GPU if available

    Args:
        config: Backend configuration

    Returns:
        ArrayBackend instance for CPU or GPU

    Raises:
        ValueError: If FF_M5_BACKEND is set to an unsupported value
        GPUNotAvailableError: If GPU is required but not available
    """
    if config is None:
        config = BackendConfig()

    # Get dtype policy
    dtype_policy = config.dtype_policy
    if dtype_policy is None:
        dtype_policy = _get_default_dtype_policy()

    # Check environment variable first
    env_backend = os.environ.get("FF_M5_BACKEND")
    forced_backend = env_backend or config.backend

    if forced_backend:
        if forced_backend == "numpy":
            # Force numpy
            xp = sys.modules.get("numpy", np)
            return ArrayBackend(
                device_type=DeviceType.CPU,
                xp=xp,
                device_id=None,
                require_gpu=False,
                dtype_policy=dtype_policy,
            )
        elif forced_backend == "cupy":
            # Force cupy - must be available
            try:
                cp = sys.modules.get("cupy")
                if cp is None:
                    import cupy as cp
            except ImportError:
                raise GPUNotAvailableError("GPU backend 'cupy' required but CuPy not importable")

            if not _check_cupy_available(cp):
                raise GPUNotAvailableError("GPU backend 'cupy' required but CUDA not available")

            return ArrayBackend(
                device_type=DeviceType.GPU,
                xp=cp,
                device_id=config.device_id,
                require_gpu=True,
                dtype_policy=dtype_policy,
            )
        else:
            raise ValueError(f"Unsupported backend: {forced_backend}")

    # No forced backend - use config preferences
    if config.require_gpu or config.prefer_gpu:
        try:
            cp = sys.modules.get("cupy")
            if cp is None:
                import cupy as cp

            if _check_cupy_available(cp):
                return ArrayBackend(
                    device_type=DeviceType.GPU,
                    xp=cp,
                    device_id=config.device_id,
                    require_gpu=config.require_gpu,
                    dtype_policy=dtype_policy,
                )
            elif config.require_gpu:
                raise GPUNotAvailableError("GPU backend required but CUDA not available")
        except ImportError:
            if config.require_gpu:
                raise GPUNotAvailableError("GPU backend required but CuPy not available")

    # Fallback to numpy
    xp = sys.modules.get("numpy", np)
    return ArrayBackend(
        device_type=DeviceType.CPU,
        xp=xp,
        device_id=None,
        require_gpu=config.require_gpu,
        dtype_policy=dtype_policy,
    )


__all__ = [
    "ArrayBackend",
    "BackendConfig",
    "DeviceType",
    "DTypePolicy",
    "select_array_backend",
]
