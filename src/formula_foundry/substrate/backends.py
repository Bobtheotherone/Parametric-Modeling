from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

BackendName = Literal["numpy", "cupy"]


class BackendSelectionError(RuntimeError):
    """Raised when no suitable numeric backend can be selected."""


class GPUNotAvailableError(BackendSelectionError):
    """Raised when GPU execution is required but unavailable."""


class HostTransferError(RuntimeError):
    """Raised when an unexpected host transfer occurs."""


@dataclass(frozen=True)
class Backend:
    name: BackendName
    module: Any
    gpu_available: bool
    require_gpu: bool


def select_backend(*, require_gpu: bool = False, prefer_gpu: bool = True) -> Backend:
    if require_gpu and not prefer_gpu:
        raise GPUNotAvailableError("GPU required but CPU-only backend requested (prefer_gpu=False).")
    cupy_module = _import_optional("cupy")
    gpu_available = _cupy_cuda_available(cupy_module)
    if prefer_gpu and gpu_available and cupy_module is not None:
        return Backend(name="cupy", module=cupy_module, gpu_available=True, require_gpu=require_gpu)
    if require_gpu:
        raise GPUNotAvailableError("GPU required but no CUDA-enabled backend is available.")
    numpy_module = _import_optional("numpy")
    if numpy_module is None:
        raise BackendSelectionError("NumPy backend unavailable; install numpy or enable GPU backend.")
    return Backend(name="numpy", module=numpy_module, gpu_available=gpu_available, require_gpu=require_gpu)


def cupy_to_torch(cupy_array: Any) -> Any:
    torch_module = _import_optional("torch")
    if torch_module is None:
        raise BackendSelectionError("torch is required for DLPack interop.")
    dlpack_module = _get_torch_dlpack(torch_module)
    from_dlpack = getattr(dlpack_module, "from_dlpack", None) if dlpack_module is not None else None
    if from_dlpack is None:
        raise BackendSelectionError("torch.utils.dlpack.from_dlpack is unavailable.")
    return from_dlpack(cupy_array)


def torch_to_cupy(torch_tensor: Any) -> Any:
    cupy_module = _import_optional("cupy")
    if cupy_module is None:
        raise BackendSelectionError("cupy is required for DLPack interop.")
    from_dlpack = _get_cupy_from_dlpack(cupy_module)
    if from_dlpack is None:
        raise BackendSelectionError("cupy.fromDlpack is unavailable.")
    if hasattr(torch_tensor, "__dlpack__"):
        return from_dlpack(torch_tensor)
    torch_module = _import_optional("torch")
    dlpack_module = _get_torch_dlpack(torch_module) if torch_module is not None else None
    to_dlpack = getattr(dlpack_module, "to_dlpack", None) if dlpack_module is not None else None
    if to_dlpack is None:
        raise BackendSelectionError("torch.utils.dlpack.to_dlpack is unavailable.")
    return from_dlpack(to_dlpack(torch_tensor))


@dataclass
class HostTransferGuard:
    fail_on_transfer: bool = True
    transfers: list[str] = field(default_factory=list)
    _patches: list[_Patch] = field(default_factory=list, init=False)

    def install(self) -> None:
        cupy_module = _import_optional("cupy")
        if cupy_module is not None:
            self._wrap(cupy_module, "asnumpy", "cupy.asnumpy")
            ndarray = getattr(cupy_module, "ndarray", None)
            if ndarray is not None:
                self._wrap(ndarray, "get", "cupy.ndarray.get")
        torch_module = _import_optional("torch")
        if torch_module is not None:
            tensor_cls = getattr(torch_module, "Tensor", None)
            if tensor_cls is not None:
                self._wrap(tensor_cls, "cpu", "torch.Tensor.cpu")
                self._wrap(tensor_cls, "numpy", "torch.Tensor.numpy")

    def uninstall(self) -> None:
        for patch in reversed(self._patches):
            try:
                setattr(patch.owner, patch.attr, patch.original)
            except Exception:
                continue
        self._patches.clear()

    def record(self, label: str) -> None:
        self.transfers.append(label)
        if self.fail_on_transfer:
            raise HostTransferError(f"Host transfer detected: {label}")

    def _wrap(self, owner: Any, attr: str, label: str) -> None:
        original = getattr(owner, attr, None)
        if original is None:
            return

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.record(label)
            return original(*args, **kwargs)

        if not _try_setattr(owner, attr, wrapper):
            return
        self._patches.append(_Patch(owner=owner, attr=attr, original=original))


@contextmanager
def host_transfer_guard(*, fail_on_transfer: bool = True) -> Iterator[HostTransferGuard]:
    guard = HostTransferGuard(fail_on_transfer=fail_on_transfer)
    guard.install()
    try:
        yield guard
    finally:
        guard.uninstall()


@dataclass(frozen=True)
class _Patch:
    owner: Any
    attr: str
    original: Any


def _cupy_cuda_available(cupy_module: Any | None) -> bool:
    if cupy_module is None:
        return False
    cuda_module = getattr(cupy_module, "cuda", None)
    if cuda_module is None:
        return False
    is_available = getattr(cuda_module, "is_available", None)
    if callable(is_available):
        try:
            return bool(is_available())
        except Exception:
            return False
    runtime = getattr(cuda_module, "runtime", None)
    get_device_count = getattr(runtime, "getDeviceCount", None) if runtime is not None else None
    if callable(get_device_count):
        try:
            return int(get_device_count()) > 0
        except Exception:
            return False
    return False


def _get_cupy_from_dlpack(cupy_module: Any) -> Any | None:
    from_dlpack = getattr(cupy_module, "fromDlpack", None)
    if from_dlpack is None:
        from_dlpack = getattr(cupy_module, "from_dlpack", None)
    return from_dlpack


def _get_torch_dlpack(torch_module: Any | None) -> Any | None:
    if torch_module is None:
        return None
    utils_module = getattr(torch_module, "utils", None)
    if utils_module is None:
        return None
    return getattr(utils_module, "dlpack", None)


def _try_setattr(owner: Any, attr: str, value: Any) -> bool:
    try:
        setattr(owner, attr, value)
    except Exception:
        return False
    return True


def _import_optional(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:
        return None
