from __future__ import annotations

import sys

import pytest

from formula_foundry.substrate import backends


class _Storage:
    def __init__(self) -> None:
        self.token = object()


class _FakeDLPack:
    def __init__(self, storage: _Storage) -> None:
        self.storage = storage


class _FakeCupyArray:
    def __init__(self, storage: _Storage) -> None:
        self.storage = storage

    def __dlpack__(self) -> _FakeDLPack:
        return _FakeDLPack(self.storage)


class _FakeTorchTensor:
    def __init__(self, storage: _Storage) -> None:
        self.storage = storage

    def __dlpack__(self) -> _FakeDLPack:
        return _FakeDLPack(self.storage)


class _FakeTorchDlpack:
    def from_dlpack(self, capsule: _FakeDLPack) -> _FakeTorchTensor:
        return _FakeTorchTensor(capsule.storage)

    def to_dlpack(self, tensor: _FakeTorchTensor) -> _FakeDLPack:
        return _FakeDLPack(tensor.storage)


class _FakeTorchUtils:
    def __init__(self) -> None:
        self.dlpack = _FakeTorchDlpack()


class _FakeTorch:
    def __init__(self) -> None:
        self.utils = _FakeTorchUtils()
        self.Tensor = _FakeTorchTensor


class _FakeCupyNdarray:
    get_calls: list[str] = []

    def get(self) -> str:
        self.__class__.get_calls.append("get")
        return "host"


class _FakeCupy:
    def __init__(self) -> None:
        self.from_dlpack_calls: list[_FakeDLPack] = []
        self.asnumpy_calls: list[str] = []
        self.ndarray = _FakeCupyNdarray

    def fromDlpack(self, capsule: _FakeDLPack | _FakeTorchTensor | _FakeCupyArray) -> _FakeCupyArray:
        if hasattr(capsule, "__dlpack__"):
            dlpack_fn = capsule.__dlpack__
            dlpack = dlpack_fn()
        else:
            dlpack = capsule
        self.from_dlpack_calls.append(dlpack)
        return _FakeCupyArray(dlpack.storage)

    def asnumpy(self, value: str) -> str:
        self.asnumpy_calls.append(value)
        return "host"


class _FakeTensor:
    cpu_calls: list[str] = []
    numpy_calls: list[str] = []

    def cpu(self) -> _FakeTensor:
        self.__class__.cpu_calls.append("cpu")
        return self

    def numpy(self) -> str:
        self.__class__.numpy_calls.append("numpy")
        return "host"


class _FakeTorchHost:
    def __init__(self) -> None:
        self.Tensor = _FakeTensor


def test_dlpack_roundtrip_zero_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cupy = _FakeCupy()
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    storage = _Storage()
    cupy_array = _FakeCupyArray(storage)

    torch_tensor = backends.cupy_to_torch(cupy_array)
    assert isinstance(torch_tensor, _FakeTorchTensor)
    assert torch_tensor.storage is storage

    cupy_roundtrip = backends.torch_to_cupy(torch_tensor)
    assert isinstance(cupy_roundtrip, _FakeCupyArray)
    assert cupy_roundtrip.storage is storage


def test_host_transfer_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cupy = _FakeCupy()
    fake_torch = _FakeTorchHost()
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    array = fake_cupy.ndarray()
    tensor = fake_torch.Tensor()

    with backends.host_transfer_guard(fail_on_transfer=False) as guard:
        assert fake_cupy.asnumpy("payload") == "host"
        assert array.get() == "host"
        assert tensor.cpu() is tensor
        assert tensor.numpy() == "host"

        assert "cupy.asnumpy" in guard.transfers
        assert "cupy.ndarray.get" in guard.transfers
        assert "torch.Tensor.cpu" in guard.transfers
        assert "torch.Tensor.numpy" in guard.transfers

    with backends.host_transfer_guard(fail_on_transfer=True) as guard:
        with pytest.raises(backends.HostTransferError, match="Host transfer"):
            fake_cupy.asnumpy("payload")
        assert guard.transfers == ["cupy.asnumpy"]
