from __future__ import annotations

import sys

import pytest

from formula_foundry.substrate import backends


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeCupy:
    def __init__(self, available: bool) -> None:
        self.cuda = _FakeCuda(available)


class _FakeNumpy:
    pass


def test_backend_defaults_to_gpu_if_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_numpy = _FakeNumpy()
    fake_cupy = _FakeCupy(available=True)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    backend = backends.select_backend()

    assert backend.name == "cupy"
    assert backend.module is fake_cupy
    assert backend.gpu_available is True

    fake_cupy_unavailable = _FakeCupy(available=False)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy_unavailable)

    backend = backends.select_backend()

    assert backend.name == "numpy"
    assert backend.module is fake_numpy
    assert backend.gpu_available is False


def test_require_gpu_mode_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_numpy = _FakeNumpy()
    fake_cupy = _FakeCupy(available=False)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    with pytest.raises(backends.GPUNotAvailableError, match="GPU"):
        backends.select_backend(require_gpu=True)

    fake_cupy_available = _FakeCupy(available=True)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy_available)

    with pytest.raises(backends.GPUNotAvailableError, match="GPU"):
        backends.select_backend(require_gpu=True, prefer_gpu=False)
