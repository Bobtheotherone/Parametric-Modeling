from __future__ import annotations

import types

from formula_foundry.postprocess import gpu_backend


def test_gpu_backend_defaults_to_cupy_and_records_no_fallback(monkeypatch) -> None:
    fake_cupy = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    fake_numpy = object()

    def fake_import(name: str):
        if name == "cupy":
            return fake_cupy
        if name == "numpy":
            return fake_numpy
        return None

    monkeypatch.setattr(gpu_backend, "_import_optional", fake_import)
    backend = gpu_backend.select_gpu_backend()

    assert backend.name == "cupy"
    assert backend.fallback_reason is None
    assert backend.module is fake_cupy


def test_gpu_backend_falls_back_to_numpy_with_reason(monkeypatch) -> None:
    fake_numpy = object()

    def fake_import(name: str):
        if name == "cupy":
            return None
        if name == "numpy":
            return fake_numpy
        return None

    monkeypatch.setattr(gpu_backend, "_import_optional", fake_import)
    backend = gpu_backend.select_gpu_backend()

    assert backend.name == "numpy"
    assert backend.fallback_reason == "cupy not installed"
    assert backend.module is fake_numpy
