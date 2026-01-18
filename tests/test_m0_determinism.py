from __future__ import annotations

import os
import random
import sys

import pytest

from formula_foundry.substrate import determinism


class _FakeNumpyRandom:
    def __init__(self) -> None:
        self.seed_calls: list[int] = []
        self._state: object = ("numpy", "initial")

    def seed(self, seed: int) -> None:
        self.seed_calls.append(seed)
        self._state = ("numpy", seed)

    def get_state(self) -> object:
        return self._state

    def set_state(self, state: object) -> None:
        self._state = state


class _FakeNumpy:
    def __init__(self) -> None:
        self.random = _FakeNumpyRandom()


class _FakeCupyRandom:
    def __init__(self) -> None:
        self.seed_calls: list[int] = []
        self._state: object = ("cupy", "initial")

    def seed(self, seed: int) -> None:
        self.seed_calls.append(seed)
        self._state = ("cupy", seed)

    def get_random_state(self) -> _FakeCupyRandom:
        return self

    def get_state(self) -> object:
        return self._state

    def set_state(self, state: object) -> None:
        self._state = state


class _FakeCupy:
    def __init__(self) -> None:
        self.random = _FakeCupyRandom()


class _FakeCudnn:
    def __init__(self) -> None:
        self.deterministic = False
        self.benchmark = True


class _FakeBackends:
    def __init__(self) -> None:
        self.cudnn = _FakeCudnn()


class _FakeCuda:
    def __init__(self) -> None:
        self.seed_calls: list[int] = []
        self._rng_state: object = ("cuda", "initial")

    def manual_seed_all(self, seed: int) -> None:
        self.seed_calls.append(seed)

    def get_rng_state_all(self) -> object:
        return self._rng_state

    def set_rng_state_all(self, state: object) -> None:
        self._rng_state = state


class _FakeTorch:
    def __init__(self) -> None:
        self.seed_calls: list[int] = []
        self._deterministic = False
        self._rng_state: object = ("torch", "initial")
        self.backends = _FakeBackends()
        self.cuda = _FakeCuda()

    def manual_seed(self, seed: int) -> None:
        self.seed_calls.append(seed)

    def use_deterministic_algorithms(self, enabled: bool) -> None:
        self._deterministic = enabled

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self._deterministic

    def get_rng_state(self) -> object:
        return self._rng_state

    def set_rng_state(self, state: object) -> None:
        self._rng_state = state


def test_determinism_modes_recorded() -> None:
    strict_manifest = determinism.determinism_manifest("strict", seed=123)
    assert strict_manifest["mode"] == "strict"
    assert strict_manifest["seeds"]["python"] == 123
    assert strict_manifest["seeds"]["numpy"] == 123
    assert strict_manifest["seeds"]["cupy"] == 123
    assert strict_manifest["seeds"]["torch"] == 123
    assert strict_manifest["cublas_workspace_config"] == determinism.DEFAULT_CUBLAS_WORKSPACE_CONFIG

    fast_manifest = determinism.determinism_manifest("fast", seed=7)
    assert fast_manifest["mode"] == "fast"
    assert fast_manifest["cublas_workspace_config"] is None


def test_strict_mode_sets_required_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_numpy = _FakeNumpy()
    fake_cupy = _FakeCupy()
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    config = determinism.apply_determinism("strict", seed=11)

    assert config.mode == "strict"
    assert os.environ["PYTHONHASHSEED"] == "11"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == determinism.DEFAULT_CUBLAS_WORKSPACE_CONFIG
    assert fake_numpy.random.seed_calls == [11]
    assert fake_cupy.random.seed_calls == [11]
    assert fake_torch.seed_calls == [11]
    assert fake_torch.cuda.seed_calls == [11]
    assert fake_torch.backends.cudnn.deterministic is True
    assert fake_torch.backends.cudnn.benchmark is False
    assert fake_torch.are_deterministic_algorithms_enabled() is True


def test_determinism_context_manager_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_numpy = _FakeNumpy()
    fake_cupy = _FakeCupy()
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("PYTHONHASHSEED", "orig")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    random.seed(1234)
    python_state = random.getstate()
    numpy_state = fake_numpy.random.get_state()
    cupy_state = fake_cupy.random.get_state()
    torch_state = fake_torch.get_rng_state()
    cuda_state = fake_torch.cuda.get_rng_state_all()
    torch_det = fake_torch.are_deterministic_algorithms_enabled()
    cudnn_det = fake_torch.backends.cudnn.deterministic
    cudnn_benchmark = fake_torch.backends.cudnn.benchmark

    with determinism.determinism_context("strict", seed=99):
        assert os.environ["PYTHONHASHSEED"] == "99"
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == determinism.DEFAULT_CUBLAS_WORKSPACE_CONFIG
        assert fake_torch.backends.cudnn.deterministic is True
        assert fake_torch.backends.cudnn.benchmark is False

    assert os.environ["PYTHONHASHSEED"] == "orig"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert random.getstate() == python_state
    assert fake_numpy.random.get_state() == numpy_state
    assert fake_cupy.random.get_state() == cupy_state
    assert fake_torch.get_rng_state() == torch_state
    assert fake_torch.cuda.get_rng_state_all() == cuda_state
    assert fake_torch.are_deterministic_algorithms_enabled() == torch_det
    assert fake_torch.backends.cudnn.deterministic == cudnn_det
    assert fake_torch.backends.cudnn.benchmark == cudnn_benchmark
