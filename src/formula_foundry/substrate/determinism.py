from __future__ import annotations

import importlib
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

DeterminismMode = Literal["strict", "fast"]

VALID_CUBLAS_WORKSPACE_CONFIGS = {":4096:8", ":16:8"}
DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


@dataclass(frozen=True)
class DeterminismConfig:
    mode: DeterminismMode
    seed: int
    cublas_workspace_config: str | None

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "seeds": {
                "python": self.seed,
                "numpy": self.seed,
                "cupy": self.seed,
                "torch": self.seed,
            },
            "cublas_workspace_config": self.cublas_workspace_config,
        }


@dataclass
class _Snapshot:
    env_pythonhashseed: str | None
    env_cublas_workspace_config: str | None
    python_state: tuple[Any, ...]
    numpy_state: object | None
    cupy_state: object | None
    torch_state: object | None
    torch_cuda_state: object | None
    torch_det_algos: bool | None
    torch_cudnn_deterministic: bool | None
    torch_cudnn_benchmark: bool | None


def determinism_manifest(
    mode: DeterminismMode,
    seed: int,
    *,
    cublas_workspace_config: str | None = None,
) -> dict[str, Any]:
    config = _build_config(mode, seed, cublas_workspace_config)
    return config.manifest_entry()


def apply_determinism(
    mode: DeterminismMode,
    seed: int,
    *,
    cublas_workspace_config: str | None = None,
) -> DeterminismConfig:
    config = _build_config(mode, seed, cublas_workspace_config)
    if mode == "strict":
        _apply_strict(seed, config.cublas_workspace_config)
    else:
        _apply_fast(seed)
    return config


@contextmanager
def determinism_context(
    mode: DeterminismMode,
    seed: int,
    *,
    cublas_workspace_config: str | None = None,
) -> Iterator[DeterminismConfig]:
    snapshot = _capture_snapshot()
    config = apply_determinism(mode, seed, cublas_workspace_config=cublas_workspace_config)
    try:
        yield config
    finally:
        _restore_snapshot(snapshot)


def _build_config(
    mode: DeterminismMode,
    seed: int,
    cublas_workspace_config: str | None,
) -> DeterminismConfig:
    if mode not in ("strict", "fast"):
        raise ValueError(f"Unknown determinism mode: {mode}")
    resolved_cublas: str | None = None
    if mode == "strict":
        resolved_cublas = _resolve_cublas_workspace_config(cublas_workspace_config)
    elif cublas_workspace_config is not None:
        _validate_cublas_workspace_config(cublas_workspace_config)
        resolved_cublas = cublas_workspace_config
    return DeterminismConfig(mode=mode, seed=seed, cublas_workspace_config=resolved_cublas)


def _apply_strict(seed: int, cublas_workspace_config: str | None) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cublas_workspace_config is None:
        raise ValueError("strict mode requires a valid CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
    _seed_everything(seed)
    torch_module = _import_optional("torch")
    if torch_module is not None:
        _set_torch_determinism(torch_module, enabled=True)


def _apply_fast(seed: int) -> None:
    _seed_everything(seed)
    torch_module = _import_optional("torch")
    if torch_module is not None:
        _set_torch_determinism(torch_module, enabled=False)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    numpy_module = _import_optional("numpy")
    if numpy_module is not None:
        _set_numpy_seed(numpy_module, seed)
    cupy_module = _import_optional("cupy")
    if cupy_module is not None:
        _set_cupy_seed(cupy_module, seed)
    torch_module = _import_optional("torch")
    if torch_module is not None:
        _set_torch_seed(torch_module, seed)


def _set_numpy_seed(numpy_module: Any, seed: int) -> None:
    random_module = getattr(numpy_module, "random", None)
    if random_module is None:
        return
    seed_fn = getattr(random_module, "seed", None)
    if seed_fn is None:
        return
    seed_fn(seed)


def _set_cupy_seed(cupy_module: Any, seed: int) -> None:
    random_module = getattr(cupy_module, "random", None)
    if random_module is None:
        return
    seed_fn = getattr(random_module, "seed", None)
    if seed_fn is None:
        return
    seed_fn(seed)


def _set_torch_seed(torch_module: Any, seed: int) -> None:
    manual_seed = getattr(torch_module, "manual_seed", None)
    if manual_seed is not None:
        manual_seed(seed)
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is None:
        return
    manual_seed_all = getattr(cuda_module, "manual_seed_all", None)
    if manual_seed_all is not None:
        manual_seed_all(seed)


def _set_torch_determinism(torch_module: Any, *, enabled: bool) -> None:
    use_deterministic_algorithms = getattr(torch_module, "use_deterministic_algorithms", None)
    if use_deterministic_algorithms is not None:
        use_deterministic_algorithms(enabled)
    cudnn_module = _get_torch_cudnn(torch_module)
    if cudnn_module is None:
        return
    cudnn_module.deterministic = enabled
    cudnn_module.benchmark = not enabled


def _resolve_cublas_workspace_config(value: str | None) -> str:
    if value is None:
        value = DEFAULT_CUBLAS_WORKSPACE_CONFIG
    _validate_cublas_workspace_config(value)
    return value


def _validate_cublas_workspace_config(value: str) -> None:
    if value not in VALID_CUBLAS_WORKSPACE_CONFIGS:
        raise ValueError(f"Invalid CUBLAS_WORKSPACE_CONFIG {value!r}; expected one of {sorted(VALID_CUBLAS_WORKSPACE_CONFIGS)}")


def _capture_snapshot() -> _Snapshot:
    numpy_module = _import_optional("numpy")
    cupy_module = _import_optional("cupy")
    torch_module = _import_optional("torch")
    return _Snapshot(
        env_pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        env_cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        python_state=random.getstate(),
        numpy_state=_get_numpy_state(numpy_module),
        cupy_state=_get_cupy_state(cupy_module),
        torch_state=_get_torch_state(torch_module),
        torch_cuda_state=_get_torch_cuda_state(torch_module),
        torch_det_algos=_get_torch_det_algos(torch_module),
        torch_cudnn_deterministic=_get_torch_cudnn_attr(torch_module, "deterministic"),
        torch_cudnn_benchmark=_get_torch_cudnn_attr(torch_module, "benchmark"),
    )


def _restore_snapshot(snapshot: _Snapshot) -> None:
    _set_env_var("PYTHONHASHSEED", snapshot.env_pythonhashseed)
    _set_env_var("CUBLAS_WORKSPACE_CONFIG", snapshot.env_cublas_workspace_config)
    random.setstate(snapshot.python_state)
    numpy_module = _import_optional("numpy")
    if numpy_module is not None and snapshot.numpy_state is not None:
        _set_numpy_state(numpy_module, snapshot.numpy_state)
    cupy_module = _import_optional("cupy")
    if cupy_module is not None and snapshot.cupy_state is not None:
        _set_cupy_state(cupy_module, snapshot.cupy_state)
    torch_module = _import_optional("torch")
    if torch_module is not None:
        _set_torch_state(torch_module, snapshot.torch_state)
        _set_torch_cuda_state(torch_module, snapshot.torch_cuda_state)
        if snapshot.torch_det_algos is not None:
            use_deterministic_algorithms = getattr(torch_module, "use_deterministic_algorithms", None)
            if use_deterministic_algorithms is not None:
                use_deterministic_algorithms(snapshot.torch_det_algos)
        _set_torch_cudnn_attr(torch_module, "deterministic", snapshot.torch_cudnn_deterministic)
        _set_torch_cudnn_attr(torch_module, "benchmark", snapshot.torch_cudnn_benchmark)


def _set_env_var(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _get_numpy_state(numpy_module: Any | None) -> object | None:
    if numpy_module is None:
        return None
    random_module = getattr(numpy_module, "random", None)
    if random_module is None:
        return None
    get_state = getattr(random_module, "get_state", None)
    if get_state is None:
        return None
    return cast(object, get_state())


def _set_numpy_state(numpy_module: Any, state: object) -> None:
    random_module = getattr(numpy_module, "random", None)
    if random_module is None:
        return
    set_state = getattr(random_module, "set_state", None)
    if set_state is None:
        return
    set_state(state)


def _get_cupy_state(cupy_module: Any | None) -> object | None:
    if cupy_module is None:
        return None
    random_module = getattr(cupy_module, "random", None)
    if random_module is None:
        return None
    get_random_state = getattr(random_module, "get_random_state", None)
    if get_random_state is None:
        return None
    rng = get_random_state()
    get_state = getattr(rng, "get_state", None)
    if get_state is None:
        return None
    return cast(object, get_state())


def _set_cupy_state(cupy_module: Any, state: object) -> None:
    random_module = getattr(cupy_module, "random", None)
    if random_module is None:
        return
    get_random_state = getattr(random_module, "get_random_state", None)
    if get_random_state is None:
        return
    rng = get_random_state()
    set_state = getattr(rng, "set_state", None)
    if set_state is None:
        return
    set_state(state)


def _get_torch_state(torch_module: Any | None) -> object | None:
    if torch_module is None:
        return None
    get_rng_state = getattr(torch_module, "get_rng_state", None)
    if get_rng_state is None:
        random_module = getattr(torch_module, "random", None)
        get_rng_state = getattr(random_module, "get_rng_state", None) if random_module is not None else None
    if get_rng_state is None:
        return None
    return cast(object, get_rng_state())


def _set_torch_state(torch_module: Any, state: object | None) -> None:
    if state is None:
        return
    set_rng_state = getattr(torch_module, "set_rng_state", None)
    if set_rng_state is None:
        random_module = getattr(torch_module, "random", None)
        set_rng_state = getattr(random_module, "set_rng_state", None) if random_module is not None else None
    if set_rng_state is None:
        return
    set_rng_state(state)


def _get_torch_cuda_state(torch_module: Any | None) -> object | None:
    if torch_module is None:
        return None
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is None:
        return None
    get_rng_state_all = getattr(cuda_module, "get_rng_state_all", None)
    if get_rng_state_all is None:
        return None
    return cast(object, get_rng_state_all())


def _set_torch_cuda_state(torch_module: Any, state: object | None) -> None:
    if state is None:
        return
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is None:
        return
    set_rng_state_all = getattr(cuda_module, "set_rng_state_all", None)
    if set_rng_state_all is None:
        return
    set_rng_state_all(state)


def _get_torch_det_algos(torch_module: Any | None) -> bool | None:
    if torch_module is None:
        return None
    checker = getattr(torch_module, "are_deterministic_algorithms_enabled", None)
    if checker is None:
        return None
    return cast(bool, checker())


def _get_torch_cudnn(torch_module: Any | None) -> Any | None:
    if torch_module is None:
        return None
    backends = getattr(torch_module, "backends", None)
    if backends is None:
        return None
    return getattr(backends, "cudnn", None)


def _get_torch_cudnn_attr(torch_module: Any | None, attr: str) -> bool | None:
    cudnn_module = _get_torch_cudnn(torch_module)
    if cudnn_module is None:
        return None
    if not hasattr(cudnn_module, attr):
        return None
    value = getattr(cudnn_module, attr)
    if not isinstance(value, bool):
        return None
    return value


def _set_torch_cudnn_attr(torch_module: Any | None, attr: str, value: bool | None) -> None:
    if value is None:
        return
    cudnn_module = _get_torch_cudnn(torch_module)
    if cudnn_module is None or not hasattr(cudnn_module, attr):
        return
    setattr(cudnn_module, attr, value)


def _import_optional(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:
        return None
