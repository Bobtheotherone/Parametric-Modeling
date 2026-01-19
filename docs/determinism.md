# Determinism Guarantees and Limitations (M0)

## What strict mode guarantees

When `tools.m0` runs in `--mode strict`, the substrate guarantees the following within a
single machine and driver stack:

- `PYTHONHASHSEED` is set and recorded.
- Seeds are applied to Python `random`, NumPy, CuPy, and Torch.
- Torch deterministic algorithms are enabled.
- cuDNN benchmarking is disabled and deterministic mode is enabled.
- `CUBLAS_WORKSPACE_CONFIG` is set to a deterministic-safe value.
- The determinism mode and seed values are recorded in each run manifest.

## Limitations and known sources of nondeterminism

Even in strict mode, some GPU edge cases can still introduce nondeterminism:

- Different GPU models or driver versions can change low-level kernel behavior.
- Certain CUDA/cuDNN kernels use atomics that are nondeterministic by design.
- Compiler/runtime upgrades (CUDA, cuDNN, Torch) can change numerics.
- Mixed precision and reduced-precision kernels can amplify rounding variance.

Strict mode is designed for repeatability on the target laptop and driver stack; it does
not guarantee bit-for-bit identity across different hardware or driver versions.
