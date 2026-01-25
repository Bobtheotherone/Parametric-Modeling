"""M5 microbench CLI wrapper.

Provides a CLI interface for running M5 evaluator throughput benchmarks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from formula_foundry.m5.backend import BackendConfig, ChunkedEvaluator, ChunkingPolicy, select_array_backend


@dataclass(frozen=True)
class BenchConfig:
    """Configuration for M5 microbench."""

    n_params: int = 256
    n_freqs: int = 512
    n_features: int = 4
    repeats: int = 5
    warmup: int = 1
    chunk_n: int | None = None
    chunk_f: int | None = None
    max_elements: int | None = 1_000_000
    require_gpu: bool = False
    prefer_gpu: bool = True
    min_elements_per_s: float = 1.0

    def chunking_policy(self) -> ChunkingPolicy:
        """Build chunking policy."""
        return ChunkingPolicy(max_elements=self.max_elements, chunk_n=self.chunk_n, chunk_f=self.chunk_f)


def run_microbench(config: BenchConfig) -> dict[str, Any]:
    """Run M5 microbench and return report."""
    backend = select_array_backend(BackendConfig(require_gpu=config.require_gpu, prefer_gpu=config.prefer_gpu))
    xp = backend.xp
    dtype = _resolve_dtype(xp)
    params = xp.arange(config.n_params * config.n_features, dtype=dtype).reshape(config.n_params, config.n_features)
    freqs = xp.linspace(0.0, 1.0, config.n_freqs, dtype=dtype)

    kernel_launches = [0]

    def eval_fn(p_chunk: Any, f_chunk: Any) -> Any:
        kernel_launches[0] += 1
        base = p_chunk[:, :1]
        if p_chunk.shape[1] > 1:
            base = base + p_chunk[:, 1:2] * 0.001
        return base * (f_chunk[None, :] + 1.0)

    evaluator = ChunkedEvaluator(backend, eval_fn, chunking=config.chunking_policy())

    def run_once() -> Any:
        result = evaluator.evaluate(params, freqs)
        return result.sum()

    elapsed = _bench_loop(run_once, repeats=config.repeats, warmup=config.warmup, xp=xp)
    total_elements = config.n_params * config.n_freqs * config.repeats
    throughput = total_elements / elapsed if elapsed > 0 else float("inf")

    status = "pass" if throughput >= config.min_elements_per_s else "fail"

    return {
        "bench": "m5-microbench",
        "status": status,
        "duration_s": elapsed,
        "total_elements": total_elements,
        "throughput_elements_per_s": throughput,
        "kernel_launches": kernel_launches[0],
        "backend": _backend_payload(backend),
        "config": _config_payload(config),
        "cpu_fallback_warnings": [],
        "gpu_assertions": [],
        "gpu_metrics": None if backend.name == "numpy" else {},
        "thresholds": {"min_elements_per_s": config.min_elements_per_s},
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run M5 microbench CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = BenchConfig(
        n_params=args.n_params,
        n_freqs=args.n_freqs,
        n_features=args.n_features,
        repeats=args.repeats,
        warmup=args.warmup,
        chunk_n=args.chunk_n,
        chunk_f=args.chunk_f,
        max_elements=args.max_elements,
        require_gpu=args.require_gpu,
        prefer_gpu=not args.cpu_only,
        min_elements_per_s=args.min_elements_per_s,
    )

    try:
        report = run_microbench(config)
        _emit_json(report, args.json)
        return 0 if report["status"] == "pass" else 1
    except Exception as exc:
        error_report: dict[str, Any] = {
            "bench": "m5-microbench",
            "status": "error",
            "error": str(exc),
            "config": _config_payload(config),
        }
        _emit_json(error_report, args.json)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(description="M5 microbench")
    parser.add_argument("--n-params", type=int, default=256)
    parser.add_argument("--n-freqs", type=int, default=512)
    parser.add_argument("--n-features", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--chunk-n", type=int, default=None)
    parser.add_argument("--chunk-f", type=int, default=None)
    parser.add_argument("--max-elements", type=int, default=1_000_000)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--min-elements-per-s", type=float, default=1.0)
    parser.add_argument("--json", "-j", default="-")
    return parser


def _bench_loop(run_once: Callable[[], Any], *, repeats: int, warmup: int, xp: Any) -> float:
    """Run benchmark loop with warmup."""
    if repeats <= 0:
        raise ValueError("repeats must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    for _ in range(warmup):
        run_once()
        _maybe_synchronize(xp)
    start = time.perf_counter()
    for _ in range(repeats):
        run_once()
        _maybe_synchronize(xp)
    return max(time.perf_counter() - start, 1e-12)


def _resolve_dtype(xp: Any) -> Any:
    """Resolve float32 dtype for array module."""
    return getattr(xp, "float32", None) or "float32"


def _maybe_synchronize(xp: Any) -> None:
    """Synchronize CUDA stream if using cupy."""
    cuda_module = getattr(xp, "cuda", None)
    if cuda_module is None:
        return
    stream = getattr(cuda_module, "Stream", None)
    if stream is None:
        return
    null_stream = getattr(stream, "null", None)
    if null_stream is None:
        return
    sync = getattr(null_stream, "synchronize", None)
    if callable(sync):
        sync()


def _backend_payload(backend: Any) -> dict[str, Any]:
    """Build backend info payload."""
    return {
        "name": backend.name,
        "device_type": "cpu" if backend.name == "numpy" else "gpu",
        "gpu_available": backend.gpu_available,
        "require_gpu": backend.require_gpu,
    }


def _config_payload(config: BenchConfig) -> dict[str, Any]:
    """Build config payload."""
    return {
        "n_params": config.n_params,
        "n_freqs": config.n_freqs,
        "n_features": config.n_features,
        "repeats": config.repeats,
        "warmup": config.warmup,
        "chunk_n": config.chunk_n,
        "chunk_f": config.chunk_f,
        "max_elements": config.max_elements,
        "require_gpu": config.require_gpu,
        "prefer_gpu": config.prefer_gpu,
    }


def _emit_json(report: dict[str, Any], target: str) -> None:
    """Emit JSON report to file or stdout."""
    text = json.dumps(report, indent=2, sort_keys=True)
    if target == "-":
        print(text)
        return
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(f"{text}\n")


if __name__ == "__main__":
    sys.exit(main())
