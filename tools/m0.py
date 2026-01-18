from __future__ import annotations

import argparse
import importlib
import json
import platform
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from formula_foundry.substrate import backends, determinism, manifest

DEFAULT_SEED = 1234


@dataclass(frozen=True)
class CheckStatus:
    status: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status}
        if self.detail:
            payload["detail"] = self.detail
        return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    run_root = _resolve_run_root(project_root, args.run_root)
    command_line = _build_command_line(argv)

    if args.command == "doctor":
        return _run_doctor(args, project_root, run_root, command_line)
    if args.command == "smoke":
        return _run_smoke(args, project_root, run_root, command_line)
    if args.command == "bench":
        return _run_bench(args, project_root, run_root, command_line)
    if args.command == "repro-check":
        return _run_repro_check(args, project_root, run_root, command_line)

    raise SystemExit(f"Unknown command: {args.command}")


def _build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--project-root", default=".")
    shared.add_argument("--run-root", default="")
    shared.add_argument("--run-id", default="")
    shared.add_argument("--mode", choices=("strict", "fast"), default="strict")
    shared.add_argument("--seed", type=int, default=DEFAULT_SEED)
    shared.add_argument("--require-gpu", action="store_true")

    parser = argparse.ArgumentParser(
        description="Formula Foundry M0 substrate CLI",
        parents=[shared],
        conflict_handler="resolve",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor",
        help="Check CUDA availability and report environment info",
        parents=[shared],
        conflict_handler="resolve",
    )
    doctor.add_argument("--json", nargs="?", const="-", default=None)

    subparsers.add_parser(
        "smoke",
        help="Run fast GPU smoke checks",
        parents=[shared],
        conflict_handler="resolve",
    )

    bench = subparsers.add_parser(
        "bench",
        help="Run microbenchmarks",
        parents=[shared],
        conflict_handler="resolve",
    )
    bench.add_argument("--json", nargs="?", const="default", default=None)

    repro = subparsers.add_parser(
        "repro-check",
        help="Run deterministic repro check",
        parents=[shared],
        conflict_handler="resolve",
    )
    repro.add_argument("--payload-bytes", type=int, default=256)

    return parser


def _resolve_run_root(project_root: Path, run_root: str) -> Path:
    if run_root:
        return Path(run_root).resolve()
    return project_root / "runs"


def _build_command_line(argv: Sequence[str] | None) -> list[str]:
    if argv is None:
        argv = sys.argv[1:]
    return ["m0", *argv]


def _run_doctor(
    args: argparse.Namespace,
    project_root: Path,
    run_root: Path,
    command_line: list[str],
) -> int:
    run_id = _resolve_run_id(args.run_id, "doctor")
    run = manifest.init_run_dir(run_root, run_id)
    report = _doctor_report(require_gpu=args.require_gpu)
    log_event(run.logs_path, "doctor.start", data={"run_id": run_id})
    log_event(run.logs_path, "doctor.report", data=report)
    artifacts: dict[str, str] = {}

    mode = cast(Literal["strict", "fast"], args.mode)
    with determinism.determinism_context(mode, args.seed) as config:
        name, digest = _write_json_artifact(run.artifacts_dir, "doctor_report.json", report)
        artifacts[name] = digest
        run_manifest = manifest.Manifest.from_environment(
            config,
            command_line=command_line,
            artifacts=artifacts,
            project_root=project_root,
        )
        run_manifest.write(run.manifest_path)

    _emit_json_report(report, args.json)

    if args.require_gpu and not report["cuda_available"]:
        log_event(run.logs_path, "doctor.failure", level="error", data={"reason": "GPU required but unavailable"})
        return 2
    return 0


def _run_smoke(
    args: argparse.Namespace,
    project_root: Path,
    run_root: Path,
    command_line: list[str],
) -> int:
    run_id = _resolve_run_id(args.run_id, "smoke")
    run = manifest.init_run_dir(run_root, run_id)
    log_event(run.logs_path, "smoke.start", data={"run_id": run_id})

    report: dict[str, Any] = {"checks": {}}
    failure = False
    artifacts: dict[str, str] = {}

    mode = cast(Literal["strict", "fast"], args.mode)
    with determinism.determinism_context(mode, args.seed) as config:
        backend = None
        try:
            backend = backends.select_backend(require_gpu=args.require_gpu)
            report["backend"] = {"name": backend.name, "gpu_available": backend.gpu_available}
        except backends.GPUNotAvailableError as exc:
            report["backend"] = {"name": "none", "gpu_available": False}
            _set_check(report, "backend", "failed", str(exc))
            failure = True
        except backends.BackendSelectionError as exc:
            report["backend"] = {"name": "none", "gpu_available": False}
            _set_check(report, "backend", "skipped", str(exc))
            failure = failure or args.require_gpu

        if backend is None or backend.name != "cupy":
            _set_check(report, "cupy_op", "skipped", "GPU backend unavailable")
        else:
            try:
                array = backend.module.arange(8)
                _ = array.sum()
                _set_check(report, "cupy_op", "ok")
            except Exception as exc:
                _set_check(report, "cupy_op", "failed", str(exc))
                failure = True

        torch_module = _import_optional("torch")
        torch_cuda_available = _torch_cuda_available(torch_module)

        if torch_module is None:
            status = "failed" if args.require_gpu else "skipped"
            _set_check(report, "torch_cuda", status, "torch not installed")
            failure = failure or args.require_gpu
        elif not torch_cuda_available:
            status = "failed" if args.require_gpu else "skipped"
            _set_check(report, "torch_cuda", status, "torch.cuda unavailable")
            failure = failure or args.require_gpu
        else:
            try:
                _ = torch_module.zeros(1, device="cuda")
                _set_check(report, "torch_cuda", "ok")
            except Exception as exc:
                _set_check(report, "torch_cuda", "failed", str(exc))
                failure = True

        _torch_compile_check(report, torch_module, torch_cuda_available)

        _dlpack_check(report, backend, torch_module, require_gpu=args.require_gpu)
        if report["checks"]["dlpack_roundtrip"]["status"] == "failed":
            failure = True

        name, digest = _write_json_artifact(run.artifacts_dir, "smoke_report.json", report)
        artifacts[name] = digest
        run_manifest = manifest.Manifest.from_environment(
            config,
            command_line=command_line,
            artifacts=artifacts,
            project_root=project_root,
        )
        run_manifest.write(run.manifest_path)

    log_event(run.logs_path, "smoke.complete", data={"failure": failure})
    return 2 if failure else 0


def _run_bench(
    args: argparse.Namespace,
    project_root: Path,
    run_root: Path,
    command_line: list[str],
) -> int:
    run_id = _resolve_run_id(args.run_id, "bench")
    run = manifest.init_run_dir(run_root, run_id)
    log_event(run.logs_path, "bench.start", data={"run_id": run_id})

    try:
        backend = backends.select_backend(require_gpu=args.require_gpu)
    except backends.GPUNotAvailableError as exc:
        report = {
            "backend": {"name": "none", "gpu_available": False},
            "status": "failed",
            "detail": str(exc),
        }
        name, digest = _write_json_artifact(run.artifacts_dir, "bench_report.json", report)
        mode = cast(Literal["strict", "fast"], args.mode)
        with determinism.determinism_context(mode, args.seed) as config:
            run_manifest = manifest.Manifest.from_environment(
                config,
                command_line=command_line,
                artifacts={name: digest},
                project_root=project_root,
            )
            run_manifest.write(run.manifest_path)
        log_event(run.logs_path, "bench.failure", level="error", data={"reason": str(exc)})
        return 2
    except backends.BackendSelectionError as exc:
        report = {
            "backend": {"name": "none", "gpu_available": False},
            "status": "skipped",
            "detail": str(exc),
        }
        name, digest = _write_json_artifact(run.artifacts_dir, "bench_report.json", report)
        mode = cast(Literal["strict", "fast"], args.mode)
        with determinism.determinism_context(mode, args.seed) as config:
            run_manifest = manifest.Manifest.from_environment(
                config,
                command_line=command_line,
                artifacts={name: digest},
                project_root=project_root,
            )
            run_manifest.write(run.manifest_path)
        _emit_json_report(report, args.json, default_path=run.run_dir / "bench.json")
        log_event(run.logs_path, "bench.complete", data={"failure": False, "skipped": True})
        return 0

    artifacts: dict[str, str] = {}
    mode = cast(Literal["strict", "fast"], args.mode)
    with determinism.determinism_context(mode, args.seed) as config:
        report, failure = _run_microbench(backend)
        name, digest = _write_json_artifact(run.artifacts_dir, "bench_report.json", report)
        artifacts[name] = digest
        run_manifest = manifest.Manifest.from_environment(
            config,
            command_line=command_line,
            artifacts=artifacts,
            project_root=project_root,
        )
        run_manifest.write(run.manifest_path)

    _emit_json_report(report, args.json, default_path=run.run_dir / "bench.json")
    log_event(run.logs_path, "bench.complete", data={"failure": failure})
    return 2 if failure else 0


def _run_repro_check(
    args: argparse.Namespace,
    project_root: Path,
    run_root: Path,
    command_line: list[str],
) -> int:
    base_id = _resolve_run_id(args.run_id, "repro")
    run_id_a = f"{base_id}-a"
    run_id_b = f"{base_id}-b"

    artifacts_a = _run_repro_pass(
        run_root,
        run_id_a,
        project_root,
        command_line,
        mode=args.mode,
        seed=args.seed,
        payload_bytes=args.payload_bytes,
    )
    artifacts_b = _run_repro_pass(
        run_root,
        run_id_b,
        project_root,
        command_line,
        mode=args.mode,
        seed=args.seed,
        payload_bytes=args.payload_bytes,
    )

    mismatch = _compare_artifact_hashes(artifacts_a, artifacts_b)
    if mismatch:
        return 2
    return 0


def _run_repro_pass(
    run_root: Path,
    run_id: str,
    project_root: Path,
    command_line: list[str],
    *,
    mode: str,
    seed: int,
    payload_bytes: int,
) -> dict[str, str]:
    run = manifest.init_run_dir(run_root, run_id)
    log_event(run.logs_path, "repro.start", data={"run_id": run_id})
    artifacts: dict[str, str] = {}
    
    safe_mode = cast(Literal["strict", "fast"], mode)
    with determinism.determinism_context(safe_mode, seed) as config:
        payload = _build_repro_payload(payload_bytes)
        name, digest = _write_bytes_artifact(run.artifacts_dir, "repro_payload.bin", payload)
        artifacts[name] = digest
        run_manifest = manifest.Manifest.from_environment(
            config,
            command_line=command_line,
            artifacts=artifacts,
            project_root=project_root,
        )
        run_manifest.write(run.manifest_path)
    log_event(run.logs_path, "repro.complete", data={"artifacts": artifacts})
    return artifacts


def _build_repro_payload(payload_bytes: int) -> bytes:
    return bytes(random.getrandbits(8) for _ in range(payload_bytes))


def _compare_artifact_hashes(a: Mapping[str, str], b: Mapping[str, str]) -> list[str]:
    mismatched: list[str] = []
    keys = set(a) | set(b)
    for key in sorted(keys):
        if a.get(key) != b.get(key):
            mismatched.append(key)
    return mismatched


def _doctor_report(*, require_gpu: bool) -> dict[str, Any]:
    cupy_module = _import_optional("cupy")
    torch_module = _import_optional("torch")
    cupy_available = cupy_module is not None
    torch_available = torch_module is not None
    cupy_cuda_available = _cupy_cuda_available(cupy_module)
    torch_cuda_available = _torch_cuda_available(torch_module)
    cuda_available = cupy_cuda_available or torch_cuda_available

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cupy_available": cupy_available,
        "cupy_cuda_available": cupy_cuda_available,
        "cupy_version": getattr(cupy_module, "__version__", None) if cupy_available else None,
        "torch_available": torch_available,
        "torch_cuda_available": torch_cuda_available,
        "torch_version": getattr(torch_module, "__version__", None) if torch_available else None,
        "cuda_available": cuda_available,
        "require_gpu": require_gpu,
    }


def _torch_compile_check(report: dict[str, Any], torch_module: Any | None, torch_cuda_available: bool) -> None:
    if torch_module is None:
        _set_check(report, "torch_compile", "skipped", "torch not installed")
        return
    if not torch_cuda_available:
        _set_check(report, "torch_compile", "skipped", "cuda unavailable")
        return
    if not hasattr(torch_module, "compile"):
        _set_check(report, "torch_compile", "skipped", "torch.compile unavailable")
        return
    try:
        def _fn(x: Any) -> Any:
            return x + 1

        compiled = torch_module.compile(_fn)
        _ = compiled(torch_module.ones(1, device="cuda"))
        _set_check(report, "torch_compile", "ok")
    except Exception as exc:
        _set_check(report, "torch_compile", "failed", str(exc))


def _dlpack_check(
    report: dict[str, Any],
    backend: backends.Backend | None,
    torch_module: Any | None,
    *,
    require_gpu: bool,
) -> None:
    if backend is None or backend.name != "cupy":
        status = "failed" if require_gpu else "skipped"
        _set_check(report, "dlpack_roundtrip", status, "cupy backend unavailable")
        return
    if torch_module is None:
        status = "failed" if require_gpu else "skipped"
        _set_check(report, "dlpack_roundtrip", status, "torch not installed")
        return
    try:
        cupy_array = backend.module.arange(4)
        torch_tensor = backends.cupy_to_torch(cupy_array)
        _ = backends.torch_to_cupy(torch_tensor)
        _set_check(report, "dlpack_roundtrip", "ok")
    except Exception as exc:
        _set_check(report, "dlpack_roundtrip", "failed", str(exc))


def _run_microbench(backend: backends.Backend) -> tuple[dict[str, Any], bool]:
    report: dict[str, Any] = {
        "backend": {"name": backend.name, "gpu_available": backend.gpu_available},
    }
    failure = False
    status = CheckStatus(status="ok")
    start = time.perf_counter()

    try:
        with backends.host_transfer_guard(fail_on_transfer=True):
            _bench_kernel(backend)
    except backends.HostTransferError as exc:
        status = CheckStatus(status="failed", detail=str(exc))
        failure = True
    except Exception as exc:
        status = CheckStatus(status="failed", detail=str(exc))
        failure = True

    elapsed = time.perf_counter() - start
    report["duration_sec"] = elapsed
    report["host_transfer_guard"] = status.to_dict()
    return report, failure


def _bench_kernel(backend: backends.Backend, *, steps: int = 8) -> None:
    xp = backend.module
    array = xp.arange(1024, dtype=getattr(xp, "float32", None) or "float32")
    for _ in range(steps):
        array = array * 1.0001 + 0.1
    _maybe_synchronize(xp)


def _maybe_synchronize(xp: Any) -> None:
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


def _resolve_run_id(run_id: str, prefix: str) -> str:
    if run_id:
        return run_id
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}"


def log_event(logs_path: Path, event: str, *, level: str = "info", data: Mapping[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"ts": _utc_timestamp(), "event": event, "level": level}
    if data:
        payload["data"] = dict(data)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with logs_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _set_check(report: dict[str, Any], name: str, status: str, detail: str | None = None) -> None:
    checks = report.setdefault("checks", {})
    payload = {"status": status}
    if detail:
        payload["detail"] = detail
    checks[name] = payload


def _write_json_artifact(
    artifacts_dir: Path,
    filename: str,
    payload: Mapping[str, Any],
) -> tuple[str, str]:
    text = manifest.canonical_json_dumps(dict(payload))
    path = artifacts_dir / filename
    path.write_text(f"{text}\n", encoding="utf-8")
    return filename, manifest.sha256_file(path)


def _write_bytes_artifact(artifacts_dir: Path, filename: str, payload: bytes) -> tuple[str, str]:
    path = artifacts_dir / filename
    path.write_bytes(payload)
    return filename, manifest.sha256_file(path)


def _emit_json_report(report: Mapping[str, Any], target: str | None, *, default_path: Path | None = None) -> None:
    if target is None:
        return
    text = json.dumps(report, indent=2, sort_keys=True)
    if target == "-":
        print(text)
        return
    if target == "default":
        if default_path is None:
            return
        out_path = default_path
    else:
        out_path = Path(target)
    out_path.write_text(f"{text}\n", encoding="utf-8")


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
    return False


def _torch_cuda_available(torch_module: Any | None) -> bool:
    if torch_module is None:
        return False
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is None:
        return False
    is_available = getattr(cuda_module, "is_available", None)
    if callable(is_available):
        try:
            return bool(is_available())
        except Exception:
            return False
    return False


def _import_optional(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
