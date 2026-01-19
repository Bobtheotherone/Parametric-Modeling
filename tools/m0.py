from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import platform
import random
import re
import subprocess
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

    smoke = subparsers.add_parser(
        "smoke",
        help="Run fast GPU smoke checks",
        parents=[shared],
        conflict_handler="resolve",
    )
    smoke.add_argument("--json", nargs="?", const="-", default=None)

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
    report, gpu_payload = _doctor_report_with_payload(require_gpu=args.require_gpu)
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
            environment_payload=_build_environment_payload(project_root, gpu_payload),
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
    gpu_payload = _gpu_environment_payload(require_gpu=args.require_gpu)

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
            environment_payload=_build_environment_payload(project_root, gpu_payload),
        )
        run_manifest.write(run.manifest_path)

    _emit_json_report(report, args.json)
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
    gpu_payload = _gpu_environment_payload(require_gpu=args.require_gpu)

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
                environment_payload=_build_environment_payload(project_root, gpu_payload),
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
                environment_payload=_build_environment_payload(project_root, gpu_payload),
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
            environment_payload=_build_environment_payload(project_root, gpu_payload),
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
    gpu_payload = _gpu_environment_payload(require_gpu=args.require_gpu)
    env_payload = _build_environment_payload(project_root, gpu_payload)

    artifacts_a = _run_repro_pass(
        run_root,
        run_id_a,
        project_root,
        command_line,
        mode=args.mode,
        seed=args.seed,
        payload_bytes=args.payload_bytes,
        environment_payload=env_payload,
    )
    artifacts_b = _run_repro_pass(
        run_root,
        run_id_b,
        project_root,
        command_line,
        mode=args.mode,
        seed=args.seed,
        payload_bytes=args.payload_bytes,
        environment_payload=env_payload,
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
    environment_payload: Mapping[str, Any] | None = None,
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
            environment_payload=environment_payload,
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
    report, _ = _doctor_report_with_payload(require_gpu=require_gpu)
    return report


def _doctor_report_with_payload(*, require_gpu: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    cupy_module = _import_optional("cupy")
    torch_module = _import_optional("torch")
    cupy_available = cupy_module is not None
    torch_available = torch_module is not None
    cupy_cuda_available = _cupy_cuda_available(cupy_module)
    torch_cuda_available = _torch_cuda_available(torch_module)
    cuda_available = cupy_cuda_available or torch_cuda_available

    nvidia_smi = _nvidia_smi_summary()
    devices, selected_device = _collect_gpu_devices(cupy_module, torch_module, nvidia_smi)
    driver_version = _resolve_driver_version(nvidia_smi)
    cuda_runtime_version = _resolve_cuda_runtime_version(cupy_module, torch_module)
    cudnn_version = _resolve_cudnn_version(torch_module)

    report = {
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
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_devices": devices,
        "selected_device": selected_device,
        "driver_version": driver_version,
        "cuda_runtime_version": cuda_runtime_version,
        "cudnn_version": cudnn_version,
        "nvidia_smi": nvidia_smi,
    }
    return report, _stable_gpu_payload(report)


def _gpu_environment_payload(*, require_gpu: bool) -> dict[str, Any]:
    _, payload = _doctor_report_with_payload(require_gpu=require_gpu)
    return payload


def _build_environment_payload(project_root: Path, gpu_payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = manifest.build_environment_payload(project_root)
    payload["gpu"] = dict(gpu_payload)
    return payload


def _stable_gpu_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    devices: list[dict[str, Any]] = []
    for device in report.get("gpu_devices", []) if isinstance(report.get("gpu_devices"), list) else []:
        if not isinstance(device, Mapping):
            continue
        devices.append(
            {
                "index": device.get("index"),
                "name": device.get("name"),
                "compute_capability": device.get("compute_capability"),
                "total_vram_bytes": device.get("total_vram_bytes"),
            }
        )
    return {
        "cuda_visible_devices": report.get("cuda_visible_devices"),
        "driver_version": report.get("driver_version"),
        "cuda_runtime_version": report.get("cuda_runtime_version"),
        "cudnn_version": report.get("cudnn_version"),
        "selected_device": report.get("selected_device"),
        "devices": devices,
    }


def _collect_gpu_devices(
    cupy_module: Any | None,
    torch_module: Any | None,
    nvidia_smi: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int | None]:
    devices, selected = _torch_gpu_devices(torch_module)
    if not devices:
        devices, selected = _cupy_gpu_devices(cupy_module)
    if not devices:
        smi_devices = nvidia_smi.get("gpus")
        if isinstance(smi_devices, list):
            for item in smi_devices:
                if not isinstance(item, Mapping):
                    continue
                devices.append(
                    {
                        "index": item.get("index"),
                        "name": item.get("name"),
                        "compute_capability": None,
                        "total_vram_bytes": _mb_to_bytes(item.get("memory_total_mb")),
                    }
                )
    return devices, selected


def _torch_gpu_devices(torch_module: Any | None) -> tuple[list[dict[str, Any]], int | None]:
    if torch_module is None or not _torch_cuda_available(torch_module):
        return [], None
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is None:
        return [], None
    try:
        count = int(cuda_module.device_count())
    except Exception:
        return [], None
    selected: int | None = None
    try:
        selected = int(cuda_module.current_device())
    except Exception:
        selected = None
    devices: list[dict[str, Any]] = []
    for idx in range(count):
        name: str | None = None
        compute_capability: str | None = None
        total_vram: int | None = None
        try:
            props = cuda_module.get_device_properties(idx)
        except Exception:
            props = None
        if props is not None:
            name = getattr(props, "name", None)
            total = getattr(props, "total_memory", None)
            if total is not None:
                try:
                    total_vram = int(total)
                except Exception:
                    total_vram = None
            major = getattr(props, "major", None)
            minor = getattr(props, "minor", None)
            if major is not None and minor is not None:
                compute_capability = f"{major}.{minor}"
        if compute_capability is None:
            try:
                cap = cuda_module.get_device_capability(idx)
                if cap is not None and len(cap) >= 2:
                    compute_capability = f"{cap[0]}.{cap[1]}"
            except Exception:
                compute_capability = None
        if name is None:
            try:
                name = cuda_module.get_device_name(idx)
            except Exception:
                name = None
        devices.append(
            {
                "index": idx,
                "name": name,
                "compute_capability": compute_capability,
                "total_vram_bytes": total_vram,
            }
        )
    return devices, selected


def _cupy_gpu_devices(cupy_module: Any | None) -> tuple[list[dict[str, Any]], int | None]:
    if cupy_module is None or not _cupy_cuda_available(cupy_module):
        return [], None
    cuda_module = getattr(cupy_module, "cuda", None)
    if cuda_module is None:
        return [], None
    runtime = getattr(cuda_module, "runtime", None) if cuda_module is not None else None
    if runtime is None:
        return [], None
    try:
        count = int(runtime.getDeviceCount())
    except Exception:
        return [], None
    selected: int | None = None
    try:
        selected = int(runtime.getDevice())
    except Exception:
        selected = None
    devices: list[dict[str, Any]] = []
    for idx in range(count):
        name: str | None = None
        compute_capability: str | None = None
        total_vram: int | None = None
        try:
            device = cuda_module.Device(idx)
        except Exception:
            device = None
        if device is not None:
            try:
                name = device.name
            except Exception:
                name = None
            try:
                cap = device.compute_capability
                if cap is not None and len(cap) >= 2:
                    compute_capability = f"{cap[0]}.{cap[1]}"
            except Exception:
                compute_capability = None
            try:
                mem_info = device.mem_info
                if mem_info is not None and len(mem_info) >= 2:
                    total_vram = int(mem_info[1])
            except Exception:
                total_vram = None
        devices.append(
            {
                "index": idx,
                "name": name,
                "compute_capability": compute_capability,
                "total_vram_bytes": total_vram,
            }
        )
    return devices, selected


def _resolve_driver_version(nvidia_smi: Mapping[str, Any]) -> str | None:
    if isinstance(nvidia_smi, Mapping):
        driver = nvidia_smi.get("driver_version")
        if isinstance(driver, str) and driver:
            return driver
    return _nvml_driver_version()


def _resolve_cuda_runtime_version(cupy_module: Any | None, torch_module: Any | None) -> str | None:
    cuda_module = getattr(cupy_module, "cuda", None) if cupy_module is not None else None
    runtime = getattr(cuda_module, "runtime", None) if cuda_module is not None else None
    get_version = getattr(runtime, "runtimeGetVersion", None) if runtime is not None else None
    if callable(get_version):
        try:
            return _format_cuda_version(int(get_version()))
        except Exception:
            pass
    if torch_module is not None:
        version = getattr(torch_module, "version", None)
        cuda_version = getattr(version, "cuda", None) if version is not None else None
        if cuda_version:
            return str(cuda_version)
    return None


def _resolve_cudnn_version(torch_module: Any | None) -> int | None:
    if torch_module is None:
        return None
    backends = getattr(torch_module, "backends", None)
    cudnn = getattr(backends, "cudnn", None) if backends is not None else None
    version = getattr(cudnn, "version", None) if cudnn is not None else None
    if callable(version):
        try:
            value = version()
        except Exception:
            return None
        if value is not None:
            try:
                return int(value)
            except Exception:
                return None
    return None


def _format_cuda_version(value: int) -> str:
    major = value // 1000
    minor = (value % 1000) // 10
    patch = value % 10
    if patch:
        return f"{major}.{minor}.{patch}"
    return f"{major}.{minor}"


def _mb_to_bytes(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value) * 1024 * 1024
    if isinstance(value, str):
        try:
            return int(float(value)) * 1024 * 1024
        except ValueError:
            return None
    return None


def _nvidia_smi_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "available": False,
        "exit_code": None,
        "driver_version": None,
        "cuda_version": None,
        "gpus": [],
    }
    rc, out, err = _run_nvidia_smi(["nvidia-smi"])
    summary["exit_code"] = rc
    if rc != 0:
        summary["error"] = err.strip() or None
        return summary
    summary["available"] = True
    driver, cuda = _parse_nvidia_smi_versions(out)
    summary["driver_version"] = driver
    summary["cuda_version"] = cuda
    q_rc, q_out, q_err = _run_nvidia_smi(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"]
    )
    summary["query_exit_code"] = q_rc
    if q_rc == 0:
        summary["gpus"] = _parse_nvidia_smi_query(q_out)
    else:
        summary["query_error"] = q_err.strip() or None
    return summary


def _parse_nvidia_smi_versions(text: str) -> tuple[str | None, str | None]:
    driver = None
    cuda = None
    driver_match = re.search(r"Driver Version:\s*([0-9A-Za-z\.\-]+)", text)
    if driver_match:
        driver = driver_match.group(1)
    cuda_match = re.search(r"CUDA Version:\s*([0-9A-Za-z\.\-]+)", text)
    if cuda_match:
        cuda = cuda_match.group(1)
    return driver, cuda


def _parse_nvidia_smi_query(text: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        index: int | None
        try:
            index = int(parts[0])
        except Exception:
            index = None
        name = parts[1] or None
        memory_total_mb = _parse_int(parts[2])
        gpus.append(
            {
                "index": index,
                "name": name,
                "memory_total_mb": memory_total_mb,
            }
        )
    return gpus


def _parse_int(value: str) -> int | None:
    tokens = value.strip().split()
    if not tokens:
        return None
    try:
        return int(float(tokens[0]))
    except Exception:
        return None


def _run_nvidia_smi(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return 127, "", "nvidia-smi not found"
    except Exception as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def _nvml_driver_version() -> str | None:
    try:
        nvml = importlib.import_module("pynvml")
    except Exception:
        return None
    try:
        nvml.nvmlInit()
        version = nvml.nvmlSystemGetDriverVersion()
        if isinstance(version, bytes):
            version = version.decode("utf-8", errors="ignore")
        return str(version)
    except Exception:
        return None
    finally:
        with contextlib.suppress(Exception):
            nvml.nvmlShutdown()


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
    report["dlpack_zero_copy_ok"] = "skip"
    report["dlpack_pointer"] = {
        "cupy_ptr": None,
        "torch_ptr": None,
        "equal": None,
        "roundtrip_ptr": None,
        "roundtrip_equal": None,
    }
    report["dlpack_write_through_ok"] = None
    report["dlpack_guard_triggered"] = None
    report["dlpack_skip_reason"] = None
    report["dlpack_error"] = None
    if backend is None or backend.name != "cupy":
        status = "failed" if require_gpu else "skipped"
        report["dlpack_skip_reason"] = "cupy backend unavailable"
        _set_check(report, "dlpack_roundtrip", status, "cupy backend unavailable")
        return
    if torch_module is None:
        status = "failed" if require_gpu else "skipped"
        report["dlpack_skip_reason"] = "torch not installed"
        _set_check(report, "dlpack_roundtrip", status, "torch not installed")
        return
    try:
        with backends.host_transfer_guard(fail_on_transfer=True):
            dtype = getattr(backend.module, "float32", None) or "float32"
            cupy_array = backend.module.arange(8, dtype=dtype)
            torch_tensor = backends.cupy_to_torch(cupy_array)
            pointer = report["dlpack_pointer"]
            cupy_ptr = int(cupy_array.data.ptr)
            torch_ptr = int(torch_tensor.data_ptr())
            pointer["cupy_ptr"] = cupy_ptr
            pointer["torch_ptr"] = torch_ptr
            pointer["equal"] = cupy_ptr == torch_ptr
            torch_tensor.add_(1)
            torch_cuda = getattr(torch_module, "cuda", None)
            sync = getattr(torch_cuda, "synchronize", None) if torch_cuda is not None else None
            if callable(sync):
                sync()
            expected_sum = backend.module.arange(8, dtype=dtype).sum() + cupy_array.size
            write_through_device = cupy_array.sum() == expected_sum
            roundtrip = backends.torch_to_cupy(torch_tensor)
            roundtrip_ptr = int(roundtrip.data.ptr)
            pointer["roundtrip_ptr"] = roundtrip_ptr
            pointer["roundtrip_equal"] = roundtrip_ptr == cupy_ptr
        report["dlpack_guard_triggered"] = False
        report["dlpack_write_through_ok"] = bool(write_through_device)
        zero_copy_ok = bool(
            pointer["equal"]
            and pointer["roundtrip_equal"]
            and report["dlpack_write_through_ok"]
        )
        report["dlpack_zero_copy_ok"] = zero_copy_ok
        if zero_copy_ok:
            _set_check(report, "dlpack_roundtrip", "ok")
        else:
            report["dlpack_error"] = "zero-copy validation failed"
            _set_check(report, "dlpack_roundtrip", "failed", "zero-copy validation failed")
    except backends.HostTransferError as exc:
        report["dlpack_guard_triggered"] = True
        report["dlpack_zero_copy_ok"] = False
        report["dlpack_error"] = str(exc)
        _set_check(report, "dlpack_roundtrip", "failed", str(exc))
    except Exception as exc:
        report["dlpack_zero_copy_ok"] = False
        report["dlpack_error"] = str(exc)
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
