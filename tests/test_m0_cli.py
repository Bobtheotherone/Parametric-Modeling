from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from tools import m0


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_doctor_command_contract(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    json_path = tmp_path / "doctor.json"

    rc = m0.main(
        [
            "doctor",
            "--run-root",
            str(run_root),
            "--run-id",
            "doctor-1",
            "--json",
            str(json_path),
        ]
    )

    assert rc == 0
    run_dir = run_root / "doctor-1"
    manifest_path = run_dir / "manifest.json"
    logs_path = run_dir / "logs.jsonl"
    report_path = run_dir / "artifacts" / "doctor_report.json"
    assert manifest_path.exists()
    assert logs_path.exists()
    assert report_path.exists()
    manifest_data = _read_json(manifest_path)
    assert "doctor_report.json" in manifest_data["artifacts"]
    doctor_json = _read_json(json_path)
    assert doctor_json["require_gpu"] is False
    assert isinstance(doctor_json["gpu_devices"], list)
    assert "cuda_visible_devices" in doctor_json
    assert "driver_version" in doctor_json
    assert "cuda_runtime_version" in doctor_json
    assert "cudnn_version" in doctor_json
    assert isinstance(doctor_json["nvidia_smi"], dict)


def test_smoke_command_contract(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    json_path = tmp_path / "smoke.json"

    rc = m0.main(
        [
            "smoke",
            "--run-root",
            str(run_root),
            "--run-id",
            "smoke-1",
            "--json",
            str(json_path),
        ]
    )

    assert rc == 0
    run_dir = run_root / "smoke-1"
    manifest_path = run_dir / "manifest.json"
    logs_path = run_dir / "logs.jsonl"
    report_path = run_dir / "artifacts" / "smoke_report.json"
    assert manifest_path.exists()
    assert logs_path.exists()
    assert report_path.exists()
    report_data = _read_json(report_path)
    assert "checks" in report_data
    assert json_path.exists()
    assert report_data["dlpack_zero_copy_ok"] in (True, False, "skip")
    assert isinstance(report_data["dlpack_pointer"], dict)


def test_bench_command_contract(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"

    rc = m0.main(
        [
            "bench",
            "--run-root",
            str(run_root),
            "--run-id",
            "bench-1",
            "--json",
        ]
    )

    assert rc == 0
    run_dir = run_root / "bench-1"
    manifest_path = run_dir / "manifest.json"
    logs_path = run_dir / "logs.jsonl"
    report_path = run_dir / "artifacts" / "bench_report.json"
    bench_json_path = run_dir / "bench.json"
    assert manifest_path.exists()
    assert logs_path.exists()
    assert report_path.exists()
    assert bench_json_path.exists()


def test_repro_check_contract(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"

    rc = m0.main(
        [
            "repro-check",
            "--run-root",
            str(run_root),
            "--run-id",
            "repro-1",
        ]
    )

    assert rc == 0
    run_a = run_root / "repro-1-a"
    run_b = run_root / "repro-1-b"
    manifest_a = _read_json(run_a / "manifest.json")
    manifest_b = _read_json(run_b / "manifest.json")
    assert (run_a / "logs.jsonl").exists()
    assert (run_b / "logs.jsonl").exists()
    assert manifest_a["artifacts"] == manifest_b["artifacts"]
    assert "repro_payload.bin" in manifest_a["artifacts"]
