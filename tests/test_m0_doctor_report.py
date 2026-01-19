from __future__ import annotations

from tools import m0


def test_doctor_report_schema_cpu() -> None:
    report = m0._doctor_report(require_gpu=False)

    assert isinstance(report["gpu_devices"], list)
    assert "cuda_visible_devices" in report
    assert "driver_version" in report
    assert "cuda_runtime_version" in report
    assert "cudnn_version" in report
    assert isinstance(report["nvidia_smi"], dict)
    assert "available" in report["nvidia_smi"]
    assert "gpus" in report["nvidia_smi"]


def test_parse_nvidia_smi_versions() -> None:
    sample = (
        "+-----------------------------------------------------------------------------+\n"
        "| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2     |\n"
        "+-----------------------------------------------------------------------------+\n"
    )
    driver, cuda = m0._parse_nvidia_smi_versions(sample)

    assert driver == "535.129.03"
    assert cuda == "12.2"


def test_parse_nvidia_smi_query() -> None:
    sample = "0, NVIDIA A100-SXM4-40GB, 40536\n1, NVIDIA A100-SXM4-40GB, 40536\n"
    gpus = m0._parse_nvidia_smi_query(sample)

    assert gpus == [
        {"index": 0, "name": "NVIDIA A100-SXM4-40GB", "memory_total_mb": 40536},
        {"index": 1, "name": "NVIDIA A100-SXM4-40GB", "memory_total_mb": 40536},
    ]
