from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest
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


class TestM0CLIGateSmoke:
    """Tests verifying `python -m tools.m0 smoke` exists and exits cleanly."""

    def test_smoke_cli_help_exit_zero(self) -> None:
        """Verify `python -m tools.m0 smoke --help` exits with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "tools.m0", "smoke", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "smoke" in result.stdout.lower() or "gpu" in result.stdout.lower()

    def test_smoke_cli_invocation_with_mocked_backend(self, tmp_path: Path) -> None:
        """Verify `python -m tools.m0 smoke` invocation completes cleanly with mocked backend."""
        run_root = tmp_path / "runs"
        json_path = tmp_path / "smoke_output.json"

        # Mock backends to avoid GPU dependency
        mock_backend = mock.MagicMock()
        mock_backend.name = "numpy"
        mock_backend.gpu_available = False

        with (
            mock.patch("tools.m0.backends.select_backend", return_value=mock_backend),
            mock.patch("tools.m0._import_optional", return_value=None),
        ):
            rc = m0.main(
                [
                    "smoke",
                    "--run-root",
                    str(run_root),
                    "--run-id",
                    "smoke-gate-test",
                    "--json",
                    str(json_path),
                ]
            )

        # smoke exits 0 when not requiring GPU and backend is available
        assert rc == 0
        run_dir = run_root / "smoke-gate-test"
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "logs.jsonl").exists()
        assert (run_dir / "artifacts" / "smoke_report.json").exists()

    def test_smoke_subcommand_exists_in_parser(self) -> None:
        """Verify the 'smoke' subcommand is registered in the CLI parser."""
        parser = m0._build_parser()
        # Check subparser actions for 'smoke'
        subparsers_action = None
        for action in parser._actions:
            if hasattr(action, "choices") and isinstance(action.choices, dict):
                subparsers_action = action
                break
        assert subparsers_action is not None
        assert "smoke" in subparsers_action.choices


class TestM0CLIGateReproCheck:
    """Tests verifying `python -m tools.m0 repro-check` exists and exits cleanly."""

    def test_repro_check_cli_help_exit_zero(self) -> None:
        """Verify `python -m tools.m0 repro-check --help` exits with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "tools.m0", "repro-check", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "repro" in result.stdout.lower() or "check" in result.stdout.lower()

    def test_repro_check_cli_invocation_exits_cleanly(self, tmp_path: Path) -> None:
        """Verify `python -m tools.m0 repro-check` invocation completes cleanly."""
        run_root = tmp_path / "runs"

        rc = m0.main(
            [
                "repro-check",
                "--run-root",
                str(run_root),
                "--run-id",
                "repro-gate-test",
                "--payload-bytes",
                "64",
            ]
        )

        assert rc == 0
        run_a = run_root / "repro-gate-test-a"
        run_b = run_root / "repro-gate-test-b"
        assert (run_a / "manifest.json").exists()
        assert (run_b / "manifest.json").exists()
        assert (run_a / "logs.jsonl").exists()
        assert (run_b / "logs.jsonl").exists()

    def test_repro_check_subcommand_exists_in_parser(self) -> None:
        """Verify the 'repro-check' subcommand is registered in the CLI parser."""
        parser = m0._build_parser()
        subparsers_action = None
        for action in parser._actions:
            if hasattr(action, "choices") and isinstance(action.choices, dict):
                subparsers_action = action
                break
        assert subparsers_action is not None
        assert "repro-check" in subparsers_action.choices

    def test_repro_check_determinism(self, tmp_path: Path) -> None:
        """Verify repro-check produces identical artifacts for both runs."""
        run_root = tmp_path / "runs"

        rc = m0.main(
            [
                "repro-check",
                "--run-root",
                str(run_root),
                "--run-id",
                "repro-det-test",
                "--seed",
                "42",
                "--payload-bytes",
                "128",
            ]
        )

        assert rc == 0
        manifest_a = _read_json(run_root / "repro-det-test-a" / "manifest.json")
        manifest_b = _read_json(run_root / "repro-det-test-b" / "manifest.json")
        # Artifacts should match for deterministic runs
        assert manifest_a["artifacts"] == manifest_b["artifacts"]


class TestM0ModuleInvocation:
    """Tests verifying `python -m tools.m0` module invocation works correctly."""

    def test_module_invocation_shows_help(self) -> None:
        """Verify `python -m tools.m0 --help` works and shows available commands."""
        result = subprocess.run(
            [sys.executable, "-m", "tools.m0", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "smoke" in result.stdout
        assert "repro-check" in result.stdout

    def test_module_invocation_requires_subcommand(self) -> None:
        """Verify `python -m tools.m0` without subcommand shows error."""
        result = subprocess.run(
            [sys.executable, "-m", "tools.m0"],
            capture_output=True,
            text=True,
            check=False,
        )
        # argparse exits with code 2 for missing required arguments
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "command" in result.stderr.lower()
