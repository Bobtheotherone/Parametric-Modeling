"""Tests for CI openEMS execution in pinned toolchain (REQ-M2-024, REQ-M2-225)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.ci.real_openems import (
    RunResult,
    ToolchainLock,
    build_metrics_summary,
    ensure_digest_ref,
    generate_minimal_case,
    load_toolchain_lock,
    validate_required_artifacts,
    validate_toolchain_lock,
)


def test_ci_workflow_runs_real_openems_case() -> None:
    """REQ-M2-024: CI workflow defines the real openEMS gate."""
    workflow_path = Path(".github") / "workflows" / "oracle-ci.yml"
    assert workflow_path.exists(), f"Missing CI workflow: {workflow_path}"

    content = workflow_path.read_text(encoding="utf-8")
    required_snippets = [
        "openems-real",
        "tools/ci/real_openems.py",
        "tools/m2/docker/Dockerfile",
        "toolchain/openems.lock.json",
        "ci_artifacts/openems",
        "metrics_summary.json",
        "actions/upload-artifact@",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in content]
    assert not missing, "CI workflow missing required openEMS gate config:\n" + "\n".join(missing)


def test_toolchain_lock_requires_base_image_digest(tmp_path: Path) -> None:
    """REQ-M2-024: Toolchain lock must include base image digest."""
    payload = {
        "openems_version": "0.0.35",
        "docker_image": "formula-foundry-openems:0.0.35",
        "base_image": "ubuntu:22.04",
        "base_image_digest": "sha256:" + ("a" * 64),
    }
    lock_path = tmp_path / "openems.lock.json"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    lock = load_toolchain_lock(lock_path)
    validate_toolchain_lock(lock)

    payload["base_image_digest"] = "missing"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    lock = load_toolchain_lock(lock_path)
    with pytest.raises(ValueError, match="base_image_digest"):
        validate_toolchain_lock(lock)


def test_ensure_digest_ref_enforces_pinning() -> None:
    """REQ-M2-225: Toolchain image reference must be digest-pinned."""
    digest = "sha256:" + ("b" * 64)
    assert ensure_digest_ref(f"ghcr.io/openems:0.0.35@{digest}") == f"ghcr.io/openems:0.0.35@{digest}"
    assert ensure_digest_ref(digest) == digest
    with pytest.raises(ValueError, match="digest-pinned"):
        ensure_digest_ref("ghcr.io/openems:0.0.35")


def test_generate_minimal_case_writes_xml(tmp_path: Path) -> None:
    """REQ-M2-024: Minimal openEMS case is generated deterministically."""
    case_path = generate_minimal_case(tmp_path, "ci_minimal_openems.xml")
    content = case_path.read_text(encoding="utf-8")
    assert "<openEMS>" in content
    assert "<RectilinearGrid" in content


def test_validate_required_artifacts_reports_missing(tmp_path: Path) -> None:
    """REQ-M2-024: Artifact validation reports missing files."""
    required = ["case.xml", "openems_stdout.log", "openems_stderr.log"]
    (tmp_path / "case.xml").write_text("x", encoding="utf-8")
    missing = validate_required_artifacts(tmp_path, required)
    assert missing == ["openems_stdout.log", "openems_stderr.log"]


def test_build_metrics_summary_records_status(tmp_path: Path) -> None:
    """REQ-M2-225: Metrics summary records run status and toolchain."""
    toolchain = ToolchainLock(
        openems_version="0.0.35",
        docker_image="formula-foundry-openems:0.0.35",
        base_image="ubuntu:22.04",
        base_image_digest="sha256:" + ("c" * 64),
    )
    run_result = RunResult(
        image_ref="sha256:" + ("d" * 64),
        image_id="sha256:" + ("d" * 64),
        stdout_path=tmp_path / "openems_stdout.log",
        stderr_path=tmp_path / "openems_stderr.log",
        exit_code=0,
        duration_sec=1.2,
    )
    summary = build_metrics_summary(
        case_id="ci_minimal_openems",
        toolchain=toolchain,
        run_result=run_result,
        required_artifacts=["case.xml"],
        missing_artifacts=[],
        error=None,
    )
    assert summary["status"] == "pass"
    assert summary["toolchain"]["openems_version"] == "0.0.35"
    assert summary["artifact_validation"]["missing"] == []
