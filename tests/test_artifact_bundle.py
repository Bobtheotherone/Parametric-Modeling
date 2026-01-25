from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from formula_foundry.artifacts.bundle import REQUIRED_FILES, ArtifactBundle, ArtifactBundleError


def _manifest_payload(artifacts: dict[str, str]) -> dict[str, Any]:
    return {
        "git_sha": "a" * 40,
        "design_doc_sha256": "b" * 64,
        "environment_fingerprint": "c" * 64,
        "determinism": {"mode": "strict", "seeds": {"python": 123}},
        "command_line": ["m0", "smoke"],
        "artifacts": artifacts,
    }


def _write_manifest(path: Path, artifacts: dict[str, str]) -> None:
    payload = _manifest_payload(artifacts)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_artifact_bundle_contains_required_files() -> None:
    assert REQUIRED_FILES == ("manifest.json", "logs.jsonl")


def test_bundle_missing_required_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    bundle = ArtifactBundle(run_dir)
    with pytest.raises(ArtifactBundleError) as excinfo:
        bundle.validate()

    assert "manifest.json" in excinfo.value.missing
    assert "logs.jsonl" in excinfo.value.missing


def test_bundle_missing_output_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "logs.jsonl").write_text("", encoding="utf-8")
    (run_dir / "artifacts").mkdir()
    _write_manifest(run_dir / "manifest.json", {"output.json": "d" * 64})

    bundle = ArtifactBundle(run_dir)
    with pytest.raises(ArtifactBundleError) as excinfo:
        bundle.validate()

    assert "artifacts/output.json" in excinfo.value.missing


def test_bundle_validates_complete_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "logs.jsonl").write_text("", encoding="utf-8")
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "output.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_dir / "manifest.json", {"output.json": "d" * 64})

    bundle = ArtifactBundle(run_dir)
    bundle.validate()
