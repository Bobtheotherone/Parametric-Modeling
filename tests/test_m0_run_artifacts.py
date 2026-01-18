from __future__ import annotations

import hashlib
from pathlib import Path

from formula_foundry.substrate import determinism, manifest


def test_run_dir_contract(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_id = "run-001"
    deterministic = determinism.determinism_manifest("strict", seed=123)
    run_manifest = manifest.Manifest(
        git_sha="a" * 40,
        design_doc_sha256="0" * 64,
        environment_fingerprint="1" * 64,
        determinism=deterministic,
        command_line=["m0", "smoke"],
        artifacts={},
    )

    artifacts = manifest.create_run(run_root, run_id, run_manifest)

    assert artifacts.run_dir == run_root / run_id
    assert artifacts.manifest_path.exists()
    assert artifacts.logs_path.exists()
    assert artifacts.artifacts_dir.is_dir()


def test_manifest_required_fields(tmp_path: Path) -> None:
    design_doc = tmp_path / "DESIGN_DOCUMENT.md"
    design_doc.write_text("Milestone: M0\n", encoding="utf-8")
    payload = {"python_version": "3.11.0", "platform": "test", "uv_lock_sha256": "deadbeef"}
    deterministic = determinism.determinism_manifest("fast", seed=7)
    expected_doc_hash = hashlib.sha256(design_doc.read_bytes()).hexdigest()
    expected_fingerprint = manifest.build_environment_fingerprint(payload)

    run_manifest = manifest.Manifest.from_environment(
        deterministic,
        command_line=["m0", "doctor"],
        artifacts={"artifact.bin": "2" * 64},
        project_root=tmp_path,
        design_doc_path=design_doc,
        environment_payload=payload,
        git_sha="b" * 40,
    )

    data = run_manifest.to_dict()

    assert data["git_sha"] == "b" * 40
    assert data["design_doc_sha256"] == expected_doc_hash
    assert data["environment_fingerprint"] == expected_fingerprint
    assert data["determinism"]["mode"] == "fast"
    assert data["determinism"]["seeds"]["python"] == 7
    assert data["command_line"] == ["m0", "doctor"]
    assert data["artifacts"]["artifact.bin"] == "2" * 64
    run_manifest.validate()


def test_manifest_canonical_json() -> None:
    run_manifest = manifest.Manifest(
        git_sha="f" * 40,
        design_doc_sha256="0" * 64,
        environment_fingerprint="1" * 64,
        determinism={"mode": "strict", "epsilon": 1.25, "seeds": {"python": 1}},
        command_line=["m0"],
        artifacts={"b.txt": "2", "a.txt": "1"},
    )

    expected = (
        '{"artifacts":{"a.txt":"1","b.txt":"2"},'
        '"command_line":["m0"],'
        '"design_doc_sha256":"0000000000000000000000000000000000000000000000000000000000000000",'
        '"determinism":{"epsilon":1.25,"mode":"strict","seeds":{"python":1}},'
        '"environment_fingerprint":"1111111111111111111111111111111111111111111111111111111111111111",'
        '"git_sha":"ffffffffffffffffffffffffffffffffffffffff"}'
    )

    assert run_manifest.to_json() == expected
