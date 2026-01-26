"""Tests for the oracle pipeline runner.

REQ-M2-001: Oracle run executes the golden pipeline by default.
REQ-M2-002: Stub mode requires explicit opt-in and labels outputs NON-GOLD.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from formula_foundry.oracle import run as run_mod


def test_oracle_run_executes_full_pipeline_real_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default run executes the golden pipeline and not the stub."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    out_dir = tmp_path / "out"

    called = {"golden": False, "stub": False}

    def fake_golden(case: Path, out: Path) -> None:
        called["golden"] = True
        (out / "golden.txt").write_text("ok", encoding="utf-8")

    def fake_stub(case: Path, out: Path) -> None:
        called["stub"] = True

    monkeypatch.setattr(run_mod, "_run_golden_pipeline", fake_golden)
    monkeypatch.setattr(run_mod, "_run_stub_pipeline", fake_stub)

    result = run_mod.run_oracle_pipeline(case_dir, out_dir=out_dir)

    assert result.mode == "golden"
    assert result.is_golden is True
    assert called["golden"] is True
    assert called["stub"] is False
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["quality_label"] == run_mod.GOLD_LABEL


def test_stub_mode_requires_explicit_allow_and_labels_non_gold(tmp_path: Path) -> None:
    """Stub mode requires explicit allow and marks outputs NON-GOLD."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    out_dir = tmp_path / "out"

    with pytest.raises(run_mod.StubRunNotAllowedError):
        run_mod.run_oracle_pipeline(case_dir, out_dir=out_dir, mode="stub")

    result = run_mod.run_oracle_pipeline(
        case_dir,
        out_dir=out_dir,
        mode="stub",
        allow_stub=True,
    )

    assert result.mode == "stub"
    assert result.is_golden is False
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["quality_label"] == run_mod.NON_GOLD_LABEL
    assert (out_dir / run_mod.NON_GOLD_MARKER_NAME).exists()
