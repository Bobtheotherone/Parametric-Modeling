from __future__ import annotations

import json
from pathlib import Path

import pytest
from formula_foundry.meta import writer


def _sample_inputs() -> dict[str, object]:
    normalized_config = {
        "format_version": 1,
        "case_id": "case-001",
        "frequency": {"start_hz": 1_000_000_000, "stop_hz": 2_000_000_000, "npoints": 3},
        "ports": [{"port_id": "P1"}],
    }
    toolchain = {
        "docker": {
            "image": "openems:0.0.35",
            "digest": "sha256:" + "a" * 64,
            "pinned_ref": "openems:0.0.35@" + "sha256:" + "a" * 64,
        },
        "versions": {"image_tag": "0.0.35", "cuda": "12.2", "ubuntu": "22.04"},
    }
    frequency_grid = writer.FrequencyGrid(start_hz=1_000_000_000, stop_hz=2_000_000_000, npoints=3)
    mesh_summary = {
        "total_cells": 100,
        "n_lines_x": 10,
        "n_lines_y": 10,
        "n_lines_z": 1,
    }
    port_map = {
        "backend": "openems",
        "port_count": 1,
        "ports": [
            {
                "port_id": "P1",
                "port_index": 0,
                "port_type": "waveguide",
                "reference_plane": {"enabled": True, "method": "reference_plane", "distance_nm": 250_000},
            }
        ],
    }
    runtime = writer.RuntimeStats(
        started_utc="2025-01-01T00:00:00Z",
        finished_utc="2025-01-01T00:00:10Z",
        duration_seconds=10.0,
    )
    termination = writer.TerminationStats(
        reason="end_criteria",
        max_steps=10000,
        actual_steps=9000,
        end_criteria_db=-50.0,
        converged=True,
    )
    return {
        "normalized_config": normalized_config,
        "toolchain": toolchain,
        "frequency_grid": frequency_grid,
        "mesh_summary": mesh_summary,
        "port_map": port_map,
        "runtime": runtime,
        "termination": termination,
    }


def test_build_meta_json_required_fields() -> None:
    inputs = _sample_inputs()
    normalized_config = inputs["normalized_config"]
    mesh_summary = inputs["mesh_summary"]

    meta = writer.build_meta_json(
        normalized_config=normalized_config,
        toolchain=inputs["toolchain"],
        frequency_grid=inputs["frequency_grid"],
        mesh_summary=mesh_summary,
        port_map=inputs["port_map"],
        runtime=inputs["runtime"],
        termination=inputs["termination"],
        git_commit="b" * 40,
    )

    writer.validate_meta_json(meta)

    assert meta["case_hash"] == writer.compute_case_hash(normalized_config)
    assert meta["git_commit"] == "b" * 40
    assert meta["toolchain"]["docker"]["digest"].startswith("sha256:")
    assert "versions" in meta["toolchain"]
    assert meta["frequency_grid"]["points_hz"][0] == 1_000_000_000
    assert meta["mesh_hash"] == writer.compute_mesh_hash(mesh_summary)
    assert meta["port_map"]["ports"][0]["reference_plane"]["enabled"] is True
    assert meta["runtime"]["duration_seconds"] == 10.0
    assert meta["termination"]["reason"] == "end_criteria"


def test_write_meta_json_round_trip(tmp_path: Path) -> None:
    inputs = _sample_inputs()
    meta = writer.build_meta_json(
        normalized_config=inputs["normalized_config"],
        toolchain=inputs["toolchain"],
        frequency_grid=inputs["frequency_grid"],
        mesh_summary=inputs["mesh_summary"],
        port_map=inputs["port_map"],
        runtime=inputs["runtime"],
        termination=inputs["termination"],
        git_commit="c" * 40,
    )

    path = tmp_path / "meta.json"
    writer.write_meta_json(path, meta)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["case_hash"] == meta["case_hash"]
    assert loaded["mesh_hash"] == meta["mesh_hash"]
    assert loaded["port_map"]["backend"] == "openems"


def test_validate_meta_json_missing_fields() -> None:
    with pytest.raises(writer.MetaWriterError, match="missing required"):
        writer.validate_meta_json({"case_hash": "0" * 64})
