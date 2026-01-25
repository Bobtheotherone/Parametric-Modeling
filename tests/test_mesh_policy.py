"""Tests for mesh planner policy enforcement and summary output."""

from __future__ import annotations

import json
from pathlib import Path

from formula_foundry.mesh.planner import (
    MeshDomain,
    MeshFeature,
    MeshPolicy,
    compute_lambda_max_cell_nm,
    plan_mesh,
)
from formula_foundry.mesh.summary import DEFAULT_BYTES_PER_CELL, compute_mesh_hash, compute_mesh_summary


def _default_domain() -> MeshDomain:
    return MeshDomain(
        x_min_nm=0,
        x_max_nm=3_000_000,
        y_min_nm=0,
        y_max_nm=3_000_000,
        z_min_nm=0,
        z_max_nm=1_000_000,
    )


def test_mesh_policy_enforces_lambda_rule_grading_thirds_rule_and_pml_clearance() -> None:
    policy = MeshPolicy(
        lambda_divisor_max_cell=20,
        max_ratio=1.5,
        thirds_rule=True,
        pml_clearance_nm=200_000,
        base_cell_nm=1_000_000,
    )
    domain = MeshDomain(
        x_min_nm=0,
        x_max_nm=1_000_000,
        y_min_nm=0,
        y_max_nm=1_000_000,
        z_min_nm=0,
        z_max_nm=1_000_000,
    )
    features = [MeshFeature(axis="x", center_nm=500_000, radius_nm=300_000)]
    frequency_hz = 100_000_000_000
    plan = plan_mesh(domain, policy, frequency_hz=frequency_hz, features=features)
    summary = compute_mesh_summary(plan)

    max_cell = compute_lambda_max_cell_nm(frequency_hz, 1.0, policy.lambda_divisor_max_cell)
    assert max(summary["deltas_nm"]["x_nm"]) <= max_cell
    assert summary["ratio_stats"]["x"]["max_ratio"] <= policy.max_ratio
    assert 400_000 in plan.lines_x_nm
    assert 600_000 in plan.lines_x_nm
    assert plan.lines_x_nm[0] == -200_000
    assert plan.lines_x_nm[-1] == 1_200_000


def test_lambda_divisor_max_cell_enforced() -> None:
    policy = MeshPolicy(
        lambda_divisor_max_cell=20,
        max_ratio=1.5,
        thirds_rule=False,
        pml_clearance_nm=0,
        base_cell_nm=1_000_000,
    )
    domain = _default_domain()
    frequency_hz = 10_000_000_000
    plan = plan_mesh(domain, policy, frequency_hz=frequency_hz, epsilon_r=1.0)

    summary = compute_mesh_summary(plan)
    max_cell = compute_lambda_max_cell_nm(frequency_hz, 1.0, policy.lambda_divisor_max_cell)
    assert max(summary["deltas_nm"]["x_nm"]) <= max_cell
    assert max(summary["deltas_nm"]["y_nm"]) <= max_cell
    assert max(summary["deltas_nm"]["z_nm"]) <= max_cell


def test_thirds_rule_inserts_lines() -> None:
    policy = MeshPolicy(
        lambda_divisor_max_cell=50,
        max_ratio=2.0,
        thirds_rule=True,
        pml_clearance_nm=0,
        base_cell_nm=450,
    )
    domain = MeshDomain(
        x_min_nm=0,
        x_max_nm=900,
        y_min_nm=0,
        y_max_nm=900,
        z_min_nm=0,
        z_max_nm=900,
    )
    # Feature: center=450, radius=300 => feature region is [150, 750]
    # Thirds rule inserts lines at 1/3 and 2/3 into feature region:
    # 1/3: 150 + 600/3 = 150 + 200 = 350
    # 2/3: 150 + 2*600/3 = 150 + 400 = 550
    features = [MeshFeature(axis="x", center_nm=450, radius_nm=300)]
    plan = plan_mesh(domain, policy, frequency_hz=1_000_000_000, features=features)

    assert 350 in plan.lines_x_nm
    assert 550 in plan.lines_x_nm

    policy_no_thirds = MeshPolicy(
        lambda_divisor_max_cell=50,
        max_ratio=2.0,
        thirds_rule=False,
        pml_clearance_nm=0,
        base_cell_nm=450,
    )
    plan_no_thirds = plan_mesh(domain, policy_no_thirds, frequency_hz=1_000_000_000, features=features)
    assert 350 not in plan_no_thirds.lines_x_nm


def test_max_ratio_enforced() -> None:
    policy = MeshPolicy(
        lambda_divisor_max_cell=100,
        max_ratio=1.5,
        thirds_rule=True,
        pml_clearance_nm=0,
        base_cell_nm=1_000,
    )
    domain = MeshDomain(
        x_min_nm=0,
        x_max_nm=3_000,
        y_min_nm=0,
        y_max_nm=3_000,
        z_min_nm=0,
        z_max_nm=3_000,
    )
    features = [MeshFeature(axis="x", center_nm=1_500, radius_nm=300)]
    plan = plan_mesh(domain, policy, frequency_hz=1_000_000_000, features=features)
    summary = compute_mesh_summary(plan)

    assert summary["ratio_stats"]["x"]["max_ratio"] <= policy.max_ratio


def test_pml_clearance_expands_bounds() -> None:
    policy = MeshPolicy(
        lambda_divisor_max_cell=50,
        max_ratio=1.5,
        thirds_rule=False,
        pml_clearance_nm=200,
        base_cell_nm=300,
    )
    domain = MeshDomain(
        x_min_nm=0,
        x_max_nm=1_000,
        y_min_nm=-500,
        y_max_nm=500,
        z_min_nm=0,
        z_max_nm=1_000,
    )
    plan = plan_mesh(domain, policy, frequency_hz=1_000_000_000)

    assert plan.lines_x_nm[0] == -200
    assert plan.lines_x_nm[-1] == 1_200
    assert plan.lines_y_nm[0] == -700
    assert plan.lines_y_nm[-1] == 700


def test_mesh_summary_json_written(tmp_path: Path) -> None:
    policy = MeshPolicy(
        lambda_divisor_max_cell=50,
        max_ratio=2.0,
        thirds_rule=False,
        pml_clearance_nm=0,
        base_cell_nm=500,
    )
    domain = _default_domain()
    summary_path = tmp_path / "mesh_summary.json"
    plan = plan_mesh(domain, policy, frequency_hz=5_000_000_000, summary_path=summary_path)

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "lines_nm" in data
    assert "deltas_nm" in data
    assert "ratio_stats" in data
    assert "cell_count" in data
    assert "memory_estimate_bytes" in data
    assert "mesh_hash" in data

    expected_hash = compute_mesh_hash(data["lines_nm"])
    assert data["mesh_hash"] == expected_hash
    expected_cell_count = (len(plan.lines_x_nm) - 1) * (len(plan.lines_y_nm) - 1) * (len(plan.lines_z_nm) - 1)
    assert data["cell_count"] == expected_cell_count
    assert data["memory_estimate_bytes"] == expected_cell_count * DEFAULT_BYTES_PER_CELL
