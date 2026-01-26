from __future__ import annotations

import pytest
from formula_foundry.preflight import resource


def test_preflight_estimates_cells_and_memory_and_passes_under_budget() -> None:
    budget = resource.ResourceBudget(max_cells=1000, max_memory_bytes=200_000)

    report = resource.preflight_from_lines(
        n_lines_x=11,
        n_lines_y=11,
        n_lines_z=11,
        budget=budget,
        bytes_per_cell=64,
    )

    assert report.within_budget is True
    assert report.estimate.total_cells == 1000
    assert report.estimate.memory_bytes == 64_000
    assert report.over_cells is False
    assert report.over_memory is False


def test_preflight_fails_over_budget_reports_estimate() -> None:
    budget = resource.ResourceBudget(max_cells=500, max_memory_bytes=20_000)

    with pytest.raises(resource.ResourceBudgetExceeded) as excinfo:
        resource.preflight_from_lines(
            n_lines_x=11,
            n_lines_y=11,
            n_lines_z=11,
            budget=budget,
            bytes_per_cell=64,
        )

    report = excinfo.value.report
    assert report.estimate.total_cells == 1000
    assert report.estimate.memory_bytes == 64_000
    assert report.over_cells is True
    assert report.over_memory is True
    assert "Resource budget exceeded" in str(excinfo.value)
