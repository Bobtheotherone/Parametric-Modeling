from __future__ import annotations

import pytest
from formula_foundry.preflight.resource import parse_resource_budget_lines


def test_parse_resource_budget_lines_with_units() -> None:
    budget = parse_resource_budget_lines(["max_cells=1_000", "max_memory=1.5GB"])
    assert budget.max_cells == 1000
    assert budget.max_memory_bytes == int(1.5 * 1024**3)


def test_parse_resource_budget_lines_missing_limits() -> None:
    with pytest.raises(ValueError, match="No resource budget limits"):
        parse_resource_budget_lines(["# comment", "  "])
