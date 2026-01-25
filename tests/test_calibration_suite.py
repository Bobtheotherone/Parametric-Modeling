from __future__ import annotations

from formula_foundry.calibration.library import REQUIRED_CALIBRATION_IDS, list_calibration_cases
from formula_foundry.calibration.runner import run_calibration_suite


def test_calibration_catalog_contains_required_cases() -> None:
    cases = list_calibration_cases()
    ids = [case.case_id for case in cases]

    assert len(ids) == len(set(ids))
    assert set(REQUIRED_CALIBRATION_IDS).issubset(ids)

    case_map = {case.case_id: case for case in cases}
    for required_id in REQUIRED_CALIBRATION_IDS:
        case = case_map[required_id]
        assert case.title
        assert case.description
        assert case.tags


def test_calibration_runner_executes_end_to_end() -> None:
    run = run_calibration_suite()
    run_again = run_calibration_suite()

    assert run.results
    assert run.summary["total"] == len(run.results)
    assert run.summary["passed"] == len(run.results)
    assert run.summary["failed"] == 0
    assert run.run_id

    run_case_ids = {result.case_id for result in run.results}
    assert set(REQUIRED_CALIBRATION_IDS).issubset(run_case_ids)

    for result in run.results:
        assert 0.0 <= result.metric <= 1.0
        assert result.status == "pass"
        assert result.detail

    assert run.run_id == run_again.run_id
    assert run.results == run_again.results
