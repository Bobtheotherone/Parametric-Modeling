"""Tests for solver termination policy handling."""

from __future__ import annotations

import pytest
from formula_foundry.solver.termination import (
    TerminationCause,
    TerminationPolicy,
    record_termination,
)


def test_policy_requires_end_criteria_and_max_steps():
    with pytest.raises(ValueError, match="end_criteria_db"):
        TerminationPolicy.from_dict({"max_steps": 100})
    with pytest.raises(ValueError, match="max_timesteps"):
        TerminationPolicy.from_dict({"end_criteria_db": -40.0})


def test_record_termination_cause_end_criteria():
    policy = TerminationPolicy.from_dict({"end_criteria_db": -40.0, "max_timesteps": 100})
    record = record_termination(policy, actual_timesteps=25, final_energy_db=-45.0)

    assert record.cause is TerminationCause.END_CRITERIA
    payload = record.to_dict()
    assert payload["cause"] == "end_criteria"
    assert payload["end_criteria_db"] == -40.0
    assert payload["max_timesteps"] == 100


def test_record_termination_cause_max_steps():
    policy = TerminationPolicy.from_dict({"end_criteria_db": -50.0, "max_steps": 5})
    record = record_termination(policy, actual_timesteps=5, final_energy_db=-10.0)

    assert record.cause is TerminationCause.MAX_STEPS
