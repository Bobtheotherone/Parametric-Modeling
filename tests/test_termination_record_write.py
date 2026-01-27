from __future__ import annotations

import json
from pathlib import Path

from formula_foundry.solver.termination import TerminationPolicy, record_termination


def test_record_termination_writes_json(tmp_path: Path) -> None:
    policy = TerminationPolicy.from_dict({"end_criteria_db": -40.0, "max_timesteps": 10})
    path = tmp_path / "termination.json"
    record = record_termination(policy, actual_timesteps=5, final_energy_db=-45.0, output_path=path)

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["cause"] == "end_criteria"
    assert payload["end_criteria_db"] == record.end_criteria_db
    assert payload["max_timesteps"] == record.max_timesteps
