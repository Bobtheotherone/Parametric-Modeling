from __future__ import annotations

import pytest
from formula_foundry.solver.field_export import (
    FIELD_EXPORT_ENV_FLAG,
    FieldExportNotAllowedError,
    enforce_field_export_policy,
)

from formula_foundry.openems.spec import SimulationControlSpec


def test_field_exports_disabled_by_default() -> None:
    control = SimulationControlSpec()
    assert control.dump_fields is False

    with pytest.raises(FieldExportNotAllowedError):
        enforce_field_export_policy(dump_fields=True, env={})


def test_field_exports_allow_explicit_opt_in() -> None:
    enforce_field_export_policy(dump_fields=True, allow_flag=True, env={})

    env = {FIELD_EXPORT_ENV_FLAG: "1"}
    enforce_field_export_policy(dump_fields=True, allow_flag=None, env=env)
