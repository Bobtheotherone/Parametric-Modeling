from __future__ import annotations

import pytest

from formula_foundry.coupongen.units import parse_length_nm


def test_lengthnm_parsing_integer_nm() -> None:
    assert parse_length_nm("0.25mm") == 250000
    assert parse_length_nm("10mil") == 254000
    assert parse_length_nm("250um") == 250000
    assert parse_length_nm("1000nm") == 1000
    assert parse_length_nm("1000") == 1000
    assert isinstance(parse_length_nm("1mm"), int)

    with pytest.raises(ValueError):
        parse_length_nm("12.34foo")
