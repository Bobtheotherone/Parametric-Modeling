from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Annotated

from pydantic import BeforeValidator, WithJsonSchema

_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_LENGTH_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*$")

_UNIT_SCALES_NM: dict[str, int] = {
    "nm": 1,
    "um": 1_000,
    "mm": 1_000_000,
    "mil": 25_400,
}

_MIN_I64 = -(2**63)
_MAX_I64 = 2**63 - 1

_LENGTH_JSON_SCHEMA = {
    "anyOf": [
        {"type": "integer"},
        {"type": "string", "pattern": r"^\s*[+-]?\d+\s*$"},
        {
            "type": "string",
            "pattern": r"^\s*[+-]?\d+(?:\.\d+)?\s*(nm|um|mm|mil)\s*$",
        },
    ],
    "title": "LengthNM",
    "description": "Integer nanometers (int or numeric string) or a string with nm/um/mm/mil units.",
}


def parse_length_nm(value: str | int | float) -> int:
    if isinstance(value, bool):
        raise ValueError("LengthNM does not accept boolean values.")
    if isinstance(value, int):
        return _check_i64(value)
    if isinstance(value, float):
        nm_decimal = _decimal_from_number(str(value))
        return _decimal_to_i64(nm_decimal)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("LengthNM requires a numeric value.")
        if _INTEGER_RE.match(text):
            return _check_i64(int(text))
        match = _LENGTH_RE.match(text)
        if not match:
            raise ValueError(
                "LengthNM string must be formatted like '0.25mm', '10mil', or '250um'."
            )
        number_text, unit = match.groups()
        unit = unit.lower()
        scale = _UNIT_SCALES_NM.get(unit)
        if scale is None:
            raise ValueError(f"Unknown LengthNM unit: {unit!r}")
        nm_decimal = _decimal_from_number(number_text) * Decimal(scale)
        return _decimal_to_i64(nm_decimal)
    raise ValueError(f"Unsupported LengthNM value: {value!r}")


def _decimal_from_number(text: str) -> Decimal:
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric value for LengthNM: {text!r}") from exc


def _decimal_to_i64(value: Decimal) -> int:
    if value != value.to_integral_value():
        raise ValueError("LengthNM must resolve to an integer number of nanometers.")
    return _check_i64(int(value))


def _check_i64(value: int) -> int:
    if value < _MIN_I64 or value > _MAX_I64:
        raise ValueError("LengthNM is outside the signed 64-bit integer range.")
    return value


LengthNM = Annotated[int, BeforeValidator(parse_length_nm), WithJsonSchema(_LENGTH_JSON_SCHEMA)]
