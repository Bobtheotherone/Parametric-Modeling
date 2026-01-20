"""Unit parsing for openEMS simulation config.

This module provides the same annotated-type pattern used in coupongen for
flexible input parsing while storing canonical values internally.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Annotated

from pydantic import BeforeValidator, WithJsonSchema

_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_FREQ_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*$")
_TIME_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*$")

_FREQ_SCALES_HZ: dict[str, int] = {
    "hz": 1,
    "khz": 1_000,
    "mhz": 1_000_000,
    "ghz": 1_000_000_000,
}

_TIME_SCALES_PS: dict[str, int] = {
    "ps": 1,
    "ns": 1_000,
    "us": 1_000_000,
    "ms": 1_000_000_000,
    "s": 1_000_000_000_000,
}

_MIN_I64 = -(2**63)
_MAX_I64 = 2**63 - 1

_FREQ_JSON_SCHEMA = {
    "anyOf": [
        {"type": "integer"},
        {"type": "string", "pattern": r"^\s*[+-]?\d+\s*$"},
        {
            "type": "string",
            "pattern": r"^\s*[+-]?\d+(?:\.\d+)?\s*(Hz|kHz|MHz|GHz|hz|khz|mhz|ghz)\s*$",
        },
    ],
    "title": "FrequencyHz",
    "description": "Frequency in Hz (int or numeric string) or string with Hz/kHz/MHz/GHz units.",
}

_TIME_JSON_SCHEMA = {
    "anyOf": [
        {"type": "integer"},
        {"type": "string", "pattern": r"^\s*[+-]?\d+\s*$"},
        {
            "type": "string",
            "pattern": r"^\s*[+-]?\d+(?:\.\d+)?\s*(ps|ns|us|ms|s)\s*$",
        },
    ],
    "title": "TimePS",
    "description": "Time in picoseconds (int or numeric string) or string with ps/ns/us/ms/s units.",
}


def _check_i64(value: int) -> int:
    if value < _MIN_I64 or value > _MAX_I64:
        raise ValueError("Value is outside the signed 64-bit integer range.")
    return value


def _decimal_from_number(text: str, type_name: str) -> Decimal:
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric value for {type_name}: {text!r}") from exc


def _decimal_to_i64(value: Decimal, type_name: str) -> int:
    if value != value.to_integral_value():
        raise ValueError(f"{type_name} must resolve to an integer.")
    return _check_i64(int(value))


def parse_frequency_hz(value: str | int | float) -> int:
    """Parse a frequency value to Hz as int."""
    if isinstance(value, bool):
        raise ValueError("FrequencyHz does not accept boolean values.")
    if isinstance(value, int):
        return _check_i64(value)
    if isinstance(value, float):
        hz_decimal = _decimal_from_number(str(value), "FrequencyHz")
        return _decimal_to_i64(hz_decimal, "FrequencyHz")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("FrequencyHz requires a numeric value.")
        if _INTEGER_RE.match(text):
            return _check_i64(int(text))
        match = _FREQ_RE.match(text)
        if not match:
            raise ValueError(
                "FrequencyHz string must be formatted like '1GHz', '100MHz', or '1000000Hz'."
            )
        number_text, unit = match.groups()
        unit = unit.lower()
        scale = _FREQ_SCALES_HZ.get(unit)
        if scale is None:
            raise ValueError(f"Unknown FrequencyHz unit: {unit!r}")
        hz_decimal = _decimal_from_number(number_text, "FrequencyHz") * Decimal(scale)
        return _decimal_to_i64(hz_decimal, "FrequencyHz")
    raise ValueError(f"Unsupported FrequencyHz value: {value!r}")


def parse_time_ps(value: str | int | float) -> int:
    """Parse a time value to picoseconds as int."""
    if isinstance(value, bool):
        raise ValueError("TimePS does not accept boolean values.")
    if isinstance(value, int):
        return _check_i64(value)
    if isinstance(value, float):
        ps_decimal = _decimal_from_number(str(value), "TimePS")
        return _decimal_to_i64(ps_decimal, "TimePS")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("TimePS requires a numeric value.")
        if _INTEGER_RE.match(text):
            return _check_i64(int(text))
        match = _TIME_RE.match(text)
        if not match:
            raise ValueError(
                "TimePS string must be formatted like '1ns', '100ps', or '1us'."
            )
        number_text, unit = match.groups()
        unit = unit.lower()
        scale = _TIME_SCALES_PS.get(unit)
        if scale is None:
            raise ValueError(f"Unknown TimePS unit: {unit!r}")
        ps_decimal = _decimal_from_number(number_text, "TimePS") * Decimal(scale)
        return _decimal_to_i64(ps_decimal, "TimePS")
    raise ValueError(f"Unsupported TimePS value: {value!r}")


FrequencyHz = Annotated[int, BeforeValidator(parse_frequency_hz), WithJsonSchema(_FREQ_JSON_SCHEMA)]
TimePS = Annotated[int, BeforeValidator(parse_time_ps), WithJsonSchema(_TIME_JSON_SCHEMA)]
