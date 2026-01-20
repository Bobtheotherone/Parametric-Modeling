from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Annotated

from pydantic import BeforeValidator, WithJsonSchema

_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_LENGTH_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*$")
_ANGLE_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*$")
_FREQ_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*$")

_UNIT_SCALES_NM: dict[str, int] = {
    "nm": 1,
    "um": 1_000,
    "mm": 1_000_000,
    "mil": 25_400,
}

# Angle scales: convert to millidegrees (mdeg) as canonical integer unit
_ANGLE_SCALES_MDEG: dict[str, int] = {
    "mdeg": 1,
    "deg": 1_000,
}

# Frequency scales: convert to Hz as canonical integer unit
_FREQ_SCALES_HZ: dict[str, int] = {
    "hz": 1,
    "khz": 1_000,
    "mhz": 1_000_000,
    "ghz": 1_000_000_000,
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

_ANGLE_MDEG_JSON_SCHEMA = {
    "anyOf": [
        {"type": "integer"},
        {"type": "string", "pattern": r"^\s*[+-]?\d+\s*$"},
        {
            "type": "string",
            "pattern": r"^\s*[+-]?\d+(?:\.\d+)?\s*(deg|mdeg)\s*$",
        },
    ],
    "title": "AngleMdeg",
    "description": "Angle in millidegrees (int or numeric string) or a string with deg/mdeg units.",
}

_FREQ_HZ_JSON_SCHEMA = {
    "anyOf": [
        {"type": "integer"},
        {"type": "string", "pattern": r"^\s*[+-]?\d+\s*$"},
        {
            "type": "string",
            "pattern": r"^\s*[+-]?\d+(?:\.\d+)?\s*(Hz|kHz|MHz|GHz|hz|khz|mhz|ghz)\s*$",
        },
    ],
    "title": "FrequencyHz",
    "description": "Frequency in Hz (int or numeric string) or a string with Hz/kHz/MHz/GHz units.",
}


def parse_length_nm(value: str | int | float) -> int:
    """Parse a length value to nanometers as int.

    Accepts:
      - Integer: treated as nanometers
      - Float: must be exactly integer-valued
      - String "0.25mm", "10mil", "250um", "1000nm": converted to nm
      - String "1000": treated as nanometers
    """
    if isinstance(value, bool):
        raise ValueError("LengthNM does not accept boolean values.")
    if isinstance(value, int):
        return _check_i64(value, "LengthNM")
    if isinstance(value, float):
        nm_decimal = _decimal_from_number(str(value), "LengthNM")
        return _decimal_to_i64(nm_decimal, "LengthNM")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("LengthNM requires a numeric value.")
        if _INTEGER_RE.match(text):
            return _check_i64(int(text), "LengthNM")
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
        nm_decimal = _decimal_from_number(number_text, "LengthNM") * Decimal(scale)
        return _decimal_to_i64(nm_decimal, "LengthNM")
    raise ValueError(f"Unsupported LengthNM value: {value!r}")


def _decimal_from_number(text: str, type_name: str) -> Decimal:
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric value for {type_name}: {text!r}") from exc


def _decimal_to_i64(value: Decimal, type_name: str) -> int:
    if value != value.to_integral_value():
        raise ValueError(f"{type_name} must resolve to an integer.")
    return _check_i64(int(value), type_name)


def _check_i64(value: int, type_name: str) -> int:
    if value < _MIN_I64 or value > _MAX_I64:
        raise ValueError(f"{type_name} is outside the signed 64-bit integer range.")
    return value


def parse_angle_mdeg(value: str | int | float) -> int:
    """Parse an angle value to millidegrees as int.

    Accepts:
      - Integer: treated as millidegrees
      - Float: must be exactly integer-valued
      - String "45deg", "-90deg", "1000mdeg": converted to millidegrees
      - String "45000": treated as millidegrees
    """
    if isinstance(value, bool):
        raise ValueError("AngleMdeg does not accept boolean values.")
    if isinstance(value, int):
        return _check_i64(value, "AngleMdeg")
    if isinstance(value, float):
        mdeg_decimal = _decimal_from_number(str(value), "AngleMdeg")
        return _decimal_to_i64(mdeg_decimal, "AngleMdeg")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("AngleMdeg requires a numeric value.")
        if _INTEGER_RE.match(text):
            return _check_i64(int(text), "AngleMdeg")
        match = _ANGLE_RE.match(text)
        if not match:
            raise ValueError(
                "AngleMdeg string must be formatted like '45deg', '90000mdeg', or '-180deg'."
            )
        number_text, unit = match.groups()
        unit = unit.lower()
        scale = _ANGLE_SCALES_MDEG.get(unit)
        if scale is None:
            raise ValueError(f"Unknown AngleMdeg unit: {unit!r}")
        mdeg_decimal = _decimal_from_number(number_text, "AngleMdeg") * Decimal(scale)
        return _decimal_to_i64(mdeg_decimal, "AngleMdeg")
    raise ValueError(f"Unsupported AngleMdeg value: {value!r}")


def parse_frequency_hz(value: str | int | float) -> int:
    """Parse a frequency value to Hz as int.

    Accepts:
      - Integer: treated as Hz
      - Float: must be exactly integer-valued
      - String "1GHz", "100MHz", "1000000Hz", "50kHz": converted to Hz
      - String "1000000000": treated as Hz
    """
    if isinstance(value, bool):
        raise ValueError("FrequencyHz does not accept boolean values.")
    if isinstance(value, int):
        return _check_i64(value, "FrequencyHz")
    if isinstance(value, float):
        hz_decimal = _decimal_from_number(str(value), "FrequencyHz")
        return _decimal_to_i64(hz_decimal, "FrequencyHz")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("FrequencyHz requires a numeric value.")
        if _INTEGER_RE.match(text):
            return _check_i64(int(text), "FrequencyHz")
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


LengthNM = Annotated[int, BeforeValidator(parse_length_nm), WithJsonSchema(_LENGTH_JSON_SCHEMA)]
AngleMdeg = Annotated[int, BeforeValidator(parse_angle_mdeg), WithJsonSchema(_ANGLE_MDEG_JSON_SCHEMA)]
FrequencyHz = Annotated[int, BeforeValidator(parse_frequency_hz), WithJsonSchema(_FREQ_HZ_JSON_SCHEMA)]
