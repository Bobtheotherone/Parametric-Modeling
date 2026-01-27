# SPDX-License-Identifier: MIT
"""Unit tests for common canonicalization and hashing utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest
from formula_foundry.common import (
    atomic_write_json,
    canonical_hash,
    canonical_json_dumps,
    canonicalize_value,
    sha256_bytes,
)


class Mode(Enum):
    FAST = "fast"
    SLOW = "slow"


@dataclass(frozen=True)
class Sample:
    path: Path
    mode: Mode
    value: float
    tags: set[int]


def test_canonicalize_dataclass_enum_path() -> None:
    sample = Sample(path=Path("root") / "file.txt", mode=Mode.FAST, value=1.25, tags={2, 1})
    text = canonical_json_dumps(sample)
    parsed = json.loads(text)
    assert parsed == {
        "mode": "fast",
        "path": "root/file.txt",
        "tags": [1, 2],
        "value": 1.25,
    }


def test_canonicalize_float_formatting() -> None:
    data = {"neg_zero": -0.0, "value": 0.1}
    text = canonical_json_dumps(data)
    expected_float = repr(0.1)
    assert f'"value":{expected_float}' in text
    assert '"neg_zero":-0.0' in text


def test_canonical_hash_matches_sha256() -> None:
    data = {"b": 2, "a": 1}
    canonical = canonical_json_dumps(data)
    expected = sha256_bytes(canonical.encode("utf-8"))
    assert canonical_hash(data) == expected


def test_atomic_write_json(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    payload = {"alpha": 1, "beta": 2}
    atomic_write_json(path, payload)
    content = path.read_text(encoding="utf-8")
    assert content == f"{canonical_json_dumps(payload)}\n"


def test_canonicalize_set_stable_order() -> None:
    data = {"items": {3, 1, 2}}
    text = canonical_json_dumps(data)
    parsed = json.loads(text)
    assert parsed["items"] == [1, 2, 3]


def test_canonicalize_numpy_array_inline_or_hash() -> None:
    np = pytest.importorskip("numpy")
    small = np.array([1.0, 2.0], dtype=np.float64)
    large = np.arange(128, dtype=np.float64)

    small_text = canonical_json_dumps({"array": small})
    small_parsed = json.loads(small_text)
    assert "__ndarray__" in small_parsed["array"]
    assert "data" in small_parsed["array"]["__ndarray__"]

    large_text = canonical_json_dumps({"array": large})
    large_parsed = json.loads(large_text)
    assert "__ndarray__" in large_parsed["array"]
    assert "sha256" in large_parsed["array"]["__ndarray__"]


def test_canonicalize_value_idempotent() -> None:
    value = {"path": Path("data"), "mode": Mode.SLOW, "values": [1, 2, 3]}
    canonical = canonicalize_value(value)
    again = canonicalize_value(canonical)
    assert canonical == again
