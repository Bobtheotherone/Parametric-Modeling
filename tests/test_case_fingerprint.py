from __future__ import annotations

from formula_foundry.oracle.fingerprint import compute_case_fingerprint


def test_fingerprint_includes_all_input_bytes_normalized_config_and_toolchain_digest() -> None:
    inputs = {
        "fab/gerbers/top.gtl": b"alpha",
        "fab/gerbers/bottom.gbl": b"beta",
    }
    config = {
        "case_id": "CAL_THRU_001",
        "format_version": "1.0",
        "frequency": {"start_hz": 1.0, "stop_hz": 2.0},
    }
    toolchain_digest = "sha256:toolchain-v1"

    baseline = compute_case_fingerprint(inputs, config, toolchain_digest)

    inputs_changed = dict(inputs)
    inputs_changed["fab/gerbers/top.gtl"] = b"alpha!"
    assert compute_case_fingerprint(inputs_changed, config, toolchain_digest) != baseline

    inputs_added = dict(inputs)
    inputs_added["fab/gerbers/inner.g2"] = b"gamma"
    assert compute_case_fingerprint(inputs_added, config, toolchain_digest) != baseline

    config_changed = dict(config)
    config_changed["frequency"] = {"start_hz": 1.0, "stop_hz": 3.0}
    assert compute_case_fingerprint(inputs, config_changed, toolchain_digest) != baseline

    assert compute_case_fingerprint(inputs, config, "sha256:toolchain-v2") != baseline


def test_case_fingerprint_is_stable_for_ordering() -> None:
    inputs_a = {
        "b.txt": b"beta",
        "a.txt": b"alpha",
    }
    inputs_b = {
        "a.txt": b"alpha",
        "b.txt": b"beta",
    }
    config_a = {
        "format_version": "1.0",
        "frequency": {"stop_hz": 2.0, "start_hz": 1.0},
        "case_id": "CAL_THRU_001",
    }
    config_b = {
        "case_id": "CAL_THRU_001",
        "frequency": {"start_hz": 1.0, "stop_hz": 2.0},
        "format_version": "1.0",
    }
    toolchain_digest = "sha256:toolchain-v1"

    assert compute_case_fingerprint(inputs_a, config_a, toolchain_digest) == compute_case_fingerprint(
        inputs_b, config_b, toolchain_digest
    )
