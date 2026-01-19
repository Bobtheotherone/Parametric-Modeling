from __future__ import annotations

import json
from pathlib import Path

from formula_foundry.coupongen import load_spec
from formula_foundry.coupongen.resolve import design_hash, resolve

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SPECS_DIR = ROOT / "tests" / "golden_specs"
GOLDEN_HASHES_PATH = ROOT / "tests" / "golden_hashes" / "design_hashes.json"


def _golden_specs() -> list[Path]:
    patterns = ("*.json", "*.yaml", "*.yml")
    specs: list[Path] = []
    for pattern in patterns:
        specs.extend(sorted(GOLDEN_SPECS_DIR.glob(pattern)))
    return sorted(specs)


def test_golden_specs_present() -> None:
    specs = _golden_specs()
    f0_specs = []
    f1_specs = []
    for path in specs:
        spec = load_spec(path)
        if spec.coupon_family == "F0_CAL_THRU_LINE":
            f0_specs.append(path)
        elif spec.coupon_family == "F1_SINGLE_ENDED_VIA":
            f1_specs.append(path)

    assert len(f0_specs) >= 10
    assert len(f1_specs) >= 10


def test_golden_hashes_match() -> None:
    specs = _golden_specs()
    expected = json.loads(GOLDEN_HASHES_PATH.read_text(encoding="utf-8"))
    mapping = expected.get("spec_hashes", {})

    for path in specs:
        spec = load_spec(path)
        resolved = resolve(spec)
        digest = design_hash(resolved)
        key = path.name
        assert key in mapping
        assert digest == mapping[key]
