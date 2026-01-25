"""CLI helpers for spec coverage linting and explain output."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

from formula_foundry.coupongen.api import load_spec, validate_spec_with_engine
from formula_foundry.coupongen.constraints import ConstraintViolationError
from formula_foundry.resolve.consumption import build_spec_consumption
from formula_foundry.substrate import canonical_json_dumps


def run_lint_spec_coverage(args: argparse.Namespace) -> int:
    """Execute the lint-spec-coverage command."""
    spec_path = Path(args.spec)
    if not spec_path.exists():
        sys.stderr.write(f"Spec file not found: {spec_path}\n")
        return 1

    try:
        spec = load_spec(spec_path)
    except Exception as exc:  # noqa: BLE001 - surface parse/validation errors
        sys.stderr.write(f"Failed to load spec: {exc}\n")
        return 1

    consumption = build_spec_consumption(spec)

    unused = _get_unused_provided(consumption)
    unconsumed = _get_unconsumed_expected(consumption)
    coverage_ratio = _get_coverage_ratio(consumption)
    is_complete = not unused and not unconsumed
    exit_code = 0 if (is_complete or not args.strict) else 1

    if args.json:
        payload = {
            "spec_path": str(spec_path),
            "coupon_family": spec.coupon_family,
            "coverage_ratio": coverage_ratio,
            "is_complete": is_complete,
            "unused_provided_paths": sorted(unused),
            "unconsumed_expected_paths": sorted(unconsumed),
        }
        sys.stdout.write(canonical_json_dumps(payload) + "\n")
        return exit_code

    lines = [
        f"Spec coverage for {spec_path}",
        f"Coupon family: {spec.coupon_family}",
        f"Coverage ratio: {coverage_ratio:.3f}",
        f"Unused provided paths: {len(unused)}",
        f"Unconsumed expected paths: {len(unconsumed)}",
    ]
    if unused:
        lines.append(f"Unused provided: {', '.join(sorted(unused))}")
    if unconsumed:
        lines.append(f"Unconsumed expected: {', '.join(sorted(unconsumed))}")
    sys.stdout.write("\n".join(lines) + "\n")
    return exit_code


def run_explain(args: argparse.Namespace) -> int:
    """Execute the explain command."""
    spec_path = Path(args.spec)
    if not spec_path.exists():
        sys.stderr.write(f"Spec file not found: {spec_path}\n")
        return 1

    try:
        spec = load_spec(spec_path)
    except Exception as exc:  # noqa: BLE001 - surface parse/validation errors
        sys.stderr.write(f"Failed to load spec: {exc}\n")
        return 1

    constraint_mode = args.constraint_mode or spec.constraints.mode

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_spec_with_engine(
                spec,
                out_dir=Path(tmpdir),
                mode=constraint_mode,
            )
    except ConstraintViolationError as exc:
        if args.json:
            payload = {
                "status": "constraint_violation",
                "tier": exc.tier,
                "constraint_ids": exc.constraint_ids,
            }
            sys.stdout.write(canonical_json_dumps(payload) + "\n")
        else:
            sys.stderr.write(f"{exc}\n")
        return 1
    except Exception as exc:  # noqa: BLE001 - surface validation errors
        sys.stderr.write(f"Failed to validate spec: {exc}\n")
        return 1

    resolved_payload = result.resolved.model_dump(mode="json")
    tightest = _tightest_constraints_by_category(result.engine_result.to_proof_document())

    payload = {
        "spec_path": str(spec_path),
        "coupon_family": result.spec.coupon_family,
        "constraint_mode": constraint_mode,
        "was_repaired": result.was_repaired,
        "constraints_passed": result.proof.passed,
        "total_constraints": len(result.proof.constraints),
        "resolved_design": resolved_payload,
        "tightest_constraints_by_category": tightest,
    }

    if args.json:
        output = canonical_json_dumps(payload) + "\n"
    else:
        output = _format_explain_report(payload)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(output, encoding="utf-8")
        sys.stdout.write(f"Explain report written to: {out_path}\n")
        return 0

    sys.stdout.write(output)
    return 0


def _get_unused_provided(consumption: Any) -> frozenset[str]:
    if hasattr(consumption, "unused_provided_paths"):
        return frozenset(consumption.unused_provided_paths)
    provided = getattr(consumption, "provided_paths", frozenset())
    consumed = getattr(consumption, "consumed_paths", frozenset())
    return frozenset(set(provided) - set(consumed))


def _get_unconsumed_expected(consumption: Any) -> frozenset[str]:
    if hasattr(consumption, "unconsumed_expected_paths"):
        return frozenset(consumption.unconsumed_expected_paths)
    expected = getattr(consumption, "expected_paths", frozenset())
    consumed = getattr(consumption, "consumed_paths", frozenset())
    return frozenset(set(expected) - set(consumed))


def _get_coverage_ratio(consumption: Any) -> float:
    ratio = getattr(consumption, "coverage_ratio", None)
    if ratio is not None:
        return float(ratio)
    expected = set(getattr(consumption, "expected_paths", frozenset()))
    if not expected:
        return 1.0
    consumed = set(getattr(consumption, "consumed_paths", frozenset()))
    return len(consumed & expected) / len(expected)


def _tightest_constraints_by_category(proof_document: Any) -> dict[str, dict[str, Any]]:
    tightest: dict[str, dict[str, Any]] = {}
    categories = getattr(proof_document, "min_margin_by_category", None)
    if not categories:
        return tightest

    for category, summary in categories.items():
        summary_dict = summary.to_dict()
        if "min_margin_constraint_id" in summary_dict:
            summary_dict["constraint_id"] = summary_dict.pop("min_margin_constraint_id")
        tightest[category] = summary_dict
    return tightest


def _format_explain_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"EXPLAIN: {payload['spec_path']}")
    lines.append("")
    lines.append("SPEC SUMMARY")
    lines.append(f"Coupon family: {payload['coupon_family']}")
    lines.append(f"Constraint mode: {payload['constraint_mode']}")
    lines.append(f"Was repaired: {payload['was_repaired']}")
    lines.append("")
    lines.append("RESOLVED DESIGN")
    lines.append(canonical_json_dumps(payload["resolved_design"]))
    lines.append("")
    lines.append("CONSTRAINT STATUS")
    lines.append(f"Passed: {payload['constraints_passed']}")
    lines.append(f"Total constraints: {payload['total_constraints']}")
    lines.append("")
    lines.append("TIGHTEST CONSTRAINTS BY CATEGORY")
    tightest = payload["tightest_constraints_by_category"]
    if tightest:
        for category, info in sorted(tightest.items()):
            lines.append(
                f"- {category}: min_margin_nm={info.get('min_margin_nm')} "
                f"constraint_id={info.get('constraint_id')} "
                f"constraint_count={info.get('constraint_count')} "
                f"failed_count={info.get('failed_count')} "
                f"passed_count={info.get('passed_count')}"
            )
    else:
        lines.append("No constraint margin summary available.")
    lines.append("")
    return "\n".join(lines) + "\n"
