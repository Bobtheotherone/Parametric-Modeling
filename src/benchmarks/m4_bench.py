"""M4 autofit benchmark CLI wrapper.

Provides a CLI interface for running M4 vector fitting benchmarks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from formula_foundry.m4.benchmarks import AutoFitBenchmarkConfig, run_autofit_benchmark
from formula_foundry.m4.types import PoleInitPolicy


def main(argv: Sequence[str] | None = None) -> int:
    """Run M4 autofit benchmark with CLI arguments.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = AutoFitBenchmarkConfig(
        n_ports=args.n_ports,
        n_freqs=args.n_freqs,
        f_min_hz=args.f_min_hz,
        f_max_hz=args.f_max_hz,
        n_poles_real=args.n_poles_real,
        n_poles_complex=args.n_poles_complex,
        pole_init_policy=PoleInitPolicy.LOGARITHMIC,
        include_constant_term=True,
        include_proportional_term=False,
        residue_scale=0.05,
        d_matrix_scale=0.1,
        noise_scale=0.0,
        seed=args.seed,
        max_poles=args.max_poles,
        min_poles=args.min_poles,
        add_poles=args.add_poles,
        improvement_threshold=args.improvement_threshold,
        spurious_relative_threshold=args.spurious_relative_threshold,
        spurious_absolute_threshold=args.spurious_absolute_threshold,
        max_duration_s=args.max_duration_s,
        max_rms_error=args.max_rms_error,
    )

    try:
        report = run_autofit_benchmark(config)
        _emit_json(report, args.json)
        return 0 if report.get("status") == "pass" else 1
    except Exception as exc:
        error_report: dict[str, Any] = {
            "bench": "m4-autofit",
            "status": "error",
            "error": str(exc),
        }
        _emit_json(error_report, args.json)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(description="M4 autofit benchmark")
    parser.add_argument("--n-ports", type=int, default=4)
    parser.add_argument("--n-freqs", type=int, default=1001)
    parser.add_argument("--f-min-hz", type=float, default=1e6)
    parser.add_argument("--f-max-hz", type=float, default=2e9)
    parser.add_argument("--n-poles-real", type=int, default=2)
    parser.add_argument("--n-poles-complex", type=int, default=4)
    parser.add_argument("--max-poles", type=int, default=12)
    parser.add_argument("--min-poles", type=int, default=2)
    parser.add_argument("--add-poles", type=int, default=1)
    parser.add_argument("--improvement-threshold", type=float, default=0.05)
    parser.add_argument("--spurious-relative-threshold", type=float, default=0.05)
    parser.add_argument("--spurious-absolute-threshold", type=float, default=0.0)
    parser.add_argument("--max-duration-s", type=float, default=5.0)
    parser.add_argument("--max-rms-error", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json", "-j", default="-", help="JSON output file (- for stdout)")
    return parser


def _emit_json(report: dict[str, Any], target: str) -> None:
    """Emit JSON report to file or stdout."""
    text = json.dumps(report, indent=2, sort_keys=True)
    if target == "-":
        print(text)
        return
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(f"{text}\n")


if __name__ == "__main__":
    sys.exit(main())
