"""Generate an M1 compliance report from verify/audit artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_PATH = Path("artifacts") / "m1_compliance_report.md"
DEFAULT_VERIFY_ROOT = Path("artifacts") / "verify"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _format_path(path: Path | None, project_root: Path) -> str:
    if path is None:
        return "missing"
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _resolve_path(project_root: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    return path


def _resolve_verify_dir(
    project_root: Path,
    verify_dir: str | None,
    verify_run: str | None,
) -> Path | None:
    if verify_dir:
        return _resolve_path(project_root, verify_dir)
    if verify_run:
        return project_root / DEFAULT_VERIFY_ROOT / verify_run
    return _find_latest_verify_run(project_root / DEFAULT_VERIFY_ROOT)


def _find_latest_verify_run(verify_root: Path) -> Path | None:
    if not verify_root.exists():
        return None
    run_dirs = sorted(path for path in verify_root.iterdir() if path.is_dir())
    return run_dirs[-1] if run_dirs else None


def _resolve_first_existing(*paths: Path | None) -> Path | None:
    for path in paths:
        if path and path.exists():
            return path
    return None


def _find_artifact(project_root: Path, filename: str) -> Path | None:
    artifacts_root = project_root / "artifacts"
    if not artifacts_root.exists():
        return None
    matches = sorted(path for path in artifacts_root.rglob(filename) if path.is_file())
    return matches[0] if matches else None


def _bool_status(value: Any) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "UNKNOWN"


def _extract_toolchain_summary(
    manifest: dict[str, Any] | None,
    lock_data: dict[str, Any] | None,
) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []

    if manifest:
        toolchain = manifest.get("toolchain")
        if isinstance(toolchain, dict):
            kicad = toolchain.get("kicad", {})
            if isinstance(kicad, dict):
                entries.append(("manifest.kicad.version", str(kicad.get("version", "unknown"))))
                entries.append(
                    ("manifest.kicad.cli_version_output", str(kicad.get("cli_version_output", "unknown")))
                )
            docker = toolchain.get("docker", {})
            if isinstance(docker, dict):
                entries.append(("manifest.docker.image_ref", str(docker.get("image_ref", "unknown"))))
            entries.append(("manifest.mode", str(toolchain.get("mode", "unknown"))))
            entries.append(("manifest.generator_git_sha", str(toolchain.get("generator_git_sha", "unknown"))))
            if "lock_file_toolchain_hash" in toolchain:
                entries.append(
                    ("manifest.lock_file_toolchain_hash", str(toolchain.get("lock_file_toolchain_hash", "unknown")))
                )

    if lock_data:
        entries.append(("lock.kicad_version", str(lock_data.get("kicad_version", "unknown"))))
        entries.append(("lock.docker_image", str(lock_data.get("docker_image", "unknown"))))
        entries.append(("lock.docker_digest", str(lock_data.get("docker_digest", "unknown"))))
        entries.append(("lock.toolchain_hash", str(lock_data.get("toolchain_hash", "unknown"))))

    return entries


def _truncate_list(values: list[Any], *, max_items: int) -> tuple[list[Any], int]:
    if len(values) <= max_items:
        return values, 0
    return values[:max_items], len(values) - max_items


def _extract_drc_excerpt(drc_report: dict[str, Any] | None) -> dict[str, Any]:
    if not drc_report:
        return {"note": "DRC report not available"}

    excerpt: dict[str, Any] = {}
    for key in ("source", "violations", "warnings", "exclusions"):
        if key not in drc_report:
            continue
        value = drc_report[key]
        if isinstance(value, list):
            trimmed, extra = _truncate_list(value, max_items=3)
            excerpt[key] = trimmed
            if extra:
                excerpt[f"{key}_truncated"] = extra
        else:
            excerpt[key] = value

    if not excerpt:
        excerpt = {"note": "No expected DRC keys found"}

    return excerpt


def _extract_manifest_excerpt(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest:
        return {"note": "Manifest not available"}

    excerpt: dict[str, Any] = {}
    for key in ("schema_version", "coupon_family", "design_hash", "toolchain_hash"):
        if key in manifest:
            excerpt[key] = manifest[key]

    toolchain = manifest.get("toolchain")
    if isinstance(toolchain, dict):
        excerpt["toolchain"] = toolchain

    exports = manifest.get("exports")
    if isinstance(exports, list):
        trimmed, extra = _truncate_list(exports, max_items=3)
        excerpt["exports_sample"] = trimmed
        if extra:
            excerpt["exports_truncated"] = extra

    verification = manifest.get("verification")
    if isinstance(verification, dict):
        excerpt["verification"] = verification

    if not excerpt:
        excerpt = {"note": "No expected manifest keys found"}

    return excerpt


def _render_verify_table(results: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Gate | Status | Note |",
        "| --- | --- | --- |",
    ]
    for item in sorted(results, key=lambda entry: entry.get("name", "")):
        name = item.get("name", "unknown")
        status = _bool_status(item.get("passed"))
        note = item.get("note", "")
        lines.append(f"| {name} | {status} | {note} |")
    return lines


def _render_audit_table(gates: dict[str, Any]) -> list[str]:
    lines = [
        "| Gate | Status | Tests | Passed | Failed | Skipped |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for gate_id in sorted(gates.keys()):
        gate = gates[gate_id]
        if not isinstance(gate, dict):
            continue
        lines.append(
            "| {gate} | {status} | {tests} | {passed} | {failed} | {skipped} |".format(
                gate=gate_id,
                status=gate.get("status", "unknown"),
                tests=gate.get("tests_count", 0),
                passed=gate.get("passed", 0),
                failed=gate.get("failed", 0),
                skipped=gate.get("skipped", 0),
            )
        )
    return lines


def build_report(
    *,
    project_root: Path,
    verify_dir: Path | None,
    verify_results: dict[str, Any] | None,
    verify_results_path: Path | None,
    audit_report: dict[str, Any] | None,
    audit_report_path: Path | None,
    manifest: dict[str, Any] | None,
    manifest_path: Path | None,
    drc_report: dict[str, Any] | None,
    drc_report_path: Path | None,
    toolchain_lock: dict[str, Any] | None,
    toolchain_lock_path: Path | None,
) -> str:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    verify_run_id = verify_dir.name if verify_dir else "unknown"

    lines: list[str] = [
        "# M1 Compliance Report",
        "",
        f"Generated: {generated_at}",
        f"Project root: {_format_path(project_root, project_root)}",
        f"Verify run: {verify_run_id}",
        "",
        "## Inputs",
        f"- verify_results: {_format_path(verify_results_path, project_root)}",
        f"- audit_report: {_format_path(audit_report_path, project_root)}",
        f"- manifest: {_format_path(manifest_path, project_root)}",
        f"- drc_report: {_format_path(drc_report_path, project_root)}",
        f"- toolchain_lock: {_format_path(toolchain_lock_path, project_root)}",
        "",
        "## Verify Summary",
    ]

    if verify_results:
        lines.append(f"- ok: {verify_results.get('ok', False)}")
        lines.append("")
        lines.extend(_render_verify_table(verify_results.get("results", [])))
    else:
        lines.append("- No verify results available.")

    lines.append("")
    lines.append("## M1 Gate Results (audit-m1)")
    if audit_report:
        lines.append(f"- overall_status: {audit_report.get('overall_status', 'unknown')}")
        lines.append(f"- pytest_returncode: {audit_report.get('pytest_returncode', 'unknown')}")
        lines.append("")
        gates = audit_report.get("gates", {})
        if isinstance(gates, dict) and gates:
            lines.extend(_render_audit_table(gates))
        else:
            lines.append("- No gate details found in audit report.")
    else:
        lines.append("- No audit report available.")

    lines.append("")
    lines.append("## Toolchain Versions")
    toolchain_entries = _extract_toolchain_summary(manifest, toolchain_lock)
    if toolchain_entries:
        for key, value in toolchain_entries:
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No toolchain metadata available.")

    lines.append("")
    lines.append("## Sample DRC Report Excerpt")
    lines.append("```json")
    lines.append(json.dumps(_extract_drc_excerpt(drc_report), indent=2, sort_keys=True))
    lines.append("```")

    lines.append("")
    lines.append("## Sample Manifest Excerpt")
    lines.append("```json")
    lines.append(json.dumps(_extract_manifest_excerpt(manifest), indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate M1 compliance report from verify/audit artifacts.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--verify-dir", default="", help="Path to a verify run directory.")
    parser.add_argument("--verify-run", default="", help="Verify run id under artifacts/verify.")
    parser.add_argument("--audit-report", default="", help="Path to audit_report.json.")
    parser.add_argument("--manifest", default="", help="Path to manifest.json to excerpt.")
    parser.add_argument("--drc-report", default="", help="Path to drc.json to excerpt.")
    parser.add_argument("--toolchain-lock", default="", help="Path to toolchain lock file.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output markdown path (default: artifacts/m1_compliance_report.md).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verify_dir and args.verify_run:
        parser.error("Use only one of --verify-dir or --verify-run.")

    project_root = Path(args.project_root).resolve()

    verify_dir = _resolve_verify_dir(project_root, args.verify_dir or None, args.verify_run or None)
    verify_results_path = verify_dir / "results.json" if verify_dir else None
    verify_results = _load_json(verify_results_path) if verify_results_path and verify_results_path.exists() else None

    audit_report_path = _resolve_path(project_root, args.audit_report or None)
    if audit_report_path is None:
        audit_report_path = _resolve_first_existing(
            project_root / "artifacts" / "audit" / "audit_report.json",
            project_root / "artifacts" / "audit_report.json",
        )
    audit_report = _load_json(audit_report_path) if audit_report_path else None

    manifest_path = _resolve_path(project_root, args.manifest or None)
    if manifest_path is None:
        manifest_path = _find_artifact(project_root, "manifest.json")
    manifest = _load_json(manifest_path) if manifest_path else None

    drc_report_path = _resolve_path(project_root, args.drc_report or None)
    if drc_report_path is None:
        drc_report_path = _find_artifact(project_root, "drc.json")
    drc_report = _load_json(drc_report_path) if drc_report_path else None

    toolchain_lock_path = _resolve_path(project_root, args.toolchain_lock or None)
    if toolchain_lock_path is None:
        toolchain_lock_path = _resolve_first_existing(project_root / "toolchain" / "kicad.lock.json")
    toolchain_lock = _load_json(toolchain_lock_path) if toolchain_lock_path else None

    report_text = build_report(
        project_root=project_root,
        verify_dir=verify_dir,
        verify_results=verify_results,
        verify_results_path=verify_results_path,
        audit_report=audit_report,
        audit_report_path=audit_report_path,
        manifest=manifest,
        manifest_path=manifest_path,
        drc_report=drc_report,
        drc_report_path=drc_report_path,
        toolchain_lock=toolchain_lock,
        toolchain_lock_path=toolchain_lock_path,
    )

    output_path = _resolve_path(project_root, args.output)
    if output_path is None:
        output_path = project_root / DEFAULT_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")

    missing_required = []
    if verify_results is None:
        missing_required.append("verify results")
    if audit_report is None:
        missing_required.append("audit report")

    return 1 if missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
