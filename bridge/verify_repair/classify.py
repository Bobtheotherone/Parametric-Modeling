"""Failure classification for verify repair.

Classifies verify failures into categories to enable targeted repair strategies.
"""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Any

from bridge.verify_repair.data import VerifyGateResult, VerifySummary


class FailureCategory(str, Enum):
    """Categories of verify failures."""

    # Lint failures (often auto-fixable)
    LINT_RUFF = "lint_ruff"
    LINT_FORMAT = "lint_format"

    # Type checking failures (require code changes)
    TYPECHECK_MYPY = "typecheck_mypy"

    # Test failures
    PYTEST_COLLECTION_IMPORT_ERROR = "pytest_collection_import_error"
    PYTEST_TEST_FAILURE = "pytest_test_failure"

    # Spec lint
    SPEC_LINT_FAILURE = "spec_lint_failure"

    # Environment/dependency issues
    MISSING_DEPENDENCY = "missing_dependency"
    MISSING_MODULE_INTERNAL = "missing_module_internal"  # Internal module not found

    # Tooling issues
    TOOLING_ERROR = "tooling_error"
    TOOLING_TIMEOUT = "tooling_timeout"

    # Git issues
    GIT_DIRTY = "git_dirty"

    # Unknown
    UNKNOWN = "unknown"


# Patterns for detecting specific failure types
IMPORT_ERROR_RE = re.compile(r"ImportError: cannot import name ['\"](\w+)['\"]")
MODULE_NOT_FOUND_RE = re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]")
COLLECTION_ERROR_RE = re.compile(r"ERROR collecting .+\.py")
MYPY_ERROR_RE = re.compile(r"^(.+\.py):(\d+): error: (.+)$", re.MULTILINE)

# Known internal module prefixes (not pip-installable)
INTERNAL_MODULE_PREFIXES = (
    "formula_foundry",
    "tools",
    "bridge",
    "tests",
    "benchmarks",
)

# Known external packages that can be bootstrap-installed
BOOTSTRAP_INSTALLABLE = {
    "numpy",
    "pytest",
    "ruff",
    "mypy",
    "pyyaml",
    "yaml",
    "mlflow",
    "cupy",
}


def _classify_pytest_failure(gate: VerifyGateResult) -> list[FailureCategory]:
    """Classify pytest failure into specific categories."""
    categories: list[FailureCategory] = []
    combined = gate.stdout + "\n" + gate.stderr

    # Check for collection errors first (import errors during collection)
    if COLLECTION_ERROR_RE.search(combined):
        # Distinguish between import errors and module not found
        if IMPORT_ERROR_RE.search(combined):
            categories.append(FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR)
        elif MODULE_NOT_FOUND_RE.search(combined):
            # Check if it's an internal or external module
            for match in MODULE_NOT_FOUND_RE.finditer(combined):
                module = match.group(1).split(".")[0]
                if module in BOOTSTRAP_INSTALLABLE:
                    categories.append(FailureCategory.MISSING_DEPENDENCY)
                elif any(module.startswith(p) for p in INTERNAL_MODULE_PREFIXES):
                    categories.append(FailureCategory.MISSING_MODULE_INTERNAL)
                else:
                    # Could be external or internal - check more carefully
                    categories.append(FailureCategory.MISSING_DEPENDENCY)
            categories.append(FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR)
        else:
            categories.append(FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR)

    # Check for actual test failures (passed collection but tests failed)
    if "FAILED" in combined and "passed" in combined.lower():
        categories.append(FailureCategory.PYTEST_TEST_FAILURE)

    # If no specific category found, mark as test failure
    if not categories and gate.returncode not in (None, 0):
        categories.append(FailureCategory.PYTEST_TEST_FAILURE)

    return categories


def _classify_mypy_failure(gate: VerifyGateResult) -> list[FailureCategory]:
    """Classify mypy failure."""
    categories: list[FailureCategory] = []
    combined = gate.stdout + "\n" + gate.stderr

    if "error:" in combined:
        categories.append(FailureCategory.TYPECHECK_MYPY)

    # Check for import errors in mypy output
    if MODULE_NOT_FOUND_RE.search(combined):
        for match in MODULE_NOT_FOUND_RE.finditer(combined):
            module = match.group(1).split(".")[0]
            if module in BOOTSTRAP_INSTALLABLE:
                categories.append(FailureCategory.MISSING_DEPENDENCY)

    if not categories:
        categories.append(FailureCategory.TYPECHECK_MYPY)

    return categories


def _classify_ruff_failure(gate: VerifyGateResult) -> list[FailureCategory]:
    """Classify ruff failure."""
    return [FailureCategory.LINT_RUFF]


def _classify_spec_lint_failure(gate: VerifyGateResult) -> list[FailureCategory]:
    """Classify spec_lint failure."""
    categories: list[FailureCategory] = []
    combined = gate.stdout + "\n" + gate.stderr

    # Check if spec_lint failure is due to pytest collection errors reported in stderr
    if COLLECTION_ERROR_RE.search(combined) or "ERROR collecting" in combined:
        categories.append(FailureCategory.PYTEST_COLLECTION_IMPORT_ERROR)

    categories.append(FailureCategory.SPEC_LINT_FAILURE)
    return categories


def _classify_git_guard_failure(gate: VerifyGateResult) -> list[FailureCategory]:
    """Classify git_guard failure."""
    return [FailureCategory.GIT_DIRTY]


def _classify_gate(gate: VerifyGateResult) -> list[FailureCategory]:
    """Classify a single gate failure."""
    if gate.passed:
        return []

    if gate.note and "timeout" in gate.note.lower():
        return [FailureCategory.TOOLING_TIMEOUT]

    name = gate.name.lower()

    if name == "pytest":
        return _classify_pytest_failure(gate)
    elif name == "mypy":
        return _classify_mypy_failure(gate)
    elif name == "ruff":
        return _classify_ruff_failure(gate)
    elif name == "spec_lint":
        return _classify_spec_lint_failure(gate)
    elif name == "git_guard":
        return _classify_git_guard_failure(gate)
    else:
        return [FailureCategory.UNKNOWN]


def classify_failures(summary: VerifySummary) -> dict[str, list[FailureCategory]]:
    """Classify all failures in a verify summary.

    Returns a dict mapping gate name to list of failure categories.
    """
    result: dict[str, list[FailureCategory]] = {}
    for gate_name in summary.failed_gates:
        gate = summary.results_by_gate.get(gate_name)
        if gate:
            result[gate_name] = _classify_gate(gate)
        else:
            result[gate_name] = [FailureCategory.UNKNOWN]
    return result


def get_all_categories(classification: dict[str, list[FailureCategory]]) -> set[FailureCategory]:
    """Get all unique categories from a classification result."""
    categories: set[FailureCategory] = set()
    for cats in classification.values():
        categories.update(cats)
    return categories


def extract_import_errors(summary: VerifySummary) -> list[dict[str, Any]]:
    """Extract detailed import error information from verify summary.

    Returns list of dicts with:
    - type: 'import_error' or 'module_not_found'
    - name: the name that couldn't be imported
    - module: the module it was being imported from (for import errors)
    - source_file: the file that triggered the error
    - is_internal: whether this appears to be an internal module
    """
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()

    for gate in summary.results_by_gate.values():
        combined = gate.stdout + "\n" + gate.stderr

        # ImportError: cannot import name 'X' from 'Y'
        for match in IMPORT_ERROR_RE.finditer(combined):
            name = match.group(1)
            key = f"import:{name}"
            if key not in seen:
                seen.add(key)
                errors.append(
                    {
                        "type": "import_error",
                        "name": name,
                        "source_gate": gate.name,
                    }
                )

        # ModuleNotFoundError: No module named 'X'
        for match in MODULE_NOT_FOUND_RE.finditer(combined):
            module = match.group(1)
            key = f"module:{module}"
            if key not in seen:
                seen.add(key)
                root_module = module.split(".")[0]
                is_internal = any(root_module.startswith(p) for p in INTERNAL_MODULE_PREFIXES)
                errors.append(
                    {
                        "type": "module_not_found",
                        "module": module,
                        "root_module": root_module,
                        "is_internal": is_internal,
                        "is_bootstrap_installable": root_module in BOOTSTRAP_INSTALLABLE,
                        "source_gate": gate.name,
                    }
                )

    return errors


def extract_mypy_errors(summary: VerifySummary) -> list[dict[str, str]]:
    """Extract mypy error details from verify summary.

    Returns list of dicts with:
    - file: path to the file
    - line: line number
    - message: error message
    """
    errors: list[dict[str, str]] = []
    gate = summary.results_by_gate.get("mypy")
    if not gate:
        return errors

    combined = gate.stdout + "\n" + gate.stderr
    for match in MYPY_ERROR_RE.finditer(combined):
        errors.append(
            {
                "file": match.group(1),
                "line": match.group(2),
                "message": match.group(3),
            }
        )

    return errors


def compute_failure_signature(summary: VerifySummary) -> str:
    """Compute a signature for the current failure state.

    Used to detect when the same failures repeat (no progress).
    The signature is based on:
    - The set of failed gates
    - First N lines of error output from each failed gate
    """
    parts: list[str] = []
    parts.append(",".join(sorted(summary.failed_gates)))

    for gate_name in sorted(summary.failed_gates):
        gate = summary.results_by_gate.get(gate_name)
        if gate:
            # Take first 20 lines of stderr/stdout for signature
            combined = (gate.stderr + "\n" + gate.stdout).strip()
            lines = combined.split("\n")[:20]
            # Normalize: strip, remove timestamps/paths that change
            normalized = []
            for line in lines:
                line = line.strip()
                # Remove path prefixes that might change
                line = re.sub(r"/[^\s]+/([^/\s]+\.py)", r"\1", line)
                # Remove timestamps
                line = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", "TIMESTAMP", line)
                if line:
                    normalized.append(line)
            parts.append(gate_name + ":" + "|".join(normalized[:10]))

    signature_text = "|||".join(parts)
    return hashlib.sha256(signature_text.encode()).hexdigest()[:16]
