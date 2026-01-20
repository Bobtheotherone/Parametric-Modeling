#!/usr/bin/env python3
"""
Schema validation script for M3 artifact, dataset, and run schemas.

This script validates JSON files against the M3 JSON schemas and can also
validate the schemas themselves for correctness.

Usage:
    python validate_schemas.py --self-test           # Validate schemas are valid JSON Schema
    python validate_schemas.py artifact file.json    # Validate a file against artifact schema
    python validate_schemas.py dataset file.json     # Validate a file against dataset schema
    python validate_schemas.py run file.json         # Validate a file against run schema
    python validate_schemas.py --all-schemas dir/    # Validate all JSON files in directory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def get_schema_dir() -> Path:
    """Get the directory containing the schemas."""
    return Path(__file__).parent.parent / "schemas"


def load_json(path: Path) -> dict[str, Any]:
    """Load and parse a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_schema_path(schema_type: str) -> Path:
    """Get the path to a schema file by type."""
    schema_dir = get_schema_dir()
    schema_map = {
        "artifact": "artifact.v1.schema.json",
        "dataset": "dataset.v1.schema.json",
        "run": "run.v1.schema.json",
    }
    if schema_type not in schema_map:
        raise ValueError(f"Unknown schema type: {schema_type}. Valid types: {list(schema_map.keys())}")
    return schema_dir / schema_map[schema_type]


def validate_json_schema_syntax(schema_path: Path) -> tuple[bool, str]:
    """
    Validate that a schema is valid JSON and follows JSON Schema structure.
    Returns (success, message).
    """
    try:
        schema = load_json(schema_path)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except FileNotFoundError:
        return False, f"Schema file not found: {schema_path}"

    # Check required top-level keys
    required_keys = ["$schema", "$id", "title", "type"]
    missing = [k for k in required_keys if k not in schema]
    if missing:
        return False, f"Missing required schema keys: {missing}"

    # Check that $schema references a valid JSON Schema draft
    valid_drafts = [
        "https://json-schema.org/draft/2020-12/schema",
        "https://json-schema.org/draft/2019-09/schema",
        "http://json-schema.org/draft-07/schema#",
    ]
    if schema.get("$schema") not in valid_drafts:
        return False, f"$schema should reference a valid JSON Schema draft. Got: {schema.get('$schema')}"

    # Check that properties and required are consistent
    if "required" in schema and "properties" in schema:
        properties = set(schema.get("properties", {}).keys())
        required = set(schema.get("required", []))
        undefined_required = required - properties
        if undefined_required:
            return False, f"Required fields not defined in properties: {undefined_required}"

    return True, "Schema syntax is valid"


def validate_document(document_path: Path, schema_type: str) -> tuple[bool, str]:
    """
    Validate a JSON document against a schema.
    Returns (success, message).
    """
    try:
        import jsonschema
        from jsonschema import Draft202012Validator
    except ImportError:
        return False, "jsonschema package not installed. Run: pip install jsonschema"

    schema_path = get_schema_path(schema_type)

    try:
        schema = load_json(schema_path)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return False, f"Failed to load schema: {e}"

    try:
        document = load_json(document_path)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in document: {e}"
    except FileNotFoundError:
        return False, f"Document file not found: {document_path}"

    # Validate using jsonschema
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(document))

    if errors:
        error_messages = []
        for error in errors[:10]:  # Limit to first 10 errors
            path = ".".join(str(p) for p in error.absolute_path) or "(root)"
            error_messages.append(f"  - {path}: {error.message}")
        return False, "Validation errors:\n" + "\n".join(error_messages)

    return True, "Document is valid"


def self_test() -> int:
    """Validate all schema files for correctness."""
    schema_dir = get_schema_dir()
    schemas = [
        "artifact.v1.schema.json",
        "dataset.v1.schema.json",
        "run.v1.schema.json",
    ]

    all_passed = True
    for schema_file in schemas:
        schema_path = schema_dir / schema_file
        success, message = validate_json_schema_syntax(schema_path)
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {schema_file}: {message}")
        if not success:
            all_passed = False

    # Also try to load schemas with jsonschema if available
    try:
        import jsonschema
        from jsonschema import Draft202012Validator

        print("\nValidating schemas with jsonschema library...")
        for schema_file in schemas:
            schema_path = schema_dir / schema_file
            try:
                schema = load_json(schema_path)
                Draft202012Validator.check_schema(schema)
                print(f"[PASS] {schema_file}: Valid JSON Schema 2020-12")
            except jsonschema.SchemaError as e:
                print(f"[FAIL] {schema_file}: {e.message}")
                all_passed = False
            except Exception as e:
                print(f"[FAIL] {schema_file}: {e}")
                all_passed = False
    except ImportError:
        print("\nNote: jsonschema package not installed. Install for full validation: pip install jsonschema")

    return 0 if all_passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate JSON files against M3 schemas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Validate that all schema files are valid JSON Schema",
    )

    parser.add_argument(
        "schema_type",
        nargs="?",
        choices=["artifact", "dataset", "run"],
        help="Schema type to validate against",
    )

    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="JSON file to validate",
    )

    parser.add_argument(
        "--all-schemas",
        type=Path,
        metavar="DIR",
        help="Validate all JSON files in directory against inferred schemas",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if args.all_schemas:
        # Validate all JSON files in directory
        all_passed = True
        for json_file in args.all_schemas.glob("**/*.json"):
            # Try to infer schema type from filename
            name = json_file.stem.lower()
            if "artifact" in name:
                schema_type = "artifact"
            elif "dataset" in name:
                schema_type = "dataset"
            elif "run" in name:
                schema_type = "run"
            else:
                if args.verbose:
                    print(f"[SKIP] {json_file}: Cannot infer schema type")
                continue

            success, message = validate_document(json_file, schema_type)
            status = "PASS" if success else "FAIL"
            print(f"[{status}] {json_file} ({schema_type})")
            if not success:
                all_passed = False
                if args.verbose:
                    print(message)

        return 0 if all_passed else 1

    if args.schema_type and args.file:
        success, message = validate_document(args.file, args.schema_type)
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {args.file}: {message}")
        return 0 if success else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
