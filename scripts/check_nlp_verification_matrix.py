#!/usr/bin/env python3
"""Validate the descriptive NLP capability verification matrix.

The matrix is intentionally descriptive. Missing and partial evidence are valid
states; this checker only rejects malformed, contradictory, or stale evidence
records so the matrix cannot silently drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "verification" / "nlp-capability-matrix.json"

ALLOWED_STATUSES = {"present", "partial", "missing", "not_applicable"}
REQUIRED_EVIDENCE = (
    "tests",
    "invariants",
    "evaluation",
    "performance",
    "cross_runtime",
    "external_consumer",
    "coverage",
    "mutation",
)
ALLOWED_DISPOSITIONS = {
    "retain",
    "rename_concrete",
    "conditional_recipe",
    "outer_boundary",
}


def load_matrix(path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _non_empty_strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item for item in value)


def validate_matrix(data: dict[str, Any], root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    if data.get("schemaVersion") != 1:
        errors.append("schemaVersion must be 1")

    capabilities = data.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        return errors + ["capabilities must be a non-empty list"]

    seen_ids: set[str] = set()
    for index, capability in enumerate(capabilities):
        prefix = f"capabilities[{index}]"
        if not isinstance(capability, dict):
            errors.append(f"{prefix} must be an object")
            continue

        capability_id = capability.get("id")
        if not isinstance(capability_id, str) or not capability_id:
            errors.append(f"{prefix}.id must be a non-empty string")
            capability_id = f"<invalid-{index}>"
        elif capability_id in seen_ids:
            errors.append(f"duplicate capability id: {capability_id}")
        seen_ids.add(capability_id)

        package = capability.get("package")
        if not isinstance(package, str) or not package:
            errors.append(f"{capability_id}.package must be a non-empty string")

        crate_path = capability.get("path")
        if not isinstance(crate_path, str) or not crate_path:
            errors.append(f"{capability_id}.path must be a non-empty string")
        elif not (root / crate_path).exists():
            errors.append(f"{capability_id}.path does not exist: {crate_path}")

        disposition = capability.get("targetDisposition")
        if disposition not in ALLOWED_DISPOSITIONS:
            errors.append(
                f"{capability_id}.targetDisposition must be one of {sorted(ALLOWED_DISPOSITIONS)}"
            )

        evidence = capability.get("evidence")
        if not isinstance(evidence, dict):
            errors.append(f"{capability_id}.evidence must be an object")
            continue

        extra_evidence = sorted(set(evidence) - set(REQUIRED_EVIDENCE))
        missing_evidence = sorted(set(REQUIRED_EVIDENCE) - set(evidence))
        if extra_evidence:
            errors.append(f"{capability_id}.evidence has unknown keys: {extra_evidence}")
        if missing_evidence:
            errors.append(f"{capability_id}.evidence is missing keys: {missing_evidence}")

        for evidence_kind in REQUIRED_EVIDENCE:
            item = evidence.get(evidence_kind)
            item_prefix = f"{capability_id}.evidence.{evidence_kind}"
            if not isinstance(item, dict):
                continue
            status = item.get("status")
            if status not in ALLOWED_STATUSES:
                errors.append(f"{item_prefix}.status must be one of {sorted(ALLOWED_STATUSES)}")

            note = item.get("note")
            if not isinstance(note, str) or not note.strip():
                errors.append(f"{item_prefix}.note must explain the evidence state")

            paths = item.get("paths", [])
            commands = item.get("commands", [])
            if not _non_empty_strings(paths):
                errors.append(f"{item_prefix}.paths must be a list of non-empty strings")
                paths = []
            if not _non_empty_strings(commands):
                errors.append(f"{item_prefix}.commands must be a list of non-empty strings")
                commands = []

            if status in {"present", "partial"} and not paths and not commands:
                errors.append(f"{item_prefix} needs at least one path or command for {status} evidence")

            for evidence_path in paths:
                if not (root / evidence_path).exists():
                    errors.append(f"{item_prefix} references missing path: {evidence_path}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate the matrix and exit")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    args = parser.parse_args()

    data = load_matrix(args.matrix)
    errors = validate_matrix(data)
    if errors:
        for error in errors:
            print(f"error: {error}")
        return 1

    print("NLP verification matrix is structurally valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
