#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path

KERNEL_DEPENDENCY_ALLOWLIST = {
    "serde",
    "serde_json",
    "unicode-casefold",
    "unicode-normalization",
    "unicode-segmentation",
}

# These are the only cross-domain dependencies grandfathered at the start of A2.
# The debt ledger must shrink as they are removed. Adding another name requires
# changing this guard explicitly rather than merely editing the ledger.
ORIGINAL_CROSS_DOMAIN_DEPENDENCIES = {
    "jobs-core": "jobs_core",
    "media-core": "media_core",
    "runtime-core": "runtime_core",
}

# These are the only parallel mirror-contract names grandfathered at A2 start.
# The ledger records which ones still exist and where; it must shrink with the
# implementation rather than remaining a permanent permission list.
ORIGINAL_MIRROR_CONTRACTS = {
    "TextDocumentContract",
    "TextSegmentContract",
    "TimebaseContract",
    "TimestampContract",
}

CONTRACT_PATTERN = re.compile(
    r"\bpub\s+struct\s+([A-Za-z_][A-Za-z0-9_]*Contract)\b"
)
# Existing direct literals all initialize the canonical byte fields first. Keep
# this narrower than `TextSpan {`, which also occurs in function return syntax
# such as `fn span_for_text(...) -> TextSpan {`.
LEGACY_TEXT_SPAN_CONSTRUCTOR_PATTERN = re.compile(
    r"\bTextSpan\s*\{\s*byte_start\s*:"
)


def _contains_import(content: str, crate_name: str) -> bool:
    return f"{crate_name}::" in content or f"use {crate_name}" in content


def _display_paths(paths: set[Path]) -> str:
    return ", ".join(sorted(path.as_posix() for path in paths)) or "<none>"


def _load_debt(root: Path) -> tuple[dict, list[str]]:
    debt_path = root / "scripts" / "text_core_a2_debt.json"
    if not debt_path.is_file():
        return {}, ["missing scripts/text_core_a2_debt.json"]
    try:
        debt = json.loads(debt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return {}, [f"invalid text-core A2 debt ledger: {error}"]
    if debt.get("schemaVersion") != 1:
        return debt, ["text-core A2 debt ledger must use schemaVersion 1"]
    return debt, []


def check_contract(root: Path) -> list[str]:
    errors: list[str] = []
    text_root = root / "crates" / "text"
    core = text_root / "text-core"
    cargo_path = core / "Cargo.toml"
    src = core / "src"

    if not cargo_path.is_file():
        return ["missing crates/text/text-core/Cargo.toml"]
    if not src.is_dir():
        return ["missing crates/text/text-core/src"]

    debt, debt_errors = _load_debt(root)
    errors.extend(debt_errors)
    if debt_errors:
        return errors

    declared_dependencies = set(debt.get("crossDomainDependencies", []))
    unknown_declared_dependencies = sorted(
        declared_dependencies - ORIGINAL_CROSS_DOMAIN_DEPENDENCIES.keys()
    )
    if unknown_declared_dependencies:
        errors.append(
            "text-core A2 ledger declares unapproved cross-domain dependencies: "
            + ", ".join(unknown_declared_dependencies)
        )

    cargo = tomllib.loads(cargo_path.read_text(encoding="utf-8"))
    dependencies = set(cargo.get("dependencies", {}))
    actual_cross_domain_dependencies = (
        dependencies & ORIGINAL_CROSS_DOMAIN_DEPENDENCIES.keys()
    )
    if actual_cross_domain_dependencies != declared_dependencies:
        errors.append(
            "text-core cross-domain dependency debt does not match ledger: "
            f"declared {', '.join(sorted(declared_dependencies)) or '<none>'}; "
            f"actual {', '.join(sorted(actual_cross_domain_dependencies)) or '<none>'}"
        )

    unexpected_dependencies = sorted(
        dependencies - KERNEL_DEPENDENCY_ALLOWLIST - declared_dependencies
    )
    if unexpected_dependencies:
        errors.append(
            "text-core dependency surface grew beyond the A2 boundary: "
            + ", ".join(unexpected_dependencies)
        )

    declared_source_files_raw = debt.get("crossDomainSourceFiles", {})
    declared_source_files = {
        crate_name: {Path(value) for value in values}
        for crate_name, values in declared_source_files_raw.items()
    }
    known_rust_crates = set(ORIGINAL_CROSS_DOMAIN_DEPENDENCIES.values())
    unknown_source_crates = sorted(declared_source_files.keys() - known_rust_crates)
    if unknown_source_crates:
        errors.append(
            "text-core A2 ledger declares unapproved source dependency names: "
            + ", ".join(unknown_source_crates)
        )

    actual_source_files = {crate_name: set() for crate_name in known_rust_crates}
    actual_contract_locations: dict[str, set[Path]] = {}
    for path in sorted(src.rglob("*.rs")):
        relative = path.relative_to(core)
        content = path.read_text(encoding="utf-8")

        for crate_name in known_rust_crates:
            if _contains_import(content, crate_name):
                actual_source_files[crate_name].add(relative)

        for contract_name in CONTRACT_PATTERN.findall(content):
            actual_contract_locations.setdefault(contract_name, set()).add(relative)

    for crate_name in sorted(known_rust_crates):
        declared = declared_source_files.get(crate_name, set())
        actual = actual_source_files[crate_name]
        if actual != declared:
            errors.append(
                f"{crate_name} source debt does not match ledger: "
                f"declared {_display_paths(declared)}; actual {_display_paths(actual)}"
            )

    declared_contracts_raw = debt.get("mirrorContracts", {})
    declared_contracts = {
        name: Path(location) for name, location in declared_contracts_raw.items()
    }
    unknown_declared_contracts = sorted(
        declared_contracts.keys() - ORIGINAL_MIRROR_CONTRACTS
    )
    if unknown_declared_contracts:
        errors.append(
            "text-core A2 ledger declares unapproved mirror *Contract types: "
            + ", ".join(unknown_declared_contracts)
        )

    unknown_actual_contracts = sorted(
        actual_contract_locations.keys() - ORIGINAL_MIRROR_CONTRACTS
    )
    if unknown_actual_contracts:
        errors.append(
            "text-core gained unapproved mirror *Contract types during A2: "
            + ", ".join(unknown_actual_contracts)
        )

    for contract_name in sorted(ORIGINAL_MIRROR_CONTRACTS):
        declared_location = declared_contracts.get(contract_name)
        actual_locations = actual_contract_locations.get(contract_name, set())
        expected_locations = (
            {declared_location} if declared_location is not None else set()
        )
        if actual_locations != expected_locations:
            errors.append(
                f"{contract_name} debt does not match ledger: "
                f"declared {_display_paths(expected_locations)}; "
                f"actual {_display_paths(actual_locations)}"
            )

    declared_span_constructors = {
        Path(value) for value in debt.get("legacyTextSpanConstructors", [])
    }
    actual_span_constructors: set[Path] = set()
    for path in sorted(text_root.rglob("*.rs")):
        if core in path.parents:
            continue
        content = path.read_text(encoding="utf-8")
        if LEGACY_TEXT_SPAN_CONSTRUCTOR_PATTERN.search(content):
            actual_span_constructors.add(path.relative_to(root))

    if actual_span_constructors != declared_span_constructors:
        errors.append(
            "legacy direct TextSpan construction debt does not match ledger: "
            f"declared {_display_paths(declared_span_constructors)}; "
            f"actual {_display_paths(actual_span_constructors)}"
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        parser.error("--check is required")

    root = Path(__file__).resolve().parents[1]
    errors = check_contract(root)
    if errors:
        for error in errors:
            print(f"text-core-a2: {error}")
        return 1

    print("text-core A2 boundary: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
