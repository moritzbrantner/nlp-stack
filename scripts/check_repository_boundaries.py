#!/usr/bin/env python3
"""Validate nlp-stack ownership, registry dependencies, and adapter boundaries."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from repository_split import (
    BUN_PACKAGE_COUNT,
    CARGO_PACKAGE_COUNT,
    DESTINATION_REPOSITORY,
    EXTRACTION_SHA,
    FOUNDATION_DEPENDENCIES,
    OWNERSHIP_PATH,
    PHASE_A_BASELINE,
    ROOT,
    SOURCE_PACKAGE_COUNT,
    SOURCE_REPOSITORY,
    SOURCE_OWNERSHIP_RECORDS_SHA256,
    cargo_metadata,
    inside_root,
    load_json,
    ownership_records_sha256,
)

FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def immutable_git_source(source: str) -> bool:
    """Accept only an exact requested revision plus Cargo's resolved commit."""

    if not source.startswith("git+"):
        return True
    parsed = urlsplit(source[4:])
    query = parse_qs(parsed.query, keep_blank_values=True)
    revisions = query.get("rev", [])
    return (
        set(query) == {"rev"}
        and len(revisions) == 1
        and FULL_SHA_RE.fullmatch(revisions[0]) is not None
        and FULL_SHA_RE.fullmatch(parsed.fragment) is not None
    )


def bun_manifests(root: Path = ROOT) -> dict[str, tuple[Path, dict]]:
    manifests: dict[str, tuple[Path, dict]] = {}
    for path in sorted((root / "packages").glob("*/package.json")):
        document = load_json(path)
        name = document.get("name")
        if isinstance(name, str):
            manifests[name] = (path, document)
    return manifests


def validate(
    metadata: dict,
    ownership: dict,
    root: Path = ROOT,
    bun_packages: dict[str, tuple[Path, dict]] | None = None,
) -> list[str]:
    errors: list[str] = []
    expected_header = {
        "schema_version": 1,
        "repository": DESTINATION_REPOSITORY,
        "source_repository": SOURCE_REPOSITORY,
        "phase_a_baseline": PHASE_A_BASELINE,
        "extraction_sha": EXTRACTION_SHA,
        "source_ownership_records_sha256": SOURCE_OWNERSHIP_RECORDS_SHA256,
        "browser_implementation_owner": "moritzbrantner/platform-packages",
        "platform_packages_ownership_checked": True,
        "npm_publication_authorized": False,
    }
    for key, expected in expected_header.items():
        if ownership.get(key) != expected:
            errors.append(f"{key} must be {expected!r}")

    records = ownership.get("packages")
    if not isinstance(records, list):
        return errors + ["packages must be a list"]
    names = [record.get("current_package_name") for record in records]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        errors.append("packages classified more than once: " + ", ".join(duplicates))

    source_records = [
        record
        for record in records
        if record.get("provenance", {}).get("kind") != "destination-authored"
    ]
    cargo_records = {
        record.get("current_package_name"): record
        for record in records
        if record.get("ecosystem") == "cargo"
    }
    bun_records = {
        record.get("current_package_name"): record
        for record in records
        if record.get("ecosystem") == "bun"
    }
    if len(source_records) != SOURCE_PACKAGE_COUNT:
        errors.append(
            f"ownership must retain exactly {SOURCE_PACKAGE_COUNT} source records, "
            f"found {len(source_records)}"
        )
    if len(cargo_records) != CARGO_PACKAGE_COUNT:
        errors.append(
            f"ownership must contain exactly {CARGO_PACKAGE_COUNT} Cargo packages, "
            f"found {len(cargo_records)}"
        )
    if len(bun_records) != BUN_PACKAGE_COUNT:
        errors.append(
            f"ownership must contain exactly {BUN_PACKAGE_COUNT} Bun packages, "
            f"found {len(bun_records)}"
        )
    if ownership_records_sha256(ownership) != SOURCE_OWNERSHIP_RECORDS_SHA256:
        errors.append("source ownership records differ from the extraction inventory")

    cargo_packages = {package["name"]: package for package in metadata.get("packages", [])}
    missing_cargo = sorted(set(cargo_packages) - set(cargo_records))
    extra_cargo = sorted(set(cargo_records) - set(cargo_packages))
    if missing_cargo:
        errors.append("unclassified Cargo packages: " + ", ".join(missing_cargo))
    if extra_cargo:
        errors.append("Cargo ownership entries absent from metadata: " + ", ".join(extra_cargo))

    actual_bun = bun_packages if bun_packages is not None else bun_manifests(root)
    missing_bun = sorted(set(actual_bun) - set(bun_records))
    extra_bun = sorted(set(bun_records) - set(actual_bun))
    if missing_bun:
        errors.append("unclassified Bun packages: " + ", ".join(missing_bun))
    if extra_bun:
        errors.append("Bun ownership entries absent from workspace: " + ", ".join(extra_bun))

    for record in records:
        name = record.get("current_package_name")
        if record.get("target_repository") != "nlp-stack":
            errors.append(f"{name}: wrong target repository")
        if record.get("intended_next_release_owner") != DESTINATION_REPOSITORY:
            errors.append(f"{name}: wrong release owner")
        manifest = record.get("manifest_path")
        if not isinstance(manifest, str):
            errors.append(f"{name}: missing manifest_path")
            continue
        path = inside_root(root, manifest)
        if path is None:
            errors.append(f"{name}: manifest_path escapes repository")
        elif not path.is_file():
            errors.append(f"{name}: manifest_path does not exist")
        if record.get("wrapped_library") not in {None, *cargo_packages}:
            errors.append(f"{name}: invalid wrapped_library {record.get('wrapped_library')!r}")
        if record.get("ecosystem") == "bun":
            if record.get("automatic_publish_eligible") is not False:
                errors.append(f"{name}: npm/WASM publication must remain separately authorized")
            if name != "@moritzbrantner/nlp-app-ui" and record.get("publication_class") != "separate npm/WASM release plan required":
                errors.append(f"{name}: npm/WASM ownership must retain the separate release gate")

    for package in cargo_packages.values():
        for dependency in package.get("dependencies", []):
            dependency_name = dependency.get("name")
            dep_path = dependency.get("path")
            source = dependency.get("source") or ""
            if dep_path:
                try:
                    Path(dep_path).resolve().relative_to(root.resolve())
                except ValueError:
                    errors.append(f"{package['name']}: dependency path escapes repository")
            if not immutable_git_source(source):
                errors.append(f"{package['name']}: non-immutable Git dependency {source}")
            if not isinstance(dependency_name, str) or not dependency_name.startswith("moenarch-"):
                continue
            if dependency_name in cargo_packages:
                continue
            expected_req = FOUNDATION_DEPENDENCIES.get(dependency_name)
            if expected_req is None:
                errors.append(
                    f"{package['name']}: forbidden NLP dependency on {dependency_name}"
                )
                continue
            if dep_path or not source.startswith("registry+"):
                errors.append(
                    f"{package['name']}: foundation dependency {dependency_name} must resolve from the registry"
                )
            if dependency.get("req") != expected_req:
                errors.append(
                    f"{package['name']}: foundation dependency {dependency_name} must use {expected_req}"
                )

    for name, (_, manifest) in actual_bun.items():
        dependencies: dict[str, str] = {}
        for field in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
            value = manifest.get(field, {})
            if isinstance(value, dict):
                dependencies.update(value)
        if "@moritzbrantner/video-analysis-ui" in dependencies:
            errors.append(f"{name}: must not absorb the rust-packages compatibility UI facade")
        for dependency_name, requirement in dependencies.items():
            if requirement == "workspace:*" and dependency_name not in actual_bun:
                errors.append(f"{name}: missing workspace dependency {dependency_name}")
            if isinstance(requirement, str) and requirement.startswith("file:"):
                errors.append(f"{name}: local file dependency {dependency_name} is forbidden")
        if name.endswith("-app"):
            if dependencies.get("@moritzbrantner/nlp-app-ui") != "workspace:*":
                errors.append(f"{name}: app must use the focused NLP workbench adapter")
        if name == "@moritzbrantner/nlp-app-ui" and manifest.get("private") is not True:
            errors.append("@moritzbrantner/nlp-app-ui: focused adapter must remain private")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--ownership", type=Path, default=OWNERSHIP_PATH)
    args = parser.parse_args()
    metadata = load_json(args.metadata) if args.metadata else cargo_metadata()
    ownership = load_json(args.ownership)
    errors = validate(metadata, ownership)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        "repository boundaries pass: 56 Cargo and 28 Bun packages; "
        "NLP depends only on exact registry foundation versions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
