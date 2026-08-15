#!/usr/bin/env python3
"""Validate the exact, intentionally non-publishing nlp-stack bootstrap plan."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from check_repository_boundaries import bun_manifests
from repository_split import (
    BUN_PACKAGE_COUNT,
    CARGO_PACKAGE_COUNT,
    DESTINATION_REPOSITORY,
    EXTRACTION_SHA,
    OWNERSHIP_PATH,
    RELEASE_PLAN_PATH,
    ROOT,
    SOURCE_REPOSITORY,
    cargo_metadata,
    inside_root,
    load_json,
)

REQUIRED_CHECKS = {
    "cargo metadata --format-version 1 --no-deps",
    "bun install --frozen-lockfile",
    "python3 scripts/check_repository_boundaries.py --check",
    "python3 scripts/check_release_plan.py --check docs/repository-split/release-plan.json",
    "python3 -m unittest discover -s scripts -p 'test_*.py'",
    "cargo fmt --all -- --check",
    "cargo clippy --workspace --all-targets --all-features -- -D warnings",
    "cargo test --workspace --all-features",
    "cargo test --workspace --no-default-features",
    "cargo doc --workspace --no-deps",
    "cargo package -p <each-public-package> --locked",
    "bun run nlp-app-ui:test",
    "bun run text-app:typecheck",
    "bun run text-app:build",
    "bun run text-wasm:test:all",
    "python3 scripts/repository_split.py --harness-audit --base-ref <reviewed-base-sha>",
}


def manifest_hashes(root: Path, ownership: dict) -> dict[str, str]:
    paths = [root / "Cargo.toml"] + [
        root / record["manifest_path"]
        for record in ownership.get("packages", [])
        if record.get("ecosystem") == "cargo"
    ]
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def package_all(plan: dict, ownership: dict, root: Path = ROOT) -> list[str]:
    """Package every Rust crate without publishing or mutating manifests."""

    before = manifest_hashes(root, ownership)
    patch_lines = ["[patch.crates-io]"]
    records = {
        record["current_package_name"]: record
        for record in ownership.get("packages", [])
        if record.get("ecosystem") == "cargo"
    }
    for name in sorted(records):
        crate = (root / records[name]["manifest_path"]).parent.resolve()
        patch_lines.append(f'"{name}" = {{ path = "{crate}" }}')
    failures: list[str] = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml") as config:
        config.write("\n".join(patch_lines) + "\n")
        config.flush()
        for name in plan.get("dependency_order", []):
            completed = subprocess.run(
                ["cargo", "package", "-p", name, "--locked", "--config", config.name],
                cwd=root,
                check=False,
            )
            if completed.returncode:
                failures.append(name)
            else:
                print(f"PACKAGED {name}")
    if before != manifest_hashes(root, ownership):
        failures.append("tracked Cargo manifests changed during packaging")
    return failures


def validate(plan: dict, ownership: dict, metadata: dict, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    if plan.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if plan.get("repository") != DESTINATION_REPOSITORY:
        errors.append("wrong repository")
    if plan.get("active_release_owner") != SOURCE_REPOSITORY:
        errors.append("wrong active release owner")
    if plan.get("source_sha") != EXTRACTION_SHA:
        errors.append("source_sha must match extraction SHA")
    if plan.get("publication_authorized") is not False:
        errors.append("bootstrap plan must explicitly deny Cargo publication")
    if plan.get("npm_publication_authorized") is not False:
        errors.append("bootstrap plan must explicitly deny npm publication")
    if plan.get("platform_packages_ownership_checked") is not True:
        errors.append("platform-packages ownership check must be recorded")

    records = ownership.get("packages", [])
    cargo_owned = {
        record.get("current_package_name"): record
        for record in records
        if record.get("ecosystem") == "cargo"
    }
    bun_owned = {
        record.get("current_package_name"): record
        for record in records
        if record.get("ecosystem") == "bun"
    }
    packages = plan.get("packages")
    npm_packages = plan.get("npm_packages")
    if not isinstance(packages, list):
        return errors + ["packages must be a list"]
    if not isinstance(npm_packages, list):
        return errors + ["npm_packages must be a list"]

    names = [package.get("name") for package in packages]
    npm_names = [package.get("name") for package in npm_packages]
    duplicates = sorted(name for name, count in Counter(names + npm_names).items() if count > 1)
    if duplicates:
        errors.append("duplicate package names: " + ", ".join(duplicates))
    if len(packages) != CARGO_PACKAGE_COUNT or set(names) != set(cargo_owned):
        errors.append(f"release plan must name all and only the {CARGO_PACKAGE_COUNT} Cargo packages")
    if len(npm_packages) != BUN_PACKAGE_COUNT or set(npm_names) != set(bun_owned):
        errors.append(f"release plan must name all and only the {BUN_PACKAGE_COUNT} Bun packages")

    metadata_packages = {package["name"]: package for package in metadata.get("packages", [])}
    metadata_names = set(metadata_packages)
    if set(names) != metadata_names:
        errors.append("release plan does not match Cargo metadata")
    for package in packages:
        name = package.get("name")
        record = cargo_owned.get(name, {})
        if package.get("intended_next_release_owner") != DESTINATION_REPOSITORY:
            errors.append(f"{name}: wrong intended next release owner")
        if package.get("publish") is not False:
            errors.append(f"{name}: publication is not authorized")
        if package.get("new_version") != package.get("old_version"):
            errors.append(f"{name}: nonpublishing plan must retain version")
        actual_version = metadata_packages.get(name, {}).get("version")
        source_version = record.get("source_version")
        if source_version != actual_version:
            errors.append(f"{name}: ownership source_version does not match workspace version {actual_version!r}")
        if package.get("old_version") != actual_version or package.get("new_version") != actual_version:
            errors.append(f"{name}: planned versions do not match workspace version {actual_version!r}")
        if package.get("expected_tag") is not None:
            errors.append(f"{name}: nonpublishing plan must not declare a tag")
        if package.get("manifest_path") != record.get("manifest_path"):
            errors.append(f"{name}: manifest_path differs from ownership")
        manifest = inside_root(root, str(package.get("manifest_path")))
        if manifest is None:
            errors.append(f"{name}: manifest_path escapes repository")
        elif not manifest.is_file():
            errors.append(f"{name}: manifest_path does not exist")
        actual_dependencies = {
            dependency["name"]
            for dependency in metadata_packages.get(name, {}).get("dependencies", [])
            if dependency.get("name") in metadata_names and dependency.get("kind") != "dev"
        }
        planned_dependencies = package.get("release_dependencies")
        if not isinstance(planned_dependencies, list):
            errors.append(f"{name}: release_dependencies must be a list")
        elif set(planned_dependencies) != actual_dependencies:
            errors.append(f"{name}: release_dependencies do not match workspace metadata")

    actual_bun = bun_manifests(root)
    for package in npm_packages:
        name = package.get("name")
        record = bun_owned.get(name, {})
        manifest = actual_bun.get(name, ({}, {}))[1]
        if package.get("intended_next_release_owner") != DESTINATION_REPOSITORY:
            errors.append(f"{name}: wrong intended next release owner")
        if package.get("publish") is not False:
            errors.append(f"{name}: npm publication is not authorized")
        if package.get("version") != manifest.get("version") or package.get("version") != record.get("source_version"):
            errors.append(f"{name}: npm version differs from ownership or manifest")
        if package.get("manifest_path") != record.get("manifest_path"):
            errors.append(f"{name}: npm manifest_path differs from ownership")
        actual_workspace_dependencies: set[str] = set()
        for field in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
            values = manifest.get(field, {})
            if isinstance(values, dict):
                actual_workspace_dependencies.update(
                    dependency for dependency, requirement in values.items() if requirement == "workspace:*"
                )
        if set(package.get("workspace_dependencies", [])) != actual_workspace_dependencies:
            errors.append(f"{name}: workspace_dependencies do not match package manifest")

    order = plan.get("dependency_order")
    if not isinstance(order, list) or len(order) != len(set(order)) or set(order) != set(names):
        errors.append("dependency_order must contain each Cargo package exactly once")
    positions = {name: index for index, name in enumerate(order or [])}
    for package in packages:
        for dependency in package.get("release_dependencies", []):
            if dependency not in positions or positions[dependency] >= positions.get(package.get("name"), -1):
                errors.append(f"wrong dependency order: {dependency} must precede {package.get('name')}")
    if plan.get("expected_tags") != []:
        errors.append("nonpublishing plan must have no expected tags")
    if plan.get("release_issue") is not None:
        errors.append("nonpublishing plan must not claim a release issue")
    required_checks = plan.get("required_checks")
    if not isinstance(required_checks, list) or set(required_checks) != REQUIRED_CHECKS:
        errors.append("required_checks must match the complete bootstrap gate set")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--package-all", action="store_true")
    parser.add_argument("plan", nargs="?", type=Path, default=RELEASE_PLAN_PATH)
    args = parser.parse_args()
    plan = load_json(args.plan)
    ownership = load_json(OWNERSHIP_PATH)
    errors = validate(plan, ownership, cargo_metadata())
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    if args.package_all:
        failures = package_all(plan, ownership)
        if failures:
            print("error: packaging failed: " + ", ".join(failures), file=sys.stderr)
            return 1
        print("package verification passes: 52 Cargo packages; tracked manifest hashes unchanged")
    else:
        print("release plan passes: 52 Cargo and 28 Bun packages retained; publication is not authorized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
