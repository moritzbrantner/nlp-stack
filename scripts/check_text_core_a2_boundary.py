#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

# These are the dependencies already present when A2 starts. The migration may
# remove any of them; adding a new dependency requires an explicit architecture
# decision rather than silently expanding the kernel again.
LEGACY_DEPENDENCY_CEILING = {
    "jobs-core",
    "media-core",
    "runtime-core",
    "serde",
    "serde_json",
    "unicode-casefold",
    "unicode-normalization",
    "unicode-segmentation",
}

# Existing cross-domain imports are migration debt. They may disappear, but
# must not spread to new text-core source files while A2 is in progress.
LEGACY_MEDIA_FILES = {Path("src/lib.rs"), Path("src/contracts.rs")}
LEGACY_RUNTIME_FILES = {
    Path("src/contracts.rs"),
    Path("src/operations.rs"),
    Path("src/surface.rs"),
}
LEGACY_JOBS_FILES = {Path("src/operations.rs")}

# A2 removes the parallel rich *Contract hierarchy. Until that migration is
# complete, the existing names are a ceiling: no additional mirror contract
# type may be introduced in text-core.
LEGACY_CONTRACT_NAMES = {
    "TextDocumentContract",
    "TextSegmentContract",
    "TimebaseContract",
    "TimestampContract",
}


def _contains_import(content: str, crate_name: str) -> bool:
    return f"{crate_name}::" in content or f"use {crate_name}" in content


def check_contract(root: Path) -> list[str]:
    errors: list[str] = []
    core = root / "crates" / "text" / "text-core"
    cargo_path = core / "Cargo.toml"
    src = core / "src"

    if not cargo_path.is_file():
        return ["missing crates/text/text-core/Cargo.toml"]
    if not src.is_dir():
        return ["missing crates/text/text-core/src"]

    cargo = tomllib.loads(cargo_path.read_text(encoding="utf-8"))
    dependencies = set(cargo.get("dependencies", {}))
    unexpected_dependencies = sorted(dependencies - LEGACY_DEPENDENCY_CEILING)
    if unexpected_dependencies:
        errors.append(
            "text-core dependency surface grew beyond the A2 baseline: "
            + ", ".join(unexpected_dependencies)
        )

    for path in sorted(src.rglob("*.rs")):
        relative = path.relative_to(core)
        content = path.read_text(encoding="utf-8")

        if _contains_import(content, "media_core") and relative not in LEGACY_MEDIA_FILES:
            errors.append(
                f"media-core usage spread outside grandfathered A2 debt: {relative.as_posix()}"
            )
        if _contains_import(content, "runtime_core") and relative not in LEGACY_RUNTIME_FILES:
            errors.append(
                f"runtime-core usage spread outside grandfathered A2 debt: {relative.as_posix()}"
            )
        if _contains_import(content, "jobs_core") and relative not in LEGACY_JOBS_FILES:
            errors.append(
                f"jobs-core usage spread outside grandfathered A2 debt: {relative.as_posix()}"
            )

    contracts_path = src / "contracts.rs"
    if contracts_path.is_file():
        contract_names = set(
            re.findall(
                r"\bpub\s+struct\s+([A-Za-z_][A-Za-z0-9_]*Contract)\b",
                contracts_path.read_text(encoding="utf-8"),
            )
        )
        new_contract_names = sorted(contract_names - LEGACY_CONTRACT_NAMES)
        if new_contract_names:
            errors.append(
                "text-core gained new parallel *Contract types during A2: "
                + ", ".join(new_contract_names)
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
