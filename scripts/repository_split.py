#!/usr/bin/env python3
"""Shared helpers for the nlp-stack ownership and release validators."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNERSHIP_PATH = ROOT / "docs/repository-split/package-ownership.json"
RELEASE_PLAN_PATH = ROOT / "docs/repository-split/release-plan.json"
SOURCE_REPOSITORY = "moritzbrantner/rust-packages"
DESTINATION_REPOSITORY = "moritzbrantner/nlp-stack"
PHASE_A_BASELINE = "d032ad2890c1df3c6a5b9eff024562f00d017fce"
EXTRACTION_SHA = "b8b29cf8db0b86ed1b133a18155adf24992f9483"
SOURCE_OWNERSHIP_RECORDS_SHA256 = (
    "0693b1cb2aed72e91a1612acc9a311ed9e3f77970e769cdb3ab5fce932c49f93"
)
SOURCE_PACKAGE_COUNT = 79
CARGO_PACKAGE_COUNT = 56
BUN_PACKAGE_COUNT = 28

FOUNDATION_DEPENDENCIES = {
    "moenarch-data-inversion-core": "=0.1.1",
    "moenarch-jobs-core": "=0.1.2",
    "moenarch-math-sparse-data": "=0.1.1",
    "moenarch-media-core": "=0.1.0",
    "moenarch-model-runtime": "=0.1.1",
    "moenarch-runtime-core": "=0.2.1",
    "moenarch-runtime-onnx": "=0.1.1",
    "moenarch-vector-analysis-core": "=0.1.1",
    "moenarch-vector-analysis-index": "=0.1.1",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def cargo_metadata(root: Path = ROOT) -> dict:
    completed = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def inside_root(root: Path, value: str, base: Path | None = None) -> Path | None:
    root = root.resolve()
    candidate = ((base or root) / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def ownership_records(document: dict) -> list[dict]:
    records = document.get("packages", [])
    return records if isinstance(records, list) else []


def ownership_records_sha256(document: dict) -> str:
    records = sorted(
        (
            record
            for record in ownership_records(document)
            if record.get("provenance", {}).get("kind") != "destination-authored"
        ),
        key=lambda record: str(record.get("current_package_name")),
    )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness-audit", action="store_true")
    parser.add_argument("--base-ref", required=True)
    args = parser.parse_args()
    if not args.harness_audit:
        parser.error("--harness-audit is required")
    resolved_base = subprocess.run(
        ["git", "rev-parse", "--verify", f"{args.base_ref}^{{commit}}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if resolved_base.returncode != 0:
        parser.error(f"--base-ref does not resolve to a commit: {args.base_ref}")
    base_sha = resolved_base.stdout.strip()
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    harness = (
        codex_home
        / "skills/moenarch-verification-harness/scripts/verification_harness.py"
    )
    requirements = ROOT / ".agent-loop/verification/requirements.json"
    return subprocess.run(
        [
            sys.executable,
            str(harness),
            "audit",
            "--repo-root",
            str(ROOT),
            "--base-ref",
            base_sha,
            "--requirements-bundle",
            str(requirements),
            "--json",
        ],
        cwd=ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
