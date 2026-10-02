#!/usr/bin/env python3
"""Compile/test/time the actual Rust helper against its frozen pre-audit body.

No Cargo dependency resolution, corpus download, or wall-clock pass threshold.
Writes one self-describing JSON report only after all steps succeed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "crates/text/text-index"
BASE = "aa425fcc8f030731a000f72c736e13d36d2829d3"
SOURCES = [
    CRATE / "src/surface/fuzzy/distance.rs",
    CRATE / "benches/support/fuzzy_distance_legacy.rs",
    CRATE / "benches/fuzzy_distance.rs",
]


def run(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    rustc = shutil.which("rustc")
    if rustc is None:
        parser.error("rustc is required; use the repository's pinned Rust toolchain")

    compiler = run([rustc, "--version", "--verbose"]).strip()
    head = run(["git", "rev-parse", "HEAD"]).strip()
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in SOURCES
    }
    dirty = run(["git", "status", "--porcelain"]).strip()
    with tempfile.TemporaryDirectory(prefix="nlp-fuzzy-distance-") as directory:
        suffix = ".exe" if platform.system() == "Windows" else ""
        tests = str(Path(directory) / f"distance-tests{suffix}")
        bench = str(Path(directory) / f"distance-bench{suffix}")
        run([rustc, "--edition=2021", "--test", str(SOURCES[2]), "-o", tests])
        test_output = run([tests])
        run([rustc, "--edition=2021", "-O", str(SOURCES[2]), "-o", bench])
        samples = [json.loads(line) for line in run([bench, str(args.iterations)]).splitlines()]

    groups: dict[tuple[str, int, int], list[dict]] = {}
    for sample in samples:
        key = (sample["fixture"], sample["chars"], sample["limit"])
        groups.setdefault(key, []).append(sample)
    summary = []
    for (fixture, chars, limit), rows in groups.items():
        before = statistics.median(row["before_ns_per_call"] for row in rows)
        after = statistics.median(row["after_ns_per_call"] for row in rows)
        summary.append({
            "fixture": fixture,
            "chars": chars,
            "limit": limit,
            "before_median_ns": before,
            "after_median_ns": after,
            "ratio_before_over_after": before / after if after > 0 else None,
        })
    report = {
        "schema": "nlp-stack.fuzzy-distance-comparison.v1",
        "baseline_commit": BASE,
        "baseline_blob": "7875625a43c1431aa2e3d29b114051ca4041a3e8",
        "head_commit": head,
        "working_tree_status": dirty,
        "source_sha256": hashes,
        "compiler": compiler,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "compile_flags": ["--edition=2021", "-O"],
        "native_test_output": test_output,
        "scope": "edit-distance kernel only; not end-to-end index search",
        "samples": samples,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"command failed ({error.returncode}): {error.cmd}\n{error.stdout}\n{error.stderr}"
        ) from error
