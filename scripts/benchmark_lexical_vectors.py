#!/usr/bin/env python3
"""Compare query-only lexical/fuzzy search with the pre-vector-repair revision.

Only the test/benchmark module and its declaration are transplanted into a
throwaway baseline worktree. Production search is never patched in that worktree.
Correctness/observed work are mandatory; wall-clock timings are observations only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "495a83b23c0454e4ad9a6ffba9dfec634e89472e"
MODULE = Path("crates/text/text-index/src/surface/fuzzy/vector_reads.rs")
FUZZY = Path("crates/text/text-index/src/surface/fuzzy.rs")
DECLARATION = '\n#[cfg(test)]\n#[path = "fuzzy/vector_reads.rs"]\nmod vector_reads;\n'
PREFIX = "surface::fuzzy::vector_reads::"
FIXED = {
    "lexical_performs_no_vector_reads_or_materialization",
    "lexical_survives_unavailable_vector_storage",
    "lexical_preserves_scores_phrases_filters_and_explanations_without_vectors",
    "fuzzy_sqlite_variants_do_not_read_corrupt_vector_payloads",
}
CONTROLS = {
    "semantic_and_hybrid_read_and_use_vectors_once",
    "semantic_and_hybrid_propagate_vector_storage_errors",
    "invalid_query_is_rejected_before_vector_storage",
}


def run(command: list[str], cwd: Path = ROOT, env: dict | None = None,
        check: bool = True) -> subprocess.CompletedProcess:
    if command[0] == "cargo":
        # Cargo can reuse an executable built from another worktree when both
        # checkouts share a target directory. A comparison must isolate outputs.
        env = dict(os.environ if env is None else env)
        parent = Path(env.get("CARGO_TARGET_DIR", str(ROOT / "target"))).resolve()
        checkout = hashlib.sha256(str(cwd.resolve()).encode()).hexdigest()[:16]
        env["CARGO_TARGET_DIR"] = str(parent / "lexical-vector-evidence" / checkout)
    return subprocess.run(command, cwd=cwd, env=env, check=check, text=True,
                          capture_output=True, timeout=900)


def validate_tests(text: str, returncode: int, before: bool) -> dict[str, str]:
    found = re.findall(r"^test " + re.escape(PREFIX) + r"(\w+) \.\.\. (ok|FAILED|ignored)",
                       text, re.MULTILINE)
    results = dict(found)
    expected = {name: "ok" for name in FIXED | CONTROLS}
    expected["query_only_benchmark"] = "ignored"
    if before:
        expected.update({name: "FAILED" for name in FIXED})
    if len(found) != len(results) or results != expected or returncode != (101 if before else 0):
        raise ValueError(f"unexpected {'before' if before else 'after'} test result: {returncode}, {results}\n{text}")
    return results


def parse_samples(text: str) -> dict[tuple[str, int, int], dict]:
    rows = {}
    marker = "NLP_VECTOR_BENCH "
    for line in text.splitlines():
        if marker not in line:
            continue
        row = json.loads(line.split(marker, 1)[1])
        key = (row["kind"], row["documents"], row["dimensions"])
        if key in rows:
            raise ValueError(f"duplicate fixture {key}")
        if not math.isfinite(row["ns_per_query"]) or row["ns_per_query"] <= 0:
            raise ValueError(f"invalid timing {key}")
        rows[key] = row
    expected = {(kind, count, dims) for kind in ("lexical", "fuzzy-one", "fuzzy-two")
                for count in (32, 256) for dims in (128, 1024)}
    if set(rows) != expected:
        raise ValueError(f"incomplete fixture set: {set(rows)}")
    return rows


def validate_pair(before: dict, after: dict) -> None:
    for field in ("kind", "documents", "dimensions", "query", "variants", "iterations", "results"):
        if before[field] != after[field]:
            raise ValueError(f"before/after mismatch in {field}")
    if before["kind"] == "lexical":
        expected_scalars = before["documents"] * before["dimensions"]
        if before["work"] != {"vector_reads": 1, "vector_scalars": expected_scalars}:
            raise ValueError("baseline vector work not reproduced")
        if after["work"] != {"vector_reads": 0, "vector_scalars": 0}:
            raise ValueError("lexical vector-work regression")
    elif before["work"] is not None or after["work"] is not None:
        raise ValueError("fuzzy work must not be presented as directly counted")


def build_benchmark(root: Path, destination: Path) -> None:
    result = run(["cargo", "test", "--locked", "-p", "moenarch-text-index", "--lib",
                  "--no-default-features", "--release", "--no-run", "--message-format=json"], cwd=root)
    executables = [event["executable"] for line in result.stdout.splitlines()
                   if line.startswith("{") for event in [json.loads(line)]
                   if event.get("reason") == "compiler-artifact" and event.get("executable")
                   and event.get("target", {}).get("name") == "text_index"]
    if len(executables) != 1:
        raise ValueError(f"expected one native test binary: {executables}")
    shutil.copy2(executables[0], destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if args.iterations <= 0 or args.samples <= 0:
        parser.error("iterations and samples must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    files = [MODULE, FUZZY, Path("crates/text/text-index/src/lib.rs"),
             Path("Cargo.lock"), Path("scripts/benchmark_lexical_vectors.py")]
    hashes = {str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in files}
    report = {
        "schema": "nlp-stack.lexical-vector-comparison.v1",
        "baseline_commit": BASELINE, "head_commit": head,
        "source_sha256": hashes,
        "working_tree_status": run(["git", "status", "--porcelain"]).stdout.strip(),
        "compiler": run(["rustc", "--version", "--verbose"]).stdout,
        "platform": platform.platform(),
        "scope": "query-only in-memory lexical/fuzzy search plus result serialization; excludes index build",
        "timing_mode": "native release test binaries; warmup; alternating revision order; no timing gate",
        "baseline_changes": "Only vector_reads.rs and its cfg(test) declaration in fuzzy.rs.",
        "work_scope": "Counting store outside timing for direct lexical calls; fuzzy vector isolation is tested with corrupted SQLite payloads, not counted in timings.",
        "build_isolation": "Separate Cargo target directories per checkout; copied release binaries per revision.",
        "validation": {}, "samples": [], "summary": [],
    }
    run(["git", "fetch", "--depth=1", "origin", BASELINE])
    with tempfile.TemporaryDirectory(prefix="nlp-lexical-vectors-") as directory:
        temporary = Path(directory)
        before_root = temporary / "before"
        run(["git", "worktree", "add", "--detach", str(before_root), BASELINE])
        try:
            (before_root / MODULE).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / MODULE, before_root / MODULE)
            with (before_root / FUZZY).open("a", encoding="utf-8") as stream:
                stream.write(DECLARATION)
            for phase, root in (("before", before_root), ("after", ROOT)):
                test = run(["cargo", "test", "--locked", "-p", "moenarch-text-index", "--lib",
                            "--features", "sqlite", PREFIX, "--", "--test-threads=1"], cwd=root, check=False)
                text = test.stdout + test.stderr
                report["validation"][phase] = {
                    "tests": validate_tests(text, test.returncode, before=phase == "before"),
                    "raw_log": text,
                }
            before_bin, after_bin = temporary / "before-bench", temporary / "after-bench"
            build_benchmark(before_root, before_bin)
            build_benchmark(ROOT, after_bin)
            env = dict(os.environ, NLP_VECTOR_BENCH_ITERATIONS=str(args.iterations))
            groups: dict[tuple, list[dict]] = {}
            for sample in range(args.samples):
                measured = {}
                order = ("before", "after") if sample % 2 == 0 else ("after", "before")
                for phase in order:
                    binary = before_bin if phase == "before" else after_bin
                    text = run([str(binary), PREFIX + "query_only_benchmark", "--exact", "--ignored",
                                "--nocapture", "--test-threads=1"], env=env).stdout
                    measured[phase] = parse_samples(text)
                for key in measured["before"]:
                    before, after = measured["before"][key], measured["after"][key]
                    validate_pair(before, after)
                    row = {"sample": sample, "order": order, "before": before, "after": after}
                    groups.setdefault(key, []).append(row)
                    report["samples"].append(row)
            for (kind, count, dimensions), rows in groups.items():
                before = statistics.median(row["before"]["ns_per_query"] for row in rows)
                after = statistics.median(row["after"]["ns_per_query"] for row in rows)
                report["summary"].append({"kind": kind, "documents": count, "dimensions": dimensions,
                    "before_median_ns": before, "after_median_ns": after,
                    "ratio_before_over_after": before / after})
        finally:
            run(["git", "worktree", "remove", "--force", str(before_root)])
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, encoding="utf-8", delete=False) as stream:
        temporary_output = Path(stream.name)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary_output, args.output)
    print(json.dumps(report["summary"], indent=2))
    print(args.output)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"command failed: {error.cmd}\n{error.stdout}\n{error.stderr}") from error
