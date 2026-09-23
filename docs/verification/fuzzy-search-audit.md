# Fuzzy-search correctness and distance-work audit

## Baseline and boundaries

Audited source: `aa425fcc8f030731a000f72c736e13d36d2829d3` (main on 2026-09-23).
Original fuzzy module blob: `7875625a43c1431aa2e3d29b114051ca4041a3e8`.

This repair stays inside the existing text-index implementation. It does not
settle A5 ownership, move query policy, change public request shapes, introduce a
production benchmark dependency, or interfere with open A2/jobs-core PRs.

## Confirmed defects and executable regressions

1. **Validation bypass:** when corrections exist, the fuzzy wrapper replaces
   `top_k` and `candidate_limit` before calling the authoritative index search.
   Consequently zero values are accepted, while identical requests without a
   correction are rejected. Validate the original query first, using the existing
   core validator rather than a duplicated validation policy.
2. **Lost phrase evidence:** the core deliberately includes required-phrase hits
   even when their lexical score is zero. The fuzzy merger discarded those hits.
   Adding an unrelated document containing `strategy` could therefore remove the
   valid `locked phrase` result for query `stratgey` with that required phrase.
   Retain finite zero scores; keep core phrase/filter validation and prefer exact
   evidence in ties.
3. **Unbounded inner work:** the distance threshold only controlled acceptance and
   length rejection. Every candidate still evaluated a complete matrix and
   allocated a new row per source character. The private helper now evaluates a
   diagonal band and rotates three row buffers.

`src/surface/fuzzy/audit_regressions.rs` contains seven memory-index regressions,
also exercised against SQLite when its feature is enabled. Six are expected to
fail on the baseline; the no-fabricated-results negative control should pass.

The distance tests include 58,564 exhaustive Unicode/threshold comparisons against
an independent full-matrix oracle, long strings with edits at every position,
empty/equal inputs, OSA-specific counterexamples, and actual recurrence-work
counts. The routine remains **optimal string alignment (restricted adjacent
transpositions)**, not unrestricted Damerau-Levenshtein.

## Deterministic work comparison

Equal-length, unequal strings; no early length/equality return:

| Unicode scalars | Baseline cells | New cells, limit 1 | New cells, limit 2 |
| --- | ---: | ---: | ---: |
| 8 | 64 | 22 | 34 |
| 16 | 256 | 46 | 74 |
| 32 | 1,024 | 94 | 154 |
| 64 | 4,096 | 190 | 314 |

The test-only counter increments inside the production recurrence. For these
fixtures it must equal `n * (2*k + 1) - k*(k + 1)`. At 64 scalars the row-buffer
allocation sites execute 66 times before versus 3 after. Those are row buffers,
not a claim about all allocations. Character decoding/storage remains linear.

These are work counts, **not measured Rust speed-up ratios**. Vocabulary building,
corpus materialization, BM25 rebuilding and per-variant query execution are outside
this kernel comparison and remain important follow-up costs.

## Run correctness and timing separately

```sh
cargo test -p moenarch-text-index surface::fuzzy
cargo test -p moenarch-text-index --features sqlite surface::fuzzy
python3 -m unittest discover -s scripts -p 'test_benchmark_fuzzy_distance.py'
python3 scripts/benchmark_fuzzy_distance.py --output /tmp/fuzzy-distance.json
```

The last command requires only the repository's Rust compiler, Python and Git;
it does not resolve Cargo dependencies. It compiles and runs the real Rust helper
tests first, then compiles an optimized native benchmark. The frozen legacy
function body comes directly from the baseline above; only visibility and signature
formatting changed.
It is test/benchmark code, never linked into the production implementation.

The JSON artifact retains baseline/head identities, actual source hashes, dirty
working-tree status, compiler/target/platform, flags, native test output, every
raw timing sample, medians and before/after ratios. Both implementations run on
the same fixtures in one process with warmups and alternating sample order.
There is no wall-clock CI threshold. Harness unit tests use explicitly mocked
samples; those values are not performance evidence.

The opt-in `fuzzy-distance-evidence` workflow runs this native comparison on
manual dispatch or PRs labeled `performance` and retains the JSON artifact for
90 days. Its separate optional search-regressions job runs the repaired crate in
memory and SQLite configurations, then transplants only the regression tests onto
the original commit. It accepts only the exact six expected test failures plus
one passing negative control, not an arbitrary nonzero exit or compilation error.
A second JSON artifact retains the named before/after results and complete logs.
Both jobs are skipped on the ordinary PR path; existing workspace gates remain.

The Cargo benchmark is also available as
`cargo bench -p moenarch-text-index --bench fuzzy_distance`.

## Reproduce the old failures without copying the fixed implementation

In a disposable worktree at the exact baseline, copy only
`crates/text/text-index/src/surface/fuzzy/audit_regressions.rs` from the repaired
revision and append this declaration to the old `surface/fuzzy.rs`:

```rust
#[cfg(test)]
#[path = "fuzzy/audit_regressions.rs"]
mod audit_regressions;
```

Run `cargo test -p moenarch-text-index audit_regressions` there, then on the repair.
Do not transplant the new distance helper or validation/filter changes into the
baseline. Source-mode worktrees need the same managed foundation setup as normal
repository development.

## Verification recorded during implementation

- Reconstructed baseline fuzzy source matched its Git blob byte-for-byte before
  applying edits.
- An executable Python translation of old/new recurrences agreed on all 58,564
  exhaustive cases and reproduced the work-count table. This is supplementary
  model-level evidence, not execution of the Rust implementation.
- Benchmark-report Python unit tests: 4 passed.
- The authoring container lacks Rust/Bun and a complete checkout; native evidence
  was obtained from GitHub Actions instead of being inferred from Python.
- Hosted kernel run `35884202222` on head
  `8d8abf14e6dbca3db542afd1271a3917a97a3b99`: all four native Rust helper tests
  passed, including the exhaustive oracle and deterministic work ratchets.
- Initial workspace run `35884179462`: ownership/release checks and all 125
  Python tests passed; formatting stopped the later stages. Formatter-suggested
  edits to the new benchmark/helper files were applied in the follow-up commit.
  Do not treat that initial run as a full workspace pass.

## First retained native timing run

Run: https://github.com/moritzbrantner/nlp-stack/actions/runs/35884202222

Artifact `10762575964`, named
`fuzzy-distance-8d8abf14e6dbca3db542afd1271a3917a97a3b99`, contains the raw
`fuzzy-distance.json`. Its source hashes matched the benchmarked files, and the
recorded working tree was clean. Compiler: Rust 1.95.0, optimized native x86_64
Linux. Each row below is the median of seven samples of 1,000 calls in this run,
with both implementations measured in the same process.

| One-edit fixture, limit 1 | Before (ns/call) | After (ns/call) | Before / after |
| --- | ---: | ---: | ---: |
| 8 scalars | 283.491 | 224.733 | 1.26x |
| 16 scalars | 746.157 | 377.540 | 1.98x |
| 32 scalars | 2,244.580 | 644.095 | 3.48x |
| 64 scalars | 8,202.438 | 1,152.149 | 7.12x |

At 64 scalars with limit 2 the same fixture measured 8,185.954 -> 1,598.982
ns/call (5.12x). These are observations for the distance kernel on this hosted
runner, **not whole-index latency or guarantees for other hardware**. The artifact
also includes rejected, equal, Unicode and transposition fixtures; none are
excluded from the raw report. Later head runs retain their own identities rather
than relabeling these measurements as belonging to another commit.
