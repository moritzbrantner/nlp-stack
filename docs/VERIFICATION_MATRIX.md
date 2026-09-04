# NLP verification matrix

`verification/nlp-capability-matrix.json` is the repository's descriptive evidence inventory for durable NLP capability candidates.

It exists to answer a narrow question before architectural migrations and feature work: **what evidence do we actually have for correctness, invariants, semantic quality, performance, runtime parity, real consumers, coverage, and mutation resistance?**

## Policy

The matrix is not a scorecard and it does not create quality thresholds.

- `present` means capability-specific evidence exists and is directly reproducible.
- `partial` means useful evidence exists but material risks or environments remain uncovered.
- `missing` is an accepted descriptive state. Missing evidence must not fail a pull request merely because it is missing.
- `not_applicable` is reserved for evidence that does not belong to that capability boundary.

The validator fails only when the matrix itself is malformed or stale: duplicate capability identities, unknown statuses, missing required evidence categories, invalid capability paths, or evidence paths that no longer exist. A future change that turns any evidence category into a merge threshold is a separate policy decision and must be reviewed as such.

## Scope

Rows represent candidate durable semantic capabilities plus the aggregate registry boundary. The matrix deliberately does **not** enumerate every extraction-era CLI/server/WASM/app package, because A9/A10 are intended to collapse ceremonial outer surfaces rather than grant them durability through documentation.

Likewise, explicit retirement targets such as `text-model-runtime` and pairwise bridge packages are not promoted into capability rows. Their safe removal is migration work, not a reason to build a permanent verification program around them.

Current target dispositions are recorded explicitly:

- `retain`: expected durable semantic owner;
- `rename_concrete`: useful current behavior whose generic package name is scheduled to become concrete;
- `conditional_recipe`: composition survives only if A8 proves independent reuse;
- `outer_boundary`: non-semantic aggregate/transport boundary.

## Evidence categories

Every row records the same categories so absence stays visible:

- `tests`: focused package-level correctness/regression tests;
- `invariants`: generated/property/idempotence/determinism evidence beyond examples;
- `evaluation`: versioned gold or calibration corpora with deterministic metrics;
- `performance`: reproducible benchmarks for latency, throughput, scaling, or memory-relevant behavior;
- `cross_runtime`: native/WASM/browser or backend parity/integration evidence;
- `external_consumer`: evidence from a real independent consumer or source-mode compatibility check;
- `coverage`: descriptive line/branch coverage evidence;
- `mutation`: mutation-testing evidence that tests detect meaningful behavioral changes.

These categories are intentionally broader than the normal workspace CI. Existing CI already proves format, strict Clippy, default/no-default builds/tests, docs, UI/WASM smoke, and packaging; the matrix is meant to expose the semantic and algorithmic evidence that those gates do not describe on their own.

## Validation

Run:

```bash
python3 scripts/check_nlp_verification_matrix.py --check
python3 -m unittest scripts.test_nlp_verification_matrix
```

The repository's existing `python3 -m unittest discover -s scripts -p 'test_*.py'` command also exercises the matrix contract, so structural drift is covered by the normal validation path without introducing semantic-quality thresholds.

## Next evidence work

Use the matrix to choose narrow improvements rather than to maximize filled cells. The highest-value gaps are expected to be capability-specific property/invariant suites, larger provenance-recorded evaluation corpora, geometric performance envelopes, descriptive coverage, and targeted mutation testing. Evidence should be added where it changes confidence in a real capability or migration decision; decorative completeness is not a goal.
