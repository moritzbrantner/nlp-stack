from pathlib import Path

root = Path('.')
path = root / 'crates/text/text-index/src/lib.rs'
old = '''        let vectors = self
            .store
            .vectors()?
            .into_iter()
            .map(|vector| (vector.chunk_id.clone(), vector))
            .collect::<BTreeMap<_, _>>();'''
new = '''        // Lexical search must not load or depend on unused semantic state.
        let vectors = if query.mode == IndexSearchMode::Lexical {
            BTreeMap::new()
        } else {
            self.store
                .vectors()?
                .into_iter()
                .map(|vector| (vector.chunk_id.clone(), vector))
                .collect::<BTreeMap<_, _>>()
        };'''
text = path.read_text()
assert text.count(old) == 1, 'expected exactly the audited vector loading block'
path.write_text(text.replace(old, new))
fuzzy = root / 'crates/text/text-index/src/surface/fuzzy.rs'
text = fuzzy.read_text()
assert 'mod vector_reads;' not in text
fuzzy.write_text(text + '\n#[cfg(test)]\n#[path = "fuzzy/vector_reads.rs"]\nmod vector_reads;\n')
workflow = root / '.github/workflows/fuzzy-distance-evidence.yml'
text = workflow.read_text()
assert '  lexical-vectors:' not in text
workflow.write_text(text + '''
  lexical-vectors:
    if: github.event_name == 'workflow_dispatch' || contains(github.event.pull_request.labels.*.name, 'performance')
    runs-on: ubuntu-latest
    timeout-minutes: 20
    env:
      CARGO_INCREMENTAL: "0"
      CARGO_PROFILE_DEV_DEBUG: "0"
      CARGO_PROFILE_TEST_DEBUG: "0"
    steps:
      - uses: actions/checkout@8e8c483db84b4bee98b60c0593521ed34d9990e8 # v6.0.1
        with:
          ref: ${{ github.event.pull_request.head.sha || github.sha }}
      - uses: dtolnay/rust-toolchain@stable
        with:
          toolchain: "1.95.0"
      - name: Query-only correctness, vector-work ratchet and paired timing
        run: |
          export CARGO_TARGET_DIR="$RUNNER_TEMP/lexical-vector-target"
          python3 scripts/benchmark_lexical_vectors.py --iterations 10 --samples 5 \\
            --output "$RUNNER_TEMP/lexical-vectors.json"
      - uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
        with:
          name: lexical-vectors-${{ github.event.pull_request.head.sha || github.sha }}
          path: ${{ runner.temp }}/lexical-vectors.json
          if-no-files-found: error
          retention-days: 90
''')
(root / 'docs/verification/lexical-vector-audit.md').write_text('''# Lexical search vector isolation

Issue #98 follows the fuzzy correctness/distance audit in PR #97.
The isolated vector-work baseline is `495a83b23c0454e4ad9a6ffba9dfec634e89472e`,
which already contains the fuzzy fixes and benchmark compilation correction.

The search method used to clone every stored embedding into a map before
checking the search mode. Lexical queries did not use those vectors, but paid
their loading cost and failed when vector storage was unavailable. The repair
leaves semantic/hybrid evaluation and error ordering intact and skips vector
loading only in lexical mode. Fuzzy variants use that same lexical implementation.

## Executable correctness and work evidence

`surface/fuzzy/vector_reads.rs` contains six memory-store tests and a SQLite
fuzzy test. The counting store verifies zero vector reads/materialized scalars
for lexical search, with semantic/hybrid positive controls. Error-injection
checks show lexical search continues without vector storage while the other
modes still propagate failures. A result-equality test covers ranking, phrases,
metadata filtering and explanations.

The SQLite test enables four actual fuzzy variants, corrupts every vector
payload, proves the corruption is detected by vector and semantic/hybrid reads,
and requires exactly unchanged fuzzy search results.

The paired runner transplants only this test module and its declaration onto
the baseline. It requires the four named failures, three passing controls and
one intentionally ignored timing test before; all seven regressions must pass
after. Compilation failure or an arbitrary nonzero exit is not accepted.

## Query-only benchmark

```sh
python3 scripts/benchmark_lexical_vectors.py --output /tmp/lexical-vectors.json
python3 -m unittest discover -s scripts -p 'test_benchmark_lexical_vectors.py'
cargo test -p moenarch-text-index --features sqlite vector_reads
```

The opt-in performance workflow compares native release test binaries from both
revisions on the same host with warmup and alternating revision order. Twelve
fixtures vary 32/256 documents, 128/1024 vector dimensions and lexical versus
one/two misspelled query terms. Index construction is excluded; search and result
serialization are included. Every paired result is checked for full JSON equality.

Direct lexical work is counted outside the timing region; the actual unwrapped
memory store is timed. Each baseline lexical query reads vectors once and loads
`documents * dimensions` scalars; the repaired path must read zero and load zero.
Fuzzy timings do not claim directly counted vector reads: their isolation is
proved by the separate corruption regression. Timings have no pass/fail threshold.

The JSON includes actual commit/compiler/source identities, lockfile hash,
working-tree status, exact before/after test logs, complete results and all raw
paired samples. The workflow preserves the report for 90 days. Python harness
tests contain synthetic values and are not timing evidence.

No public APIs, ownership, dependency versions or persistence formats change.
No frontend changes are included. The ordinary workspace gate is unchanged.
''')
