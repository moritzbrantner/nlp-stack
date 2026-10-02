# nlp-stack

`nlp-stack` is the canonical text and natural-language capability repository for the Moenarch ecosystem. It owns NLP source, architecture, tests, issues, versions, and releases for capabilities assigned to this domain.

The repository was extracted from `moritzbrantner/rust-packages`; see [docs/PROVENANCE.md](docs/PROVENANCE.md). Historical copies in `rust-packages` are compatibility/provenance material rather than a second source of truth.

The current workspace still contains the broad extraction-era package, adapter, WASM, and demo inventory. That inventory is transitional. The target architecture intentionally reduces it to semantic capability libraries, earned adapters, a thin aggregate registry boundary, and one default NLP workbench. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before treating an existing package boundary as durable.

Ownership does not itself authorize publication, tags, releases, consumer migration, or source removal. Those remain explicit release or migration tasks.

## Browser workbench

The static GitHub Pages workbench can ingest text-bearing files, PDFs, images, and pasted text locally in the browser, then run the existing Rust/Wasm analysis surfaces without an `nlp-stack` API server.

Analysis runs in a dedicated worker for each studio job. Both studios offer cancellation, which terminates active analysis and releases the worker; queued operations within a job run serially. Browser execution accepts at most 256 KiB of UTF-8 source text and 512 sentences across all supplied documents, counting repeated sentences separately. Oversized inputs are rejected with guidance rather than truncated. The sentence ceiling bounds quadratic clustering storage; it is a browser policy, not a native Rust API limit. Worker requests have a 60-second deadline, including initialization and queue time.

The corpus-baseline flow exports a versioned `nlp-stack.semantic-corpus-baseline` JSON artifact containing the extracted source text, provenance, semantic-corpus options, and Rust semantic result. The artifact deliberately omits generated timestamps so unchanged inputs and analysis output serialize deterministically.

## Source development

Normal feature work may use the exact `moenarch-foundation` revision declared in `.coding-tooling.source-deps.json` without publishing intermediate crates. Run `bash scripts/source-deps activate` before cross-repository work and `bash scripts/source-deps deactivate` before registry-only release verification.

See [docs/SOURCE_DEVELOPMENT.md](docs/SOURCE_DEVELOPMENT.md).

## Local verification

```bash
bun install --frozen-lockfile
cargo metadata --locked --format-version 1
python3 scripts/check_repository_boundaries.py --check
python3 scripts/check_release_plan.py --check docs/repository-split/release-plan.json
python3 -m unittest discover -s scripts -p 'test_*.py'
cargo test --locked --workspace --all-features
cargo test --locked --workspace --no-default-features
cargo doc --locked --workspace --no-deps
bun run nlp-app-ui:test
bun run text-app:typecheck
bun run text-app:build
bun run text-wasm:test:all
```

`scripts/check-preflight.sh` runs the Rust, shared UI, and capability-app gate. `scripts/check.sh` adds archive verification for every Rust package. Cargo verification and WASM builds use `--locked`: prepare source-mode resolution explicitly as described in [source development](docs/SOURCE_DEVELOPMENT.md) before running either gate.

For changes to Pages, its shared UI, or its Rust/WASM dependencies, also run the canonical Pages gate:

```bash
# One-time browser acquisition after installing the pinned web dependencies:
bun install --cwd web --frozen-lockfile
(cd web && bunx playwright install --with-deps chromium)
bun run pages:check
```

`pages:check` installs frozen dependencies, builds shared UI and WASM assets, runs all Pages unit tests and typechecking, builds Storybook and the local export, and runs browser tests. Pages CI invokes the same command and additionally leaves a deployment export using its `/nlp-stack` base path. Browser installation is an explicit prerequisite, not part of verification. The browser test server serves a deterministic embedding-worker fixture at the HTTP boundary, including for nested workers, while running the real analysis worker and Rust/WASM. The gate uses no retries or pre-existing server.

The Pages runtime tests use the built Rust/WASM modules to verify the browser wire contract. Build those assets and the shared UI before running the tests:

```bash
bun run nlp-app-ui:build
bash web/scripts/build-wasm-assets.sh
bun install --cwd web --frozen-lockfile
bun run --cwd web test
```

The repeated-sentence clustering benchmark covers 500, 1,000, and 2,000 units through the semantic analysis API:

```bash
cargo bench -p moenarch-text-analysis --bench document_and_corpus -- semantic_repeated_sentences
```
