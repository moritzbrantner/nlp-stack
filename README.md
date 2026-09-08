# nlp-stack

`nlp-stack` is the canonical text and natural-language capability repository for the Moenarch ecosystem. It owns NLP source, architecture, tests, issues, versions, and releases for capabilities assigned to this domain.

The repository was extracted from `moritzbrantner/rust-packages`; see [docs/PROVENANCE.md](docs/PROVENANCE.md). Historical copies in `rust-packages` are compatibility/provenance material rather than a second source of truth.

The current workspace still contains the broad extraction-era package, adapter, WASM, and demo inventory. That inventory is transitional. The target architecture intentionally reduces it to semantic capability libraries, earned adapters, a thin aggregate registry boundary, and one default NLP workbench. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before treating an existing package boundary as durable.

Ownership does not itself authorize publication, tags, releases, consumer migration, or source removal. Those remain explicit release or migration tasks.

## Browser workbench

The static GitHub Pages workbench can ingest text-bearing files, PDFs, images, and pasted text locally in the browser, then run the existing Rust/Wasm analysis surfaces without an `nlp-stack` API server.

The corpus-baseline flow exports a versioned `nlp-stack.semantic-corpus-baseline` JSON artifact containing the extracted source text, provenance, semantic-corpus options, and Rust semantic result. The artifact deliberately omits generated timestamps so unchanged inputs and analysis output serialize deterministically.

## Source development

Normal feature work may use the exact `moenarch-foundation` revision declared in `.coding-tooling.source-deps.json` without publishing intermediate crates. Run `bash scripts/source-deps activate` before cross-repository work and `bash scripts/source-deps deactivate` before registry-only release verification.

See [docs/SOURCE_DEVELOPMENT.md](docs/SOURCE_DEVELOPMENT.md).

## Local verification

```bash
bun install --frozen-lockfile
cargo metadata --format-version 1 --no-deps
python3 scripts/check_repository_boundaries.py --check
python3 scripts/check_release_plan.py --check docs/repository-split/release-plan.json
python3 -m unittest discover -s scripts -p 'test_*.py'
cargo test --workspace --all-features
cargo test --workspace --no-default-features
cargo doc --workspace --no-deps
bun run nlp-app-ui:test
bun run text-app:typecheck
bun run text-app:build
bun run text-wasm:test:all
```

`scripts/check-preflight.sh` runs the normal clean-checkout gate. `scripts/check.sh` adds archive verification for every Rust package.
