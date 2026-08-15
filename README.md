# nlp-stack

`nlp-stack` is the text and natural-language capability repository for the
Moenarch ecosystem. It owns 52 Rust crates for text documents, lexical and
linguistic analysis, classification, embeddings, indexing, retrieval,
question answering, generation, and purified transcript contracts. It also
retains 27 focused Bun app/WASM surfaces and one private NLP workbench adapter.

This repository was bootstrapped as a clean copy from
`moritzbrantner/rust-packages`; see [docs/PROVENANCE.md](docs/PROVENANCE.md).
The source repository remains the active release owner. This bootstrap does not
authorize Cargo or npm publication, tags, releases, consumer migration, or
source removal.

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

`scripts/check-preflight.sh` runs the normal clean-checkout gate.
`scripts/check.sh` adds archive verification for every Rust package.
