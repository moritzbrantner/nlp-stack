# Clean-copy provenance

This repository was created by clean-copy extraction. Git history was not
rewritten or filtered.

- Source repository: `moritzbrantner/rust-packages`
- Reviewed Phase-A ownership baseline: `d032ad2890c1df3c6a5b9eff024562f00d017fce`
- Exact extraction commit: `b8b29cf8db0b86ed1b133a18155adf24992f9483`
- Extraction issue: `moritzbrantner/rust-packages#113`
- Parent PRD: `moritzbrantner/rust-packages#106`
- Destination: `moritzbrantner/nlp-stack`
- History note: original per-file history remains in the source repository; this
  destination begins with one attributed bootstrap change.

The 52 source-derived Cargo directories and 27 Bun package directories named by the source
records in `docs/repository-split/package-ownership.json` were copied from the
exact extraction commit. Rust crate trees remain byte-identical except for four
strict Rust 1.95 Clippy repairs: two boolean test assertions use `assert!`, one
model-bundle test reads the supported revision accessor instead of its
deprecated compatibility field, and one length divisibility guard uses
`is_multiple_of`. These preserve behavior and public APIs. The npm/WASM
wrappers remain byte-identical. The 13 app trees differ only where their unpublished
`@moritzbrantner/video-analysis-ui` workspace dependency and source aliases were
retargeted to the private focused adapter.

The `text-transcripts` README example is also self-contained in this repository:
its inline Whisper JSON replaces a monolith-root fixture path that cannot exist
in an independent checkout or packaged crate.

`packages/nlp-app-ui` is destination-authored from the exact extraction
commit's `packages/video-analysis-ui/src/package-surface/**` plus the shared
primitive component seam. Stories and the 29 package-surface behavior tests are
retained. A minimal local `cn` helper replaces imports from broader video UI
types. The adapter is private, is not automatically publishable, and does not
transfer ownership of the broader compatibility UI or any `platform-packages`
implementation.

The following basic inputs were copied byte-identically:

- `LICENSE-APACHE`, `LICENSE-MIT`, `rust-toolchain.toml`, `.editorconfig`

Destination-authored or materially adapted support includes the root Cargo and
Bun manifests, regenerated lockfiles, repository docs, CI, draft Harness,
ownership/release inventories, validators, and local check scripts. The
ownership validator binds the canonical digest of all 79 reviewed source
records and separately validates the destination-authored private adapter plus
the four additive aggregate-registry Cargo packages introduced by issue #122.

Generated `pkg`, `dist`, `target`, and `node_modules` output was not copied.
Dual-licensed material retains the source MIT OR Apache-2.0 terms. No Cargo or
npm publication, tag, release, consumer migration, or source removal is
authorized by this bootstrap.
