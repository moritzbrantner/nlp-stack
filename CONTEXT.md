# Repository context

This repository is the NLP layer of the Moenarch capability graph. Production Rust code may depend on crates owned here and on foundation crates. During normal development, the committed source-dependency declaration may replace the registry packages with one exact `moenarch-foundation` revision; distributed builds use the exact released versions declared in the root workspace. Committed package manifests must not depend on audio-analysis, visual-analysis, spatial-analysis, application repositories, another checkout, or moving Git branches.

The workspace owns 56 Cargo packages: 52 clean-copied source packages and four
destination-authored aggregate-registry packages. The accompanying Bun inventory contains
13 apps, 13 npm/WASM wrappers, one benchmark package, and the private
`@moritzbrantner/nlp-app-ui` workbench adapter. The adapter is a focused copy of
the package-surface seam; the broader compatibility UI remains in
`rust-packages`, while browser/application implementations remain owned by
`platform-packages`. No Bun surface is publication-eligible without a separate
exact npm/WASM ownership and release decision.

Public Rust APIs, serialized transcript shapes, package names, and operation IDs
are retained from extraction commit
`b8b29cf8db0b86ed1b133a18155adf24992f9483`. Repository movement is additive:
`rust-packages` remains the active source and release owner until a later exact
release issue, registry proof, and consumer gates transfer that responsibility.
