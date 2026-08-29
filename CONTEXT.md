# Repository context

This repository is the canonical NLP layer of the Moenarch capability graph. Production Rust code may depend on capabilities owned here and on lower-level `moenarch-foundation` crates. Historical NLP copies in `rust-packages` are compatibility/provenance material only.

During normal development, the committed source-dependency declaration may replace registry packages with one exact `moenarch-foundation` revision. Distributed builds use released dependency coordinates. Committed package manifests must not depend on audio-analysis, visual-analysis, spatial-analysis, application repositories, another checkout, or moving Git branches.

The current workspace is an extraction-era inventory rather than the target package graph. It still contains focused CLI/server/WASM packages, pairwise bridges, model-runtime glue, and per-capability Bun applications that the target architecture does not require. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) is authoritative for durable capability boundaries and the bottom-up simplification order.

Important repository-level rules:

- semantic capability libraries are the durable center of the workspace;
- typed Rust APIs and typed data are the source of truth;
- generic JSON/operation/transport surfaces belong at outer adapters, not inside domain crates;
- adapters, backend crates, contract crates, compatibility layers, and shared recipes are created only when real use proves the need;
- `text-core` is a small media/runtime-agnostic text kernel;
- neutral timed text belongs below NLP in `moenarch-foundation::media-core` for now;
- publication is a separate decision from repository ownership;
- compatibility work is driven by real consumers, not by preserving every extracted 0.1.x seam.

Browser/product implementations remain application concerns. This repository may own one NLP workbench/showcase and narrow browser smoke fixtures; a dedicated app for every capability is not an invariant.

For source-development mechanics see `docs/SOURCE_DEVELOPMENT.md`. For extraction history see `docs/PROVENANCE.md`. For architectural decisions and target boundaries see `docs/ARCHITECTURE.md`.
