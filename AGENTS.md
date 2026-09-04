# Agent instructions

Read `CONTEXT.md`, `docs/ARCHITECTURE.md`, `docs/PROVENANCE.md`, `docs/SOURCE_DEVELOPMENT.md`, and the relevant release/ownership material before changing package boundaries, cross-repository dependencies, public APIs, or release metadata.

## Agent startup

- On a fresh checkout or freshly provisioned trusted base image, execute `bash scripts/codex-environment.sh setup` before repository work. The base environment must already provide the repository's exact Bun version plus `rustup`; environment-v1 installs the recorded dependency state without silently changing pins or lockfiles.
- Before an implementation run, execute `bash scripts/check-agent-readiness.sh`. It verifies the semantic environment fingerprint, resolves the live `coding-agent-conventions` policy for this repository, and validates the current `coding-agent-skills` catalog plus the `standard` profile.
- Before work that activates the local foundation source graph, execute `bash scripts/check-agent-readiness.sh --with-source`. The outer workspace must provide the exact sibling `moenarch-foundation` revision; the canary verifies source-development environment identity, activation, and Cargo metadata without turning private Git authentication into repository configuration.
- Shared engineering policy remains live in `coding-agent-conventions`; do not copy shared rule text into this repository. `.coding-tooling.json` only names stable convention IDs whose continued availability is part of this repository's agent contract.

## Architecture invariants

- `nlp-stack` is the canonical NLP source/architecture owner. Do not add new NLP behavior to historical `rust-packages` copies.
- Treat the current extraction-era package count and adapter inventory as transitional. `docs/ARCHITECTURE.md` defines the target boundaries.
- Organize code by semantic capability. Do not create pairwise bridge crates, one adapter per capability, one contract crate per result type, or one backend crate per runtime without concrete reuse/dependency/lifecycle pressure.
- Domain crates expose typed Rust APIs and typed domain data. Do not make `PackageSurface`, string operation IDs, JSON dispatch, transport DTOs, or workflow presentation metadata part of a semantic capability API.
- `nlp-package-registry` is an outermost aggregate composition root. Semantic/domain crates must not depend on it.
- `text-core` is a small media/runtime-agnostic kernel. Do not add generic analyzer/pipeline orchestration, model execution policy, media timestamps/events, indexing/retrieval, or capability-specific result graphs to it.
- Use UTF-8 half-open byte offsets as the canonical text-span coordinate system. Add explicitly typed conversion helpers at boundaries that require UTF-16/grapheme/etc. positions rather than duplicating offsets in core spans.
- Keep semantic requests backend-neutral. Device/model/download/cache/credential/retry policy belongs in execution configuration/context, not in the domain request shape.
- Errors are capability-local and typed. Do not introduce a universal NLP error enum or reuse media-specific errors for unrelated text capabilities.
- Neutral timed-text contracts belong below NLP in `moenarch-foundation::media-core` for now. `text-core` must remain independent of `media-core`.
- `text-embeddings` may own direct pairwise embedding similarity but not collection indexing/search. `text-index` owns materialization/storage/update; `text-retrieval` owns query-time ranking/fusion/filter/reranking.
- `text-linguistics` may remain one crate, but expose useful linguistic operations independently. Treat full Fast/Balanced/Rich analysis as a convenience recipe rather than the fundamental API.
- `text-analysis` contains only proven reusable stateless recipes. Start cross-capability workflows in their consuming application and promote them only after independent reuse proves the seam.
- Use concrete capability names. The current Markov implementation should not be generalized beyond evidence; extractive QA should not claim generic QA behavior.
- Compatibility is consumer-driven. Preserve or deliberately migrate APIs used by real consumers, but do not add ceremonial shims for obsolete adapters, bridge crates, runtime layers, or unused 0.1.x seams.
- Published sibling libraries should use normal semver-compatible requirements by default. Exact sibling pins require a concrete compatibility reason.

## Repository boundaries and releases

- Keep NLP dependent only on `moenarch-foundation` plus NLP capabilities whose dependency direction is allowed by `docs/ARCHITECTURE.md`.
- The outer coding workspace owns the sibling foundation checkout/worktree. It must exist at the exact pinned revision before source activation; do not add private-repository tokens or authenticated Git fallback to hosted CI.
- Do not add committed path dependencies that escape this checkout or Git dependencies on moving branches. Source overrides belong only to managed, ignored Cargo configuration generated by `coding-tooling`.
- Ordinary feature work is source-first. Do not publish crates, bump versions, create tags, or start a release train merely to unblock implementation.
- Repository ownership does not authorize publication, source deletion, or consumer migration. Dedicated release/migration issues own those actions.
- If implementation uncovers an architectural choice not resolved by `docs/ARCHITECTURE.md` or the issue, stop at that seam and return the decision to an architecture review instead of inventing a new abstraction.

## Checks

Run the narrowest relevant Cargo or Bun check while developing. Always run:

```bash
python3 scripts/check_repository_boundaries.py --check
python3 scripts/check_release_plan.py --check docs/repository-split/release-plan.json
python3 -m unittest discover -s scripts -p 'test_*.py'
```

Use `scripts/check-preflight.sh` before handoff and `scripts/check.sh` when archive verification is required. Never weaken checks or report unrun evidence as passing.
