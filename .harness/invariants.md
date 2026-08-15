# Project invariants

## INV-001 — The reviewed Rust surface remains independently usable

- Requirement: Exactly 52 reviewed Cargo packages build, test, document, and package from this checkout at their source names and exact release-manifest versions.
- Forbidden behavior: omitted packages, unreviewed packages, broken public behavior, or dependence on another checkout.
- Authority/source: repo:docs/repository-split/package-ownership.json
- Affected surfaces: Cargo.toml, Cargo.lock, crates/**
- Linked tests: repo:scripts/test_check_repository_boundaries.py
- Compatibility promise: Bootstrap does not intentionally alter public APIs, serialized shapes, or operation IDs.
- Required evidence: contract, behavioral, static
- Sensitivity: optional
- Risk dimensions: security=covered:INV-002; recovery=covered:INV-003; persistence=not-applicable:no-storage-migration; concurrency=not-applicable:no-concurrency-contract-change; migration=covered:INV-003; partial-failure=covered:INV-003; operational=covered:INV-004

## INV-002 — NLP has no dependency escape or reverse capability edge

- Requirement: Local dependencies resolve inside this checkout and external Moenarch dependencies are exact released foundation versions.
- Forbidden behavior: sibling paths, moving Git branches, audio/visual/spatial/application edges, or unpublished foundation sources.
- Authority/source: repo:CONTEXT.md
- Affected surfaces: Cargo.toml, Cargo.lock, crates/**/Cargo.toml, packages/*/package.json
- Linked tests: repo:scripts/test_check_repository_boundaries.py
- Compatibility promise: Consumers can build from a clean clone without a sibling repository.
- Required evidence: contract
- Sensitivity: optional
- Risk dimensions: security=covered:INV-002; recovery=covered:INV-003; persistence=not-applicable:no-storage-migration; concurrency=not-applicable:no-concurrency-contract-change; migration=covered:INV-002; partial-failure=covered:INV-003; operational=covered:INV-004

## INV-003 — Publication is exact, receipt-gated, and recoverable

- Requirement: Cargo publication uses only destination issue #2, the checked `releases/nlp-wave-1.toml`, its original immutable package/tag source, an optional fixed-surface release-control repair source, a manifest-only exact control head, a current independent review, a passing exact-head Agent Loop receipt, and `release:approved`.
- Forbidden behavior: implicit or npm publication, undeclared packages or versions, mutable dependency sources, tags before registry verification, republishing an existing version, automatic yanks, source removal, or product-logic changes in downstream consumers.
- Authority/source: repo:releases/nlp-wave-1.toml
- Affected surfaces: .agent-loop.toml, releases/**, scripts/publish_release.py, scripts/check_release_plan.py, docs/AGENT_DRIVEN_RELEASES.md, docs/RELEASE_CHECKLIST.md
- Linked tests: repo:scripts/test_publish_release.py, repo:scripts/test_check_release_plan.py
- Compatibility promise: npm packages and rust-packages source remain unchanged; a partial Cargo wave preserves its published prefix and resumes at the first absent package.
- Required evidence: contract, behavioral, integration
- Sensitivity: required
- Risk dimensions: security=covered:INV-002; recovery=covered:INV-003; persistence=not-applicable:no-state-migration; concurrency=not-applicable:no-concurrent-release; migration=covered:INV-003; partial-failure=covered:INV-003; operational=covered:INV-003

## INV-004 — Focused app and WASM package surfaces remain compatible

- Requirement: The 13 apps retain their operation IDs and compile against the private focused workbench; every npm/WASM wrapper completes its pack/install smoke.
- Forbidden behavior: restoring the unpublished compatibility UI dependency, absorbing platform implementations, or silently dropping an app/wrapper.
- Authority/source: repo:CONTEXT.md
- Affected surfaces: package.json, bun.lock, packages/**, crates/bindings/**
- Linked tests: repo:packages/nlp-app-ui/src/package-surface/package-surface.test.tsx
- Compatibility promise: Existing focused app and WASM entrypoints remain available without publishing them.
- Required evidence: behavioral, integration
- Sensitivity: optional
- Risk dimensions: security=covered:INV-002; recovery=covered:INV-003; persistence=not-applicable:no-app-storage-migration; concurrency=not-applicable:no-concurrency-change; migration=covered:INV-004; partial-failure=covered:INV-004; operational=covered:INV-004
