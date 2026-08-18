# Project invariants

## INV-001 — The reviewed Rust surface remains independently usable

- Requirement: Exactly 52 reviewed source Cargo packages plus four destination-authored aggregate-registry packages build, test, document, and package from this checkout at their declared versions.
- Forbidden behavior: omitted packages, unreviewed destination additions, broken public behavior, or dependence on another checkout.
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

## INV-003 — Bootstrap cannot authorize Cargo, npm, or source removal

- Requirement: Every package remains at its source version with publish=false, no tags, and no release issue; npm/WASM ownership remains separately gated.
- Forbidden behavior: implicit publication, version bump, tag, release, consumer migration, or source removal.
- Authority/source: repo:docs/repository-split/release-plan.json
- Affected surfaces: docs/repository-split/**, docs/AGENT_DRIVEN_RELEASES.md, docs/RELEASE_CHECKLIST.md
- Linked tests: repo:scripts/test_check_release_plan.py
- Compatibility promise: rust-packages remains active source/release owner until later gates complete.
- Required evidence: contract
- Sensitivity: optional
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
