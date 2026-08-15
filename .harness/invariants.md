# Project invariants

## INV-001 — The exact Rust release surface remains structurally declared

- Requirement: Exactly 52 reviewed Cargo packages remain present in Cargo metadata at their release-manifest names, versions, ownership, and dependency order.
- Forbidden behavior: omitted or extra packages, undeclared versions, wrong ownership, or dependency-order drift.
- Authority/source: repo:docs/repository-split/package-ownership.json
- Affected surfaces: Cargo.toml, Cargo.lock, crates/**
- Compatibility promise: Public behavior is outside the publication gate and remains a separately accepted restructuring risk.
- Required evidence: contract, static
- Sensitivity: optional
- Risk dimensions: security=covered:INV-002; recovery=covered:INV-003; persistence=not-applicable:no-storage-migration; concurrency=not-applicable:no-concurrency-contract-change; migration=covered:INV-003; partial-failure=covered:INV-003; operational=covered:INV-004

## INV-002 — NLP has no dependency escape or reverse capability edge

- Requirement: Local dependencies resolve inside this checkout and external Moenarch dependencies are exact released foundation versions.
- Forbidden behavior: sibling paths, moving Git branches, audio/visual/spatial/application edges, or unpublished foundation sources.
- Authority/source: repo:CONTEXT.md
- Affected surfaces: Cargo.toml, Cargo.lock, crates/**/Cargo.toml, packages/*/package.json
- Compatibility promise: Consumers can build from a clean clone without a sibling repository.
- Required evidence: contract
- Sensitivity: optional
- Risk dimensions: security=covered:INV-002; recovery=covered:INV-003; persistence=not-applicable:no-storage-migration; concurrency=not-applicable:no-concurrency-contract-change; migration=covered:INV-002; partial-failure=covered:INV-003; operational=covered:INV-004

## INV-003 — Irreversible publication effects remain exact and recoverable

- Requirement: Cargo publication uses only destination issue #2, the checked `releases/nlp-wave-1.toml`, its original immutable package/tag source, a fixed-surface release-control source, a manifest-only exact control head, current independent static review, the structural safeguards receipt, and `release:approved`. Fast continuation additionally requires both the checked manifest flag and the matching operator flag.
- Forbidden behavior: implicit or npm publication, undeclared packages or versions, mutable dependency sources, tags before registry verification, republishing an existing version, automatic yanks, source removal, or product-logic changes in downstream consumers.
- Authority/source: repo:releases/nlp-wave-1.toml
- Affected surfaces: .agent-loop.toml, releases/**, scripts/publish_release.py, scripts/check_release_plan.py, docs/AGENT_DRIVEN_RELEASES.md, docs/RELEASE_CHECKLIST.md
- Compatibility promise: npm packages and rust-packages source remain unchanged; a partial Cargo wave preserves its published prefix and resumes at the first absent package.
- Required evidence: contract, static
- Sensitivity: optional
- Risk dimensions: security=covered:INV-002; recovery=covered:INV-003; persistence=not-applicable:no-state-migration; concurrency=not-applicable:no-concurrent-release; migration=covered:INV-003; partial-failure=covered:INV-003; operational=covered:INV-003

## INV-004 — Behavioral compatibility is separate from publication

- Requirement: Publication does not authorize npm artifacts, source removal, or product-logic changes; app, WASM, consumer, and compatibility verification remains separate restructuring work.
- Forbidden behavior: treating publication as authorization for npm release, source removal, or downstream product mutation.
- Authority/source: repo:CONTEXT.md
- Affected surfaces: package.json, bun.lock, packages/**, crates/bindings/**
- Compatibility promise: Existing surfaces are not modified by the publication operation itself.
- Required evidence: contract, static
- Sensitivity: optional
- Risk dimensions: security=covered:INV-002; recovery=covered:INV-003; persistence=not-applicable:no-app-storage-migration; concurrency=not-applicable:no-concurrency-change; migration=covered:INV-004; partial-failure=covered:INV-004; operational=covered:INV-004
