# Agent instructions

Read `CONTEXT.md`, `docs/PROVENANCE.md`, the ADR, ownership map, and release plan
before changing package boundaries or release metadata.

## Boundaries

- Keep NLP dependent only on released `moenarch-foundation` crates.
- Do not add path dependencies that escape this checkout or Git dependencies on
  moving branches.
- Preserve public Rust APIs, transcript serialization, package names, operation
  IDs, and focused CLI/server/WASM/app adapters unless a separate semver issue
  authorizes a change.
- Keep browser/product implementations in `platform-packages`; the private NLP
  workbench is only the focused package-surface adapter documented in provenance.
- Do not publish Cargo or npm packages, create tags/releases, migrate consumers,
  or remove source from `rust-packages` without a later exact release contract.
- The `.harness/` profile is draft. Its structural audit is required for this
  bootstrap, but the issue-required repository checks remain authoritative.

## Checks

Run the narrowest relevant Cargo or Bun check while developing. Always run:

```bash
python3 scripts/check_repository_boundaries.py --check
python3 scripts/check_release_plan.py --check docs/repository-split/release-plan.json
python3 -m unittest discover -s scripts -p 'test_*.py'
```

Use `scripts/check-preflight.sh` before handoff and `scripts/check.sh` when
archive verification is required. Never weaken checks or report unrun evidence
as passing.

<!-- verification-harness:start -->
## Verification harness
Run the installed `moenarch-verification-harness` skill's `audit` command before changing verification surfaces.
Early selection is advisory; `full` remains the handoff gate. See `.harness/README.md`.
<!-- verification-harness:end -->
