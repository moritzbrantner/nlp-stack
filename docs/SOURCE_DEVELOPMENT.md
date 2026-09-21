# NLP source development

`nlp-stack` keeps registry coordinates as its distribution contract, but ordinary cross-repository development does not require publishing foundation crates first.

The committed `.coding-tooling.source-deps.json` pins the foundation packages used by this workspace to one exact `moenarch-foundation` revision and enables local-only resolution. Run `bash scripts/source-deps activate` only after the sibling `../moenarch-foundation` checkout exists at that exact Git `HEAD`. Missing local source is an error; source mode never falls back to cloning the private repository or authenticated Git fetches.

The outer coding loop owns the sibling checkout/worktree and may advance the pinned revision when a task deliberately validates a newer foundation source head. Use `bash scripts/source-deps status` to inspect the mode and `bash scripts/source-deps deactivate` before registry-only release verification.

## Development contract

- Feature work may change NLP and the immediate foundation source without starting a crates.io release.
- Keep package versions compatible during source work. Version bumps and publication belong to a dedicated release task.
- Update every affected entry to the same exact foundation revision when the validated source head changes.
- Do not commit generated Cargo configuration, sibling paths, or moving Git dependencies to package manifests.
- Do not add private-repository credentials or authenticated Git fallback merely to reproduce the local multi-repository workspace in hosted CI.
- Keep a normal consumer task to the consumer plus at most two upstream repositories unless broader migration scope was explicitly assigned.

## Verification boundary

Source-mode verification in the local multi-repository workspace proves the exact source graph under development and is valid implementation evidence before publication. Hosted CI remains repository-local. Distribution still requires a later release task to deactivate source mode, publish or select the minimal required crate closure, and prove clean registry-only resolution.

## Prepare resolution before verification

Source activation changes dependency origins. Prepare the corresponding local lockfile explicitly before running locked verification:

```bash
bash scripts/source-deps activate
cargo metadata --format-version 1 > /dev/null
bash scripts/check-preflight.sh
# Also required for Pages, shared UI, or Pages Rust/WASM changes:
bun run pages:check
```

The metadata command is a deliberate dependency-resolution step; inspect its `Cargo.lock` diff. Keep source-only lockfile changes local to the source-development checkout. Prefer an isolated worktree when the ordinary checkout must retain the committed registry resolution. Verification and WASM builds use `--locked` and fail if resolution has not been prepared; they do not silently repair the lockfile. After changing source pins or returning to registry mode, prepare and inspect that mode's resolution again before verification. Registry-only release proof still belongs to the dedicated release workflow.
