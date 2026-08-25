# NLP source development

`nlp-stack` keeps registry coordinates as its distribution contract, but ordinary cross-repository development does not require publishing foundation crates first.

The committed `.coding-tooling.source-deps.json` pins the foundation packages used by this workspace to one exact `moenarch-foundation` revision. Run `bash scripts/source-deps activate` to ask `coding-tooling` to materialize the ignored `.cargo/config.toml`. If a sibling foundation checkout exists, its Git `HEAD` must equal the declared revision; otherwise coding-tooling uses the exact Git revision when the repository is accessible.

Use `bash scripts/source-deps status` to inspect the mode and `bash scripts/source-deps deactivate` before registry-only release verification.

## Development contract

- Feature work may change NLP and the immediate foundation source without starting a crates.io release.
- Keep package versions compatible during source work. Version bumps and publication belong to a dedicated release task.
- Update every affected entry to the same exact foundation revision when the validated source head changes.
- Do not commit generated Cargo configuration, sibling paths, or moving Git dependencies to package manifests.
- Keep a normal consumer task to the consumer plus at most two upstream repositories unless broader migration scope was explicitly assigned.

## Verification boundary

Source-mode verification proves the exact source graph under development and is valid implementation evidence before publication. Distribution still requires a later release task to deactivate source mode, publish or select the minimal required crate closure, and prove clean registry-only resolution.
