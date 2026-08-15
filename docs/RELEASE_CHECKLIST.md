# Release checklist

The bootstrap plan is non-publishing. Do not run `cargo publish`, `npm publish`,
create tags/releases, or change package versions for issue #113.

For a future authorized release:

1. Use an open destination-local issue and exact reviewed publishing manifest.
2. Bind exact package names, versions, registry, source/base commits, dependency
   order, checks, consumers, and tags.
3. Keep Cargo and npm/WASM authorization separate; re-check ownership against
   `platform-packages` before any browser wrapper publication.
4. Require a clean immutable commit, full repository checks, every Cargo archive,
   WASM pack/install smoke, and candidate consumer gates.
5. Publish topologically, verify immutable registry artifacts, then create tags.
6. Stop on partial failure and resume from the first unpublished artifact; never
   overwrite, delete, or automatically yank a published version.
7. Keep `rust-packages` source until release, consumer, compatibility, and
   rollback gates all pass.
