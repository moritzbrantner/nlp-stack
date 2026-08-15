# Release checklist

The issue #113 bootstrap plan remains nonpublishing historical inventory. The
issue #114 Cargo wave is authorized only through destination issue #2 and the
checked `releases/nlp-wave-1.toml`; npm publication remains unauthorized.

For the authorized Cargo wave:

1. Create the tested source commit, then a control commit whose only difference
   is the exact publishing manifest.
2. Bind exact package names, versions, registry, source/base commits, dependency
   order, checks, consumers, and tags.
3. Keep Cargo and npm/WASM authorization separate; re-check ownership against
   `platform-packages` before any browser wrapper publication.
4. Require a clean immutable commit, full repository checks, every Cargo archive,
   WASM pack/install smoke, and candidate consumer gates.
5. Run independent exact-head review and Agent Loop verification. Update the
   open destination issue with exactly one control SHA and manifest digest line,
   then apply `release:approved`.
6. Publish topologically through the receipt-gated wrapper, verify immutable
   registry artifacts, then create tags and declared GitHub Releases.
7. Run `scripts/check_nlp_wave_1_registry_consumer.sh` without any Cargo patch,
   then open one downstream update PR per affected repository.
8. Stop on partial failure and resume from the first unpublished artifact; never
   overwrite, delete, or automatically yank a published version.
9. Keep `rust-packages` source until release, consumer, compatibility, and
   rollback gates all pass.
