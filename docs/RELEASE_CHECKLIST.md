# Release checklist

The issue #113 bootstrap plan remains nonpublishing historical inventory. The
issue #114 Cargo wave is authorized only through destination issue #2 and the
checked `releases/nlp-wave-1.toml`; npm publication remains unauthorized.

For the authorized Cargo wave:

1. Preserve the tested source commit as package/tag provenance. A recovery after
   partial publication may add one fixed-surface release-control repair commit;
   create a final control commit whose only difference from that repair source
   is the exact publishing manifest.
2. Bind exact package names, versions, registry, source/base commits, dependency
   order, checks, consumers, and tags.
3. Keep Cargo and npm/WASM authorization separate; re-check ownership against
   `platform-packages` before any browser wrapper publication.
4. Require a clean immutable commit, full repository checks, every Cargo archive,
   WASM pack/install smoke, the source-pure native-whisperx candidate gate, and
   unchanged pinned baselines for migrations deferred to #124/#125/#127/#128.
5. Run independent exact-head review and Agent Loop verification. The Agent
   Loop master validates the exact receipt for every `required_check`; the
   repository hook replays consumer and archive gates. Update the
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
