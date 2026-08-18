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
4. For restructuring-first continuation, set reviewed manifest flag
   `fast_continuation = true` and invoke with
   `NLP_RELEASE_FAST_CONTINUATION=1`. Keep only clean exact-head authority,
   exact dependency/version/order validation, registry absence/checksum/yank
   checks, immutable source/tag binding, and `cargo package` for the next absent
   crate. Do not replay repository, unit, integration, consumer, WASM, or
   all-candidate archive suites.
5. Run independent exact-head review of the control-policy change. Update the
   open destination issue with exactly one control SHA and manifest digest line,
   then apply `release:approved` only for an active publication attempt.
6. Publish topologically through the receipt-gated wrapper, verify immutable
   registry artifacts, then create tags and declared GitHub Releases.
7. Treat registry-only consumers and downstream update PRs as separate
   restructuring work, not publication gates.
8. Stop on partial failure and resume from the first unpublished artifact; never
   overwrite, delete, or automatically yank a published version.
9. Keep `rust-packages` source until release, consumer, compatibility, and
   rollback gates all pass.
