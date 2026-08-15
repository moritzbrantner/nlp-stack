# Agent-driven releases

Issue #113 authorized extraction but no publication. The historical JSON plan
therefore remains a nonpublishing inventory. Issue #2 in this repository is the
destination-local authorization surface for the issue #114 Cargo wave; only its
exact checked TOML manifest may authorize those 52 package versions.

The Cargo wave keeps the original tested source commit as immutable package and
tag provenance. After a partial publication, a fixed release-control repair
commit may update only release scripts, their tests, documentation, and the
manifest; a final manifest-only control commit binds that repair source. The
open issue must identify the exact control SHA and manifest SHA-256 and must
carry `release:approved` before the receipt-gated publisher can run.

The default repository hook replays the pinned `required_consumer_checks` and
packages every candidate before publication. A restructuring-first maintainer
directive may instead set the checked manifest flag `fast_continuation = true`
and require the operator to pass `NLP_RELEASE_FAST_CONTINUATION=1`. That explicit
two-part policy skips repository, unit, integration, consumer, and all-candidate
archive suites. It retains the irreversible-operation safeguards: clean exact
head, issue/manifest/approval authority, exact dependency/version/order
validation, exact non-yanked registry-prefix checksums, immutable source/tag
binding, and `cargo package` for only the next absent crate immediately before
its publish attempt. Neither the fast flag nor a prior receipt alone authorizes
publication. The maintainer explicitly accepts that this fast-continuation
branch is reviewed statically and untested; no behavioral, integration,
consumer, sensitivity, or final broad suite is required for publication.

The downstream gate records exact source commits and never edits consumer
product source. It compiles native-whisperx against the local candidate.
media-similarity, youtube-corpus, document-search, and philosophy-extractor run
unchanged compatibility baselines; their candidate migrations remain deferred
to rust-packages issues #124, #125, #127, and #128 until registry-only proof is
available. video-analysis-studio and stutter-tracker remain on pinned
compatibility packages, and the rust-packages ownership baseline is also pinned.
Candidate compatibility and deferred baseline evidence are distinct outcomes.

All 28 Bun packages remain nonpublishing. Any npm/WASM publication requires its
own exact authorization after ownership is checked again against
`platform-packages`.

Credentials remain in their normal tool-specific stores and must never be
printed or copied into repository files. Partial publication stops at the first
failure. Published artifacts are never overwritten, deleted, silently skipped,
automatically yanked, or inferred beyond a reviewed manifest. Source removal
from `rust-packages` remains separate restructuring work.
