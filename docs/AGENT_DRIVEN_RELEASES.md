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

The Agent Loop master runs all ordered `required_checks` and validates their
exact-head receipt before invoking the repository hook. The hook then replays
the pinned `required_consumer_checks`, packages every candidate, publishes in
manifest order, verifies crates.io, and creates immutable tags at the original
source commit plus GitHub Releases. The postpublication
`scripts/check_nlp_wave_1_registry_consumer.sh` must then resolve all 52 packages
from registry sources with no local patch.

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
from `rust-packages` is a later gate after registry-only consumer evidence.
