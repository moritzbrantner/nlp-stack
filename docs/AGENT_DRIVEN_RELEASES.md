# Agent-driven releases

Issue #113 authorized extraction but no publication. The historical JSON plan
therefore remains a nonpublishing inventory. Issue #2 in this repository is the
destination-local authorization surface for the issue #114 Cargo wave; only its
exact checked TOML manifest may authorize those 52 package versions.

The Cargo wave uses a source commit followed by a manifest-only control commit.
The open issue must identify that exact control SHA and the manifest SHA-256 and
must carry `release:approved` before the receipt-gated publisher can run. The
publisher replays all ordered repository, package, WASM, and pinned downstream
consumer checks before publishing in manifest order, verifying crates.io, and
creating immutable tags and GitHub Releases. The postpublication
`scripts/check_nlp_wave_1_registry_consumer.sh` must then resolve all 52 packages
from registry sources with no local patch.

The downstream gate records exact source commits. It compiles native-whisperx,
media-similarity, youtube-corpus, and philosophy-extractor against the local
candidate, and runs document-search tests/build against the public WASM package
exports. video-analysis-studio and stutter-tracker remain on pinned compatibility
packages pending their repository-scoped migrations; the rust-packages ownership
baseline is also pinned. Each eventual update is a separate downstream PR.

All 28 Bun packages remain nonpublishing. Any npm/WASM publication requires its
own exact authorization after ownership is checked again against
`platform-packages`.

Credentials remain in their normal tool-specific stores and must never be
printed or copied into repository files. Partial publication stops at the first
failure. Published artifacts are never overwritten, deleted, silently skipped,
automatically yanked, or inferred beyond a reviewed manifest. Source removal
from `rust-packages` is a later gate after registry-only consumer evidence.
