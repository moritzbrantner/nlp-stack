# Agent-driven releases

Issue #113 authorizes no publication. The checked-in release plan retains all
52 Cargo and 28 Bun surfaces at their current versions, sets both Cargo and npm
publication to false, declares no tags, and has no release issue.

A future Cargo release requires a destination-local exact issue, reviewed
publishing manifest, clean immutable commit, full package and consumer checks,
topological publication, crates.io verification, and immutable package tags.
Any npm/WASM publication requires its own exact authorization after ownership is
checked again against `platform-packages`.

Credentials remain in their normal tool-specific stores and must never be
printed or copied into repository files. Partial publication stops at the first
failure. Published artifacts are never overwritten, deleted, silently skipped,
automatically yanked, or inferred beyond a reviewed manifest. Source removal
from `rust-packages` is a later gate after registry-only consumer evidence.
