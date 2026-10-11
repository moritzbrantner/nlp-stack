#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

mode="${1:-registry}"
case "$mode" in
  registry|workspace) lock_args=(--locked) ;;
  *)
    printf 'usage: %s [registry|workspace]\n' "$0" >&2
    exit 2
    ;;
esac

# Registry mode checks the committed registry contract. Workspace mode checks
# the pinned moenarch-foundation source override after the managed outer workspace
# has activated it with coding-tooling and prepared the local lockfile. Both modes
# reject stale resolution instead of silently rewriting Cargo.lock.
cargo check "${lock_args[@]}" -p moenarch-text-linguistics --features model-bundles --all-targets
cargo check "${lock_args[@]}" -p moenarch-text-linguistics --features "candle model-bundles" --all-targets
cargo check "${lock_args[@]}" -p moenarch-text-analysis --features model-bundles --all-targets
cargo check "${lock_args[@]}" -p moenarch-text-analysis --features local-models --all-targets
