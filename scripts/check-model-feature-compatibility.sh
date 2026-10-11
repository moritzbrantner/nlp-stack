#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

mode="${1:-registry}"
case "$mode" in
  registry) lock_args=(--locked) ;;
  workspace) lock_args=() ;;
  *)
    printf 'usage: %s [registry|workspace]\n' "$0" >&2
    exit 2
    ;;
esac

# Registry mode checks the committed registry contract. Workspace mode checks
# the pinned moenarch-foundation source override after the managed outer workspace
# has activated it with coding-tooling.
cargo check "${lock_args[@]}" -p moenarch-text-linguistics --features model-bundles --all-targets
cargo check "${lock_args[@]}" -p moenarch-text-linguistics --features "candle model-bundles" --all-targets
cargo check "${lock_args[@]}" -p moenarch-text-analysis --features model-bundles --all-targets
cargo check "${lock_args[@]}" -p moenarch-text-analysis --features local-models --all-targets
