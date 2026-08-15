#!/usr/bin/env bash
set -euo pipefail

repository_root=$(git rev-parse --show-toplevel)
consumer_manifest="$repository_root/scripts/fixtures/nlp_wave_1_consumer/Cargo.toml"
scratch_root=${TMPDIR:-/tmp}
patch_config=$(mktemp "$scratch_root/nlp-wave-1-patches.XXXXXX.toml")

cleanup() {
  case "$patch_config" in
    "$scratch_root"/nlp-wave-1-patches.*.toml)
      unlink -- "$patch_config"
      ;;
    *)
      echo "refusing to remove unexpected patch config: $patch_config" >&2
      ;;
  esac
}
trap cleanup EXIT

{
  echo '[patch.crates-io]'
  jq -r --arg root "$repository_root" '
    .packages[]
    | select(.ecosystem == "cargo")
    | "\"" + .current_package_name + "\" = { path = \"" + $root + "/" + (.manifest_path | sub("/Cargo.toml$"; "")) + "\" }"
  ' "$repository_root/docs/repository-split/package-ownership.json"
} >"$patch_config"

export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"$repository_root/target"}
cargo check \
  --locked \
  --manifest-path "$consumer_manifest" \
  --config "$patch_config"

echo "NLP wave 1 candidate consumer passed"
