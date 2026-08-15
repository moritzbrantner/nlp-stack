#!/usr/bin/env bash
set -euo pipefail

repository_root=$(git rev-parse --show-toplevel)
scratch_parent=${NLP_WAVE_1_SCRATCH_PARENT:-"$(dirname "$repository_root")"}
scratch_root=$(mktemp -d "$scratch_parent/nlp-wave-1-registry.XXXXXX")

cleanup() {
  case "$scratch_root" in
    "$scratch_parent"/nlp-wave-1-registry.*)
      rm -rf -- "$scratch_root"
      ;;
    *)
      echo "refusing to remove unexpected registry scratch directory: $scratch_root" >&2
      ;;
  esac
}
trap cleanup EXIT

fixture="$repository_root/scripts/fixtures/nlp_wave_1_consumer"
cp "$fixture/Cargo.toml" "$scratch_root/Cargo.toml"
cp -R "$fixture/src" "$scratch_root/src"

export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"$repository_root/target"}
cargo generate-lockfile --manifest-path "$scratch_root/Cargo.toml"
cargo check --locked --manifest-path "$scratch_root/Cargo.toml"

cargo metadata \
  --format-version 1 \
  --locked \
  --manifest-path "$scratch_root/Cargo.toml" \
  | jq -e --slurpfile ownership "$repository_root/docs/repository-split/package-ownership.json" '
      ($ownership[0].packages
        | map(select(
            .ecosystem == "cargo"
              and .intended_next_release_owner == "moritzbrantner/nlp-stack"
          ))
        | map(.current_package_name)) as $wave
      | [.packages[] | select(.name as $name | $wave | index($name))]
      | length == 52
        and all(.[]; (.source // "") | startswith("registry+"))
    ' >/dev/null

echo "NLP wave 1 registry-only consumer passed"
