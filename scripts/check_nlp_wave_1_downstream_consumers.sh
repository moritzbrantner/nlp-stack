#!/usr/bin/env bash
set -euo pipefail

repository_root=$(git rev-parse --show-toplevel)
mkdir -p "$repository_root/target"
if [[ -n "${NLP_WAVE_1_SCRATCH_PARENT:-}" ]]; then
  scratch_parent=$NLP_WAVE_1_SCRATCH_PARENT
elif [[ -n "${CARGO_TARGET_DIR:-}" && "$CARGO_TARGET_DIR" = /* ]]; then
  # The Agent Loop publisher runs from a small temporary filesystem but points
  # CARGO_TARGET_DIR at durable build storage. Keep the eight cloned consumers
  # on that same filesystem so a clean publication checkout cannot exhaust /tmp.
  scratch_parent=$(dirname "$CARGO_TARGET_DIR")
else
  scratch_parent=$(dirname "$repository_root")
fi
mkdir -p "$scratch_parent"
scratch_root=$(mktemp -d "$scratch_parent/nlp-wave-1-downstream.XXXXXX")

cleanup() {
  case "$scratch_root" in
    "$scratch_parent"/nlp-wave-1-downstream.*)
      rm -rf -- "$scratch_root"
      ;;
    *)
      echo "refusing to remove unexpected downstream scratch directory: $scratch_root" >&2
      ;;
  esac
}
trap cleanup EXIT

clone_pinned() {
  local repository=$1
  local revision=$2
  local destination="$scratch_root/$repository"

  git init --quiet "$destination"
  git -C "$destination" remote add origin "https://github.com/moritzbrantner/$repository.git"
  git -C "$destination" fetch --quiet --depth=1 origin "$revision"
  git -C "$destination" checkout --quiet --detach FETCH_HEAD
  test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
}

replace_exact() {
  local path=$1
  local before=$2
  local after=$3
  python3 - "$path" "$before" "$after" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
before = sys.argv[2]
after = sys.argv[3]
contents = path.read_text(encoding="utf-8")
count = contents.count(before)
if count != 1:
    raise SystemExit(f"expected one occurrence of {before!r} in {path}, found {count}")
path.write_text(contents.replace(before, after), encoding="utf-8")
PY
}

clone_pinned "native-whisperx" "b0ba12342fbb36b057fbe620f62d52c4fde0b36d"
clone_pinned "media-similarity" "d015b36187a9c3ebd202f81175081608fb307aa3"
clone_pinned "youtube-corpus" "8ab21570348e7d636685a51f110f11fc2eacf363"
clone_pinned "document-search" "0221b65b7aebc7a638662c5651bcd549d431b3d8"
clone_pinned "philosophy-extractor" "f945e77657c7c6cc0d56446c23482b68648ee2a4"
clone_pinned "video-analysis-studio" "93ceeb1c43764be9d31c35258145604559e0a0aa"
clone_pinned "stutter-tracker" "6c68b7a343ac8470405a79f240263f9e8ca7af80"
clone_pinned "rust-packages" "b8b29cf8db0b86ed1b133a18155adf24992f9483"

patch_config="$scratch_root/nlp-wave-1-patches.toml"
{
  echo '[patch.crates-io]'
  jq -r --arg root "$repository_root" '
    .packages[]
    | select(.ecosystem == "cargo")
    | "\"" + .current_package_name + "\" = { path = \"" + $root + "/" + (.manifest_path | sub("/Cargo.toml$"; "")) + "\" }"
  ' "$repository_root/docs/repository-split/package-ownership.json"
} >"$patch_config"

native="$scratch_root/native-whisperx/Cargo.toml"
replace_exact "$native" 'text-model-runtime = { package = "moenarch-text-model-runtime", version = "0.1.0", default-features = false }' 'text-model-runtime = { package = "moenarch-text-model-runtime", version = "=0.1.1", default-features = false }'
replace_exact "$native" 'text-transcripts = { package = "moenarch-text-transcripts", version = "0.1.1", default-features = false }' 'text-transcripts = { package = "moenarch-text-transcripts", version = "=0.1.3", default-features = false }'

# native-whisperx is the source-pure candidate check. The next four consumers
# retain their exact, unchanged compatibility baselines until their separately
# reviewed postpublication migrations (#124, #125, #127, and #128). The pinned
# rust-packages checkout already occupies the sibling path their manifests use.
rg --fixed-strings 'text-transcripts = { package = "moenarch-text-transcripts", version = "0.1.2", features = ["native"] }' "$scratch_root/media-similarity/backend/Cargo.toml"
rg --fixed-strings 'text-transcripts = { package = "moritzbrantner-text-transcripts", path = "../rust-packages/crates/text/text-transcripts" }' "$scratch_root/youtube-corpus/Cargo.toml"
rg --fixed-strings '"@moritzbrantner/text-index-wasm": "file:../rust-packages/packages/text-index-wasm"' "$scratch_root/document-search/package.json"
rg --fixed-strings 'text-retrieval = { path = "../rust-packages/crates/text/text-retrieval" }' "$scratch_root/philosophy-extractor/Cargo.toml"

# These consumers also remain on compatibility packages that are intentionally
# not part of this release wave. Pin and assert those baselines instead of
# silently inventing a migration mapping inside a publication gate.
rg --fixed-strings 'text-analysis-features = { version = "0.1.0", path = "../rust-packages/crates/text/text-analysis-features" }' "$scratch_root/video-analysis-studio/Cargo.toml"
rg --fixed-strings 'text-analysis-transcription = { version = "0.1.0", path = "../rust-packages/crates/text/text-analysis-transcription" }' "$scratch_root/video-analysis-studio/Cargo.toml"
rg --fixed-strings 'rev = "78a9c6e9eb33730b60c9584ceffb9dc982f5b9da", package = "text-analysis-core"' "$scratch_root/stutter-tracker/apps/desktop/src-tauri/Cargo.toml"
rg --fixed-strings 'rev = "78a9c6e9eb33730b60c9584ceffb9dc982f5b9da", package = "text-analysis-transcription"' "$scratch_root/stutter-tracker/apps/desktop/src-tauri/Cargo.toml"
jq -e '
  [.packages[] | select(
    .ecosystem == "cargo"
      and .intended_next_release_owner == "moritzbrantner/nlp-stack"
  )]
  | length == 52
    and all(.[]; .automatic_publish_eligible == true)
' "$scratch_root/rust-packages/docs/repository-split/package-ownership.json" >/dev/null

export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"$repository_root/target"}
cargo check --manifest-path "$scratch_root/native-whisperx/Cargo.toml" -p native-whisperx --config "$patch_config"
cargo check --manifest-path "$scratch_root/media-similarity/backend/Cargo.toml" --bin image-similarity-service
echo "media-similarity baseline pinned; candidate migration deferred to rust-packages#124"
echo "youtube-corpus baseline pinned; candidate migration deferred to rust-packages#125"
echo "document-search baseline pinned; WASM migration deferred to rust-packages#127"
echo "philosophy-extractor baseline pinned; candidate migration deferred to rust-packages#128"

echo "NLP wave 1 downstream consumer gate passed"
