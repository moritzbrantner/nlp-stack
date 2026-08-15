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

media="$scratch_root/media-similarity/backend/Cargo.toml"
replace_exact "$media" 'text-transcripts = { package = "moenarch-text-transcripts", version = "0.1.2", features = ["native"] }' 'text-transcripts = { package = "moenarch-text-transcripts", version = "=0.1.3" }'
replace_exact "$media" $'publish = false\n\n[lib]' $'publish = false\n\n[workspace]\n\n[lib]'
media_models="$scratch_root/media-similarity/backend/src/workers/media/models.rs"
replace_exact "$media_models" $'use text_transcripts::{WhisperCppModel, WhisperCppModelStore};\n' ''
replace_exact "$media_models" $'pub fn parse_whisper_cpp_model(value: &str) -> Result<WhisperCppModel, String> {\n    let normalized = value.trim();\n    WhisperCppModel::ALL\n        .into_iter()\n        .find(|model| model.id().eq_ignore_ascii_case(normalized))\n        .ok_or_else(|| format!("Unknown whisper.cpp model `{normalized}`"))\n}\n\npub fn audio_transcription_model_store(settings: &Settings) -> WhisperCppModelStore {\n    settings\n        .audio_transcription_cache_dir\n        .clone()\n        .map(WhisperCppModelStore::new)\n        .unwrap_or_default()\n}\n\n' ''

youtube="$scratch_root/youtube-corpus/Cargo.toml"
replace_exact "$youtube" 'text-core = { package = "moritzbrantner-text-core", path = "../rust-packages/crates/text/text-core" }' 'text-core = { package = "moenarch-text-core", version = "=0.1.1" }'
replace_exact "$youtube" 'text-embeddings = { package = "moritzbrantner-text-embeddings", path = "../rust-packages/crates/text/text-embeddings" }' 'text-embeddings = { package = "moenarch-text-embeddings", version = "=0.1.1" }'
replace_exact "$youtube" 'text-lexical = { package = "moritzbrantner-text-lexical", path = "../rust-packages/crates/text/text-lexical" }' 'text-lexical = { package = "moenarch-text-lexical", version = "=0.1.1" }'
replace_exact "$youtube" 'text-transcripts = { package = "moritzbrantner-text-transcripts", path = "../rust-packages/crates/text/text-transcripts" }' 'text-transcripts = { package = "moenarch-text-transcripts", version = "=0.1.3" }'
replace_exact "$youtube" $'text-transcripts = { package = "moenarch-text-transcripts", version = "=0.1.3" }\n' $'text-transcripts = { package = "moenarch-text-transcripts", version = "=0.1.3" }\nlegacy-text-transcripts = { package = "moritzbrantner-text-transcripts", version = "=0.1.1" }\n'
replace_exact "$youtube" 'runtime-core = { package = "moritzbrantner-runtime-core", path = "../rust-packages/crates/runtime/runtime-core" }' 'runtime-core = { package = "moenarch-runtime-core", version = "=0.2.1" }'
replace_exact "$youtube" 'jobs-core = { package = "moritzbrantner-jobs-core", path = "../rust-packages/crates/jobs/jobs-core" }' 'jobs-core = { package = "moenarch-jobs-core", version = "=0.1.2" }'
replace_exact "$youtube" 'video-analysis-ffmpeg = { package = "moritzbrantner-video-analysis-ffmpeg", path = "../rust-packages/crates/video/video-analysis-ffmpeg" }' 'video-analysis-ffmpeg = { package = "moenarch-video-analysis-ffmpeg", version = "=0.1.1" }'
replace_exact "$youtube" 'video-analysis-ingest = { package = "moritzbrantner-video-analysis-ingest", path = "../rust-packages/crates/video/video-analysis-ingest" }' 'video-analysis-ingest = { package = "moenarch-video-analysis-ingest", version = "=0.1.0" }'
replace_exact "$youtube" $'\n[dev-dependencies]\n' $'\n[workspace]\n\n[dev-dependencies]\n'
sed -i 's/text_transcripts/legacy_text_transcripts/g' "$scratch_root/youtube-corpus/src/asr.rs"
replace_exact "$scratch_root/youtube-corpus/src/asr.rs" 'parsed.segments.into_iter().map(Into::into).collect()' $'parsed\n            .segments\n            .into_iter()\n            .map(|segment| text_transcripts::TranscriptSegmentContract {\n                index: segment.index,\n                start_seconds: segment.start_seconds,\n                end_seconds: segment.end_seconds,\n                text: segment.text,\n                language: segment.language,\n                speaker: segment.speaker,\n                confidence: segment.confidence,\n                is_final: segment.is_final,\n                words: Vec::new(),\n                chars: Vec::new(),\n                attributes: Default::default(),\n            })\n            .collect()'

document="$scratch_root/document-search/package.json"
replace_exact "$document" '"@moritzbrantner/text-core-wasm": "file:../rust-packages/packages/text-core-wasm"' '"@moritzbrantner/text-core-wasm": "file:__NLP_STACK__/packages/text-core-wasm"'
replace_exact "$document" '"@moritzbrantner/text-index-wasm": "file:../rust-packages/packages/text-index-wasm"' '"@moritzbrantner/text-index-wasm": "file:__NLP_STACK__/packages/text-index-wasm"'
sed -i "s|__NLP_STACK__|$repository_root|g" "$document"
document_index="$scratch_root/document-search/src/wasm/textIndex.ts"
replace_exact "$document_index" $'import initWasm, {\n  runOperation,\n} from "@moritzbrantner/text-index-wasm/pkg/moritzbrantner_text_index_wasm.js";\nimport type { SurfaceResponse } from "@moritzbrantner/text-index-wasm";\n\nlet initPromise: Promise<unknown> | undefined;\n' $'import { runOperation } from "@moritzbrantner/text-index-wasm";\nimport type { SurfaceResponse } from "@moritzbrantner/text-index-wasm";\n'
replace_exact "$document_index" $'  initPromise ??= initWasm();\n  await initPromise;\n  return fromWasm(runOperation(request)) as SurfaceResponse;\n' $'  return (await runOperation(request)) as SurfaceResponse;\n'
replace_exact "$document_index" $'\nfunction fromWasm(value: unknown): unknown {\n  if (value instanceof Map) {\n    return Object.fromEntries(\n      Array.from(value.entries(), ([key, entry]) => [key, fromWasm(entry)]),\n    );\n  }\n  if (Array.isArray(value)) {\n    return value.map(fromWasm);\n  }\n  if (value && typeof value === "object") {\n    return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, fromWasm(entry)]));\n  }\n  return value;\n}\n' $'\n'

philosophy="$scratch_root/philosophy-extractor/Cargo.toml"
replace_exact "$philosophy" 'text-core = { path = "../rust-packages/crates/text/text-core" }' 'text-core = { package = "moenarch-text-core", version = "=0.1.1" }'
replace_exact "$philosophy" 'text-embeddings = { path = "../rust-packages/crates/text/text-embeddings" }' 'text-embeddings = { package = "moenarch-text-embeddings", version = "=0.1.1" }'
replace_exact "$philosophy" 'text-linguistics = { path = "../rust-packages/crates/text/text-linguistics" }' 'text-linguistics = { package = "moenarch-text-linguistics", version = "=0.1.1" }'
replace_exact "$philosophy" 'text-retrieval = { path = "../rust-packages/crates/text/text-retrieval" }' 'text-retrieval = { package = "moenarch-text-retrieval", version = "=0.1.1" }'
replace_exact "$scratch_root/philosophy-extractor/packages/pipeline/src/pipeline/relate.rs" $'                    rerank_window: candidates.len().min(16).max(1),\n' $'                    rerank_window: candidates.len().min(16).max(1),\n                    rerank: false,\n'

# These consumers still use compatibility packages that are intentionally not
# part of this release wave. Pin and assert those baselines instead of silently
# inventing a migration mapping inside a publication gate.
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
cargo check --manifest-path "$scratch_root/media-similarity/backend/Cargo.toml" --bin image-similarity-service --config "$patch_config"
cargo check --manifest-path "$scratch_root/youtube-corpus/Cargo.toml" --config "$patch_config"
cargo check --manifest-path "$scratch_root/philosophy-extractor/Cargo.toml" -p philosophy-extractor --config "$patch_config"
# These packages intentionally ignore wasm-pack output. Generate the two JS/WASM
# surfaces used by document-search inside every clean publication checkout.
bun run --cwd "$repository_root/packages/text-core-wasm" build
bun run --cwd "$repository_root/packages/text-index-wasm" build
bun install --cwd "$scratch_root/document-search"
bun test --cwd "$scratch_root/document-search"
bun run --cwd "$scratch_root/document-search" build

echo "NLP wave 1 downstream consumer gate passed"
