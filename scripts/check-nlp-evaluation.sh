#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

artifact_dir="${NLP_EVAL_ARTIFACT_DIR:-$root/.artifacts/nlp-eval}"
corpus="$root/evaluation/corpora/sentence-boundaries-v1.jsonl"
predictions="$artifact_dir/sentence-boundaries-v1.predictions.jsonl"
report="$artifact_dir/text-core-sentence-boundaries-v1.json"
minimum_f1="${NLP_EVAL_MIN_BOUNDARY_F1:-1.0}"

mkdir -p "$artifact_dir"

cargo run --quiet --locked -p moenarch-text-core --example sentence_boundary_predictions -- "$corpus" > "$predictions"

python3 -m evaluation.runner boundaries \
  --gold "$corpus" \
  --predictions "$predictions" \
  --suite sentence-boundaries-v1 \
  --system text-core/builtin-sentence-boundaries \
  --source-revision "$(git rev-parse HEAD)" \
  --output "$report" \
  --min-f1 "$minimum_f1"

printf '%s\n' "NLP evaluation report: $report"
