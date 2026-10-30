#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

bash "$ROOT_DIR/packages/text-core-wasm/scripts/build-wasm.sh"
bash "$ROOT_DIR/packages/text-analysis-wasm/scripts/build-wasm.sh"

rm -rf "$ROOT_DIR/web/public/wasm"
mkdir -p "$ROOT_DIR/web/public/wasm"

cp "$ROOT_DIR/packages/text-core-wasm/pkg/moenarch_text_core_wasm.js" "$ROOT_DIR/web/public/wasm/"
cp "$ROOT_DIR/packages/text-core-wasm/pkg/moenarch_text_core_wasm_bg.wasm" "$ROOT_DIR/web/public/wasm/"
cp "$ROOT_DIR/packages/text-analysis-wasm/pkg/moenarch_text_analysis_wasm.js" "$ROOT_DIR/web/public/wasm/"
cp "$ROOT_DIR/packages/text-analysis-wasm/pkg/moenarch_text_analysis_wasm_bg.wasm" "$ROOT_DIR/web/public/wasm/"

# Canonical browser NER module (no inference copy in the Pages app).
rm -rf "$ROOT_DIR/web/public/ner"
mkdir -p "$ROOT_DIR/web/public/ner"
cp "$ROOT_DIR"/packages/text-linguistics-wasm/browser-ner{,-core,-worker}.js "$ROOT_DIR/web/public/ner/"
