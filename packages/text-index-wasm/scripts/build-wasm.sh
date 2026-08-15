#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
wasm-pack build crates/bindings/text-index-wasm --target web --out-dir ../../../packages/text-index-wasm/pkg
