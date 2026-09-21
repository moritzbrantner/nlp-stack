#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

cargo metadata --locked --format-version 1 >/dev/null
bun install --frozen-lockfile
bun install --cwd web --frozen-lockfile
bun run nlp-app-ui:build
bash web/scripts/build-wasm-assets.sh
bun run --cwd web test
bun run --cwd web typecheck
bun run --cwd web storybook:build
GITHUB_ACTIONS=false bun run --cwd web build
bun run --cwd web e2e
# Leave the export in the caller's deployment mode (including the Pages base path).
if [[ "${GITHUB_ACTIONS:-false}" == "true" ]]; then
  bun run --cwd web build
fi
