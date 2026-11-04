#!/usr/bin/env bash
# Installs the pinned Stanza sidecar environment and its digest-verified models.
# Idempotent: re-running reuses a matching venv and skips verified model files.
#
#   bash sidecars/stanza/setup.sh [--languages en,de,es] [--model-dir DIR]
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required (https://docs.astral.sh/uv/)" >&2
  exit 1
fi

# The venv lives in the ignored sidecars/stanza/.venv; --frozen refuses lock drift.
uv sync --frozen --no-dev
uv run --frozen --no-dev python -m stanza_sidecar.fetch "$@"
