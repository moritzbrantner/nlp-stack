#!/usr/bin/env bash
set -euo pipefail

scripts/check-preflight.sh
bun run text-wasm:test:all
python3 scripts/check_release_plan.py --package-all docs/repository-split/release-plan.json
