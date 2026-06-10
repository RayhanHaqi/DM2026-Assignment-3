#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
exec python scripts/run_break080_10slot.py --max-slots "${MAX_SLOTS:-10}" --device "${DEVICE:-cuda}" "$@"
