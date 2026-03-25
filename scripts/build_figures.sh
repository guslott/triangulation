#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT/results/raw"
PROCESSED_DIR="$ROOT/results/processed"
MPL_CACHE_DIR="$PROCESSED_DIR/.mplcache"
XDG_CACHE_DIR="$PROCESSED_DIR/.cache"

mkdir -p "$RAW_DIR" "$PROCESSED_DIR" "$MPL_CACHE_DIR" "$XDG_CACHE_DIR/fontconfig"
export MPLCONFIGDIR="$MPL_CACHE_DIR"
export XDG_CACHE_HOME="$XDG_CACHE_DIR"

python3 "$ROOT/scripts/build_figures.py" \
  --raw-dir "$RAW_DIR" \
  --processed-dir "$PROCESSED_DIR" \
  "$@"
