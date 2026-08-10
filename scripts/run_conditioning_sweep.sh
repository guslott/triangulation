#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build/conditioning-sweep"
OUT_DIR="$ROOT/evidence/conditioning"

cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --target conditioning_sweep -j
"$BUILD_DIR/conditioning_sweep" \
  "$OUT_DIR/2026-08-10_conditioning_cases.csv" \
  "$OUT_DIR/2026-08-10_conditioning_bins.csv" \
  "$OUT_DIR/2026-08-10_conditioning_summary.md"
