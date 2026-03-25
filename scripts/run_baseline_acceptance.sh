#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build"
OUT_DIR="$ROOT/results/raw"
STAMP="$(date -u +"%Y-%m-%dT%H%M%SZ")"
OUT_FILE="$OUT_DIR/${STAMP}_baseline_acceptance.txt"

mkdir -p "$OUT_DIR"

cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_DIR" -j >/dev/null

"$BUILD_DIR/baseline_acceptance" > "$OUT_FILE"

echo "Wrote:"
echo "  $OUT_FILE"
