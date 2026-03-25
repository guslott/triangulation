#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build"
OUT_DIR="$ROOT/results/raw"
STAMP="$(date -u +"%Y-%m-%dT%H%M%SZ")"
POINTS_OUT="$OUT_DIR/${STAMP}_bench_correctness_points.csv"

mkdir -p "$OUT_DIR"

cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_DIR" -j >/dev/null

"$BUILD_DIR/bench_speed" > "$OUT_DIR/${STAMP}_bench_speed.txt"
"$BUILD_DIR/bench_scaling" > "$OUT_DIR/${STAMP}_bench_scaling.txt"
"$BUILD_DIR/bench_correctness" --cert-all --csv-out "$POINTS_OUT" > "$OUT_DIR/${STAMP}_bench_correctness.txt"
"$BUILD_DIR/bench_approximation" > "$OUT_DIR/${STAMP}_bench_approximation.txt"

if [[ -x "$ROOT/scripts/build_figures.sh" ]]; then
  "$ROOT/scripts/build_figures.sh" --stamp "$STAMP"
fi

if [[ -x "$ROOT/scripts/capture_env.sh" ]]; then
  "$ROOT/scripts/capture_env.sh"
fi

echo "Wrote:"
echo "  $OUT_DIR/${STAMP}_bench_speed.txt"
echo "  $OUT_DIR/${STAMP}_bench_scaling.txt"
echo "  $OUT_DIR/${STAMP}_bench_correctness.txt"
echo "  $OUT_DIR/${STAMP}_bench_approximation.txt"
echo "  $POINTS_OUT"
