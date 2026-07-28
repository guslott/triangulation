#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +"%Y-%m-%dT%H%M%SZ")"
BUILD_DIR="$ROOT/build/$STAMP"
OUT_DIR="$ROOT/results/raw"
POINTS_OUT="$OUT_DIR/${STAMP}_bench_correctness_points.csv"

finalize_run() {
  local status=$?
  trap - EXIT
  "$ROOT/scripts/capture_env.sh" "$STAMP" "$BUILD_DIR" "$OUT_DIR" "$status" || true
  exit "$status"
}
trap finalize_run EXIT

mkdir -p "$OUT_DIR"

cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_DIR" -j >/dev/null

"$BUILD_DIR/theorem_regression" > "$OUT_DIR/${STAMP}_theorem_regression.txt" 2>&1
"$BUILD_DIR/baseline_acceptance" > "$OUT_DIR/${STAMP}_baseline_acceptance.txt" 2>&1
"$BUILD_DIR/bench_speed" > "$OUT_DIR/${STAMP}_bench_speed.txt" 2>&1
"$BUILD_DIR/bench_scaling" > "$OUT_DIR/${STAMP}_bench_scaling.txt" 2>&1
"$BUILD_DIR/bench_correctness" --cert-all --csv-out "$POINTS_OUT" > "$OUT_DIR/${STAMP}_bench_correctness.txt" 2>&1
"$BUILD_DIR/bench_approximation" > "$OUT_DIR/${STAMP}_bench_approximation.txt" 2>&1

if [[ -x "$ROOT/scripts/build_figures.sh" ]]; then
  "$ROOT/scripts/build_figures.sh" --stamp "$STAMP"
fi

echo "Wrote:"
echo "  $OUT_DIR/${STAMP}_theorem_regression.txt"
echo "  $OUT_DIR/${STAMP}_baseline_acceptance.txt"
echo "  $OUT_DIR/${STAMP}_bench_speed.txt"
echo "  $OUT_DIR/${STAMP}_bench_scaling.txt"
echo "  $OUT_DIR/${STAMP}_bench_correctness.txt"
echo "  $OUT_DIR/${STAMP}_bench_approximation.txt"
echo "  $POINTS_OUT"
