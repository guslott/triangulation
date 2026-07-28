#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${1:-$(date -u +"%Y-%m-%dT%H%M%SZ")}"
BUILD_DIR="${2:-$ROOT/build}"
RAW_DIR="${3:-$ROOT/results/raw}"
RUN_STATUS="${4:-not-recorded}"
EVIDENCE_DIR="$ROOT/evidence/runs"
OUT="$EVIDENCE_DIR/${STAMP}_manifest.md"
PROCESSED_DIR="$ROOT/results/processed"
MPL_CACHE_DIR="$PROCESSED_DIR/.mplcache"
XDG_CACHE_DIR="$PROCESSED_DIR/.cache"

mkdir -p "$EVIDENCE_DIR" "$MPL_CACHE_DIR" "$XDG_CACHE_DIR/fontconfig"
export MPLCONFIGDIR="$MPL_CACHE_DIR"
export XDG_CACHE_HOME="$XDG_CACHE_DIR"

CPU_MODEL="unknown"
if command -v sysctl >/dev/null 2>&1; then
  CPU_MODEL="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
fi
if [[ "$CPU_MODEL" == "unknown" ]]; then
  CPU_MODEL="$(uname -m)"
fi

GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unavailable)"
GIT_BRANCH="$(git -C "$ROOT" branch --show-current 2>/dev/null || echo unavailable)"
GIT_STATUS="$(git -C "$ROOT" status --short 2>/dev/null || true)"
if [[ -z "$GIT_STATUS" ]]; then
  GIT_STATE="clean"
else
  GIT_STATE="dirty"
fi

hash_file() {
  local path="$1"
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    sha256sum "$path" | awk '{print $1}'
  fi
}

{
  echo "# Reproducibility run manifest"
  echo
  echo "- Run stamp (UTC): \`$STAMP\`"
  echo "- Pipeline exit status: \`$RUN_STATUS\`"
  echo "- Git commit: \`$GIT_COMMIT\`"
  echo "- Git branch: \`$GIT_BRANCH\`"
  echo "- Git state before manifest creation: \`$GIT_STATE\`"
  echo "- Build directory: \`$BUILD_DIR\`"
  echo "- Raw-output directory: \`$RAW_DIR\`"
  echo
  if [[ "$GIT_STATE" == "dirty" ]]; then
    echo "## Dirty-worktree paths"
    echo
    echo '```text'
    printf '%s\n' "$GIT_STATUS"
    echo '```'
    echo
  fi
  echo "## System"
  echo
  echo "- OS: $(uname -srm)"
  echo "- CPU: ${CPU_MODEL}"
  echo
  echo "## Toolchain"
  echo
  echo "- C++ compiler: $(c++ --version | head -n 1)"
  echo "- CMake: $(cmake --version | head -n 1)"
  echo "- Python: $(python3 --version 2>/dev/null || true)"
  echo
  echo "## Python packages"
  echo
  python3 - <<'PY'
import importlib
for name in ["numpy", "matplotlib"]:
    try:
        module = importlib.import_module(name)
        print(f"- {name}: {getattr(module, '__version__', 'unknown')}")
    except Exception:
        print(f"- {name}: unavailable")
PY
  echo
  echo "## Commands"
  echo
  echo '```sh'
  echo "cmake -S . -B build/$STAMP -DCMAKE_BUILD_TYPE=Release"
  echo "cmake --build build/$STAMP -j"
  echo "build/$STAMP/theorem_regression"
  echo "build/$STAMP/baseline_acceptance"
  echo "build/$STAMP/bench_speed"
  echo "build/$STAMP/bench_scaling"
  echo "build/$STAMP/bench_correctness --cert-all --csv-out results/raw/${STAMP}_bench_correctness_points.csv"
  echo "build/$STAMP/bench_approximation"
  echo '```'
  echo
  echo "## Deterministic seeds"
  echo
  echo '```text'
  rg -n "mt19937[^;]*\(|_seed=" "$ROOT/benchmarks"/*.cpp || true
  echo '```'
  echo
  echo "## CMake cache"
  echo
  if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo '```text'
    rg -n "CMAKE_BUILD_TYPE:STRING|CMAKE_CXX_COMPILER:FILEPATH|CMAKE_CXX_FLAGS_RELEASE:STRING|EIGEN3_INCLUDE_DIR:PATH|Eigen3_DIR:PATH" "$BUILD_DIR/CMakeCache.txt" || true
    echo '```'
  else
    echo "_No CMake cache was produced._"
  fi
  echo
  echo "## Artifact SHA-256"
  echo
  found_artifact=0
  for artifact in "$RAW_DIR"/"$STAMP"_*; do
    if [[ ! -f "$artifact" ]]; then
      continue
    fi
    found_artifact=1
    echo "- \`$(basename "$artifact")\`: \`$(hash_file "$artifact")\`"
  done
  if [[ "$found_artifact" -eq 0 ]]; then
    echo "_No stamped raw artifacts were present; the run stopped before producing them._"
  fi
} > "$OUT"

echo "Wrote $OUT"
