#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/manifests/repro_env.md"
BUILD_DIR="$ROOT/build"
PROCESSED_DIR="$ROOT/results/processed"
MPL_CACHE_DIR="$PROCESSED_DIR/.mplcache"
XDG_CACHE_DIR="$PROCESSED_DIR/.cache"

mkdir -p "$MPL_CACHE_DIR" "$XDG_CACHE_DIR/fontconfig"
export MPLCONFIGDIR="$MPL_CACHE_DIR"
export XDG_CACHE_HOME="$XDG_CACHE_DIR"

CPU_MODEL="unknown"
if command -v sysctl >/dev/null 2>&1; then
  CPU_MODEL="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
fi
if [[ "$CPU_MODEL" == "unknown" ]]; then
  CPU_MODEL="$(uname -m)"
fi

{
  echo "# Reproducibility Environment Manifest"
  echo
  echo "Generated (UTC): $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo
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
  echo "## Python Packages"
  echo
  python3 - <<'PY'
import importlib
pkgs = ["numpy", "matplotlib"]
for name in pkgs:
    try:
        m = importlib.import_module(name)
        print(f"- {name}: {getattr(m, '__version__', 'unknown')}")
    except Exception:
        print(f"- {name}: unavailable")
PY
  echo
  echo "## CMake Cache (if present)"
  echo
  if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo '```'
    rg -n "CMAKE_BUILD_TYPE:STRING|CMAKE_CXX_COMPILER:FILEPATH|CMAKE_CXX_FLAGS_RELEASE:STRING|EIGEN3_INCLUDE_DIR:PATH|Eigen3_DIR:PATH" "$BUILD_DIR/CMakeCache.txt" || true
    echo '```'
  else
    echo "_No CMake cache found at \`$BUILD_DIR/CMakeCache.txt\`._"
  fi
} > "$OUT"

echo "Wrote $OUT"
