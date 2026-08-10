#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
data_dir=${OXFORD_DINO_DATA_DIR:-/private/tmp/oxford_vgg_dinosaur}
output_dir=${OXFORD_DINO_OUTPUT_DIR:-/private/tmp/oxford_vgg_dinosaur_probe}
cxx=${CXX:-c++}

mkdir -p "$data_dir" "$output_dir"

if [ "${OXFORD_DINO_OFFLINE:-0}" = "1" ]; then
  python3 "$script_dir/prepare_oxford_dinosaur.py" --dest "$data_dir" --offline
else
  python3 "$script_dir/prepare_oxford_dinosaur.py" --dest "$data_dir"
fi

if [ -n "${EIGEN3_INCLUDE_DIR:-}" ]; then
  eigen_include=$EIGEN3_INCLUDE_DIR
elif [ -d /usr/local/include/eigen3 ]; then
  eigen_include=/usr/local/include/eigen3
elif [ -d /opt/homebrew/include/eigen3 ]; then
  eigen_include=/opt/homebrew/include/eigen3
elif [ -d /usr/include/eigen3 ]; then
  eigen_include=/usr/include/eigen3
elif command -v pkg-config >/dev/null 2>&1 && pkg-config --exists eigen3; then
  eigen_include=$(pkg-config --cflags-only-I eigen3 | awk '{sub(/^-I/, "", $1); print $1; exit}')
else
  echo "Eigen 3 include directory not found; set EIGEN3_INCLUDE_DIR." >&2
  exit 1
fi
if [ -z "$eigen_include" ] || [ ! -d "$eigen_include/Eigen" ]; then
  echo "Resolved Eigen 3 include directory is invalid: $eigen_include" >&2
  exit 1
fi

binary="$output_dir/oxford_dinosaur_real"
"$cxx" -std=c++20 -O3 \
  -I"$repo_dir" -I"$repo_dir/benchmarks" -I"$eigen_include" \
  "$repo_dir/benchmarks/oxford_dinosaur_real.cpp" -o "$binary"

"$binary" \
  --tracks "$data_dir/viff.xy" \
  --cameras "$data_dir/dino_cameras.tsv" \
  --pair-csv "$output_dir/pairs.csv" \
  --point-csv "$output_dir/points.csv" \
  | tee "$output_dir/summary.txt"

shasum -a 256 \
  "$repo_dir/scripts/run_oxford_dinosaur_real.sh" \
  "$repo_dir/scripts/prepare_oxford_dinosaur.py" \
  "$repo_dir/benchmarks/oxford_dinosaur_real.cpp" \
  "$data_dir/provenance.json" \
  "$output_dir/summary.txt" \
  "$output_dir/pairs.csv" \
  "$output_dir/points.csv" \
  "$binary"
