#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
data_dir=${OXFORD_DINO_DATA_DIR:-/private/tmp/oxford_vgg_dinosaur}
real_output_dir=${OXFORD_DINO_OUTPUT_DIR:-/private/tmp/oxford_vgg_dinosaur_probe}
sdp_output_dir=${TPAMI_SDP_OUTPUT_DIR:-/private/tmp/oxford_vgg_dinosaur_sdp}
sdp_python=${TPAMI_SDP_PYTHON:-python3}

mkdir -p "$sdp_output_dir"

# Rebuild the point-level input every time so the SDP population cannot silently
# use stale corrections or camera data. The preparation step reuses only
# checksum-valid cached Oxford source files and honors OXFORD_DINO_OFFLINE.
OXFORD_DINO_DATA_DIR="$data_dir" \
  OXFORD_DINO_OUTPUT_DIR="$real_output_dir" \
  "$script_dir/run_oxford_dinosaur_real.sh"

"$sdp_python" "$repo_dir/benchmarks/sdp_global_comparator.py" \
  --tracks "$data_dir/viff.xy" \
  --cameras "$data_dir/dino_cameras.tsv" \
  --points "$real_output_dir/points.csv" \
  --output "$sdp_output_dir/sdp_points.csv" \
  --pair-output "$sdp_output_dir/sdp_pairs.csv" \
  --summary "$sdp_output_dir/summary.txt" \
  --all-rows \
  --expected-samples 27080

{
  "$sdp_python" --version
  "$sdp_python" -c 'import platform, numpy, cvxopt; print("platform=" + platform.platform()); print("numpy=" + numpy.__version__); print("cvxopt=" + cvxopt.__version__)'
} > "$sdp_output_dir/environment.txt"

shasum -a 256 \
  "$repo_dir/benchmarks/sdp_global_comparator.py" \
  "$repo_dir/scripts/run_sdp_global_comparator.sh" \
  "$repo_dir/scripts/requirements-sdp.txt" \
  "$sdp_output_dir/sdp_points.csv" \
  "$sdp_output_dir/sdp_pairs.csv" \
  "$sdp_output_dir/summary.txt" \
  "$sdp_output_dir/environment.txt"
