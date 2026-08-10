# Benchmark evidence index

This directory records immutable manifests for revision benchmark runs.  Each
manifest identifies the exact commit, clean/dirty state, commands, toolchain,
seeds, and SHA-256 hashes of the locally retained raw outputs.

## Run dispositions

- `2026-07-28T143835Z_manifest.md`: **superseded for timing claims** because
  methods were timed in sequential blocks.  The run remains evidence for its
  passing theorem, baseline-acceptance, and correctness checks.
- `2026-07-28T145115Z_manifest.md`: **authoritative revision run**.  It was
  produced from clean commit `e53b19c`, reports a zero pipeline exit status,
  and uses the deterministic interleaved timing protocol.

The small authoritative text outputs are force-tracked under `results/raw/` so
the reported aggregate evidence travels with the source. Large point-level CSV
files and generated plots/tables remain ignored; their hashes in the manifest
make locally retained artifacts auditable, and the release package may include
them separately.

## Conditioning evidence

`conditioning/` contains the deterministic coefficient-space sweep added for
the major revision. Its tracked case and bin tables cover singular-value-ratio,
affine-limit, PSD-endpoint, and common-normalization stress families and can be
regenerated with `./scripts/run_conditioning_sweep.sh`.

## Real-data evidence

`real_data/` contains compact results from the official Oxford VGG Dinosaur
tracks and reconstruction cameras. The tracked package includes pinned source
provenance, the aggregate solver summary, and pair-level results. The larger
point-level table is deliberately omitted; its byte size and SHA-256 hash are
recorded in `real_data/README.md`. Regenerate locally with
`./scripts/run_oxford_dinosaur_real.sh`.

## SDP-comparator evidence

`sdp_comparator/` contains the full-population lifted-SDP cross-check added as
the methodologically distinct global comparator. It archives pair-level
diagnostics across all 27,080 rows and 364 nonempty Oxford view pairs, the
gate-bearing PASS summary, the exact Python/NumPy/CVXOPT environment record,
and the hash of the locally retained point-level output. Regenerate with
`TPAMI_SDP_PYTHON=/path/to/python ./scripts/run_sdp_global_comparator.sh`.
