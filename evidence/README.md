# Benchmark evidence index

This directory records immutable manifests for revision benchmark runs.  Each
manifest identifies the exact commit, clean/dirty state, commands, toolchain,
seeds, and SHA-256 hashes of the locally retained raw outputs.

## Run dispositions

- `2026-07-28T143835Z_manifest.md`: **superseded for timing claims** because
  methods were timed in sequential blocks.  The run remains evidence for its
  passing theorem, baseline-acceptance, and correctness checks.
- The newest manifest produced after the interleaved timing protocol was
  committed is the authoritative timing run.  It must report a clean Git state
  and a zero pipeline exit status before its numbers are copied into the paper.

Raw and processed benchmark outputs stay under the ignored `results/` tree so
large point-level CSV files do not enter source control.  Their hashes in each
manifest make the local artifacts auditable; release packaging may include the
corresponding raw files separately.
