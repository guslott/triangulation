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

Raw and processed benchmark outputs stay under the ignored `results/` tree so
large point-level CSV files do not enter source control.  Their hashes in each
manifest make the local artifacts auditable; release packaging may include the
corresponding raw files separately.
