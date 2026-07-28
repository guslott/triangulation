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
