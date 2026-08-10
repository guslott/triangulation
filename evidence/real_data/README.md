# Oxford VGG Dinosaur real-data probe

This directory is the compact, tracked record of the 2026-08-10 DATA-1
experiment. It validates two-view image correction on real tracked
correspondences; it does **not** claim ground-truth 3D reconstruction accuracy.

## Source and scope

The inputs come from the official [Oxford VGG multi-view data
page](https://www.robots.ox.ac.uk/~vgg/data/mview/):

- `README.txt`: `0f88ffa9d193d7cbd6e784092f02c8c9b009d367b8e191670f88f82b2c752c8b`
- `viff.xy`: `a23e0044853968dcfbc899bcf80cbdfd5c04f3664ccd7ccd9a7b5701f6d53a8b`
- `dino_Ps.mat`: `61adf55edf43764ab50ce389fd3e95516046cd4ed833584b6f6a2e7ea268d281`

The source files are downloaded only to `/private/tmp` and are not vendored.
The official README declares 4,838 tracks, while the checksummed `viff.xy`
contains 4,983 well-formed 72-value rows. The preprocessing script fails closed
on malformed rows, camera-array shape changes, or source checksum changes, and
the discrepancy is preserved in `provenance.json`.

The observed file contains 16,432 visible point/view measurements across 36
views. Of 630 possible unordered view pairs, 364 have at least one co-visible
track, yielding 27,080 pair-correspondence evaluations. A track can contribute
to several view pairs, so these evaluations are not statistically independent.

## Geometry and solver checks

For each nonempty pair, the harness derives

\[
F=[e']_\times P'P^+,
\]

projects the result explicitly to rank two, and independently checks it against
the 4-by-4 camera-row minor construction. It also checks projected synthetic
homogeneous points and verifies the measured tracks prefer the intended
`x'^T F x` convention over its transpose.

The current theorem-aligned Lott solver and the Hartley--Sturm reference are
then run on every co-visible correspondence. The recorded run produced:

| Check | Result |
|---|---:|
| Lott certified / failures | 27,080 / 0 |
| Lott status | 27,080 regular-interior |
| Hartley--Sturm finite | 27,080 / 27,080 |
| Both normalized residuals at most `1e-12` | 27,080 / 27,080 |
| Maximum Lott normalized residual | `9.04e-20` |
| Mean objective, Lott / Hartley--Sturm | `0.26358766954360291` / `0.26358766954360435` |
| Median / p95 absolute objective gap | `1.31e-14` / `1.09e-13` |
| Maximum absolute / relative objective gap | `2.05e-12` / `4.46e-13` |
| Scaled objective disagreements above `1e-8` | 0 |
| Maximum camera-formula disagreement | `3.68e-15` |
| Maximum projected-point normalized residual | `2.42e-17` |
| Maximum final `sigma3/sigma1` | `9.14e-22` |
| Median measured-track Sampson residual, `F` / `F^T` | `0.200` / `8.61` pixels |
| Selected x / y / z / w charts | 9,970 / 3,281 / 9,921 / 3,908 |

## Reproduction

An online run downloads and checksum-validates the three small official files:

```bash
./scripts/run_oxford_dinosaur_real.sh
```

To prohibit network access and require valid cached inputs:

```bash
OXFORD_DINO_OFFLINE=1 \
OXFORD_DINO_DATA_DIR=/private/tmp/oxford_vgg_dinosaur \
OXFORD_DINO_OUTPUT_DIR=/private/tmp/oxford_vgg_dinosaur_probe \
./scripts/run_oxford_dinosaur_real.sh
```

The runner uses Bash `errexit` plus `pipefail`, so a nonzero benchmark exit is
not masked by `tee`. A normal CMake target is also available:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target oxford_dinosaur_real
```

The archived run used base commit
`1bc33f4b46f110152f85908323729210eef7b629` with the uncommitted revision
changes represented by these source hashes:

- `scripts/run_oxford_dinosaur_real.sh`: `2d070a97230391477f3e53fbe569c13df7d7b99fb099c22753c0eda72a77a862`
- `scripts/prepare_oxford_dinosaur.py`: `3b9ba368a4feca1a369b111f55cce4b012fb4393cd262d3ec00775d1c36a0923`
- `benchmarks/oxford_dinosaur_real.cpp`: `987a1b2489329a87adaef8488dc4ef10f0de956cf1fdf11547987285d5b78089`

Toolchain: Apple Clang 21.0.0, C++20, Eigen 3.4.0, CMake 3.31.6, arm64
Darwin 25.5.0. The temporary release binary hash was
`22b4f50e132e1a6a0d48d846b4bb7c7c4affa6e786456aa13426b8256190aee7`.

## Archived artifacts

| File | Bytes | Rows | SHA-256 |
|---|---:|---:|---|
| `provenance.json` | 1,348 | -- | `0fafc96909b62f036d46315eec09d1af34b3f16f361c17eec77565ccefc127f0` |
| `summary.txt` | 1,601 | 43 | `4aee53cceeff7f854c830c5e893b31212df1466337f0e586535e8f525ce9dbf9` |
| `pairs.csv` | 110,200 | 365 | `72b6c99d71d44e43f6e5a0b11fca3a2fbfea0049fc38315e4f420279a9d24dd2` |
| `points.csv` (local only; intentionally omitted) | 6,363,142 | 27,081 | `679cdd5e75c29515d928938abe2e6c06768b8eed787204aba0c69092f201a95e` |

The network and cached-input runs produced identical `summary.txt`,
`pairs.csv`, and omitted `points.csv` hashes. The camera text emitted by the
standard-library MATLAB-v5 parser was also compared with SciPy's interpretation
and matched bit-for-bit.

## Verification performed

- clean CMake configure and build of `oxford_dinosaur_real`,
  `theorem_regression`, and `conditioning_sweep`;
- CTest pass for the theorem regression and conditioning sweep, including
  580/580 theorem checks;
- baseline-acceptance T1--T4 overall pass;
- offline runner exit status zero on the pinned cached data;
- AddressSanitizer and UndefinedBehaviorSanitizer run with no findings;
- explicit `pipefail` failure probe returned status 1; and
- `git diff --check` passed.

The cameras are reconstruction estimates supplied with the dataset, not
independent 3D ground truth. Accordingly, the defensible claim is real-image
correction feasibility and objective agreement with Hartley--Sturm, not metric
3D accuracy or statistical generalization.
