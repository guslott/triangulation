# Fast Optimal L2 Two-View Triangulation via Orthogonal Distance to the Epipolar Quadric

**Author:** Gus Lott (guslott@gmail.com)

A bounded-cost method for two-view triangulation based on orthogonal distance to the epipolar quadric. For a rank-two fundamental matrix, exact mode solves a scalar secular equation only on the positive-definite Lagrange-multiplier interval. Every accepted output is checked for feasibility, KKT stationarity, and a positive-definite or positive-semidefinite Lagrangian Hessian. Numerically unresolved cases fail closed; an optional wrapper invokes the Hartley–Sturm all-candidate reference implementation on those failures.

## Repository Structure

```
├── lott_triangulate.h            # Core solver (header-only C++)
├── svd2x2_lott.h                 # 2x2 SVD for the joint rotation
├── lott_triangulate_certified.h  # Certified wrapper with Hartley–Sturm fallback
├── TriangulatorLott.m            # MATLAB reference implementation
├── Triangulation_OrthogonalDistance.ipynb  # Historical exploratory notebook
├── CMakeLists.txt                # Build system (requires Eigen3)
├── benchmarks/                   # Evaluation harnesses and baselines
│   ├── bench_speed.cpp           # Runtime benchmarks
│   ├── bench_correctness.cpp     # Objective-gap vs Hartley–Sturm
│   ├── bench_scaling.cpp         # O(N) scaling verification
│   ├── bench_approximation.cpp   # Approximation ladder (H1–H4)
│   ├── baseline_acceptance.cpp   # Baseline convention validation
│   ├── theorem_regression.cpp    # KKT/PSD and degeneracy regression suite
│   ├── triangulate_hs.h          # Hartley–Sturm baseline
│   ├── triangulate_kanatani.h    # Kanatani baseline
│   ├── triangulate_lindstrom.h   # Lindstrom baseline
│   ├── so3_utils.h               # Rotation utilities for data generation
│   ├── Polynomial.h              # Polynomial root-finding (for HS)
│   ├── Polynomial.cpp
│   └── rpoly.h                   # Jenkins–Traub root finder
└── scripts/                      # Reproducible benchmark pipeline
    ├── run_benchmarks.sh          # Run all benchmarks
    ├── run_baseline_acceptance.sh # Run baseline validation
    ├── build_figures.py           # Generate paper figures/tables
    ├── build_figures.sh           # Figure generation wrapper
    └── capture_env.sh             # Record build environment
```

## Quick Start

### Requirements
- C++20 compiler (clang or gcc)
- [Eigen3](https://eigen.tuxfamily.org/) (header-only linear algebra library)
- CMake >= 3.16

### Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Run Benchmarks
```bash
# Synthetic benchmark and regression suite
bash scripts/run_benchmarks.sh

# Baseline convention validation only
bash scripts/run_baseline_acceptance.sh
```

Each run uses a fresh timestamped build, executes the theorem and baseline
gates before timing, and writes raw outputs to `results/raw/`. A tracked-ready
manifest under `evidence/runs/` records the commit/worktree state, toolchain,
commands, and SHA-256 hashes for the run artifacts.

### Use in Your Code

The core solver is a single header file with one dependency (Eigen3). The input
fundamental matrix must have rank two; project an estimated matrix to rank two
before calling the solver.

```cpp
#include <Eigen/Dense>
#include "svd2x2_lott.h"
#include "lott_triangulate.h"

// F: 3x3 fundamental matrix
// A: 4xN matrix of [u0, v0, u1, v1] point pairs
Eigen::Matrix<double, 4, Eigen::Dynamic> X;
lott_triangulate(A, F, X);
// X now contains corrected point pairs satisfying x1'*F*x0 = 0
```

Exact mode certifies its returned rounded-double correction. To inspect the
per-point path, request the optional status and uniqueness-sentinel vectors:

```cpp
LottSolverDiagnostics diagnostics;
Eigen::VectorXi certified_solution_kind;
Eigen::VectorXi status;
lott_triangulate(A, F, X, &diagnostics, true, 0,
                 &certified_solution_kind, &status);
// certified_solution_kind: 1 = unique, 2 = certified nonunique PSD case,
//                           -1 = numerical failure (X is NaN for that point)
```

For an operational Hartley–Sturm fallback on numerical certificate failures,
include the wrapper. It also uses the baseline headers under `benchmarks/`:

```cpp
#include "lott_triangulate_certified.h"

Eigen::Matrix<double, 4, Eigen::Dynamic> X;
lott_triangulate_certified(A, F, X);
```

The core returns a deterministic certified member of a nonunique PSD-boundary
solution set. The wrapper accepts that member and falls back only when the core
reports failure; it does not turn the reference implementation itself into a
formal post-hoc certificate.

## Key Properties

- **O(1) per correspondence** — constant work, batch-vectorizable
- **Multiplier-safe root isolation** — safeguarded Newton/bisection stays in the common positive-definite interval
- **Global optimality certificate** — feasibility plus KKT plus a PD Hessian proves a regular solution is the unique global minimizer
- **Explicit PSD endpoint policy** — nullspace classification returns a certified deterministic optimum and records nonuniqueness
- **Fail-closed exact mode** — rejected candidates are returned as NaNs with a negative status; the optional wrapper supplies an all-candidate reference fallback
- **Tunable approximation ladder** — Householder H1–H4 one-step approximations for ultra-fast modes

The explanatory Sage notebook and exact verifier used for the TPAMI revision
are maintained in the submission supplement. The notebook in this public
baseline checkout is exploratory and should not be treated as the verification
artifact until the supplement is synchronized after its clean Sage run.

## License

MIT License — see [LICENSE](LICENSE).
