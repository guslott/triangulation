# Fast Projectively Invariant Triangulation via Orthogonal Distance to the Epipolar Quadric

**Author:** Gus Lott (guslott@gmail.com)

A bounded-cost method for two-view triangulation based on orthogonal distance to the epipolar quadric. The method avoids global polynomial solving by combining explicit root bounds with safeguarded Newton refinement from the origin branch, achieving Hartley–Sturm objective accuracy at substantially lower runtime.

## Repository Structure

```
├── lott_triangulate.h            # Core solver (header-only C++)
├── svd2x2_lott.h                 # 2x2 SVD for the joint rotation
├── lott_triangulate_certified.h  # Certified wrapper with Hartley–Sturm fallback
├── TriangulatorLott.m            # MATLAB reference implementation
├── Triangulation_OrthogonalDistance.ipynb  # Derivation notebook (Jupyter/SymPy)
├── CMakeLists.txt                # Build system (requires Eigen3)
├── benchmarks/                   # Evaluation harnesses and baselines
│   ├── bench_speed.cpp           # Runtime benchmarks
│   ├── bench_correctness.cpp     # Objective-gap vs Hartley–Sturm
│   ├── bench_scaling.cpp         # O(N) scaling verification
│   ├── bench_approximation.cpp   # Approximation ladder (H1–H4)
│   ├── baseline_acceptance.cpp   # Baseline convention validation
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
```

### Run Benchmarks
```bash
# Full benchmark suite (speed, correctness, scaling, approximation)
bash scripts/run_benchmarks.sh

# Baseline convention validation only
bash scripts/run_baseline_acceptance.sh
```

Results are written to `results/raw/` (timestamped) and `results/processed/` (tables, figures).

### Use in Your Code

The solver is a single header file with one dependency (Eigen3):

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

For certified mode (exact Hartley–Sturm fallback on non-unique branches):
```cpp
#include "lott_triangulate_certified.h"

Eigen::Matrix<double, 4, Eigen::Dynamic> X;
lott_triangulate_certified(A, F, X);
```

## Key Properties

- **O(1) per correspondence** — constant work, batch-vectorizable
- **Bounded-interval root isolation** — no global polynomial root finding
- **Branch-scoped optimality** — when the Sturm certificate reports one interval root, the solution is unique on the bounded origin branch
- **Certified-fast mode** — optional Hartley–Sturm fallback for unconditional exactness
- **Tunable approximation ladder** — Householder H1–H4 one-step approximations for ultra-fast modes

## License

MIT License — see [LICENSE](LICENSE).
