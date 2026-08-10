# Conditioning sweep evidence

Run the deterministic sweep from the repository root:

```sh
./scripts/run_conditioning_sweep.sh
```

The runner constructs canonical rank-two coefficient families with known
regular multipliers and corrections, then records point-level and bin-level
metrics.  It covers `b/a` down to zero, `b/a` approaching one, the normalized
quadratic coefficient `a` approaching the affine path, the Hessian endpoint
margin `1-mu` approaching zero, the exact PSD boundary, and common positive
coefficient scalings from `1e-200` through `1e200`.

For the positive-`a` affine-limit family, the target `mu` and unscaled linear
coefficients are fixed, so `g` grows like `1/a`; after common normalization,
both the quadratic and linear coefficients approach zero. The exact `a=0`
rows are separately constructed rank-two affine projections. They test the
solver's affine route but are not the finite pointwise limit of the positive-
`a` corrections.

The tracked outputs are:

- `2026-08-10_conditioning_cases.csv`: all point-level diagnostics;
- `2026-08-10_conditioning_bins.csv`: certificate/fallback/pass rates and
  extrema for every parameter bin;
- `2026-08-10_conditioning_summary.md`: compact results and interpretation
  limits.

The sweep treats any negative exact-solver status as requiring the public
Hartley--Sturm fallback. It deliberately does not invoke that fallback, so the
reported fallback rate measures certificate/solver routing rather than the
fallback implementation's accuracy or cost. This synthetic coefficient-space
evidence also does not replace the requested real-image evaluation.
