# Deterministic conditioning and degeneracy sweep

The runner evaluates analytically constructed canonical rank-two families. For every regular case, the target multiplier and correction are known independently of the numerical solve. A negative point status is counted as requiring the public Hartley--Sturm fallback; the sweep does not time or invoke that fallback.

## Overall result

- Cases: 188 across 47 parameter bins
- Certified: 188/188
- Fallback required: 0/188
- All assertions passed: 188/188
- Maximum normalized feasibility residual: `9.89255192643953373e-15`
- Maximum normalized KKT residual: `1.65388209683485278e-16`
- Maximum observed residual-condition / theoretical-bound ratio: `1.22472682349665893e-01`
- Maximum safeguarded iterations: 85
- Maximum bisection steps: 41

## Bin-level results

| Suite | Parameter | Bins | Cases | Certified | Fallback | Pass | Max iter. | Max bisect. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a_to_affine | `a_raw` | 8 | 32 | 32 | 0 | 32 | 8 | 0 |
| b_over_a | `b_over_a` | 11 | 44 | 44 | 0 | 44 | 8 | 0 |
| common_coefficient_scale | `scale` | 11 | 44 | 44 | 0 | 44 | 8 | 0 |
| endpoint_margin | `one_minus_mu` | 9 | 36 | 36 | 0 | 36 | 85 | 41 |
| near_equal_singular_values | `one_minus_b_over_a` | 8 | 32 | 32 | 0 | 32 | 8 | 0 |

## Interpretation limits

- These are canonical coefficient-space stress fixtures, not a real-image experiment.
- `residual_root_condition` is the scale-normalized `1/|phi'|`; `residual_root_bound` is `2/||q||^2`. They describe scalar residual-to-multiplier sensitivity, not end-to-end image conditioning.
- `hessian_condition` and `reconstruction_condition` expose the separate endpoint sensitivity. The exact `one_minus_mu=0` rows are PSD-boundary/nonunique and therefore do not receive a regular root condition number.
- The common-scale suite checks invariance after the solver's positive coefficient normalization; it is not a substitute for camera-coordinate scale tests.
- In the positive-`a` affine-limit family, the target `mu` and the unscaled linear coefficients are fixed, so `g` grows like `1/a`; after common normalization both the quadratic and linear terms approach zero. The exact `a=0` rows are separately constructed rank-two affine projections, not the finite pointwise limit of those positive-`a` corrections.
