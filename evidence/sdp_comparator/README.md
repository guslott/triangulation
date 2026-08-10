# Lifted-SDP global-comparator evidence

This directory records the `BASE-2` numerical cross-check added for the TPAMI
major revision. It uses a methodologically distinct Shor semidefinite
relaxation of the two-view correction QCQP; it does not call the paper's
sextic or scalar multiplier solver.

## Formulation and claim boundary

For observed joint-image coordinate `z0` and correction `p`, the epipolar
constraint is translated to

```text
p^T A p + 2 g^T p + c = 0.
```

The comparator lifts `[p;1][p;1]^T` to a symmetric 5-by-5 matrix `Y` and
solves

```text
minimize    sum(Y[0:4,0:4] diagonal)
subject to  <G,Y> = 0,  Y[4,4] = 1,  Y positive semidefinite.
```

An exactly feasible rank-one optimum of this relaxation recovers a global
minimizer of the original QCQP. CVXOPT is a floating-point solver, so the
experiment is reported more narrowly as a deterministic numerical cross-check:
all returned lifts must be numerically rank one, their extracted points must be
feasible, the SDP optimality residuals must pass fixed thresholds, and their
objectives must agree with the independently certified Lott output. This is not
an exact-arithmetic SDP certificate.

## Population and acceptance gates

The input is the complete 27,080-row Oxford Dinosaur point-level result from
`evidence/real_data`. The canonical runner evaluates every row from every one
of the 364 nonempty view pairs; there is no subsampling or outcome-conditioned
selection.

The Python comparator reconstructs each fundamental matrix independently from
the released cameras rather than consuming the C++ matrix. In the audit, its
maximum up-to-sign difference from the C++ Gram-inverse construction was
`2.97e-15`. The upstream point CSV records the certified Lott objective and
residual but not its corrected coordinates, so this is an objective/relaxation
comparison rather than a bit-identical two-implementation replay.

The runner fails unless all 27,080 rows satisfy all of the following:

- the input Lott result is certified, finite, and feasible at normalized
  residual `1e-12`;
- CVXOPT reports `optimal`;
- the lifted tail spectral ratio and PSD-violation ratio are at most `1e-7`;
- the extracted lifted-constraint residual is at most `1e-7`;
- primal and dual infeasibilities are at most `1e-8`;
- the scaled duality gap is at most `1e-8`; and
- the relaxation and extracted objectives agree with the Lott cost within the
  scaled `1e-8` threshold.

## Result

All 27,080 rows pass. The strongest observed diagnostics are:

| Quantity | Maximum |
|---|---:|
| Lott vs. extracted SDP objective, scaled gap | `3.05528e-10` |
| Lott vs. SDP relaxation objective, scaled gap | `4.72996e-10` |
| Lifted tail spectral ratio | `1.37171e-10` |
| PSD-violation ratio | `9.02157e-11` |
| Extracted coefficient-normalized lifted-constraint residual | `4.66849e-14` |
| Primal / dual infeasibility | `9.72152e-10` / `1.81306e-10` |
| Scaled duality gap | `9.80976e-10` |

This comparator is used for global-objective validation, not a runtime claim.
The implementation is Python/CVXOPT, while the timed methods are C++/Eigen, so
cross-framework wall-clock numbers would not isolate algorithmic cost.

## Reproduction

Install the pinned optional dependencies into a Python 3.12 environment, then
run from the repository root:

```bash
python3.12 -m pip install -r scripts/requirements-sdp.txt
TPAMI_SDP_PYTHON=python3.12 ./scripts/run_sdp_global_comparator.sh
```

The runner first regenerates the checksum-pinned Oxford point-level input, then
runs the SDP population and records its environment. To prohibit downloads and
require the cached official files:

```bash
OXFORD_DINO_OFFLINE=1 \
TPAMI_SDP_PYTHON=/path/to/python3.12 \
./scripts/run_sdp_global_comparator.sh
```

The archived run used Python 3.12.13, NumPy 2.5.2, and CVXOPT 1.3.3 on macOS
arm64. Independent complete runs reproduced the same aggregate metrics; the
canonical runner records exact hashes for the final population, pair, summary,
and environment outputs.

## Archived hashes

| Artifact | SHA-256 |
|---|---|
| `benchmarks/sdp_global_comparator.py` | `7e7175718dd4f0398bc8b947625b2e43077d9398da9173340be9d40171f44d45` |
| `scripts/run_sdp_global_comparator.sh` | `95979b2ecbdc34fd5dcbd07874ae77871e43a098fe9cfd97800cadd648e091bc` |
| `scripts/requirements-sdp.txt` | `2c3b21ade7fb6625f226dacee67f4a5f3e2a203e8aca82fa86b752b7f2399c4b` |
| full `sdp_points.csv` (11,017,722 bytes, 27,081 rows; locally retained) | `b4285c543807f0e6039561021c6e246102d25dce2aa98c94bcaa4f7f43879590` |
| `2026-08-10_sdp_pairs.csv` | `f1f45c61120ff95380c369c7267ae4a9926b59c5e347f52a6524cc9663e32f36` |
| `2026-08-10_sdp_summary.txt` | `87271c3fe9cf708c1c4548df6ff5f6915d82ca14fc7ff17fd2709420107b010d` |
| `2026-08-10_sdp_environment.txt` | `e86053eca9da4b07a324cb86d10c9f55af64511cfcf6bc734b036a0e2699a898` |
