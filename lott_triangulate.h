/*
        Fast Optimal Triangulation

        Given a pair of 2D image correspondences and a fundamental matrix
        describing the projective relationship of the views, the triangulation
        algorithm finds the nearest image points which perfectly satisfies
        the projective relationship so that back projected rays perfectly
   intersect in space.

        MIT License

        Copyright (c) 2021 Dr. Gus K. Lott, guslott@gmail.com

        Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "svd2x2_lott.h"

#define MAX_ROOT_STEPS 24
#define MAX_BRACKET_EXPANSIONS 24
#define MAX_BISECTION_FINISH_STEPS 128
#define CONVERGENCE_THRESHOLD 1e-15
constexpr double C_NEAR_ZERO_RATIO_TOL = 1e-3;

struct LottRootDiagnostics {
  bool used_sign_bracket = false;
  bool converged = false;
  // In certified mode these endpoints are Lagrange multipliers, not chart
  // coordinates.  Approximation modes do not claim a bracket.
  bool bracket_is_multiplier = false;
  int iterations = 0;
  int bisection_steps = 0;
  int guarded_halfsteps = 0;
  int nonfinite_eval_steps = 0;
  double bracket_left = std::numeric_limits<double>::quiet_NaN();
  double bracket_right = std::numeric_limits<double>::quiet_NaN();
  double multiplier = std::numeric_limits<double>::quiet_NaN();
  double minimum_hessian_eigenvalue =
      std::numeric_limits<double>::quiet_NaN();
};

// Per-correspondence outcome.  Positive values identify the path that returned
// a certified optimum; LOTT_STATUS_UNCERTIFIED_APPROXIMATE is reserved for the
// timing-oriented one-step Householder modes; negative values fail closed.
enum LottPointStatus : int {
  LOTT_STATUS_UNSET = 0,
  LOTT_STATUS_ALREADY_FEASIBLE = 1,
  LOTT_STATUS_AFFINE = 2,
  LOTT_STATUS_REGULAR_INTERIOR = 3,
  LOTT_STATUS_BOUNDARY_PSD_UNIQUE = 4,
  LOTT_STATUS_BOUNDARY_PSD_NONUNIQUE = 5,
  LOTT_STATUS_UNCERTIFIED_APPROXIMATE = 6,
  LOTT_STATUS_FAILED_INVALID_INPUT = -1,
  LOTT_STATUS_FAILED_BRACKET = -2,
  LOTT_STATUS_FAILED_CERTIFICATE = -3
};

inline bool lott_status_is_certified(const LottPointStatus status) {
  return status >= LOTT_STATUS_ALREADY_FEASIBLE &&
         status <= LOTT_STATUS_BOUNDARY_PSD_NONUNIQUE;
}

struct LottSolverDiagnostics {
  long long points_total = 0;
  long long roots_bracketed = 0;
  long long roots_unbracketed = 0;
  long long roots_converged = 0;
  // Historical name: in exact mode this counts any failed solver path,
  // including invalid input, bracketing failure, or certificate rejection.
  long long roots_max_steps = 0;
  long long total_iterations = 0;
  long long bisection_steps = 0;
  long long guarded_halfsteps = 0;
  long long nonfinite_eval_steps = 0;
  long long chart_non_x_points = 0;
  long long c_near_zero_points = 0;
  long long c_near_zero_non_x_points = 0;
  // In exact mode, eq1/gt1 count unique/nonunique theorem certificates, not
  // Sturm roots.  eq0 and the older Sturm/IVT/long-double fields below remain
  // for source compatibility and are not populated by the theorem solver.
  long long cert_points = 0;
  long long cert_rootcount_eq0 = 0;
  long long cert_rootcount_eq1 = 0;
  long long cert_rootcount_gt1 = 0;
  long long cert_failures = 0;
  long long cert_missing_bracket = 0;
  long long cert_nonfinite_endpoints = 0;
  long long cert_no_sign_change = 0;
  long long cert_endpoint_root_left = 0;
  long long cert_endpoint_root_right = 0;
  long long cert_sturm_invalid = 0;
  long long cert_ivt_conflict = 0;
  long long cert_longdouble_attempts = 0;
  long long cert_longdouble_rescues = 0;
  long long cert_longdouble_failures = 0;
  long long already_feasible_points = 0;
  long long affine_points = 0;
  long long regular_interior_points = 0;
  long long boundary_psd_unique_points = 0;
  long long boundary_psd_nonunique_points = 0;
  long long uncertified_approximate_points = 0;
  long long failed_invalid_input_points = 0;
  long long failed_bracket_points = 0;
  long long failed_certificate_points = 0;
  long long cert_feasibility_failures = 0;
  long long cert_kkt_failures = 0;
  long long cert_psd_failures = 0;
  long long chart_points[4] = {0, 0, 0, 0};
};

template <int PORDER> double polyval(const double *p, const double x) {
  // Evaluate polynomial using Horner's recursion
  double px = p[0];
  for (int i = 1; i <= PORDER; ++i) {
    px = px * x + p[i];
    // Perhaps Use FMA for better precision and speed where available
    // px = std::fma(px, x, p[i]);
  }
  return px;
}

template <int PORDER> void poly_derivative(const double *p, double *dp) {
  for (int i = 0; i < PORDER; i++) {
    dp[i] = p[i] * (PORDER - i);
  }
}

template <int HORDER> double householder_step_from_origin(const double *p) {

  // Householder Methods for polynomial root finding "from the origin"
  // Only need the first HORDER+1 coefficients of the polynomial
  // Converges to the root at an HORDER+1 rate.  Does not iterate
  const double &k0 = p[6];
  const double &k1 = p[5];
  const double &k2 = p[4];
  const double &k3 = p[3];
  const double &k4 = p[2];
  const double &k5 = p[1];
  const double &k6 = p[0];

  if (HORDER == 1) {
    // Newton-Raphson's Method
    const double num = -k0;
    const double den = k1;
    return (num / den);
  }
  if (HORDER == 2) {
    // Halley's method
    const double num = -k0 * k1;
    const double den = (k1 * k1 - k0 * k2);
    return (num / den);
  }
  // Higher order methods
  if (HORDER == 3) {
    const double num = -(k0 * k1 * k1 - k0 * k0 * k2);
    const double den = (k1 * k1 * k1 - 2 * k0 * k1 * k2 + k0 * k0 * k3);
    return (num / den);
  }
  if (HORDER == 4) {
    const double num =
        -(k0 * k1 * k1 * k1 - 2 * k0 * k0 * k1 * k2 + k0 * k0 * k0 * k3);
    const double den =
        (k1 * k1 * k1 * k1 - 3 * k0 * k1 * k1 * k2 + k0 * k0 * k2 * k2 +
         2 * k0 * k0 * k1 * k3 - k0 * k0 * k0 * k4);
    return (num / den);
  }
  if (HORDER == 5) {
    const double num = -k0 * k1 * k1 * k1 * k1 + 3 * k0 * k0 * k1 * k1 * k2 -
                       k0 * k0 * k0 * k2 * k2 - 2 * k0 * k0 * k0 * k1 * k3 +
                       k0 * k0 * k0 * k0 * k4;
    const double den = k1 * k1 * k1 * k1 * k1 - 4 * k0 * k1 * k1 * k1 * k2 +
                       3 * k0 * k0 * k1 * k2 * k2 - 2 * k0 * k0 * k0 * k1 * k4 +
                       k0 * k0 * k0 * k0 * k5 + 3 * k0 * k0 * k1 * k1 * k3 -
                       2 * k0 * k0 * k0 * k2 * k3;
    return (num / den);
  }

  return 0; // is an error, no update to the root
}

template <int HORDER>
double householder_step_from_origin(const double a, const double b,
                                    const double c, const double d,
                                    const double e, const double f,
                                    const double g) {
  // Simply compute the householder step of the specified order
  //  This is the approximate triangulation.  Only needs a few coefficients
  double p[7];
  lott_poly6_cx<HORDER + 1>(a, b, c, d, e, f, g, p);
  return c * householder_step_from_origin<HORDER>(
                 p); // c factor due to change of variables
}

template <int NCOEF>
void lott_poly6_cx(const double a, const double b, const double c,
                   const double d, const double e, const double f,
                   const double g,
                   double *p) // p assumed to have 7 doubles of space
{
  // Chart-normalized polynomial in the dimensionless variable t.
  // In the x-chart, the physical coordinate is x = c*t; other charts use
  // analogous normalized coordinates via chart-specific permutations.
  const double a2 = a * a;
  const double b2 = b * b;
  const double c2 = c * c;
  const double d2 = d * d;
  const double e2 = e * e;
  const double f2 = f * f;
  const double nu2 = c2 + d2 + e2 + f2;
  // const double rho = a*(c2-e2) + b*(d2-f2);
  // using difference-of-squares reduces the magnitude of intermediate values
  // before
  // multiplication, which can slightly improve numerical stability for float
  // (vs double) implementations.
  const double S1 = (c - e) * (c + e);
  const double S2 = (d - f) * (d + f);
  const double rho = a * S1 + b * S2;
  const double delta = (a - b) * (a + b);

  // TODO: Optimize this computation by precomputing common sub-expressions
  //  (13*a2-2*b2), 8*(a2-b2), 4*a*(3*a2+b2), (a2-b2), (4*a2*a2 + 3*a2*b2 +
  //  b2*b2), (4*(5*a2-b2)*(a2-b2)), (4*a*(a2-b2)*(a2-b2)) Will be different for
  //  different polynomial variables (x, y, z, w)

  if (NCOEF >= 1)
    p[6] = g; // x^0 term
  if (NCOEF >= 2)
    p[5] = (6 * a * g + 2 * nu2);
  if (NCOEF >= 3)
    p[4] = (3 * rho + g * (13 * a2 - 2 * b2) + 10 * a * nu2);
  if (NCOEF >= 4)
    p[3] =
        (8 * delta * (2 * c2 - e2) + g * 4 * a * (3 * a2 + b2) + 16 * a2 * nu2);
  if (NCOEF >= 5)
    p[2] = (delta * (a * (29 * c2 - 5 * e2) + b * S2) +
            g * (4 * a2 * a2 + 3 * a2 * b2 + b2 * b2) + 8 * a * a2 * nu2);
  if (NCOEF >= 6)
    p[1] = c2 * (4 * (5 * a2 - b2) * delta);
  if (NCOEF >= 7)
    p[0] = c2 * (4 * a * delta * delta); // x^6 term
}

/*
        returns root value
        loops will contain the actual number of iterations used
*/
double full_root_iterative(const double p[7], int &loops,
                           LottRootDiagnostics *diag = nullptr) {
  LottRootDiagnostics local_diag;
  if (diag == nullptr) {
    diag = &local_diag;
  }
  *diag = LottRootDiagnostics{};
  const auto bracket_width_converged = [](const double l,
                                          const double r) -> bool {
    const double scale = std::max({1.0, std::abs(l), std::abs(r)});
    const double abs_tol = CONVERGENCE_THRESHOLD;
    const double rel_tol = 64.0 * std::numeric_limits<double>::epsilon();
    return std::abs(r - l) <= (abs_tol + rel_tol * scale);
  };

  // In normalized variable t, p(0)=g and preconditioning enforces g>=0.
  // If p(0) is already close to zero, the root is at the origin.
  const double xr0 = 0.0;
  const double fr0 = p[6];
  if (std::abs(fr0) < CONVERGENCE_THRESHOLD) {
    loops = 0;
    diag->converged = true;
    return xr0;
  }

  // Initial guess from origin-based Householder step.
  double x = householder_step_from_origin<4>(p);
  if (!std::isfinite(x) || x >= 0.0) {
    x = householder_step_from_origin<1>(p);
  }
  if (!std::isfinite(x) || x >= 0.0) {
    x = -1.0;
  }

  // Build a sign-changing bracket [xl, xr], with xr fixed at 0.
  double xl = x;
  double xr = xr0;
  double fl = polyval<6>(p, xl);
  double fr = fr0;
  if (!std::isfinite(fl)) {
    xl = -1.0;
    fl = polyval<6>(p, xl);
  }

  for (int e = 0;
       (fl > 0.0 || !std::isfinite(fl)) && e < MAX_BRACKET_EXPANSIONS; ++e) {
    xl *= 2.0;
    fl = polyval<6>(p, xl);
  }

  // If a strict sign-changing bracket cannot be established, use guarded Newton
  // updates with fallback to half-step toward the origin.
  if (fl > 0.0 || !std::isfinite(fl)) {
    diag->used_sign_bracket = false;
    if (x >= 0.0 || !std::isfinite(x)) {
      x = -1.0;
    }
    for (loops = 0; loops < MAX_ROOT_STEPS; ++loops) {
      double val = p[0];
      double der = 0.0;
      for (int i = 1; i < 7; ++i) {
        der = der * x + val;
        val = val * x + p[i];
      }
      if (std::abs(val) < CONVERGENCE_THRESHOLD) {
        diag->converged = true;
        break;
      }
      double x_new = x;
      if (std::isfinite(der) && std::abs(der) > 1e-18) {
        x_new = x - val / der;
      } else {
        x_new *= 0.5;
        ++diag->guarded_halfsteps;
      }
      if (!std::isfinite(x_new) || x_new >= 0.0) {
        x_new = 0.5 * x;
        ++diag->guarded_halfsteps;
      }
      x = x_new;
    }
    diag->iterations = loops;
    if (loops < MAX_ROOT_STEPS) {
      diag->converged = true;
    }
    return x;
  }

  diag->used_sign_bracket = true;
  diag->bracket_left = xl;
  diag->bracket_right = xr;
  x = std::clamp(x, xl, xr);
  int extra_steps = 0;
  for (loops = 0; loops < MAX_ROOT_STEPS; ++loops) {
    double val = p[0];
    double der = 0.0;
    for (int i = 1; i < 7; ++i) {
      der = der * x + val;
      val = val * x + p[i];
    }
    if (std::abs(val) < CONVERGENCE_THRESHOLD) {
      diag->converged = true;
      break;
    }

    double x_new;
    if (!std::isfinite(der) || std::abs(der) < 1e-18) {
      x_new = 0.5 * (xl + xr); // bisection fallback
      ++diag->bisection_steps;
    } else {
      x_new = x - val / der; // Newton proposal
      if (!(x_new > xl && x_new < xr) || !std::isfinite(x_new)) {
        x_new = 0.5 * (xl + xr); // keep bracket valid
        ++diag->bisection_steps;
      }
    }

    double f_new = polyval<6>(p, x_new);
    if (!std::isfinite(f_new)) {
      ++diag->nonfinite_eval_steps;
      x_new = 0.5 * (xl + xr);
      f_new = polyval<6>(p, x_new);
      ++diag->bisection_steps;
    }

    // Maintain sign-changing bracket.
    if ((fl <= 0.0 && f_new <= 0.0) || (fl >= 0.0 && f_new >= 0.0)) {
      xl = x_new;
      fl = f_new;
    } else {
      xr = x_new;
      fr = f_new;
    }

    x = x_new;
    if (bracket_width_converged(xl, xr) ||
        std::abs(fr - fl) < CONVERGENCE_THRESHOLD) {
      diag->converged = true;
      break;
    }
  }

  // Rare fallback: if Newton+guarded updates hit the iteration cap while the
  // sign-changing bracket is still valid, finish with pure bisection.
  if (!diag->converged) {
    for (extra_steps = 0; extra_steps < MAX_BISECTION_FINISH_STEPS;
         ++extra_steps) {
      x = 0.5 * (xl + xr);
      const double fx = polyval<6>(p, x);
      ++diag->bisection_steps;
      if (!std::isfinite(fx)) {
        ++diag->nonfinite_eval_steps;
        continue;
      }
      if ((fl <= 0.0 && fx <= 0.0) || (fl >= 0.0 && fx >= 0.0)) {
        xl = x;
        fl = fx;
      } else {
        xr = x;
        fr = fx;
      }
      if (std::abs(fx) < CONVERGENCE_THRESHOLD ||
          bracket_width_converged(xl, xr)) {
        diag->converged = true;
        ++extra_steps;
        break;
      }
    }
  }

  diag->iterations = loops + extra_steps;
  loops = diag->iterations;

  return x;
}

inline void trim_leading_small(std::vector<double> &poly, const double tol) {
  while (poly.size() > 1 && std::abs(poly.front()) <= tol) {
    poly.erase(poly.begin());
  }
  if (poly.empty()) {
    poly.push_back(0.0);
  }
}

inline double polyval_vec(const std::vector<double> &poly, const double x) {
  double px = poly.empty() ? 0.0 : poly.front();
  for (size_t i = 1; i < poly.size(); ++i) {
    px = px * x + poly[i];
  }
  return px;
}

inline std::vector<double> poly_derivative_vec(const std::vector<double> &poly) {
  if (poly.size() <= 1) {
    return {0.0};
  }
  std::vector<double> d(poly.size() - 1, 0.0);
  const int deg = static_cast<int>(poly.size()) - 1;
  for (size_t i = 0; i + 1 < poly.size(); ++i) {
    d[i] = poly[i] * static_cast<double>(deg - static_cast<int>(i));
  }
  return d;
}

inline std::vector<double> poly_remainder_vec(std::vector<double> numer,
                                              std::vector<double> denom,
                                              const double tol) {
  trim_leading_small(numer, tol);
  trim_leading_small(denom, tol);
  if (denom.size() == 1 && std::abs(denom[0]) <= tol) {
    return {0.0};
  }

  while (numer.size() >= denom.size()) {
    if (std::abs(numer.front()) <= tol) {
      numer.erase(numer.begin());
      if (numer.empty()) {
        numer.push_back(0.0);
      }
      continue;
    }
    const double scale = numer.front() / denom.front();
    for (size_t j = 0; j < denom.size(); ++j) {
      numer[j] -= scale * denom[j];
    }
    trim_leading_small(numer, tol);
    if (numer.size() == 1 && std::abs(numer[0]) <= tol) {
      break;
    }
  }
  trim_leading_small(numer, tol);
  return numer;
}

inline void trim_leading_small_ld(std::vector<long double> &poly,
                                  const long double tol) {
  while (poly.size() > 1 && std::abs(poly.front()) <= tol) {
    poly.erase(poly.begin());
  }
  if (poly.empty()) {
    poly.push_back(0.0L);
  }
}

inline long double polyval_vec_ld(const std::vector<long double> &poly,
                                  const long double x) {
  long double px = poly.empty() ? 0.0L : poly.front();
  for (size_t i = 1; i < poly.size(); ++i) {
    px = px * x + poly[i];
  }
  return px;
}

inline std::vector<long double>
poly_derivative_vec_ld(const std::vector<long double> &poly) {
  if (poly.size() <= 1) {
    return {0.0L};
  }
  std::vector<long double> d(poly.size() - 1, 0.0L);
  const int deg = static_cast<int>(poly.size()) - 1;
  for (size_t i = 0; i + 1 < poly.size(); ++i) {
    d[i] = poly[i] * static_cast<long double>(deg - static_cast<int>(i));
  }
  return d;
}

inline std::vector<long double>
poly_remainder_vec_ld(std::vector<long double> numer,
                      std::vector<long double> denom, const long double tol) {
  trim_leading_small_ld(numer, tol);
  trim_leading_small_ld(denom, tol);
  if (denom.size() == 1 && std::abs(denom[0]) <= tol) {
    return {0.0L};
  }

  while (numer.size() >= denom.size()) {
    if (std::abs(numer.front()) <= tol) {
      numer.erase(numer.begin());
      if (numer.empty()) {
        numer.push_back(0.0L);
      }
      continue;
    }
    const long double scale = numer.front() / denom.front();
    for (size_t j = 0; j < denom.size(); ++j) {
      numer[j] -= scale * denom[j];
    }
    trim_leading_small_ld(numer, tol);
    if (numer.size() == 1 && std::abs(numer[0]) <= tol) {
      break;
    }
  }
  trim_leading_small_ld(numer, tol);
  return numer;
}

inline int sturm_root_count_open_interval_poly6_long_double(const double p[7],
                                                             const double left,
                                                             const double right) {
  if (!(right > left) || !std::isfinite(left) || !std::isfinite(right)) {
    return -1;
  }
  constexpr long double kPolyTol = 1e-18L;
  constexpr long double kEvalTol = 1e-15L;

  std::vector<long double> s0(7, 0.0L);
  for (int i = 0; i < 7; ++i) {
    s0[i] = static_cast<long double>(p[i]);
  }
  trim_leading_small_ld(s0, kPolyTol);
  if (s0.size() <= 1) {
    return -1;
  }
  std::vector<long double> s1 = poly_derivative_vec_ld(s0);
  trim_leading_small_ld(s1, kPolyTol);
  if (s1.size() == 1 && std::abs(s1[0]) <= kPolyTol) {
    return -1;
  }

  std::vector<std::vector<long double>> sturm;
  sturm.reserve(8);
  sturm.push_back(s0);
  sturm.push_back(s1);

  for (int k = 0; k < 8; ++k) {
    std::vector<long double> rem =
        poly_remainder_vec_ld(sturm[sturm.size() - 2], sturm.back(), kPolyTol);
    bool all_small = true;
    for (long double &coef : rem) {
      coef = -coef;
      if (std::abs(coef) > kPolyTol) {
        all_small = false;
      }
    }
    trim_leading_small_ld(rem, kPolyTol);
    if (all_small) {
      break;
    }
    sturm.push_back(rem);
    if (rem.size() == 1) {
      break;
    }
  }
  if (sturm.size() < 2) {
    return -1;
  }

  const long double span =
      static_cast<long double>(right) - static_cast<long double>(left);
  const long double eps = 1e-12L * std::max(1.0L, span);
  long double x0 = static_cast<long double>(left) + eps;
  long double x1 = static_cast<long double>(right) - eps;
  if (!(x1 > x0)) {
    x0 = static_cast<long double>(left);
    x1 = static_cast<long double>(right);
  }

  const auto sign_variations = [&](const long double x) -> int {
    int prev_sign = 0;
    int variations = 0;
    for (const auto &poly : sturm) {
      const long double v = polyval_vec_ld(poly, x);
      int s = 0;
      if (v > kEvalTol) {
        s = 1;
      } else if (v < -kEvalTol) {
        s = -1;
      } else {
        continue;
      }
      if (prev_sign != 0 && s != prev_sign) {
        ++variations;
      }
      prev_sign = s;
    }
    return variations;
  };

  const int v0 = sign_variations(x0);
  const int v1 = sign_variations(x1);
  const int count = v0 - v1;
  if (count < 0) {
    return -1;
  }
  return count;
}

inline int sturm_root_count_open_interval_poly6(const double p[7],
                                                const double left,
                                                const double right) {
  if (!(right > left) || !std::isfinite(left) || !std::isfinite(right)) {
    return -1;
  }
  constexpr double kPolyTol = 1e-14;
  constexpr double kEvalTol = 1e-12;

  std::vector<double> s0(p, p + 7);
  trim_leading_small(s0, kPolyTol);
  if (s0.size() <= 1) {
    return -1;
  }
  std::vector<double> s1 = poly_derivative_vec(s0);
  trim_leading_small(s1, kPolyTol);
  if (s1.size() == 1 && std::abs(s1[0]) <= kPolyTol) {
    return -1;
  }

  std::vector<std::vector<double>> sturm;
  sturm.reserve(8);
  sturm.push_back(s0);
  sturm.push_back(s1);

  for (int k = 0; k < 8; ++k) {
    std::vector<double> rem =
        poly_remainder_vec(sturm[sturm.size() - 2], sturm.back(), kPolyTol);
    bool all_small = true;
    for (double &coef : rem) {
      coef = -coef;
      if (std::abs(coef) > kPolyTol) {
        all_small = false;
      }
    }
    trim_leading_small(rem, kPolyTol);
    if (all_small) {
      break;
    }
    sturm.push_back(rem);
    if (rem.size() == 1) {
      break;
    }
  }
  if (sturm.size() < 2) {
    return -1;
  }

  const double span = right - left;
  const double eps = 1e-10 * std::max(1.0, span);
  double x0 = left + eps;
  double x1 = right - eps;
  if (!(x1 > x0)) {
    x0 = left;
    x1 = right;
  }

  const auto sign_variations = [&](const double x) -> int {
    int prev_sign = 0;
    int variations = 0;
    for (const auto &poly : sturm) {
      const double v = polyval_vec(poly, x);
      int s = 0;
      if (v > kEvalTol) {
        s = 1;
      } else if (v < -kEvalTol) {
        s = -1;
      } else {
        continue;
      }
      if (prev_sign != 0 && s != prev_sign) {
        ++variations;
      }
      prev_sign = s;
    }
    return variations;
  };

  const int v0 = sign_variations(x0);
  const int v1 = sign_variations(x1);
  const int count = v0 - v1;
  if (count < 0) {
    return -1;
  }
  return count;
}

struct LottCertifiedPointResult {
  Eigen::Vector4d correction = Eigen::Vector4d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  LottPointStatus status = LOTT_STATUS_UNSET;
  // Convention retained by lott_triangulate_certified_fallback: one means a
  // unique certified optimum, two means a nonunique PSD-boundary optimum, and
  // minus one means fail closed and request the external all-candidate fallback.
  int certified_solution_count = -1;
  LottRootDiagnostics root;
  bool feasibility_ok = false;
  bool kkt_ok = false;
  bool hessian_ok = false;
};

namespace lott_certified_detail {

using LD4 = std::array<long double, 4>;

inline bool finite4(const LD4 &u) {
  return std::isfinite(u[0]) && std::isfinite(u[1]) &&
         std::isfinite(u[2]) && std::isfinite(u[3]);
}

inline long double constraint_value(const long double a, const long double b,
                                    const LD4 &q, const long double g,
                                    const LD4 &u) {
  return a * u[0] * u[0] + b * u[1] * u[1] - a * u[2] * u[2] -
         b * u[3] * u[3] + 2.0L * (q[0] * u[0] + q[1] * u[1] +
                                      q[2] * u[2] + q[3] * u[3]) +
         g;
}

inline long double constraint_scale(const long double a, const long double b,
                                    const LD4 &q, const long double g,
                                    const LD4 &u) {
  return std::max(
      1.0L, std::abs(g) + std::abs(a * u[0] * u[0]) +
                std::abs(b * u[1] * u[1]) +
                std::abs(a * u[2] * u[2]) +
                std::abs(b * u[3] * u[3]) +
                2.0L * (std::abs(q[0] * u[0]) + std::abs(q[1] * u[1]) +
                         std::abs(q[2] * u[2]) +
                         std::abs(q[3] * u[3])));
}

inline void set_double_correction(const LD4 &u,
                                  LottCertifiedPointResult &result) {
  for (int j = 0; j < 4; ++j) {
    result.correction(j) = static_cast<double>(u[static_cast<size_t>(j)]);
  }
}

inline LD4 rounded_correction(const LottCertifiedPointResult &result) {
  return {static_cast<long double>(result.correction(0)),
          static_cast<long double>(result.correction(1)),
          static_cast<long double>(result.correction(2)),
          static_cast<long double>(result.correction(3))};
}

// Verify the three numerical facts used by the global-optimality proof:
// feasibility, stationarity, and a positive-(semi)definite Lagrangian Hessian.
// All coefficients below have first been divided by one common positive scale,
// which leaves both the feasible set and correction unchanged.
inline bool certify(const long double a, const long double b, const LD4 &q,
                    const long double g, const LD4 &u,
                    const long double mu, const bool boundary_psd,
                    LottCertifiedPointResult &result) {
  constexpr long double kTol =
      8192.0L * static_cast<long double>(std::numeric_limits<double>::epsilon());
  if (!finite4(u)) {
    return false;
  }

  const long double h = constraint_value(a, b, q, g, u);
  const long double h_scale = constraint_scale(a, b, q, g, u);
  result.feasibility_ok =
      std::isfinite(h) && std::abs(h) <= kTol * h_scale;

  if (!(a > 0.0L)) {
    // Affine path: the mu argument carries lambda for the normalized
    // constraint (there is no finite lambda_b with which to normalize it).
    const long double two_lambda = 2.0L * mu;
    long double residual = 0.0L;
    long double residual_scale = 1.0L;
    for (int j = 0; j < 4; ++j) {
      const long double rhs = two_lambda * q[static_cast<size_t>(j)];
      residual = std::max(
          residual, std::abs(u[static_cast<size_t>(j)] + rhs));
      residual_scale = std::max(
          residual_scale,
          std::abs(u[static_cast<size_t>(j)]) + std::abs(rhs));
    }
    result.kkt_ok = residual <= kTol * residual_scale;
    result.hessian_ok = true;
    result.root.minimum_hessian_eigenvalue = 1.0;
    return result.feasibility_ok && result.kkt_ok;
  }

  const long double ratio = b / a;
  const LD4 hdiag = {1.0L + mu, 1.0L + ratio * mu, 1.0L - mu,
                     1.0L - ratio * mu};
  const long double two_lambda = mu / a;
  long double residual = 0.0L;
  long double residual_scale = 1.0L;
  long double min_h = hdiag[0];
  for (int j = 0; j < 4; ++j) {
    const long double rhs = two_lambda * q[static_cast<size_t>(j)];
    const long double lhs =
        hdiag[static_cast<size_t>(j)] * u[static_cast<size_t>(j)];
    residual = std::max(residual, std::abs(lhs + rhs));
    residual_scale =
        std::max(residual_scale, std::abs(lhs) + std::abs(rhs));
    min_h = std::min(min_h, hdiag[static_cast<size_t>(j)]);
  }
  result.kkt_ok = residual <= kTol * residual_scale;
  result.hessian_ok = boundary_psd ? (min_h >= -kTol) : (min_h > 0.0L);
  result.root.minimum_hessian_eigenvalue = static_cast<double>(min_h);
  return result.feasibility_ok && result.kkt_ok && result.hessian_ok;
}

struct PhiEvaluation {
  long double value = std::numeric_limits<long double>::quiet_NaN();
  long double derivative = std::numeric_limits<long double>::quiet_NaN();
  long double scale = 1.0L;
};

inline PhiEvaluation evaluate_phi(const long double a, const long double b,
                                  const LD4 &q, const long double g,
                                  const long double mu) {
  PhiEvaluation out;
  if (!(a > 0.0L) || !(mu >= 0.0L) || !(mu < 1.0L)) {
    return out;
  }
  const long double ratio = b / a;
  const LD4 signs = {1.0L, ratio, -1.0L, -ratio};
  long double reduction = 0.0L;
  long double reduction_abs = 0.0L;
  long double derivative_sum = 0.0L;
  for (int j = 0; j < 4; ++j) {
    const long double s = signs[static_cast<size_t>(j)];
    const long double den = 1.0L + s * mu;
    if (!(den > 0.0L)) {
      return out;
    }
    const long double q2 = q[static_cast<size_t>(j)] *
                           q[static_cast<size_t>(j)];
    const long double term =
        (q2 / a) * mu * (2.0L + s * mu) / (den * den);
    reduction += term;
    reduction_abs += std::abs(term);
    derivative_sum += q2 / (den * den * den);
  }
  out.value = g - reduction;
  out.derivative = -2.0L * derivative_sum / a;
  out.scale = std::max(1.0L, std::abs(g) + reduction_abs);
  return out;
}

inline LD4 reconstruct_regular(const long double a, const long double b,
                               const LD4 &q, const long double mu) {
  const long double ratio = b / a;
  const LD4 hdiag = {1.0L + mu, 1.0L + ratio * mu, 1.0L - mu,
                     1.0L - ratio * mu};
  const long double two_lambda = mu / a;
  LD4 u{};
  for (int j = 0; j < 4; ++j) {
    u[static_cast<size_t>(j)] =
        -two_lambda * q[static_cast<size_t>(j)] /
        hdiag[static_cast<size_t>(j)];
  }
  return u;
}

// Complementary evaluation for roots near the PSD endpoint.  Representing
// tau=1-mu directly avoids losing all relative accuracy in the singular
// denominator when mu rounds close to one.
inline PhiEvaluation evaluate_phi_tau(const long double a,
                                      const long double b, const LD4 &q,
                                      const long double g,
                                      const long double tau) {
  PhiEvaluation out;
  if (!(a > 0.0L) || !(tau > 0.0L) || !(tau <= 1.0L)) {
    return out;
  }
  const long double ratio = b / a;
  const long double mu = 1.0L - tau;
  const LD4 hdiag = {2.0L - tau, 1.0L + ratio - ratio * tau, tau,
                     1.0L - ratio + ratio * tau};
  const LD4 signs = {1.0L, ratio, -1.0L, -ratio};
  long double reduction = 0.0L;
  long double reduction_abs = 0.0L;
  long double derivative_sum = 0.0L;
  for (int j = 0; j < 4; ++j) {
    const long double den = hdiag[static_cast<size_t>(j)];
    if (!(den > 0.0L)) {
      return out;
    }
    const long double q2 = q[static_cast<size_t>(j)] *
                           q[static_cast<size_t>(j)];
    long double numerator =
        mu * (2.0L + signs[static_cast<size_t>(j)] * mu);
    if (j == 2) {
      // mu(2-mu)=1-tau^2, evaluated without subtracting tau from one.
      numerator = 1.0L - tau * tau;
    } else if (j == 3 && ratio == 1.0L) {
      numerator = 1.0L - tau * tau;
    }
    const long double term = (q2 / a) * numerator / (den * den);
    reduction += term;
    reduction_abs += std::abs(term);
    derivative_sum += q2 / (den * den * den);
  }
  out.value = g - reduction;
  // d phi / d tau = -d phi / d mu > 0.
  out.derivative = 2.0L * derivative_sum / a;
  out.scale = std::max(1.0L, std::abs(g) + reduction_abs);
  return out;
}

inline LD4 reconstruct_regular_tau(const long double a, const long double b,
                                   const LD4 &q, const long double tau) {
  const long double ratio = b / a;
  const long double mu = 1.0L - tau;
  const LD4 hdiag = {2.0L - tau, 1.0L + ratio - ratio * tau, tau,
                     1.0L - ratio + ratio * tau};
  const long double two_lambda = mu / a;
  LD4 u{};
  for (int j = 0; j < 4; ++j) {
    u[static_cast<size_t>(j)] =
        -two_lambda * q[static_cast<size_t>(j)] /
        hdiag[static_cast<size_t>(j)];
  }
  return u;
}

inline bool certify_tau(const long double a, const long double b, const LD4 &q,
                        const long double g, const LD4 &u,
                        const long double tau,
                        LottCertifiedPointResult &result) {
  constexpr long double kTol =
      8192.0L * static_cast<long double>(std::numeric_limits<double>::epsilon());
  if (!finite4(u) || !(tau > 0.0L) || !(tau < 1.0L)) {
    return false;
  }
  const long double h = constraint_value(a, b, q, g, u);
  const long double h_scale = constraint_scale(a, b, q, g, u);
  result.feasibility_ok =
      std::isfinite(h) && std::abs(h) <= kTol * h_scale;

  const long double ratio = b / a;
  const long double mu = 1.0L - tau;
  const LD4 hdiag = {2.0L - tau, 1.0L + ratio - ratio * tau, tau,
                     1.0L - ratio + ratio * tau};
  const long double two_lambda = mu / a;
  long double residual = 0.0L;
  long double residual_scale = 1.0L;
  long double min_h = hdiag[0];
  for (int j = 0; j < 4; ++j) {
    const long double rhs = two_lambda * q[static_cast<size_t>(j)];
    const long double lhs =
        hdiag[static_cast<size_t>(j)] * u[static_cast<size_t>(j)];
    residual = std::max(residual, std::abs(lhs + rhs));
    residual_scale =
        std::max(residual_scale, std::abs(lhs) + std::abs(rhs));
    min_h = std::min(min_h, hdiag[static_cast<size_t>(j)]);
  }
  result.kkt_ok = residual <= kTol * residual_scale;
  result.hessian_ok = min_h > 0.0L;
  result.root.minimum_hessian_eigenvalue = static_cast<double>(min_h);
  return result.feasibility_ok && result.kkt_ok && result.hessian_ok;
}

inline LottCertifiedPointResult solve_regular_tau(
    const long double a, const long double b, const LD4 &q,
    const long double g, const double a_input) {
  LottCertifiedPointResult result;
  long double left = 0.5L;
  PhiEvaluation left_eval = evaluate_phi_tau(a, b, q, g, left);
  int bracket_steps = 0;
  constexpr int kMaxBracketSteps = 256;
  while ((std::isnan(left_eval.value) || left_eval.value >= 0.0L) &&
         bracket_steps < kMaxBracketSteps) {
    const long double next = 0.5L * left;
    if (!(next > 0.0L) || !(next < left)) {
      break;
    }
    left = next;
    left_eval = evaluate_phi_tau(a, b, q, g, left);
    ++bracket_steps;
  }
  long double right = 1.0L;
  PhiEvaluation right_eval = evaluate_phi_tau(a, b, q, g, right);
  if (std::isnan(left_eval.value) || !(left_eval.value < 0.0L) ||
      std::isnan(right_eval.value) || !(right_eval.value > 0.0L)) {
    result.status = LOTT_STATUS_FAILED_BRACKET;
    return result;
  }

  long double x = 0.5L * (left + right);
  PhiEvaluation x_eval = evaluate_phi_tau(a, b, q, g, x);
  constexpr int kMaxSolveSteps = 192;
  constexpr long double kPhiTol =
      128.0L * static_cast<long double>(std::numeric_limits<double>::epsilon());
  int iterations = 0;
  int bisections = 0;
  for (; iterations < kMaxSolveSteps; ++iterations) {
    if (std::isnan(x_eval.value)) {
      ++result.root.nonfinite_eval_steps;
      x = 0.5L * (left + right);
      x_eval = evaluate_phi_tau(a, b, q, g, x);
      ++bisections;
      if (std::isnan(x_eval.value)) {
        break;
      }
    }
    if (std::abs(x_eval.value) <= kPhiTol * x_eval.scale) {
      break;
    }
    if (x_eval.value < 0.0L) {
      left = x;
      left_eval = x_eval;
    } else {
      right = x;
      right_eval = x_eval;
    }
    const long double midpoint = 0.5L * (left + right);
    if (!(midpoint > left) || !(midpoint < right)) {
      if (std::abs(left_eval.value) <= std::abs(right_eval.value)) {
        x = left;
        x_eval = left_eval;
      } else {
        x = right;
        x_eval = right_eval;
      }
      break;
    }
    long double proposal = std::numeric_limits<long double>::quiet_NaN();
    if (std::isfinite(x_eval.derivative) && x_eval.derivative > 0.0L) {
      proposal = x - x_eval.value / x_eval.derivative;
    }
    if (!(proposal > left) || !(proposal < right) ||
        !std::isfinite(proposal)) {
      proposal = midpoint;
      ++bisections;
    }
    x = proposal;
    x_eval = evaluate_phi_tau(a, b, q, g, x);
  }

  result.root.used_sign_bracket = true;
  result.root.bracket_is_multiplier = true;
  result.root.iterations = iterations + bracket_steps;
  result.root.bisection_steps = bisections;
  result.root.bracket_left = static_cast<double>(
      (1.0L - right) / (2.0L * static_cast<long double>(a_input)));
  result.root.bracket_right = static_cast<double>(
      (1.0L - left) / (2.0L * static_cast<long double>(a_input)));
  result.root.multiplier = static_cast<double>(
      (1.0L - x) / (2.0L * static_cast<long double>(a_input)));

  if (std::isnan(x_eval.value) || !(x > 0.0L) || !(x < 1.0L)) {
    result.status = LOTT_STATUS_FAILED_BRACKET;
    return result;
  }
  const LD4 u = reconstruct_regular_tau(a, b, q, x);
  set_double_correction(u, result);
  if (result.correction.allFinite() &&
      certify_tau(a, b, q, g, rounded_correction(result), x, result)) {
    result.root.converged = true;
    result.status = LOTT_STATUS_REGULAR_INTERIOR;
    result.certified_solution_count = 1;
    return result;
  }
  result.status = LOTT_STATUS_FAILED_CERTIFICATE;
  result.correction.setConstant(std::numeric_limits<double>::quiet_NaN());
  return result;
}

} // namespace lott_certified_detail

// The theorem-aligned exact point solver.  The input has already undergone the
// g>=0 image swap, and selected_chart is the largest-|q_i| chart (0..3).
inline LottCertifiedPointResult lott_solve_certified_point(
    const double a_input, const double b_input, const double c_input,
    const double d_input, const double e_input, const double f_input,
    const double g_input, const int selected_chart,
    const double initial_mu_input =
        std::numeric_limits<double>::quiet_NaN()) {
  using namespace lott_certified_detail;
  LottCertifiedPointResult result;

  const std::array<double, 7> raw = {a_input, b_input, c_input, d_input,
                                     e_input, f_input, g_input};
  for (const double v : raw) {
    if (!std::isfinite(v)) {
      result.status = LOTT_STATUS_FAILED_INVALID_INPUT;
      return result;
    }
  }
  if (!(a_input >= 0.0) || !(b_input >= 0.0) || b_input > a_input ||
      !(g_input >= 0.0) || selected_chart < 0 || selected_chart > 3) {
    result.status = LOTT_STATUS_FAILED_INVALID_INPUT;
    return result;
  }

  long double coefficient_scale = 0.0L;
  for (const double v : raw) {
    coefficient_scale =
        std::max(coefficient_scale, std::abs(static_cast<long double>(v)));
  }
  if (!(coefficient_scale > 0.0L)) {
    result.correction.setZero();
    result.status = LOTT_STATUS_ALREADY_FEASIBLE;
    result.certified_solution_count = 1;
    result.feasibility_ok = result.kkt_ok = result.hessian_ok = true;
    result.root.converged = true;
    result.root.multiplier = 0.0;
    result.root.minimum_hessian_eigenvalue = 1.0;
    return result;
  }

  const long double a = static_cast<long double>(a_input) / coefficient_scale;
  const long double b = static_cast<long double>(b_input) / coefficient_scale;
  const LD4 q = {static_cast<long double>(c_input) / coefficient_scale,
                 static_cast<long double>(d_input) / coefficient_scale,
                 static_cast<long double>(e_input) / coefficient_scale,
                 static_cast<long double>(f_input) / coefficient_scale};
  const long double g = static_cast<long double>(g_input) / coefficient_scale;
  constexpr long double kNearZero =
      256.0L * static_cast<long double>(std::numeric_limits<double>::epsilon());

  // g=0 has the zero correction as the unique minimum.  Treat a residual below
  // the scale-aware roundoff floor the same way and still run the certificate.
  if (std::abs(g) <= kNearZero) {
    const LD4 u = {0.0L, 0.0L, 0.0L, 0.0L};
    result.root.converged = true;
    result.root.multiplier = 0.0;
    if (certify(a, b, q, g, u, 0.0L, false, result)) {
      result.correction.setZero();
      result.status = LOTT_STATUS_ALREADY_FEASIBLE;
      result.certified_solution_count = 1;
      return result;
    }
    result.status = LOTT_STATUS_FAILED_CERTIFICATE;
    result.correction.setConstant(std::numeric_limits<double>::quiet_NaN());
    return result;
  }

  // With a=0 the quadric is the affine hyperplane 2 q^T u + g=0.
  if (!(a > 0.0L)) {
    long double q2 = 0.0L;
    for (const long double qj : q) {
      q2 += qj * qj;
    }
    if (!(q2 > 0.0L)) {
      const bool nonzero_q =
          q[0] != 0.0L || q[1] != 0.0L || q[2] != 0.0L || q[3] != 0.0L;
      // A literal zero normal makes the nonzero affine constraint infeasible.
      // A nonzero normal whose squared norm underflows instead means working
      // precision cannot represent/certify the projection, so fail closed as a
      // certificate failure and allow the external fallback policy to decide.
      result.status = nonzero_q ? LOTT_STATUS_FAILED_CERTIFICATE
                                : LOTT_STATUS_FAILED_INVALID_INPUT;
      return result;
    }
    const long double alpha = -g / (2.0L * q2);
    LD4 u{};
    for (int j = 0; j < 4; ++j) {
      u[static_cast<size_t>(j)] = alpha * q[static_cast<size_t>(j)];
    }
    const long double lambda_normalized = g / (4.0L * q2);
    result.root.converged = true;
    result.root.multiplier = static_cast<double>(
        lambda_normalized / coefficient_scale);
    set_double_correction(u, result);
    if (result.correction.allFinite() &&
        certify(a, b, q, g, rounded_correction(result), lambda_normalized,
                false, result)) {
      result.status = LOTT_STATUS_AFFINE;
      result.certified_solution_count = 1;
      return result;
    }
    result.status = LOTT_STATUS_FAILED_CERTIFICATE;
    result.correction.setConstant(std::numeric_limits<double>::quiet_NaN());
    return result;
  }

  // Classify the PSD endpoint mu=1 before attempting an interior root.  For
  // a>b the nullspace is span(e_z); for exactly equal singular values it is
  // span(e_z,e_w).  Near equality remains on the mathematically correct a>b
  // branch and is evaluated in long double rather than inventing a nullspace.
  const bool equal_singular = (a_input == b_input);
  const long double q_norm =
      std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const long double qn_norm = std::sqrt(
      q[2] * q[2] + (equal_singular ? q[3] * q[3] : 0.0L));
  // The orthogonal canonicalization introduces a few ulps even when q_N is
  // algebraically zero.  Remove only that bounded transform noise; a genuinely
  // small value such as 1e-13 at unit scale remains on the unique interior
  // stratum and is never relabeled as a PSD-boundary optimum.
  const long double qn_canonicalization_tol =
      32.0L * static_cast<long double>(std::numeric_limits<double>::epsilon()) *
      std::max(1.0L, q_norm);
  // Thus the numerical classifier is backward-stable with respect to the
  // canonicalization, rather than a literal exact-stratum test.  No boundary
  // label is returned unless the final feasibility/KKT/PSD certificate passes.
  const bool qn_is_zero = qn_norm <= qn_canonicalization_tol;

  LD4 ubar = {-q[0] / (2.0L * a), -q[1] / (a + b), 0.0L, 0.0L};
  if (!equal_singular) {
    ubar[3] = -q[3] / (a - b);
  }
  const long double r = constraint_value(a, b, q, g, ubar);
  // Do not merge genuinely positive/negative strata.  This narrow guard only
  // absorbs the forward-error floor of evaluating h(ubar); values such as
  // +/-1e-13 remain distinct at unit scale.
  const long double r_roundoff_tol =
      32.0L * static_cast<long double>(std::numeric_limits<double>::epsilon()) *
      constraint_scale(a, b, q, g, ubar);
  const bool r_is_roundoff_zero = std::abs(r) <= r_roundoff_tol;
  if (qn_is_zero && std::isfinite(r) &&
      (r > 0.0L || r_is_roundoff_zero)) {
    LD4 u = ubar;
    bool nonunique = false;
    if (r > r_roundoff_tol) {
      // Deterministic first-null-vector representative.  Keep the complete
      // quadratic formula even though exact stratum membership gives q_z=0.
      const long double disc = q[2] * q[2] + a * r;
      if (!(disc >= 0.0L) || !std::isfinite(disc)) {
        result.status = LOTT_STATUS_FAILED_CERTIFICATE;
        return result;
      }
      u[2] = (q[2] + std::sqrt(disc)) / a;
      nonunique = true;
    }
    result.root.converged = true;
    result.root.multiplier = 1.0 / (2.0 * a_input);
    set_double_correction(u, result);
    if (result.correction.allFinite() &&
        certify(a, b, q, g, rounded_correction(result), 1.0L, true,
                result)) {
      result.status = nonunique ? LOTT_STATUS_BOUNDARY_PSD_NONUNIQUE
                                : LOTT_STATUS_BOUNDARY_PSD_UNIQUE;
      result.certified_solution_count = nonunique ? 2 : 1;
      return result;
    }
    result.status = LOTT_STATUS_FAILED_CERTIFICATE;
    result.correction.setConstant(std::numeric_limits<double>::quiet_NaN());
    return result;
  }

  // The remaining cases have a unique root in 0<mu<1.  phi is strictly
  // decreasing there.  Establish a finite negative right endpoint without
  // ever crossing or evaluating the singular boundary.
  long double q2 = 0.0L;
  for (const long double qj : q) {
    q2 += qj * qj;
  }
  // A valid selected-chart Householder proposal is useful as a starting point,
  // but never trusted as a root: it is first evaluated as a multiplier and the
  // bracket remains entirely inside [0,1).  The Newton-at-zero estimate is the
  // deterministic fallback when the chart proposal is unavailable.
  long double right = static_cast<long double>(initial_mu_input);
  if (!(right > 0.0L) || !(right < 1.0L) || !std::isfinite(right)) {
    right = (q2 > 0.0L) ? (a * g / (2.0L * q2)) : 0.5L;
  }
  if (!(right > 0.0L) || !std::isfinite(right)) {
    right = 0.25L;
  }
  right = std::min(right, 0.5L);
  PhiEvaluation right_eval = evaluate_phi(a, b, q, g, right);
  int bracket_steps = 0;
  constexpr int kMaxBracketSteps = 256;
  while ((std::isnan(right_eval.value) || right_eval.value > 0.0L) &&
         bracket_steps < kMaxBracketSteps) {
    const long double next = (right < 0.5L) ? std::min(2.0L * right, 0.5L)
                                            : 0.5L * (right + 1.0L);
    if (!(next > right) || !(next < 1.0L)) {
      break;
    }
    right = next;
    right_eval = evaluate_phi(a, b, q, g, right);
    ++bracket_steps;
  }
  if (std::isnan(right_eval.value) || right_eval.value > 0.0L ||
      !(right < 1.0L)) {
    // The mu representation may run out of spacing before a very near-endpoint
    // sign change becomes representable.  Retry in tau=1-mu rather than
    // crossing the PSD endpoint or accepting an uncertified candidate.
    return solve_regular_tau(a, b, q, g, a_input);
  }
  if (right > 0.75L) {
    // Near the endpoint, tau retains relative accuracy in the singular
    // denominator while mu does not.  This also keeps the finite root distinct
    // from the excluded tau=0 boundary.
    return solve_regular_tau(a, b, q, g, a_input);
  }

  long double left = 0.0L;
  PhiEvaluation left_eval = evaluate_phi(a, b, q, g, left);
  long double x = 0.5L * right;
  PhiEvaluation x_eval = evaluate_phi(a, b, q, g, x);
  constexpr int kMaxSolveSteps = 160;
  constexpr long double kPhiTol =
      128.0L * static_cast<long double>(std::numeric_limits<double>::epsilon());
  int iterations = 0;
  int bisections = 0;
  for (; iterations < kMaxSolveSteps; ++iterations) {
    if (std::isnan(x_eval.value)) {
      ++result.root.nonfinite_eval_steps;
      x = 0.5L * (left + right);
      x_eval = evaluate_phi(a, b, q, g, x);
      ++bisections;
      if (std::isnan(x_eval.value)) {
        break;
      }
    }

    if (std::abs(x_eval.value) <= kPhiTol * x_eval.scale) {
      break;
    }
    if (x_eval.value > 0.0L) {
      left = x;
      left_eval = x_eval;
    } else {
      right = x;
      right_eval = x_eval;
    }

    const long double midpoint = 0.5L * (left + right);
    if (!(midpoint > left) || !(midpoint < right)) {
      // The two endpoints are adjacent at working precision.  Return the one
      // with the smaller secular residual and let the explicit certificate
      // decide whether double precision is adequate for this instance.
      if (std::abs(left_eval.value) <= std::abs(right_eval.value)) {
        x = left;
        x_eval = left_eval;
      } else {
        x = right;
        x_eval = right_eval;
      }
      break;
    }

    long double proposal = std::numeric_limits<long double>::quiet_NaN();
    if (std::isfinite(x_eval.derivative) && x_eval.derivative < 0.0L) {
      proposal = x - x_eval.value / x_eval.derivative;
    }
    if (!(proposal > left) || !(proposal < right) ||
        !std::isfinite(proposal)) {
      proposal = midpoint;
      ++bisections;
    }
    x = proposal;
    x_eval = evaluate_phi(a, b, q, g, x);
  }

  result.root.used_sign_bracket = true;
  result.root.bracket_is_multiplier = true;
  result.root.iterations = iterations + bracket_steps;
  result.root.bisection_steps = bisections;
  result.root.bracket_left = static_cast<double>(
      left / (2.0L * static_cast<long double>(a_input)));
  result.root.bracket_right = static_cast<double>(
      right / (2.0L * static_cast<long double>(a_input)));
  result.root.multiplier = static_cast<double>(
      x / (2.0L * static_cast<long double>(a_input)));

  if (std::isnan(x_eval.value) || !(x >= 0.0L) || !(x < 1.0L)) {
    result.status = LOTT_STATUS_FAILED_BRACKET;
    return result;
  }

  const LD4 u = reconstruct_regular(a, b, q, x);
  set_double_correction(u, result);
  if (result.correction.allFinite() &&
      certify(a, b, q, g, rounded_correction(result), x, false, result)) {
    result.root.converged = true;
    result.status = LOTT_STATUS_REGULAR_INTERIOR;
    result.certified_solution_count = 1;
    return result;
  }
  result.status = LOTT_STATUS_FAILED_CERTIFICATE;
  result.correction.setConstant(std::numeric_limits<double>::quiet_NaN());
  return result;
}

// Distance Metric - return squared reprojection error
//  This is a quadratic upgrade to the linear sampson distance
double lott_distance_quadratic(const Eigen::Matrix<double, 3, 3> &F,
                               const Eigen::Matrix<double, 4, 1> &A) {
  // Singular value, a - This factor is point independent,
  //  may be extracted for speed - compiler will probably do that for you given
  //  "const" keyword
  const double r1 = sqrt((F(0, 0) + F(1, 1)) * (F(0, 0) + F(1, 1)) +
                         (F(0, 1) - F(1, 0)) * (F(0, 1) - F(1, 0)));
  const double r2 = sqrt((F(0, 0) - F(1, 1)) * (F(0, 0) - F(1, 1)) +
                         (F(0, 1) + F(1, 0)) * (F(0, 1) + F(1, 0)));
  const double a = 0.5 * (r1 + r2);

  // Point dependent parameters:
  // parameter nu^2
  const double Fr0x0 = F(0, 0) * A(0) + F(0, 1) * A(1) + F(0, 2);
  const double Fr1x0 = F(1, 0) * A(0) + F(1, 1) * A(1) + F(1, 2);
  const double Fc0x1 = F(0, 0) * A(2) + F(1, 0) * A(3) + F(2, 0);
  const double Fc1x1 = F(0, 1) * A(2) + F(1, 1) * A(3) + F(2, 1);
  const double nu2 =
      Fr0x0 * Fr0x0 + Fr1x0 * Fr1x0 + Fc0x1 * Fc0x1 + Fc1x1 * Fc1x1;

  // parameter g
  const double g = 2 * (Fr0x0 * A(2) + Fr1x0 * A(3) + F(2, 0) * A(0) +
                        F(2, 1) * A(1) + F(2, 2));

  // Compute the squared reprojection error
  const double den = (6 * a * g + 2 * nu2);
  const double eps2 = g * g * nu2 / (den * den);
  return eps2;
}

void lott_triangulate(const Eigen::Matrix<double, 4, -1> &A,
                      const Eigen::Matrix<double, 3, 3> &F,
                      Eigen::Matrix<double, 4, -1> &X,
                      LottSolverDiagnostics *solver_diag = nullptr,
                      const bool enable_root_count_certificate = false,
                      const int root_solver_mode = 0,
                      Eigen::VectorXi *cert_closed_count_per_point = nullptr,
                      Eigen::VectorXi *status_per_point = nullptr) {
  X.resize(4, A.cols());
  if (cert_closed_count_per_point != nullptr) {
    cert_closed_count_per_point->resize(A.cols());
    cert_closed_count_per_point->setConstant(std::numeric_limits<int>::min());
  }
  if (status_per_point != nullptr) {
    status_per_point->resize(A.cols());
    status_per_point->setConstant(static_cast<int>(LOTT_STATUS_UNSET));
  }

  // Step 1: Compute the SVD of the upper left 2x2 of F
  //  Eigen's SVD is also good, but a little slower.  This is provided for
  //  portability.
  const SVD2x2_Jacobi svd(F.block<2, 2>(0, 0));

  // Build the joint rotation of the 4D image space
  Eigen::Matrix<double, 4, 4> R;
  R.block<2, 2>(0, 0) = svd.V().transpose();
  R.block<2, 2>(2, 0) = -R.block<2, 2>(0, 0);
  R.block<2, 2>(0, 2) = svd.U().transpose();
  R.block<2, 2>(2, 2) = R.block<2, 2>(0, 2);
  R *= M_SQRT1_2;

  // Step 2: Create the coefficient 4-vector that will become [c,d,e,f]
  const Eigen::Matrix<double, 4, 1> beta(F(2, 0), F(2, 1), F(0, 2), F(1, 2));
  // Rotate the coefficients into the new blended joint image space
  const Eigen::Matrix<double, 4, 1> beta_r = R * beta;

  Eigen::Matrix<double, 4, 1> Xod, X_hat;
  Eigen::Matrix<double, 4, 1> beta_rt;

  const double a = svd.d(0);
  const double b = svd.d(1);
  // std::cout << "Singular values: a = " << a << ", b = " << b << std::endl;
  // std::cout << 4*(a*a-b*b)*(5*a*a-b*b) << std::endl;

  // Eigen::Matrix<double,7,5> Mx;
  // const double S1 = (a-b)*(a+b);
  // Mx.row(6) << 0,0,0,0,1;
  // Mx.row(5) << 2,2,2,2,6*a;
  // Mx.row(4) << 13*a, 10*a+3*b, 7*a, 10*a - 3*b, 13*a*a - 2*b*b;
  // Mx.row(3) << 16*(2*a*a - b*b), 16*a*a, 8*(a*a + b*b), 16*a*a, 4*a*(3*a*a
  // +b*b); Mx.row(2) << a*(37*a*a - 29*b*b), 8*a*a*a + S1*b, a*(3*a*a + 5*b*b),
  // 8*a*a*a-a*a*b+b*b*b, 4*a*a*a*a + 3*a*a*b*b + b*b*b*b; Mx.row(1) <<
  // 4*S1*(4*a*a+S1),0,0,0,0; Mx.row(0) << 4*a*S1*S1,0,0,0,0;

  // std::cout << Mx << std::endl;
  // For each point pair, triangulate the nearest valid point
  for (int i = 0; i < A.cols(); i++) {
    int cert_closed_count = std::numeric_limits<int>::min();

    if (solver_diag != nullptr) {
      ++solver_diag->points_total;
    }
    // Step 3: Rotate point into the joint space:
    //  A = [u0,v0,u1,v1]^T
    const Eigen::Matrix<double, 4, 1> Ar = R * A.col(i);

    // Step 4: Translate quadric (compute [c,d,e,f])
    //  const double & a = svd.d(0);
    //  const double & b = svd.d(1);

    beta_rt(0) = a * Ar(0) + beta_r(0);
    beta_rt(1) = b * Ar(1) + beta_r(1);
    beta_rt(2) = -a * Ar(2) + beta_r(2);
    beta_rt(3) = -b * Ar(3) + beta_r(3);

    double &c = beta_rt(0);
    double &d = beta_rt(1);
    double &e = beta_rt(2);
    double &f = beta_rt(3);

    // Step 5: compute g
    double g = Ar.dot(beta_rt + beta_r) + 2.0 * F(2, 2);

    // Step 6: conditionally swap images if g is negative
    const bool swap_images = (g < 0);
    if (swap_images) {
      const double cp = -e;
      const double dp = -f;
      e = -c;
      f = -d;
      c = cp;
      d = dp;
      g = -g;
    }

    // Step 7: determine which polynomial to use based on the magnitude of c,d,e,f.
    // The near-zero policy uses rho_c = |c| / max(|c|,|d|,|e|,|f|). When rho_c is
    // small, argmax chart selection naturally avoids the x-chart and reduces
    // sensitivity to c ~ 0 in finite precision.
    const double absc = std::abs(c);
    const double absd = std::abs(d);
    const double abse = std::abs(e);
    const double absf = std::abs(f);
    const double c_ratio =
        absc / std::max({absc, absd, abse, absf, 1e-18});
    const bool c_near_zero = (c_ratio <= C_NEAR_ZERO_RATIO_TOL);

    // Find index (1-4) of coefficient with largest magnitude
    int largest_idx = 1;
    double largest_val = absc;

    if (absd > largest_val) {
      largest_idx = 2;
      largest_val = absd;
    }
    if (abse > largest_val) {
      largest_idx = 3;
      largest_val = abse;
    }
    if (absf > largest_val) {
      largest_idx = 4;
      // largest_val = absf;
    }
    if (solver_diag != nullptr) {
      ++solver_diag->chart_points[largest_idx - 1];
      if (largest_idx != 1) {
        ++solver_diag->chart_non_x_points;
      }
      if (c_near_zero) {
        ++solver_diag->c_near_zero_points;
        if (largest_idx != 1) {
          ++solver_diag->c_near_zero_non_x_points;
        }
      }
    }

    // Step 7:
    //  According to which of c, d, e, f has the larger amplitude,
    //  pick the appropriate normalized polynomial
    double p[7];
    if (largest_idx == 1) {
      lott_poly6_cx<7>(a, b, c, d, e, f, g, p);
    } else if (largest_idx == 2) {
      // Note that this is the same as the x-polynomial,
      //  but with a & b swapped, d & c swapped, and f & e swapped (see Jupyter
      //  notebook)
      lott_poly6_cx<7>(b, a, d, c, f, e, g, p);
    } else if (largest_idx == 3) {
      // Note that this is the same as the x-polynomial,
      //  but with a swapped with -a, e & c are swapped,
      lott_poly6_cx<7>(-a, b, e, d, c, f, g, p);
    } else // largest_idx == 4
    {
      // The w-poly is the same as the x-polynomial,
      // but a & -b swapped, b & -a swapped, c & f swapped, d & e swapped
      lott_poly6_cx<7>(-b, -a, f, e, d, c, g, p);
    }

    // Step 8-10.  Mode zero (and unsupported mode values) use the theorem-
    // aligned multiplier solver.  Modes 1..5 deliberately retain the original
    // one-step Householder approximations for timing/accuracy experiments and
    // are explicitly reported as uncertified.
    const bool use_certified_solver =
        (root_solver_mode == 0 || root_solver_mode < 1 || root_solver_mode > 5);
    LottPointStatus point_status = LOTT_STATUS_UNSET;
    LottRootDiagnostics root_diag;

    if (use_certified_solver) {
      const double chart_t0 = householder_step_from_origin<4>(p);
      const double chart_delta =
          (largest_idx == 1) ? a
          : (largest_idx == 2) ? b
          : (largest_idx == 3) ? -a
                               : -b;
      const double chart_map_den = 1.0 + chart_delta * chart_t0;
      const double chart_mu0 =
          (std::isfinite(chart_t0) && std::isfinite(chart_map_den) &&
           chart_map_den != 0.0)
              ? (-a * chart_t0 / chart_map_den)
              : std::numeric_limits<double>::quiet_NaN();
      const LottCertifiedPointResult solved = lott_solve_certified_point(
          a, b, c, d, e, f, g, largest_idx - 1, chart_mu0);
      Xod = solved.correction;
      point_status = solved.status;
      root_diag = solved.root;
      cert_closed_count = solved.certified_solution_count;

      if (solver_diag != nullptr) {
        if (root_diag.used_sign_bracket) {
          ++solver_diag->roots_bracketed;
        }
        if (root_diag.converged) {
          ++solver_diag->roots_converged;
        } else {
          ++solver_diag->roots_max_steps;
          if (!root_diag.used_sign_bracket) {
            ++solver_diag->roots_unbracketed;
          }
        }
        solver_diag->total_iterations += root_diag.iterations;
        solver_diag->bisection_steps += root_diag.bisection_steps;
        solver_diag->guarded_halfsteps += root_diag.guarded_halfsteps;
        solver_diag->nonfinite_eval_steps += root_diag.nonfinite_eval_steps;

        switch (point_status) {
        case LOTT_STATUS_ALREADY_FEASIBLE:
          ++solver_diag->already_feasible_points;
          break;
        case LOTT_STATUS_AFFINE:
          ++solver_diag->affine_points;
          break;
        case LOTT_STATUS_REGULAR_INTERIOR:
          ++solver_diag->regular_interior_points;
          break;
        case LOTT_STATUS_BOUNDARY_PSD_UNIQUE:
          ++solver_diag->boundary_psd_unique_points;
          break;
        case LOTT_STATUS_BOUNDARY_PSD_NONUNIQUE:
          ++solver_diag->boundary_psd_nonunique_points;
          break;
        case LOTT_STATUS_FAILED_INVALID_INPUT:
          ++solver_diag->failed_invalid_input_points;
          break;
        case LOTT_STATUS_FAILED_BRACKET:
          ++solver_diag->failed_bracket_points;
          break;
        case LOTT_STATUS_FAILED_CERTIFICATE:
          ++solver_diag->failed_certificate_points;
          break;
        default:
          break;
        }

        if (enable_root_count_certificate) {
          ++solver_diag->cert_points;
          if (cert_closed_count == 1) {
            ++solver_diag->cert_rootcount_eq1;
          } else if (cert_closed_count > 1) {
            ++solver_diag->cert_rootcount_gt1;
          } else {
            ++solver_diag->cert_failures;
            if (!solved.feasibility_ok) {
              ++solver_diag->cert_feasibility_failures;
            }
            if (!solved.kkt_ok) {
              ++solver_diag->cert_kkt_failures;
            }
            if (!solved.hessian_ok) {
              ++solver_diag->cert_psd_failures;
            }
            if (point_status == LOTT_STATUS_FAILED_BRACKET) {
              ++solver_diag->cert_missing_bracket;
            }
          }
        }
      }
    } else {
      double rt = std::numeric_limits<double>::quiet_NaN();
      if (root_solver_mode == 1) {
        rt = householder_step_from_origin<1>(p);
      } else if (root_solver_mode == 2) {
        rt = householder_step_from_origin<2>(p);
      } else if (root_solver_mode == 3) {
        rt = householder_step_from_origin<3>(p);
      } else if (root_solver_mode == 4) {
        rt = householder_step_from_origin<4>(p);
      } else if (root_solver_mode == 5) {
        rt = householder_step_from_origin<5>(p);
      }
      root_diag.converged = std::isfinite(rt);
      point_status = LOTT_STATUS_UNCERTIFIED_APPROXIMATE;
      cert_closed_count = -1;

      if (std::isfinite(rt)) {
        if (largest_idx == 1) {
          Xod(0) = c * rt;
          Xod(1) = (d * rt) / ((a - b) * rt + 1.0);
          Xod(2) = (e * rt) / (2.0 * a * rt + 1.0);
          Xod(3) = (f * rt) / ((a + b) * rt + 1.0);
        } else if (largest_idx == 2) {
          Xod(0) = (-c * rt) / ((a - b) * rt - 1.0);
          Xod(1) = d * rt;
          Xod(2) = (e * rt) / ((a + b) * rt + 1.0);
          Xod(3) = (f * rt) / (2.0 * b * rt + 1.0);
        } else if (largest_idx == 3) {
          Xod(0) = (-c * rt) / (2.0 * a * rt - 1.0);
          Xod(1) = (-d * rt) / ((a + b) * rt - 1.0);
          Xod(2) = e * rt;
          Xod(3) = (-f * rt) / ((a - b) * rt - 1.0);
        } else {
          Xod(0) = (-c * rt) / ((a + b) * rt - 1.0);
          Xod(1) = (-d * rt) / (2.0 * b * rt - 1.0);
          Xod(2) = (e * rt) / ((a - b) * rt + 1.0);
          Xod(3) = f * rt;
        }
      } else {
        // Fail visibly.  Returning the observation (rt=0) would silently turn
        // an arithmetic failure into an infeasible result.
        Xod.setConstant(std::numeric_limits<double>::quiet_NaN());
      }
      if (solver_diag != nullptr) {
        ++solver_diag->roots_unbracketed;
        ++solver_diag->uncertified_approximate_points;
        if (root_diag.converged) {
          ++solver_diag->roots_converged;
        } else {
          ++solver_diag->roots_max_steps;
        }
      }
    }

    // conditionally swap images back
    if (swap_images) // [z w x y] <- [x y z w]
    {
      const double x = Xod(0);
      const double y = Xod(1);
      Xod(0) = Xod(2);
      Xod(1) = Xod(3);
      Xod(2) = x;
      Xod(3) = y;
    }

    // Step 11: Transform back to original frame
    Xod += Ar;
    X_hat.head(2) = svd.V() * (Xod.head(2) - Xod.tail(2)) * M_SQRT1_2;
    X_hat.tail(2) = svd.U() * (Xod.head(2) + Xod.tail(2)) * M_SQRT1_2;
    // X_hat = [u0, v0, u1, v1]^T such that [u1,v1,1]*F*[u0,v0,1]^T = 0 exactly

    // Output
    X.col(i) = X_hat;
    if (cert_closed_count_per_point != nullptr) {
      (*cert_closed_count_per_point)(i) = cert_closed_count;
    }
    if (status_per_point != nullptr) {
      (*status_per_point)(i) = static_cast<int>(point_status);
    }
  }
}
