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
  int iterations = 0;
  int bisection_steps = 0;
  int guarded_halfsteps = 0;
  int nonfinite_eval_steps = 0;
  double bracket_left = std::numeric_limits<double>::quiet_NaN();
  double bracket_right = std::numeric_limits<double>::quiet_NaN();
};

struct LottSolverDiagnostics {
  long long points_total = 0;
  long long roots_bracketed = 0;
  long long roots_unbracketed = 0;
  long long roots_converged = 0;
  long long roots_max_steps = 0;
  long long total_iterations = 0;
  long long bisection_steps = 0;
  long long guarded_halfsteps = 0;
  long long nonfinite_eval_steps = 0;
  long long chart_non_x_points = 0;
  long long c_near_zero_points = 0;
  long long c_near_zero_non_x_points = 0;
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
                      Eigen::VectorXi *cert_closed_count_per_point = nullptr) {
  X.resize(4, A.cols());
  if (cert_closed_count_per_point != nullptr) {
    cert_closed_count_per_point->resize(A.cols());
    cert_closed_count_per_point->setConstant(std::numeric_limits<int>::min());
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

    // Step 8-9: solve the selected chart polynomial.
    // root_solver_mode == 0 uses the full safeguarded branch solver.
    // root_solver_mode in {1..5} are one-step Householder approximations.
    int nloops = 0;
    LottRootDiagnostics root_diag;
    double rt = 0.0;
    if (root_solver_mode == 0) {
      rt = full_root_iterative(p, nloops, &root_diag);
    } else if (root_solver_mode == 1) {
      rt = householder_step_from_origin<1>(p);
      root_diag.converged = std::isfinite(rt);
    } else if (root_solver_mode == 2) {
      rt = householder_step_from_origin<2>(p);
      root_diag.converged = std::isfinite(rt);
    } else if (root_solver_mode == 3) {
      rt = householder_step_from_origin<3>(p);
      root_diag.converged = std::isfinite(rt);
    } else if (root_solver_mode == 4) {
      rt = householder_step_from_origin<4>(p);
      root_diag.converged = std::isfinite(rt);
    } else if (root_solver_mode == 5) {
      rt = householder_step_from_origin<5>(p);
      root_diag.converged = std::isfinite(rt);
    } else {
      // Fallback for unsupported modes: use the full safeguarded solver.
      rt = full_root_iterative(p, nloops, &root_diag);
    }
    if (!std::isfinite(rt)) {
      rt = 0.0;
      root_diag.converged = false;
    }
    if (solver_diag != nullptr) {
      if (root_solver_mode == 0) {
        if (root_diag.used_sign_bracket) {
          ++solver_diag->roots_bracketed;
        } else {
          ++solver_diag->roots_unbracketed;
        }
        if (root_diag.converged) {
          ++solver_diag->roots_converged;
        } else {
          ++solver_diag->roots_max_steps;
        }
        solver_diag->total_iterations += root_diag.iterations;
        solver_diag->bisection_steps += root_diag.bisection_steps;
        solver_diag->guarded_halfsteps += root_diag.guarded_halfsteps;
        solver_diag->nonfinite_eval_steps += root_diag.nonfinite_eval_steps;
      } else {
        if (root_diag.converged) {
          ++solver_diag->roots_converged;
        } else {
          ++solver_diag->roots_max_steps;
        }
        ++solver_diag->roots_unbracketed;
      }
      if (enable_root_count_certificate && root_solver_mode == 0) {
        constexpr double kCertEndpointTol = 1e-12;
        ++solver_diag->cert_points;
        if (root_diag.used_sign_bracket &&
            std::isfinite(root_diag.bracket_left) &&
            std::isfinite(root_diag.bracket_right) &&
            (root_diag.bracket_right > root_diag.bracket_left)) {
          const double f_left = polyval<6>(p, root_diag.bracket_left);
          const double f_right = polyval<6>(p, root_diag.bracket_right);
          if (!std::isfinite(f_left) || !std::isfinite(f_right)) {
            ++solver_diag->cert_nonfinite_endpoints;
            ++solver_diag->cert_failures;
          } else {
            auto poly_abs_scale = [&](const double x) {
              const double ax = std::abs(x);
              double s = std::abs(p[0]);
              for (int j = 1; j < 7; ++j) {
                s = s * ax + std::abs(p[j]);
              }
              return std::max(1.0, s);
            };
            const double left_tol = kCertEndpointTol * poly_abs_scale(root_diag.bracket_left);
            const double right_tol =
                kCertEndpointTol * poly_abs_scale(root_diag.bracket_right);

            const bool left_root = (std::abs(f_left) <= left_tol);
            const bool right_root = (std::abs(f_right) <= right_tol);
            if (left_root) {
              ++solver_diag->cert_endpoint_root_left;
            }
            if (right_root) {
              ++solver_diag->cert_endpoint_root_right;
            }

            const int sign_left = (f_left > left_tol)
                                      ? 1
                                      : ((f_left < -left_tol) ? -1 : 0);
            const int sign_right = (f_right > right_tol)
                                       ? 1
                                       : ((f_right < -right_tol) ? -1 : 0);
            const bool strict_sign_change =
                (sign_left != 0 && sign_right != 0 && sign_left != sign_right);
            if (!strict_sign_change) {
              ++solver_diag->cert_no_sign_change;
            }

            const int cert_count = sturm_root_count_open_interval_poly6(
                p, root_diag.bracket_left, root_diag.bracket_right);
            const bool needs_fallback =
                (cert_count < 0) ||
                (strict_sign_change && cert_count == 0 && !left_root &&
                 !right_root);
            int resolved_count = cert_count;
            if (needs_fallback) {
              ++solver_diag->cert_longdouble_attempts;
              const int ld_count =
                  sturm_root_count_open_interval_poly6_long_double(
                      p, root_diag.bracket_left, root_diag.bracket_right);
              if (ld_count >= 0) {
                resolved_count = ld_count;
                ++solver_diag->cert_longdouble_rescues;
              } else {
                ++solver_diag->cert_longdouble_failures;
              }
            }

            if (resolved_count < 0) {
              ++solver_diag->cert_sturm_invalid;
              ++solver_diag->cert_failures;
              cert_closed_count = -1;
            } else if (strict_sign_change && resolved_count == 0 && !left_root &&
                       !right_root) {
              // Inconsistent with IVT/sign bracket after fallback.
              ++solver_diag->cert_ivt_conflict;
              ++solver_diag->cert_failures;
              cert_closed_count = -1;
            } else {
              const int closed_count =
                  resolved_count + (left_root ? 1 : 0) + (right_root ? 1 : 0);
              cert_closed_count = closed_count;
              if (closed_count == 0) {
                ++solver_diag->cert_rootcount_eq0;
              } else if (closed_count == 1) {
                ++solver_diag->cert_rootcount_eq1;
              } else {
                ++solver_diag->cert_rootcount_gt1;
              }
            }
          }
        } else {
          ++solver_diag->cert_missing_bracket;
          ++solver_diag->cert_failures;
          cert_closed_count = -1;
        }
      }
    }

    // OR A variable approximation of the root can be faster though slightly
    // less accurate Select a performance speed versus accuracy from less to
    // more accurate, faster to slower const double rt =
    // householder_step_from_origin<4>(a,b,c,d,e,f,g);

    // Step 10: Assemble the point
    //  Note, for example that the real root is c*rt in the x case
    //   this means that the y value is d*c*rt/((a-b)*c*rt + c)
    //   So we cancel out the c's for a simpler solution less dependent on c

    if (largest_idx == 1) {
      // x = c*rt
      //(x, d*x/((a - b)*x + c), e*x/(2*a*x + c), f*x/((a + b)*x + c), 1)
      Xod(0) = (c * rt);
      Xod(1) = (d * rt) / ((a - b) * rt + 1);
      Xod(2) = (e * rt) / ((a + a) * rt + 1);
      Xod(3) = (f * rt) / ((a + b) * rt + 1);
    } else if (largest_idx == 2) {
      // (-c*y/((a - b)*y - d), y, e*y/((a + b)*y + d), f*y/(2*b*y + d), 1)
      Xod(0) = (-c * rt) / ((a - b) * rt - 1);
      Xod(1) = (d * rt);
      Xod(2) = (e * rt) / ((a + b) * rt + 1);
      Xod(3) = (f * rt) / ((b + b) * rt + 1);
    } else if (largest_idx == 3) {
      //(-c*z/(2*a*z - e), -d*z/((a + b)*z - e), z, -f*z/((a - b)*z - e), 1)
      Xod(0) = (-c * rt) / ((a + a) * rt - 1);
      Xod(1) = (-d * rt) / ((a + b) * rt - 1);
      Xod(2) = (e * rt);
      Xod(3) = (-f * rt) / ((a - b) * rt - 1);
    } else // largest_idx == 4
    {
      //(-c*w/((a + b)*w - f), -d*w/(2*b*w - f), e*w/((a - b)*w + f), w, 1)
      Xod(0) = (-c * rt) / ((a + b) * rt - 1);
      Xod(1) = (-d * rt) / ((b + b) * rt - 1);
      Xod(2) = (e * rt) / ((a - b) * rt + 1);
      Xod(3) = (f * rt);
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
  }
}
