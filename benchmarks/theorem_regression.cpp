#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "lott_triangulate_certified.h"

namespace {

constexpr double kSqrtHalf = 0.707106781186547524400844362104849039;

enum class ExpectedKind {
  kAlreadyFeasible,
  kAffine,
  kRegularInterior,
  kBoundaryUnique,
  kBoundaryTwoPoint,
  kBoundaryCircle,
};

LottPointStatus expected_status(const ExpectedKind kind) {
  switch (kind) {
    case ExpectedKind::kAlreadyFeasible:
      return LOTT_STATUS_ALREADY_FEASIBLE;
    case ExpectedKind::kAffine:
      return LOTT_STATUS_AFFINE;
    case ExpectedKind::kRegularInterior:
      return LOTT_STATUS_REGULAR_INTERIOR;
    case ExpectedKind::kBoundaryUnique:
      return LOTT_STATUS_BOUNDARY_PSD_UNIQUE;
    case ExpectedKind::kBoundaryTwoPoint:
    case ExpectedKind::kBoundaryCircle:
      return LOTT_STATUS_BOUNDARY_PSD_NONUNIQUE;
  }
  return LOTT_STATUS_UNSET;
}

struct Fixture {
  std::string name;
  Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
  Eigen::Vector4d observation = Eigen::Vector4d::Zero();
  ExpectedKind expected = ExpectedKind::kRegularInterior;
  int expected_chart = -1;
  bool compare_hartley_sturm = false;
};

struct CanonicalProblem {
  Eigen::Matrix4d R = Eigen::Matrix4d::Zero();
  Eigen::Vector4d Ar = Eigen::Vector4d::Zero();
  Eigen::Vector4d q = Eigen::Vector4d::Zero();
  double a = 0.0;
  double b = 0.0;
  double g = 0.0;
  bool swapped = false;
};

struct TestState {
  int checks = 0;
  int failures = 0;

  void check(const bool condition, const std::string &fixture,
             const std::string &message) {
    ++checks;
    if (condition) {
      return;
    }
    ++failures;
    std::cerr << "FAIL [" << fixture << "] " << message << '\n';
  }

  void near(const double actual, const double expected, const double abs_tol,
            const double rel_tol, const std::string &fixture,
            const std::string &quantity) {
    const double tol =
        abs_tol + rel_tol * std::max(std::abs(actual), std::abs(expected));
    std::ostringstream message;
    message << std::setprecision(17) << quantity << " = " << actual
            << ", expected " << expected << " (tol " << tol << ")";
    check(std::isfinite(actual) && std::abs(actual - expected) <= tol, fixture,
          message.str());
  }
};

// For A=0 and a diagonal upper-left block, this matrix realizes the canonical
// coefficients q=(c,d,e,f).  Its determinant is
//   (a*b*g - b*(c*c-e*e) - a*(d*d-f*f))/2,
// so callers can construct exact rank-two strata directly.
Eigen::Matrix3d canonical_fundamental(const double a, const double b,
                                      const Eigen::Vector4d &q,
                                      const double g) {
  const double c = q(0);
  const double d = q(1);
  const double e = q(2);
  const double f = q(3);
  Eigen::Matrix3d F;
  F << a, 0.0, (c + e) * kSqrtHalf, 0.0, b,
      (d + f) * kSqrtHalf, (c - e) * kSqrtHalf,
      (d - f) * kSqrtHalf, 0.5 * g;
  return F;
}

Eigen::Matrix3d planar_rotation(const double radians) {
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  const double c = std::cos(radians);
  const double s = std::sin(radians);
  rotation.block<2, 2>(0, 0) << c, -s, s, c;
  return rotation;
}

double rank_two_g(const double a, const double b, const Eigen::Vector4d &q) {
  const double numerator =
      b * (q(0) * q(0) - q(2) * q(2)) +
      a * (q(1) * q(1) - q(3) * q(3));
  return numerator / (a * b);
}

CanonicalProblem canonicalize(const Eigen::Matrix3d &F,
                              const Eigen::Vector4d &observation) {
  CanonicalProblem cp;
  const SVD2x2_Jacobi svd(F.block<2, 2>(0, 0));
  cp.a = svd.d(0);
  cp.b = svd.d(1);
  cp.R.block<2, 2>(0, 0) = svd.V().transpose();
  cp.R.block<2, 2>(2, 0) = -cp.R.block<2, 2>(0, 0);
  cp.R.block<2, 2>(0, 2) = svd.U().transpose();
  cp.R.block<2, 2>(2, 2) = cp.R.block<2, 2>(0, 2);
  cp.R *= kSqrtHalf;

  const Eigen::Vector4d beta(F(2, 0), F(2, 1), F(0, 2), F(1, 2));
  const Eigen::Vector4d beta_r = cp.R * beta;
  cp.Ar = cp.R * observation;
  cp.q << cp.a * cp.Ar(0) + beta_r(0),
      cp.b * cp.Ar(1) + beta_r(1),
      -cp.a * cp.Ar(2) + beta_r(2),
      -cp.b * cp.Ar(3) + beta_r(3);
  cp.g = cp.Ar.dot(cp.q + beta_r) + 2.0 * F(2, 2);

  if (cp.g < 0.0) {
    cp.swapped = true;
    const double c = cp.q(0);
    const double d = cp.q(1);
    const double e = cp.q(2);
    const double f = cp.q(3);
    cp.q << -e, -f, -c, -d;
    cp.g = -cp.g;
  }
  return cp;
}

Eigen::Vector4d canonical_correction(const CanonicalProblem &cp,
                                     const Eigen::Vector4d &corrected,
                                     const Eigen::Vector4d &observation) {
  Eigen::Vector4d u = cp.R * (corrected - observation);
  if (cp.swapped) {
    const Eigen::Vector4d original = u;
    u << original(2), original(3), original(0), original(1);
  }
  return u;
}

Eigen::Vector4d diagonal(const CanonicalProblem &cp) {
  return Eigen::Vector4d(cp.a, cp.b, -cp.a, -cp.b);
}

double canonical_constraint(const CanonicalProblem &cp,
                            const Eigen::Vector4d &u) {
  return (diagonal(cp).array() * u.array().square()).sum() +
         2.0 * cp.q.dot(u) + cp.g;
}

double normalized_epipolar_residual(const Eigen::Vector4d &point,
                                    const Eigen::Matrix3d &F) {
  const Eigen::Vector3d x(point(0), point(1), 1.0);
  const Eigen::Vector3d xp(point(2), point(3), 1.0);
  const double scale = std::max(1.0, F.norm() * x.norm() * xp.norm());
  return std::abs(xp.dot(F * x)) / scale;
}

double correction_cost(const Eigen::Vector4d &point,
                       const Eigen::Vector4d &observation) {
  return (point - observation).squaredNorm();
}

int largest_component(const Eigen::Vector4d &q) {
  int index = 0;
  double magnitude = std::abs(q(0));
  for (int i = 1; i < 4; ++i) {
    if (std::abs(q(i)) > magnitude) {
      index = i;
      magnitude = std::abs(q(i));
    }
  }
  return index;
}

double kkt_multiplier(const CanonicalProblem &cp, const Eigen::Vector4d &u) {
  const Eigen::Vector4d gradient =
      diagonal(cp).array() * u.array() + cp.q.array();
  const Eigen::Vector4d twice_gradient = 2.0 * gradient;
  if (twice_gradient.squaredNorm() == 0.0) {
    return 0.0;
  }
  return -u.dot(twice_gradient) / twice_gradient.squaredNorm();
}

double normalized_kkt_residual(const CanonicalProblem &cp,
                               const Eigen::Vector4d &u,
                               const double lambda) {
  const Eigen::Vector4d gradient =
      diagonal(cp).array() * u.array() + cp.q.array();
  const Eigen::Vector4d residual = u + 2.0 * lambda * gradient;
  const double scale =
      1.0 + u.norm() + 2.0 * std::abs(lambda) * gradient.norm();
  return residual.norm() / scale;
}

double minimum_hessian_eigenvalue(const CanonicalProblem &cp,
                                  const double lambda) {
  const Eigen::Vector4d eigenvalues =
      Eigen::Vector4d::Ones() + 2.0 * lambda * diagonal(cp);
  return eigenvalues.minCoeff();
}

Eigen::Vector4d boundary_center(const CanonicalProblem &cp) {
  const double lambda_b = 1.0 / (2.0 * cp.a);
  const Eigen::Vector4d Hdiag =
      Eigen::Vector4d::Ones() + 2.0 * lambda_b * diagonal(cp);
  Eigen::Vector4d center = Eigen::Vector4d::Zero();
  for (int i = 0; i < 4; ++i) {
    if (std::abs(Hdiag(i)) > 1e-13) {
      center(i) = -2.0 * lambda_b * cp.q(i) / Hdiag(i);
    }
  }
  return center;
}

Eigen::Vector4d solve_hartley_sturm(const Fixture &fixture) {
  Eigen::Matrix<double, 3, 1> x;
  Eigen::Matrix<double, 3, 1> xp;
  x << fixture.observation(0), fixture.observation(1), 1.0;
  xp << fixture.observation(2), fixture.observation(3), 1.0;
  Eigen::Matrix<double, 4, -1> answer(4, 1);
  hartley_triangulate(x, xp, fixture.F, answer);
  return answer.col(0);
}

std::vector<Fixture> make_fixtures() {
  std::vector<Fixture> fixtures;

  fixtures.push_back(
      {"already_feasible_g_zero",
       canonical_fundamental(1.0, 0.5, Eigen::Vector4d(1.0, 0.2, 1.0, 0.2),
                             0.0),
       Eigen::Vector4d::Zero(), ExpectedKind::kAlreadyFeasible, 0, false});

  fixtures.push_back(
      {"affine_rank_two",
       canonical_fundamental(0.0, 0.0, Eigen::Vector4d(0.0, 1.0, 0.0, 0.0),
                             1.0),
       Eigen::Vector4d::Zero(), ExpectedKind::kAffine, 1, false});

  // Generic regular problems whose largest canonical coefficient deliberately
  // selects each of the x/y/z/w charts.
  const std::array<Eigen::Vector4d, 4> chart_q = {
      Eigen::Vector4d(4.0, 0.2, 0.1, 0.1),
      Eigen::Vector4d(0.2, 4.0, 0.1, 0.1),
      Eigen::Vector4d(3.9, 3.9, 4.0, 0.1),
      Eigen::Vector4d(3.9, 3.9, 0.1, 4.0),
  };
  const std::array<const char *, 4> chart_names = {"x", "y", "z", "w"};
  for (int i = 0; i < 4; ++i) {
    const double g = rank_two_g(2.0, 1.0, chart_q[i]);
    fixtures.push_back({std::string("generic_chart_") + chart_names[i],
                        canonical_fundamental(2.0, 1.0, chart_q[i], g),
                        Eigen::Vector4d::Zero(),
                        ExpectedKind::kRegularInterior, i, true});
  }

  const Eigen::Vector4d q_small_b(1.0, 0.5, 0.3, 0.5);
  fixtures.push_back(
      {"near_zero_b_over_a",
       canonical_fundamental(1.0, 1e-6, q_small_b,
                             rank_two_g(1.0, 1e-6, q_small_b)),
       Eigen::Vector4d::Zero(), ExpectedKind::kRegularInterior, 0, true});

  const Eigen::Vector4d q_near_equal(1.0, 0.8, 0.2, 0.1);
  fixtures.push_back(
      {"near_equal_singular_values",
       canonical_fundamental(1.0, 1.0 - 1e-10, q_near_equal,
                             rank_two_g(1.0, 1.0 - 1e-10, q_near_equal)),
       Eigen::Vector4d::Zero(), ExpectedKind::kRegularInterior, 0, true});

  const Eigen::Vector4d q_near_endpoint(1.0, 0.0, 5e-5, 0.0);
  fixtures.push_back(
      {"regular_near_psd_endpoint",
       canonical_fundamental(1.0, 0.5, q_near_endpoint,
                             rank_two_g(1.0, 0.5, q_near_endpoint)),
       Eigen::Vector4d::Zero(), ExpectedKind::kRegularInterior, 0, true});

  const Eigen::Matrix3d swap_F = canonical_fundamental(
      1.0, 0.5, Eigen::Vector4d(1.0, 0.0, 0.0, 0.0), 1.0);
  fixtures.push_back({"negative_g_image_swap", swap_F,
                      Eigen::Vector4d(-1.0, 0.0, 0.0, 0.0),
                      ExpectedKind::kRegularInterior, 0, true});

  const Eigen::Vector4d rotated_q(1.2, 0.7, 0.3, 0.2);
  const Eigen::Matrix3d canonical_rotated_F =
      canonical_fundamental(2.0, 0.8, rotated_q,
                            rank_two_g(2.0, 0.8, rotated_q));
  const Eigen::Matrix3d rotated_F =
      planar_rotation(0.37) * canonical_rotated_F *
      planar_rotation(-0.61).transpose();
  const Eigen::Vector4d rotated_observation(0.4, -0.2, -0.3, 0.5);
  const int rotated_chart =
      largest_component(canonicalize(rotated_F, rotated_observation).q);
  fixtures.push_back({"rotated_nonzero_observation", rotated_F,
                      rotated_observation, ExpectedKind::kRegularInterior,
                      rotated_chart, true});

  fixtures.push_back(
      {"b_zero_interior_r_negative",
       canonical_fundamental(1.0, 0.0, Eigen::Vector4d(1.0, 1.0, 0.0, 1.0),
                             4.5),
       Eigen::Vector4d::Zero(), ExpectedKind::kRegularInterior, 0, true});

  fixtures.push_back(
      {"b_zero_boundary_unique_r_zero",
       canonical_fundamental(1.0, 0.0, Eigen::Vector4d(1.0, 1.0, 0.0, 1.0),
                             4.75),
       Eigen::Vector4d::Zero(), ExpectedKind::kBoundaryUnique, 0, false});

  fixtures.push_back(
      {"b_zero_boundary_two_point",
       canonical_fundamental(1.0, 0.0, Eigen::Vector4d(1.0, 1.0, 0.0, 1.0),
                             5.0),
       Eigen::Vector4d::Zero(), ExpectedKind::kBoundaryTwoPoint, 0, false});

  fixtures.push_back(
      {"positive_b_boundary_two_point",
       canonical_fundamental(1.0, 0.5, Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
                             1.0),
       Eigen::Vector4d::Zero(), ExpectedKind::kBoundaryTwoPoint, 0, false});

  fixtures.push_back(
      {"equal_singular_boundary_circle",
       canonical_fundamental(1.0, 1.0, Eigen::Vector4d(1.0, 2.0, 0.0, 0.0),
                             5.0),
       Eigen::Vector4d::Zero(), ExpectedKind::kBoundaryCircle, 1, false});

  return fixtures;
}

void check_fixture(const Fixture &fixture, TestState &state) {
  const int failures_before = state.failures;
  const CanonicalProblem cp = canonicalize(fixture.F, fixture.observation);
  const double determinant_scale = std::max(1.0, std::pow(fixture.F.norm(), 3));
  state.check(std::abs(fixture.F.determinant()) / determinant_scale < 2e-15,
              fixture.name, "constructed F is not rank-two to roundoff");
  const Eigen::JacobiSVD<Eigen::Matrix3d> full_svd(fixture.F);
  const Eigen::Vector3d singular_values = full_svd.singularValues();
  state.check(singular_values(1) > 1e-12 * singular_values(0) &&
                  singular_values(2) <= 2e-14 * singular_values(0),
              fixture.name,
              "constructed F does not have numerical rank exactly two");
  state.check(cp.a + 1e-14 >= cp.b && cp.b >= -1e-14, fixture.name,
              "canonical singular values do not satisfy a >= b >= 0");
  if (fixture.expected_chart >= 0 && cp.q.cwiseAbs().maxCoeff() > 0.0) {
    state.check(largest_component(cp.q) == fixture.expected_chart, fixture.name,
                "fixture did not realize its intended largest-component chart");
  }

  Eigen::Matrix<double, 4, -1> observations(4, 1);
  observations.col(0) = fixture.observation;
  Eigen::Matrix<double, 4, -1> corrected;
  LottSolverDiagnostics diagnostics;
  Eigen::VectorXi certificate_count;
  Eigen::VectorXi point_status;
  lott_triangulate(observations, fixture.F, corrected, &diagnostics, true, 0,
                   &certificate_count, &point_status);

  state.check(corrected.rows() == 4 && corrected.cols() == 1, fixture.name,
              "solver returned the wrong output shape");
  if (corrected.rows() != 4 || corrected.cols() != 1) {
    return;
  }
  const Eigen::Vector4d point = corrected.col(0);
  state.check(point.allFinite(), fixture.name, "solver output is non-finite");
  if (!point.allFinite()) {
    return;
  }

  state.check(diagnostics.points_total == 1, fixture.name,
              "diagnostics.points_total is not one");
  state.check(diagnostics.cert_points == 1, fixture.name,
              "certificate was requested but diagnostics.cert_points is not one");
  state.check(diagnostics.cert_failures == 0, fixture.name,
              "solver reported a certificate failure");
  state.check(certificate_count.size() == 1, fixture.name,
              "per-point certificate count has the wrong size");
  state.check(point_status.size() == 1, fixture.name,
              "per-point status vector has the wrong size");
  if (point_status.size() == 1) {
    const LottPointStatus status =
        static_cast<LottPointStatus>(point_status(0));
    state.check(status == expected_status(fixture.expected), fixture.name,
                "per-point status does not match the expected solver path");
    state.check(lott_status_is_certified(status), fixture.name,
                "returned point status is not certified");
  }

  const long long classified_points =
      diagnostics.already_feasible_points + diagnostics.affine_points +
      diagnostics.regular_interior_points +
      diagnostics.boundary_psd_unique_points +
      diagnostics.boundary_psd_nonunique_points +
      diagnostics.uncertified_approximate_points +
      diagnostics.failed_invalid_input_points +
      diagnostics.failed_bracket_points +
      diagnostics.failed_certificate_points;
  state.check(classified_points == 1, fixture.name,
              "diagnostic path counters do not classify exactly one point");
  state.check(diagnostics.failed_invalid_input_points == 0 &&
                  diagnostics.failed_bracket_points == 0 &&
                  diagnostics.failed_certificate_points == 0,
              fixture.name, "a fail-closed diagnostic path was activated");
  state.check(diagnostics.cert_feasibility_failures == 0 &&
                  diagnostics.cert_kkt_failures == 0 &&
                  diagnostics.cert_psd_failures == 0,
              fixture.name, "a theorem-certificate invariant failed");

  switch (fixture.expected) {
    case ExpectedKind::kAlreadyFeasible:
      state.check(diagnostics.already_feasible_points == 1, fixture.name,
                  "already-feasible path counter was not incremented");
      break;
    case ExpectedKind::kAffine:
      state.check(diagnostics.affine_points == 1, fixture.name,
                  "affine path counter was not incremented");
      break;
    case ExpectedKind::kRegularInterior:
      state.check(diagnostics.regular_interior_points == 1, fixture.name,
                  "regular-interior path counter was not incremented");
      state.check(diagnostics.roots_bracketed == 1, fixture.name,
                  "regular solution did not report a multiplier bracket");
      break;
    case ExpectedKind::kBoundaryUnique:
      state.check(diagnostics.boundary_psd_unique_points == 1, fixture.name,
                  "unique PSD-boundary counter was not incremented");
      break;
    case ExpectedKind::kBoundaryTwoPoint:
    case ExpectedKind::kBoundaryCircle:
      state.check(diagnostics.boundary_psd_nonunique_points == 1, fixture.name,
                  "nonunique PSD-boundary counter was not incremented");
      break;
  }

  if (fixture.expected_chart >= 0) {
    for (int chart = 0; chart < 4; ++chart) {
      const long long expected = (chart == fixture.expected_chart) ? 1 : 0;
      state.check(diagnostics.chart_points[chart] == expected, fixture.name,
                  "largest-component chart diagnostic is inconsistent");
    }
  }

  const bool expected_nonunique =
      fixture.expected == ExpectedKind::kBoundaryTwoPoint ||
      fixture.expected == ExpectedKind::kBoundaryCircle;
  if (certificate_count.size() == 1) {
    if (expected_nonunique) {
      state.check(certificate_count(0) > 1, fixture.name,
                  "nonunique PSD boundary was not reported as nonunique");
      state.check(diagnostics.cert_rootcount_gt1 == 1, fixture.name,
                  "nonunique certificate counter was not incremented");
    } else {
      state.check(certificate_count(0) == 1, fixture.name,
                  "unique solution did not receive certificate count one");
      state.check(diagnostics.cert_rootcount_eq1 == 1, fixture.name,
                  "unique certificate counter was not incremented");
    }
  }

  const double image_residual = normalized_epipolar_residual(point, fixture.F);
  state.check(image_residual <= 2e-10, fixture.name,
              "normalized image-space epipolar residual exceeds 2e-10");

  const Eigen::Vector4d u =
      canonical_correction(cp, point, fixture.observation);
  const double h = canonical_constraint(cp, u);
  const double h_scale =
      1.0 + std::abs(cp.g) + 2.0 * cp.q.norm() * u.norm() +
      std::max(cp.a, cp.b) * u.squaredNorm();
  state.check(std::abs(h) / h_scale <= 2e-10, fixture.name,
              "canonical feasibility residual exceeds 2e-10");

  if (fixture.expected == ExpectedKind::kAlreadyFeasible) {
    state.check((point - fixture.observation).norm() <= 2e-13, fixture.name,
                "g=0 path changed an already-feasible observation");
  } else if (fixture.expected == ExpectedKind::kAffine) {
    const Eigen::Vector4d expected_u(0.0, -0.5, 0.0, 0.0);
    state.check((u - expected_u).norm() <= 2e-12, fixture.name,
                "affine solution is not the exact hyperplane projection");
  } else {
    const double lambda = kkt_multiplier(cp, u);
    const double kkt_residual = normalized_kkt_residual(cp, u, lambda);
    state.check(kkt_residual <= 2e-9, fixture.name,
                "normalized KKT residual exceeds 2e-9");
    state.check(lambda >= -2e-12, fixture.name,
                "KKT multiplier is negative on the certified branch");

    const double lambda_b = 1.0 / (2.0 * cp.a);
    const bool boundary = fixture.expected == ExpectedKind::kBoundaryUnique ||
                          fixture.expected == ExpectedKind::kBoundaryTwoPoint ||
                          fixture.expected == ExpectedKind::kBoundaryCircle;
    if (!boundary) {
      state.check(lambda < lambda_b * (1.0 - 1e-10), fixture.name,
                  "regular solution did not remain inside the PD interval");
      state.check(minimum_hessian_eigenvalue(cp, lambda) > 1e-10,
                  fixture.name, "regular solution Hessian is not PD");
    } else {
      state.near(lambda, lambda_b, 2e-10, 2e-9, fixture.name,
                 "boundary multiplier");
      state.check(minimum_hessian_eigenvalue(cp, lambda) >= -2e-9,
                  fixture.name, "boundary Hessian is not PSD");

      const Eigen::Vector4d center = boundary_center(cp);
      const double r = canonical_constraint(cp, center);
      state.check(r >= -2e-11, fixture.name,
                  "expected boundary fixture has negative computed r");
      const double radius = std::sqrt(std::max(0.0, r / cp.a));
      if (fixture.expected == ExpectedKind::kBoundaryUnique) {
        state.check((u - center).norm() <= 2e-9, fixture.name,
                    "r=0 boundary output differs from the pseudoinverse center");
        state.check(radius <= 2e-9, fixture.name,
                    "r=0 boundary fixture has nonzero nullspace radius");
      } else if (fixture.expected == ExpectedKind::kBoundaryTwoPoint) {
        Eigen::Vector4d regular_difference = u - center;
        const double z = regular_difference(2);
        regular_difference(2) = 0.0;
        state.check(regular_difference.norm() <= 2e-9, fixture.name,
                    "two-point boundary output left the z nullspace");
        state.near(std::abs(z), radius, 2e-9, 2e-9, fixture.name,
                   "two-point boundary radius");
      } else {
        Eigen::Vector4d regular_difference = u - center;
        const double null_radius = regular_difference.tail<2>().norm();
        regular_difference.tail<2>().setZero();
        state.check(regular_difference.norm() <= 2e-9, fixture.name,
                    "circle boundary output left the (z,w) nullspace");
        state.near(null_radius, radius, 2e-9, 2e-9, fixture.name,
                   "circle boundary radius");
      }
    }
  }

  if (fixture.compare_hartley_sturm) {
    const Eigen::Vector4d hs = solve_hartley_sturm(fixture);
    state.check(hs.allFinite(), fixture.name,
                "Hartley-Sturm comparison returned a non-finite point");
    if (hs.allFinite()) {
      const double hs_residual = normalized_epipolar_residual(hs, fixture.F);
      state.check(hs_residual <= 2e-8, fixture.name,
                  "Hartley-Sturm comparison is not epipolar feasible");
      const double cost = correction_cost(point, fixture.observation);
      const double hs_cost = correction_cost(hs, fixture.observation);
      const double tolerance = 2e-8 * (1.0 + std::abs(hs_cost));
      std::ostringstream message;
      message << std::setprecision(17) << "cost " << cost
              << " exceeds Hartley-Sturm cost " << hs_cost << " by more than "
              << tolerance;
      state.check(cost <= hs_cost + tolerance, fixture.name, message.str());
    }
  }

  std::cout << (state.failures == failures_before ? "PASS" : "FAIL") << " ["
            << fixture.name << "] residual=" << std::scientific
            << std::setprecision(3) << image_residual
            << " cost=" << correction_cost(point, fixture.observation) << '\n';
}

void check_fail_closed_point_helpers(TestState &state) {
  constexpr const char *kName = "fail_closed_point_helpers";
  const int failures_before = state.failures;

  // The affine constraint 2*q^T*u+g=0 is infeasible when q=0 and g>0.
  const LottCertifiedPointResult invalid_affine =
      lott_solve_certified_point(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0);
  state.check(invalid_affine.status == LOTT_STATUS_FAILED_INVALID_INPUT, kName,
              "infeasible affine problem did not fail as invalid input");
  state.check(!lott_status_is_certified(invalid_affine.status), kName,
              "infeasible affine problem was marked certified");
  state.check(invalid_affine.certified_solution_count == -1, kName,
              "infeasible affine problem did not retain fail-closed count -1");
  state.check(!invalid_affine.correction.allFinite(), kName,
              "infeasible affine problem returned a finite correction");
  state.check(!invalid_affine.feasibility_ok && !invalid_affine.kkt_ok &&
                  !invalid_affine.hessian_ok,
              kName, "infeasible affine problem set certificate flags");

  const LottCertifiedPointResult nonfinite = lott_solve_certified_point(
      1.0, 0.5, std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0,
      1.0, 0);
  state.check(nonfinite.status == LOTT_STATUS_FAILED_INVALID_INPUT, kName,
              "non-finite coefficient did not fail as invalid input");
  state.check(nonfinite.certified_solution_count == -1, kName,
              "non-finite coefficient did not retain fail-closed count -1");
  state.check(!nonfinite.correction.allFinite(), kName,
              "non-finite coefficient returned a finite correction");

  // This affine problem is mathematically feasible, but its q^T*q computation
  // lies in the subnormal regime at double precision.  It deterministically
  // exercises status=-3 and guards against leaking the finite/partial candidate
  // that originally survived a failed final certificate.
  const LottCertifiedPointResult unrepresentable_affine =
      lott_solve_certified_point(0.0, 0.0, 1e-160, 0.0, 0.0, 0.0, 1.0, 0);
  state.check(unrepresentable_affine.status ==
                  LOTT_STATUS_FAILED_CERTIFICATE,
              kName,
              "unrepresentable affine correction did not fail certification");
  state.check(unrepresentable_affine.certified_solution_count == -1, kName,
              "status=-3 probe did not retain fail-closed count -1");
  state.check(!unrepresentable_affine.correction.allFinite(), kName,
              "status=-3 probe leaked a finite correction");

  std::cout << (state.failures == failures_before ? "PASS" : "FAIL") << " ["
            << kName << "]\n";
}

void check_mixed_status_batch(TestState &state) {
  constexpr const char *kName = "mixed_status_batch";
  const int failures_before = state.failures;
  const Eigen::Matrix3d F = canonical_fundamental(
      1.0, 0.5, Eigen::Vector4d(1.0, 0.0, 0.0, 0.0), 1.0);

  Eigen::Matrix<double, 4, -1> observations(4, 3);
  observations.col(0).setZero();  // PSD boundary, two optima.
  // This is one of the two feasible boundary points expressed back in the raw
  // joint-image coordinates, so its zero correction exercises g=0.
  observations.col(1) << -kSqrtHalf, 0.0, 0.0, 0.0;
  observations.col(2) << 1.0, 0.0, 0.0, 0.0;  // q_N != 0: regular interior.

  Eigen::Matrix<double, 4, -1> corrected;
  LottSolverDiagnostics diagnostics;
  Eigen::VectorXi certificate_count;
  Eigen::VectorXi point_status;
  lott_triangulate(observations, F, corrected, &diagnostics, true, 0,
                   &certificate_count, &point_status);

  state.check(corrected.rows() == 4 && corrected.cols() == 3, kName,
              "mixed batch returned the wrong output shape");
  state.check(certificate_count.size() == 3 && point_status.size() == 3, kName,
              "mixed batch returned wrong-sized per-point metadata");
  if (certificate_count.size() == 3 && point_status.size() == 3) {
    state.check(point_status(0) == LOTT_STATUS_BOUNDARY_PSD_NONUNIQUE &&
                    point_status(1) == LOTT_STATUS_ALREADY_FEASIBLE &&
                    point_status(2) == LOTT_STATUS_REGULAR_INTERIOR,
                kName, "mixed batch statuses are incorrect or out of order");
    state.check(certificate_count(0) == 2 && certificate_count(1) == 1 &&
                    certificate_count(2) == 1,
                kName,
                "mixed batch unique/nonunique certificate counts are incorrect");
  }

  state.check(diagnostics.points_total == 3 && diagnostics.cert_points == 3,
              kName, "mixed batch total/certificate counters are incorrect");
  state.check(diagnostics.boundary_psd_nonunique_points == 1 &&
                  diagnostics.already_feasible_points == 1 &&
                  diagnostics.regular_interior_points == 1,
              kName, "mixed batch path counters are incorrect");
  state.check(diagnostics.affine_points == 0 &&
                  diagnostics.boundary_psd_unique_points == 0 &&
                  diagnostics.uncertified_approximate_points == 0,
              kName, "mixed batch activated an unexpected successful path");
  state.check(diagnostics.failed_invalid_input_points == 0 &&
                  diagnostics.failed_bracket_points == 0 &&
                  diagnostics.failed_certificate_points == 0 &&
                  diagnostics.cert_failures == 0,
              kName, "mixed batch activated a fail-closed path");
  state.check(diagnostics.cert_rootcount_eq1 == 2 &&
                  diagnostics.cert_rootcount_gt1 == 1 &&
                  diagnostics.cert_rootcount_eq0 == 0,
              kName, "mixed batch certificate summary counters are incorrect");
  state.check(diagnostics.chart_points[0] == 3 &&
                  diagnostics.chart_points[1] == 0 &&
                  diagnostics.chart_points[2] == 0 &&
                  diagnostics.chart_points[3] == 0,
              kName, "mixed batch chart counters are incorrect");
  state.check(diagnostics.roots_bracketed == 1, kName,
              "mixed batch should contain exactly one bracketed interior root");

  if (corrected.rows() == 4 && corrected.cols() == 3) {
    for (int i = 0; i < corrected.cols(); ++i) {
      state.check(corrected.col(i).allFinite(), kName,
                  "mixed batch contains a non-finite output");
      if (corrected.col(i).allFinite()) {
        state.check(normalized_epipolar_residual(corrected.col(i), F) <= 2e-10,
                    kName,
                    "mixed batch contains an epipolar-infeasible output");
      }
    }
    state.check((corrected.col(1) - observations.col(1)).norm() <= 2e-13,
                kName,
                "already-feasible member of mixed batch was unnecessarily moved");
  }

  std::cout << (state.failures == failures_before ? "PASS" : "FAIL") << " ["
            << kName << "]\n";
}

void check_near_endpoint_root_diagnostics(TestState &state) {
  constexpr const char *kName = "near_endpoint_root_diagnostics";
  const int failures_before = state.failures;
  constexpr double a = 1.0;
  constexpr double b = 0.5;
  constexpr double c = 1.0;
  constexpr double e = 5e-5;
  constexpr double g = 1.0 - e * e;
  const LottCertifiedPointResult result =
      lott_solve_certified_point(a, b, c, 0.0, e, 0.0, g, 0);

  state.check(result.status == LOTT_STATUS_REGULAR_INTERIOR, kName,
              "near-endpoint point solve was not regular and certified");
  state.check(result.certified_solution_count == 1, kName,
              "near-endpoint point solve was not uniquely certified");
  state.check(result.root.converged && result.root.used_sign_bracket &&
                  result.root.bracket_is_multiplier,
              kName,
              "near-endpoint solve did not report a multiplier-space bracket");
  state.check(std::isfinite(result.root.bracket_left) &&
                  std::isfinite(result.root.bracket_right) &&
                  std::isfinite(result.root.multiplier),
              kName, "near-endpoint bracket metadata is non-finite");
  const double lambda_b = 1.0 / (2.0 * a);
  state.check(result.root.bracket_left >= 0.0 &&
                  result.root.bracket_left <= result.root.multiplier &&
                  result.root.multiplier <= result.root.bracket_right &&
                  result.root.bracket_right < lambda_b,
              kName,
              "reported multiplier is not inside a safe sub-boundary bracket");
  const double mu = 2.0 * a * result.root.multiplier;
  state.check(mu > 0.999 && mu < 1.0, kName,
              "engineered regular root is not close to the PSD endpoint");
  state.check(result.root.minimum_hessian_eigenvalue > 0.0, kName,
              "near-endpoint regular Hessian is not positive definite");
  state.near(result.root.minimum_hessian_eigenvalue, 1.0 - mu, 2e-12, 2e-9,
             kName, "reported minimum Hessian eigenvalue");
  state.check(result.feasibility_ok && result.kkt_ok && result.hessian_ok,
              kName, "near-endpoint theorem certificate flags are incomplete");
  state.check(result.correction.allFinite(), kName,
              "near-endpoint point solver returned a non-finite correction");

  std::cout << (state.failures == failures_before ? "PASS" : "FAIL") << " ["
            << kName << "] mu=" << std::scientific << std::setprecision(6)
            << mu << " status=" << static_cast<int>(result.status)
            << " cert_flags=" << result.feasibility_ok << result.kkt_ok
            << result.hessian_ok << " bracket=" << result.root.bracket_left
            << "," << result.root.bracket_right << '\n';
}

void check_fundamental_scale_invariance(TestState &state) {
  constexpr const char *kName = "fundamental_scale_invariance";
  const int failures_before = state.failures;
  const Eigen::Vector4d q(1.2, 0.7, 0.3, 0.2);
  const Eigen::Matrix3d canonical_F =
      canonical_fundamental(2.0, 0.8, q, rank_two_g(2.0, 0.8, q));
  const Eigen::Matrix3d F = planar_rotation(0.37) * canonical_F *
                            planar_rotation(-0.61).transpose();
  Eigen::Matrix<double, 4, -1> observations(4, 1);
  observations.col(0) << 0.4, -0.2, -0.3, 0.5;

  struct SolveResult {
    Eigen::Vector4d point = Eigen::Vector4d::Constant(
        std::numeric_limits<double>::quiet_NaN());
    int certificate_count = std::numeric_limits<int>::min();
    int status = LOTT_STATUS_UNSET;
    LottSolverDiagnostics diagnostics;
  };
  const auto solve = [&](const double scale) {
    SolveResult result;
    Eigen::Matrix<double, 4, -1> output;
    Eigen::VectorXi certificate_count;
    Eigen::VectorXi status;
    lott_triangulate(observations, scale * F, output, &result.diagnostics, true,
                     0, &certificate_count, &status);
    if (output.rows() == 4 && output.cols() == 1) {
      result.point = output.col(0);
    }
    if (certificate_count.size() == 1) {
      result.certificate_count = certificate_count(0);
    }
    if (status.size() == 1) {
      result.status = status(0);
    }
    return result;
  };

  const SolveResult reference = solve(1.0);
  state.check(reference.point.allFinite() &&
                  reference.status == LOTT_STATUS_REGULAR_INTERIOR &&
                  reference.certificate_count == 1,
              kName, "unscaled reference solve is not a certified interior point");
  for (const double scale : {1e-12, 1e12}) {
    const SolveResult scaled = solve(scale);
    state.check(scaled.point.allFinite(), kName,
                "scaled fundamental matrix produced a non-finite output");
    state.check(scaled.status == reference.status &&
                    scaled.certificate_count == reference.certificate_count,
                kName, "scaling changed status or certificate count");
    if (scaled.point.allFinite() && reference.point.allFinite()) {
      const double tolerance = 2e-9 * (1.0 + reference.point.norm());
      state.check((scaled.point - reference.point).norm() <= tolerance, kName,
                  "scaling F changed the corrected image point");
      state.check(normalized_epipolar_residual(scaled.point, scale * F) <=
                      2e-10,
                  kName, "scaled solve is not epipolar feasible");
    }
    state.check(scaled.diagnostics.points_total == 1 &&
                    scaled.diagnostics.regular_interior_points == 1 &&
                    scaled.diagnostics.cert_failures == 0,
                kName, "scaled solve diagnostics are inconsistent");
  }

  std::cout << (state.failures == failures_before ? "PASS" : "FAIL") << " ["
            << kName << "]\n";
}

void check_exact_stratum_contracts(TestState &state) {
  constexpr const char *kName = "exact_stratum_contracts";
  const int failures_before = state.failures;

  const auto check_regular_or_fail_closed =
      [&](const LottCertifiedPointResult &result,
          const std::string &description) {
        const bool certified_regular =
            result.status == LOTT_STATUS_REGULAR_INTERIOR &&
            result.certified_solution_count == 1 &&
            result.correction.allFinite() && result.feasibility_ok &&
            result.kkt_ok && result.hessian_ok;
        const bool failed_closed =
            result.status < 0 && result.certified_solution_count == -1 &&
            !result.correction.allFinite();
        state.check(certified_regular || failed_closed, kName,
                    description +
                        " was neither certified regular nor failed closed");
        state.check(result.status != LOTT_STATUS_BOUNDARY_PSD_UNIQUE &&
                        result.status != LOTT_STATUS_BOUNDARY_PSD_NONUNIQUE,
                    kName, description + " was incorrectly snapped to a PSD stratum");
      };

  // A nonzero q_N is algebraically regular even when its magnitude is tiny.
  // This was the finite-candidate/status=-3 reproducer during red-team review.
  const LottCertifiedPointResult qn_1e10 = lott_solve_certified_point(
      1.0, 0.0, 1.0, 1.0, 1e-10, 1.0, 5.0, 0);
  check_regular_or_fail_closed(qn_1e10, "q_N=1e-10 point helper");

  const LottCertifiedPointResult qn_1e13 = lott_solve_certified_point(
      1.0, 0.0, 1.0, 1.0, 1e-13, 1.0, 5.0, 0);
  check_regular_or_fail_closed(qn_1e13, "q_N=1e-13 point helper");

  // Here r=g-4.75 exactly in the canonical problem.  Neither sign may be
  // collapsed into the r=0 unique-boundary stratum.
  constexpr double g_center = 4.75;
  constexpr double r_delta = 1e-13;
  const LottCertifiedPointResult r_positive = lott_solve_certified_point(
      1.0, 0.0, 1.0, 1.0, 0.0, 1.0, g_center + r_delta, 0);
  state.check(r_positive.status == LOTT_STATUS_BOUNDARY_PSD_NONUNIQUE &&
                  r_positive.certified_solution_count == 2 &&
                  r_positive.correction.allFinite() &&
                  r_positive.feasibility_ok && r_positive.kkt_ok &&
                  r_positive.hessian_ok,
              kName, "r=+1e-13 was not classified as nonunique PSD boundary");
  state.check(r_positive.status != LOTT_STATUS_BOUNDARY_PSD_UNIQUE, kName,
              "r=+1e-13 collapsed to the r=0 unique boundary");

  const LottCertifiedPointResult r_negative = lott_solve_certified_point(
      1.0, 0.0, 1.0, 1.0, 0.0, 1.0, g_center - r_delta, 0);
  check_regular_or_fail_closed(r_negative, "r=-1e-13 point helper");
  state.check(r_negative.status != LOTT_STATUS_BOUNDARY_PSD_UNIQUE, kName,
              "r=-1e-13 collapsed to the r=0 unique boundary");

  // Exercise the same q_N=1e-10 case through the public batch API.  A failed
  // certification must publish NaN, status<0, and count=-1; a successful one
  // must be finite and feasible.  There is no permitted finite uncertified
  // third state.
  const Eigen::Matrix3d F_qn = canonical_fundamental(
      1.0, 0.0, Eigen::Vector4d(1.0, 1.0, 1e-10, 1.0), 5.0);
  Eigen::Matrix<double, 4, -1> observation(4, 1);
  observation.setZero();
  Eigen::Matrix<double, 4, -1> output;
  LottSolverDiagnostics diagnostics;
  Eigen::VectorXi count;
  Eigen::VectorXi status;
  lott_triangulate(observation, F_qn, output, &diagnostics, true, 0, &count,
                   &status);
  state.check(output.rows() == 4 && output.cols() == 1 && count.size() == 1 &&
                  status.size() == 1,
              kName, "q_N=1e-10 public solve returned malformed outputs");
  if (output.rows() == 4 && output.cols() == 1 && count.size() == 1 &&
      status.size() == 1) {
    const bool public_success =
        status(0) == LOTT_STATUS_REGULAR_INTERIOR && count(0) == 1 &&
        output.col(0).allFinite() &&
        normalized_epipolar_residual(output.col(0), F_qn) <= 2e-10;
    const bool public_failed_closed =
        status(0) < 0 && count(0) == -1 && !output.col(0).allFinite();
    state.check(public_success || public_failed_closed, kName,
                "q_N=1e-10 public solve leaked a finite uncertified candidate");
    state.check(status(0) != LOTT_STATUS_BOUNDARY_PSD_UNIQUE &&
                    status(0) != LOTT_STATUS_BOUNDARY_PSD_NONUNIQUE,
                kName, "q_N=1e-10 public solve snapped to a PSD boundary");

    Eigen::Matrix<double, 4, -1> fallback_output;
    LottSolverDiagnostics fallback_solver_diagnostics;
    LottCertifiedFallbackDiagnostics fallback_diagnostics;
    lott_triangulate_certified_fallback(
        observation, F_qn, fallback_output, &fallback_solver_diagnostics,
        &fallback_diagnostics);
    state.check(fallback_output.rows() == 4 && fallback_output.cols() == 1 &&
                    fallback_output.col(0).allFinite(),
                kName, "certified wrapper did not return a finite q_N=1e-10 point");
    if (fallback_output.rows() == 4 && fallback_output.cols() == 1 &&
        fallback_output.col(0).allFinite()) {
      state.check(normalized_epipolar_residual(fallback_output.col(0), F_qn) <=
                      2e-8,
                  kName, "certified-wrapper result is not epipolar feasible");
    }
    const long long expected_fallbacks = public_success ? 0 : 1;
    state.check(fallback_diagnostics.points_total == 1 &&
                    fallback_diagnostics.fallback_points ==
                        expected_fallbacks &&
                    fallback_diagnostics.fallback_cert_failure_points ==
                        expected_fallbacks,
                kName, "certified-wrapper fallback counters are inconsistent");
  }

  const Eigen::Matrix3d F_qn_1e13 = canonical_fundamental(
      1.0, 0.0, Eigen::Vector4d(1.0, 1.0, 1e-13, 1.0), 5.0);
  Eigen::Matrix<double, 4, -1> output_1e13;
  Eigen::VectorXi count_1e13;
  Eigen::VectorXi status_1e13;
  lott_triangulate(observation, F_qn_1e13, output_1e13, nullptr, true, 0,
                   &count_1e13, &status_1e13);
  state.check(output_1e13.rows() == 4 && output_1e13.cols() == 1 &&
                  count_1e13.size() == 1 && status_1e13.size() == 1,
              kName, "q_N=1e-13 public solve returned malformed outputs");
  if (output_1e13.rows() == 4 && output_1e13.cols() == 1 &&
      count_1e13.size() == 1 && status_1e13.size() == 1) {
    const bool success_1e13 =
        status_1e13(0) == LOTT_STATUS_REGULAR_INTERIOR &&
        count_1e13(0) == 1 && output_1e13.col(0).allFinite() &&
        normalized_epipolar_residual(output_1e13.col(0), F_qn_1e13) <= 2e-10;
    const bool failed_1e13 = status_1e13(0) < 0 && count_1e13(0) == -1 &&
                                !output_1e13.col(0).allFinite();
    state.check(success_1e13 || failed_1e13, kName,
                "q_N=1e-13 public solve violated the certified/fail-closed contract");
    state.check(status_1e13(0) != LOTT_STATUS_BOUNDARY_PSD_UNIQUE &&
                    status_1e13(0) != LOTT_STATUS_BOUNDARY_PSD_NONUNIQUE,
                kName, "q_N=1e-13 public solve was mislabeled as PSD boundary");
  }

  // A certified nonunique endpoint is not a numerical failure.  The wrapper
  // must retain the core's deterministic representative rather than replacing
  // it with the all-candidate fallback.
  const Eigen::Matrix3d F_nonunique = canonical_fundamental(
      1.0, 0.5, Eigen::Vector4d(1.0, 0.0, 0.0, 0.0), 1.0);
  Eigen::Matrix<double, 4, -1> nonunique_output;
  LottCertifiedFallbackDiagnostics nonunique_diagnostics;
  lott_triangulate_certified(observation, F_nonunique, nonunique_output,
                             nullptr, &nonunique_diagnostics);
  state.check(nonunique_output.rows() == 4 &&
                  nonunique_output.cols() == 1 &&
                  nonunique_output.col(0).allFinite() &&
                  normalized_epipolar_residual(nonunique_output.col(0),
                                               F_nonunique) <= 2e-12,
              kName, "wrapper rejected a certified nonunique PSD optimum");
  state.check(nonunique_diagnostics.points_total == 1 &&
                  nonunique_diagnostics.certified_nonunique_points == 1 &&
                  nonunique_diagnostics.fallback_points == 0 &&
                  nonunique_diagnostics.fallback_cert_failure_points == 0,
              kName, "wrapper treated certified nonuniqueness as failure");

  std::cout << (state.failures == failures_before ? "PASS" : "FAIL") << " ["
            << kName << "] qn_statuses="
            << static_cast<int>(qn_1e10.status) << ","
            << static_cast<int>(qn_1e13.status) << " r_statuses="
            << static_cast<int>(r_negative.status) << ","
            << static_cast<int>(r_positive.status) << '\n';
}

}  // namespace

int main() {
  TestState state;
  const std::vector<Fixture> fixtures = make_fixtures();
  for (const Fixture &fixture : fixtures) {
    check_fixture(fixture, state);
  }
  check_fail_closed_point_helpers(state);
  check_mixed_status_batch(state);
  check_near_endpoint_root_diagnostics(state);
  check_fundamental_scale_invariance(state);
  check_exact_stratum_contracts(state);

  std::cout << "theorem_regression: " << (state.checks - state.failures) << "/"
            << state.checks << " checks passed across " << fixtures.size()
            << " fixtures" << '\n';
  return state.failures == 0 ? 0 : 1;
}
