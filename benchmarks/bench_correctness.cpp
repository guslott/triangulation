#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "lott_triangulate.h"
#include "so3_utils.h"
#include "triangulate_hs.h"

namespace {

struct Args {
  std::string csv_out;
  bool cert_all = false;
};

void print_usage(const char *argv0) {
  std::cout << "Usage: " << argv0
            << " [--csv-out <path>] [--cert-all]" << std::endl;
}

bool parse_args(int argc, char **argv, Args &args) {
  for (int i = 1; i < argc; ++i) {
    const std::string token(argv[i]);
    if (token == "--help" || token == "-h") {
      print_usage(argv[0]);
      return false;
    }
    if (token == "--csv-out") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --csv-out" << std::endl;
        return false;
      }
      args.csv_out = argv[++i];
      continue;
    }
    if (token == "--cert-all") {
      args.cert_all = true;
      continue;
    }
    std::cerr << "Unknown argument: " << token << std::endl;
    print_usage(argv[0]);
    return false;
  }
  return true;
}

double reprojection_cost(const Eigen::Vector4d &corr, const Eigen::Vector4d &obs) {
  return (corr.head<2>() - obs.head<2>()).squaredNorm() +
         (corr.tail<2>() - obs.tail<2>()).squaredNorm();
}

double epipolar_residual(const Eigen::Vector4d &A, const Eigen::Matrix3d &F) {
  Eigen::Vector3d lf = F.block<3, 2>(0, 0) * A.head<2>() + F.block<3, 1>(0, 2);
  return std::abs(lf.head<2>().dot(A.tail<2>()) + lf(2));
}

bool finite_epipole_radius(const Eigen::Vector3d &e, const double focal,
                           double &radius_normalized) {
  if (!e.allFinite() || std::abs(e(2)) < 1e-12 || focal <= 0.0) {
    radius_normalized = std::numeric_limits<double>::infinity();
    return false;
  }
  const double u = (e(0) / e(2)) / focal;
  const double v = (e(1) / e(2)) / focal;
  radius_normalized = std::sqrt(u * u + v * v);
  return std::isfinite(radius_normalized);
}

Eigen::Matrix3d random_rotation(std::mt19937 &rng, double max_deg) {
  std::uniform_real_distribution<double> unif(-1.0, 1.0);
  std::uniform_real_distribution<double> ang(0.0, max_deg * M_PI / 180.0);
  Eigen::Vector3d axis(unif(rng), unif(rng), unif(rng));
  const double n = axis.norm();
  if (n < 1e-12) {
    axis = Eigen::Vector3d(1.0, 0.0, 0.0);
  } else {
    axis /= n;
  }
  const double theta = ang(rng);
  return Eigen::AngleAxisd(theta, axis).toRotationMatrix();
}

struct ConditioningParams {
  Eigen::Matrix<double, 4, 4> Rmix = Eigen::Matrix<double, 4, 4>::Zero();
  Eigen::Matrix<double, 4, 1> beta_r = Eigen::Matrix<double, 4, 1>::Zero();
  double a = 0.0;
  double b = 0.0;
  double two_f33 = 0.0;
};

ConditioningParams make_conditioning_params(const Eigen::Matrix3d &F,
                                            const SVD2x2_Jacobi &svd) {
  ConditioningParams cp;
  cp.a = svd.d(0);
  cp.b = svd.d(1);
  cp.two_f33 = 2.0 * F(2, 2);

  cp.Rmix.block<2, 2>(0, 0) = svd.V().transpose();
  cp.Rmix.block<2, 2>(2, 0) = -cp.Rmix.block<2, 2>(0, 0);
  cp.Rmix.block<2, 2>(0, 2) = svd.U().transpose();
  cp.Rmix.block<2, 2>(2, 2) = cp.Rmix.block<2, 2>(0, 2);
  cp.Rmix *= M_SQRT1_2;

  const Eigen::Matrix<double, 4, 1> beta(F(2, 0), F(2, 1), F(0, 2), F(1, 2));
  cp.beta_r = cp.Rmix * beta;
  return cp;
}

double conditioned_c_ratio(const ConditioningParams &cp,
                           const Eigen::Vector4d &obs, double &c_abs) {
  const Eigen::Matrix<double, 4, 1> Ar = cp.Rmix * obs;
  double c = cp.a * Ar(0) + cp.beta_r(0);
  double d = cp.b * Ar(1) + cp.beta_r(1);
  double e = -cp.a * Ar(2) + cp.beta_r(2);
  double f = -cp.b * Ar(3) + cp.beta_r(3);
  double g =
      Ar.dot((Eigen::Vector4d() << c, d, e, f).finished() + cp.beta_r) +
      cp.two_f33;

  if (g < 0.0) {
    const double cp_swap = -e;
    const double dp_swap = -f;
    e = -c;
    f = -d;
    c = cp_swap;
    d = dp_swap;
  }

  c_abs = std::abs(c);
  const double denom =
      std::max({std::abs(c), std::abs(d), std::abs(e), std::abs(f), 1e-18});
  return c_abs / denom;
}

double quantile(std::vector<double> data, const double q) {
  if (data.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double q_clamped = std::clamp(q, 0.0, 1.0);
  const size_t idx = static_cast<size_t>(q_clamped * (data.size() - 1));
  std::nth_element(data.begin(), data.begin() + static_cast<long>(idx), data.end());
  return data[idx];
}

struct SuiteStats {
  int nan_count_lott = 0;
  int nan_count_hs = 0;
  int gap_gt_1e8_count = 0;
  int gap_gt_1e6_count = 0;
  double worst_cost_gap = -std::numeric_limits<double>::infinity();
  double sum_cost_gap = 0.0;
  double sum_lott_res = 0.0;
  double sum_hs_res = 0.0;
  double worst_lott_res = 0.0;
  double worst_hs_res = 0.0;
  long long valid_pts = 0;
  std::vector<double> gaps;
  std::vector<double> lott_residuals;
  std::vector<double> hs_residuals;
  std::vector<double> b_over_a;
  long long solver_points_total = 0;
  long long solver_bracketed = 0;
  long long solver_unbracketed = 0;
  long long solver_converged = 0;
  long long solver_max_steps = 0;
  long long solver_total_iterations = 0;
  long long solver_bisection_steps = 0;
  long long solver_guarded_halfsteps = 0;
  long long solver_nonfinite_eval_steps = 0;
  long long solver_chart_non_x = 0;
  long long solver_c_near_zero_points = 0;
  long long solver_c_near_zero_non_x_points = 0;
  long long solver_cert_points = 0;
  long long solver_cert_rootcount_eq0 = 0;
  long long solver_cert_rootcount_eq1 = 0;
  long long solver_cert_rootcount_gt1 = 0;
  long long solver_cert_failures = 0;
  long long solver_cert_missing_bracket = 0;
  long long solver_cert_nonfinite_endpoints = 0;
  long long solver_cert_no_sign_change = 0;
  long long solver_cert_endpoint_root_left = 0;
  long long solver_cert_endpoint_root_right = 0;
  long long solver_cert_sturm_invalid = 0;
  long long solver_cert_ivt_conflict = 0;
  long long solver_cert_longdouble_attempts = 0;
  long long solver_cert_longdouble_rescues = 0;
  long long solver_cert_longdouble_failures = 0;
  long long epipoles_finite_both = 0;
  long long epipoles_inside_unit_focal = 0;
  long long epipoles_near_unit_focal = 0;
  std::vector<double> epipole_max_radius;
  double c_target_ratio = 1.0;
  long long c_sampling_attempts = 0;
  long long c_target_hits = 0;
  long long c_target_misses = 0;
  long long c_ratio_le_1e_2 = 0;
  long long c_ratio_le_1e_3 = 0;
  std::vector<double> c_ratio;
  std::vector<double> c_abs;
};

void write_point_csv(std::ostream &os, const std::string &suite, const int trial,
                     const int point_idx, const double focal, const double sigma,
                     const double a, const double b, const double b_over_a,
                     const bool lott_finite, const bool hs_finite,
                     const double e_ours, const double e_hs, const double gap,
                     const double epi_ours, const double epi_hs) {
  os << suite << "," << trial << "," << point_idx << "," << std::setprecision(17)
     << focal << "," << sigma << "," << a << "," << b << "," << b_over_a << ","
     << (lott_finite ? 1 : 0) << "," << (hs_finite ? 1 : 0) << "," << e_ours
     << "," << e_hs << "," << gap << "," << epi_ours << "," << epi_hs << "\n";
}

template <typename ScenarioFn>
SuiteStats run_suite(const std::string &suite_name, const int trials, const int npts,
                     std::mt19937 &rng,
                     std::uniform_real_distribution<double> &focal_dist,
                     std::uniform_real_distribution<double> &xy_dist,
                     std::uniform_real_distribution<double> &z_dist,
                     std::uniform_real_distribution<double> &sigma_dist,
                     ScenarioFn scenario, std::ostream *point_csv,
                     const double c_ratio_target = 1.0,
                     const int max_point_attempts = 1,
                     const bool enable_root_certificate = false) {
  SuiteStats s;
  s.c_target_ratio = c_ratio_target;
  s.gaps.reserve(static_cast<size_t>(trials) * static_cast<size_t>(npts));
  s.lott_residuals.reserve(static_cast<size_t>(trials) * static_cast<size_t>(npts));
  s.hs_residuals.reserve(static_cast<size_t>(trials) * static_cast<size_t>(npts));
  s.b_over_a.reserve(static_cast<size_t>(trials));
  s.epipole_max_radius.reserve(static_cast<size_t>(trials));
  s.c_ratio.reserve(static_cast<size_t>(trials) * static_cast<size_t>(npts));
  s.c_abs.reserve(static_cast<size_t>(trials) * static_cast<size_t>(npts));

  for (int tr = 0; tr < trials; ++tr) {
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    scenario(rng, R, t);

    const double f = focal_dist(rng);
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = f;
    K(1, 1) = f;

    Eigen::Matrix3d F = K.inverse().transpose() * skew(t) * R * K.inverse();
    F /= F.norm();

    const SVD2x2_Jacobi svd(F.block<2, 2>(0, 0));
    const ConditioningParams cond = make_conditioning_params(F, svd);
    const double a = svd.d(0);
    const double b = svd.d(1);
    const double ratio = (a > 1e-15) ? (b / a) : 0.0;
    s.b_over_a.push_back(ratio);

    const Eigen::JacobiSVD<Eigen::Matrix3d> svd_f(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Vector3d epi0 = svd_f.matrixV().col(2);  // F * epi0 = 0
    const Eigen::Vector3d epi1 = svd_f.matrixU().col(2);  // F^T * epi1 = 0
    double radius0 = std::numeric_limits<double>::infinity();
    double radius1 = std::numeric_limits<double>::infinity();
    const bool finite0 = finite_epipole_radius(epi0, f, radius0);
    const bool finite1 = finite_epipole_radius(epi1, f, radius1);
    if (finite0 && finite1) {
      ++s.epipoles_finite_both;
      const double max_radius = std::max(radius0, radius1);
      s.epipole_max_radius.push_back(max_radius);
      if (max_radius <= 1.0) {
        ++s.epipoles_inside_unit_focal;
      }
      if (max_radius <= 1.5) {
        ++s.epipoles_near_unit_focal;
      }
    }

    const double sigma = sigma_dist(rng);
    std::normal_distribution<double> noise(0.0, sigma);

    Eigen::Matrix<double, 3, -1> x(3, npts), xp(3, npts);
    Eigen::Matrix<double, 4, -1> Aobs(4, npts);

    const int attempts_cap = std::max(1, max_point_attempts);
    for (int i = 0; i < npts; ++i) {
      Eigen::Vector3d best_xi = Eigen::Vector3d::Zero();
      Eigen::Vector3d best_xpi = Eigen::Vector3d::Zero();
      Eigen::Vector4d best_obs = Eigen::Vector4d::Zero();
      double best_c_ratio = std::numeric_limits<double>::infinity();
      double best_c_abs = std::numeric_limits<double>::infinity();
      int attempts_used = 0;

      for (int attempt = 0; attempt < attempts_cap; ++attempt) {
        ++attempts_used;
        Eigen::Vector3d X(xy_dist(rng), xy_dist(rng), z_dist(rng));
        Eigen::Vector3d Xp = R * X + t;
        if (Xp(2) < 0.1) {
          Xp(2) = 0.1;
        }

        Eigen::Vector3d xi = K * X / X(2);
        Eigen::Vector3d xpi = K * Xp / Xp(2);
        xi(0) += noise(rng);
        xi(1) += noise(rng);
        xpi(0) += noise(rng);
        xpi(1) += noise(rng);
        xi(2) = 1.0;
        xpi(2) = 1.0;

        Eigen::Vector4d obs = Eigen::Vector4d::Zero();
        obs.head<2>() = xi.head<2>();
        obs.tail<2>() = xpi.head<2>();

        double c_abs = std::numeric_limits<double>::quiet_NaN();
        const double c_ratio = conditioned_c_ratio(cond, obs, c_abs);
        if (c_ratio < best_c_ratio) {
          best_c_ratio = c_ratio;
          best_c_abs = c_abs;
          best_xi = xi;
          best_xpi = xpi;
          best_obs = obs;
        }
        if (c_ratio <= c_ratio_target) {
          break;
        }
      }

      s.c_sampling_attempts += attempts_used;
      if (best_c_ratio <= c_ratio_target) {
        ++s.c_target_hits;
      } else {
        ++s.c_target_misses;
      }
      if (best_c_ratio <= 1e-2) {
        ++s.c_ratio_le_1e_2;
      }
      if (best_c_ratio <= 1e-3) {
        ++s.c_ratio_le_1e_3;
      }
      s.c_ratio.push_back(best_c_ratio);
      s.c_abs.push_back(best_c_abs);

      x.col(i) = best_xi;
      xp.col(i) = best_xpi;
      Aobs.col(i) = best_obs;
    }

    Eigen::Matrix<double, 4, -1> Xl(4, npts), Xh(4, npts);
    LottSolverDiagnostics solver_diag;
    lott_triangulate(Aobs, F, Xl, &solver_diag, enable_root_certificate);
    hartley_triangulate(x, xp, F, Xh);
    s.solver_points_total += solver_diag.points_total;
    s.solver_bracketed += solver_diag.roots_bracketed;
    s.solver_unbracketed += solver_diag.roots_unbracketed;
    s.solver_converged += solver_diag.roots_converged;
    s.solver_max_steps += solver_diag.roots_max_steps;
    s.solver_total_iterations += solver_diag.total_iterations;
    s.solver_bisection_steps += solver_diag.bisection_steps;
    s.solver_guarded_halfsteps += solver_diag.guarded_halfsteps;
    s.solver_nonfinite_eval_steps += solver_diag.nonfinite_eval_steps;
    s.solver_chart_non_x += solver_diag.chart_non_x_points;
    s.solver_c_near_zero_points += solver_diag.c_near_zero_points;
    s.solver_c_near_zero_non_x_points += solver_diag.c_near_zero_non_x_points;
    s.solver_cert_points += solver_diag.cert_points;
    s.solver_cert_rootcount_eq0 += solver_diag.cert_rootcount_eq0;
    s.solver_cert_rootcount_eq1 += solver_diag.cert_rootcount_eq1;
    s.solver_cert_rootcount_gt1 += solver_diag.cert_rootcount_gt1;
    s.solver_cert_failures += solver_diag.cert_failures;
    s.solver_cert_missing_bracket += solver_diag.cert_missing_bracket;
    s.solver_cert_nonfinite_endpoints += solver_diag.cert_nonfinite_endpoints;
    s.solver_cert_no_sign_change += solver_diag.cert_no_sign_change;
    s.solver_cert_endpoint_root_left += solver_diag.cert_endpoint_root_left;
    s.solver_cert_endpoint_root_right += solver_diag.cert_endpoint_root_right;
    s.solver_cert_sturm_invalid += solver_diag.cert_sturm_invalid;
    s.solver_cert_ivt_conflict += solver_diag.cert_ivt_conflict;
    s.solver_cert_longdouble_attempts += solver_diag.cert_longdouble_attempts;
    s.solver_cert_longdouble_rescues += solver_diag.cert_longdouble_rescues;
    s.solver_cert_longdouble_failures += solver_diag.cert_longdouble_failures;

    for (int i = 0; i < npts; ++i) {
      const Eigen::Vector4d vl = Xl.col(i);
      const Eigen::Vector4d vh = Xh.col(i);
      const Eigen::Vector4d vo = Aobs.col(i);
      const bool lott_finite = vl.allFinite();
      const bool hs_finite = vh.allFinite();

      double cl = std::numeric_limits<double>::quiet_NaN();
      double ch = std::numeric_limits<double>::quiet_NaN();
      double gap = std::numeric_limits<double>::quiet_NaN();
      double rl = std::numeric_limits<double>::quiet_NaN();
      double rh = std::numeric_limits<double>::quiet_NaN();

      if (!lott_finite) {
        ++s.nan_count_lott;
      }
      if (!hs_finite) {
        ++s.nan_count_hs;
      }

      if (lott_finite && hs_finite) {
        cl = reprojection_cost(vl, vo);
        ch = reprojection_cost(vh, vo);
        gap = cl - ch;
        rl = epipolar_residual(vl, F);
        rh = epipolar_residual(vh, F);

        s.gaps.push_back(gap);
        s.sum_cost_gap += gap;
        if (gap > 1e-8) {
          ++s.gap_gt_1e8_count;
        }
        if (gap > 1e-6) {
          ++s.gap_gt_1e6_count;
        }
        if (gap > s.worst_cost_gap) {
          s.worst_cost_gap = gap;
        }

        s.sum_lott_res += rl;
        s.sum_hs_res += rh;
        s.lott_residuals.push_back(rl);
        s.hs_residuals.push_back(rh);
        if (rl > s.worst_lott_res) {
          s.worst_lott_res = rl;
        }
        if (rh > s.worst_hs_res) {
          s.worst_hs_res = rh;
        }
        ++s.valid_pts;
      }

      if (point_csv != nullptr) {
        write_point_csv(*point_csv, suite_name, tr, i, f, sigma, a, b, ratio,
                        lott_finite, hs_finite, cl, ch, gap, rl, rh);
      }
    }
  }

  return s;
}

void print_suite(const std::string &name, const int trials, const int npts,
                 const SuiteStats &s) {
  const double mean_gap = (s.valid_pts > 0)
                              ? (s.sum_cost_gap / static_cast<double>(s.valid_pts))
                              : std::numeric_limits<double>::quiet_NaN();
  const double mean_lott = (s.valid_pts > 0)
                               ? (s.sum_lott_res / static_cast<double>(s.valid_pts))
                               : std::numeric_limits<double>::quiet_NaN();
  const double mean_hs = (s.valid_pts > 0)
                             ? (s.sum_hs_res / static_cast<double>(s.valid_pts))
                             : std::numeric_limits<double>::quiet_NaN();
  const long long attempted = static_cast<long long>(trials) * npts;

  std::cout << "suite=" << name << std::endl;
  std::cout << "trials=" << trials << " npts_per_trial=" << npts
            << " total_points=" << attempted << std::endl;
  std::cout << "valid_points=" << s.valid_pts << std::endl;
  std::cout << "nan_count=" << s.nan_count_lott << std::endl;
  std::cout << "nan_count_hs=" << s.nan_count_hs << std::endl;
  std::cout << "mean_cost_gap_Eours_minus_Ehs=" << mean_gap << std::endl;
  std::cout << "worst_cost_gap_Eours_minus_Ehs=" << s.worst_cost_gap
            << std::endl;
  std::cout << "count_gap_gt_1e-8=" << s.gap_gt_1e8_count << std::endl;
  std::cout << "count_gap_gt_1e-6=" << s.gap_gt_1e6_count << std::endl;
  std::cout << "mean_abs_epi_lott=" << mean_lott << std::endl;
  std::cout << "mean_abs_epi_hs=" << mean_hs << std::endl;
  std::cout << "worst_abs_epi_lott=" << s.worst_lott_res << std::endl;
  std::cout << "worst_abs_epi_hs=" << s.worst_hs_res << std::endl;
  std::cout << "gap_p50=" << quantile(s.gaps, 0.50) << std::endl;
  std::cout << "gap_p95=" << quantile(s.gaps, 0.95) << std::endl;
  std::cout << "gap_p99=" << quantile(s.gaps, 0.99) << std::endl;
  std::cout << "lott_abs_epi_p95=" << quantile(s.lott_residuals, 0.95)
            << std::endl;
  std::cout << "lott_abs_epi_p99=" << quantile(s.lott_residuals, 0.99)
            << std::endl;
  std::cout << "hs_abs_epi_p95=" << quantile(s.hs_residuals, 0.95) << std::endl;
  std::cout << "hs_abs_epi_p99=" << quantile(s.hs_residuals, 0.99) << std::endl;
  std::cout << "b_over_a_p50=" << quantile(s.b_over_a, 0.50) << std::endl;
  std::cout << "b_over_a_p95=" << quantile(s.b_over_a, 0.95) << std::endl;
  std::cout << "epipoles_finite_both=" << s.epipoles_finite_both << std::endl;
  std::cout << "epipoles_inside_unit_focal=" << s.epipoles_inside_unit_focal << std::endl;
  std::cout << "epipoles_near_unit_focal=" << s.epipoles_near_unit_focal << std::endl;
  std::cout << "epipole_max_radius_p50=" << quantile(s.epipole_max_radius, 0.50)
            << std::endl;
  std::cout << "epipole_max_radius_p95=" << quantile(s.epipole_max_radius, 0.95)
            << std::endl;
  std::cout << "epipole_max_radius_max=" << quantile(s.epipole_max_radius, 1.00)
            << std::endl;
  std::cout << "c_target_ratio=" << s.c_target_ratio << std::endl;
  std::cout << "c_sampling_attempts=" << s.c_sampling_attempts << std::endl;
  std::cout << "c_target_hits=" << s.c_target_hits << std::endl;
  std::cout << "c_target_misses=" << s.c_target_misses << std::endl;
  std::cout << "c_ratio_le_1e-2=" << s.c_ratio_le_1e_2 << std::endl;
  std::cout << "c_ratio_le_1e-3=" << s.c_ratio_le_1e_3 << std::endl;
  std::cout << "c_ratio_p50=" << quantile(s.c_ratio, 0.50) << std::endl;
  std::cout << "c_ratio_p95=" << quantile(s.c_ratio, 0.95) << std::endl;
  std::cout << "c_ratio_max=" << quantile(s.c_ratio, 1.00) << std::endl;
  std::cout << "c_abs_p50=" << quantile(s.c_abs, 0.50) << std::endl;
  std::cout << "c_abs_p95=" << quantile(s.c_abs, 0.95) << std::endl;
  std::cout << "c_abs_max=" << quantile(s.c_abs, 1.00) << std::endl;
  std::cout << "solver_points_total=" << s.solver_points_total << std::endl;
  std::cout << "solver_roots_bracketed=" << s.solver_bracketed << std::endl;
  std::cout << "solver_roots_unbracketed=" << s.solver_unbracketed << std::endl;
  std::cout << "solver_roots_converged=" << s.solver_converged << std::endl;
  std::cout << "solver_roots_max_steps=" << s.solver_max_steps << std::endl;
  std::cout << "solver_total_iterations=" << s.solver_total_iterations << std::endl;
  std::cout << "solver_bisection_steps=" << s.solver_bisection_steps << std::endl;
  std::cout << "solver_guarded_halfsteps=" << s.solver_guarded_halfsteps
            << std::endl;
  std::cout << "solver_nonfinite_eval_steps=" << s.solver_nonfinite_eval_steps
            << std::endl;
  std::cout << "solver_chart_non_x_points=" << s.solver_chart_non_x << std::endl;
  std::cout << "solver_c_near_zero_points=" << s.solver_c_near_zero_points
            << std::endl;
  std::cout << "solver_c_near_zero_non_x_points="
            << s.solver_c_near_zero_non_x_points << std::endl;
  std::cout << "solver_cert_points=" << s.solver_cert_points << std::endl;
  std::cout << "solver_cert_rootcount_eq0=" << s.solver_cert_rootcount_eq0
            << std::endl;
  std::cout << "solver_cert_rootcount_eq1=" << s.solver_cert_rootcount_eq1
            << std::endl;
  std::cout << "solver_cert_rootcount_gt1=" << s.solver_cert_rootcount_gt1
            << std::endl;
  std::cout << "solver_cert_failures=" << s.solver_cert_failures << std::endl;
  const long long cert_not_eq1 =
      std::max(0LL, s.solver_cert_points - s.solver_cert_rootcount_eq1);
  const double cert_not_eq1_rate =
      (s.solver_cert_points > 0)
          ? (static_cast<double>(cert_not_eq1) /
             static_cast<double>(s.solver_cert_points))
          : std::numeric_limits<double>::quiet_NaN();
  std::cout << "solver_cert_nonunique_or_failure=" << cert_not_eq1 << std::endl;
  std::cout << "solver_cert_nonunique_or_failure_rate=" << cert_not_eq1_rate
            << std::endl;
  std::cout << "solver_cert_missing_bracket=" << s.solver_cert_missing_bracket
            << std::endl;
  std::cout << "solver_cert_nonfinite_endpoints="
            << s.solver_cert_nonfinite_endpoints << std::endl;
  std::cout << "solver_cert_no_sign_change=" << s.solver_cert_no_sign_change
            << std::endl;
  std::cout << "solver_cert_endpoint_root_left="
            << s.solver_cert_endpoint_root_left << std::endl;
  std::cout << "solver_cert_endpoint_root_right="
            << s.solver_cert_endpoint_root_right << std::endl;
  std::cout << "solver_cert_sturm_invalid=" << s.solver_cert_sturm_invalid
            << std::endl;
  std::cout << "solver_cert_ivt_conflict=" << s.solver_cert_ivt_conflict
            << std::endl;
  std::cout << "solver_cert_longdouble_attempts="
            << s.solver_cert_longdouble_attempts << std::endl;
  std::cout << "solver_cert_longdouble_rescues="
            << s.solver_cert_longdouble_rescues << std::endl;
  std::cout << "solver_cert_longdouble_failures="
            << s.solver_cert_longdouble_failures << std::endl;
  std::cout << std::endl;
}

} // namespace

int main(int argc, char **argv) {
  Args args;
  if (!parse_args(argc, argv, args)) {
    return (argc > 1) ? 1 : 0;
  }

  std::ofstream point_csv_file;
  std::ostream *point_csv = nullptr;
  if (!args.csv_out.empty()) {
    point_csv_file.open(args.csv_out);
    if (!point_csv_file.is_open()) {
      std::cerr << "Failed to open CSV output: " << args.csv_out << std::endl;
      return 2;
    }
    point_csv = &point_csv_file;
    *point_csv
        << "suite,trial,point_idx,focal,sigma,a,b,b_over_a,lott_finite,hs_finite,"
           "E_ours,E_hs,deltaE,abs_epi_ours,abs_epi_hs\n";
  }

  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> focal_dist(200.0, 1200.0);
  std::uniform_real_distribution<double> t_dist(-1.0, 1.0);
  std::uniform_real_distribution<double> ang_z(-20.0 * M_PI / 180.0,
                                               20.0 * M_PI / 180.0);
  std::uniform_real_distribution<double> xy_dist(-2.0, 2.0);
  std::uniform_real_distribution<double> z_dist(1.0, 6.0);
  std::uniform_real_distribution<double> sigma_dist(0.1, 2.0);

  const int trials = 200;
  const int deg_trials = 120;
  const int c_near_zero_trials = 60;
  const int npts = 200;
  const double c_near_zero_target = 1e-2;
  const int c_near_zero_attempts = 1024;

  const auto general_scenario = [&t_dist](std::mt19937 &local_rng,
                                          Eigen::Matrix3d &R, Eigen::Vector3d &t) {
    R = random_rotation(local_rng, 30.0);
    t = Eigen::Vector3d(t_dist(local_rng), t_dist(local_rng), t_dist(local_rng));
    if (t.norm() < 1e-3) {
      t(0) += 0.1;
    }
  };

  const auto translation_cyclotorsion = [&t_dist, &ang_z](
                                             std::mt19937 &local_rng,
                                             Eigen::Matrix3d &R, Eigen::Vector3d &t) {
    const double a = ang_z(local_rng);
    R = Eigen::AngleAxisd(a, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    t = Eigen::Vector3d(t_dist(local_rng), t_dist(local_rng), 0.0);
    if (t.head<2>().norm() < 1e-3) {
      t(0) += 0.1;
    }
  };

  const auto affine_like = [&t_dist](std::mt19937 &local_rng, Eigen::Matrix3d &R,
                                     Eigen::Vector3d &t) {
    R = Eigen::Matrix3d::Identity();
    t = Eigen::Vector3d(t_dist(local_rng), t_dist(local_rng), 0.0);
    if (t.head<2>().norm() < 1e-3) {
      t(0) += 0.1;
    }
  };

  const auto epipole_inside = [](std::mt19937 &local_rng, Eigen::Matrix3d &R,
                                 Eigen::Vector3d &t) {
    std::uniform_real_distribution<double> tz_dist(0.8, 2.2);
    std::uniform_real_distribution<double> radius_dist(0.05, 0.55);
    std::uniform_real_distribution<double> theta_dist(0.0, 2.0 * M_PI);
    for (int attempt = 0; attempt < 64; ++attempt) {
      R = random_rotation(local_rng, 8.0);
      const double tz = tz_dist(local_rng);
      const double radius = radius_dist(local_rng);
      const double theta = theta_dist(local_rng);
      t = Eigen::Vector3d(radius * tz * std::cos(theta), radius * tz * std::sin(theta), tz);
      const Eigen::Vector3d c2 = -R.transpose() * t;
      if (std::abs(c2(2)) < 1e-9 || std::abs(t(2)) < 1e-9) {
        continue;
      }
      const double r0 = std::hypot(c2(0) / c2(2), c2(1) / c2(2));
      const double r1 = std::hypot(t(0) / t(2), t(1) / t(2));
      if (std::max(r0, r1) <= 0.75) {
        return;
      }
    }
    R = Eigen::Matrix3d::Identity();
    t = Eigen::Vector3d(0.1, -0.08, 1.2);
  };

  const auto epipole_near_image = [](std::mt19937 &local_rng, Eigen::Matrix3d &R,
                                     Eigen::Vector3d &t) {
    std::uniform_real_distribution<double> tz_dist(0.7, 2.0);
    std::uniform_real_distribution<double> radius_dist(0.75, 1.15);
    std::uniform_real_distribution<double> theta_dist(0.0, 2.0 * M_PI);
    for (int attempt = 0; attempt < 96; ++attempt) {
      R = random_rotation(local_rng, 10.0);
      const double tz = tz_dist(local_rng);
      const double radius = radius_dist(local_rng);
      const double theta = theta_dist(local_rng);
      t = Eigen::Vector3d(radius * tz * std::cos(theta), radius * tz * std::sin(theta), tz);
      const Eigen::Vector3d c2 = -R.transpose() * t;
      if (std::abs(c2(2)) < 1e-9 || std::abs(t(2)) < 1e-9) {
        continue;
      }
      const double r0 = std::hypot(c2(0) / c2(2), c2(1) / c2(2));
      const double r1 = std::hypot(t(0) / t(2), t(1) / t(2));
      const double max_r = std::max(r0, r1);
      const double min_r = std::min(r0, r1);
      if (max_r <= 1.35 && min_r >= 0.6) {
        return;
      }
    }
    R = Eigen::Matrix3d::Identity();
    t = Eigen::Vector3d(0.95, 0.0, 1.0);
  };

  const SuiteStats general =
      run_suite("general", trials, npts, rng, focal_dist, xy_dist, z_dist,
                sigma_dist, general_scenario, point_csv, 1.0, 1, args.cert_all);
  print_suite("general", trials, npts, general);

  const SuiteStats trans =
      run_suite("translation_cyclotorsion", deg_trials, npts, rng, focal_dist,
                xy_dist, z_dist, sigma_dist, translation_cyclotorsion, point_csv,
                1.0, 1, args.cert_all);
  print_suite("translation_cyclotorsion", deg_trials, npts, trans);

  const SuiteStats affine =
      run_suite("affine_like", deg_trials, npts, rng, focal_dist, xy_dist, z_dist,
                sigma_dist, affine_like, point_csv, 1.0, 1, args.cert_all);
  print_suite("affine_like", deg_trials, npts, affine);

  const SuiteStats epi_inside =
      run_suite("epipole_inside", deg_trials, npts, rng, focal_dist, xy_dist, z_dist,
                sigma_dist, epipole_inside, point_csv, 1.0, 1, args.cert_all);
  print_suite("epipole_inside", deg_trials, npts, epi_inside);

  const SuiteStats epi_near =
      run_suite("epipole_near_image", deg_trials, npts, rng, focal_dist, xy_dist,
                z_dist, sigma_dist, epipole_near_image, point_csv, 1.0, 1,
                args.cert_all);
  print_suite("epipole_near_image", deg_trials, npts, epi_near);

  const SuiteStats c_near_zero = run_suite(
      "c_near_zero", c_near_zero_trials, npts, rng, focal_dist, xy_dist, z_dist,
      sigma_dist, general_scenario, point_csv, c_near_zero_target,
      c_near_zero_attempts, true);
  print_suite("c_near_zero", c_near_zero_trials, npts, c_near_zero);

  return 0;
}
