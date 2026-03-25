#include <cmath>
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
#include "triangulate_kanatani.h"
#include "triangulate_lindstrom.h"

namespace {

struct Dataset {
  Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 3, -1> x;
  Eigen::Matrix<double, 3, -1> xp;
  Eigen::Matrix<double, 4, -1> obs;
};

struct CheckResult {
  std::string id;
  bool pass = true;
};

Eigen::Matrix3d random_rotation(std::mt19937 &rng, const double max_deg) {
  std::uniform_real_distribution<double> unif(-1.0, 1.0);
  std::uniform_real_distribution<double> ang(0.0, max_deg * M_PI / 180.0);
  Eigen::Vector3d axis(unif(rng), unif(rng), unif(rng));
  const double n = axis.norm();
  axis = (n < 1e-12) ? Eigen::Vector3d(1.0, 0.0, 0.0) : axis / n;
  return Eigen::AngleAxisd(ang(rng), axis).toRotationMatrix();
}

void swap_output_packing(const Eigen::Matrix<double, 4, -1> &in,
                         Eigen::Matrix<double, 4, -1> &out) {
  out.resize(4, in.cols());
  out.topRows<2>() = in.bottomRows<2>();
  out.bottomRows<2>() = in.topRows<2>();
}

double xpFx(const Eigen::Vector4d &A, const Eigen::Matrix3d &F) {
  Eigen::Vector3d lf = F.block<3, 2>(0, 0) * A.head<2>() + F.block<3, 1>(0, 2);
  return lf.head<2>().dot(A.tail<2>()) + lf(2);
}

double mean_abs_residual(const Eigen::Matrix<double, 4, -1> &X,
                         const Eigen::Matrix3d &F) {
  double sum = 0.0;
  long long finite = 0;
  for (int i = 0; i < X.cols(); ++i) {
    const Eigen::Vector4d p = X.col(i);
    if (!p.allFinite()) {
      continue;
    }
    sum += std::abs(xpFx(p, F));
    ++finite;
  }
  return (finite > 0)
             ? (sum / static_cast<double>(finite))
             : std::numeric_limits<double>::infinity();
}

double mean_correction_cost(const Eigen::Matrix<double, 4, -1> &X,
                            const Eigen::Matrix<double, 4, -1> &obs) {
  double sum = 0.0;
  long long finite = 0;
  for (int i = 0; i < X.cols(); ++i) {
    if (!X.col(i).allFinite()) {
      continue;
    }
    sum += (X.col(i).head<2>() - obs.col(i).head<2>()).squaredNorm() +
           (X.col(i).tail<2>() - obs.col(i).tail<2>()).squaredNorm();
    ++finite;
  }
  return (finite > 0)
             ? (sum / static_cast<double>(finite))
             : std::numeric_limits<double>::infinity();
}

long long count_nonfinite(const Eigen::Matrix<double, 4, -1> &X) {
  long long count = 0;
  for (int i = 0; i < X.cols(); ++i) {
    if (!X.col(i).allFinite()) {
      ++count;
    }
  }
  return count;
}

Dataset make_dataset(const int npts, const double sigma, std::mt19937 &rng,
                     const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
                     const double focal = 800.0) {
  Dataset d;
  d.x.resize(3, npts);
  d.xp.resize(3, npts);
  d.obs.resize(4, npts);

  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  K(0, 0) = focal;
  K(1, 1) = focal;
  d.F = K.inverse().transpose() * skew(t) * R * K.inverse();
  d.F /= d.F.norm();

  std::uniform_real_distribution<double> xy_dist(-2.0, 2.0);
  std::uniform_real_distribution<double> z_dist(1.0, 6.0);
  std::normal_distribution<double> noise(0.0, sigma);

  for (int i = 0; i < npts; ++i) {
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

    d.x.col(i) = xi;
    d.xp.col(i) = xpi;
    d.obs.col(i).head<2>() = xi.head<2>();
    d.obs.col(i).tail<2>() = xpi.head<2>();
  }
  return d;
}

Dataset make_general_dataset(const int npts, const double sigma, std::mt19937 &rng) {
  std::uniform_real_distribution<double> t_dist(-1.0, 1.0);
  Eigen::Matrix3d R = random_rotation(rng, 20.0);
  Eigen::Vector3d t(t_dist(rng), t_dist(rng), t_dist(rng));
  if (t.norm() < 1e-3) {
    t(0) += 0.1;
  }
  return make_dataset(npts, sigma, rng, R, t);
}

Dataset make_translation_dataset(const int npts, const double sigma, std::mt19937 &rng) {
  std::uniform_real_distribution<double> t_dist(-1.0, 1.0);
  std::uniform_real_distribution<double> ang_z(-15.0 * M_PI / 180.0,
                                               15.0 * M_PI / 180.0);
  const double a = ang_z(rng);
  Eigen::Matrix3d R = Eigen::AngleAxisd(a, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Eigen::Vector3d t(t_dist(rng), t_dist(rng), 0.0);
  if (t.head<2>().norm() < 1e-3) {
    t(0) += 0.1;
  }
  return make_dataset(npts, sigma, rng, R, t);
}

Dataset make_epipole_inside_dataset(const int npts, const double sigma, std::mt19937 &rng) {
  std::uniform_real_distribution<double> tz_dist(0.8, 2.2);
  std::uniform_real_distribution<double> radius_dist(0.05, 0.55);
  std::uniform_real_distribution<double> theta_dist(0.0, 2.0 * M_PI);
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t(0.1, -0.08, 1.2);
  for (int attempt = 0; attempt < 64; ++attempt) {
    R = random_rotation(rng, 8.0);
    const double tz = tz_dist(rng);
    const double radius = radius_dist(rng);
    const double theta = theta_dist(rng);
    t = Eigen::Vector3d(radius * tz * std::cos(theta), radius * tz * std::sin(theta),
                        tz);
    const Eigen::Vector3d c2 = -R.transpose() * t;
    if (std::abs(c2(2)) < 1e-9 || std::abs(t(2)) < 1e-9) {
      continue;
    }
    const double r0 = std::hypot(c2(0) / c2(2), c2(1) / c2(2));
    const double r1 = std::hypot(t(0) / t(2), t(1) / t(2));
    if (std::max(r0, r1) <= 0.75) {
      break;
    }
  }
  return make_dataset(npts, sigma, rng, R, t);
}

void run_lott(const Dataset &d, Eigen::Matrix<double, 4, -1> &X) {
  lott_triangulate(d.obs, d.F, X);
}

void run_hs(const Dataset &d, const Eigen::Matrix3d &F_in, Eigen::Matrix<double, 4, -1> &X) {
  X.resize(4, d.obs.cols());
  hartley_triangulate(d.x, d.xp, F_in, X);
}

void run_lind1_raw(const Dataset &d, const Eigen::Matrix3d &F_in,
                   Eigen::Matrix<double, 4, -1> &X) {
  triangulation::lindstrom_niter1(d.x, d.xp, F_in, X);
}

void run_lind2_raw(const Dataset &d, const Eigen::Matrix3d &F_in,
                   Eigen::Matrix<double, 4, -1> &X) {
  triangulation::lindstrom_niter2(d.x, d.xp, F_in, X);
}

void run_kan_raw(const Dataset &d, const Eigen::Matrix3d &F_in,
                 Eigen::Matrix<double, 4, -1> &X) {
  triangulation::kanatani_triangulate(d.x, d.xp, F_in, X);
}

void run_lind1_canonical(const Dataset &d, const Eigen::Matrix3d &F_in,
                         Eigen::Matrix<double, 4, -1> &X) {
  Eigen::Matrix<double, 4, -1> raw;
  run_lind1_raw(d, F_in, raw);
  swap_output_packing(raw, X);
}

void run_lind2_canonical(const Dataset &d, const Eigen::Matrix3d &F_in,
                         Eigen::Matrix<double, 4, -1> &X) {
  Eigen::Matrix<double, 4, -1> raw;
  run_lind2_raw(d, F_in, raw);
  swap_output_packing(raw, X);
}

void run_kan_canonical(const Dataset &d, const Eigen::Matrix3d &F_in,
                       Eigen::Matrix<double, 4, -1> &X) {
  Eigen::Matrix<double, 4, -1> raw;
  run_kan_raw(d, F_in, raw);
  swap_output_packing(raw, X);
}

bool check_ratio(const double preferred, const double alternate,
                 const double ratio_limit) {
  if (!std::isfinite(preferred) || !std::isfinite(alternate)) {
    return false;
  }
  if (alternate <= 0.0) {
    return preferred <= ratio_limit;
  }
  return preferred <= ratio_limit * alternate;
}

CheckResult run_convention_test(const Dataset &d) {
  CheckResult r;
  r.id = "T1_convention";
  std::cout << "== T1 Convention Test ==" << std::endl;
  std::cout << "criterion: preferred packing residual <= 0.1 * alternate"
            << std::endl;

  bool pass = true;
  const double ratio_limit = 0.1;

  auto print_line = [&](const std::string &method, const std::string &preferred,
                        const double m_pref, const double m_alt,
                        const bool ok) {
    std::cout << "method=" << method << " preferred=" << preferred
              << " mean_pref=" << std::setprecision(17) << m_pref
              << " mean_alt=" << m_alt << " status=" << (ok ? "PASS" : "FAIL")
              << std::endl;
  };

  Eigen::Matrix<double, 4, -1> Xraw, Xswap;

  run_lott(d, Xraw);
  swap_output_packing(Xraw, Xswap);
  double lott_asis = mean_abs_residual(Xraw, d.F);
  double lott_swap = mean_abs_residual(Xswap, d.F);
  bool ok = check_ratio(lott_asis, lott_swap, ratio_limit);
  pass = pass && ok;
  print_line("Lott", "as_is", lott_asis, lott_swap, ok);

  run_hs(d, d.F, Xraw);
  swap_output_packing(Xraw, Xswap);
  double hs_asis = mean_abs_residual(Xraw, d.F);
  double hs_swap = mean_abs_residual(Xswap, d.F);
  ok = check_ratio(hs_asis, hs_swap, ratio_limit);
  pass = pass && ok;
  print_line("Hartley-Sturm", "as_is", hs_asis, hs_swap, ok);

  run_lind1_raw(d, d.F.transpose(), Xraw);
  swap_output_packing(Xraw, Xswap);
  double l1_asis = mean_abs_residual(Xraw, d.F);
  double l1_swap = mean_abs_residual(Xswap, d.F);
  ok = check_ratio(l1_swap, l1_asis, ratio_limit);
  pass = pass && ok;
  print_line("Lindstrom niter1", "swapped", l1_swap, l1_asis, ok);

  run_lind2_raw(d, d.F.transpose(), Xraw);
  swap_output_packing(Xraw, Xswap);
  double l2_asis = mean_abs_residual(Xraw, d.F);
  double l2_swap = mean_abs_residual(Xswap, d.F);
  ok = check_ratio(l2_swap, l2_asis, ratio_limit);
  pass = pass && ok;
  print_line("Lindstrom niter2", "swapped", l2_swap, l2_asis, ok);

  run_kan_raw(d, d.F.transpose(), Xraw);
  swap_output_packing(Xraw, Xswap);
  double k_asis = mean_abs_residual(Xraw, d.F);
  double k_swap = mean_abs_residual(Xswap, d.F);
  ok = check_ratio(k_swap, k_asis, ratio_limit);
  pass = pass && ok;
  print_line("Kanatani", "swapped", k_swap, k_asis, ok);

  std::cout << "T1_status=" << (pass ? "PASS" : "FAIL") << std::endl << std::endl;
  r.pass = pass;
  return r;
}

CheckResult run_sanity_test(const Dataset &general, const Dataset &translation,
                            const Dataset &epi_inside) {
  CheckResult r;
  r.id = "T2_sanity";
  std::cout << "== T2 Sanity Test ==" << std::endl;
  std::cout << "criterion: nonfinite outputs = 0 across deterministic suites"
            << std::endl;

  bool pass = true;

  auto check_suite = [&](const std::string &suite, const Dataset &d) {
    Eigen::Matrix<double, 4, -1> X;
    auto eval = [&](const std::string &method, auto fn) {
      fn(d, X);
      const long long bad = count_nonfinite(X);
      const bool ok = (bad == 0);
      pass = pass && ok;
      std::cout << "suite=" << suite << " method=" << method
                << " nonfinite=" << bad << " status=" << (ok ? "PASS" : "FAIL")
                << std::endl;
    };

    eval("Lott", [&](const Dataset &dd, Eigen::Matrix<double, 4, -1> &Y) {
      run_lott(dd, Y);
    });
    eval("Hartley-Sturm", [&](const Dataset &dd, Eigen::Matrix<double, 4, -1> &Y) {
      run_hs(dd, dd.F, Y);
    });
    eval("Lindstrom niter1", [&](const Dataset &dd, Eigen::Matrix<double, 4, -1> &Y) {
      run_lind1_canonical(dd, dd.F.transpose(), Y);
    });
    eval("Lindstrom niter2", [&](const Dataset &dd, Eigen::Matrix<double, 4, -1> &Y) {
      run_lind2_canonical(dd, dd.F.transpose(), Y);
    });
    eval("Kanatani", [&](const Dataset &dd, Eigen::Matrix<double, 4, -1> &Y) {
      run_kan_canonical(dd, dd.F.transpose(), Y);
    });
  };

  check_suite("general", general);
  check_suite("translation", translation);
  check_suite("epipole_inside", epi_inside);

  std::cout << "T2_status=" << (pass ? "PASS" : "FAIL") << std::endl << std::endl;
  r.pass = pass;
  return r;
}

CheckResult run_known_case_test(std::mt19937 &rng) {
  CheckResult r;
  r.id = "T3_known_case";
  std::cout << "== T3 Known-Case Test ==" << std::endl;
  std::cout << "case: noiseless correspondences (already on epipolar quadric)"
            << std::endl;
  std::cout << "criteria: exact methods require near-machine precision; "
               "approximate methods require bounded small deviation"
            << std::endl;

  const int npts = 128;
  Eigen::Matrix3d R = random_rotation(rng, 10.0);
  Eigen::Vector3d t(0.2, 0.05, 0.3);
  Dataset noiseless = make_dataset(npts, 0.0, rng, R, t);

  bool pass = true;

  auto evaluate = [&](const std::string &method,
                      const Eigen::Matrix<double, 4, -1> &X,
                      const double max_mean_cost, const double max_mean_residual) {
    const double mean_cost = mean_correction_cost(X, noiseless.obs);
    const double mean_res = mean_abs_residual(X, noiseless.F);
    const bool ok = (mean_cost <= max_mean_cost && mean_res <= max_mean_residual);
    pass = pass && ok;
    std::cout << "method=" << method << " mean_cost=" << std::setprecision(17)
              << mean_cost << " mean_abs_epi=" << mean_res << " status="
              << (ok ? "PASS" : "FAIL") << std::endl;
  };

  Eigen::Matrix<double, 4, -1> X;
  run_lott(noiseless, X);
  evaluate("Lott", X, 1e-8, 1e-10);

  run_hs(noiseless, noiseless.F, X);
  evaluate("Hartley-Sturm", X, 1e-8, 1e-10);

  run_lind1_canonical(noiseless, noiseless.F.transpose(), X);
  evaluate("Lindstrom niter1", X, 1e-8, 1e-10);

  run_lind2_canonical(noiseless, noiseless.F.transpose(), X);
  evaluate("Lindstrom niter2", X, 5e-3, 5e-3);

  run_kan_canonical(noiseless, noiseless.F.transpose(), X);
  evaluate("Kanatani", X, 1e-8, 1e-6);

  std::cout << "T3_status=" << (pass ? "PASS" : "FAIL") << std::endl << std::endl;
  r.pass = pass;
  return r;
}

CheckResult run_consistency_test(const Dataset &d) {
  CheckResult r;
  r.id = "T4_consistency";
  std::cout << "== T4 Consistency Test ==" << std::endl;
  std::cout << "criterion: exact baselines enforce strong wrapper separation; "
               "approximate baselines enforce non-regression under documented wrapper"
            << std::endl;

  bool pass = true;
  const double strict_ratio = 0.1;
  Eigen::Matrix<double, 4, -1> X;

  run_hs(d, d.F, X);
  const double hs_doc = mean_abs_residual(X, d.F);
  run_hs(d, d.F.transpose(), X);
  const double hs_alt = mean_abs_residual(X, d.F);
  bool ok = check_ratio(hs_doc, hs_alt, strict_ratio);
  pass = pass && ok;
  std::cout << "method=Hartley-Sturm documented=F mean_doc=" << std::setprecision(17)
            << hs_doc << " mean_alt=" << hs_alt << " status=" << (ok ? "PASS" : "FAIL")
            << std::endl;

  run_lind1_canonical(d, d.F.transpose(), X);
  const double l1_doc = mean_abs_residual(X, d.F);
  run_lind1_canonical(d, d.F, X);
  const double l1_alt = mean_abs_residual(X, d.F);
  ok = check_ratio(l1_doc, l1_alt, strict_ratio);
  pass = pass && ok;
  std::cout << "method=Lindstrom niter1 documented=F^T mean_doc=" << l1_doc
            << " mean_alt=" << l1_alt << " status=" << (ok ? "PASS" : "FAIL")
            << std::endl;

  run_lind2_canonical(d, d.F.transpose(), X);
  const double l2_doc = mean_abs_residual(X, d.F);
  run_lind2_canonical(d, d.F, X);
  const double l2_alt = mean_abs_residual(X, d.F);
  // niter2 is an approximation method; require documented wrapper to not regress materially.
  ok = std::isfinite(l2_doc) && std::isfinite(l2_alt) && (l2_doc <= 1.05 * l2_alt);
  pass = pass && ok;
  std::cout << "method=Lindstrom niter2 documented=F^T mean_doc=" << l2_doc
            << " mean_alt=" << l2_alt << " criterion=doc<=1.05*alt"
            << " status=" << (ok ? "PASS" : "FAIL") << std::endl;

  run_kan_canonical(d, d.F.transpose(), X);
  const double k_doc = mean_abs_residual(X, d.F);
  run_kan_canonical(d, d.F, X);
  const double k_alt = mean_abs_residual(X, d.F);
  ok = std::isfinite(k_doc) && std::isfinite(k_alt) && (k_doc <= 0.7 * k_alt);
  pass = pass && ok;
  std::cout << "method=Kanatani documented=F^T mean_doc=" << k_doc
            << " mean_alt=" << k_alt << " criterion=doc<=0.7*alt"
            << " status=" << (ok ? "PASS" : "FAIL") << std::endl;

  std::cout << "T4_status=" << (pass ? "PASS" : "FAIL") << std::endl << std::endl;
  r.pass = pass;
  return r;
}

} // namespace

int main() {
  std::mt19937 rng(20260214);
  const Dataset general = make_general_dataset(1024, 1.0, rng);
  const Dataset translation = make_translation_dataset(1024, 1.0, rng);
  const Dataset epi_inside = make_epipole_inside_dataset(1024, 1.0, rng);

  std::cout << std::setprecision(17);
  std::cout << "baseline_acceptance_seed=20260214" << std::endl;
  std::cout << "npts_per_suite=1024" << std::endl;
  std::cout << std::endl;

  const CheckResult t1 = run_convention_test(general);
  const CheckResult t2 = run_sanity_test(general, translation, epi_inside);
  const CheckResult t3 = run_known_case_test(rng);
  const CheckResult t4 = run_consistency_test(general);

  const bool all_pass = t1.pass && t2.pass && t3.pass && t4.pass;
  std::cout << "overall_status=" << (all_pass ? "PASS" : "FAIL") << std::endl;

  return all_pass ? 0 : 1;
}
