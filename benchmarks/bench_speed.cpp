#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "lott_triangulate.h"
#include "lott_triangulate_certified.h"
#include "so3_utils.h"
#include "triangulate_hs.h"
#include "triangulate_kanatani.h"
#include "triangulate_lindstrom.h"

namespace {

struct TimingStats {
  double mean = std::numeric_limits<double>::quiet_NaN();
  double stddev = std::numeric_limits<double>::quiet_NaN();
  double ci95 = std::numeric_limits<double>::quiet_NaN();
};

void swap_output_packing(const Eigen::Matrix<double, 4, -1> &in,
                         Eigen::Matrix<double, 4, -1> &out) {
  out.resize(4, in.cols());
  out.topRows<2>() = in.bottomRows<2>();
  out.bottomRows<2>() = in.topRows<2>();
}

double xpFx(const Eigen::Matrix<double, 4, 1> &A, const Eigen::Matrix3d &F) {
  Eigen::Vector3d lf = F.block<3, 2>(0, 0) * A.head<2>() + F.block<3, 1>(0, 2);
  return lf.head<2>().dot(A.tail<2>()) + lf(2);
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

template <typename Fn>
std::vector<double> benchmark_ns_per_pt(Fn fn, const int repeats, const int npts) {
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(repeats));
  for (int r = 0; r < repeats; ++r) {
    const auto start = std::chrono::high_resolution_clock::now();
    fn();
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    samples.push_back(static_cast<double>(duration.count()) / npts);
  }
  return samples;
}

TimingStats summarize_samples(const std::vector<double> &samples) {
  TimingStats stats;
  if (samples.empty()) {
    return stats;
  }
  const double mean =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  double var = 0.0;
  for (const double x : samples) {
    const double d = x - mean;
    var += d * d;
  }
  var /= (samples.size() > 1) ? static_cast<double>(samples.size() - 1) : 1.0;
  const double stddev = std::sqrt(var);
  const double ci95 = 1.96 * stddev / std::sqrt(static_cast<double>(samples.size()));

  stats.mean = mean;
  stats.stddev = stddev;
  stats.ci95 = ci95;
  return stats;
}

double mean_abs_epi(const Eigen::Matrix<double, 4, -1> &X_hat,
                    const Eigen::Matrix3d &F, double &finite_ratio) {
  double sum = 0.0;
  int finite_count = 0;
  for (int i = 0; i < X_hat.cols(); ++i) {
    const Eigen::Vector4d p = X_hat.col(i);
    if (!p.allFinite()) {
      continue;
    }
    sum += std::abs(xpFx(p, F));
    ++finite_count;
  }
  finite_ratio = static_cast<double>(finite_count) / X_hat.cols();
  if (finite_count == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return sum / finite_count;
}

void print_timing_block(const std::string &name, const TimingStats &stats) {
  std::cout << name << " ns/pt mean: " << stats.mean << std::endl;
  std::cout << name << " ns/pt std: " << stats.stddev << std::endl;
  std::cout << name << " ns/pt ci95: " << stats.ci95 << std::endl;
}

} // namespace

int main() {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> t_dist(-1.0, 1.0);
  std::uniform_real_distribution<double> xy_dist(-2.0, 2.0);
  std::uniform_real_distribution<double> z_dist(1.0, 6.0);
  std::normal_distribution<double> noise(0.0, 2.0);

  const int npts = 5000;
  const int repeats = 40;
  const double focal_length = 800.0;
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  K(0, 0) = focal_length;
  K(1, 1) = focal_length;

  const Eigen::Matrix3d R = random_rotation(rng, 20.0);
  Eigen::Vector3d t(t_dist(rng), t_dist(rng), t_dist(rng));
  if (t.norm() < 1e-3) {
    t(0) += 0.1;
  }
  Eigen::Matrix3d F = K.inverse().transpose() * skew(t) * R * K.inverse();
  F = F / F.norm();
  const Eigen::Matrix3d F_baseline = F.transpose();

  Eigen::Matrix<double, 3, -1> x(3, npts), xp(3, npts);
  Eigen::Matrix<double, 4, -1> A(4, npts);

  for (int i = 0; i < npts; ++i) {
    const Eigen::Vector3d X(xy_dist(rng), xy_dist(rng), z_dist(rng));
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
    x.col(i) = xi;
    xp.col(i) = xpi;
    A.col(i).head<2>() = xi.head<2>();
    A.col(i).tail<2>() = xpi.head<2>();
  }

  Eigen::Matrix<double, 4, -1> X_hat(4, npts), X_hat_hs(4, npts),
      X_hat_ls1(4, npts), X_hat_ls2(4, npts), X_hat_kt(4, npts),
      X_hat_cf(4, npts);

  // Warm-up for fairer timing.
  lott_triangulate(A, F, X_hat);
  lott_triangulate_certified_fallback(A, F, X_hat_cf);
  hartley_triangulate(x, xp, F, X_hat_hs);
  triangulation::lindstrom_niter1(x, xp, F_baseline, X_hat_ls1);
  triangulation::lindstrom_niter2(x, xp, F_baseline, X_hat_ls2);
  triangulation::kanatani_triangulate(x, xp, F_baseline, X_hat_kt);

  const auto lott_samples =
      benchmark_ns_per_pt([&]() { lott_triangulate(A, F, X_hat); }, repeats, npts);
  const auto hs_samples = benchmark_ns_per_pt(
      [&]() { hartley_triangulate(x, xp, F, X_hat_hs); }, repeats, npts);
  const auto cf_samples = benchmark_ns_per_pt(
      [&]() { lott_triangulate_certified_fallback(A, F, X_hat_cf); }, repeats,
      npts);
  const auto ls1_samples = benchmark_ns_per_pt(
      [&]() { triangulation::lindstrom_niter1(x, xp, F_baseline, X_hat_ls1); }, repeats, npts);
  const auto ls_samples = benchmark_ns_per_pt(
      [&]() { triangulation::lindstrom_niter2(x, xp, F_baseline, X_hat_ls2); }, repeats, npts);
  const auto kt_samples = benchmark_ns_per_pt(
      [&]() { triangulation::kanatani_triangulate(x, xp, F_baseline, X_hat_kt); }, repeats, npts);

  const TimingStats lott_stats = summarize_samples(lott_samples);
  const TimingStats hs_stats = summarize_samples(hs_samples);
  const TimingStats cf_stats = summarize_samples(cf_samples);
  const TimingStats ls1_stats = summarize_samples(ls1_samples);
  const TimingStats ls_stats = summarize_samples(ls_samples);
  const TimingStats kt_stats = summarize_samples(kt_samples);

  // Lindstrom/Kanatani implementations return [x', x]; convert to [x, x']
  // before evaluating the canonical x'^T F x residual.
  Eigen::Matrix<double, 4, -1> X_hat_ls1_canon(4, npts), X_hat_ls2_canon(4, npts),
      X_hat_kt_canon(4, npts);
  swap_output_packing(X_hat_ls1, X_hat_ls1_canon);
  swap_output_packing(X_hat_ls2, X_hat_ls2_canon);
  swap_output_packing(X_hat_kt, X_hat_kt_canon);

  double lott_ratio = 0.0;
  double hs_ratio = 0.0;
  double ls1_ratio = 0.0;
  double ls_ratio = 0.0;
  double kt_ratio = 0.0;
  const double mean_lott = mean_abs_epi(X_hat, F, lott_ratio);
  const double mean_hs = mean_abs_epi(X_hat_hs, F, hs_ratio);
  const double mean_ls1 = mean_abs_epi(X_hat_ls1_canon, F, ls1_ratio);
  const double mean_ls = mean_abs_epi(X_hat_ls2_canon, F, ls_ratio);
  const double mean_kt = mean_abs_epi(X_hat_kt_canon, F, kt_ratio);
  LottSolverDiagnostics cf_solver_diag;
  LottCertifiedFallbackDiagnostics cf_diag;
  lott_triangulate_certified_fallback(A, F, X_hat_cf, &cf_solver_diag, &cf_diag);
  double cf_ratio = 0.0;
  const double mean_cf_res = mean_abs_epi(X_hat_cf, F, cf_ratio);

  std::cout << "repeats=" << repeats << std::endl;
  std::cout << "npts=" << npts << std::endl;
  print_timing_block("Lott triangulation", lott_stats);
  print_timing_block("Lott certified+fallback triangulation", cf_stats);
  print_timing_block("Hartley-Sturm triangulation", hs_stats);
  print_timing_block("Lindstrom niter1 triangulation", ls1_stats);
  print_timing_block("Lindstrom niter2 triangulation", ls_stats);
  print_timing_block("Kanatani triangulation", kt_stats);
  std::cout << "Mean |x'Fx| Lott: " << mean_lott << std::endl;
  std::cout << "Mean |x'Fx| Lott certified+fallback: " << mean_cf_res << std::endl;
  std::cout << "Mean |x'Fx| HS: " << mean_hs << std::endl;
  std::cout << "Mean |x'Fx| Lindstrom niter1: " << mean_ls1 << std::endl;
  std::cout << "Mean |x'Fx| Lindstrom niter2: " << mean_ls << std::endl;
  std::cout << "Mean |x'Fx| Kanatani: " << mean_kt << std::endl;
  std::cout << "Convention note Lindstrom/Kanatani: wrappers use F^T for internal x^TFx' convention; output [x',x], residual evaluated on swapped canonical packing [x,x']"
            << std::endl;
  std::cout << "Finite ratio Lott: " << lott_ratio << std::endl;
  std::cout << "Finite ratio Lott certified+fallback: " << cf_ratio << std::endl;
  std::cout << "Finite ratio HS: " << hs_ratio << std::endl;
  std::cout << "Finite ratio Lindstrom niter1: " << ls1_ratio << std::endl;
  std::cout << "Finite ratio Lindstrom niter2: " << ls_ratio << std::endl;
  std::cout << "Finite ratio Kanatani: " << kt_ratio << std::endl;
  std::cout << "certified_eq1_points=" << cf_diag.cert_eq1_points << std::endl;
  std::cout << "certified_fallback_points=" << cf_diag.fallback_points << std::endl;
  std::cout << "certified_fallback_nonunique_points=" << cf_diag.fallback_nonunique_points << std::endl;
  std::cout << "certified_fallback_cert_failure_points=" << cf_diag.fallback_cert_failure_points << std::endl;
  std::cout << "certified_solver_cert_failures=" << cf_solver_diag.cert_failures
            << std::endl;
  return 0;
}
