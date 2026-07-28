#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
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

enum class TimingMethod : std::size_t {
  Lott = 0,
  LottCertifiedFallback,
  HartleySturm,
  LindstromNiter1,
  LindstromNiter2,
  Kanatani,
  Count
};

constexpr std::size_t kTimingMethodCount =
    static_cast<std::size_t>(TimingMethod::Count);

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
  const int warmup_rounds = 8;
  constexpr std::uint32_t timing_order_seed = 0x6c6f7474U;
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

  const auto run_method = [&](const TimingMethod method) {
    switch (method) {
    case TimingMethod::Lott:
      lott_triangulate(A, F, X_hat);
      break;
    case TimingMethod::LottCertifiedFallback:
      lott_triangulate_certified_fallback(A, F, X_hat_cf);
      break;
    case TimingMethod::HartleySturm:
      hartley_triangulate(x, xp, F, X_hat_hs);
      break;
    case TimingMethod::LindstromNiter1:
      triangulation::lindstrom_niter1(x, xp, F_baseline, X_hat_ls1);
      break;
    case TimingMethod::LindstromNiter2:
      triangulation::lindstrom_niter2(x, xp, F_baseline, X_hat_ls2);
      break;
    case TimingMethod::Kanatani:
      triangulation::kanatani_triangulate(x, xp, F_baseline, X_hat_kt);
      break;
    case TimingMethod::Count:
      break;
    }
  };

  const auto run_interleaved = [&](const int rounds, std::mt19937 &order_rng,
                                   std::array<std::vector<double>,
                                              kTimingMethodCount> *samples) {
    std::array<TimingMethod, kTimingMethodCount> base_order{
        TimingMethod::Lott, TimingMethod::LottCertifiedFallback,
        TimingMethod::HartleySturm, TimingMethod::LindstromNiter1,
        TimingMethod::LindstromNiter2, TimingMethod::Kanatani};

    for (int round = 0; round < rounds; ++round) {
      const std::size_t rotation =
          static_cast<std::size_t>(round) % kTimingMethodCount;
      if (rotation == 0) {
        std::shuffle(base_order.begin(), base_order.end(), order_rng);
      }

      for (std::size_t position = 0; position < kTimingMethodCount;
           ++position) {
        const TimingMethod method =
            base_order[(position + rotation) % kTimingMethodCount];
        if (samples == nullptr) {
          run_method(method);
          continue;
        }

        const auto start = std::chrono::steady_clock::now();
        run_method(method);
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ns =
            std::chrono::duration<double, std::nano>(end - start).count();
        (*samples)[static_cast<std::size_t>(method)].push_back(elapsed_ns /
                                                               npts);
      }
    }
  };

  // Warm every implementation and its output allocation repeatedly before any
  // samples are collected. Use a separate deterministic schedule so the timed
  // order is reproducible solely from timing_order_seed.
  std::mt19937 warmup_order_rng(timing_order_seed ^ 0x9e3779b9U);
  run_interleaved(warmup_rounds, warmup_order_rng, nullptr);

  std::array<std::vector<double>, kTimingMethodCount> timing_samples;
  for (auto &method_samples : timing_samples) {
    method_samples.reserve(static_cast<std::size_t>(repeats));
  }
  std::mt19937 timing_order_rng(timing_order_seed);
  run_interleaved(repeats, timing_order_rng, &timing_samples);

  const auto &lott_samples =
      timing_samples[static_cast<std::size_t>(TimingMethod::Lott)];
  const auto &cf_samples = timing_samples[static_cast<std::size_t>(
      TimingMethod::LottCertifiedFallback)];
  const auto &hs_samples =
      timing_samples[static_cast<std::size_t>(TimingMethod::HartleySturm)];
  const auto &ls1_samples = timing_samples[static_cast<std::size_t>(
      TimingMethod::LindstromNiter1)];
  const auto &ls_samples = timing_samples[static_cast<std::size_t>(
      TimingMethod::LindstromNiter2)];
  const auto &kt_samples =
      timing_samples[static_cast<std::size_t>(TimingMethod::Kanatani)];

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
  std::cout << "timing_protocol=seeded_shuffle_balanced_rotation" << std::endl;
  std::cout << "timing_order_seed=" << timing_order_seed << std::endl;
  std::cout << "timing_warmup_rounds=" << warmup_rounds << std::endl;
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
  std::cout << "certified_nonunique_points="
            << cf_diag.certified_nonunique_points << std::endl;
  std::cout << "certified_fallback_points=" << cf_diag.fallback_points << std::endl;
  std::cout << "certified_fallback_nonunique_points=" << cf_diag.fallback_nonunique_points << std::endl;
  std::cout << "certified_fallback_cert_failure_points=" << cf_diag.fallback_cert_failure_points << std::endl;
  std::cout << "certified_solver_cert_failures=" << cf_solver_diag.cert_failures
            << std::endl;
  return 0;
}
