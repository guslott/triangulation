#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "lott_triangulate.h"
#include "so3_utils.h"

namespace {

struct TimingStats {
  double mean = std::numeric_limits<double>::quiet_NaN();
  double stddev = std::numeric_limits<double>::quiet_NaN();
  double ci95 = std::numeric_limits<double>::quiet_NaN();
};

struct MethodConfig {
  std::string method;
  int root_solver_mode = 0; // 0=full, 1..4=Householder order
};

struct MethodStats {
  std::string method;
  int root_solver_mode = 0;
  TimingStats speed;
  double speed_mean_abs_epi = std::numeric_limits<double>::quiet_NaN();
  double speed_finite_ratio = std::numeric_limits<double>::quiet_NaN();
  long long total_points = 0;
  long long finite_points = 0;
  double deltaE_mean = std::numeric_limits<double>::quiet_NaN();
  double deltaE_abs_p50 = std::numeric_limits<double>::quiet_NaN();
  double deltaE_abs_p95 = std::numeric_limits<double>::quiet_NaN();
  double deltaE_abs_p99 = std::numeric_limits<double>::quiet_NaN();
  double deltaE_abs_max = std::numeric_limits<double>::quiet_NaN();
  double abs_epi_p95 = std::numeric_limits<double>::quiet_NaN();
  double abs_epi_p99 = std::numeric_limits<double>::quiet_NaN();
};

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

inline double epipolar_residual(const Eigen::Vector4d &A,
                                const Eigen::Matrix3d &F) {
  Eigen::Vector3d lf = F.block<3, 2>(0, 0) * A.head<2>() + F.block<3, 1>(0, 2);
  return std::abs(lf.head<2>().dot(A.tail<2>()) + lf(2));
}

inline double reprojection_cost(const Eigen::Vector4d &corr,
                                const Eigen::Vector4d &obs) {
  return (corr.head<2>() - obs.head<2>()).squaredNorm() +
         (corr.tail<2>() - obs.tail<2>()).squaredNorm();
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

template <typename Fn>
std::vector<double> benchmark_ns_per_pt(Fn fn, const int repeats, const int npts) {
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(repeats));
  for (int r = 0; r < repeats; ++r) {
    const auto start = std::chrono::high_resolution_clock::now();
    fn();
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
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
    sum += epipolar_residual(p, F);
    ++finite_count;
  }
  finite_ratio = static_cast<double>(finite_count) / X_hat.cols();
  if (finite_count == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return sum / finite_count;
}

} // namespace

int main() {
  const std::vector<MethodConfig> methods = {
      {"Lott Full", 0},
      {"Householder H1", 1},
      {"Householder H2", 2},
      {"Householder H3", 3},
      {"Householder H4", 4},
  };

  // Speed benchmark setup (single deterministic scene).
  std::mt19937 speed_rng(12345);
  std::uniform_real_distribution<double> t_dist(-1.0, 1.0);
  std::uniform_real_distribution<double> xy_dist(-2.0, 2.0);
  std::uniform_real_distribution<double> z_dist(1.0, 6.0);
  std::normal_distribution<double> noise(0.0, 2.0);
  const int npts_speed = 5000;
  const int repeats = 40;
  const double focal_length = 800.0;

  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  K(0, 0) = focal_length;
  K(1, 1) = focal_length;

  const Eigen::Matrix3d R_speed = random_rotation(speed_rng, 20.0);
  Eigen::Vector3d t_speed(t_dist(speed_rng), t_dist(speed_rng), t_dist(speed_rng));
  if (t_speed.norm() < 1e-3) {
    t_speed(0) += 0.1;
  }
  Eigen::Matrix3d F_speed = K.inverse().transpose() * skew(t_speed) * R_speed * K.inverse();
  F_speed /= F_speed.norm();

  Eigen::Matrix<double, 4, -1> A_speed(4, npts_speed);
  for (int i = 0; i < npts_speed; ++i) {
    const Eigen::Vector3d X(xy_dist(speed_rng), xy_dist(speed_rng), z_dist(speed_rng));
    Eigen::Vector3d Xp = R_speed * X + t_speed;
    if (Xp(2) < 0.1) {
      Xp(2) = 0.1;
    }
    Eigen::Vector3d xi = K * X / X(2);
    Eigen::Vector3d xpi = K * Xp / Xp(2);
    xi(0) += noise(speed_rng);
    xi(1) += noise(speed_rng);
    xpi(0) += noise(speed_rng);
    xpi(1) += noise(speed_rng);
    A_speed.col(i).head<2>() = xi.head<2>();
    A_speed.col(i).tail<2>() = xpi.head<2>();
  }

  std::vector<MethodStats> stats;
  stats.reserve(methods.size());
  for (const auto &m : methods) {
    MethodStats s;
    s.method = m.method;
    s.root_solver_mode = m.root_solver_mode;

    Eigen::Matrix<double, 4, -1> X_hat(4, npts_speed);
    lott_triangulate(A_speed, F_speed, X_hat, nullptr, false, m.root_solver_mode); // warmup

    const auto samples = benchmark_ns_per_pt(
        [&]() { lott_triangulate(A_speed, F_speed, X_hat, nullptr, false, m.root_solver_mode); },
        repeats, npts_speed);
    s.speed = summarize_samples(samples);
    s.speed_mean_abs_epi = mean_abs_epi(X_hat, F_speed, s.speed_finite_ratio);
    stats.push_back(s);
  }

  // Accuracy ablation setup (broad Monte Carlo).
  std::mt19937 acc_rng(54321);
  std::uniform_real_distribution<double> focal_dist(200.0, 1200.0);
  std::uniform_real_distribution<double> t_acc_dist(-1.0, 1.0);
  std::uniform_real_distribution<double> xy_acc_dist(-2.0, 2.0);
  std::uniform_real_distribution<double> z_acc_dist(1.0, 6.0);
  std::uniform_real_distribution<double> sigma_dist(0.1, 2.0);

  const int trials = 200;
  const int npts_per_trial = 200;

  std::vector<std::vector<double>> delta_abs(methods.size());
  std::vector<std::vector<double>> abs_epi(methods.size());
  std::vector<double> delta_sum(methods.size(), 0.0);
  std::vector<long long> finite_points(methods.size(), 0);
  std::vector<long long> total_points(methods.size(), 0);

  for (int tr = 0; tr < trials; ++tr) {
    const Eigen::Matrix3d R = random_rotation(acc_rng, 30.0);
    Eigen::Vector3d t(t_acc_dist(acc_rng), t_acc_dist(acc_rng), t_acc_dist(acc_rng));
    if (t.norm() < 1e-3) {
      t(0) += 0.1;
    }

    const double f = focal_dist(acc_rng);
    Eigen::Matrix3d Kt = Eigen::Matrix3d::Identity();
    Kt(0, 0) = f;
    Kt(1, 1) = f;
    Eigen::Matrix3d F = Kt.inverse().transpose() * skew(t) * R * Kt.inverse();
    F /= F.norm();

    const double sigma = sigma_dist(acc_rng);
    std::normal_distribution<double> trial_noise(0.0, sigma);

    Eigen::Matrix<double, 4, -1> Aobs(4, npts_per_trial);
    for (int i = 0; i < npts_per_trial; ++i) {
      Eigen::Vector3d X(xy_acc_dist(acc_rng), xy_acc_dist(acc_rng), z_acc_dist(acc_rng));
      Eigen::Vector3d Xp = R * X + t;
      if (Xp(2) < 0.1) {
        Xp(2) = 0.1;
      }
      Eigen::Vector3d xi = Kt * X / X(2);
      Eigen::Vector3d xpi = Kt * Xp / Xp(2);
      xi(0) += trial_noise(acc_rng);
      xi(1) += trial_noise(acc_rng);
      xpi(0) += trial_noise(acc_rng);
      xpi(1) += trial_noise(acc_rng);
      Aobs.col(i).head<2>() = xi.head<2>();
      Aobs.col(i).tail<2>() = xpi.head<2>();
    }

    Eigen::Matrix<double, 4, -1> X_full(4, npts_per_trial);
    lott_triangulate(Aobs, F, X_full, nullptr, false, 0);

    std::vector<Eigen::Matrix<double, 4, -1>> X_methods(methods.size());
    for (size_t k = 0; k < methods.size(); ++k) {
      X_methods[k].resize(4, npts_per_trial);
      lott_triangulate(Aobs, F, X_methods[k], nullptr, false,
                       methods[k].root_solver_mode);
    }

    for (int i = 0; i < npts_per_trial; ++i) {
      const Eigen::Vector4d obs = Aobs.col(i);
      const Eigen::Vector4d full = X_full.col(i);
      const bool full_finite = full.allFinite();
      const double e_full = full_finite ? reprojection_cost(full, obs)
                                        : std::numeric_limits<double>::quiet_NaN();
      for (size_t k = 0; k < methods.size(); ++k) {
        ++total_points[k];
        const Eigen::Vector4d est = X_methods[k].col(i);
        if (!full_finite || !est.allFinite()) {
          continue;
        }
        const double e_est = reprojection_cost(est, obs);
        const double delta = e_est - e_full;
        delta_sum[k] += delta;
        delta_abs[k].push_back(std::abs(delta));
        abs_epi[k].push_back(epipolar_residual(est, F));
        ++finite_points[k];
      }
    }
  }

  for (size_t k = 0; k < methods.size(); ++k) {
    MethodStats &s = stats[k];
    s.total_points = total_points[k];
    s.finite_points = finite_points[k];
    if (finite_points[k] > 0) {
      s.deltaE_mean = delta_sum[k] / static_cast<double>(finite_points[k]);
      s.deltaE_abs_p50 = quantile(delta_abs[k], 0.50);
      s.deltaE_abs_p95 = quantile(delta_abs[k], 0.95);
      s.deltaE_abs_p99 = quantile(delta_abs[k], 0.99);
      s.deltaE_abs_max = quantile(delta_abs[k], 1.00);
      s.abs_epi_p95 = quantile(abs_epi[k], 0.95);
      s.abs_epi_p99 = quantile(abs_epi[k], 0.99);
    }
  }

  std::cout << "repeats=" << repeats << std::endl;
  std::cout << "npts_speed=" << npts_speed << std::endl;
  std::cout << "trials=" << trials << std::endl;
  std::cout << "npts_per_trial=" << npts_per_trial << std::endl;
  std::cout << "total_points=" << (trials * npts_per_trial) << std::endl;
  std::cout << std::endl;

  for (const auto &s : stats) {
    std::cout << "method=" << s.method << std::endl;
    std::cout << "root_solver_mode=" << s.root_solver_mode << std::endl;
    std::cout << "speed_ns_per_pt_mean=" << s.speed.mean << std::endl;
    std::cout << "speed_ns_per_pt_std=" << s.speed.stddev << std::endl;
    std::cout << "speed_ns_per_pt_ci95=" << s.speed.ci95 << std::endl;
    std::cout << "speed_mean_abs_epi=" << s.speed_mean_abs_epi << std::endl;
    std::cout << "speed_finite_ratio=" << s.speed_finite_ratio << std::endl;
    std::cout << "points_total=" << s.total_points << std::endl;
    std::cout << "points_finite=" << s.finite_points << std::endl;
    std::cout << "deltaE_mean=" << s.deltaE_mean << std::endl;
    std::cout << "deltaE_abs_p50=" << s.deltaE_abs_p50 << std::endl;
    std::cout << "deltaE_abs_p95=" << s.deltaE_abs_p95 << std::endl;
    std::cout << "deltaE_abs_p99=" << s.deltaE_abs_p99 << std::endl;
    std::cout << "deltaE_abs_max=" << s.deltaE_abs_max << std::endl;
    std::cout << "abs_epi_p95=" << s.abs_epi_p95 << std::endl;
    std::cout << "abs_epi_p99=" << s.abs_epi_p99 << std::endl;
    std::cout << std::endl;
  }

  return 0;
}
