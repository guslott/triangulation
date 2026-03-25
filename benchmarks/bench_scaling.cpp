#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
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

struct ScalingRow {
  int npts = 0;
  TimingStats total_ns;
  TimingStats ns_per_pt;
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

template <typename Fn>
std::vector<double> benchmark_total_ns(Fn fn, int repeats) {
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(repeats));
  for (int r = 0; r < repeats; ++r) {
    const auto start = std::chrono::high_resolution_clock::now();
    fn();
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    samples.push_back(static_cast<double>(duration.count()));
  }
  return samples;
}

std::vector<double> divide_samples(const std::vector<double> &samples, double denom) {
  std::vector<double> out;
  out.reserve(samples.size());
  for (const double s : samples) {
    out.push_back(s / denom);
  }
  return out;
}

void linear_fit(const std::vector<ScalingRow> &rows, double &slope, double &intercept, double &r2) {
  slope = std::numeric_limits<double>::quiet_NaN();
  intercept = std::numeric_limits<double>::quiet_NaN();
  r2 = std::numeric_limits<double>::quiet_NaN();
  if (rows.size() < 2) {
    return;
  }

  std::vector<double> x;
  std::vector<double> y;
  x.reserve(rows.size());
  y.reserve(rows.size());
  for (const auto &row : rows) {
    x.push_back(static_cast<double>(row.npts));
    y.push_back(row.total_ns.mean);
  }

  const double x_mean = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
  const double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();

  double sxx = 0.0;
  double sxy = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    const double dx = x[i] - x_mean;
    sxx += dx * dx;
    sxy += dx * (y[i] - y_mean);
  }
  if (sxx <= 0.0) {
    return;
  }
  slope = sxy / sxx;
  intercept = y_mean - slope * x_mean;

  double ss_tot = 0.0;
  double ss_res = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    const double yi = y[i];
    const double yp = slope * x[i] + intercept;
    const double d_tot = yi - y_mean;
    const double d_res = yi - yp;
    ss_tot += d_tot * d_tot;
    ss_res += d_res * d_res;
  }
  if (ss_tot <= 0.0) {
    r2 = 1.0;
  } else {
    r2 = 1.0 - ss_res / ss_tot;
  }
}

std::string join_sizes(const std::vector<int> &sizes) {
  std::ostringstream oss;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << sizes[i];
  }
  return oss.str();
}

}  // namespace

int main() {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> t_dist(-1.0, 1.0);
  std::uniform_real_distribution<double> xy_dist(-2.0, 2.0);
  std::uniform_real_distribution<double> z_dist(1.0, 6.0);
  std::normal_distribution<double> noise(0.0, 2.0);

  const std::vector<int> sizes = {500, 1000, 2000, 5000, 10000, 20000};
  const int repeats = 40;
  const int max_npts = sizes.back();

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

  Eigen::Matrix<double, 4, -1> A(4, max_npts);
  for (int i = 0; i < max_npts; ++i) {
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
    A.col(i).head<2>() = xi.head<2>();
    A.col(i).tail<2>() = xpi.head<2>();
  }

  std::vector<ScalingRow> rows;
  rows.reserve(sizes.size());

  for (const int npts : sizes) {
    Eigen::Matrix<double, 4, -1> X_hat(4, npts);
    // Warm-up once per size to avoid first-call effects.
    lott_triangulate(A.leftCols(npts), F, X_hat);

    const auto total_samples =
        benchmark_total_ns([&]() { lott_triangulate(A.leftCols(npts), F, X_hat); }, repeats);
    const auto ns_per_pt_samples = divide_samples(total_samples, static_cast<double>(npts));

    ScalingRow row;
    row.npts = npts;
    row.total_ns = summarize_samples(total_samples);
    row.ns_per_pt = summarize_samples(ns_per_pt_samples);
    rows.push_back(row);
  }

  double fit_slope = std::numeric_limits<double>::quiet_NaN();
  double fit_intercept = std::numeric_limits<double>::quiet_NaN();
  double fit_r2 = std::numeric_limits<double>::quiet_NaN();
  linear_fit(rows, fit_slope, fit_intercept, fit_r2);

  double ns_per_pt_min = std::numeric_limits<double>::infinity();
  double ns_per_pt_max = -std::numeric_limits<double>::infinity();
  for (const auto &row : rows) {
    ns_per_pt_min = std::min(ns_per_pt_min, row.ns_per_pt.mean);
    ns_per_pt_max = std::max(ns_per_pt_max, row.ns_per_pt.mean);
  }
  const double ns_ratio =
      (ns_per_pt_min > 0.0) ? (ns_per_pt_max / ns_per_pt_min) : std::numeric_limits<double>::quiet_NaN();

  std::cout << "repeats=" << repeats << std::endl;
  std::cout << "method=Lott Full" << std::endl;
  std::cout << "sizes=" << join_sizes(sizes) << std::endl;
  for (const auto &row : rows) {
    std::cout << "npts=" << row.npts
              << " total_ns_mean=" << row.total_ns.mean
              << " total_ns_std=" << row.total_ns.stddev
              << " total_ns_ci95=" << row.total_ns.ci95
              << " ns_per_pt_mean=" << row.ns_per_pt.mean
              << " ns_per_pt_std=" << row.ns_per_pt.stddev
              << " ns_per_pt_ci95=" << row.ns_per_pt.ci95
              << std::endl;
  }
  std::cout << "fit_slope_ns_per_pt=" << fit_slope << std::endl;
  std::cout << "fit_intercept_ns=" << fit_intercept << std::endl;
  std::cout << "fit_r2=" << fit_r2 << std::endl;
  std::cout << "ns_per_pt_min=" << ns_per_pt_min << std::endl;
  std::cout << "ns_per_pt_max=" << ns_per_pt_max << std::endl;
  std::cout << "ns_per_pt_ratio_max_over_min=" << ns_ratio << std::endl;

  return 0;
}
