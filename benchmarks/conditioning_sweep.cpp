#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "lott_triangulate.h"

namespace {

using LD4 = std::array<long double, 4>;

constexpr long double kNaN =
    std::numeric_limits<long double>::quiet_NaN();

enum class ExpectedPath { kRegular, kBoundaryNonunique, kAffine };

struct SweepCase {
  std::string suite;
  std::string parameter;
  long double parameter_value = kNaN;
  int replicate = 0;
  long double a = 0.0L;
  long double b = 0.0L;
  LD4 q{};
  long double g = 0.0L;
  long double target_mu = kNaN;
  ExpectedPath expected = ExpectedPath::kRegular;
};

struct CaseMetrics {
  SweepCase input;
  LottCertifiedPointResult result;
  int selected_chart = 0;
  long double coefficient_scale = kNaN;
  long double b_over_a = kNaN;
  long double qn_ratio = kNaN;
  long double pivot_ratio = kNaN;
  long double lambda_bar = kNaN;
  long double mu = kNaN;
  long double endpoint_margin = kNaN;
  long double normalized_phi = kNaN;
  long double normalized_phi_derivative = kNaN;
  long double residual_root_condition = kNaN;
  long double residual_root_bound = kNaN;
  long double residual_root_bound_ratio = kNaN;
  long double residual_implied_lambda_error_bound = kNaN;
  long double lambda_bar_absolute_error = kNaN;
  long double hessian_condition = kNaN;
  long double reconstruction_condition = kNaN;
  long double reconstruction_condition_bound = kNaN;
  long double normalized_feasibility_residual = kNaN;
  long double normalized_kkt_residual = kNaN;
  long double normalized_rank_two_identity_residual = kNaN;
  long double relative_reconstruction_error = kNaN;
  long double lambda_bar_relative_error = kNaN;
  bool certified = false;
  bool fallback_required = false;
  bool passed = false;
};

struct BinStats {
  std::string suite;
  std::string parameter;
  long double parameter_value = kNaN;
  int cases = 0;
  int certified = 0;
  int fallback_required = 0;
  int regular = 0;
  int boundary = 0;
  int affine = 0;
  int passed = 0;
  long long iterations = 0;
  int max_iterations = 0;
  long long bisections = 0;
  int max_bisections = 0;
  long double max_feasibility_residual = 0.0L;
  long double max_kkt_residual = 0.0L;
  long double max_rank_two_identity_residual = 0.0L;
  long double min_endpoint_margin = std::numeric_limits<long double>::infinity();
  long double max_residual_root_condition = 0.0L;
  long double max_residual_root_bound_ratio = 0.0L;
  long double max_hessian_condition = 0.0L;
  long double max_reconstruction_error = 0.0L;
  long double max_lambda_error = 0.0L;
  long double max_lambda_relative_error = 0.0L;
};

std::string status_name(const LottPointStatus status) {
  switch (status) {
    case LOTT_STATUS_UNSET:
      return "unset";
    case LOTT_STATUS_ALREADY_FEASIBLE:
      return "already_feasible";
    case LOTT_STATUS_AFFINE:
      return "affine";
    case LOTT_STATUS_REGULAR_INTERIOR:
      return "regular_interior";
    case LOTT_STATUS_BOUNDARY_PSD_UNIQUE:
      return "boundary_psd_unique";
    case LOTT_STATUS_BOUNDARY_PSD_NONUNIQUE:
      return "boundary_psd_nonunique";
    case LOTT_STATUS_UNCERTIFIED_APPROXIMATE:
      return "uncertified_approximate";
    case LOTT_STATUS_FAILED_INVALID_INPUT:
      return "failed_invalid_input";
    case LOTT_STATUS_FAILED_BRACKET:
      return "failed_bracket";
    case LOTT_STATUS_FAILED_CERTIFICATE:
      return "failed_certificate";
  }
  return "unknown";
}

std::string expected_name(const ExpectedPath path) {
  switch (path) {
    case ExpectedPath::kRegular:
      return "regular_interior";
    case ExpectedPath::kBoundaryNonunique:
      return "boundary_psd_nonunique";
    case ExpectedPath::kAffine:
      return "affine";
  }
  return "unknown";
}

long double norm2(const LD4 &x) {
  long double value = 0.0L;
  for (const long double xi : x) {
    value += xi * xi;
  }
  return std::sqrt(value);
}

long double max_abs(const LD4 &x) {
  long double value = 0.0L;
  for (const long double xi : x) {
    value = std::max(value, std::abs(xi));
  }
  return value;
}

int largest_component(const LD4 &q) {
  int index = 0;
  long double magnitude = std::abs(q[0]);
  for (int i = 1; i < 4; ++i) {
    if (std::abs(q[static_cast<size_t>(i)]) > magnitude) {
      index = i;
      magnitude = std::abs(q[static_cast<size_t>(i)]);
    }
  }
  return index;
}

long double finite_max(const long double lhs, const long double rhs) {
  if (std::isnan(lhs)) {
    return rhs;
  }
  if (std::isnan(rhs)) {
    return lhs;
  }
  return std::max(lhs, rhs);
}

std::string number(const long double value) {
  if (std::isnan(value)) {
    return "nan";
  }
  if (std::isinf(value)) {
    return value > 0.0L ? "inf" : "-inf";
  }
  std::ostringstream stream;
  stream << std::scientific << std::setprecision(17) << value;
  return stream.str();
}

// For a positive singular-value pair d and a prescribed
// mu=2*a*lambda, choosing
//   q_minus/q_plus = (1-(d/a)mu)/(1+(d/a)mu)
// makes that pair satisfy both the rank-two identity and the secular equation
// at the prescribed multiplier.  The b=0 expression below is its continuous
// limit and also obeys the exact b=0 rank-two condition q_1^2=q_3^2.
void add_rank_two_family(std::vector<SweepCase> &cases,
                         const std::string &suite,
                         const std::string &parameter,
                         const long double parameter_value,
                         const long double a, const long double ratio,
                         const long double mu, const long double common_scale,
                         const ExpectedPath expected) {
  constexpr std::array<std::array<long double, 2>, 4> amplitudes = {{
      {1.0L, 0.75L},
      {0.8L, -1.1L},
      {-0.6L, 0.9L},
      {1.25L, -0.4L},
  }};
  for (int replicate = 0; replicate < static_cast<int>(amplitudes.size());
       ++replicate) {
    const long double q0 = amplitudes[static_cast<size_t>(replicate)][0];
    const long double q1 = amplitudes[static_cast<size_t>(replicate)][1];
    const long double q2 = q0 * (1.0L - mu) / (1.0L + mu);
    const long double q3 =
        q1 * (1.0L - ratio * mu) / (1.0L + ratio * mu);
    // Stable form of (q0^2-q2^2)/a + (q1^2-q3^2)/b.
    const long double g =
        4.0L * mu / a *
        (q0 * q0 / ((1.0L + mu) * (1.0L + mu)) +
         q1 * q1 /
             ((1.0L + ratio * mu) * (1.0L + ratio * mu)));
    SweepCase item;
    item.suite = suite;
    item.parameter = parameter;
    item.parameter_value = parameter_value;
    item.replicate = replicate;
    item.a = common_scale * a;
    item.b = common_scale * a * ratio;
    item.q = {common_scale * q0, common_scale * q1, common_scale * q2,
              common_scale * q3};
    item.g = common_scale * g;
    item.target_mu = mu;
    item.expected = expected;
    cases.push_back(item);
  }
}

std::vector<SweepCase> make_cases() {
  std::vector<SweepCase> cases;

  const std::array<long double, 11> b_ratios = {
      0.0L,  1e-12L, 1e-10L, 1e-8L, 1e-6L, 1e-4L,
      1e-2L, 1e-1L,  5e-1L,  9e-1L, 1.0L};
  for (const long double ratio : b_ratios) {
    add_rank_two_family(cases, "b_over_a", "b_over_a", ratio, 1.0L,
                        ratio, 0.6L, 1.0L, ExpectedPath::kRegular);
  }

  const std::array<long double, 8> equality_gaps = {
      1e-2L, 1e-4L, 1e-6L, 1e-8L,
      1e-10L, 1e-12L, 1e-14L, 0.0L};
  for (const long double gap : equality_gaps) {
    add_rank_two_family(cases, "near_equal_singular_values",
                        "one_minus_b_over_a", gap, 1.0L, 1.0L - gap,
                        0.6L, 1.0L, ExpectedPath::kRegular);
  }

  const std::array<long double, 7> a_values = {
      1.0L, 1e-2L, 1e-4L, 1e-6L, 1e-8L, 1e-10L, 1e-12L};
  for (const long double a : a_values) {
    add_rank_two_family(cases, "a_to_affine", "a_raw", a, a, 0.5L,
                        0.6L, 1.0L, ExpectedPath::kRegular);
  }
  constexpr std::array<std::array<long double, 4>, 4> affine_q = {{
      {1.0L, 0.75L, 0.25L, -0.375L},
      {0.8L, -1.1L, -0.2L, -0.55L},
      {-0.6L, 0.9L, 0.15L, 0.45L},
      {1.25L, -0.4L, -0.3125L, -0.2L},
  }};
  for (int replicate = 0; replicate < static_cast<int>(affine_q.size());
       ++replicate) {
    SweepCase item;
    item.suite = "a_to_affine";
    item.parameter = "a_raw";
    item.parameter_value = 0.0L;
    item.replicate = replicate;
    item.a = 0.0L;
    item.b = 0.0L;
    item.q = affine_q[static_cast<size_t>(replicate)];
    item.g = 1.0L;
    item.expected = ExpectedPath::kAffine;
    cases.push_back(item);
  }

  const std::array<long double, 9> endpoint_margins = {
      5e-1L, 1e-1L, 1e-2L, 1e-4L, 1e-6L,
      1e-8L, 1e-10L, 1e-12L, 0.0L};
  for (const long double tau : endpoint_margins) {
    const ExpectedPath expected = tau > 0.0L
                                      ? ExpectedPath::kRegular
                                      : ExpectedPath::kBoundaryNonunique;
    add_rank_two_family(cases, "endpoint_margin", "one_minus_mu", tau,
                        1.0L, 0.5L, 1.0L - tau, 1.0L, expected);
  }

  const std::array<long double, 11> common_scales = {
      1e-200L, 1e-150L, 1e-100L, 1e-50L, 1e-20L, 1.0L,
      1e20L,   1e50L,   1e100L,  1e150L, 1e200L};
  for (const long double scale : common_scales) {
    add_rank_two_family(cases, "common_coefficient_scale", "scale", scale,
                        1.0L, 0.5L, 0.6L, scale,
                        ExpectedPath::kRegular);
  }

  return cases;
}

CaseMetrics evaluate_case(const SweepCase &input) {
  CaseMetrics metrics;
  metrics.input = input;

  const double a = static_cast<double>(input.a);
  const double b = static_cast<double>(input.b);
  const double c = static_cast<double>(input.q[0]);
  const double d = static_cast<double>(input.q[1]);
  const double e = static_cast<double>(input.q[2]);
  const double f = static_cast<double>(input.q[3]);
  const double g = static_cast<double>(input.g);
  const long double actual_a = static_cast<long double>(a);
  const long double actual_b = static_cast<long double>(b);
  const LD4 actual_q = {static_cast<long double>(c),
                        static_cast<long double>(d),
                        static_cast<long double>(e),
                        static_cast<long double>(f)};
  const long double actual_g = static_cast<long double>(g);
  metrics.selected_chart = largest_component(actual_q);
  metrics.result = lott_solve_certified_point(
      a, b, c, d, e, f, g, metrics.selected_chart);

  metrics.coefficient_scale =
      std::max({std::abs(actual_a), std::abs(actual_b), max_abs(actual_q),
                std::abs(actual_g)});
  const long double scale = metrics.coefficient_scale;
  const long double abar = actual_a / scale;
  const long double bbar = actual_b / scale;
  LD4 qbar{};
  for (int j = 0; j < 4; ++j) {
    qbar[static_cast<size_t>(j)] = actual_q[static_cast<size_t>(j)] / scale;
  }
  const long double gbar = actual_g / scale;
  const long double qnorm = norm2(qbar);
  metrics.b_over_a = actual_a > 0.0L ? actual_b / actual_a : kNaN;
  metrics.pivot_ratio = max_abs(qbar) / qnorm;
  if (actual_a > 0.0L) {
    const bool equal_singular = actual_a == actual_b;
    const long double qn2 = qbar[2] * qbar[2] +
                            (equal_singular ? qbar[3] * qbar[3] : 0.0L);
    metrics.qn_ratio = std::sqrt(qn2) / qnorm;
  }

  metrics.certified = lott_status_is_certified(metrics.result.status);
  metrics.fallback_required = !metrics.certified;

  const long double rank_two_identity =
      abar * bbar * gbar -
      bbar * (qbar[0] * qbar[0] - qbar[2] * qbar[2]) -
      abar * (qbar[1] * qbar[1] - qbar[3] * qbar[3]);
  const long double rank_two_scale =
      std::max(1.0L, std::abs(abar * bbar * gbar) +
                         std::abs(bbar * (qbar[0] * qbar[0] -
                                          qbar[2] * qbar[2])) +
                         std::abs(abar * (qbar[1] * qbar[1] -
                                          qbar[3] * qbar[3])));
  metrics.normalized_rank_two_identity_residual =
      std::abs(rank_two_identity) / rank_two_scale;
  if (std::isfinite(metrics.result.root.multiplier)) {
    metrics.lambda_bar =
        scale * static_cast<long double>(metrics.result.root.multiplier);
  }

  LD4 u{};
  for (int j = 0; j < 4; ++j) {
    u[static_cast<size_t>(j)] =
        static_cast<long double>(metrics.result.correction(j));
  }
  const bool finite_u = std::all_of(u.begin(), u.end(), [](long double value) {
    return std::isfinite(value);
  });

  if (finite_u && std::isfinite(metrics.lambda_bar)) {
    const LD4 delta = {abar, bbar, -abar, -bbar};
    long double constraint = gbar;
    long double constraint_scale = std::max(1.0L, std::abs(gbar));
    long double kkt2 = 0.0L;
    long double kkt_scale2 = 1.0L;
    for (int j = 0; j < 4; ++j) {
      const size_t k = static_cast<size_t>(j);
      const long double quadratic = delta[k] * u[k] * u[k];
      const long double linear = 2.0L * qbar[k] * u[k];
      constraint += quadratic + linear;
      constraint_scale += std::abs(quadratic) + std::abs(linear);
      const long double gradient = delta[k] * u[k] + qbar[k];
      const long double residual =
          u[k] + 2.0L * metrics.lambda_bar * gradient;
      const long double local_scale =
          std::abs(u[k]) +
          2.0L * std::abs(metrics.lambda_bar) * std::abs(gradient);
      kkt2 += residual * residual;
      kkt_scale2 += local_scale * local_scale;
    }
    metrics.normalized_feasibility_residual =
        std::abs(constraint) / constraint_scale;
    metrics.normalized_kkt_residual = std::sqrt(kkt2 / kkt_scale2);
  }

  if (actual_a > 0.0L && std::isfinite(metrics.lambda_bar)) {
    const LD4 delta = {abar, bbar, -abar, -bbar};
    metrics.mu = 2.0L * abar * metrics.lambda_bar;
    std::array<long double, 4> hdiag{};
    long double reduction = 0.0L;
    long double derivative_sum = 0.0L;
    long double reconstruction_sum = 0.0L;
    long double hmin = std::numeric_limits<long double>::infinity();
    long double hmax = 0.0L;
    for (int j = 0; j < 4; ++j) {
      const size_t k = static_cast<size_t>(j);
      hdiag[k] = 1.0L + 2.0L * delta[k] * metrics.lambda_bar;
      hmin = std::min(hmin, hdiag[k]);
      hmax = std::max(hmax, hdiag[k]);
      if (hdiag[k] > 0.0L) {
        const long double q2 = qbar[k] * qbar[k];
        reduction += 4.0L * metrics.lambda_bar * q2 *
                     (1.0L + delta[k] * metrics.lambda_bar) /
                     (hdiag[k] * hdiag[k]);
        derivative_sum += q2 / (hdiag[k] * hdiag[k] * hdiag[k]);
        reconstruction_sum +=
            q2 / (hdiag[k] * hdiag[k] * hdiag[k] * hdiag[k]);
      }
    }
    const bool regular =
        metrics.result.status == LOTT_STATUS_REGULAR_INTERIOR;
    metrics.endpoint_margin = regular ? hmin : 0.0L;
    metrics.hessian_condition =
        regular && hmin > 0.0L
            ? hmax / hmin
            : std::numeric_limits<long double>::infinity();
    if (regular && hmin > 0.0L) {
      metrics.normalized_phi = gbar - reduction;
      metrics.normalized_phi_derivative = -4.0L * derivative_sum;
      metrics.residual_root_condition =
          1.0L / std::abs(metrics.normalized_phi_derivative);
      metrics.residual_root_bound = 2.0L / (qnorm * qnorm);
      metrics.residual_root_bound_ratio =
          metrics.residual_root_condition / metrics.residual_root_bound;
      metrics.residual_implied_lambda_error_bound =
          metrics.residual_root_bound * std::abs(metrics.normalized_phi);
      metrics.reconstruction_condition = 2.0L * std::sqrt(reconstruction_sum);
      metrics.reconstruction_condition_bound =
          2.0L * qnorm / (hmin * hmin);
    }
    if (std::isfinite(input.target_mu)) {
      const long double target_lambda_bar =
          scale * input.target_mu / (2.0L * actual_a);
      metrics.lambda_bar_absolute_error =
          std::abs(metrics.lambda_bar - target_lambda_bar);
      metrics.lambda_bar_relative_error =
          metrics.lambda_bar_absolute_error /
          std::max(1.0L, std::abs(target_lambda_bar));
      if (finite_u && input.target_mu < 1.0L) {
        LD4 expected_u{};
        const long double ratio = actual_b / actual_a;
        const LD4 target_h = {1.0L + input.target_mu,
                              1.0L + ratio * input.target_mu,
                              1.0L - input.target_mu,
                              1.0L - ratio * input.target_mu};
        long double error2 = 0.0L;
        long double expected2 = 0.0L;
        for (int j = 0; j < 4; ++j) {
          const size_t k = static_cast<size_t>(j);
          expected_u[k] = -input.target_mu * actual_q[k] /
                          (actual_a * target_h[k]);
          const long double error = u[k] - expected_u[k];
          error2 += error * error;
          expected2 += expected_u[k] * expected_u[k];
        }
        metrics.relative_reconstruction_error =
            std::sqrt(error2) / std::max(1.0L, std::sqrt(expected2));
      }
    }
  } else if (actual_a == 0.0L && finite_u &&
             std::isfinite(metrics.lambda_bar)) {
    metrics.endpoint_margin = 1.0L;
    metrics.hessian_condition = 1.0L;
  }

  bool expected_status = false;
  if (input.expected == ExpectedPath::kRegular) {
    expected_status = metrics.result.status == LOTT_STATUS_REGULAR_INTERIOR;
  } else if (input.expected == ExpectedPath::kBoundaryNonunique) {
    expected_status =
        metrics.result.status == LOTT_STATUS_BOUNDARY_PSD_NONUNIQUE;
  } else {
    expected_status = metrics.result.status == LOTT_STATUS_AFFINE;
  }
  const bool certificate_flags = metrics.result.feasibility_ok &&
                                 metrics.result.kkt_ok &&
                                 metrics.result.hessian_ok;
  const bool residuals_ok =
      std::isfinite(metrics.normalized_feasibility_residual) &&
      metrics.normalized_feasibility_residual <= 5e-11L &&
      std::isfinite(metrics.normalized_kkt_residual) &&
      metrics.normalized_kkt_residual <= 5e-11L;
  const bool pivot_ok = metrics.pivot_ratio >= 0.5L - 1e-15L;
  const bool rank_two_ok =
      std::isfinite(metrics.normalized_rank_two_identity_residual) &&
      metrics.normalized_rank_two_identity_residual <= 5e-14L;
  bool regular_metrics_ok = true;
  if (input.expected == ExpectedPath::kRegular) {
    regular_metrics_ok =
        std::isfinite(metrics.endpoint_margin) &&
        metrics.endpoint_margin > 0.0L &&
        std::isfinite(metrics.residual_root_bound_ratio) &&
        metrics.residual_root_bound_ratio <= 1.0L + 1e-10L &&
        std::isfinite(metrics.relative_reconstruction_error) &&
        metrics.relative_reconstruction_error <= 5e-10L &&
        std::isfinite(metrics.mu) &&
        std::abs(metrics.mu - input.target_mu) <= 5e-11L;
  }
  metrics.passed = metrics.certified && expected_status && certificate_flags &&
                   residuals_ok && pivot_ok && rank_two_ok &&
                   regular_metrics_ok;
  return metrics;
}

void write_case_csv(const std::filesystem::path &path,
                    const std::vector<CaseMetrics> &rows) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot open case CSV: " + path.string());
  }
  out << "suite,parameter,parameter_value,replicate,expected_path,status,"
         "certified,fallback_required,passed,a_raw,b_raw,b_over_a,"
         "coefficient_scale,selected_chart,pivot_ratio,qn_ratio,target_mu,"
         "mu,endpoint_margin,lambda_bar,lambda_bar_absolute_error,"
         "normalized_phi,normalized_phi_derivative,residual_root_condition,"
         "residual_root_bound,residual_root_bound_ratio,"
         "residual_implied_lambda_error_bound,minimum_hessian_eigenvalue,"
         "hessian_condition,reconstruction_condition,"
         "reconstruction_condition_bound,normalized_feasibility_residual,"
         "normalized_kkt_residual,normalized_rank_two_identity_residual,"
         "relative_reconstruction_error,lambda_bar_relative_error,iterations,"
         "bisection_steps,guarded_halfsteps,nonfinite_eval_steps,"
         "certificate_solution_count\n";
  for (const CaseMetrics &row : rows) {
    out << row.input.suite << ',' << row.input.parameter << ','
        << number(row.input.parameter_value) << ',' << row.input.replicate
        << ',' << expected_name(row.input.expected) << ','
        << status_name(row.result.status) << ',' << (row.certified ? 1 : 0)
        << ',' << (row.fallback_required ? 1 : 0) << ','
        << (row.passed ? 1 : 0) << ',' << number(row.input.a) << ','
        << number(row.input.b) << ',' << number(row.b_over_a) << ','
        << number(row.coefficient_scale) << ',' << row.selected_chart << ','
        << number(row.pivot_ratio) << ',' << number(row.qn_ratio) << ','
        << number(row.input.target_mu) << ',' << number(row.mu) << ','
        << number(row.endpoint_margin) << ',' << number(row.lambda_bar) << ','
        << number(row.lambda_bar_absolute_error) << ','
        << number(row.normalized_phi) << ','
        << number(row.normalized_phi_derivative) << ','
        << number(row.residual_root_condition) << ','
        << number(row.residual_root_bound) << ','
        << number(row.residual_root_bound_ratio) << ','
        << number(row.residual_implied_lambda_error_bound) << ','
        << number(static_cast<long double>(
               row.result.root.minimum_hessian_eigenvalue))
        << ',' << number(row.hessian_condition) << ','
        << number(row.reconstruction_condition) << ','
        << number(row.reconstruction_condition_bound) << ','
        << number(row.normalized_feasibility_residual) << ','
        << number(row.normalized_kkt_residual) << ','
        << number(row.normalized_rank_two_identity_residual) << ','
        << number(row.relative_reconstruction_error) << ','
        << number(row.lambda_bar_relative_error) << ','
        << row.result.root.iterations << ',' << row.result.root.bisection_steps
        << ',' << row.result.root.guarded_halfsteps << ','
        << row.result.root.nonfinite_eval_steps << ','
        << row.result.certified_solution_count << '\n';
  }
}

std::vector<BinStats> aggregate(const std::vector<CaseMetrics> &rows) {
  using Key = std::tuple<std::string, std::string, std::string>;
  std::map<Key, BinStats> bins;
  for (const CaseMetrics &row : rows) {
    const Key key = {row.input.suite, row.input.parameter,
                     number(row.input.parameter_value)};
    BinStats &bin = bins[key];
    bin.suite = row.input.suite;
    bin.parameter = row.input.parameter;
    bin.parameter_value = row.input.parameter_value;
    ++bin.cases;
    bin.certified += row.certified ? 1 : 0;
    bin.fallback_required += row.fallback_required ? 1 : 0;
    bin.passed += row.passed ? 1 : 0;
    bin.regular += row.result.status == LOTT_STATUS_REGULAR_INTERIOR ? 1 : 0;
    bin.boundary +=
        (row.result.status == LOTT_STATUS_BOUNDARY_PSD_UNIQUE ||
         row.result.status == LOTT_STATUS_BOUNDARY_PSD_NONUNIQUE)
            ? 1
            : 0;
    bin.affine += row.result.status == LOTT_STATUS_AFFINE ? 1 : 0;
    bin.iterations += row.result.root.iterations;
    bin.max_iterations =
        std::max(bin.max_iterations, row.result.root.iterations);
    bin.bisections += row.result.root.bisection_steps;
    bin.max_bisections =
        std::max(bin.max_bisections, row.result.root.bisection_steps);
    bin.max_feasibility_residual = finite_max(
        bin.max_feasibility_residual, row.normalized_feasibility_residual);
    bin.max_kkt_residual =
        finite_max(bin.max_kkt_residual, row.normalized_kkt_residual);
    bin.max_rank_two_identity_residual =
        finite_max(bin.max_rank_two_identity_residual,
                   row.normalized_rank_two_identity_residual);
    if (std::isfinite(row.endpoint_margin)) {
      bin.min_endpoint_margin =
          std::min(bin.min_endpoint_margin, row.endpoint_margin);
    }
    bin.max_residual_root_condition = finite_max(
        bin.max_residual_root_condition, row.residual_root_condition);
    bin.max_residual_root_bound_ratio = finite_max(
        bin.max_residual_root_bound_ratio, row.residual_root_bound_ratio);
    if (std::isinf(row.hessian_condition)) {
      bin.max_hessian_condition =
          std::numeric_limits<long double>::infinity();
    } else {
      bin.max_hessian_condition =
          finite_max(bin.max_hessian_condition, row.hessian_condition);
    }
    bin.max_reconstruction_error = finite_max(
        bin.max_reconstruction_error, row.relative_reconstruction_error);
    bin.max_lambda_error =
        finite_max(bin.max_lambda_error, row.lambda_bar_absolute_error);
    bin.max_lambda_relative_error = finite_max(
        bin.max_lambda_relative_error, row.lambda_bar_relative_error);
  }
  std::vector<BinStats> result;
  result.reserve(bins.size());
  for (const auto &[key, bin] : bins) {
    static_cast<void>(key);
    result.push_back(bin);
  }
  std::sort(result.begin(), result.end(), [](const BinStats &lhs,
                                             const BinStats &rhs) {
    if (lhs.suite != rhs.suite) {
      return lhs.suite < rhs.suite;
    }
    return lhs.parameter_value < rhs.parameter_value;
  });
  return result;
}

void write_bin_csv(const std::filesystem::path &path,
                   const std::vector<BinStats> &bins) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot open bin CSV: " + path.string());
  }
  out << "suite,parameter,parameter_value,cases,certified_rate,"
         "fallback_required_rate,pass_rate,regular_count,boundary_count,"
         "affine_count,mean_iterations,max_iterations,mean_bisections,"
         "max_bisections,max_normalized_feasibility_residual,"
         "max_normalized_kkt_residual,min_endpoint_margin,"
         "max_normalized_rank_two_identity_residual,"
         "max_residual_root_condition,max_condition_to_bound_ratio,"
         "max_hessian_condition,max_relative_reconstruction_error,"
         "max_lambda_bar_absolute_error,max_lambda_bar_relative_error\n";
  for (const BinStats &bin : bins) {
    out << bin.suite << ',' << bin.parameter << ','
        << number(bin.parameter_value) << ',' << bin.cases << ','
        << number(static_cast<long double>(bin.certified) / bin.cases) << ','
        << number(static_cast<long double>(bin.fallback_required) / bin.cases)
        << ',' << number(static_cast<long double>(bin.passed) / bin.cases)
        << ',' << bin.regular << ',' << bin.boundary << ',' << bin.affine
        << ',' << number(static_cast<long double>(bin.iterations) / bin.cases)
        << ',' << bin.max_iterations << ','
        << number(static_cast<long double>(bin.bisections) / bin.cases) << ','
        << bin.max_bisections << ','
        << number(bin.max_feasibility_residual) << ','
        << number(bin.max_kkt_residual) << ','
        << number(std::isfinite(bin.min_endpoint_margin)
                      ? bin.min_endpoint_margin
                      : kNaN)
        << ',' << number(bin.max_rank_two_identity_residual) << ','
        << number(bin.max_residual_root_condition) << ','
        << number(bin.max_residual_root_bound_ratio) << ','
        << number(bin.max_hessian_condition) << ','
        << number(bin.max_reconstruction_error) << ','
        << number(bin.max_lambda_error) << ','
        << number(bin.max_lambda_relative_error) << '\n';
  }
}

void write_summary(const std::filesystem::path &path,
                   const std::vector<CaseMetrics> &rows,
                   const std::vector<BinStats> &bins) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot open summary: " + path.string());
  }
  const int certified = static_cast<int>(std::count_if(
      rows.begin(), rows.end(), [](const CaseMetrics &row) {
        return row.certified;
      }));
  const int fallback = static_cast<int>(std::count_if(
      rows.begin(), rows.end(), [](const CaseMetrics &row) {
        return row.fallback_required;
      }));
  const int passed = static_cast<int>(std::count_if(
      rows.begin(), rows.end(), [](const CaseMetrics &row) {
        return row.passed;
      }));
  long double max_feasibility = 0.0L;
  long double max_kkt = 0.0L;
  long double max_bound_ratio = 0.0L;
  int max_iterations = 0;
  int max_bisections = 0;
  for (const CaseMetrics &row : rows) {
    max_feasibility =
        finite_max(max_feasibility, row.normalized_feasibility_residual);
    max_kkt = finite_max(max_kkt, row.normalized_kkt_residual);
    max_bound_ratio =
        finite_max(max_bound_ratio, row.residual_root_bound_ratio);
    max_iterations = std::max(max_iterations, row.result.root.iterations);
    max_bisections =
        std::max(max_bisections, row.result.root.bisection_steps);
  }

  out << "# Deterministic conditioning and degeneracy sweep\n\n"
      << "The runner evaluates analytically constructed canonical rank-two "
         "families. For every regular case, the target multiplier and "
         "correction are known independently of the numerical solve. A "
         "negative point status is counted as requiring the public "
         "Hartley--Sturm fallback; the sweep does not time or invoke that "
         "fallback.\n\n"
      << "## Overall result\n\n"
      << "- Cases: " << rows.size() << " across " << bins.size()
      << " parameter bins\n"
      << "- Certified: " << certified << "/" << rows.size() << "\n"
      << "- Fallback required: " << fallback << "/" << rows.size() << "\n"
      << "- All assertions passed: " << passed << "/" << rows.size()
      << "\n"
      << "- Maximum normalized feasibility residual: `"
      << number(max_feasibility) << "`\n"
      << "- Maximum normalized KKT residual: `" << number(max_kkt)
      << "`\n"
      << "- Maximum observed residual-condition / theoretical-bound ratio: `"
      << number(max_bound_ratio) << "`\n"
      << "- Maximum safeguarded iterations: " << max_iterations << "\n"
      << "- Maximum bisection steps: " << max_bisections << "\n\n"
      << "## Bin-level results\n\n"
      << "| Suite | Parameter | Bins | Cases | Certified | Fallback | Pass | "
         "Max iter. | Max bisect. |\n"
      << "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n";

  struct SuiteStats {
    std::string parameter;
    int bins = 0;
    int cases = 0;
    int certified = 0;
    int fallback = 0;
    int passed = 0;
    int max_iterations = 0;
    int max_bisections = 0;
  };
  std::map<std::string, SuiteStats> suites;
  for (const BinStats &bin : bins) {
    SuiteStats &suite = suites[bin.suite];
    suite.parameter = bin.parameter;
    ++suite.bins;
    suite.cases += bin.cases;
    suite.certified += bin.certified;
    suite.fallback += bin.fallback_required;
    suite.passed += bin.passed;
    suite.max_iterations =
        std::max(suite.max_iterations, bin.max_iterations);
    suite.max_bisections =
        std::max(suite.max_bisections, bin.max_bisections);
  }
  for (const auto &[name, suite] : suites) {
    out << "| " << name << " | `" << suite.parameter << "` | "
        << suite.bins << " | " << suite.cases << " | " << suite.certified
        << " | " << suite.fallback << " | " << suite.passed << " | "
        << suite.max_iterations << " | " << suite.max_bisections << " |\n";
  }
  out << "\n## Interpretation limits\n\n"
      << "- These are canonical coefficient-space stress fixtures, not a "
         "real-image experiment.\n"
      << "- `residual_root_condition` is the scale-normalized "
         "`1/|phi'|`; `residual_root_bound` is `2/||q||^2`. They describe "
         "scalar residual-to-multiplier sensitivity, not end-to-end image "
         "conditioning.\n"
      << "- `hessian_condition` and `reconstruction_condition` expose the "
         "separate endpoint sensitivity. The exact `one_minus_mu=0` rows "
         "are PSD-boundary/nonunique and therefore do not receive a regular "
         "root condition number.\n"
      << "- The common-scale suite checks invariance after the solver's "
         "positive coefficient normalization; it is not a substitute for "
         "camera-coordinate scale tests.\n"
      << "- In the positive-`a` affine-limit family, the target `mu` and the "
         "unscaled linear coefficients are fixed, so `g` grows like `1/a`; "
         "after common normalization both the quadratic and linear terms "
         "approach zero. The exact `a=0` rows are separately constructed "
         "rank-two affine projections, not the finite pointwise limit of "
         "those positive-`a` corrections.\n";
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: conditioning_sweep CASES.csv BINS.csv SUMMARY.md\n";
    return 2;
  }
  try {
    const std::filesystem::path cases_path(argv[1]);
    const std::filesystem::path bins_path(argv[2]);
    const std::filesystem::path summary_path(argv[3]);
    for (const std::filesystem::path &path :
         {cases_path, bins_path, summary_path}) {
      if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
      }
    }

    const std::vector<SweepCase> inputs = make_cases();
    std::vector<CaseMetrics> rows;
    rows.reserve(inputs.size());
    for (const SweepCase &input : inputs) {
      rows.push_back(evaluate_case(input));
    }
    const std::vector<BinStats> bins = aggregate(rows);
    write_case_csv(cases_path, rows);
    write_bin_csv(bins_path, bins);
    write_summary(summary_path, rows, bins);

    const int failures = static_cast<int>(std::count_if(
        rows.begin(), rows.end(), [](const CaseMetrics &row) {
          return !row.passed;
        }));
    std::cout << "conditioning_sweep_cases=" << rows.size() << '\n'
              << "conditioning_sweep_bins=" << bins.size() << '\n'
              << "conditioning_sweep_failures=" << failures << '\n'
              << "conditioning_sweep_cases_csv=" << cases_path.string()
              << '\n'
              << "conditioning_sweep_bins_csv=" << bins_path.string() << '\n'
              << "conditioning_sweep_summary=" << summary_path.string()
              << '\n';
    return failures == 0 ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "conditioning_sweep error: " << error.what() << '\n';
    return 2;
  }
}
