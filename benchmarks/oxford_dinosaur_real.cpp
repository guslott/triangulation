#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "lott_triangulate.h"
#include "triangulate_hs.h"

namespace {

using Camera = Eigen::Matrix<double, 3, 4>;

struct Args {
  std::string tracks;
  std::string cameras;
  std::string pair_csv;
  std::string point_csv;
};

struct Track {
  std::array<double, 72> values{};
};

struct FundamentalResult {
  Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
  double formula_relative_error = std::numeric_limits<double>::infinity();
  double camera_center_relative_residual = std::numeric_limits<double>::infinity();
  double projected_point_sampson_max = std::numeric_limits<double>::infinity();
  double projected_point_normalized_residual_max =
      std::numeric_limits<double>::infinity();
  double rank_ratio = std::numeric_limits<double>::infinity();
};

void usage(const char *program) {
  std::cout << "Usage: " << program
            << " --tracks <viff.xy> --cameras <dino_cameras.tsv>"
               " [--pair-csv <path>] [--point-csv <path>]\n";
}

bool parse_args(const int argc, char **argv, Args &args) {
  for (int i = 1; i < argc; ++i) {
    const std::string token(argv[i]);
    if (token == "-h" || token == "--help") {
      usage(argv[0]);
      return false;
    }
    auto consume = [&](std::string &destination) -> bool {
      if (i + 1 >= argc) {
        std::cerr << "Missing value after " << token << "\n";
        return false;
      }
      destination = argv[++i];
      return true;
    };
    if (token == "--tracks") {
      if (!consume(args.tracks)) {
        return false;
      }
    } else if (token == "--cameras") {
      if (!consume(args.cameras)) {
        return false;
      }
    } else if (token == "--pair-csv") {
      if (!consume(args.pair_csv)) {
        return false;
      }
    } else if (token == "--point-csv") {
      if (!consume(args.point_csv)) {
        return false;
      }
    } else {
      std::cerr << "Unknown argument: " << token << "\n";
      return false;
    }
  }
  if (args.tracks.empty() || args.cameras.empty()) {
    usage(argv[0]);
    return false;
  }
  return true;
}

std::vector<Track> read_tracks(const std::string &path,
                               long long &visible_observations) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot open tracks: " + path);
  }
  std::vector<Track> tracks;
  std::string line;
  int line_number = 0;
  visible_observations = 0;
  while (std::getline(stream, line)) {
    ++line_number;
    if (line.empty()) {
      continue;
    }
    std::istringstream row(line);
    Track track;
    for (double &value : track.values) {
      if (!(row >> value) || !std::isfinite(value)) {
        throw std::runtime_error("invalid track value on row " +
                                 std::to_string(line_number));
      }
    }
    double extra = 0.0;
    if (row >> extra) {
      throw std::runtime_error("more than 72 values on track row " +
                               std::to_string(line_number));
    }
    for (int view = 0; view < 36; ++view) {
      const bool x_missing = track.values[2 * view] == -1.0;
      const bool y_missing = track.values[2 * view + 1] == -1.0;
      if (x_missing != y_missing) {
        throw std::runtime_error("split missing-value sentinel on track row " +
                                 std::to_string(line_number));
      }
      if (!x_missing) {
        ++visible_observations;
      }
    }
    tracks.push_back(track);
  }
  if (tracks.empty()) {
    throw std::runtime_error("track file is empty");
  }
  return tracks;
}

std::vector<Camera> read_cameras(const std::string &path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot open cameras: " + path);
  }
  std::vector<Camera> cameras;
  std::string line;
  int line_number = 0;
  while (std::getline(stream, line)) {
    ++line_number;
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream row(line);
    int view = 0;
    if (!(row >> view) || view != static_cast<int>(cameras.size()) + 1) {
      throw std::runtime_error("unexpected camera index on row " +
                               std::to_string(line_number));
    }
    Camera camera;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 4; ++c) {
        if (!(row >> camera(r, c)) || !std::isfinite(camera(r, c))) {
          throw std::runtime_error("invalid camera value on row " +
                                   std::to_string(line_number));
        }
      }
    }
    double extra = 0.0;
    if (row >> extra) {
      throw std::runtime_error("extra camera value on row " +
                               std::to_string(line_number));
    }
    cameras.push_back(camera);
  }
  if (cameras.size() != 36) {
    throw std::runtime_error("expected 36 cameras, found " +
                             std::to_string(cameras.size()));
  }
  return cameras;
}

Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d result;
  result << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
  return result;
}

double up_to_sign_distance(const Eigen::Matrix3d &lhs,
                           const Eigen::Matrix3d &rhs) {
  return std::min((lhs - rhs).norm(), (lhs + rhs).norm());
}

double sampson_distance(const Eigen::Vector3d &x,
                        const Eigen::Vector3d &xp,
                        const Eigen::Matrix3d &F) {
  const Eigen::Vector3d line_p = F * x;
  const Eigen::Vector3d line = F.transpose() * xp;
  const double denominator = std::sqrt(
      line_p.head<2>().squaredNorm() + line.head<2>().squaredNorm());
  if (!(denominator > 0.0) || !std::isfinite(denominator)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::abs(xp.dot(F * x)) / denominator;
}

double normalized_algebraic_residual(const Eigen::Vector4d &point,
                                     const Eigen::Matrix3d &F) {
  const Eigen::Vector3d x(point(0), point(1), 1.0);
  const Eigen::Vector3d xp(point(2), point(3), 1.0);
  const double denominator = std::max(1.0, F.norm() * x.norm() * xp.norm());
  return std::abs(xp.dot(F * x)) / denominator;
}

FundamentalResult derive_fundamental(const Camera &P, const Camera &Pp) {
  FundamentalResult result;
  const Eigen::JacobiSVD<Camera> camera_svd(P, Eigen::ComputeFullV);
  Eigen::Vector4d center = camera_svd.matrixV().col(3);
  result.camera_center_relative_residual =
      (P * center).norm() / std::max(1.0, P.norm() * center.norm());

  const Eigen::Matrix3d gram = P * P.transpose();
  const Camera pinv_transpose = gram.inverse() * P;
  const Eigen::Matrix<double, 4, 3> pinv = pinv_transpose.transpose();
  const Eigen::Vector3d epipole_p = Pp * center;
  Eigen::Matrix3d raw = skew(epipole_p) * Pp * pinv;
  if (!raw.allFinite() || !(raw.norm() > 0.0)) {
    throw std::runtime_error("camera formula produced an invalid fundamental matrix");
  }

  const Eigen::JacobiSVD<Eigen::Matrix3d> raw_svd(
      raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d rank_two_values = raw_svd.singularValues();
  rank_two_values(2) = 0.0;
  result.F = raw_svd.matrixU() * rank_two_values.asDiagonal() *
             raw_svd.matrixV().transpose();
  result.F /= result.F.norm();

  // Independent determinant construction:
  // F(i,j) is, up to a global sign, the determinant of the four camera rows
  // remaining after removing row j from P and row i from P'.
  Eigen::Matrix3d minors;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Eigen::Matrix4d stacked;
      int target_row = 0;
      for (int source_row = 0; source_row < 3; ++source_row) {
        if (source_row != j) {
          stacked.row(target_row++) = P.row(source_row);
        }
      }
      for (int source_row = 0; source_row < 3; ++source_row) {
        if (source_row != i) {
          stacked.row(target_row++) = Pp.row(source_row);
        }
      }
      const double checkerboard_sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
      minors(i, j) = checkerboard_sign * stacked.determinant();
    }
  }
  if (!minors.allFinite() || !(minors.norm() > 0.0)) {
    throw std::runtime_error("minor formula produced an invalid fundamental matrix");
  }
  minors /= minors.norm();
  result.formula_relative_error = up_to_sign_distance(result.F, minors);

  const Eigen::JacobiSVD<Eigen::Matrix3d> final_svd(result.F);
  result.rank_ratio = final_svd.singularValues()(2) /
                      final_svd.singularValues()(0);

  const std::array<Eigen::Vector4d, 5> scene_points = {
      Eigen::Vector4d(0.2, -0.3, 1.1, 1.0),
      Eigen::Vector4d(-0.8, 0.4, 0.7, 1.0),
      Eigen::Vector4d(1.3, 0.2, -0.5, 1.0),
      Eigen::Vector4d(-0.1, -1.2, 0.9, 1.0),
      Eigen::Vector4d(0.6, 0.8, 1.7, 1.0),
  };
  result.projected_point_sampson_max = 0.0;
  result.projected_point_normalized_residual_max = 0.0;
  for (const Eigen::Vector4d &scene_point : scene_points) {
    Eigen::Vector3d x = P * scene_point;
    Eigen::Vector3d xp = Pp * scene_point;
    if (std::abs(x(2)) < 1e-14 || std::abs(xp(2)) < 1e-14) {
      continue;
    }
    x /= x(2);
    xp /= xp(2);
    result.projected_point_sampson_max =
        std::max(result.projected_point_sampson_max,
                 sampson_distance(x, xp, result.F));
    const Eigen::Vector4d projected(x(0), x(1), xp(0), xp(1));
    result.projected_point_normalized_residual_max =
        std::max(result.projected_point_normalized_residual_max,
                 normalized_algebraic_residual(projected, result.F));
  }
  return result;
}

double correction_cost(const Eigen::Vector4d &corrected,
                       const Eigen::Vector4d &observed) {
  return (corrected - observed).squaredNorm();
}

double quantile(std::vector<double> values, const double q) {
  values.erase(std::remove_if(values.begin(), values.end(),
                              [](const double value) {
                                return !std::isfinite(value);
                              }),
               values.end());
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const double position =
      std::clamp(q, 0.0, 1.0) * static_cast<double>(values.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

double mean(const std::vector<double> &values) {
  long double sum = 0.0L;
  long long count = 0;
  for (const double value : values) {
    if (std::isfinite(value)) {
      sum += value;
      ++count;
    }
  }
  return count ? static_cast<double>(sum / count)
               : std::numeric_limits<double>::quiet_NaN();
}

double finite_max(const std::vector<double> &values) {
  double result = -std::numeric_limits<double>::infinity();
  for (const double value : values) {
    if (std::isfinite(value)) {
      result = std::max(result, value);
    }
  }
  return std::isfinite(result) ? result
                               : std::numeric_limits<double>::quiet_NaN();
}

const char *status_name(const int status) {
  switch (static_cast<LottPointStatus>(status)) {
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
  default:
    return "unset_or_unknown";
  }
}

} // namespace

int main(int argc, char **argv) {
  Args args;
  if (!parse_args(argc, argv, args)) {
    return 2;
  }

  try {
    long long visible_observations = 0;
    const std::vector<Track> tracks = read_tracks(args.tracks, visible_observations);
    const std::vector<Camera> cameras = read_cameras(args.cameras);

    std::ofstream pair_csv;
    if (!args.pair_csv.empty()) {
      pair_csv.open(args.pair_csv);
      if (!pair_csv) {
        throw std::runtime_error("cannot open pair CSV: " + args.pair_csv);
      }
      pair_csv
          << "view1,view2,correspondences,formula_relative_error,"
             "camera_center_relative_residual,projected_point_sampson_max,"
             "projected_point_normalized_residual_max,rank_ratio,"
             "observation_sampson_median,transpose_sampson_median,"
             "lott_certified,lott_failures,hs_finite,both_finite,"
             "lott_cost_mean,hs_cost_mean,gap_mean,gap_abs_max,"
             "lott_normalized_residual_max,hs_normalized_residual_max\n";
      pair_csv << std::setprecision(17);
    }

    std::ofstream point_csv;
    if (!args.point_csv.empty()) {
      point_csv.open(args.point_csv);
      if (!point_csv) {
        throw std::runtime_error("cannot open point CSV: " + args.point_csv);
      }
      point_csv
          << "view1,view2,track,status,status_name,certificate_count,"
             "lott_finite,hs_finite,lott_cost,hs_cost,gap,absolute_gap,"
             "relative_gap,observation_sampson,lott_sampson,hs_sampson,"
             "lott_normalized_residual,hs_normalized_residual\n";
      point_csv << std::setprecision(17);
    }

    long long nonempty_pairs = 0;
    long long pair_correspondences = 0;
    long long lott_certified = 0;
    long long lott_failures = 0;
    long long lott_finite = 0;
    long long hs_finite = 0;
    long long both_finite = 0;
    long long lott_feasible_1e12 = 0;
    long long hs_feasible_1e12 = 0;
    long long lott_worse_beyond_tolerance = 0;
    long long hs_worse_beyond_tolerance = 0;
    std::map<int, long long> status_counts;
    LottSolverDiagnostics solver_diagnostics;

    double max_formula_error = 0.0;
    double max_camera_center_residual = 0.0;
    double max_projection_sampson = 0.0;
    double max_projection_normalized_residual = 0.0;
    double max_rank_ratio = 0.0;
    std::vector<double> observation_sampson;
    std::vector<double> transpose_sampson;
    std::vector<double> lott_costs;
    std::vector<double> hs_costs;
    std::vector<double> gaps;
    std::vector<double> absolute_gaps;
    std::vector<double> relative_gaps;
    std::vector<double> lott_sampson;
    std::vector<double> hs_sampson;
    std::vector<double> lott_normalized_residuals;
    std::vector<double> hs_normalized_residuals;

    for (int first = 0; first < 36; ++first) {
      for (int second = first + 1; second < 36; ++second) {
        std::vector<int> track_indices;
        for (size_t track_index = 0; track_index < tracks.size(); ++track_index) {
          const Track &track = tracks[track_index];
          if (track.values[2 * first] != -1.0 &&
              track.values[2 * second] != -1.0) {
            track_indices.push_back(static_cast<int>(track_index));
          }
        }
        if (track_indices.empty()) {
          continue;
        }
        ++nonempty_pairs;
        pair_correspondences += static_cast<long long>(track_indices.size());

        const FundamentalResult fundamental =
            derive_fundamental(cameras[first], cameras[second]);
        max_formula_error =
            std::max(max_formula_error, fundamental.formula_relative_error);
        max_camera_center_residual =
            std::max(max_camera_center_residual,
                     fundamental.camera_center_relative_residual);
        max_projection_sampson =
            std::max(max_projection_sampson,
                     fundamental.projected_point_sampson_max);
        max_projection_normalized_residual =
            std::max(max_projection_normalized_residual,
                     fundamental.projected_point_normalized_residual_max);
        max_rank_ratio = std::max(max_rank_ratio, fundamental.rank_ratio);

        const int count = static_cast<int>(track_indices.size());
        Eigen::Matrix<double, 4, -1> observations(4, count);
        Eigen::Matrix<double, 3, -1> x(3, count), xp(3, count);
        for (int column = 0; column < count; ++column) {
          const Track &track = tracks[static_cast<size_t>(track_indices[column])];
          observations.col(column) << track.values[2 * first],
              track.values[2 * first + 1], track.values[2 * second],
              track.values[2 * second + 1];
          x.col(column) << observations(0, column), observations(1, column), 1.0;
          xp.col(column) << observations(2, column), observations(3, column), 1.0;
        }

        Eigen::Matrix<double, 4, -1> lott_output;
        Eigen::Matrix<double, 4, -1> hs_output(4, count);
        Eigen::VectorXi certificate_counts;
        Eigen::VectorXi statuses;
        lott_triangulate(observations, fundamental.F, lott_output,
                         &solver_diagnostics, true, 0, &certificate_counts,
                         &statuses);
        hartley_triangulate(x, xp, fundamental.F, hs_output);

        long long pair_lott_certified = 0;
        long long pair_lott_failures = 0;
        long long pair_hs_finite = 0;
        long long pair_both_finite = 0;
        std::vector<double> pair_observation_sampson;
        std::vector<double> pair_transpose_sampson;
        std::vector<double> pair_lott_costs;
        std::vector<double> pair_hs_costs;
        std::vector<double> pair_gaps;
        std::vector<double> pair_absolute_gaps;
        std::vector<double> pair_lott_residuals;
        std::vector<double> pair_hs_residuals;

        for (int column = 0; column < count; ++column) {
          const int status = statuses(column);
          ++status_counts[status];
          const bool certified = lott_status_is_certified(
              static_cast<LottPointStatus>(status));
          const bool lott_is_finite = lott_output.col(column).allFinite();
          const bool hs_is_finite = hs_output.col(column).allFinite();
          if (certified) {
            ++lott_certified;
            ++pair_lott_certified;
          } else {
            ++lott_failures;
            ++pair_lott_failures;
          }
          if (lott_is_finite) {
            ++lott_finite;
          }
          if (hs_is_finite) {
            ++hs_finite;
            ++pair_hs_finite;
          }

          const Eigen::Vector3d observed_x = x.col(column);
          const Eigen::Vector3d observed_xp = xp.col(column);
          const double observed_sampson =
              sampson_distance(observed_x, observed_xp, fundamental.F);
          const double transposed_sampson =
              sampson_distance(observed_x, observed_xp, fundamental.F.transpose());
          observation_sampson.push_back(observed_sampson);
          transpose_sampson.push_back(transposed_sampson);
          pair_observation_sampson.push_back(observed_sampson);
          pair_transpose_sampson.push_back(transposed_sampson);

          double lott_cost = std::numeric_limits<double>::quiet_NaN();
          double hs_cost = std::numeric_limits<double>::quiet_NaN();
          double gap = std::numeric_limits<double>::quiet_NaN();
          double absolute_gap = std::numeric_limits<double>::quiet_NaN();
          double relative_gap = std::numeric_limits<double>::quiet_NaN();
          double lott_point_sampson = std::numeric_limits<double>::quiet_NaN();
          double hs_point_sampson = std::numeric_limits<double>::quiet_NaN();
          double lott_residual = std::numeric_limits<double>::quiet_NaN();
          double hs_residual = std::numeric_limits<double>::quiet_NaN();

          if (lott_is_finite) {
            lott_cost = correction_cost(lott_output.col(column),
                                        observations.col(column));
            const Eigen::Vector3d corrected_x(lott_output(0, column),
                                              lott_output(1, column), 1.0);
            const Eigen::Vector3d corrected_xp(lott_output(2, column),
                                               lott_output(3, column), 1.0);
            lott_point_sampson =
                sampson_distance(corrected_x, corrected_xp, fundamental.F);
            lott_residual = normalized_algebraic_residual(
                lott_output.col(column), fundamental.F);
            lott_costs.push_back(lott_cost);
            lott_sampson.push_back(lott_point_sampson);
            lott_normalized_residuals.push_back(lott_residual);
            pair_lott_costs.push_back(lott_cost);
            pair_lott_residuals.push_back(lott_residual);
            if (lott_residual <= 1e-12) {
              ++lott_feasible_1e12;
            }
          }
          if (hs_is_finite) {
            hs_cost = correction_cost(hs_output.col(column),
                                      observations.col(column));
            const Eigen::Vector3d corrected_x(hs_output(0, column),
                                              hs_output(1, column), 1.0);
            const Eigen::Vector3d corrected_xp(hs_output(2, column),
                                               hs_output(3, column), 1.0);
            hs_point_sampson =
                sampson_distance(corrected_x, corrected_xp, fundamental.F);
            hs_residual = normalized_algebraic_residual(hs_output.col(column),
                                                        fundamental.F);
            hs_costs.push_back(hs_cost);
            hs_sampson.push_back(hs_point_sampson);
            hs_normalized_residuals.push_back(hs_residual);
            pair_hs_costs.push_back(hs_cost);
            pair_hs_residuals.push_back(hs_residual);
            if (hs_residual <= 1e-12) {
              ++hs_feasible_1e12;
            }
          }
          if (lott_is_finite && hs_is_finite) {
            ++both_finite;
            ++pair_both_finite;
            gap = lott_cost - hs_cost;
            absolute_gap = std::abs(gap);
            const double scale = std::max({1.0, std::abs(lott_cost),
                                           std::abs(hs_cost)});
            relative_gap = absolute_gap / scale;
            const double tolerance = 1e-8 * scale;
            if (gap > tolerance) {
              ++lott_worse_beyond_tolerance;
            } else if (gap < -tolerance) {
              ++hs_worse_beyond_tolerance;
            }
            gaps.push_back(gap);
            absolute_gaps.push_back(absolute_gap);
            relative_gaps.push_back(relative_gap);
            pair_gaps.push_back(gap);
            pair_absolute_gaps.push_back(absolute_gap);
          }

          if (point_csv) {
            point_csv << first + 1 << ',' << second + 1 << ','
                      << track_indices[column] + 1 << ',' << status << ','
                      << status_name(status) << ',' << certificate_counts(column)
                      << ',' << (lott_is_finite ? 1 : 0) << ','
                      << (hs_is_finite ? 1 : 0) << ',' << lott_cost << ','
                      << hs_cost << ',' << gap << ',' << absolute_gap << ','
                      << relative_gap << ',' << observed_sampson << ','
                      << lott_point_sampson << ',' << hs_point_sampson << ','
                      << lott_residual << ',' << hs_residual << '\n';
          }
        }

        if (pair_csv) {
          pair_csv << first + 1 << ',' << second + 1 << ',' << count << ','
                   << fundamental.formula_relative_error << ','
                   << fundamental.camera_center_relative_residual << ','
                   << fundamental.projected_point_sampson_max << ','
                   << fundamental.projected_point_normalized_residual_max << ','
                   << fundamental.rank_ratio << ','
                   << quantile(pair_observation_sampson, 0.5) << ','
                   << quantile(pair_transpose_sampson, 0.5) << ','
                   << pair_lott_certified << ',' << pair_lott_failures << ','
                   << pair_hs_finite << ',' << pair_both_finite << ','
                   << mean(pair_lott_costs) << ',' << mean(pair_hs_costs) << ','
                   << mean(pair_gaps) << ',' << finite_max(pair_absolute_gaps)
                   << ',' << finite_max(pair_lott_residuals) << ','
                   << finite_max(pair_hs_residuals) << '\n';
        }
      }
    }

    const double observation_median = quantile(observation_sampson, 0.5);
    const double transpose_median = quantile(transpose_sampson, 0.5);
    const bool camera_geometry_valid = max_formula_error <= 1e-10 &&
                                       max_camera_center_residual <= 1e-12 &&
                                       max_projection_sampson <= 1e-5 &&
                                       max_projection_normalized_residual <=
                                           1e-12 &&
                                       max_rank_ratio <= 1e-12;
    const bool track_convention_valid =
        std::isfinite(observation_median) && std::isfinite(transpose_median) &&
        observation_median < transpose_median;

    std::cout << std::setprecision(17);
    std::cout << "dataset_tracks=" << tracks.size() << '\n';
    std::cout << "dataset_visible_observations=" << visible_observations << '\n';
    std::cout << "camera_count=" << cameras.size() << '\n';
    std::cout << "possible_view_pairs=630\n";
    std::cout << "nonempty_view_pairs=" << nonempty_pairs << '\n';
    std::cout << "pair_correspondences=" << pair_correspondences << '\n';
    std::cout << "max_f_formula_relative_error=" << max_formula_error << '\n';
    std::cout << "max_camera_center_relative_residual="
              << max_camera_center_residual << '\n';
    std::cout << "max_projected_point_sampson=" << max_projection_sampson
              << '\n';
    std::cout << "max_projected_point_normalized_residual="
              << max_projection_normalized_residual << '\n';
    std::cout << "max_rank_ratio_sigma3_sigma1=" << max_rank_ratio << '\n';
    std::cout << "observation_sampson_median=" << observation_median << '\n';
    std::cout << "observation_sampson_p95="
              << quantile(observation_sampson, 0.95) << '\n';
    std::cout << "transpose_convention_sampson_median=" << transpose_median
              << '\n';
    std::cout << "camera_geometry_validation="
              << (camera_geometry_valid ? "PASS" : "FAIL") << '\n';
    std::cout << "track_convention_validation="
              << (track_convention_valid ? "PASS" : "FAIL") << '\n';
    std::cout << "lott_certified=" << lott_certified << '\n';
    std::cout << "lott_failures=" << lott_failures << '\n';
    std::cout << "lott_finite=" << lott_finite << '\n';
    std::cout << "hartley_sturm_finite=" << hs_finite << '\n';
    std::cout << "both_finite=" << both_finite << '\n';
    std::cout << "lott_feasible_normalized_1e-12=" << lott_feasible_1e12
              << '\n';
    std::cout << "hartley_sturm_feasible_normalized_1e-12="
              << hs_feasible_1e12 << '\n';
    std::cout << "lott_normalized_residual_max="
              << finite_max(lott_normalized_residuals) << '\n';
    std::cout << "hartley_sturm_normalized_residual_max="
              << finite_max(hs_normalized_residuals) << '\n';
    std::cout << "lott_sampson_max=" << finite_max(lott_sampson) << '\n';
    std::cout << "hartley_sturm_sampson_max=" << finite_max(hs_sampson)
              << '\n';
    std::cout << "lott_cost_mean=" << mean(lott_costs) << '\n';
    std::cout << "hartley_sturm_cost_mean=" << mean(hs_costs) << '\n';
    std::cout << "objective_gap_lott_minus_hs_mean=" << mean(gaps) << '\n';
    std::cout << "objective_absolute_gap_median="
              << quantile(absolute_gaps, 0.5) << '\n';
    std::cout << "objective_absolute_gap_p95="
              << quantile(absolute_gaps, 0.95) << '\n';
    std::cout << "objective_absolute_gap_max=" << finite_max(absolute_gaps)
              << '\n';
    std::cout << "objective_relative_gap_max=" << finite_max(relative_gaps)
              << '\n';
    std::cout << "lott_worse_beyond_1e-8_scaled="
              << lott_worse_beyond_tolerance << '\n';
    std::cout << "hartley_sturm_worse_beyond_1e-8_scaled="
              << hs_worse_beyond_tolerance << '\n';
    for (const auto &[status, count] : status_counts) {
      std::cout << "status_" << status_name(status) << '=' << count << '\n';
    }
    std::cout << "solver_total_iterations="
              << solver_diagnostics.total_iterations << '\n';
    std::cout << "solver_bisection_steps=" << solver_diagnostics.bisection_steps
              << '\n';
    std::cout << "solver_chart_x=" << solver_diagnostics.chart_points[0]
              << '\n';
    std::cout << "solver_chart_y=" << solver_diagnostics.chart_points[1]
              << '\n';
    std::cout << "solver_chart_z=" << solver_diagnostics.chart_points[2]
              << '\n';
    std::cout << "solver_chart_w=" << solver_diagnostics.chart_points[3]
              << '\n';

    if (!camera_geometry_valid || !track_convention_valid) {
      std::cerr << "Camera geometry or x'^T F x convention validation failed; "
                   "results must not be used.\n";
      return 3;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
