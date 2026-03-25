#pragma once

#include <vector>

#include <Eigen/Dense>

#include "lott_triangulate.h"
#include "triangulate_hs.h"

struct LottCertifiedFallbackDiagnostics {
  long long points_total = 0;
  long long cert_eq1_points = 0;
  long long fallback_points = 0;
  long long fallback_nonunique_points = 0;
  long long fallback_cert_failure_points = 0;
};

inline void lott_triangulate_certified_fallback(
    const Eigen::Matrix<double, 4, -1> &A, const Eigen::Matrix<double, 3, 3> &F,
    Eigen::Matrix<double, 4, -1> &X,
    LottSolverDiagnostics *solver_diag = nullptr,
    LottCertifiedFallbackDiagnostics *fallback_diag = nullptr) {
  Eigen::VectorXi cert_closed_count;
  lott_triangulate(A, F, X, solver_diag, true, 0, &cert_closed_count);

  LottCertifiedFallbackDiagnostics local_diag;
  if (fallback_diag == nullptr) {
    fallback_diag = &local_diag;
  }
  fallback_diag->points_total = A.cols();

  std::vector<int> fallback_indices;
  fallback_indices.reserve(static_cast<size_t>(A.cols()));
  for (int i = 0; i < A.cols(); ++i) {
    const int c = cert_closed_count(i);
    if (c == 1) {
      ++fallback_diag->cert_eq1_points;
      continue;
    }
    fallback_indices.push_back(i);
    if (c < 0) {
      ++fallback_diag->fallback_cert_failure_points;
    } else {
      ++fallback_diag->fallback_nonunique_points;
    }
  }

  fallback_diag->fallback_points = static_cast<long long>(fallback_indices.size());
  if (fallback_indices.empty()) {
    return;
  }

  const int m = static_cast<int>(fallback_indices.size());
  Eigen::Matrix<double, 3, -1> x(3, m), xp(3, m);
  for (int j = 0; j < m; ++j) {
    const int i = fallback_indices[static_cast<size_t>(j)];
    x.col(j) << A(0, i), A(1, i), 1.0;
    xp.col(j) << A(2, i), A(3, i), 1.0;
  }

  Eigen::Matrix<double, 4, -1> X_hs(4, m);
  hartley_triangulate(x, xp, F, X_hs);
  for (int j = 0; j < m; ++j) {
    const int i = fallback_indices[static_cast<size_t>(j)];
    X.col(i) = X_hs.col(j);
  }
}
