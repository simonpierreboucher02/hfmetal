#pragma once

#include "hfm/core/types.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/vector.hpp"

namespace hfm {

Matrix<f64> classical_covariance(const Matrix<f64>& X, const Vector<f64>& residuals,
                                  std::size_t n, std::size_t k);

Matrix<f64> white_covariance(const Matrix<f64>& X, const Vector<f64>& residuals,
                              const Matrix<f64>& XtX_inv);

Matrix<f64> newey_west_covariance(const Matrix<f64>& X, const Vector<f64>& residuals,
                                   const Matrix<f64>& XtX_inv, i64 max_lag = -1);

Matrix<f64> clustered_covariance(const Matrix<f64>& X, const Vector<f64>& residuals,
                                  const Matrix<f64>& XtX_inv,
                                  const Vector<i64>& cluster_ids);

Matrix<f64> twoway_clustered_covariance(const Matrix<f64>& X, const Vector<f64>& residuals,
                                         const Matrix<f64>& XtX_inv,
                                         const Vector<i64>& cluster1_ids,
                                         const Vector<i64>& cluster2_ids);

i64 newey_west_auto_lag(std::size_t n);

} // namespace hfm
