#pragma once

#include <vector>
#include <functional>
#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/estimators/ols.hpp"

namespace hfm {

struct BatchedOLSOptions {
    CovarianceType covariance = CovarianceType::Classical;
    i64 hac_lag = -1;
    bool add_intercept = false;
    Backend backend = Backend::Auto;
    std::size_t n_threads = 0;  // 0 = hardware_concurrency

    BatchedOLSOptions& set_covariance(CovarianceType c) { covariance = c; return *this; }
    BatchedOLSOptions& set_hac_lag(i64 lag) { hac_lag = lag; return *this; }
    BatchedOLSOptions& set_n_threads(std::size_t n) { n_threads = n; return *this; }
};

struct BatchedOLSResult {
    Matrix<f64> betas;       // n_regressions x k
    Matrix<f64> std_errors;  // n_regressions x k
    Matrix<f64> t_stats;     // n_regressions x k
    Vector<f64> r_squared;   // n_regressions
    std::size_t n_regressions = 0;
    std::size_t n_succeeded = 0;
    std::vector<bool> converged;
    f64 elapsed_ms = 0.0;
    Backend backend_used = Backend::CPU;
};

// Same X, many y vectors
Result<BatchedOLSResult> batched_ols(const Matrix<f64>& Y_columns,
                                      const Matrix<f64>& X,
                                      const BatchedOLSOptions& opts = BatchedOLSOptions{});

// Different (y, X) pairs
Result<BatchedOLSResult> batched_ols(
    const std::vector<std::pair<Vector<f64>, Matrix<f64>>>& problems,
    const BatchedOLSOptions& opts = BatchedOLSOptions{});

} // namespace hfm
