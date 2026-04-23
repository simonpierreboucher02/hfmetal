#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/estimators/ols.hpp"

namespace hfm {

struct VAROptions {
    std::size_t p = 1;
    bool add_intercept = true;
    CovarianceType covariance = CovarianceType::NeweyWest;
    i64 hac_lag = -1;

    VAROptions& set_p(std::size_t lags) { p = lags; return *this; }
    VAROptions& set_intercept(bool v) { add_intercept = v; return *this; }
    VAROptions& set_covariance(CovarianceType c) { covariance = c; return *this; }
};

struct VARResult {
    Matrix<f64> coefficients;   // k x (n_vars*p + intercept) companion form
    Matrix<f64> residuals;      // T x n_vars
    Matrix<f64> sigma_u;        // n_vars x n_vars residual covariance
    std::vector<Matrix<f64>> covariance_matrices; // per-equation covariance
    std::vector<Vector<f64>> std_errors;          // per-equation SE
    std::vector<Vector<f64>> t_stats;             // per-equation t-stats
    f64 log_likelihood = 0.0;
    f64 aic = 0.0;
    f64 bic = 0.0;
    std::size_t p = 0;
    std::size_t n_vars = 0;
    std::size_t n_obs = 0;
    bool has_intercept = true;
    f64 elapsed_ms = 0.0;
};

// Y is T x n_vars matrix (each column is a variable)
Result<VARResult> var(const Matrix<f64>& Y, const VAROptions& opts = VAROptions{});

// Build the companion-form lag design matrix for VAR
Matrix<f64> var_lag_design_matrix(const Matrix<f64>& Y, std::size_t p, bool intercept = true);

} // namespace hfm
