#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/estimators/ols.hpp"

namespace hfm {

struct AROptions {
    std::size_t p = 1;
    bool add_intercept = true;
    CovarianceType covariance = CovarianceType::NeweyWest;
    i64 hac_lag = -1;

    AROptions& set_p(std::size_t lags) { p = lags; return *this; }
    AROptions& set_intercept(bool v) { add_intercept = v; return *this; }
};

struct ARResult {
    Vector<f64> coefficients;  // [const, phi_1, ..., phi_p]
    Vector<f64> std_errors;
    Vector<f64> t_stats;
    Vector<f64> residuals;
    f64 sigma = 0.0;
    f64 r_squared = 0.0;
    f64 aic = 0.0;
    f64 bic = 0.0;
    std::size_t p = 0;
    std::size_t n_obs = 0;
    bool has_intercept = true;
    f64 elapsed_ms = 0.0;
};

Result<ARResult> ar(const Vector<f64>& y, const AROptions& opts = AROptions{});

Matrix<f64> lag_design_matrix(const Vector<f64>& y, std::size_t p, bool intercept = true);

} // namespace hfm
