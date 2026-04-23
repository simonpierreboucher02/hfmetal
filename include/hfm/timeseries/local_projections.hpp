#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/estimators/ols.hpp"

namespace hfm {

struct LPOptions {
    std::size_t max_horizon = 12;
    std::size_t n_lags = 4;        // control lags
    bool add_intercept = true;
    CovarianceType covariance = CovarianceType::NeweyWest;
    i64 hac_lag = -1;

    LPOptions& set_max_horizon(std::size_t h) { max_horizon = h; return *this; }
    LPOptions& set_n_lags(std::size_t l) { n_lags = l; return *this; }
    LPOptions& set_covariance(CovarianceType c) { covariance = c; return *this; }
};

struct LPResult {
    Vector<f64> irf;               // impulse response at each horizon
    Vector<f64> irf_se;            // standard errors
    Vector<f64> irf_lower;         // CI lower
    Vector<f64> irf_upper;         // CI upper
    Vector<f64> cumulative_irf;    // cumulative IRF
    std::size_t max_horizon = 0;
    std::size_t n_obs = 0;
    f64 confidence_level = 0.95;
    f64 elapsed_ms = 0.0;
};

// Local projections: y_{t+h} = alpha_h + beta_h * x_t + gamma_h' * controls_t + e_{t+h}
// y: response variable
// x: impulse variable (shock)
// controls: optional control variables (can be empty matrix with 0 cols)
Result<LPResult> local_projections(const Vector<f64>& y,
                                    const Vector<f64>& x,
                                    const Matrix<f64>& controls,
                                    const LPOptions& opts = LPOptions{});

// Simplified version: univariate LP (y responds to its own shock)
Result<LPResult> local_projections(const Vector<f64>& y,
                                    const LPOptions& opts = LPOptions{});

} // namespace hfm
