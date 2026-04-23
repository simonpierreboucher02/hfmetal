#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/data/series.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/estimators/ols.hpp"

namespace hfm {

struct HAROptions {
    std::size_t daily_lag = 1;     // RV(d)
    std::size_t weekly_lag = 5;    // RV(w) = avg of 5 days
    std::size_t monthly_lag = 22;  // RV(m) = avg of 22 days
    bool add_intercept = true;
    CovarianceType covariance = CovarianceType::NeweyWest;
    i64 hac_lag = -1;

    HAROptions& set_lags(std::size_t d, std::size_t w, std::size_t m) {
        daily_lag = d; weekly_lag = w; monthly_lag = m; return *this;
    }
};

struct HARResult {
    f64 alpha = 0.0;
    f64 beta_d = 0.0;
    f64 beta_w = 0.0;
    f64 beta_m = 0.0;
    Vector<f64> std_errors;
    Vector<f64> t_stats;
    f64 r_squared = 0.0;
    std::size_t n_obs = 0;
    f64 elapsed_ms = 0.0;
};

Result<HARResult> har_rv(const Series<f64>& daily_rv,
                          const HAROptions& opts = HAROptions{});

} // namespace hfm
