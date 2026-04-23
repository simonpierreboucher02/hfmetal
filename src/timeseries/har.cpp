#include "hfm/timeseries/har.hpp"
#include "hfm/linalg/matrix.hpp"
#include <chrono>
#include <cmath>

namespace hfm {

Result<HARResult> har_rv(const Series<f64>& daily_rv,
                          const HAROptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();

    std::size_t n = daily_rv.size();
    std::size_t max_lag = std::max({opts.daily_lag, opts.weekly_lag, opts.monthly_lag});

    if (n <= max_lag + 1) {
        return Status::error(ErrorCode::InvalidArgument, "har_rv: not enough observations");
    }

    std::size_t T = n - max_lag;
    std::size_t k = opts.add_intercept ? 4 : 3;

    Matrix<f64> X(T, k, 0.0);
    Vector<f64> y(T);

    for (std::size_t t = 0; t < T; ++t) {
        std::size_t idx = max_lag + t;
        y[t] = daily_rv[idx];

        std::size_t col = 0;
        if (opts.add_intercept) {
            X(t, col++) = 1.0;
        }

        // Daily RV: RV(t-1)
        X(t, col++) = daily_rv[idx - opts.daily_lag];

        // Weekly RV: average of past weekly_lag days
        f64 rv_w = 0.0;
        for (std::size_t j = 0; j < opts.weekly_lag; ++j) {
            rv_w += daily_rv[idx - 1 - j];
        }
        X(t, col++) = rv_w / static_cast<f64>(opts.weekly_lag);

        // Monthly RV: average of past monthly_lag days
        f64 rv_m = 0.0;
        for (std::size_t j = 0; j < opts.monthly_lag; ++j) {
            rv_m += daily_rv[idx - 1 - j];
        }
        X(t, col++) = rv_m / static_cast<f64>(opts.monthly_lag);
    }

    OLSOptions ols_opts;
    ols_opts.covariance = opts.covariance;
    ols_opts.hac_lag = opts.hac_lag;

    auto ols_res = ols(y, X, ols_opts);
    if (!ols_res) return ols_res.status();

    auto& res = ols_res.value();
    HARResult result;
    result.n_obs = T;
    result.r_squared = res.r_squared();
    result.std_errors = res.std_errors();
    result.t_stats = res.t_stats();

    std::size_t col = 0;
    if (opts.add_intercept) result.alpha = res.coefficients()[col++];
    result.beta_d = res.coefficients()[col++];
    result.beta_w = res.coefficients()[col++];
    result.beta_m = res.coefficients()[col++];

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();

    return result;
}

} // namespace hfm
