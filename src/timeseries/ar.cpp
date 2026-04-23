#include "hfm/timeseries/ar.hpp"
#include "hfm/estimators/ols.hpp"
#include <chrono>
#include <cmath>
#include <numbers>

namespace hfm {

Matrix<f64> lag_design_matrix(const Vector<f64>& y, std::size_t p, bool intercept) {
    std::size_t n = y.size();
    if (n <= p) return Matrix<f64>(0, 0);

    std::size_t T = n - p;
    std::size_t k = intercept ? (p + 1) : p;
    Matrix<f64> X(T, k, 0.0);

    for (std::size_t t = 0; t < T; ++t) {
        std::size_t col = 0;
        if (intercept) {
            X(t, col++) = 1.0;
        }
        for (std::size_t lag = 1; lag <= p; ++lag) {
            X(t, col++) = y[p + t - lag];
        }
    }
    return X;
}

Result<ARResult> ar(const Vector<f64>& y, const AROptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();

    std::size_t n = y.size();
    std::size_t p = opts.p;

    if (n <= p + 1) {
        return Status::error(ErrorCode::InvalidArgument, "ar: not enough observations");
    }

    auto X = lag_design_matrix(y, p, opts.add_intercept);
    std::size_t T = n - p;

    Vector<f64> y_dep(T);
    for (std::size_t t = 0; t < T; ++t) {
        y_dep[t] = y[p + t];
    }

    OLSOptions ols_opts;
    ols_opts.covariance = opts.covariance;
    ols_opts.hac_lag = opts.hac_lag;

    auto ols_result = ols(y_dep, X, ols_opts);
    if (!ols_result) return ols_result.status();

    auto& res = ols_result.value();

    ARResult result;
    result.coefficients = res.coefficients();
    result.std_errors = res.std_errors();
    result.t_stats = res.t_stats();
    result.residuals = res.residuals();
    result.sigma = res.sigma();
    result.r_squared = res.r_squared();
    result.p = p;
    result.n_obs = T;
    result.has_intercept = opts.add_intercept;

    // Information criteria
    std::size_t k = res.n_regressors();
    f64 ll = -0.5 * static_cast<f64>(T) * (1.0 + std::log(2.0 * std::numbers::pi) +
             std::log(result.sigma * result.sigma));
    result.aic = -2.0 * ll + 2.0 * static_cast<f64>(k);
    result.bic = -2.0 * ll + std::log(static_cast<f64>(T)) * static_cast<f64>(k);

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();

    return result;
}

} // namespace hfm
