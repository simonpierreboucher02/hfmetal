#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/vector.hpp"

namespace hfm {

struct ARIMAOptions {
    std::size_t p = 1;
    std::size_t d = 0;
    std::size_t q = 0;
    bool add_intercept = true;
    std::size_t max_iter = 100;
    f64 tol = 1e-8;

    ARIMAOptions& set_p(std::size_t v) { p = v; return *this; }
    ARIMAOptions& set_d(std::size_t v) { d = v; return *this; }
    ARIMAOptions& set_q(std::size_t v) { q = v; return *this; }
};

struct ARIMAResult {
    Vector<f64> ar_coefficients;
    Vector<f64> ma_coefficients;
    f64 intercept = 0.0;
    f64 sigma2 = 0.0;
    Vector<f64> residuals;
    Vector<f64> fitted;
    f64 log_likelihood = 0.0;
    f64 aic = 0.0;
    f64 bic = 0.0;
    std::size_t p = 0;
    std::size_t d = 0;
    std::size_t q = 0;
    std::size_t n_obs = 0;
    bool converged = true;
    f64 elapsed_ms = 0.0;
};

Result<ARIMAResult> arima(const Vector<f64>& y,
                           const ARIMAOptions& opts = ARIMAOptions{});

Vector<f64> difference(const Vector<f64>& y, std::size_t d = 1);
Vector<f64> undifference(const Vector<f64>& dy, const Vector<f64>& y_orig,
                          std::size_t d = 1);

struct ARIMAForecastResult {
    Vector<f64> forecast;
    Vector<f64> lower;
    Vector<f64> upper;
    f64 confidence = 0.95;
    std::size_t n_ahead = 0;
    f64 elapsed_ms = 0.0;
};

Result<ARIMAForecastResult> arima_forecast(const Vector<f64>& y,
                                            const ARIMAResult& model,
                                            std::size_t n_ahead = 10,
                                            f64 confidence = 0.95);

} // namespace hfm
