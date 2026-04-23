#include "hfm/timeseries/arima.hpp"
#include "hfm/linalg/solver.hpp"
#include "hfm/linalg/matrix.hpp"
#include <chrono>
#include <cmath>
#include <numbers>
#include <algorithm>

namespace hfm {

Vector<f64> difference(const Vector<f64>& y, std::size_t d) {
    if (d == 0) return y;
    Vector<f64> result(y.size() - 1);
    for (std::size_t i = 1; i < y.size(); ++i) {
        result[i - 1] = y[i] - y[i - 1];
    }
    if (d > 1) return difference(result, d - 1);
    return result;
}

Vector<f64> undifference(const Vector<f64>& dy, const Vector<f64>& y_orig,
                          std::size_t d) {
    if (d == 0) return dy;
    Vector<f64> result(dy.size() + 1);
    result[0] = y_orig[y_orig.size() - 1];
    for (std::size_t i = 0; i < dy.size(); ++i) {
        result[i + 1] = result[i] + dy[i];
    }
    if (d > 1) return undifference(result, y_orig, d - 1);
    Vector<f64> out(dy.size());
    for (std::size_t i = 0; i < dy.size(); ++i) out[i] = result[i + 1];
    return out;
}

Result<ARIMAResult> arima(const Vector<f64>& y, const ARIMAOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();

    if (y.size() < opts.p + opts.d + opts.q + 5) {
        return Status::error(ErrorCode::InvalidArgument,
                             "arima: insufficient observations");
    }

    Vector<f64> z = difference(y, opts.d);
    std::size_t n = z.size();
    std::size_t p = opts.p;
    std::size_t q = opts.q;

    ARIMAResult result;
    result.p = p;
    result.d = opts.d;
    result.q = q;
    result.n_obs = y.size();

    if (q == 0) {
        std::size_t T = n - p;
        std::size_t k = p + (opts.add_intercept ? 1 : 0);
        Matrix<f64> X(T, k);
        Vector<f64> yy(T);

        for (std::size_t t = 0; t < T; ++t) {
            std::size_t idx = t + p;
            yy[t] = z[idx];
            std::size_t col = 0;
            if (opts.add_intercept) X(t, col++) = 1.0;
            for (std::size_t j = 0; j < p; ++j) {
                X(t, col++) = z[idx - j - 1];
            }
        }

        auto beta = solve_least_squares(X, yy);
        if (!beta) {
            return Status::error(ErrorCode::SingularMatrix, "arima: AR regression failed");
        }

        std::size_t col = 0;
        if (opts.add_intercept) result.intercept = beta.value()[col++];
        result.ar_coefficients.resize(p);
        for (std::size_t j = 0; j < p; ++j) {
            result.ar_coefficients[j] = beta.value()[col++];
        }

        auto fitted = matvec(X, beta.value());
        result.residuals.resize(T);
        result.fitted.resize(T);
        f64 sse = 0.0;
        for (std::size_t t = 0; t < T; ++t) {
            result.fitted[t] = fitted[t];
            result.residuals[t] = yy[t] - fitted[t];
            sse += result.residuals[t] * result.residuals[t];
        }
        result.sigma2 = sse / static_cast<f64>(T - k);

        f64 fT = static_cast<f64>(T);
        result.log_likelihood = -0.5 * fT * (std::log(2.0 * std::numbers::pi) +
                                 std::log(sse / fT) + 1.0);
        result.aic = -2.0 * result.log_likelihood + 2.0 * static_cast<f64>(k + 1);
        result.bic = -2.0 * result.log_likelihood +
                     static_cast<f64>(k + 1) * std::log(fT);

    } else {
        result.ar_coefficients.resize(p, 0.0);
        result.ma_coefficients.resize(q, 0.0);
        if (opts.add_intercept) result.intercept = z.mean();

        Vector<f64> residuals(n, 0.0);
        bool conv = false;

        for (std::size_t iter = 0; iter < opts.max_iter; ++iter) {
            std::size_t start_t = std::max(p, q);
            std::size_t T = n - start_t;
            std::size_t k = p + q + (opts.add_intercept ? 1 : 0);
            Matrix<f64> X(T, k);
            Vector<f64> yy(T);

            for (std::size_t t = 0; t < T; ++t) {
                std::size_t idx = t + start_t;
                yy[t] = z[idx];
                std::size_t col = 0;
                if (opts.add_intercept) X(t, col++) = 1.0;
                for (std::size_t j = 0; j < p; ++j) {
                    X(t, col++) = z[idx - j - 1];
                }
                for (std::size_t j = 0; j < q; ++j) {
                    X(t, col++) = residuals[idx - j - 1];
                }
            }

            auto beta = solve_least_squares(X, yy);
            if (!beta) break;

            Vector<f64> old_ar = result.ar_coefficients;
            Vector<f64> old_ma = result.ma_coefficients;
            f64 old_c = result.intercept;

            std::size_t col = 0;
            if (opts.add_intercept) result.intercept = beta.value()[col++];
            for (std::size_t j = 0; j < p; ++j)
                result.ar_coefficients[j] = beta.value()[col++];
            for (std::size_t j = 0; j < q; ++j)
                result.ma_coefficients[j] = beta.value()[col++];

            auto fitted = matvec(X, beta.value());
            for (std::size_t t = 0; t < T; ++t) {
                residuals[t + start_t] = yy[t] - fitted[t];
            }

            f64 change = 0.0;
            for (std::size_t j = 0; j < p; ++j)
                change += std::abs(result.ar_coefficients[j] - old_ar[j]);
            for (std::size_t j = 0; j < q; ++j)
                change += std::abs(result.ma_coefficients[j] - old_ma[j]);
            change += std::abs(result.intercept - old_c);

            if (change < opts.tol) {
                conv = true;
                break;
            }
        }
        result.converged = conv;

        std::size_t start_t = std::max(p, q);
        std::size_t T = n - start_t;
        result.residuals.resize(T);
        result.fitted.resize(T);
        f64 sse = 0.0;
        for (std::size_t t = 0; t < T; ++t) {
            result.residuals[t] = residuals[t + start_t];
            result.fitted[t] = z[t + start_t] - result.residuals[t];
            sse += result.residuals[t] * result.residuals[t];
        }
        std::size_t k = p + q + (opts.add_intercept ? 1 : 0);
        result.sigma2 = sse / static_cast<f64>(T - k);

        f64 fT = static_cast<f64>(T);
        result.log_likelihood = -0.5 * fT * (std::log(2.0 * std::numbers::pi) +
                                 std::log(sse / fT) + 1.0);
        result.aic = -2.0 * result.log_likelihood + 2.0 * static_cast<f64>(k + 1);
        result.bic = -2.0 * result.log_likelihood +
                     static_cast<f64>(k + 1) * std::log(fT);
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

Result<ARIMAForecastResult> arima_forecast(const Vector<f64>& y,
                                            const ARIMAResult& model,
                                            std::size_t n_ahead,
                                            f64 confidence) {
    auto start = std::chrono::high_resolution_clock::now();
    Vector<f64> z = difference(y, model.d);
    std::size_t n = z.size();

    ARIMAForecastResult result;
    result.n_ahead = n_ahead;
    result.confidence = confidence;
    result.forecast.resize(n_ahead);
    result.lower.resize(n_ahead);
    result.upper.resize(n_ahead);

    std::vector<f64> extended(n + n_ahead, 0.0);
    for (std::size_t i = 0; i < n; ++i) extended[i] = z[i];

    std::vector<f64> errors(n + n_ahead, 0.0);
    if (model.residuals.size() > 0) {
        std::size_t offset = n - model.residuals.size();
        for (std::size_t i = 0; i < model.residuals.size(); ++i) {
            errors[offset + i] = model.residuals[i];
        }
    }

    for (std::size_t h = 0; h < n_ahead; ++h) {
        std::size_t idx = n + h;
        f64 forecast = model.intercept;
        for (std::size_t j = 0; j < model.p; ++j) {
            if (idx > j) forecast += model.ar_coefficients[j] * extended[idx - j - 1];
        }
        for (std::size_t j = 0; j < model.q; ++j) {
            if (idx > j) forecast += model.ma_coefficients[j] * errors[idx - j - 1];
        }
        extended[idx] = forecast;
        result.forecast[h] = forecast;
    }

    f64 z_alpha = 1.96;
    if (confidence == 0.99) z_alpha = 2.576;
    else if (confidence == 0.90) z_alpha = 1.645;

    f64 sigma = std::sqrt(std::max(model.sigma2, 1e-20));
    for (std::size_t h = 0; h < n_ahead; ++h) {
        f64 se = sigma * std::sqrt(static_cast<f64>(h + 1));
        result.lower[h] = result.forecast[h] - z_alpha * se;
        result.upper[h] = result.forecast[h] + z_alpha * se;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

} // namespace hfm
